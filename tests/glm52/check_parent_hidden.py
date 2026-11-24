"""Check optional parent transfer with CPU Torch and a HIXL WRITE double."""

import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.pd_vllm import ascend_layout as layout_api  # noqa: E402
from tilert.pd_vllm.ascend_cache import (  # noqa: E402
    AscendReceiveBuffer, CHECKPOINT_FILES,
)
from tilert.pd_vllm.ascend_parent import ParentHiddenCapture  # noqa: E402
from tilert.pd_vllm.ascend_receiver import AscendReceiveServer  # noqa: E402
from tilert.pd_vllm.ascend_source import bind_vllm_caches  # noqa: E402
from tilert.pd_vllm.prefix_capture import save_rank_prefix  # noqa: E402
from tilert.models.glm_5_2_ascend.session import PrefixRequest  # noqa: E402
from check_ascend_connector import configuration, request_output  # noqa: E402
from check_ascend_layout import fixture  # noqa: E402
from check_ascend_pd import endpoints, wait_for  # noqa: E402


REGION = layout_api.PARENT_HIDDEN_REGION
SOURCE_FILES = (
    "tilert/pd_vllm/ascend_parent.py", "tilert/pd_vllm/ascend_layout.py",
    "tilert/pd_vllm/ascend_source.py", "tilert/pd_vllm/ascend_connector.py",
    "tilert/pd_vllm/ascend_cache.py", "tilert/pd_vllm/ascend_receiver.py",
    "tilert/pd_vllm/prefix_capture.py", "tests/glm52/capture_pd_prefix.py",
    "tests/glm52/serve_vllm_prefill.sh", "tests/glm52/check_parent_hidden.py",
)


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


class ParentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import torch_npu

        cls.torch = torch
        cls.npu = torch_npu
        torch.set_num_threads(1)
        if torch_npu.npu._initialized:
            raise RuntimeError("This checker requires an uninitialized NPU")
        from tilert.pd_vllm import ascend_connector

        cls.connector = ascend_connector

    def setUp(self):
        torch = self.torch

        class Target(torch.nn.Module):
            def __init__(self, length, value):
                super().__init__()
                rows = ((torch.arange(length) % 31) + value).to(
                    torch.bfloat16,
                )
                self.output = rows[:, None].expand(length, 6144)

            def forward(self, *, input_ids, positions):
                return self.output

        self.target = Target

    def forward(self, target, length):
        ids = self.torch.arange(length)
        return target(input_ids=ids, positions=ids)

    def test_default_layout_and_optional_fixed_size(self):
        for length in (1, 128, 129, 8192):
            old = layout_api.cache_layout(length)
            new = layout_api.cache_layout(length, parent_hidden=True)
            self.assertEqual((old["version"], len(old["regions"])), (2, 196))
            self.assertEqual((new["version"], len(new["regions"])), (3, 197))
            self.assertEqual(new["nbytes"] - old["nbytes"], 12288)
            self.assertTrue(all(new["regions"][name] == value
                                for name, value in old["regions"].items()))
            layout_api.validate_layout(json.loads(json.dumps(new)))

    def test_real_hook_keeps_storage_identity_and_copies_last_row(self):
        capture = ParentHiddenCapture("cpu")
        pointer = capture.tensor.data_ptr()
        for generation, length in enumerate((1, 128, 129, 8192), 1):
            request = SimpleNamespace(rid=f"r{generation}", num_tokens=length)
            target = self.target(length, generation)
            capture.bind(target, request)
            self.forward(target, length)
            expected = generation + (length - 1) % 31
            target.output[-1].fill_(-7)
            self.assertEqual(capture.pointer_for(request), pointer)
            self.assertTrue(self.torch.all(capture.tensor == expected))
            self.assertEqual(capture.tensor.dtype, self.torch.bfloat16)
            capture.clear()
            with self.assertRaises(RuntimeError):
                capture.pointer_for(request)

    def test_hook_rejects_missing_duplicate_and_wrong_identity(self):
        request = SimpleNamespace(rid="r", num_tokens=2)
        capture = ParentHiddenCapture("cpu")
        target = self.target(2, 3)
        capture.bind(target, request)
        with self.assertRaises(RuntimeError):
            capture.pointer_for(request)
        self.forward(target, 2)
        for bad in (SimpleNamespace(rid="other", num_tokens=2),
                    SimpleNamespace(rid="r", num_tokens=1)):
            with self.assertRaises(RuntimeError):
                capture.pointer_for(bad)
        with self.assertRaises(RuntimeError):
            self.forward(target, 2)
        capture.clear()

    def test_hook_rejects_output_dtype_shape_and_input_shape(self):
        torch = self.torch
        for output in (torch.zeros(2, 6144), torch.zeros(1, 6144),
                       (torch.zeros(2, 6144),)):
            capture = ParentHiddenCapture("cpu")
            target = self.target(2, 1)
            target.output = output
            capture.bind(target, SimpleNamespace(rid="r", num_tokens=2))
            with self.assertRaises(ValueError):
                self.forward(target, 2)
            capture.clear()
        capture = ParentHiddenCapture("cpu")
        target = self.target(2, 1)
        capture.bind(target, SimpleNamespace(rid="r", num_tokens=2))
        with self.assertRaises(ValueError):
            self.forward(target, 1)
        capture.clear()

    def test_parent_contract_rejects_wrong_position_dtype_and_rotation(self):
        expected = layout_api.parent_hidden_metadata(2)
        for key, value in (("position", True), ("position", 2),
                           ("normalized", 1), ("normalized", False),
                           ("rotated", True), ("dtype", "float32"),
                           ("shape", [True, 6144])):
            bad = {**expected, key: value}
            with self.assertRaises(ValueError):
                layout_api.validate_parent_hidden_metadata(bad, 2)

    def test_transfer_plan_appends_one_registered_row(self):
        caches, cache_config = fixture(layout_api, strided=True)
        binding = bind_vllm_caches(caches, cache_config)
        tables = [list(range(group, group + 65)) for group in range(5)]
        base, pointer = 2**44, 2**48
        for length in (1, 128, 129, 8192):
            old = layout_api.cache_layout(length)
            new = layout_api.cache_layout(length, parent_hidden=True)
            common = dict(mtp_prefix_tokens=length - 1)
            original = binding.transfer_plan(tables, base, length,
                                             destination_layout=old, **common)
            updated = binding.transfer_plan(
                tables, base, length, destination_layout=new,
                parent_hidden_ptr=pointer, **common,
            )
            self.assertEqual(tuple(values[:-1] for values in updated),
                             original)
            self.assertEqual(tuple(values[-1] for values in updated),
                             (pointer, base + old["nbytes"], 12288))
            with self.assertRaises(ValueError):
                binding.transfer_plan(tables, base, length,
                                      destination_layout=new, **common)
            with self.assertRaises(ValueError):
                binding.transfer_plan(tables, base, length,
                                      parent_hidden_ptr=pointer, **common)

    def test_cpu_freeze_serializes_parent_without_receive_alias(self):
        torch = self.torch
        transport = mock.Mock()
        buffer = AscendReceiveBuffer(0, "cpu", 128, transport, "127.0.0.1",
                                    parent_hidden=True)
        buffer.storage.zero_()
        buffer.views[REGION].fill_(7)
        request = PrefixRequest("r", 2, {
            "generation": 1, "mtp_seq_len": 1, "last_prompt_token": 9,
            "prompt_token_ids": [8, 9],
            REGION: layout_api.parent_hidden_metadata(2),
        })
        with tempfile.TemporaryDirectory() as directory:
            receipt = save_rank_prefix(buffer, request, {"config": "hash"},
                                       directory)
            buffer.views[REGION].zero_()
            bundle = torch.load(receipt["files"]["mtp"]["path"],
                                map_location="cpu", weights_only=True)
            parent = bundle["parent_hidden"]
            self.assertTrue(torch.all(parent["tensor"] == 7))
            self.assertEqual(parent["tensor"].untyped_storage().nbytes(),
                             12288)
            self.assertEqual(parent["request_id"], "r")
            self.assertEqual(parent["generation"], 1)
            self.assertEqual(parent["position"], 1)
            self.assertTrue(parent["normalized"])
            self.assertFalse(parent["rotated"])
            self.assertEqual(bundle["layer"]["latent"].shape[0], 1)
        with self.assertRaises(ValueError):
            buffer.frozen_prefix(request, {})

    def test_receiver_requires_matching_parent_contract_before_admission(self):
        for enabled in (False, True):
            layout = layout_api.cache_layout(128, parent_hidden=enabled)
            server = AscendReceiveServer(layout, endpoints(layout), {},
                                         host="127.0.0.1", ctrl_port=0)
            incoming = {
                "rid": "r", "rank": 0, "seq_len": 2, "mtp_seq_len": 1,
                "last_prompt_token": 9, "prompt_token_ids": [8, 9],
            }
            try:
                candidates = (
                    [None, {**layout_api.parent_hidden_metadata(2),
                            "position": 2}]
                    if enabled else [layout_api.parent_hidden_metadata(2)]
                )
                for parent in candidates:
                    request = dict(incoming)
                    if parent is not None:
                        request[REGION] = parent
                    self.assertFalse(server._admit(request).get("accepted"))
                    self.assertEqual(server.state_snapshot()["state"],
                                     "free")
            finally:
                server.close()

    def test_real_connector_tcp_waits_for_all_parent_writes(self):
        connector = self.connector
        layout = layout_api.cache_layout(8192, parent_hidden=True)
        destinations = endpoints(layout)
        written = {}
        errors = []
        workers = []
        threads = []
        transports = []
        blocked = threading.Event()
        release = threading.Event()

        class HixlDouble:
            def __init__(self, **kwargs):
                self.regions = []
                self.sealed = False
                transports.append(self)

            def init(self, host):
                pass

            def register(self, pointer, size, device):
                if self.sealed:
                    raise RuntimeError("Late HIXL registration")
                self.regions.append((pointer, size))
                self.rank = device

            def write(self, endpoint, sources, targets, lengths):
                self.sealed = True
                for source, size in zip(sources, lengths):
                    if not any(base <= source and source + size <= base + total
                               for base, total in self.regions):
                        raise ValueError("Unregistered source")
                capture = workers[self.rank]._parent_capture
                expected = (
                    capture.tensor.data_ptr(), endpoint["base_ptr"]
                    + layout["regions"][REGION]["offset"], 12288,
                )
                actual = tuple(values[-1] for values in
                               (sources, targets, lengths))
                if actual != expected:
                    raise ValueError("Parent is not part of the KV WRITE")
                if self.rank == 15:
                    blocked.set()
                    if not release.wait(10):
                        raise TimeoutError("Parent WRITE was not released")
                written[self.rank] = (
                    workers[self.rank]._parent_capture.tensor.clone()
                )

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in CHECKPOINT_FILES:
                (directory / name).write_text("{}")
            config = configuration(directory)
            config.model_config.enforce_eager = True
            config.kv_transfer_config.kv_connector_extra_config[
                "transfer_mtp_parent_hidden"
            ] = True
            checkpoint = {name: hashlib.sha256(b"{}").hexdigest()
                          for name in CHECKPOINT_FILES}
            server = AscendReceiveServer(layout, destinations, checkpoint,
                                         host="127.0.0.1", ctrl_port=0)
            tables = [list(range(group, group + 65)) for group in range(5)]
            try:
                for rank in range(16):
                    caches, cache_config = fixture(layout_api, strided=True)
                    for tensors in caches.values():
                        for tensor in tensors:
                            tensor.device.index = rank
                    worker = connector.AscendTileRTConnector(config, "worker",
                                                            cache_config)
                    with (
                        mock.patch(
                            "vllm.distributed.get_tensor_model_parallel_rank",
                            return_value=rank,
                        ),
                        mock.patch(
                            "tilert.pd_vllm.ascend_parent.ParentHiddenCapture",
                            side_effect=lambda device: ParentHiddenCapture(
                                "cpu",
                            ),
                        ),
                    ):
                        worker.register_kv_caches(caches)
                    target = self.target(8192, rank + 1)
                    metadata = worker.build_connector_meta(
                        request_output(8192, tables, server.ctrl_port),
                    )
                    context = SimpleNamespace(model_instance=target,
                                              is_draft_model=False)
                    with mock.patch("vllm.forward_context.get_forward_context",
                                    return_value=context):
                        worker.bind_connector_metadata(metadata)
                    self.forward(target, 8192)
                    worker._written_layers = {
                        region.layer_name
                        for region in worker._binding.regions.values()
                        if region.name.startswith("mla_latent.")
                    }
                    workers.append(worker)

                def send(worker):
                    try:
                        worker.wait_for_save()
                    except Exception as error:
                        errors.append(repr(error))

                with (
                    mock.patch.object(connector, "HixlTransport", HixlDouble),
                    mock.patch.object(self.torch.npu, "synchronize")
                    as synchronize,
                ):
                    for worker in workers:
                        thread = threading.Thread(target=send, args=(worker,))
                        threads.append(thread)
                        thread.start()
                    self.assertTrue(blocked.wait(10))
                    wait_for(
                        lambda: len(server.state_snapshot()["done_ranks"])
                        == 15, timeout=10,
                    )
                    self.assertTrue(server.completed.empty())
                    self.assertNotIn(15, written)
                    release.set()
                    for thread in threads:
                        thread.join(12)
                    self.assertFalse(errors)
                    self.assertEqual(synchronize.call_count, 16)
                received = server.completed.get(timeout=2)
                request = server.profile.convert(None, None, 8192,
                                                 received, 16)
                self.assertEqual(request.metadata[REGION],
                                 layout_api.parent_hidden_metadata(8192))
                self.assertEqual(received.done_ranks, set(range(16)))
                self.assertEqual(set(written), set(range(16)))
                for rank, value in written.items():
                    self.assertTrue(self.torch.all(
                        value == rank + 1 + 8191 % 31,
                    ))
                for transport in transports:
                    binding = workers[transport.rank]._binding
                    self.assertEqual(len(transport.regions),
                                     len(binding.registration_ranges()) + 1)
                wrong = deepcopy(server._hello())
                wrong["layout_version"] = 2
                with self.assertRaises(ValueError):
                    workers[0]._validate_hello(wrong)
                server.release(received.rid)
            finally:
                release.set()
                for thread in threads:
                    thread.join(12)
                for worker in workers:
                    worker.clear_connector_metadata()
                server.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    before = hashes()
    started = time.monotonic()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ParentChecks)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    initialized = bool(ParentChecks.npu.npu._initialized)
    report = {
        "passed": (result.wasSuccessful() and not initialized
                   and before == hashes()),
        "tests_run": result.testsRun, "elapsed_s": time.monotonic() - started,
        "scope": ("CPU Torch hooks, real V1 base/localhost TCP; "
                  "synthetic KV addresses"),
        "stubbed": ["NPU allocation", "forward context", "TP rank",
                    "device synchronization", "HIXL WRITE"],
        "not_executed": ["NPU", "real model forward", "HIXL",
                         "MTP computation"],
        "torch": ParentChecks.torch.__version__,
        "torch_npu": ParentChecks.npu.__version__,
        "npu_initialized": initialized, "python": platform.python_version(),
        "source_sha256": before, "command": sys.argv,
        "log": stream.getvalue(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(stream.getvalue(), end="")
    print(json.dumps({key: report[key] for key in
                     ("passed", "tests_run", "npu_initialized")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
