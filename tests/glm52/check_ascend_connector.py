"""Check the Ascend connector's wire protocol with stdlib metadata doubles."""

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import pickle
import platform
import runpy
import subprocess
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace


def install_vllm_boundary():
    """Stub the external V1 base; import the real TileRT connector normally."""
    if "torch" in sys.modules or "vllm" in sys.modules:
        raise RuntimeError("Run this checker in a fresh stdlib Python process")
    path = "vllm.distributed.kv_transfer.kv_connector.v1.base"
    for length in range(1, len(path.split(".")) + 1):
        name = ".".join(path.split(".")[:length])
        module = ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module

    class Base:
        def __init__(self, config, role, cache_config):
            self._vllm_config = config
            self._kv_cache_config = cache_config
            self._connector_metadata = None

        def bind_connector_metadata(self, metadata):
            self._connector_metadata = metadata

        def _get_connector_metadata(self):
            return self._connector_metadata

    module.KVConnectorBase_V1 = Base
    module.KVConnectorMetadata = type("Metadata", (), {})
    module.SupportsHMA = type("SupportsHMA", (), {})
    distributed = sys.modules["vllm.distributed"]
    distributed.rank = 0
    distributed.get_tensor_model_parallel_rank = lambda: distributed.rank
    return distributed


@dataclass(slots=True)
class Sampling:
    temperature: float = 0.0
    repetition_penalty: float = 1.0
    extra_args: dict = field(default_factory=dict)


def configuration(directory):
    return SimpleNamespace(
        model_config=SimpleNamespace(model=str(directory), max_model_len=9216),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=16, pipeline_parallel_size=1,
            data_parallel_size=1, prefill_context_parallel_size=1,
            decode_context_parallel_size=1, enable_expert_parallel=False,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=1, enable_chunked_prefill=False,
            max_num_batched_tokens=9216,
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False, block_size=128,
        ),
        speculative_config=SimpleNamespace(
            method="mtp", num_speculative_tokens=3,
            num_speculative_tokens_per_batch_size=None,
            parallel_drafting=False,
            draft_model_config=SimpleNamespace(
                model=str(directory), max_model_len=9216,
            ),
        ),
        kv_transfer_config=SimpleNamespace(
            is_kv_producer=True,
            kv_connector_extra_config={
                "hixl_host": "127.0.0.1", "tilert_timeout_s": 10.0,
                "tilert_admission_attempts": 1,
                "hixl_library": "/unused/fake/libtilert_hixl.so",
            },
        ),
    )


def request_output(length, tables, port):
    request = SimpleNamespace(
        req_id="ascend-pd-check", num_computed_tokens=0,
        prompt_token_ids=[(7 + row * 73) % 154880 for row in range(length)],
        block_ids=tables,
        sampling_params=Sampling(extra_args={"kv_transfer_params": {
            "tilert_host": "127.0.0.1", "tilert_ctrl_port": port,
            "sampling": {"temperature": 0.0},
        }}),
    )
    return SimpleNamespace(
        scheduled_new_reqs=[request],
        num_scheduled_tokens={request.req_id: length},
    )


def check_plan(binding, tables, layout, base, prefix, plan):
    """Compare each copied page edge with an independent address oracle."""
    sources, destinations, lengths = plan
    cursor = 0
    covered_bytes = 0
    for name, region in layout["regions"].items():
        rows = prefix - 1 if name.endswith(".78") else prefix
        source = binding.regions[name]
        start = base + region["offset"]
        end = start + rows * region["row_bytes"]
        covered = start
        while cursor < len(lengths) and destinations[cursor] < end:
            destination = destinations[cursor]
            size = lengths[cursor]
            if destination != covered or destination + size > end:
                return False
            for offset in range(0, size, 128 * region["row_bytes"]):
                last_byte = min(
                    offset + 128 * region["row_bytes"], size,
                ) - 1
                for byte in (offset, last_byte):
                    row, inner = divmod(
                        destination + byte - start, region["row_bytes"],
                    )
                    block = tables[source.group_id][row // 128]
                    expected = (
                        source.data_ptr + block * source.page_stride_bytes
                        + row % 128 * region["row_bytes"] + inner
                    )
                    if sources[cursor] + byte != expected:
                        return False
            covered += size
            covered_bytes += size
            cursor += 1
        if covered != end:
            return False
    expected_bytes = sum(
        (prefix - int(name.endswith(".78"))) * region["row_bytes"]
        for name, region in layout["regions"].items()
    )
    return cursor == len(lengths) and covered_bytes == expected_bytes


def run(root):
    distributed = install_vllm_boundary()
    from tilert.pd_vllm import ascend_connector as connector
    from tilert.pd_vllm import ascend_layout
    from tilert.pd_vllm.ascend_cache import (
        CHECKPOINT_FILES, checkpoint_fingerprint, receive_device_id,
    )
    from tilert.pd_vllm.ascend_receiver import AscendReceiveServer

    fixture = runpy.run_path(
        str(root / "tests/glm52/check_ascend_layout.py"),
    )["fixture"]
    checks = []

    def check(name, value):
        checks.append({"name": name, "passed": bool(value)})

    def rejects(name, operation):
        try:
            operation()
        except (ValueError, RuntimeError):
            check(name, True)
        else:
            check(name, False)

    synchronize_calls = []
    torch_stub = ModuleType("torch")
    torch_stub.npu = SimpleNamespace(synchronize=synchronize_calls.append)
    sys.modules["torch"] = torch_stub
    layout = ascend_layout.cache_layout(8320)
    endpoints = [{
        "rank": rank, "base_ptr": 2 ** 44 + rank * 2 ** 32,
        "transport": "hixl", "hixl_engine": f"127.0.0.1:{26100 + rank}",
        "hixl_device": receive_device_id(rank),
        "hixl_regions": [{"base": 2 ** 44 + rank * 2 ** 32,
                          "bytes": layout["nbytes"]}],
    } for rank in range(16)]
    tables = [list(range(group, group + 65)) for group in range(5)]
    writes = []
    admissions = {}
    metadata_checks = []
    parameter_checks = []
    transports = []
    entered = threading.Barrier(17, timeout=10)
    release = threading.Event()
    ack_entered = threading.Event()
    release_ack = threading.Event()
    returned = set()
    errors = []

    class FakeHixl:
        def __init__(self, **kwargs):
            self.regions = []
            self.port = kwargs["port"]
            self.library = kwargs["library_path"]
            transports.append(self)

        def init(self, host):
            self.host = host

        def register(self, pointer, size, device):
            self.regions.append((pointer, size))
            self.device = device

        def write(self, endpoint, sources, destinations, lengths):
            if self.device not in synchronize_calls:
                raise RuntimeError(
                    "WRITE began before producer synchronization"
                )
            writes.append((
                self.device, endpoint, (sources, destinations, lengths),
            ))
            entered.wait()
            if not release.wait(10):
                raise TimeoutError("Checker did not release fake WRITE")

    connector.HixlTransport = FakeHixl
    with tempfile.TemporaryDirectory(prefix="ascend-connector-check-") as temp:
        directory = Path(temp)
        for name in CHECKPOINT_FILES:
            (directory / name).write_text(json.dumps({"fixture": name}))
        config = configuration(directory)
        config.scheduler_config.enable_chunked_prefill = True
        config.scheduler_config.max_num_batched_tokens = 8192
        hashes = checkpoint_fingerprint(directory)
        server = AscendReceiveServer(
            layout, endpoints, hashes, host="127.0.0.1", ctrl_port=0,
            request_timeout=10,
        )
        complete_rank = server._complete_rank
        admit = server._admit

        def record_admission(request):
            admissions[request["rank"]] = request
            return admit(request)

        server._admit = record_admission

        def delay_rank_zero_ack(request, generation, done):
            reply = complete_rank(request, generation, done)
            if request["rank"] == 0:
                ack_entered.set()
                if not release_ack.wait(10):
                    raise TimeoutError("Checker did not release done ack")
            return reply

        server._complete_rank = delay_rank_zero_ack
        workers = []
        threads = []
        try:
            for rank in range(16):
                distributed.rank = rank
                caches, cache_config = fixture(ascend_layout, strided=True)
                for tensors in caches.values():
                    for tensor in tensors:
                        tensor.device.index = rank
                worker = connector.AscendTileRTConnector(
                    config, "worker", cache_config,
                )
                worker.register_kv_caches(caches)
                output = request_output(8192, tables, server.ctrl_port)
                source = output.scheduled_new_reqs[0]
                params = source.sampling_params.extra_args
                original = json.loads(json.dumps(params))
                metadata = worker.build_connector_meta(output)
                request = metadata.requests[0]
                metadata_checks.append(
                    request.prompt_token_ids == source.prompt_token_ids
                    and request.prompt_token_ids is not source.prompt_token_ids
                )
                parameter_checks.append(params == original)
                metadata = pickle.loads(pickle.dumps(metadata))
                metadata_checks.append(
                    metadata.requests[0].prompt_token_ids
                    == source.prompt_token_ids
                )
                worker.bind_connector_metadata(metadata)
                for region in worker._binding.regions.values():
                    if region.name.startswith("mla_latent."):
                        worker.save_kv_layer(region.layer_name, None, None)
                workers.append(worker)

            first = workers[0]
            check("chunking_enabled_complete_8192_prefill", (
                config.scheduler_config.enable_chunked_prefill
                and len(workers) == 16 and all(
                    worker._get_connector_metadata().requests[0].num_tokens
                    == 8192 for worker in workers
                )
            ))
            config.scheduler_config.enable_chunked_prefill = False
            check("chunking_disabled_still_supported", (
                first._check_config(config) is None
                and len(first.build_connector_meta(output).requests) == 1
            ))
            config.scheduler_config.enable_chunked_prefill = True
            config.speculative_config.parallel_drafting = True
            rejects("parallel_drafting_changes_MTP_prefix", (
                lambda: first._check_config(config)
            ))
            config.speculative_config.parallel_drafting = False
            for prefix in (1, 128, 129, 8192):
                plan = first._binding.transfer_plan(
                    tables, endpoints[0]["base_ptr"], prefix,
                    mtp_prefix_tokens=prefix - 1, destination_layout=layout,
                )
                check(f"page_and_MTP_lengths_P{prefix}", check_plan(
                    first._binding, tables, layout,
                    endpoints[0]["base_ptr"], prefix, plan,
                ))
            mtp_name = first._binding.regions["mla_latent.78"].layer_name
            first._written_layers.remove(mtp_name)
            rejects("missing_MTP_prevents_send", first.wait_for_save)
            check("missing_MTP_before_device_or_transport", (
                not synchronize_calls and not transports
            ))
            first._written_layers.add(mtp_name)
            bad_hello = server._hello()
            bad_hello["checkpoint_sha256"] = {}
            rejects(
                "checkpoint_mismatch",
                lambda: first._validate_hello(bad_hello),
            )
            partial = request_output(8192, tables, server.ctrl_port)
            partial.num_scheduled_tokens["ascend-pd-check"] = 4096
            rejects(
                "chunked_prefix", lambda: first.build_connector_meta(partial),
            )
            check("partial_prefill_rejected_before_transport", (
                not synchronize_calls and not transports and not writes
            ))
            partial.num_scheduled_tokens["ascend-pd-check"] = 8192
            partial.scheduled_new_reqs[0].num_computed_tokens = 128
            rejects(
                "prefix_hit", lambda: first.build_connector_meta(partial),
            )

            def send(rank):
                try:
                    workers[rank].wait_for_save()
                    returned.add(rank)
                except Exception as error:
                    errors.append({"rank": rank, "error": repr(error)})

            for rank in range(16):
                thread = threading.Thread(target=send, args=(rank,))
                threads.append(thread)
                thread.start()
            entered.wait()
            check("wait_for_save_blocks_until_WRITE", not returned)
            snapshot = server.state_snapshot()
            check("all_ranks_admitted_no_ready", (
                snapshot["active_writers"] == 16
                and snapshot["done_ranks"] == [] and server.completed.empty()
            ))
            release.set()
            check("done_ack_reached", ack_entered.wait(10))
            check("wait_for_save_blocks_until_done_ack", 0 not in returned)
            release_ack.set()
            for thread in threads:
                thread.join(12)
            check("all_rank_calls_returned", (
                returned == set(range(16)) and not errors
            ))
            complete = server.completed.get(timeout=2)
            check("receiver_complete_79_caches", (
                complete.done_ranks == set(range(16))
                and complete.seq_len == 8192 and complete.mtp_seq_len == 8191
            ))
            expected_prompt = [(7 + row * 73) % 154880 for row in range(8192)]
            check("all16_worker_metadata_preserve_prompt_and_pickle", (
                len(metadata_checks) == 32 and all(metadata_checks)
            ))
            check("source_sampling_parameters_unchanged", (
                len(parameter_checks) == 16 and all(parameter_checks)
            ))
            check("all16_wire_admissions_preserve_identical_prompt", (
                set(admissions) == set(range(16)) and all(
                    request.get("prompt_token_ids") == expected_prompt
                    and request["last_prompt_token"] == expected_prompt[-1]
                    for request in admissions.values()
                )
            ))
            check("completed_request_retains_original_prompt", (
                complete.prompt_token_ids == expected_prompt
                and complete.last_prompt_token == expected_prompt[-1]
            ))
            check("NPU_sync_precedes_WRITE", (
                sorted(synchronize_calls) == list(range(16))
            ))
            check("source_storage_registered_once", all(
                transport.regions
                == workers[transport.device]._binding.registration_ranges()
                and transport.port == 26000 + transport.device
                for transport in transports
            ))
            check("hixl_library_forwarded", all(
                transport.library == "/unused/fake/libtilert_hixl.so"
                for transport in transports
            ))
            check("rank_r_to_rank_r_all_region_bytes", (
                len(writes) == 16 and all(
                    endpoint == endpoints[rank] and check_plan(
                        workers[rank]._binding, tables, layout,
                        endpoint["base_ptr"], 8192, plan,
                    ) for rank, endpoint, plan in writes
                )
            ))
            check("scheduler_frees_after_synchronous_save", all(
                worker.request_finished(
                    SimpleNamespace(request_id="ascend-pd-check"), tables,
                )
                == (False, None) for worker in workers
            ))
            server.release(complete.rid)
        finally:
            release.set()
            release_ack.set()
            for thread in threads:
                thread.join(12)
            server.close()
    check("external_runtime_is_stubbed", sys.modules["torch"] is torch_stub)
    return checks, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vllm-root", type=Path)
    parser.add_argument("--vllm-ascend-root", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    checks, errors = run(root)
    source_files = (
        "tilert/pd_vllm/ascend_connector.py",
        "tilert/pd_vllm/ascend_source.py", "tilert/pd_vllm/ascend_layout.py",
        "tilert/pd_vllm/ascend_receiver.py",
        "tilert/pd_vllm/ascend_cache.py",
        "tilert/pd_vllm/prefill_connector.py",
        "tilert/pd_vllm/wire.py", "tests/glm52/check_ascend_connector.py",
        "tests/glm52/check_ascend_layout.py",
    )
    report = {
        "passed": all(item["passed"] for item in checks) and not errors,
        "scope": (
            "stdlib metadata, synthetic addresses, real localhost TCP; "
            "V1 base and HIXL WRITE stubbed"
        ),
        "not_executed": [
            "vLLM worker", "Torch", "NPU", "HIXL", "KV numerical parity",
            "performance",
        ],
        "python": platform.python_version(),
        "checks": checks, "errors": errors,
        "command": sys.argv,
        "source_api": {
            "scope": "source anchors; external vLLM runtime was stubbed",
            "vllm": source_evidence(args.vllm_root, {
                "vllm/config/parallel.py": (
                    "pipeline_parallel_size:", "tensor_parallel_size:",
                    "data_parallel_size:", "prefill_context_parallel_size:",
                    "decode_context_parallel_size:", "enable_expert_parallel:",
                ),
                "vllm/config/cache.py": (
                    "block_size:", "enable_prefix_caching:",
                ),
                "vllm/config/scheduler.py": (
                    "max_num_batched_tokens:", "max_num_seqs:",
                    "enable_chunked_prefill:",
                ),
                "vllm/config/speculative.py": (
                    "num_speculative_tokens:", "method:",
                    "num_speculative_tokens_per_batch_size:",
                    "parallel_drafting:",
                    "draft_model_config:",
                ),
                "vllm/config/kv_transfer.py": ("def is_kv_producer(",),
                "vllm/v1/worker/kv_connector_model_runner_mixin.py": (
                    "def finalize_kv_connector(",
                    "kv_connector.wait_for_save()",
                ),
                "vllm/model_executor/models/deepseek_mtp.py": (
                    "self.mtp_block = DeepseekV2DecoderLayer(",
                    'vllm_config, f"{prefix}.layers.{idx}"',
                ),
            }),
            "vllm_ascend": source_evidence(args.vllm_ascend_root, {
                "vllm_ascend/worker/model_runner_v1.py": (
                    "defer_kv_connector_finalize =",
                    "self.finalize_kv_connector()",
                    "get_kv_transfer_group().register_kv_caches(kv_caches)",
                    "if current_kv_cache_spec.cache_sparse_li_c4:",
                    "indexer_scale_cache_shape =",
                ),
                "vllm_ascend/spec_decode/llm_base_proposer.py": (
                    "if self.num_speculative_tokens == 0:",
                    "self.input_ids[: num_tokens - 1] = target_token_ids[1:]",
                    "self.input_ids[token_indices_to_sample] = next_token_ids",
                    "self._set_positions(num_tokens, target_positions)",
                ),
                "vllm_ascend/attention/sfa_v1.py": (
                    "def _compose_sfa_kv_cache(",
                    "maybe_save_kv_layer_to_connector(",
                ),
            }),
        },
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in source_files
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": len(checks),
                      "output": str(args.output)}))
    raise SystemExit(0 if report["passed"] else 1)


def source_evidence(directory, anchors):
    if directory is None:
        return None
    files = {}
    for name, symbols in anchors.items():
        source = (directory / name).read_bytes()
        lines = source.decode().splitlines()
        locations = {
            symbol: [number for number, line in enumerate(lines, 1)
                     if line.strip().startswith(symbol)]
            for symbol in symbols
        }
        if not all(locations.values()):
            raise ValueError(f"Missing fixed-source API anchor: {name}")
        files[name] = {
            "sha256": hashlib.sha256(source).hexdigest(),
            "anchors": locations,
        }
    return {
        "root": str(directory),
        "revision": subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "files": files,
    }


if __name__ == "__main__":
    main()
