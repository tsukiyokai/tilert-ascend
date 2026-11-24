"""Check TP16 prefix export with symbolic CPU rows, not tensor arithmetic."""

import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import platform
from queue import Queue
import sys
import tempfile
import threading
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.models.glm_5_2_ascend.session import PrefixRequest  # noqa: E402
from tilert.models.glm_5_2_ascend.tp_runtime import (  # noqa: E402
    CoordinatedBackend, RankExecutor,
)
from tilert.pd_vllm.ascend_cache import AscendReceiveBuffer  # noqa: E402
from tilert.pd_vllm.ascend_layout import cache_layout  # noqa: E402
from tilert.pd_vllm import prefix_capture as capture  # noqa: E402


CHECKPOINT = {"config.json": "c" * 64,
              "quant_model_description.json": "d" * 64,
              "quant_model_weights.safetensors.index.json": "e" * 64}


class SymbolicRows:
    """A non-Tensor view: one string per row represents its source identity."""

    def __init__(self, storage, dtype, width, start=0, stop=None, flat=False):
        self.storage = storage
        self.dtype = dtype
        self.width = width
        self.start = start
        self.stop = len(storage) if stop is None else stop
        self.flat = flat

    @property
    def values(self):
        return self.storage[self.start:self.stop]

    @property
    def shape(self):
        length = self.stop - self.start
        return (length,) if self.flat else (length, self.width)

    def view(self, *shape):
        if shape != (-1, self.width) and not (
            shape == (-1,) and self.width == 1
        ):
            raise ValueError("Unsupported symbolic view")
        return SymbolicRows(self.storage, self.dtype, self.width,
                            self.start, self.stop, len(shape) == 1)

    def __getitem__(self, selected):
        start, stop, step = selected.indices(self.stop - self.start)
        if step != 1:
            raise ValueError("Only contiguous symbolic rows are supported")
        return SymbolicRows(self.storage, self.dtype, self.width,
                            self.start + start, self.start + stop, self.flat)

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return SymbolicRows(list(self.values), self.dtype, self.width,
                            flat=self.flat)


def encode_rows(value):
    if not isinstance(value, SymbolicRows):
        raise TypeError(type(value).__name__)
    return {"fixture": "symbolic_rows", "dtype": value.dtype,
            "shape": value.shape, "rows": value.values}


def make_buffer(rank, capacity=256):
    buffer = object.__new__(AscendReceiveBuffer)
    buffer.rank = rank
    buffer.layout = cache_layout(capacity)
    buffer._last_generation = 0
    buffer.views = {}
    for name, region in buffer.layout["regions"].items():
        width = region["row_bytes"] // (
            1 if region["dtype"] == "int8" else 2
        )
        rows = [f"{rank}:{name}:{row}" for row in range(capacity)]
        buffer.views[name] = SymbolicRows(rows, region["dtype"], width)
    return buffer


def request(length=3, generation=1):
    tokens = [(generation * 7 + row * 73) % 154880
              for row in range(length)]
    return PrefixRequest(f"request-{generation}", length, {
        "generation": generation, "mtp_seq_len": length - 1,
        "last_prompt_token": tokens[-1], "prompt_token_ids": tokens,
    })


class Gather:
    """Every rank observes the same receipts after all sixteen arrivals."""

    def __init__(self):
        self.values = [None] * 16
        self.barrier = threading.Barrier(16, timeout=10)
        self.condition = threading.Condition()
        self.arrivals = set()

    def __call__(self, receipt):
        with self.condition:
            self.values[receipt["rank"]] = receipt
            self.arrivals.add(receipt["rank"])
            self.condition.notify_all()
        self.barrier.wait()
        result = deepcopy(self.values)
        self.barrier.wait()
        return result


class CaptureTeam:
    """Execute actual rank commands concurrently over stdlib queues."""

    def __init__(self, directory, *, failed_rank=None, blocked_rank=None):
        self.directory = Path(directory)
        self.failed_rank = failed_rank
        self.blocked_rank = blocked_rank
        self.blocked = threading.Event()
        self.release = threading.Event()
        self.gather = Gather()
        self.saved = {}
        self.commands = []
        self.errors = {}
        self.finished = {}
        self.condition = threading.Condition()
        self.buffers = [make_buffer(rank) for rank in range(16)]
        self.ranks = []
        for rank, buffer in enumerate(self.buffers):
            engine = capture.CaptureRankEngine(
                buffer, CHECKPOINT, directory, self.gather,
                save=lambda bundle, file, rank=rank:
                self.save(rank, bundle, file),
            )
            self.ranks.append(RankExecutor(engine, rank,
                                           lambda prefix, rank: prefix))
        self.queues = {rank: Queue() for rank in range(1, 16)}
        self.threads = [threading.Thread(target=self.worker, args=(rank,))
                        for rank in range(1, 16)]
        for thread in self.threads:
            thread.start()
        self.backend = CoordinatedBackend(self.ranks[0], self)
        self.engine = capture.PrefixCaptureEngine(
            self.backend, self.ranks[0].engine, directory,
            {"fixture": "symbolic CPU rows; no numerical execution"},
        )

    def save(self, rank, bundle, file):
        role = "main" if "layers" in bundle else "mtp"
        if rank == self.blocked_rank and role == "mtp":
            self.blocked.set()
            if not self.release.wait(10):
                raise TimeoutError("The checker did not release the writer")
        if rank == self.failed_rank:
            raise OSError(f"rank {rank} cannot write its bundle")
        self.saved[rank, role] = bundle
        file.write(json.dumps(bundle, default=encode_rows).encode())

    def send(self, command):
        self.commands.append(deepcopy(command))
        for queue in self.queues.values():
            queue.put(deepcopy(command))

    def worker(self, rank):
        while True:
            command = self.queues[rank].get()
            if command is None:
                return
            try:
                self.ranks[rank].execute(command)
            except BaseException as error:
                self.errors[rank] = error
            finally:
                with self.condition:
                    self.finished[rank] = command["serial"]
                    self.condition.notify_all()
            if rank in self.errors or self.ranks[rank].closed:
                return

    def wait_workers(self):
        serial = self.commands[-1]["serial"]
        with self.condition:
            if not self.condition.wait_for(lambda: all(
                self.finished.get(rank, 0) >= serial for rank in range(1, 16)
            ), timeout=10):
                raise TimeoutError("Workers did not finish the command")

    def stop(self):
        self.release.set()
        self.gather.barrier.abort()
        for queue in self.queues.values():
            queue.put(None)
        for thread in self.threads:
            thread.join(10)
            if thread.is_alive():
                raise RuntimeError("A checker worker did not stop")


class CaptureChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def team(self, **kwargs):
        team = CaptureTeam(self.directory, **kwargs)
        self.addCleanup(team.stop)
        return team

    def manifest(self, index=0):
        path = self.directory / f"case-{index:02d}" / "manifest.json"
        return json.loads(path.read_text())

    def test_cpu_copy_clones_views_and_excludes_unused_rows(self):
        source = SymbolicRows(["prefix", "unused"], "bfloat16", 512)[:1]
        copied = capture.copy_to_cpu(source)
        source.storage[0] = "overwritten"
        self.assertEqual((copied.dtype, copied.shape, copied.values,
                          len(copied.storage)),
                         ("bfloat16", (1, 512), ["prefix"], 1))
        self.assertIsNot(copied.storage, source.storage)

    def test_rank_bundles_keep_all_fields_dtypes_and_original_rows(self):
        team = self.team()
        receipt = capture.save_rank_prefix(
            team.buffers[15], request(129), CHECKPOINT, self.directory,
            save=lambda bundle, file: team.save(15, bundle, file),
        )
        main, mtp = team.saved[15, "main"], team.saved[15, "mtp"]
        self.assertEqual(len(main["layers"]), 78)
        self.assertEqual(main["prompt_token_ids"],
                         request(129).metadata["prompt_token_ids"])
        self.assertEqual(main["metadata_sha256"], CHECKPOINT)
        self.assertEqual(mtp["metadata_sha256"], CHECKPOINT)
        for layer in range(79):
            values = main["layers"][layer] if layer < 78 else mtp["layer"]
            length = 129 if layer < 78 else 128
            expected = {"latent": ("bfloat16", (length, 512)),
                        "rope": ("bfloat16", (length, 64))}
            full = layer < 3 or (layer - 2) % 4 == 0
            if full:
                dtype = "int8" if 6 <= layer <= 66 else "bfloat16"
                expected["index_keys"] = dtype, (length, 128)
                if dtype == "int8":
                    expected["index_scales"] = "float16", (length,)
            self.assertEqual({name: (value.dtype, value.shape)
                              for name, value in values.items()}, expected)
            for name, value in values.items():
                region = {"latent": "mla_latent", "rope": "mla_rope",
                          "index_keys": "index",
                          "index_scales": "index_scale"}[name]
                source = team.buffers[15].views[f"{region}.{layer}"]
                self.assertIsNot(value.storage, source.storage)
                self.assertEqual(value.values, source.values[:length])
                self.assertEqual(len(value.storage), length)
        before = main["layers"][0]["latent"].values
        team.buffers[15].views["mla_latent.0"].storage[0] = "reused"
        self.assertEqual(main["layers"][0]["latent"].values, before)
        for record in receipt["files"].values():
            self.assertEqual(capture.digest(record["path"]), record["sha256"])
            self.assertEqual(Path(record["path"]).stat().st_size,
                             record["bytes"])
        self.assertEqual(sorted(path.name for path in
                                (self.directory / "main-prefix").iterdir()),
                         ["rank-15.pt"])

    def test_first_token_waits_for_all_rank_files_and_no_model_step(self):
        team = self.team(blocked_rank=15)
        delivered, failures = [], []

        def callback(token):
            manifest = self.manifest()
            self.assertEqual(manifest["status"], "prefix_saved")
            self.assertEqual(len(manifest["ranks"]), 16)
            for receipt in manifest["ranks"]:
                for record in receipt["files"].values():
                    self.assertEqual(capture.digest(record["path"]),
                                     record["sha256"])
            delivered.append(token)

        def run_request():
            try:
                team.engine.inject(request(129))
                team.engine.decode(901, max_tokens=1, on_token=callback)
            except BaseException as error:
                failures.append(error)

        caller = threading.Thread(target=run_request)
        caller.start()
        try:
            self.assertTrue(team.blocked.wait(10))
            with team.gather.condition:
                self.assertTrue(team.gather.condition.wait_for(
                    lambda: len(team.gather.arrivals) == 15, timeout=10,
                ))
            self.assertTrue(caller.is_alive())
            self.assertEqual(delivered, [])
            self.assertEqual(self.manifest()["status"], "exporting")
        finally:
            team.release.set()
            caller.join(10)
        self.assertFalse(caller.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(delivered, [901])
        team.engine.reset()
        team.wait_workers()
        team.backend.close()
        team.wait_workers()
        self.assertEqual([item["operation"] for item in team.commands],
                         ["inject", "reset", "close"])
        self.assertTrue(all(rank.closed for rank in team.ranks))
        manifest = self.manifest()
        self.assertEqual((manifest["first_token_id"],
                          manifest["delivered_tokens"],
                          manifest["reset_complete"]), (901, [901], True))

    def test_repeat_capture_preserves_prompt_generation_and_old_files(self):
        team = self.team()
        old_files = {}
        for index, length in enumerate((1, 129, 3)):
            incoming = request(length, index + 1)
            original = list(incoming.metadata["prompt_token_ids"])
            team.engine.inject(incoming)
            incoming.metadata["prompt_token_ids"][0] = 154879
            self.assertEqual(team.engine.decode(901 + index, max_tokens=1),
                             [901 + index])
            team.engine.reset()
            team.wait_workers()
            manifest = self.manifest(index)
            self.assertEqual(manifest["prompt_token_ids"], original)
            self.assertEqual(manifest["prompt_sha256"],
                             capture.prompt_digest(original))
            self.assertEqual(manifest["generation"], index + 1)
            self.assertEqual(manifest["capture_index"], index)
            self.assertTrue(all(rank.engine.position == 0
                                and rank.request_id is None
                                for rank in team.ranks))
            self.assertEqual(team.saved[0, "mtp"]["layer"]["latent"].shape,
                             (length - 1, 512))
            for receipt in manifest["ranks"]:
                for record in receipt["files"].values():
                    old_files[record["path"]] = record["sha256"]
            self.assertTrue(all(capture.digest(path) == digest
                                for path, digest in old_files.items()))
        team.backend.close()
        team.wait_workers()
        self.assertFalse(team.errors)

    def test_failed_rank_is_observed_by_every_rank_before_delivery(self):
        team = self.team(failed_rank=7)
        with self.assertRaisesRegex(RuntimeError, "rank 7"):
            team.engine.inject(request())
        team.wait_workers()
        self.assertEqual(set(team.errors), set(range(1, 16)))
        self.assertTrue(all("rank 7" in str(error)
                            for error in team.errors.values()))
        self.assertTrue(all(rank.engine.receipts[7]["status"] == "error"
                            and rank.engine.position == 0
                            for rank in team.ranks))
        self.assertEqual(self.manifest()["status"], "error")
        self.assertIsNone(self.manifest()["first_token_id"])
        self.assertTrue(all(path.suffix == ".pt" for path in
                            self.directory.glob("case-00/*/*")))
        count = len(team.commands)
        for operation in (team.engine.reset, team.backend.close):
            with self.assertRaisesRegex(RuntimeError, "restart all sixteen"):
                operation()
        self.assertEqual(len(team.commands), count)

    def test_stale_generation_fails_all_ranks_after_reset(self):
        team = self.team()
        team.engine.inject(request())
        team.engine.decode(901, max_tokens=1)
        team.engine.reset()
        team.wait_workers()
        with self.assertRaisesRegex(RuntimeError, "stale"):
            team.engine.inject(request())
        team.wait_workers()
        self.assertEqual(set(team.errors), set(range(1, 16)))
        self.assertEqual(self.manifest(1)["status"], "error")

    def test_receipts_bind_prompt_checkpoint_generation_and_all_ranks(self):
        team = self.team()
        incoming = request()
        team.engine.inject(incoming)
        receipts = team.ranks[0].engine.receipts
        for field, value in (("generation", 2), ("prompt_sha256", "x" * 64),
                             ("checkpoint_sha256", {}), ("rank", 0),
                             ("seq_len", 2), ("mtp_seq_len", 3)):
            altered = deepcopy(receipts)
            altered[15][field] = value
            with self.assertRaises(RuntimeError):
                capture.validate_receipts(altered, incoming, CHECKPOINT)
        with self.assertRaises(RuntimeError):
            capture.validate_receipts(receipts[:15], incoming, CHECKPOINT)
        team.engine.reset()
        team.wait_workers()
        team.backend.close()
        team.wait_workers()

    def test_only_one_prefill_token_and_one_active_capture(self):
        team = self.team()
        team.engine.inject(request())
        count = len(team.commands)
        with self.assertRaises(RuntimeError):
            team.engine.inject(request(generation=2))
        for token, budget in ((901, 2), (-1, 1), (154880, 1), (True, 1)):
            with self.assertRaises(ValueError):
                team.engine.decode(token, max_tokens=budget)
        self.assertEqual(len(team.commands), count)
        team.engine.decode(901, max_tokens=1)
        with self.assertRaises(RuntimeError):
            team.engine.decode(901, max_tokens=1)
        team.engine.reset()
        team.wait_workers()
        team.backend.close()
        team.wait_workers()


def source_hashes():
    names = (
        "tests/glm52/check_prefix_capture.py",
        "tilert/pd_vllm/prefix_capture.py", "tilert/pd_vllm/ascend_cache.py",
        "tilert/pd_vllm/ascend_receiver.py",
        "tilert/pd_vllm/ascend_layout.py", "tilert/pd_vllm/decode_timing.py",
        "tilert/models/glm_5_2_ascend/tp_runtime.py",
        "tilert/models/glm_5_2_ascend/session.py",
    )
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = source_hashes()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CaptureChecks)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    after = source_hashes()
    passed = result.wasSuccessful() and before == after
    passed = passed and "torch" not in sys.modules
    report = {
        "status": "pass" if passed else "fail", "tests": result.testsRun,
        "scope": "TP16 stdlib commands, symbolic rows and JSON fixture files",
        "numerical_execution": False, "tpot_measured": False,
        "python": platform.python_version(),
        "torch_imported": "torch" in sys.modules,
        "failures": len(result.failures), "errors": len(result.errors),
        "source_stable": before == after, "source_sha256": after,
        "test_output": stream.getvalue(),
    }
    capture.atomic_json(args.output, report)
    print(stream.getvalue())
    print(json.dumps({key: report[key] for key in (
        "status", "tests", "failures", "errors", "source_stable",
    )}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
