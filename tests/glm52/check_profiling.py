"""Check capture scheduling with sixteen real rank executors and CPU oracles.

Fake profiler contexts validate thread ownership and marker coverage only.
No Torch import, NPU activity, profiler API execution or timing is performed.
"""

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import platform
from queue import Queue
import sys
from tempfile import TemporaryDirectory
import threading
import traceback
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_mtp_tp import (  # noqa: E402
    ScalarDraft, ScalarTarget, load_prefix, serial_oracle,
)
from profile_native_session import (  # noqa: E402
    bind_prefixes, main as profile_main, parse_arguments, trace_inventory,
)
from tilert.models.glm_5_2_ascend.profiling import (  # noqa: E402
    PROFILER_CONTRACT, StageMarkers, capture_phase, instrument_engine,
    reject_msprof_environment, run_phase,
)
from tilert.models.glm_5_2_ascend.session import PrefixRequest  # noqa: E402


SOURCES = (
    "tests/glm52/check_profiling.py",
    "tests/glm52/profile_native_session.py",
    "tests/glm52/precision_gate.py", "tests/glm52/prefix_io.py",
    "tests/glm52/check_mtp_tp.py",
    "tilert/models/glm_5_2_ascend/profiling.py",
    "tilert/models/glm_5_2_ascend/tp_runtime.py",
    "tilert/models/glm_5_2_ascend/session.py",
    "tilert/models/glm_5_2_ascend/mtp_session.py",
    "tilert/models/glm_5_2_ascend/mtp_round.py",
    "tilert/models/glm_5_2_ascend/mtp_state.py",
    "tilert/pd_vllm/ascend_cache.py", "tilert/pd_vllm/ascend_server.py",
)


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCES}


class Mailboxes:
    def __init__(self):
        self.queues = [Queue() for _ in range(16)]
        self.commands = []

    def channel(self, rank):
        mailboxes = self

        class Channel:
            def send(self, command):
                if rank != 0:
                    raise AssertionError("Only rank zero sends commands")
                mailboxes.commands.append(deepcopy(command))
                for queue in mailboxes.queues[1:]:
                    queue.put(deepcopy(command))

            def receive(self):
                return mailboxes.queues[rank].get(timeout=15)

        return Channel()


class FakeProfiler:
    def __init__(self, rank):
        self.rank = rank
        self.events = []
        self.thread = None
        self.active = False

    def __enter__(self):
        if self.active:
            raise AssertionError("Profiler started twice")
        self.thread = threading.get_ident()
        self.active = True
        self.events.append(("start", self.rank, self.thread))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.check_thread()
        self.events.append(("stop", self.rank, threading.get_ident()))
        self.active = False

    def check_thread(self):
        if not self.active or threading.get_ident() != self.thread:
            raise AssertionError("Decode ran outside its profiling thread")

    @contextmanager
    def marker(self, name):
        self.check_thread()
        yield
        self.check_thread()


def threaded_fixture(width, mode="accept", prefix=128, cancel_after=None,
                     eos_at=None):
    prompt = tuple((index * 7 + 3) % 251 for index in range(prefix))
    expected, _ = serial_oracle(prompt, 6, prefix + 8)
    stops = () if eos_at is None else (expected[eos_at],)
    cancelled = threading.Event()
    delivered = []

    def callback(token):
        delivered.append(token)
        if len(delivered) == cancel_after:
            cancelled.set()

    barrier = threading.Barrier(16, timeout=15)
    mailboxes = Mailboxes()
    results, failures = {}, []
    lock = threading.Lock()

    def execute_rank(rank):
        try:
            markers = StageMarkers(rank)
            engine = instrument_engine(ScalarTarget(prefix + 8),
                                       ScalarDraft(prefix + 8, mode), markers)
            request = PrefixRequest("request", prefix, {"prompt": prompt})
            arguments = (engine, rank, mailboxes.channel(rank), load_prefix,
                         request, width, 6, markers)
            warmup = run_phase(*arguments, requests=3, ignore_eos=True)
            barrier.wait()
            if rank == 0 and cancel_after == 0:
                cancelled.set()
            markers.phase = "capture"
            profiler = FakeProfiler(rank)
            markers.range_factory = profiler.marker
            capture = capture_phase(
                profiler, markers, barrier.wait,
                lambda: run_phase(*arguments, stop_ids=stops,
                                  on_token=callback,
                                  cancel_event=cancelled),
            )
            with lock:
                results[rank] = {
                    "warmup": warmup, "capture": capture,
                    "events": markers.events, "profiler": profiler.events,
                    "position": engine.position,
                    "draft_position": engine.draft.position,
                }
        except BaseException:
            with lock:
                failures.append((rank, traceback.format_exc()))
            barrier.abort()

    threads = [threading.Thread(target=execute_rank, args=(rank,),
                                name=f"rank-{rank}") for rank in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    if any(thread.is_alive() for thread in threads):
        raise AssertionError("Rank command loop did not terminate")
    if failures:
        raise AssertionError(failures)
    end = len(expected)
    if eos_at is not None:
        end = min(end, eos_at)
    if cancel_after is not None:
        end = min(end, cancel_after)
    return results, expected[:end], mailboxes.commands


class ProfilingChecks(unittest.TestCase):
    fixtures = 0
    thread_contexts = 0

    def check_fixture(self, results, expected, commands):
        self.assertEqual(set(results), set(range(16)))
        threads = set()
        for rank, result in results.items():
            start, stop = result["profiler"]
            self.assertEqual(start[:2], ("start", rank))
            self.assertEqual(stop, ("stop", rank, start[2]))
            threads.add(start[2])
            events = [event for event in result["events"]
                      if event["phase"] == "capture"]
            self.assertTrue(events)
            stack = []
            for event in events:
                self.assertEqual(event["thread_id"], start[2])
                if event["event"] == "begin":
                    stack.append(event["name"])
                else:
                    self.assertEqual(stack.pop(), event["name"])
            self.assertEqual(stack, [])
            self.assertEqual(result["position"], 0)
            self.assertEqual(result["draft_position"], 0)
            self.assertEqual(result["capture"]["command_serial"],
                             results[0]["capture"]["command_serial"])
            warmup = [event for event in result["events"]
                      if event["phase"] == "warmup"]
            self.assertEqual(sum(event["name"] == "prompt_replay"
                                 and event["event"] == "begin"
                                 for event in warmup), 3)
        self.assertEqual(len(threads), 16)
        root = results[0]["capture"]["requests"][0]
        self.assertEqual(root["tokens"], expected)
        calls = root.get("round_calls", root.get("step_calls", 0))
        self.assertLessEqual(calls, 5)
        self.assertEqual(sum(command["operation"] == "close"
                             for command in commands), 2)
        type(self).fixtures += 1
        type(self).thread_contexts += 16

    def test_normal_import_and_profiler_contract(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("torch_npu", sys.modules)
        self.assertEqual(PROFILER_CONTRACT["level"], 1)
        self.assertIn("HCCL_TRACE", PROFILER_CONTRACT["level_flags"])
        self.assertFalse(PROFILER_CONTRACT["analyse"])

    def test_real_sixteen_rank_sessions_preserve_output(self):
        for width in (1, 2, 4):
            for mode in ("accept", "reject", "mixed"):
                with self.subTest(width=width, mode=mode):
                    self.check_fixture(*threaded_fixture(width, mode))

    def test_round_markers_include_seed_chain_verify_and_commit(self):
        results, expected, commands = threaded_fixture(4, prefix=8192)
        self.check_fixture(results, expected, commands)
        for result in results.values():
            names = [event["name"] for event in result["events"]
                     if event["phase"] == "capture"
                     and event["event"] == "begin"]
            self.assertEqual(names.count("seed"), 1)
            self.assertEqual(names.count("prompt_replay"), 1)
            self.assertEqual(names.count("verify"), names.count("round"))
            self.assertEqual(names.count("conditioned"), names.count("round"))
            self.assertEqual(names.count("commit"), names.count("round"))
            self.assertEqual(names.count("chain"), 4)
            first_round = names[names.index("round"):names.index("finish")]
            self.assertEqual(first_round[:7],
                             ["round", "seed", "chain", "chain", "verify",
                              "conditioned", "commit"])
            self.assertLess(names.index("prompt_replay"), names.index("seed"))
            self.assertLess(names.index("finish"), names.index("reset"))

    def test_cancellation_closes_every_context_and_worker(self):
        for count in (0, 1, 2, 5):
            with self.subTest(delivered=count):
                results, expected, commands = threaded_fixture(
                    4, cancel_after=count,
                )
                self.check_fixture(results, expected, commands)
                root = results[0]["capture"]["requests"][0]
                self.assertEqual(root["finish_reason"], "cancelled")
                if count < 2:
                    self.assertEqual(root["round_calls"], 0)

    def test_eos_ends_after_the_atomic_round(self):
        for position in (0, 1, 3):
            with self.subTest(eos_at=position):
                results, expected, commands = threaded_fixture(
                    4, eos_at=position,
                )
                self.check_fixture(results, expected, commands)
                self.assertEqual(results[0]["capture"]["requests"][0]
                                 ["finish_reason"], "stop")

    def test_msprof_environment_conflicts_are_explicit(self):
        reject_msprof_environment({})
        for name in ("PROFILING_MODE", "PROFILER_SAMPLECONFIG"):
            with self.assertRaisesRegex(RuntimeError, name):
                reject_msprof_environment({name: "enabled"})

    def test_prefix_hash_and_replay_token_binding(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("main", "mtp"):
                (root / name).mkdir()
                (root / name / "rank-0.pt").write_bytes(name.encode())
            arguments = self.arguments(root)
            args = parse_arguments(arguments)
            hashes = {name: hashlib.sha256(name.encode()).hexdigest()
                      for name in ("main", "mtp")}
            gate = {"prefix_sha256": {"0": dict(hashes)},
                    "last_prompt_token": 17}
            self.assertEqual(bind_prefixes(args, gate, 0)[1], hashes)
            gate["last_prompt_token"] = 18
            with self.assertRaisesRegex(ValueError, "replay token"):
                bind_prefixes(args, gate, 0)
            gate["last_prompt_token"] = None
            gate["prefix_sha256"]["0"]["mtp"] = None
            self.assertEqual(bind_prefixes(args, gate, 0)[1], hashes)
            (root / "main/rank-0.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "main prefix"):
                bind_prefixes(args, gate, 0)

    @staticmethod
    def arguments(root):
        return ["--checkpoint", str(root / "checkpoint"), "--precision-dir",
                str(root / "precision"), "--prefix", str(root / "main"),
                "--mtp-prefix", str(root / "mtp"), "--last-prompt-token",
                "17", "--output", str(root / "output"),
                "--source-commit", "cpu-fixture"]

    def test_missing_precision_stops_before_torch_import(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "checkpoint").mkdir()
            for name in ("config.json", "quant_model_description.json",
                         "quant_model_weights.safetensors.index.json"):
                (root / "checkpoint" / name).write_text("{}")
            with redirect_stdout(io.StringIO()):
                status = profile_main(self.arguments(root))
            report = json.loads((root / "output/rank-0.json").read_text())
            self.assertEqual(status, 1)
            self.assertIn("Precision report is unavailable", report["error"])
            self.assertFalse(report["tpot_measured"])
            self.assertNotIn("torch", sys.modules)

    def test_capture_budget_is_bounded(self):
        with TemporaryDirectory() as temporary:
            arguments = self.arguments(Path(temporary))
            for tokens in (1, 7, 100):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_arguments([*arguments, "--max-tokens",
                                         str(tokens)])

    def test_raw_inventory_requires_device_payload(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            framework = root / "rank_ascend_pt/FRAMEWORK/torch.op_range"
            framework.parent.mkdir(parents=True)
            framework.write_bytes(b"synthetic CPU metadata")
            device = root / "rank_ascend_pt/PROF_1_2_A/device_0"
            device.mkdir(parents=True)
            (device / "info.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "device data"):
                trace_inventory(root)
            (device / "data").mkdir()
            (device / "data/test.slice_0").write_bytes(b"synthetic payload")
            self.assertEqual(len(trace_inventory(root)), 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = source_hashes()
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProfilingChecks)
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    after = source_hashes()
    report = {
        "scope": "CPU scheduling/markers; profiler and NPU not executed",
        "passed": result.wasSuccessful() and before == after
        and "torch" not in sys.modules,
        "tests": result.testsRun, "log": log.getvalue(),
        "fixtures": ProfilingChecks.fixtures,
        "thread_contexts": ProfilingChecks.thread_contexts,
        "torch_imported": "torch" in sys.modules,
        "torch_npu_imported": "torch_npu" in sys.modules,
        "source_stable": before == after, "source_sha256": after,
        "python": platform.python_version(), "hostname": platform.node(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: report[name] for name in (
        "passed", "tests", "fixtures", "thread_contexts", "source_stable",
        "torch_imported",
    )}))
    if not report["passed"]:
        print(log.getvalue())
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
