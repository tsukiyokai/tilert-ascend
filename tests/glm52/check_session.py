"""Check TP16 command ordering and greedy request lifecycle on the CPU."""

import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
from threading import Event
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.models.glm_5_2_ascend.session import (  # noqa: E402
    AscendPDEngine, PrefixRequest,
)
from tilert.models.glm_5_2_ascend.tp_runtime import (  # noqa: E402
    CoordinatedBackend, RankExecutor,
)
from tilert.pd_vllm.capabilities import (  # noqa: E402
    CapabilityUnavailable, InvalidParameter,
)
from tilert.pd_vllm.grammar_spec import GrammarUnsupported  # noqa: E402


class ScalarEngine:
    """Keep token rows and return a deterministic scalar successor."""

    def __init__(self, rank, capacity=9216):
        self.rank = rank
        self.capacity = capacity
        self.position = 0
        self.rows = []

    def inject(self, prefix):
        owner, rows = prefix
        if owner != self.rank:
            raise ValueError("Prefix came from another rank")
        self.rows = list(rows)
        self.position = len(rows)

    def step(self, token):
        if self.position == self.capacity:
            raise ValueError("Scalar cache is full")
        self.rows.append(token)
        self.position += 1
        return (token * 17 + self.position) % 257

    def reset(self):
        self.rows = []
        self.position = 0


def load_prefix(request, rank):
    if request.seq_len == 0:
        return None
    return rank, range(request.seq_len)


class FanoutChannel:
    """Apply real commands to fifteen CPU executors before rank zero."""

    def __init__(self, workers):
        self.workers = workers
        self.commands = []

    def send(self, command):
        self.commands.append(deepcopy(command))
        for worker in self.workers:
            worker.execute(deepcopy(command))


def make_session(capacity=9216, stop_ids=()):
    ranks = [RankExecutor(ScalarEngine(rank, capacity), rank, load_prefix)
             for rank in range(16)]
    channel = FanoutChannel(ranks[1:])
    backend = CoordinatedBackend(ranks[0], channel)
    session = AscendPDEngine(backend, stop_ids, select_token=lambda item: item)
    return session, ranks, channel


def serial_oracle(prefix, first, count, capacity, stop_ids=()):
    rows, output, token = list(range(prefix)), [], first
    while len(output) < count and len(rows) <= capacity:
        if token in stop_ids:
            break
        output.append(token)
        if len(output) == count or len(rows) == capacity:
            break
        rows.append(token)
        token = (token * 17 + len(rows)) % 257
    return output, rows


class SessionChecks(unittest.TestCase):
    def test_normal_import_does_not_load_torch(self):
        self.assertNotIn("torch", sys.modules)

    def test_token_and_cache_oracle_on_all_sixteen_ranks(self):
        for prefix in (0, 1, 127, 128, 8192):
            for count in (0, 1, 2, 4, 32):
                with self.subTest(prefix=prefix, count=count):
                    session, ranks, channel = make_session()
                    session.inject(PrefixRequest("request", prefix))
                    actual = session.decode(13, count)
                    expected, rows = serial_oracle(prefix, 13, count, 9216)
                    self.assertEqual(actual, expected)
                    for executor in ranks:
                        self.assertEqual(executor.engine.rows, rows)
                        self.assertEqual(executor.engine.position, len(rows))
                        self.assertEqual(executor.serial,
                                         len(channel.commands))
                    self.assertEqual(session.last_stats["step_calls"],
                                     max(0, len(actual) - 1))

    def test_first_token_needs_no_step(self):
        session, ranks, channel = make_session()
        session.inject(PrefixRequest("one", 8192))
        self.assertEqual(session.decode(11, 1), [11])
        self.assertEqual(len(channel.commands), 1)
        self.assertEqual(ranks[0].engine.position, 8192)

    def test_prefix_descriptor_has_no_tensor_or_other_rank_storage(self):
        session, _, channel = make_session()
        session.inject(PrefixRequest("id", 128, {"directory": "/prefix"}))
        self.assertEqual(channel.commands[0]["request"], {
            "request_id": "id", "seq_len": 128,
            "metadata": {"directory": "/prefix"},
        })

    def test_eos_is_not_emitted(self):
        session, ranks, _ = make_session(stop_ids=(11,))
        session.inject(PrefixRequest("eos", 8))
        self.assertEqual(session.decode(11, 4), [])
        self.assertEqual(session.last_stats["finish_reason"], "stop")
        self.assertTrue(all(rank.engine.position == 8 for rank in ranks))

    def test_eos_successor_consumes_predecessor_once(self):
        first, prefix = 11, 8
        stop = (first * 17 + prefix + 1) % 257
        session, ranks, _ = make_session(stop_ids=(stop,))
        session.inject(PrefixRequest("eos-successor", prefix))
        self.assertEqual(session.decode(first, 4), [first])
        self.assertEqual(session.last_stats["step_calls"], 1)
        self.assertTrue(all(rank.engine.position == 9 for rank in ranks))

    def test_ignore_eos(self):
        session, _, _ = make_session(stop_ids=(11,))
        session.inject(PrefixRequest("ignore", 8))
        self.assertEqual(len(session.decode(11, 4, {"ignore_eos": True})), 4)

    def test_capacity_preserves_last_output_outside_kv(self):
        for prefix in (0, 7, 8):
            session, ranks, _ = make_session(capacity=8)
            session.inject(PrefixRequest("full", prefix))
            output = session.decode(11, 20)
            expected, rows = serial_oracle(prefix, 11, 20, 8)
            self.assertEqual(output, expected)
            self.assertTrue(all(rank.engine.rows == rows for rank in ranks))
            self.assertEqual(ranks[0].engine.position, 8)

    def test_cancel_before_first_token(self):
        session, ranks, _ = make_session()
        session.inject(PrefixRequest("cancel", 8))
        cancel = Event()
        cancel.set()
        self.assertEqual(session.decode(11, 4, cancel_event=cancel), [])
        self.assertEqual(session.last_stats["finish_reason"], "cancelled")
        self.assertTrue(all(rank.engine.position == 8 for rank in ranks))

    def test_callback_cancellation_does_not_launch_another_step(self):
        session, ranks, _ = make_session()
        session.inject(PrefixRequest("cancel", 8))
        cancel, delivered = Event(), []

        def callback(token):
            delivered.append(token)
            cancel.set()

        self.assertEqual(session.decode(11, 4, on_token=callback,
                                        cancel_event=cancel), [11])
        self.assertEqual(delivered, [11])
        self.assertTrue(all(rank.engine.position == 8 for rank in ranks))

    def test_reset_then_new_request_replays_exactly(self):
        session, ranks, _ = make_session()
        outputs = []
        for index in range(3):
            session.inject(PrefixRequest(str(index), 128))
            outputs.append(session.decode(11, 32))
            session.reset()
            self.assertTrue(all(rank.engine.position == 0 for rank in ranks))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])

    def test_decode_requires_a_fresh_prefix(self):
        session, _, _ = make_session()
        with self.assertRaises(RuntimeError):
            session.decode(11, 4)
        session.inject(PrefixRequest("once", 1))
        session.decode(11, 4)
        with self.assertRaises(RuntimeError):
            session.decode(11, 4)

    def test_rejects_invalid_output_budget(self):
        session, _, _ = make_session()
        session.inject(PrefixRequest("budget", 1))
        for budget in (-1, True, 1.5):
            with self.assertRaises(InvalidParameter):
                session.decode(11, budget)

    def test_rejects_unsupported_sampling_before_advancing(self):
        session, ranks, channel = make_session()
        session.inject(PrefixRequest("sampling", 1))
        for sampling in ({"temperature": 1}, {"presence_penalty": 0.1},
                         {"frequency_penalty": 0.1}):
            with self.assertRaises(CapabilityUnavailable):
                session.decode(11, 4, sampling)
        for value in (float("nan"), float("inf")):
            with self.assertRaises(InvalidParameter):
                session.decode(11, 4, {"temperature": value})
        self.assertEqual(len(channel.commands), 1)
        self.assertTrue(all(rank.engine.position == 1 for rank in ranks))

    def test_accepts_greedy_top_one(self):
        session, _, _ = make_session()
        session.inject(PrefixRequest("top1", 1))
        self.assertEqual(len(session.decode(
            11, 4, {"temperature": 1, "top_k": 1}
        )), 4)

    def test_grammar_and_logprobs_rejected(self):
        session, _, _ = make_session()
        session.inject(PrefixRequest("features", 1))
        self.assertIsNone(session.prepare_grammar(None))
        with self.assertRaises(GrammarUnsupported):
            session.prepare_grammar({"type": "regex"})
        with self.assertRaises(CapabilityUnavailable):
            session.decode(11, 4, top_logprobs=1)

    def test_out_of_order_and_stale_position_rejected(self):
        executor = RankExecutor(ScalarEngine(0), 0, load_prefix)
        for serial, position in ((2, 0), (1, 1)):
            with self.assertRaises(ValueError):
                executor.execute({"serial": serial, "position": position,
                                  "operation": "reset"})
            self.assertEqual(executor.serial, 0)

    def test_uninjected_worker_cannot_step(self):
        executor = RankExecutor(ScalarEngine(0), 0, load_prefix)
        with self.assertRaises(RuntimeError):
            executor.execute({"serial": 1, "position": 0,
                              "operation": "step", "token": 11})

    def test_close_is_broadcast_to_every_rank(self):
        session, ranks, _ = make_session()
        session.backend.close()
        self.assertTrue(all(rank.closed for rank in ranks))

    def test_bad_prefix_length_rejected_before_broadcast(self):
        session, _, channel = make_session(capacity=8)
        for length in (-1, 9):
            with self.assertRaises(ValueError):
                session.inject(PrefixRequest("bad", length))
        self.assertEqual(channel.commands, [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SessionChecks)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    sources = [Path(__file__), *(ROOT / path for path in (
        "tilert/__init__.py", "tilert/tilert_init.py",
        "tilert/models/glm_5_2_ascend/__init__.py",
        "tilert/models/glm_5_2_ascend/session.py",
        "tilert/models/glm_5_2_ascend/tp_runtime.py",
    ))]
    report = {
        "status": "pass" if result.wasSuccessful() else "fail",
        "scope": "CPU scalar request/control; no CANN, HCCL or HIXL execution",
        "python": platform.python_version(), "tests": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "torch_imported": "torch" in sys.modules,
        "source_sha256": {str(path.relative_to(ROOT)):
                          hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
        "test_output": stream.getvalue(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(stream.getvalue())
    print(json.dumps({key: report[key] for key in (
        "status", "tests", "failures", "errors", "torch_imported",
    )}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
