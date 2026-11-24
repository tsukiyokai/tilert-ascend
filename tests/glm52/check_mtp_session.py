"""Check MTP serving delivery through sixteen real rank command executors."""

import argparse
import hashlib
import io
from itertools import product
import json
from pathlib import Path
import platform
import sys
from threading import Event
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

from check_mtp_tp import make_backend  # noqa: E402
from tilert.models.glm_5_2_ascend.mtp_session import (  # noqa: E402
    AscendMtpPDEngine,
)
from tilert.models.glm_5_2_ascend.session import (  # noqa: E402
    AscendPDEngine, PrefixRequest,
)
from tilert.pd_vllm.ascend_server import create_session  # noqa: E402
from tilert.pd_vllm.capabilities import (  # noqa: E402
    CapabilityUnavailable, InvalidParameter,
)


SOURCES = (
    "tests/glm52/check_mtp_session.py",
    "tests/glm52/check_mtp_tp.py",
    "tilert/__init__.py",
    "tilert/tilert_init.py",
    "tilert/models/glm_5_2_ascend/__init__.py",
    "tilert/models/glm_5_2_ascend/session.py",
    "tilert/models/glm_5_2_ascend/mtp_session.py",
    "tilert/models/glm_5_2_ascend/mtp_state.py",
    "tilert/models/glm_5_2_ascend/mtp_round.py",
    "tilert/models/glm_5_2_ascend/tp_runtime.py",
    "tilert/pd_vllm/ascend_cache.py",
    "tilert/pd_vllm/ascend_server.py",
    "tilert/pd_vllm/capabilities.py",
    "tilert/pd_vllm/grammar_spec.py",
)


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCES}


def serial_oracle(prompt, count, capacity, stops=(), cancel_after=None):
    """Consume one row at a time, without any MTP state or draft proposal."""
    rows, digest = [], 0
    for token in prompt:
        digest = (digest * 1000003 + token + 17) % 2147483647
        rows.append((token, digest))
    token = (digest * 19 + len(rows) * 7) % 251 + 1
    output, reason = [], "length"
    limit = min(count, capacity - len(prompt) + 1)
    while len(output) < limit:
        if cancel_after is not None and len(output) == cancel_after:
            reason = "cancelled"
            break
        if token in stops:
            reason = "stop"
            break
        output.append(token)
        if len(output) == limit:
            break
        if cancel_after is not None and len(output) == cancel_after:
            reason = "cancelled"
            break
        digest = (digest * 1000003 + token + 17) % 2147483647
        rows.append((token, digest))
        token = (digest * 19 + len(rows) * 7) % 251 + 1
    return output, rows, reason


def make_session(prefix=128, width=4, mode="accept", room=32, stops=(),
                 rounds_per_command=1):
    backend, ranks, channel = make_backend(mode, prefix + room)
    session = create_session(backend, width, stops,
                             rounds_per_command=rounds_per_command)
    prompt = tuple((row * 7 + 3) % 251 for row in range(prefix))
    session.inject(PrefixRequest("request", prefix, {"prompt": prompt}))
    first = ranks[0].engine.first_token_id
    return session, ranks, channel, prompt, first


def operations(channel):
    return [command["operation"] for command in channel.commands]


class MtpSessionChecks(unittest.TestCase):
    coverage = {
        "fixtures": 0, "round_calls": 0, "verified_positions": 0,
        "logical_accepted": 0, "output_tokens": 0,
    }
    widths = set()
    accepted_spans = set()

    def check_result(self, session, ranks, channel, actual, expected):
        tokens, rows, reason = expected
        self.assertEqual(actual, tokens)
        stats = session.last_stats
        self.assertEqual(stats["finish_reason"], reason)
        self.assertEqual(stats["output_tokens"], len(tokens))
        self.assertEqual(stats["committed_position"], len(rows))
        self.assertEqual(stats["accepted"], sum(
            item["accepted"] for item in stats["accepted_spans"]
        ))
        self.assertEqual(stats["verified_positions"], sum(
            item["width"] for item in stats["accepted_spans"]
        ))
        self.assertEqual(sum(item["output_tokens"]
                             for item in stats["accepted_spans"]),
                         max(0, len(tokens) - 1))
        for rank in ranks:
            target, draft = rank.engine.target, rank.engine.draft
            self.assertEqual(target.rows[:target.position], rows)
            self.assertEqual(target.position, len(rows))
            self.assertLessEqual(draft.position, target.position)
            for position, (token, parent) in enumerate(
                draft.rows[:draft.position]
            ):
                self.assertEqual(parent.kind, "target")
                self.assertEqual(parent.length, position + 1)
                self.assertEqual(parent.digest, rows[position][1])
                if position + 1 < len(rows):
                    self.assertEqual(token, rows[position + 1][0])
                else:
                    predicted = (
                        parent.digest * 19 + parent.length * 7
                    ) % 251 + 1
                    self.assertEqual(token, predicted)
            self.assertIsNone(rank.engine._mtp)
            self.assertEqual(rank.serial, len(channel.commands))
        commands = operations(channel)
        self.assertNotIn("step", commands)
        self.assertEqual(commands.count("start_mtp"),
                         commands.count("finish_mtp"))
        batch_commands = (commands.count("advance_mtp")
                          + commands.count("advance_mtp_n"))
        chunk = getattr(session, "rounds_per_command", 1)
        self.assertLessEqual(batch_commands, stats["round_calls"])
        self.assertLessEqual(stats["round_calls"], batch_commands * chunk)
        self.coverage["fixtures"] += 1
        for name in ("round_calls", "verified_positions", "output_tokens"):
            self.coverage[name] += stats[name]
        self.coverage["logical_accepted"] += stats["accepted"]
        for item in stats["accepted_spans"]:
            self.widths.add(item["width"])
            self.accepted_spans.add((item["width"], item["accepted"]))

    def test_factory_import_and_width_selection(self):
        backend, _, _ = make_backend()
        self.assertIs(type(create_session(backend, 1, ())), AscendPDEngine)
        for width in (2, 4):
            self.assertIsInstance(create_session(backend, width, ()),
                                  AscendMtpPDEngine)
        self.assertNotIn("torch", sys.modules)

    def test_serial_output_and_cache_for_accept_reject_and_mixed(self):
        for prefix, width, mode, budget in product(
            (1, 128, 8192), (2, 4), ("accept", "reject", "mixed"),
            (0, 1, 2, 5, 9),
        ):
            with self.subTest(prefix=prefix, width=width, mode=mode,
                              budget=budget):
                session, ranks, channel, prompt, first = make_session(
                    prefix, width, mode,
                )
                delivered = []
                actual = session.decode(first, budget,
                                        on_token=delivered.append)
                expected = serial_oracle(prompt, budget, prefix + 32)
                self.assertEqual(delivered, actual)
                self.check_result(session, ranks, channel, actual, expected)

    def test_batched_rounds_deliver_the_single_round_stream(self):
        for chunk, mode, budget in product(
            (2, 3, 5), ("accept", "reject", "mixed"), (5, 9),
        ):
            with self.subTest(chunk=chunk, mode=mode, budget=budget):
                session, ranks, channel, prompt, first = make_session(
                    mode=mode, rounds_per_command=chunk,
                )
                delivered = []
                actual = session.decode(first, budget,
                                        on_token=delivered.append)
                expected = serial_oracle(prompt, budget, 160)
                self.assertEqual(delivered, actual)
                self.check_result(session, ranks, channel, actual, expected)
                commands = operations(channel)
                self.assertEqual(commands.count("advance_mtp"), 0)
                self.assertLess(commands.count("advance_mtp_n"),
                                session.last_stats["round_calls"] + 1)

    def test_zero_and_one_output_do_not_start_or_finish_mtp(self):
        for budget in (0, 1):
            session, ranks, channel, prompt, first = make_session()
            output = session.decode(first, budget)
            self.assertEqual(operations(channel), ["inject"])
            self.assertTrue(all(rank.engine.draft.position == 127
                                for rank in ranks))
            self.check_result(session, ranks, channel, output,
                              serial_oracle(prompt, budget, 160))

    def test_zero_and_one_budget_eos_cancel_precedence(self):
        for budget, room, stop, cancelled in product(
            (0, 1), (0, 1), (False, True), (False, True),
        ):
            with self.subTest(budget=budget, room=room, stop=stop,
                              cancelled=cancelled):
                session, ranks, channel, prompt, first = make_session(
                    room=room,
                )
                stops = (first,) if stop else ()
                session.stop_ids = frozenset(stops)
                cancel = Event()
                if cancelled:
                    cancel.set()
                actual = session.decode(first, budget, cancel_event=cancel)
                expected = serial_oracle(
                    prompt, budget, 128 + room, stops,
                    cancel_after=0 if cancelled else None,
                )
                self.check_result(session, ranks, channel, actual, expected)
                self.assertEqual(operations(channel), ["inject"])

    def test_capacity_tail_uses_width_one_or_two(self):
        seen = set()
        for room, width, mode in product(
            range(8), (2, 4), ("accept", "reject", "mixed"),
        ):
            with self.subTest(room=room, width=width, mode=mode):
                session, ranks, channel, prompt, first = make_session(
                    128, width, mode, room,
                )
                actual = session.decode(first, 32)
                self.check_result(
                    session, ranks, channel, actual,
                    serial_oracle(prompt, 32, 128 + room),
                )
                seen.update(item["width"]
                            for item in session.last_stats["accepted_spans"])
        self.assertEqual(seen, {1, 2, 4})

    def test_eos_at_first_and_every_accepted_position(self):
        for eos_at, mode in product(range(5), ("accept", "reject", "mixed")):
            with self.subTest(eos_at=eos_at, mode=mode):
                session, ranks, channel, prompt, first = make_session(
                    mode=mode,
                )
                full, _, _ = serial_oracle(prompt, 5, 160)
                self.assertEqual(len(set(full)), 5)
                stop = full[eos_at]
                session.stop_ids = frozenset((stop,))
                delivered = []
                actual = session.decode(first, 12,
                                        on_token=delivered.append)
                self.assertEqual(actual, full[:eos_at])
                self.assertEqual(delivered, actual)
                self.check_result(
                    session, ranks, channel, actual,
                    serial_oracle(prompt, 12, 160, (stop,)),
                )
                if eos_at == 0:
                    self.assertEqual(operations(channel), ["inject"])
                else:
                    self.assertEqual(session.last_stats["accepted"], eos_at)

    def test_ignore_eos_clears_backend_stop_set(self):
        session, ranks, channel, prompt, first = make_session()
        full, _, _ = serial_oracle(prompt, 9, 160)
        session.stop_ids = frozenset(full[:5])
        actual = session.decode(first, 9, sampling={"ignore_eos": True})
        self.check_result(session, ranks, channel, actual,
                          serial_oracle(prompt, 9, 160))
        start = channel.commands[1]
        self.assertEqual(start["eos_tokens"], ())
        self.assertEqual(start["max_tokens"], 9)

    def test_callback_cancellation_at_every_output_position(self):
        for prefix, width, mode, cancel_after in product(
            (1, 128, 8192), (2, 4), ("accept", "reject", "mixed"),
            range(10),
        ):
            with self.subTest(prefix=prefix, width=width, mode=mode,
                              cancel_after=cancel_after):
                session, ranks, channel, prompt, first = make_session(
                    prefix, width, mode,
                )
                cancel, delivered = Event(), []
                if cancel_after == 0:
                    cancel.set()

                def callback(token):
                    delivered.append(token)
                    if len(delivered) == cancel_after:
                        cancel.set()

                actual = session.decode(first, 9, on_token=callback,
                                        cancel_event=cancel)
                expected = serial_oracle(
                    prompt, 9, prefix + 32, cancel_after=cancel_after,
                )
                self.assertEqual(delivered, actual)
                self.check_result(session, ranks, channel, actual, expected)
                if cancel_after <= 1:
                    self.assertEqual(operations(channel), ["inject"])

    def test_cancellation_inside_a_round_finishes_all_ranks_then_trims(self):
        for mode in ("accept", "reject", "mixed"):
            session, ranks, channel, prompt, first = make_session(mode=mode)
            cancel, delivered = Event(), []
            verify = ranks[0].engine.target.verify

            def cancel_during_verify(tokens, **kwargs):
                result = verify(tokens, **kwargs)
                cancel.set()
                return result

            ranks[0].engine.target.verify = cancel_during_verify
            actual = session.decode(first, 9, on_token=delivered.append,
                                    cancel_event=cancel)
            expected, rows, _ = serial_oracle(prompt, 1, 160)
            self.check_result(session, ranks, channel, actual,
                              (expected, rows, "cancelled"))
            self.assertEqual(delivered, [first])
            self.assertEqual(operations(channel),
                             ["inject", "start_mtp", "advance_mtp_n",
                              "finish_mtp"])
            self.assertEqual(session.last_stats["completed_rounds"], 1)
            self.assertGreater(session.last_stats["accepted"], 0)
            self.assertEqual(session.last_stats["accepted_spans"][0]
                             ["output_tokens"], 0)

    def test_cancellation_after_start_preserves_missing_draft_seed(self):
        session, ranks, channel, prompt, first = make_session()
        cancel = Event()
        start = session.backend.start_mtp

        def cancel_after_start(*args):
            start(*args)
            cancel.set()

        session.backend.start_mtp = cancel_after_start
        actual = session.decode(first, 9, cancel_event=cancel)
        expected, rows, _ = serial_oracle(prompt, 1, 160)
        self.check_result(session, ranks, channel, actual,
                          (expected, rows, "cancelled"))
        self.assertEqual(operations(channel),
                         ["inject", "start_mtp", "finish_mtp"])
        self.assertTrue(all(rank.engine.draft.position == 127
                            for rank in ranks))

    def test_callback_exception_excludes_failed_delivery_and_trims(self):
        for fail_at in range(1, 6):
            with self.subTest(fail_at=fail_at):
                session, ranks, channel, prompt, first = make_session()
                delivered, attempts = [], []

                def callback(token):
                    attempts.append(token)
                    if len(attempts) == fail_at:
                        raise LookupError("callback failed")
                    delivered.append(token)

                with self.assertRaisesRegex(LookupError, "callback failed"):
                    session.decode(first, 9, on_token=callback)
                expected, rows, _ = serial_oracle(
                    prompt, 9, 160, cancel_after=fail_at - 1,
                )
                self.check_result(session, ranks, channel, delivered,
                                  (expected, rows, "error"))
                self.assertEqual(len(attempts), fail_at)
                self.assertFalse(session._decoding)
                self.assertTrue(session._finished)

    def test_rank_start_failure_does_not_send_finish(self):
        session, ranks, channel, _, first = make_session()

        def fail_start(*args):
            raise LookupError("rank zero start failed")

        ranks[0].engine.start_mtp = fail_start
        with self.assertRaisesRegex(LookupError, "rank zero start failed"):
            session.decode(first, 9)
        self.assertTrue(session.backend.failed)
        self.assertEqual(operations(channel), ["inject", "start_mtp"])
        self.assertIsNone(session.last_stats["finish_error"])
        self.assertEqual(session.last_stats["finish_reason"], "error")

    def test_rank_round_failure_preserves_original_error(self):
        session, ranks, channel, _, first = make_session()

        def fail_round():
            raise LookupError("rank zero verification failed")

        ranks[0].engine.advance_mtp = fail_round
        with self.assertRaisesRegex(LookupError,
                                    "rank zero verification failed"):
            session.decode(first, 9)
        self.assertTrue(session.backend.failed)
        self.assertEqual(operations(channel),
                         ["inject", "start_mtp", "advance_mtp_n"])
        self.assertIn("TP command stream failed",
                      session.last_stats["finish_error"])
        self.assertEqual(session.last_stats["finish_reason"], "error")
        self.assertEqual(session.last_stats["round_calls"], 1)
        self.assertEqual(session.last_stats["completed_rounds"], 0)
        with self.assertRaisesRegex(RuntimeError, "TP command stream failed"):
            session.reset()
        self.assertEqual(len(channel.commands), 3)

    def test_first_finish_failure_is_propagated(self):
        session, ranks, channel, _, first = make_session()
        delivered = []

        def fail_finish(retained):
            raise ArithmeticError("rank zero finish failed")

        ranks[0].engine.finish_mtp = fail_finish
        with self.assertRaisesRegex(ArithmeticError,
                                    "rank zero finish failed"):
            session.decode(first, 5, on_token=delivered.append)
        self.assertTrue(session.backend.failed)
        self.assertEqual(operations(channel),
                         ["inject", "start_mtp", "advance_mtp_n",
                          "finish_mtp"])
        self.assertEqual(session.last_stats["output_tokens"], len(delivered))
        self.assertEqual(session.last_stats["finish_reason"], "error")
        self.assertEqual(session.last_stats["finish_error"],
                         "ArithmeticError: rank zero finish failed")

    def test_callback_error_survives_a_failed_finish(self):
        session, ranks, channel, _, first = make_session()
        delivered = []

        def callback(token):
            if delivered:
                raise LookupError("delivery failed")
            delivered.append(token)

        def fail_finish(retained):
            raise ArithmeticError("rank zero finish failed")

        ranks[0].engine.finish_mtp = fail_finish
        with self.assertRaisesRegex(LookupError, "delivery failed"):
            session.decode(first, 9, on_token=callback)
        self.assertTrue(session.backend.failed)
        self.assertEqual(operations(channel)[-1], "finish_mtp")
        self.assertEqual(delivered, [first])
        self.assertEqual(session.last_stats["output_tokens"], 1)
        self.assertEqual(session.last_stats["finish_error"],
                         "ArithmeticError: rank zero finish failed")

    def test_logical_accepted_count_includes_undelivered_round_tail(self):
        session, ranks, channel, prompt, first = make_session()
        cancel, delivered, logical = Event(), [], []
        finish = session.backend.finish_mtp

        def snapshot_then_finish(retained):
            logical.append(ranks[0].engine._mtp.state.emitted_count)
            finish(retained)

        def callback(token):
            delivered.append(token)
            if len(delivered) == 2:
                cancel.set()

        session.backend.finish_mtp = snapshot_then_finish
        actual = session.decode(first, 9, on_token=callback,
                                cancel_event=cancel)
        self.check_result(session, ranks, channel, actual,
                          serial_oracle(prompt, 9, 160, cancel_after=2))
        self.assertEqual(logical, [5])
        self.assertEqual(session.last_stats["accepted"], 4)
        self.assertEqual(session.last_stats["output_tokens"], 2)

    def test_cancel_before_eos_observation_wins(self):
        session, ranks, channel, prompt, first = make_session()
        full, _, _ = serial_oracle(prompt, 5, 160)
        session.stop_ids = frozenset((full[2],))
        cancel, delivered = Event(), []

        def callback(token):
            delivered.append(token)
            if len(delivered) == 2:
                cancel.set()

        actual = session.decode(first, 9, on_token=callback,
                                cancel_event=cancel)
        self.check_result(
            session, ranks, channel, actual,
            serial_oracle(prompt, 9, 160, (full[2],), cancel_after=2),
        )
        self.assertEqual(session.last_stats["accepted"], 2)

    def test_eos_observed_before_late_cancel_keeps_stop_reason(self):
        session, ranks, channel, prompt, first = make_session()
        full, _, _ = serial_oracle(prompt, 5, 160)
        session.stop_ids = frozenset((full[1],))
        cancel = Event()
        finish = session.backend.finish_mtp

        def cancel_during_finish(retained):
            cancel.set()
            finish(retained)

        session.backend.finish_mtp = cancel_during_finish
        actual = session.decode(first, 9, cancel_event=cancel)
        self.check_result(session, ranks, channel, actual,
                          serial_oracle(prompt, 9, 160, (full[1],)))

    def test_reset_after_partial_delivery_replays_a_fresh_request(self):
        session, ranks, channel, prompt, first = make_session()
        expected, _, _ = serial_oracle(prompt, 9, 160)
        for number in range(3):
            if number:
                session.inject(PrefixRequest(str(number), len(prompt),
                                             {"prompt": prompt}))
            cancel, delivered = Event(), []

            def callback(token):
                delivered.append(token)
                if number == 0 and len(delivered) == 2:
                    cancel.set()

            output = session.decode(first, 9, on_token=callback,
                                    cancel_event=cancel)
            self.assertEqual(output, expected[:2] if number == 0 else expected)
            self.assertTrue(all(rank.engine._mtp is None for rank in ranks))
            session.reset()
            self.assertEqual(session.last_stats, {})
            self.assertTrue(all(rank.engine.target.position == 0
                                and rank.engine.draft.position == 0
                                and rank.engine.parent_hidden is None
                                and rank.request_id is None
                                for rank in ranks))
        self.assertTrue(all(rank.serial == len(channel.commands)
                            for rank in ranks))

    def test_active_callback_cannot_reset_or_decode_recursively(self):
        session, _, _, _, first = make_session()

        def callback(token):
            with self.assertRaises(RuntimeError):
                session.reset()
            with self.assertRaises(RuntimeError):
                session.decode(first, 1)

        self.assertEqual(len(session.decode(first, 5, on_token=callback)), 5)
        with self.assertRaises(RuntimeError):
            session.decode(first, 1)

    def test_invalid_width_prefix_and_sampling_do_not_start(self):
        backend, _, channel = make_backend()
        for width in (1, 3, 17, True, 2.0):
            with self.assertRaises(ValueError):
                AscendMtpPDEngine(backend, width)
        session = AscendMtpPDEngine(backend)
        with self.assertRaises(ValueError):
            session.inject(PrefixRequest("empty", 0))
        self.assertEqual(channel.commands, [])
        session, _, channel, _, first = make_session()
        for budget in (-1, True, 1.5):
            with self.assertRaises(InvalidParameter):
                session.decode(first, budget)
        for sampling in ({"temperature": 1}, {"presence_penalty": 0.1},
                         {"frequency_penalty": 0.1}):
            with self.assertRaises(CapabilityUnavailable):
                session.decode(first, 9, sampling)
        for kwargs in ({"grammar_session": object()}, {"top_logprobs": 1}):
            with self.assertRaises(CapabilityUnavailable):
                session.decode(first, 9, **kwargs)
        self.assertEqual(operations(channel), ["inject"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = source_hashes()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MtpSessionChecks)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    after = source_hashes()
    success = result.wasSuccessful() and before == after
    success = success and "torch" not in sys.modules
    report = {
        "status": "pass" if success else "fail",
        "scope": "stdlib scalar TP16 MTP serving; no Torch, CANN or HCCL",
        "python": platform.python_version(), "tests": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "torch_imported": "torch" in sys.modules,
        "source_stable_during_execution": before == after,
        "source_sha256": after,
        "coverage": {
            **MtpSessionChecks.coverage,
            "verify_widths": sorted(MtpSessionChecks.widths),
            "accepted_spans": sorted(MtpSessionChecks.accepted_spans),
        },
        "test_output": stream.getvalue(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(stream.getvalue())
    print(json.dumps({key: report[key] for key in (
        "status", "tests", "failures", "errors", "torch_imported",
        "coverage",
    )}))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
