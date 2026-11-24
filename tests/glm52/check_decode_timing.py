"""Check TP16 callback timing and NDJSON accounting using only stdlib."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_pd_decode as bench  # noqa: E402
from check_mtp_tp import make_backend  # noqa: E402
from prefix_io import prompt_identity  # noqa: E402
from tilert.models.glm_5_2_ascend.session import PrefixRequest  # noqa: E402
from tilert.pd_vllm import ascend_server  # noqa: E402
from tilert.pd_vllm.ascend_server import create_session  # noqa: E402
from tilert.pd_vllm.decode_timing import (  # noqa: E402
    TimedBackend, TimedEngine, load_timing, timing_path, token_metrics,
)


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        self.now += 1000000
        return self.now


def wire_message(value):
    return json.dumps(value).encode() + b"\n"


def terminal(count=3):
    return {"done": True, "n": count, "seq_len": 8192,
            "finish_reason": "length"}


def benchmark_request(tokens, number=0, producer_ns=1000000,
                      client_ns=1000000):
    return {
        "rid": str(number), "client_token_ids": list(tokens),
        "producer_report": {"token_ids": list(tokens),
                            "returned_token_ids": list(tokens)},
        "diversity": bench.output_diversity(tokens),
        "producer": token_metrics(tokens, [i * producer_ns
                                          for i in range(len(tokens))]),
        "client": token_metrics(tokens, [i * client_ns
                                        for i in range(len(tokens))]),
        "prompt_sha256": bench.prompt_digest([number % 8] * 8192),
    }


def save_request(directory, name, request):
    bench.atomic_json(directory / name, request)
    return hashlib.sha256((directory / name).read_bytes()).hexdigest()


def benchmark_summary(requests, precision, directory, warmup=3, policy=None):
    names = [f"request-{number:04d}.json" for number in range(len(requests))]
    (directory / "diversity-policy.json").unlink(missing_ok=True)
    return {
        **{source: bench.aggregate(requests, source)
           for source in ("producer", "client")},
        "requests": names,
        "request_sha256": {name: save_request(directory, name, request)
                           for name, request in zip(names, requests)},
        "diversity_policy": bench.freeze_diversity_policy(policy, directory),
        "warmup_requests": list(range(warmup)), "precision": precision,
        "measured_prompt_counts": dict(Counter(
            item["prompt_sha256"] for item in requests
        )),
    }


class CpuServingEngine(ascend_server.AscendServingEngine):
    """Use the production factory with a scalar session and no NPU context."""

    def inject(self, request):
        return self.session.inject(request)

    def decode(self, first_token_id, **kwargs):
        return self.session.decode(first_token_id, **kwargs)

    def reset(self):
        return self.session.reset()


class DecodeTimingChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, width=4, mode="accept", prefix=3, budget=9,
                rounds_per_command=1):
        backend, ranks, channel = make_backend(mode, prefix + budget + 8)
        clock = Clock()
        timed = TimedBackend(backend, clock)
        session = create_session(timed, width, (),
                                 rounds_per_command=rounds_per_command)
        engine = TimedEngine(session, timed, self.directory, clock,
                             metadata={"mtp_width": width})
        prompt = tuple((row * 7 + 3) % 251 for row in range(prefix))
        engine.inject(PrefixRequest("request", prefix, {"prompt": prompt}))
        first = ranks[0].engine.first_token_id
        return engine, first, ranks, channel

    def test_real_tp16_factory_serial_and_mtp_callbacks(self):
        for width in (1, 2, 4):
            for mode in ("accept", "reject", "mixed"):
                with self.subTest(width=width, mode=mode):
                    engine, first, ranks, channel = self.fixture(width, mode)
                    delivered = []
                    tokens = engine.decode(first, 9, on_token=delivered.append)
                    engine.reset()
                    report = load_timing(self.directory, "request")
                    self.assertEqual(tokens, delivered)
                    self.assertEqual(report["token_ids"], tokens)
                    self.assertEqual(report["returned_token_ids"], tokens)
                    self.assertEqual(report["producer"]["output_tokens"], 9)
                    self.assertTrue(report["reset_complete"])
                    self.assertTrue(all(rank.engine.position == 0
                                        for rank in ranks))
                    self.assertEqual(len(channel.commands),
                                     len(report["operations"]))
                    rounds = [item for item in report["operations"]
                              if item["operation"] in (
                                  "advance_mtp", "advance_mtp_n")]
                    if width > 1:
                        self.assertEqual(sum(item["accepted"]
                                             for item in rounds), 8)

    def test_production_factory_wires_the_same_timed_backend(self):
        for width in (1, 2, 4):
            for timing in (False, True):
                with self.subTest(width=width, timing=timing):
                    backend, ranks, _ = make_backend("mixed", 32)
                    rank_engine = ranks[0].engine
                    directory = self.directory if timing else None
                    with patch.object(ascend_server, "AscendServingEngine",
                                      CpuServingEngine):
                        engine = ascend_server.create_serving_engine(
                            backend, rank_engine, "cpu", mtp_width=width,
                            stop_ids=(), timing_dir=directory,
                            metadata={"mtp_width": width},
                        )
                    if timing:
                        self.assertIsInstance(engine, TimedEngine)
                        serving = engine.engine
                        self.assertIs(serving.session.backend,
                                      engine.timed_backend)
                        self.assertIs(engine.timed_backend.backend, backend)
                    else:
                        self.assertIsInstance(engine, CpuServingEngine)
                        serving = engine
                        self.assertIs(serving.session.backend, backend)
                    self.assertIs(serving.rank_engine, rank_engine)
                    engine.inject(PrefixRequest("factory", 3,
                                                {"prompt": (3, 10, 17)}))
                    delivered = []
                    tokens = engine.decode(rank_engine.first_token_id,
                                           max_tokens=9,
                                           on_token=delivered.append)
                    engine.reset()
                    self.assertEqual(tokens, delivered)
                    self.assertEqual(len(tokens), 9)
                    self.assertTrue(all(rank.engine.position == 0
                                        for rank in ranks))
                    if timing:
                        report = load_timing(self.directory, "factory")
                        self.assertEqual(report["token_ids"], tokens)
                        self.assertTrue(report["reset_complete"])
                        self.assertEqual(report["runtime"]["mtp_width"],
                                         width)

    def test_actual_8k_prefix_and_1024_delivery(self):
        engine, first, ranks, _ = self.fixture(prefix=8192, budget=1024)
        tokens = engine.decode(first, 1024, sampling={"ignore_eos": True})
        engine.reset()
        report = load_timing(self.directory, "request")
        received = bench.DecodeStream()
        received.feed(wire_message({"t": tokens})
                      + wire_message(terminal(1024)), 100)
        received.finish(first)
        metrics, rounds = bench.validate_producer(report, received, first, 4)
        self.assertEqual(metrics["output_tokens"], 1024)
        self.assertEqual(sum(item["accepted"] for item in rounds), 1023)
        self.assertTrue(all(rank.engine.position == 0 for rank in ranks))

    def test_single_round_commands_remain_accepted(self):
        engine, first, _, _ = self.fixture(prefix=8192, budget=1024)

        def single_round(count):
            self.assertEqual(count, 1)
            return [engine.timed_backend.advance_mtp()]

        with patch.object(engine.timed_backend, "advance_mtp_n", single_round):
            tokens = engine.decode(first, 1024, sampling={"ignore_eos": True})
        engine.reset()
        report = load_timing(self.directory, "request")
        _, commands = bench.validate_producer(
            report, SimpleNamespace(token_ids=tokens), first, 4,
        )
        self.assertTrue(all(item["operation"] == "advance_mtp"
                            for item in commands))
        summary = bench.round_summary(commands)
        self.assertEqual(summary["count"], summary["command_count"])
        self.assertEqual(summary["accepted"], 1023)

    def test_batched_commands_preserve_round_counts_and_command_timing(self):
        engine, first, _, _ = self.fixture(prefix=8192, budget=1024,
                                           rounds_per_command=3)
        tokens = engine.decode(first, 1024, sampling={"ignore_eos": True})
        engine.reset()
        report = load_timing(self.directory, "request")
        received = SimpleNamespace(token_ids=tokens)
        _, commands = bench.validate_producer(report, received, first, 4)
        summary = bench.round_summary(commands)
        self.assertEqual(summary["count"], report["engine_stats"]["round_calls"])
        self.assertEqual(summary["command_count"], len(commands))
        self.assertGreater(summary["count"], summary["command_count"])
        self.assertEqual(summary["accepted"], 1023)
        self.assertEqual(summary["mean_accepted"], 1023 / summary["count"])
        self.assertNotIn("p50_ms", summary)
        self.assertEqual(summary["p50_command_ms"], 1)
        for changes in ({"rounds": 0}, {"rounds": True}, {"rounds": 4},
                        {"accepted": 1}, {"accepted_tokens": [0]},
                        {"start_ns": 10**30}):
            damaged = deepcopy(report)
            command = next(item for item in damaged["operations"]
                           if item["operation"] == "advance_mtp_n")
            command.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                bench.validate_producer(damaged, received, first, 4)
        for changes in ({"width": 3}, {"width": 5}, {"accepted": 0},
                        {"output_tokens": 0},
                        {"main_start": 0}, {"main_retained": 0}):
            damaged = deepcopy(report)
            damaged["engine_stats"]["accepted_spans"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                bench.validate_producer(damaged, received, first, 4)
        for key in ("round_calls", "completed_rounds"):
            damaged = deepcopy(report)
            damaged["engine_stats"][key] += 1
            with self.subTest(field=key), self.assertRaises(ValueError):
                bench.validate_producer(damaged, received, first, 4)

    def test_first_mismatch_cannot_claim_a_complete_accepted_round(self):
        for width in (4, 32):
            with self.subTest(width=width):
                engine, first, _, _ = self.fixture(
                    width=width, prefix=8192, budget=1024,
                    rounds_per_command=3,
                )
                tokens = engine.decode(first, 1024,
                                       sampling={"ignore_eos": True})
                engine.reset()
                report = load_timing(self.directory, "request")
                received = bench.DecodeStream()
                received.feed(wire_message({"t": tokens})
                              + wire_message(terminal(1024)), 100)
                received.finish(first)
                bench.validate_producer(report, received, first, width)
                span = report["engine_stats"]["accepted_spans"][0]
                self.assertEqual(span["accepted"], width)
                span["candidates"][1] = (
                    span["predictions"][0] + 1
                ) % bench.VOCAB_SIZE
                with self.assertRaisesRegex(ValueError, "acceptance differs"):
                    bench.validate_producer(report, received, first, width)

    def test_linear_acceptance_retains_rejections_and_final_budget_clipping(self):
        cases = [(1, "accept"), *((width, mode)
                 for width in (2, 4, 32)
                 for mode in ("accept", "reject", "mixed"))]
        for width, mode in cases:
            with self.subTest(width=width, mode=mode):
                engine, first, _, _ = self.fixture(
                    width=width, mode=mode, prefix=8192, budget=1024,
                    rounds_per_command=3,
                )
                tokens = engine.decode(first, 1024,
                                       sampling={"ignore_eos": True})
                engine.reset()
                report = load_timing(self.directory, "request")
                metrics, commands = bench.validate_producer(
                    report, SimpleNamespace(token_ids=tokens), first, width,
                )
                self.assertEqual(metrics["output_tokens"], 1024)
                if width == 1:
                    self.assertEqual(commands, [])
                elif mode == "accept":
                    last = report["engine_stats"]["accepted_spans"][-1]
                    self.assertEqual(last["natural_accepted"], width)
                    self.assertEqual(last["accepted"], 1023 % width)

    def test_linear_round_rows_reject_unknown_tree_and_invalid_tokens(self):
        engine, first, _, _ = self.fixture(prefix=8192, budget=1024)
        tokens = engine.decode(first, 1024, sampling={"ignore_eos": True})
        engine.reset()
        report = load_timing(self.directory, "request")
        changes = (
            {"parents": [None, 0, 1, 2]},
            {"candidates": None},
            {"predictions": [1]},
            {"candidates": [first + 1, 1, 2, 3]},
            {"predictions": [True, 1, 2, 3]},
            {"predictions": [bench.VOCAB_SIZE, 1, 2, 3]},
        )
        for values in changes:
            with self.subTest(values=values):
                damaged = deepcopy(report)
                damaged["engine_stats"]["accepted_spans"][0].update(values)
                with self.assertRaisesRegex(ValueError, "linear candidate"):
                    bench.validate_producer(
                        damaged, SimpleNamespace(token_ids=tokens), first, 4,
                    )

    def test_eos_clipped_delivery_cannot_enter_the_fixed_length_gate(self):
        engine, first, _, _ = self.fixture(prefix=8192, budget=1024)
        baseline = engine.decode(first, 1024, sampling={"ignore_eos": True})
        engine, first, _, _ = self.fixture(prefix=8192, budget=1024)
        engine.engine.stop_ids = frozenset((baseline[2],))
        tokens = engine.decode(first, 1024)
        engine.reset()
        report = load_timing(self.directory, "request")
        self.assertEqual(report["engine_stats"]["finish_reason"], "stop")
        with self.assertRaisesRegex(ValueError, "incomplete delivery"):
            bench.validate_producer(
                report, SimpleNamespace(token_ids=tokens), first, 4,
            )

    def test_zero_one_and_first_eos_have_no_tpot(self):
        for count, stop in ((0, False), (1, False), (9, True)):
            engine, first, _, _ = self.fixture()
            if stop:
                engine.engine.stop_ids = frozenset((first,))
            tokens = engine.decode(first, count)
            report = load_timing(self.directory, "request")
            self.assertEqual(len(tokens), 0 if stop else count)
            self.assertIsNone(report["producer"]["tpot_ms"])
            self.assertNotIn("advance_mtp", [item["operation"]
                                             for item in report["operations"]])

    def test_callback_timestamp_is_after_callback_success(self):
        engine, first, _, _ = self.fixture()
        before_return = []

        def callback(token):
            before_return.append(engine.clock())

        engine.decode(first, 3, on_token=callback)
        report = load_timing(self.directory, "request")
        self.assertTrue(all(after > before for before, after in
                            zip(before_return, report["callback_ns"])))

    def test_cancelled_burst_counts_successful_delivery(self):
        for limit in (0, 1, 2):
            engine, first, _, _ = self.fixture()
            cancellation, delivered = Event(), []
            if limit == 0:
                cancellation.set()

            def callback(token):
                delivered.append(token)
                if len(delivered) == limit:
                    cancellation.set()

            engine.decode(first, 9, on_token=callback,
                          cancel_event=cancellation)
            report = load_timing(self.directory, "request")
            self.assertEqual(report["producer"]["output_tokens"], limit)
            self.assertEqual(report["engine_stats"]["finish_reason"],
                             "cancelled")
            if limit == 2:
                self.assertGreater(report["engine_stats"]["accepted"], 1)

    def test_callback_exception_does_not_count_failed_token(self):
        engine, first, _, _ = self.fixture()
        successful = []

        def callback(token):
            if len(successful) == 2:
                raise ValueError("sink failed")
            successful.append(token)

        with self.assertRaisesRegex(ValueError, "sink failed"):
            engine.decode(first, 9, on_token=callback)
        report = load_timing(self.directory, "request")
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["token_ids"], successful)
        self.assertEqual(report["producer"]["output_tokens"], 2)

    def test_command_error_is_saved_and_preserved(self):
        engine, first, ranks, channel = self.fixture()
        with patch.object(ranks[0].engine, "advance_mtp",
                          side_effect=RuntimeError("rank command failed")):
            with self.assertRaisesRegex(RuntimeError, "rank command failed"):
                engine.decode(first, 9)
        report = load_timing(self.directory, "request")
        self.assertIn("rank command failed", report["error"])
        self.assertEqual(report["producer"]["output_tokens"], 1)
        before = len(channel.commands)
        with self.assertRaisesRegex(RuntimeError, "restart all sixteen"):
            engine.reset()
        self.assertEqual(len(channel.commands), before)

    def test_safe_request_path_and_atomic_json(self):
        path = timing_path(self.directory, "../../outside")
        self.assertEqual(path.parent, self.directory)
        engine, first, _, _ = self.fixture()
        engine.decode(first, 1)
        self.assertEqual(list(self.directory.iterdir()),
                         [timing_path(self.directory, "request")])

    def test_fragmented_and_coalesced_lines_keep_received_times(self):
        received = bench.DecodeStream()
        first = wire_message({"t": [1]})
        received.feed(first[:4], 10)
        received.feed(first[4:] + wire_message({"t": [2, 3]})
                      + wire_message(terminal()), 20)
        result = received.finish(1, output_tokens=3)
        self.assertEqual(received.token_ids, [1, 2, 3])
        self.assertEqual(received.timestamps_ns, [20, 20, 20])
        self.assertEqual(result["zero_intervals"], 2)
        sizes = [len(item["token_ids"]) for item in received.messages]
        self.assertEqual(sizes, [1, 2])

    def test_missing_done_truncated_line_and_bad_terminal_rejected(self):
        cases = (
            b"", b'{"done":true', wire_message(terminal(4)),
            wire_message({**terminal(), "seq_len": 8191}),
            wire_message({**terminal(), "finish_reason": "stop"}),
        )
        for tail in cases:
            received = bench.DecodeStream()
            received.feed(wire_message({"t": [1, 2, 3]}) + tail, 10)
            with self.assertRaises(ValueError):
                received.finish(1, output_tokens=3)

    def test_invalid_tokens_errors_and_post_done_data_rejected(self):
        for value in ({"t": [True]}, {"t": []}, {"error": "failed"},
                      {"t": [1], "done": True}):
            with self.assertRaises(ValueError):
                bench.DecodeStream().feed(wire_message(value), 10)
        received = bench.DecodeStream()
        received.feed(wire_message(terminal()), 10)
        with self.assertRaises(ValueError):
            received.feed(wire_message({"t": [2]}), 20)

    def test_http_reader_uses_read1_and_counts_tokens_not_messages(self):
        chunks = iter((wire_message({"t": [7]}),
                       wire_message({"t": [8] * 1023})
                       + wire_message(terminal(1024)), b""))
        response = SimpleNamespace(read1=lambda _: next(chunks))
        connection = SimpleNamespace(close=lambda: None)
        args = SimpleNamespace(decode_url="http://unused", timeout=1)
        with patch.object(bench, "post", return_value=(connection, response)):
            received = bench.decode(args, "request", 7)
        self.assertEqual(len(received.token_ids), 1024)
        self.assertEqual(len(received.messages), 2)

    def test_buffered_client_cannot_hide_slow_producer(self):
        tokens = list(range(1024))
        requests = [benchmark_request(tokens, number, 3000000, 0)
                    for number in range(32)]
        precision = {"prompt_sha256s": sorted(set(
            request["prompt_sha256"] for request in requests
        ))}
        summary = benchmark_summary(requests, precision, self.directory,
                                    policy=self.diversity_policy())
        self.assertEqual(summary["client"]["request_p95_ms"], 0)
        self.assertFalse(bench.passes_target(
            summary, requests, self.directory,
        ))

    def diversity_policy(self, tokens=2, bigrams=2):
        path = self.directory / "policy-source.json"
        bench.atomic_json(path, {"min_distinct_tokens": tokens,
                                 "min_distinct_bigrams": bigrams})
        return bench.load_diversity_policy(path)

    def frozen_diversity_policy(self, tokens=2, bigrams=2):
        policy = self.diversity_policy(tokens, bigrams)
        (self.directory / "diversity-policy.json").unlink(missing_ok=True)
        return bench.freeze_diversity_policy(policy, self.directory)

    def diversity_case(self, tokens=None):
        if tokens is None:
            tokens = list(range(1024))
        requests = [benchmark_request(tokens, number) for number in range(32)]
        precision = {"prompt_sha256s": sorted({
            request["prompt_sha256"] for request in requests
        })}
        summary = benchmark_summary(requests, precision, self.directory,
                                    policy=self.diversity_policy())
        return summary, requests

    def test_explicit_diversity_policy_has_no_default_minima(self):
        self.assertIsNone(bench.load_diversity_policy(None))
        policy = self.diversity_policy(1024, 1023)
        self.assertEqual(policy["sha256"], hashlib.sha256(
            Path(policy["file"]).read_bytes()).hexdigest())
        for value in (None, {}, [], {"min_distinct_tokens": 2}):
            path = self.directory / "invalid-policy.json"
            bench.atomic_json(path, value)
            with self.assertRaises(ValueError):
                bench.load_diversity_policy(path)
        for field in ("min_distinct_tokens", "min_distinct_bigrams"):
            for value in (True, 1, 0, -1, 2.0, "2"):
                values = {"min_distinct_tokens": 2, "min_distinct_bigrams": 2}
                values[field] = value
                bench.atomic_json(path, values)
                with self.assertRaises(ValueError):
                    bench.load_diversity_policy(path)
        summary, requests = self.diversity_case()
        source = Path(summary["diversity_policy"]["source"]["file"])
        raw = b'{ "min_distinct_tokens": 2, "min_distinct_bigrams": 2 }\n'
        source.write_bytes(raw)
        frozen = self.directory / "diversity-policy.json"
        frozen.unlink()
        summary["diversity_policy"] = bench.freeze_diversity_policy(
            bench.load_diversity_policy(source), self.directory,
        )
        self.assertEqual(frozen.read_bytes(), raw)
        source.unlink()
        self.assertTrue(bench.passes_target(
            summary, requests, self.directory,
        ))

    def test_fast_complete_outputs_need_the_explicit_policy(self):
        summary, requests = self.diversity_case()
        summary["diversity_policy"] = None
        summary["diversity_passed"] = True
        self.assertFalse(bench.passes_target(
            summary, requests, self.directory,
        ))

    def test_one_ms_constant_output_cannot_pass_the_target(self):
        summary, requests = self.diversity_case([0] * 1024)
        for source in ("producer", "client"):
            self.assertEqual(summary[source]["request_p95_ms"], 1.0)
        self.assertEqual(len(summary["measured_prompt_counts"]), 8)
        self.assertFalse(bench.passes_target(
            summary, requests, self.directory,
        ))

    def test_whole_output_period_rejects_even_truncated_repetitions(self):
        for period in (257, 512, 513):
            tokens = [row % period for row in range(1024)]
            summary, requests = self.diversity_case(tokens)
            summary["diversity_policy"] = self.frozen_diversity_policy(128, 128)
            self.assertEqual(
                requests[0]["diversity"]["minimum_period"], period,
            )
            self.assertEqual(
                bench.passes_target(summary, requests, self.directory),
                period > 512,
            )

    def test_diversity_minima_are_applied_to_actual_tokens_and_bigrams(self):
        for tokens, bigrams, passed in ((1024, 1023, True),
                                       (1025, 1023, False),
                                       (1024, 1024, False)):
            summary, requests = self.diversity_case()
            summary["diversity_policy"] = self.frozen_diversity_policy(
                tokens, bigrams,
            )
            self.assertEqual(
                bench.passes_target(summary, requests, self.directory), passed,
            )

    def test_missing_or_forged_diversity_statistics_are_recomputed(self):
        mutations = (None, "output_sha256", "distinct_output_tokens",
                     "distinct_output_bigrams", "minimum_period",
                     "output_tokens")
        for field in mutations:
            summary, requests = self.diversity_case()
            if field is None:
                del requests[0]["diversity"]
            elif field == "output_sha256":
                requests[0]["diversity"][field] = "0" * 64
            else:
                requests[0]["diversity"][field] += 1
            name = summary["requests"][0]
            summary["request_sha256"][name] = save_request(
                self.directory, name, requests[0],
            )
            self.assertFalse(bench.passes_target(
                summary, requests, self.directory,
            ), msg=str(field))

    def test_request_bytes_counts_and_complete_token_lists_are_required(self):
        for field in ("sha", "raw", "tokens", "returned", "count", "coverage",
                      "measured", "policy"):
            summary, requests = self.diversity_case()
            name = summary["requests"][0]
            if field == "sha":
                summary["request_sha256"][name] = "0" * 64
            elif field == "raw":
                with (self.directory / name).open("a") as stream:
                    stream.write(" ")
            elif field == "tokens":
                requests[0]["client_token_ids"].pop()
            elif field == "returned":
                requests[0]["producer_report"]["returned_token_ids"][0] = 7
            elif field == "count":
                requests[0]["client"]["output_tokens"] -= 1
                summary["client"] = bench.aggregate(requests, "client")
            elif field == "coverage":
                prompt = requests[0]["prompt_sha256"]
                summary["measured_prompt_counts"][prompt] += 1
            elif field == "measured":
                requests.pop()
            else:
                with (self.directory / "diversity-policy.json").open("ab") as stream:
                    stream.write(b" ")
            if field in ("tokens", "returned", "count"):
                summary["request_sha256"][name] = save_request(
                    self.directory, name, requests[0],
                )
            self.assertFalse(bench.passes_target(
                summary, requests, self.directory,
            ), msg=field)

    def test_prompt_identity_matches_the_prefix_producer(self):
        prompts = [[token] * 8192 for token in range(8)]
        digests = [prompt_identity(prompt)["prompt_sha256"]
                   for prompt in prompts]
        precision = {"prompt_sha256s": digests}
        self.assertEqual(bench.validate_prompts(prompts, precision), digests)
        self.assertEqual(bench.validate_prompts(prompts[:1], precision),
                         digests[:1])
        with self.assertRaisesRegex(ValueError, "no matching precision"):
            bench.validate_prompts([[8] * 8192], precision)
        for invalid in (None, digests[:7], digests[:7] + digests[:1],
                        digests[:7] + ["z" * 64]):
            with self.assertRaisesRegex(ValueError, "eight distinct"):
                bench.validate_prompts(prompts,
                                       {"prompt_sha256s": invalid})

    def benchmark_args(self, warmup=3, requests=32):
        return SimpleNamespace(
            prompt_ids=self.directory / "input.json",
            precision_dir=self.directory / "precision",
            checkpoint=self.directory / "checkpoint", mtp_width=4,
            expert_bank=False, output=self.directory / "benchmark",
            diversity_policy=self.diversity_policy()["file"],
            warmup=warmup, requests=requests,
        )

    def test_unqualified_prompt_rejected_before_any_request(self):
        args = self.benchmark_args()
        bench.atomic_json(args.prompt_ids, [[9] * 8192])
        precision = {"prompt_sha256s": [bench.prompt_digest([i] * 8192)
                                       for i in range(8)]}
        with patch.object(bench, "parse_arguments", return_value=args), \
                patch("precision_gate.validate_workload_precision",
                      return_value=precision), \
                patch.object(bench, "run_request") as request:
            with self.assertRaisesRegex(ValueError, "no matching precision"):
                bench.main()
        request.assert_not_called()

    def test_target_requires_actual_measured_prompt_coverage(self):
        prompts = [[token] * 8192 for token in range(8)]
        precision = {"prompt_sha256s": [bench.prompt_digest(prompt)
                                       for prompt in prompts]}
        cases = (
            (prompts, 3, 32, True, 8, True),
            (prompts, 3, 32, False, 8, False),
            (prompts, 2, 32, False, 8, True),
            (prompts, 3, 31, False, 8, True),
            (prompts[:1], 3, 32, False, 1, True),
            ([prompts[7]] + prompts[:7] * 5, 3, 32, False, 7, True),
            (prompts[:1], 0, 1, False, 1, True),
        )
        for number, case in enumerate(cases):
            submitted, warmup, count, passes, coverage, policy = case
            args = self.benchmark_args(warmup, count)
            args.output = self.directory / f"benchmark-{number}"
            if not policy:
                args.diversity_policy = None
            bench.atomic_json(args.prompt_ids, submitted)
            calls = []

            def measured(_args, prompt, _precision, calls=calls):
                calls.append(prompt)
                return {**benchmark_request(list(range(1024)), len(calls),
                                            2000000, 2000000), "rounds": [],
                        "client_ttft_ms": 1,
                        "prompt_sha256": bench.prompt_digest(prompt)}

            with patch.object(bench, "parse_arguments", return_value=args), \
                    patch("precision_gate.validate_workload_precision",
                          return_value=precision), \
                    patch.object(bench, "run_request", side_effect=measured), \
                    redirect_stdout(io.StringIO()):
                bench.main()
            result = json.loads((args.output / "summary.json").read_text())
            self.assertEqual(result["passes_2ms"], passes)
            self.assertEqual(len(result["requests"]), count)
            self.assertEqual(len(result["warmup_requests"]), warmup)
            self.assertEqual(len(result["measured_prompt_counts"]), coverage)
            self.assertEqual(result["status"], "measured")
            self.assertEqual(result["diversity_policy"] is not None, policy)

    def test_diversity_does_not_relax_either_clock_or_trust_aggregate_timing(self):
        for source in ("producer", "client"):
            summary, requests = self.diversity_case()
            for request in requests:
                request[source] = token_metrics(
                    request["client_token_ids"],
                    [i * 3000000 for i in range(1024)],
                )
            summary[source] = bench.aggregate(requests, source)
            self.assertFalse(bench.passes_target(
                summary, requests, self.directory,
            ))
            summary[source]["request_p50_ms"] = 1.0
            summary[source]["request_p95_ms"] = 1.0
            self.assertFalse(bench.passes_target(
                summary, requests, self.directory,
            ))

    def test_request_mean_and_token_weighted_statistics_differ(self):
        requests = [
            {"producer": token_metrics([1, 2], [0, 1000000])},
            {"producer": token_metrics(list(range(10)),
                                       [i * 3000000 for i in range(10)])},
        ]
        stats = bench.aggregate(requests, "producer")
        self.assertEqual(stats["request_mean_ms"], 2)
        self.assertEqual(stats["token_weighted_ms"], 2.8)
        self.assertEqual(stats["request_p95_ms"], 3)

    def test_runtime_identity_must_match_precision(self):
        name = "tilert/pd_vllm/decode_timing.py"
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        precision = {"mtp_width": 4, "expert_backend": "bank",
                     "checkpoint_sha256": {"config.json": "test"},
                     "packages": {"torch": "cpu-fixture"},
                     "source_sha256": {name: digest}}
        producer = {"runtime": {**precision, "rank": 0, "world_size": 16,
                                "torch": "cpu-fixture"},
                    "timing_source_sha256": digest}
        bench.validate_runtime(producer, precision)
        for field, value in (("mtp_width", 2), ("world_size", 8),
                             ("expert_backend", "per_expert"),
                             ("torch", "wrong-version")):
            wrong = deepcopy(producer)
            wrong["runtime"][field] = value
            with self.assertRaises(ValueError):
                bench.validate_runtime(wrong, precision)

    def test_failed_prefill_validation_cancels_owned_request(self):
        args = SimpleNamespace(output=self.directory,
                               decode_url="http://unused", timeout=1)
        response = {"id": "cmpl-abcdefgh", "usage": {"prompt_tokens": 3}}
        connection = SimpleNamespace(close=lambda: None)
        reply = SimpleNamespace(read=lambda: b"{}")
        with patch.object(bench, "prefill", return_value=response):
            with patch.object(bench, "post",
                              return_value=(connection, reply)) as send:
                with self.assertRaisesRegex(ValueError, "8192 input"):
                    bench.run_request(args, [1, 2, 3], {})
        self.assertEqual(send.call_args.args[1], "/pd/cancel")
        report = json.loads(timing_path(self.directory / "failures",
                                       "abcdefgh").read_text())
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["client_token_ids"], [])

    def test_short_prefill_usage_and_default_8k_contract(self):
        response = {
            "usage": {"prompt_tokens": 32, "completion_tokens": 1},
            "choices": [{"logprobs": {"tokens": ["token_id:17"]}}],
        }
        self.assertEqual(bench.prefill_token(response, prompt_tokens=32), 17)
        with self.assertRaisesRegex(ValueError, "8192 input"):
            bench.prefill_token(response)
        for inputs, outputs in ((31, 1), (33, 1), (32, 0), (32, 2)):
            response["usage"] = {"prompt_tokens": inputs,
                                 "completion_tokens": outputs}
            with self.assertRaisesRegex(ValueError, "32 input"):
                bench.prefill_token(response, prompt_tokens=32)
        response["usage"] = {"prompt_tokens": 8192, "completion_tokens": 1}
        self.assertEqual(bench.prefill_token(response), 17)

    def test_no_device_package_imported(self):
        self.assertNotIn("torch", sys.modules)


def source_hashes():
    paths = [Path(__file__), Path(__file__).with_name("bench_pd_decode.py"),
             Path(__file__).with_name("check_mtp_tp.py"),
             Path(__file__).with_name("prefix_io.py"),
             Path(__file__).with_name("precision_gate.py")]
    paths.extend((ROOT / "tilert/models/glm_5_2_ascend").glob("*.py"))
    paths.extend(ROOT / "tilert/pd_vllm" / name for name in (
        "decode_timing.py", "ascend_server.py", "ascend_cache.py", "wire.py",
    ))
    return {str(path.relative_to(ROOT)): hashlib.sha256(
        path.read_bytes()
    ).hexdigest() for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = source_hashes()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        DecodeTimingChecks,
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    after = source_hashes()
    passed = result.wasSuccessful() and before == after
    passed = passed and "torch" not in sys.modules
    report = {
        "status": "pass" if passed else "fail", "tests": result.testsRun,
        "scope": "stdlib TP16 scalar commands and HTTP framing; no NPU timing",
        "python": platform.python_version(), "failures": len(result.failures),
        "errors": len(result.errors), "torch_imported": "torch" in sys.modules,
        "source_stable": before == after, "source_sha256": after,
        "test_output": stream.getvalue(),
    }
    bench.atomic_json(args.output, report)
    print(stream.getvalue())
    print(json.dumps({key: report[key] for key in (
        "status", "tests", "failures", "errors", "torch_imported",
    )}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
