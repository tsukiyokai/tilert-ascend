"""Check profiling lifetime and lookup accounting without NPU devices."""

from contextlib import redirect_stderr
from copy import deepcopy
import io
from itertools import product
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_cann_mtp as bench  # noqa: E402
import read_levers  # noqa: E402
from check_mtp_tp import make_backend, serial_oracle  # noqa: E402
from tilert.models.glm_5_2_ascend.session import PrefixRequest  # noqa: E402


class BenchmarkChecks(unittest.TestCase):
    tree_coverage = {"exhaustive_cases": 0, "sessions": 0, "spans": 0}

    def drive(self, output, fail_during_capture=False):
        backend, ranks, channel = make_backend("mixed", capacity=32)
        prompt = (3, 10, 17)
        first = serial_oracle(prompt, 1, 32)[0][0]
        args = SimpleNamespace(width=4, warmup=1, requests=2,
                               max_tokens=9, rounds_per_command=1,
                               profile_last=True, output=output)
        events = []
        active = [False]

        class Capture:
            def __enter__(capture):
                events.append(("enter", channel.commands[-1]["operation"],
                               backend.position))
                active[0] = True
                return capture

            def __exit__(capture, error_type, error, traceback):
                events.append(("exit", error_type))
                active[0] = False

        profiler = SimpleNamespace(
            profile=lambda **kwargs: Capture(),
            ProfilerActivity=SimpleNamespace(NPU="npu"),
            ProfilerLevel=SimpleNamespace(Level1=1),
            _ExperimentalConfig=lambda **kwargs: kwargs,
            tensorboard_trace_handler=lambda path: path,
            profiler=SimpleNamespace(analyse=lambda **kwargs:
                                     events.append(("analyse", kwargs))),
        )
        advance = backend.advance_mtp_n

        def advance_or_fail(count):
            if fail_during_capture and active[0]:
                raise RuntimeError("decode failed during capture")
            return advance(count)

        def request(rid, length):
            return PrefixRequest(rid, length, {"prompt": prompt})

        with patch.dict(sys.modules, {"torch_npu": SimpleNamespace(profiler=profiler)}), \
                patch.object(bench, "PREFIX_LENGTH", len(prompt)), \
                patch.object(bench, "PrefixRequest", side_effect=request), \
                patch.object(backend, "advance_mtp_n", side_effect=advance_or_fail):
            if fail_during_capture:
                with self.assertRaisesRegex(RuntimeError, "decode failed during capture"):
                    bench.drive(backend, first, args)
                records = None
            else:
                records = bench.drive(backend, first, args)
        self.assertTrue(all(rank.closed for rank in ranks))
        self.assertTrue(all(rank.engine.position == 0 for rank in ranks))
        self.assertFalse(active[0])
        return records, events, args

    def test_real_session_profiles_only_last_decode_and_excludes_its_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            records, events, args = self.drive(Path(directory))
        self.assertEqual([row["warmup"] for row in records], [True, False, False])
        self.assertEqual([row["profiled"] for row in records], [False, False, True])
        self.assertEqual(events, [
            ("enter", "inject", 3), ("exit", None),
            ("analyse", {"profiler_path": str(args.output / "capture")}),
        ])
        self.assertEqual(bench.timing_requests(records), [records[1]])
        self.assertTrue(all(row["tokens"] == row["returned_tokens"] for row in records))
        self.assertTrue(all(row["operations"][-1]["operation"] == "reset"
                            for row in records))

    def test_decode_failure_closes_profiler_and_resets_the_session(self):
        with tempfile.TemporaryDirectory() as directory:
            _, events, _ = self.drive(Path(directory), fail_during_capture=True)
        self.assertEqual(events, [("enter", "inject", 3), ("exit", RuntimeError)])

    def state(self, lookup):
        draft = None if lookup else 8193
        reset = [0, None if lookup else 0]
        request = {
            "first_token": 7, "injected": [8192, None if lookup else 8191],
            "finished": [8193, draft], "reset": reset,
            "rounds": [{
                "width": 2, "candidates": [7, 99], "predictions": [8, 100],
                "accepted_tokens": [8], "matched_drafts": 0,
                "natural_accepted": 1, "before": 8192, "written": 8194,
                "retained": 8193, "draft_retained": draft,
                "start_ns": 1, "end_ns": 2, "elapsed_ns": 1,
            }],
        }
        args = SimpleNamespace(lookup=lookup, warmup=0, requests=1,
                               max_tokens=2, width=2, target_graph=False,
                               draft_graph=False, conditioned_graph=False,
                               automaton=False, tree=False)
        report = {"rank": 1, "closed": True, "final_positions": reset,
                  "rank_requests": [request], "command_serial": 6}
        return args, report

    def test_lookup_accepts_absent_draft_and_rejects_fabricated_draft_state(self):
        args, report = self.state(lookup=True)
        self.assertIsNone(bench.MeasuredRank.draft_position.fget(
            SimpleNamespace(draft=None)))
        self.assertEqual(bench.validate_local(report, args)[0]["tokens"], [7, 8])
        for key, value in (("injected", [8192, 8191]), ("reset", [0, 0]),
                           ("finished", [8193, 8193])):
            corrupted = deepcopy(report)
            corrupted["rank_requests"][0][key] = value
            with self.assertRaises(ValueError):
                bench.validate_local(corrupted, args)
        corrupted = deepcopy(report)
        corrupted["final_positions"] = [0, 0]
        with self.assertRaises(ValueError):
            bench.validate_local(corrupted, args)

    def test_mtp_still_requires_the_draft_positions(self):
        args, report = self.state(lookup=False)
        self.assertEqual(bench.validate_local(report, args)[0]["tokens"], [7, 8])
        report["rank_requests"][0]["injected"][1] = None
        with self.assertRaises(ValueError):
            bench.validate_local(report, args)

    def tree_state(self):
        args, report = self.state(lookup=True)
        args.tree, args.width, args.max_tokens = True, 4, 4
        request = report["rank_requests"][0]
        request["finished"] = [8195, None]
        request["rounds"][0].update(
            width=4, candidates=[7, 99, 8, 9], parents=[-1, 0, 0, 2],
            predictions=[8, 100, 9, 10], accepted_tokens=[8, 9, 10],
            matched_drafts=2, natural_accepted=3, written=8196, retained=8195,
        )
        return args, report

    def test_tree_report_follows_the_selected_branch(self):
        args, report = self.tree_state()
        self.assertEqual(bench.validate_local(report, args)[0]["tokens"],
                         [7, 8, 9, 10])
        args.max_tokens = 3
        request = report["rank_requests"][0]
        request["finished"] = [8194, None]
        request["rounds"][0].update(accepted_tokens=[8, 9], retained=8194)
        self.assertEqual(bench.validate_local(report, args)[0]["tokens"],
                         [7, 8, 9])

    def test_tree_report_rejects_missing_malformed_or_mislabelled_lineage(self):
        args, report = self.tree_state()
        for parents in (None, [], [-1, 0], [0, 0, 0, 2], [-1, -1, 0, 2],
                        [-1, 1, 0, 2], [-1, 3, 0, 2], [-1, True, 0, 2]):
            damaged = deepcopy(report)
            damaged["rank_requests"][0]["rounds"][0]["parents"] = parents
            with self.subTest(parents=parents), self.assertRaises(ValueError):
                bench.validate_local(damaged, args)
        args.tree = False
        with self.assertRaises(ValueError):
            bench.validate_local(report, args)

    def test_tree_report_rejects_a_fabricated_accepted_path(self):
        args, report = self.tree_state()
        for changes in ({"accepted_tokens": [8, 100, 9]},
                        {"matched_drafts": 1}, {"natural_accepted": 2},
                        {"parents": [-1, 0, 1, 2]}):
            damaged = deepcopy(report)
            damaged["rank_requests"][0]["rounds"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                bench.validate_local(damaged, args)

    def test_tree_report_ties_use_the_first_deepest_leaf(self):
        self.assertEqual(bench.tree_predictions(
            [1, 2, 2, 3, 3], [-1, 0, 0, 1, 2], [2, 3, 3, 9, 10],
        ), [2, 3, 9])
        self.assertEqual(bench.tree_predictions([7], [-1], [8]), [8])
        self.assertEqual(bench.tree_predictions(
            [7, 8, 99], [-1, 0, 1], [8, 9, 10],
        ), [8, 9])
        with self.assertRaises(ValueError):
            bench.tree_predictions([], [], [])

    def test_tree_report_matches_exhaustive_small_tree_resolution(self):
        from tilert.models.glm_5_2_ascend.tree_resolve import accepted_path

        for width in range(1, 6):
            for tail in product(*(range(node) for node in range(1, width))):
                parents = [-1, *tail]
                for labels in product((0, 1), repeat=2 * width):
                    candidates = list(labels[:width])
                    predictions = list(labels[width:])
                    path = accepted_path(candidates, parents, predictions)
                    self.assertEqual(bench.tree_predictions(
                        candidates, parents, predictions,
                    ), [predictions[node] for node in path])
                    self.tree_coverage["exhaustive_cases"] += 1

    def test_measured_tree_round_preserves_lineage_and_rebuilt_tokens(self):
        from check_round_invariance import (
            DRAFTERS, PROMPT, RULES, WIDTHS, ScalarTarget,
        )
        from tilert.models.glm_5_2_ascend.mtp_round import MtpRoundExecutor

        for rule in RULES:
            for width in WIDTHS:
                with self.subTest(rule=rule.__name__, width=width):
                    target = ScalarTarget(rule, PROMPT, len(PROMPT) + 128)
                    first = rule.predict(target.rows[-1][1], len(PROMPT))
                    loop = MtpRoundExecutor(
                        target, None, width, 32,
                        drafter=DRAFTERS["tree"](list(PROMPT)),
                    )
                    loop.tree_mask = target.install_mask
                    loop.start(first, None)
                    measured = bench.MeasuredRank.__new__(bench.MeasuredRank)
                    measured.target, measured.draft = target, None
                    measured._mtp = loop
                    measured.current = {"rounds": []}
                    while loop.state.finish_reason is None:
                        result = measured.advance_mtp()
                        row = measured.current["rounds"][-1]
                        self.assertEqual(row["parents"], list(result.parents))
                        rebuilt = bench.tree_predictions(
                            row["candidates"], row["parents"], row["predictions"],
                        )
                        self.assertEqual(row["accepted_tokens"],
                                         rebuilt[:result.commit.committed])
                        self.assertEqual(row["natural_accepted"], len(rebuilt))

    def test_tree_session_preserves_measured_parent_receipts(self):
        from check_round_invariance import (
            PROMPT, RULES, WIDTHS, ScalarTarget, serial_decode,
        )
        from tilert.models.glm_5_2_ascend.mtp_round import MtpRoundExecutor
        from tilert.models.glm_5_2_ascend.mtp_session import AscendMtpPDEngine
        from tilert.models.glm_5_2_ascend.tree_draft import TreeDrafter

        for rule in RULES:
            for width in WIDTHS:
                with self.subTest(rule=rule.__name__, width=width):
                    target = ScalarTarget(rule, PROMPT, len(PROMPT) + 160)
                    first = rule.predict(target.rows[-1][1], len(PROMPT))
                    loop = MtpRoundExecutor(
                        target, None, width, 96,
                        drafter=TreeDrafter(list(PROMPT)),
                    )
                    loop.tree_mask = target.install_mask
                    measured = bench.MeasuredRank(
                        target, None, None, PROMPT, drafter="tree",
                    )
                    measured._mtp = loop
                    measured.current = {"rounds": []}

                    class Backend:
                        """Adapt the scalar target's existing prefix."""

                        @property
                        def position(self):
                            return target.position

                        @property
                        def capacity(self):
                            return target.capacity

                        def inject(self, request):
                            assert request.seq_len == self.position

                        def start_mtp(self, opening, *unused):
                            loop.start(opening, None)

                        def advance_mtp_n(self, count):
                            assert count == 1
                            return [measured.advance_mtp()]

                        def finish_mtp(self, retained):
                            target.truncate(retained)

                    session = AscendMtpPDEngine(Backend(), width)
                    session.inject(PrefixRequest("tree", len(PROMPT)))
                    output = session.decode(first, 96)
                    self.assertEqual(output, serial_decode(rule, PROMPT, 96))
                    spans = session.last_stats["accepted_spans"]
                    rows = measured.current["rounds"]
                    self.tree_coverage["sessions"] += 1
                    self.tree_coverage["spans"] += len(spans)
                    self.assertEqual(len(spans), len(rows))
                    for span, row in zip(spans, rows):
                        self.assertEqual(span["parents"], row["parents"])
                        self.assertEqual(len(span["parents"]), width)
                        self.assertEqual(span["parents"][0], -1)
                        self.assertEqual(span["candidates"], row["candidates"])
                        self.assertEqual(span["predictions"], row["predictions"])
                        rebuilt = bench.tree_predictions(
                            row["candidates"], row["parents"], row["predictions"],
                        )
                        self.assertEqual(rebuilt[:span["accepted"]],
                                         row["accepted_tokens"])
                        self.assertEqual(span["natural_accepted"],
                                         row["natural_accepted"])

    def test_only_device_drafting_loads_the_mtp_model(self):
        target = object()
        modes = ((True, False, False), (False, True, False),
                 (False, False, True), (False, False, False))
        for lookup, automaton, tree in modes:
            args = SimpleNamespace(lookup=lookup, automaton=automaton,
                                   tree=tree, draft_projection_tp=True)
            candidate = SimpleNamespace(
                model=SimpleNamespace(do_not_compile=True),
            )
            with self.subTest(mode=(lookup, automaton, tree)), patch(
                "tilert.models.glm_5_2_ascend.cann_mtp.CannMtp",
                return_value=candidate,
            ) as load:
                draft = bench.create_draft(target, args)
                if lookup or automaton or tree:
                    self.assertIsNone(draft)
                    load.assert_not_called()
                else:
                    self.assertIs(draft, candidate)
                    load.assert_called_once_with(target,
                                                 shard_projections=True)

    def test_host_modes_reject_unused_mtp_flags_before_loading(self):
        required = ["bench", "--checkpoint", "checkpoint", "--capture-case",
                    "prefix", "--output", "out", "--source-commit", "test"]
        for mode in ("--lookup", "--automaton", "--tree"):
            for options in (("--draft-graph",), ("--draft-projection-tp",),
                            ("--mtp-tokens", "1")):
                with (
                    self.subTest(mode=mode, options=options),
                    patch.object(sys, "argv", [*required, mode, *options]),
                    redirect_stderr(io.StringIO()),
                ):
                    with self.assertRaises(SystemExit):
                        bench.arguments()

    def test_cli_requires_a_sample_without_profiler(self):
        command = ["bench", "--checkpoint", "checkpoint", "--capture-case", "prefix",
                   "--output", "out", "--source-commit", "test", "--requests", "1"]
        with patch.object(sys, "argv", command):
            self.assertEqual(bench.arguments().requests, 1)
        with patch.object(sys, "argv", command + ["--profile-last"]), \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            bench.arguments()
        self.assertEqual(error.exception.code, 2)

    def test_saved_result_readers_exclude_profiled_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number in range(1, 4):
                for name, elapsed in (("control", 10), ("candidate", 12)):
                    output = root / f"{name}-{number}"
                    output.mkdir()
                    records = [{
                        "warmup": False, "tpot_ms": 2, "output_tokens": 9,
                        "operations": [{"operation": "advance_mtp_n",
                                        "elapsed_ns": elapsed * 1000000}],
                    }]
                    profiled = deepcopy(records[0])
                    profiled.update(profiled=True, tpot_ms=1000)
                    profiled["operations"][0]["elapsed_ns"] = 1
                    records.append(profiled)
                    (output / "rank-0.json").write_text(json.dumps({
                        "requests": records}))
                    result = read_levers.read(output)
                    self.assertEqual(result["tpot"], 2)
                    self.assertEqual(result["round"], elapsed)
                    self.assertEqual(result["rounds"], 1)
            result = subprocess.check_output([
                sys.executable, str(Path(__file__).with_name("paired.py")),
                str(root), "control", "candidate"], text=True)
            self.assertIn("paired mean delta +2.000 ms a round", result)


if __name__ == "__main__":
    unittest.main()
