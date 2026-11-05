"""CPU checks of raw-clock accounting, falsification rules and provenance."""

import json
from pathlib import Path
import struct
import tempfile
import unittest

import device_bench as bench
import device_check as check


IDLE = """| 0 Ascend910 | OK |
| NPU Chip | Process id | Process name | Process memory(MB) |
+----------------------+
| No running processes found in NPU 0 |
+----------------------+
"""


def put(path, value):
    path.write_text(json.dumps(value, separators=(",", ":")))


def make_evidence(root, cycles=1, ranks=2, tail=False, skew_ns=100):
    config = {"ranks": ranks, "rows": [1], "blocks": 4, "warmup": 10,
              "samples": 30, "pressure_mib": 256, "replays_per_sample": 32,
              "start_lead_ms": 20}
    plan = bench.registration(config, cycles)
    plan["libraries"] = {arm: {"library_sha256": arm} for arm in bench.ARMS}
    plan["launch"] = {"hostname": "cpu-fixture"}
    put(root / "registration.json", plan)
    for world in plan["worlds"]:
        location = root / f"world-{world['id']:03d}-{world['arm']}"
        location.mkdir()
        invocation = {"returncode": 0}
        for phase in ("before", "after"):
            path = location / f"occupancy-{phase}.txt"
            path.write_text(IDLE)
            invocation[f"occupancy_{phase}"] = {"idle": True, "sha256": bench.digest(path),
                                               "cards": [0]}
        put(location / "invocation.json", invocation)
        samples = []
        for sample in range(30):
            duration = 100 if world["arm"] == "control" else 88
            if tail and world["arm"] == "candidate":
                duration = 80 if sample < 28 else 130
            timings = [{"rank": rank, "start_ns": 100000000 + rank * skew_ns,
                        "enqueue_before_ns": 100000100 + rank * skew_ns,
                        "enqueue_after_ns": 100000200 + rank * skew_ns,
                        "end_ns": 100000000 + duration * 32000 + rank * skew_ns,
                        "event_span_us": duration * 32,
                        "host_span_us": duration * 32}
                       for rank in range(ranks)]
            samples.append({"rows": 1, "sample": sample, "all_rank_timing": timings,
                            "bounds": bench.timing_bounds(timings, 32), "pass": True,
                            "repetitions": 32, "guard_regions_intact": True,
                            "all_rank_checks": [True] * ranks,
                            "checks": {name: {"bitwise_equal": True}
                                       for name in ("first", "output", "input", "weight")}})
        for rank in range(ranks):
            own = [dict(sample, timing=sample["all_rank_timing"][rank]) for sample in samples]
            put(location / f"rank-{rank}.json", {
                "world": world, "rank": rank, "config": config,
                "registration_sha256": bench.digest(root / "registration.json"),
                "binding": plan["libraries"][world["arm"]], "host": "cpu-fixture",
                "all_ranks_completed": True, "pass": True, "samples": own})
    return plan


class BenchContract(unittest.TestCase):
    def test_quantiles_from_known_raw(self):
        self.assertEqual(bench.quantile(list(range(1, 102)), .95), 96)
        timings = [{"rank": 0, "start_ns": 1000, "enqueue_before_ns": 1100,
                    "enqueue_after_ns": 1200, "end_ns": 41000, "event_span_us": 30},
                   {"rank": 1, "start_ns": 2000, "enqueue_before_ns": 2100,
                    "enqueue_after_ns": 2200, "end_ns": 52000, "event_span_us": 40}]
        result = bench.timing_bounds(timings, 2)
        self.assertEqual(result["world_host_us_per_probe"], 25.5)
        self.assertEqual(result["world_event_us_per_probe"], 20)
        self.assertEqual(result["submission_envelope_us"], 1.2)
        timings[1]["rank"] = 0
        with self.assertRaises(ValueError):
            bench.timing_bounds(timings, 2)

    def test_complete_tp16_acceptance_and_pilot_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_evidence(root, cycles=15, ranks=16)
            result = bench.summarize(root)
            self.assertTrue(result["accepted"])
            self.assertEqual(result["world_pairs"], 30)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_evidence(root)
            result = bench.summarize(root)
            self.assertFalse(result["accepted"])
            self.assertTrue(result["pilot_only"])
            self.assertTrue(result["rows"]["1"]["metrics"]["event"]["hypothesis_pass"])

    def test_p95_regression_and_skew_are_not_filtered(self):
        for options in ({"tail": True}, {"skew_ns": 80000}):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                make_evidence(root, **options)
                result = bench.summarize(root)
                self.assertFalse(result["accepted"])
                if "tail" in options:
                    metric = result["rows"]["1"]["metrics"]["event"]
                    self.assertGreater(metric["p95_ratio"], 1)
                    self.assertFalse(metric["hypothesis_pass"])
                else:
                    self.assertFalse(result["rows"]["1"]["launch_skew_valid"])
                self.assertEqual(len(result["worlds"][0]["rows"]["1"]["event"]["raw"]), 30)

    def test_rank_loss_or_mutated_clocks_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_evidence(root)
            path = root / "world-000-control/rank-1.json"
            value = json.loads(path.read_text())
            value["samples"][0]["timing"]["end_ns"] += 1
            put(path, value)
            with self.assertRaises(ValueError):
                bench.summarize(root)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                bench.summarize(root)

    def test_process_guard_requires_idle_and_known_complete_format(self):
        self.assertEqual(bench.idle_process_table(IDLE), [0])
        for text in ("", IDLE.replace("No running processes found in NPU 0", "0 0 | 123 | python | 20"),
                     IDLE.replace("NPU 0", "NPU 1")):
            with self.assertRaises(ValueError):
                bench.idle_process_table(text)

    def test_frozen_driver_and_compiled_source_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "compiled.cpp").write_text("// CPU identity fixture, not a device artifact\n")
            library = root / "library-identity-fixture"
            library.write_bytes(b"CPU identity fixture")
            frozen = bench.HERE / "frozen/device_check_v1.py"
            manifest = {"sources": {"compiled.cpp": check.digest(root / "compiled.cpp")},
                        "experiment_sources": {"device_check.py": check.digest(frozen)}}
            put(root / "manifest.json", manifest)
            put(root / "build-report.json", {
                "prepared_manifest_sha256": check.digest(root / "manifest.json"),
                "arms": {"control": {"library": {"path": str(library),
                                                  "sha256": check.digest(library)}}}})
            with self.assertRaises(ValueError):
                check.bind_library(root, "control")
            _, binding = check.bind_library(root, "control", frozen)
            self.assertTrue(binding["driver_changed"])
            self.assertEqual(binding["compiled_manifest_driver_sha256"],
                             "0bda19190327e52fa1b1d4f0e3f11882fe616fcfd980387990c40dbcfd3eb875")
            (root / "compiled.cpp").write_text("// changed\n")
            with self.assertRaises(ValueError):
                check.bind_library(root, "control", frozen)

    def test_four_rank_order_and_finite_tie_inputs(self):
        def f32(value):
            return struct.unpack("<f", struct.pack("<f", value))[0]
        def bf16(bits):
            return struct.unpack("<f", struct.pack("<I", bits << 16))[0]
        bits = check.hccl_columns(4, "witness")[0][1]
        values = [bf16(value) for value in bits]
        tree = f32(f32(values[0] + values[2]) + f32(values[1] + values[3]))
        serial = f32(f32(f32(values[0] + values[1]) + values[2]) + values[3])
        self.assertEqual(tree, 2 ** -25)
        self.assertEqual(serial, 0)
        for family in ("witness", "finite", "ties", "edges"):
            for _, column in check.hccl_columns(4, family):
                self.assertTrue(all((value & 0x7F80) != 0x7F80 for value in column))
        for name, column in check.hccl_columns(4, "ties"):
            total = f32(sum(bf16(value) for value in column))
            bits = struct.unpack("<I", struct.pack("<f", total))[0]
            rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
            expected = {"even_bf16_tie": 0x3F80, "odd_bf16_tie": 0x3F82,
                        "negative_even_tie": 0xBF80, "negative_odd_tie": 0xBF82}
            self.assertEqual(rounded, expected[name])


if __name__ == "__main__":
    unittest.main()
