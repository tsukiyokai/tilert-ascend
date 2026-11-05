"""Prove arch22 C8 rounding witnesses with standard-library arithmetic."""

import argparse
import ast
import hashlib
import json
import math
import platform
import struct
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tilert.models.glm_5_2_ascend.ops.indexer import (
    CANN_ARCH22_SOURCE_SHA256,
    cann_arch22_input_domain,
    cann_arch22_scoring_contract,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    Path(__file__).relative_to(ROOT),
    Path("tilert/models/glm_5_2_ascend/ops/indexer.py"),
    Path("tilert/models/glm_5_2_ascend/reference.py"),
    Path("tests/glm52/check_indexer.py"),
    Path("tests/glm52/check_a8_shapes.py"),
    Path("tests/glm52/run_hf_reference.py"),
    Path("tests/glm52/run_mtp_reference.py"),
    Path("tests/glm52/run_native_decode.py"),
    Path("tests/glm52/run_native_mtp.py"),
    Path("tests/glm52/check_hf_reference.py"),
    Path("tests/glm52/check_mtp_contract.py"),
)


def fp32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fp16(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def bits(value, dtype):
    floating, integer, width = {
        "fp16": ("e", "H", 4), "fp32": ("f", "I", 8),
    }[dtype]
    word = struct.unpack("<" + integer, struct.pack("<" + floating, value))[0]
    return f"0x{word:0{width}x}"


def vector(entries):
    result = [0] * 128
    for coordinate, value in entries.items():
        result[coordinate] = value
    return result


def scalar_scores(query, keys, query_scale, key_scale, weights):
    """Use exact INT32 dots and individually rounded scalar FP32 operations."""
    ideal, source, stages = [], [], []
    for key, scale in zip(keys, key_scale):
        ideal_score, source_score = 0.0, 0.0
        key_stages = []
        for head, (row, qscale, weight) in enumerate(zip(
            query, query_scale, weights
        )):
            dot = sum(left * right for left, right in zip(row, key))
            product = fp32(fp32(dot * qscale) * scale)
            ideal_score = fp32(
                ideal_score + fp32(max(product, 0.0) * weight)
            )
            rounded_dot = fp16(max(fp32(dot / 1024.0), 0.0))
            rounded_weight = fp16(fp32(weight * qscale))
            source_score = fp32(
                source_score + fp32(rounded_dot * rounded_weight)
            )
            if weight:
                key_stages.append({
                    "head": head, "int32_dot": dot,
                    "weight_fp16_bits": bits(weight, "fp16"),
                    "query_scale_fp16_bits": bits(qscale, "fp16"),
                    "weight_scale_before_round": fp32(weight * qscale),
                    "weight_scale_fp16": rounded_weight,
                    "weight_scale_fp16_bits": bits(rounded_weight, "fp16"),
                    "positive_dot_fp16": rounded_dot,
                    "positive_dot_fp16_bits": bits(rounded_dot, "fp16"),
                })
        ideal.append(ideal_score)
        source.append(fp32(source_score * scale))
        stages.append(key_stages)
    return ideal, source, stages


def cutoff(scores):
    """Expand four score classes into 8192 positions with a strict cutoff."""
    sentinel, first, second, zero = scores
    expanded = [sentinel] * 2047 + [first, second] + [zero] * 6143
    order = sorted(range(8192), key=lambda index: (-expanded[index], index))
    selected = order[:2048]
    return {
        "selected": selected,
        "cutoff": expanded[order[2047]],
        "next": expanded[order[2048]],
        "strict_boundary": expanded[order[2047]] > expanded[order[2048]],
    }


def witness(kind):
    query = [vector({10: 127}) for _ in range(32)]
    weights = [0.0] * 32
    query_scale = [1.0] * 32
    if kind == "qk_fp16":
        query[0] = vector({0: 16, 1: 1, 3: 127})
        query[1] = vector({2: 1, 4: 127})
        weights[:2] = [1.0, 0.5]
        entries = [
            {0: 127, 1: 127}, {0: 127, 1: 16, 2: 1},
            {0: 127, 1: 17}, {5: 127},
        ]
    elif kind == "weight_query_scale_fp16":
        query[0] = vector({0: 16, 1: 1, 6: 127})
        query[1] = vector({2: 16, 3: 1, 7: 127})
        query[2] = vector({4: 1, 8: 127})
        query_scale[0] = 129 / 128
        weights[:3] = [129 / 128, 1.0, 1 / 16]
        entries = [
            {0: 127, 1: 127}, {0: 127, 1: 16},
            {2: 127, 3: 48, 4: 1}, {9: 127},
        ]
    else:
        raise ValueError("Unknown rounding witness")
    keys = [vector(entry) for entry in entries]
    ideal, source, stages = scalar_scores(
        query, keys, query_scale, [1.0] * len(keys), weights
    )
    domain = cann_arch22_input_domain(query_scale, [1.0] * 8192, weights)
    return {
        "query_shape": [32, 128], "key_shape": [8192, 128],
        "query_nonzero": [
            {str(index): value for index, value in enumerate(row) if value}
            for row in query
        ],
        "key_class_nonzero": entries,
        "key_class_ranges": [[0, 2047], [2047, 2048], [2048, 2049],
                             [2049, 8192]],
        "query_scale": query_scale, "key_scale": 1.0, "weights": weights,
        "ideal_scores": ideal, "arch22_scores": source,
        "arch22_score_fp32_bits": [bits(value, "fp32") for value in source],
        "stages": stages, "fp16_domain": domain,
        "ideal_topk": cutoff(ideal), "arch22_topk": cutoff(source),
    }


class ScoringChecks(unittest.TestCase):
    source_directory = None
    fixtures = {}

    def assert_cutoff_reversal(self, name, ideal_pair, source_pair):
        result = witness(name)
        self.fixtures[name] = result
        self.assertEqual(result["ideal_scores"][1:3], ideal_pair)
        self.assertEqual(result["arch22_scores"][1:3], source_pair)
        self.assertTrue(result["fp16_domain"]["accepted"])
        ideal = result["ideal_topk"]
        source = result["arch22_topk"]
        self.assertTrue(ideal["strict_boundary"])
        self.assertTrue(source["strict_boundary"])
        self.assertEqual(len(set(ideal["selected"]) &
                             set(source["selected"])), 2047)
        self.assertEqual(set(ideal["selected"]) ^ set(source["selected"]),
                         {2047, 2048})
        for summary in (ideal, source):
            selected = summary.pop("selected")
            summary["selected_count"] = len(selected)
            summary["selected_special_indices"] = [
                index for index in selected if index in (2047, 2048)
            ]
            summary["selected_int32_sha256"] = hashlib.sha256(
                struct.pack("<2048i", *sorted(selected))
            ).hexdigest()

    def test_qk_fp16_cutoff_reversal(self):
        self.assert_cutoff_reversal(
            "qk_fp16", [2048.5, 2049.0], [2.00048828125, 2.0]
        )

    def test_weight_scale_fp16_cutoff_reversal(self):
        self.assert_cutoff_reversal(
            "weight_query_scale_fp16", [2080.125, 2080.0625],
            [2.03125, 2.03131103515625],
        )

    def test_qk_conversion_range(self):
        self.assertEqual(fp16((128 * 128**2) / 1024), 2048.0)
        self.assertEqual(fp16(1 / 1024), 2**-10)
        self.assertGreater(2**-10, 2**-14)
        self.assertEqual(fp16(2049 / 1024), 2.0)

    def test_fp16_domain(self):
        for first_weight, first_scale, accepted in (
            (0.0, 0.0, True), (-1.0, 1.0, True),
            (65504.0, 1.0, True), (2**-14, 1.0, True),
            (2**-15, 1.0, False), (2**-14, 0.5, False),
            (2**-14, 2**-14, False), (65504.0, 2.0, False),
            (math.inf, 1.0, False), (math.nan, 1.0, False),
            (1.0, -1.0, False),
        ):
            with self.subTest(weight=first_weight, scale=first_scale):
                weights = [first_weight] + [1.0] * 31
                scales = [first_scale] + [1.0] * 31
                result = cann_arch22_input_domain(scales, [1.0], weights)
                self.assertEqual(result["accepted"], accepted)
        self.assertFalse(cann_arch22_input_domain(
            [1.0] * 32, [2**-15], [1.0] * 32
        )["accepted"])

    def test_source_binding(self):
        contract = cann_arch22_scoring_contract(self.source_directory)
        self.assertTrue(contract["source_verified"])
        self.assertEqual(contract["source_sha256"], CANN_ARCH22_SOURCE_SHA256)
        self.assertEqual(contract["device_validation"], "pending")

    def test_python_sources_and_reference_wiring(self):
        for name in SOURCE_FILES:
            ast.parse((ROOT / name).read_text(), feature_version=(3, 11))
        tree = ast.parse((ROOT / SOURCE_FILES[2]).read_text())
        indexer = next(node for node in tree.body
                       if isinstance(node, ast.ClassDef)
                       and node.name == "ReferenceIndexer")
        calls = {node.func.id for node in ast.walk(indexer)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)}
        self.assertIn("cann_arch22_index_scores", calls)
        self.assertIn("cann_arch22_input_domain", calls)
        self.assertNotIn("torch", sys.modules)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cann-source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ScoringChecks.source_directory = args.cann_source_dir
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ScoringChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "python": platform.python_version(),
        "command": [sys.executable, *sys.argv],
        "status": "pass" if result.wasSuccessful() else "fail",
        "tests_run": result.testsRun,
        "failures": [str(test) for test, _ in result.failures],
        "errors": [str(test) for test, _ in result.errors],
        "scope": "stdlib witnesses, input domain, source hashes, Python AST",
        "torch_executed": False, "npu_executed": False,
        "historical_indexer_report_revalidated": False,
        "source_sha256": {
            str(name): hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES
        },
        "cann_source_sha256": CANN_ARCH22_SOURCE_SHA256,
        "fixtures": ScoringChecks.fixtures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.write_text(payload)
    print(json.dumps({key: report[key] for key in ("status", "tests_run")}))
    return int(not result.wasSuccessful())


if __name__ == "__main__":
    raise SystemExit(main())
