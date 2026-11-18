"""CPU artifact-integrity and coverage checks for the saved strict comparator."""

import copy
import json
import tempfile
import unittest
from importlib import util
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
spec = util.spec_from_file_location("saved_precision", HERE / "compare.py")
saved = util.module_from_spec(spec)
spec.loader.exec_module(saved)


def frame():
    hidden = torch.ones(1, 6144, dtype=torch.bfloat16)
    logits = torch.arange(154880, dtype=torch.float32).reshape(1, -1)
    cache = []
    for layer in range(78):
        full = layer < 3 or (layer - 2) % 4 == 0
        quantized = full and 6 <= layer <= 66
        cache.append(
            {
                "latent": torch.ones(1, 512, dtype=torch.bfloat16),
                "rope": torch.ones(1, 64, dtype=torch.bfloat16),
                "index": torch.ones(1, 128, dtype=torch.int8 if quantized else torch.bfloat16)
                if full
                else None,
                "index_scale": torch.ones(1, dtype=torch.float16) if quantized else None,
            }
        )
    return {
        "token": [1],
        "position": 8192,
        "committed_position": 8193,
        "token_ids": torch.tensor([154879]),
        "logits": logits,
        "last_hidden": hidden,
        "hidden_states": [hidden] * 79,
        "layer_logits": [logits] * 78,
        "indices": [torch.arange(2048, dtype=torch.int32).reshape(1, -1)] * 78,
        "routes": [None] * 3 + [torch.arange(8, dtype=torch.int32).reshape(1, -1)] * 75,
        "cache_rows": cache,
    }


def write_receipts(directory):
    states = [{"phase": "serial", "position": 8224, "predictions": [1] * 32}]
    flags = {
        "case_precision_passed": False,
        "precision_verified": False,
        "formal_precision_gate_passed": False,
        "performance_measured": False,
        "compiled_graph_verified": False,
        "super_kernel_verified": False,
    }
    common = {
        **flags,
        "mode": "capture_only",
        "status": "captured_unverified",
        "capture_complete": True,
        "capture_checks_passed": True,
        "entry_passed": True,
        "checkpoint_sha256": {"config": "identity"},
        "source_sha256": saved.entry.source_identity(),
        "source_commit": "0" * 40,
        "command": ["python", "run.py", "--capture-only"],
        "execution": {
            "widths": [1, 32],
            "layers": 78,
            "fp32_head": True,
            "attention_kernel": "sfa",
            "enforce_eager": True,
            "compile_target": False,
            "super_kernel": False,
            "graph": None,
            "mtp_model_loaded": False,
            "capacity": 9344,
        },
        "model_tp_reduction": "float32",
        "loaded_libraries_sha256": {"cpu.fixture": "identity"},
        "packages": {name: "cpu.fixture" for name in ("torch", "torch-npu", "vllm", "vllm-ascend")},
    }
    receipts = []
    for rank in range(16):
        report = {
            **common,
            "schema": "cann-single-case-eager-v1",
            "rank": rank,
            "device": f"npu:{rank}",
            "world_size": 16,
            "npu_executed": True,
            "timed": False,
            "profiled": False,
            "checks": [{"name": "cpu.fixture", "passed": True}],
            "states": states,
            "final_position": 0,
            "trace_sha256": dict.fromkeys(saved.TRACE_NAMES, "raw"),
            "teacher": {"sha256": "teacher"},
        }
        path = directory / f"rank-{rank}.json"
        path.write_text(json.dumps(report))
        receipts.append(
            {
                "rank": rank,
                "report_sha256": saved.digest(path),
                "states": states,
                "final_position": 0,
            }
        )
    summary = {
        **common,
        "ranks": receipts,
        "all_rank_state_agreement": True,
        "teacher_sha256": "teacher",
    }
    (directory / "summary.json").write_text(json.dumps(summary))
    exit_receipt = directory / "launcher-exit.json"
    exit_receipt.write_text(
        json.dumps(
            {
                "schema": "cann-capture-exit-v1",
                "returncode": 0,
                "summary_sha256": saved.digest(directory / "summary.json"),
                "source_commit": common["source_commit"],
                "command": common["command"],
            }
        )
    )
    return exit_receipt


class SavedArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_complete_actual_layout_precedes_original_strict_comparison(self):
        actual = frame()
        saved.validate_frame(actual, [1], 8192)
        row = saved.entry.single_row(actual, 0)
        expected = copy.deepcopy(row)
        checks = []
        saved.entry.strict_row("cpu", row, expected, checks, exact=True)
        self.assertTrue(all(item["passed"] for item in checks))
        expected["logits"][0, 0] += 1
        checks = []
        saved.entry.strict_row("cpu", row, expected, checks, exact=True)
        self.assertFalse(all(item["passed"] for item in checks))

    def test_skippable_missing_layers_truncated_head_and_invalid_identity_are_rejected(
        self,
    ):
        for kind in ("missing", "short", "head", "typed", "cache"):
            actual = frame()
            if kind == "missing":
                del actual["hidden_states"]
            elif kind == "short":
                actual["layer_logits"].pop()
            elif kind == "head":
                actual["logits"] = actual["logits"][:, :17]
            elif kind == "typed":
                actual["token"] = [True]
            else:
                actual["cache_rows"][6]["index_scale"] = torch.zeros(1, 2)
            with self.assertRaises((KeyError, ValueError), msg=kind):
                saved.validate_frame(actual, [1], 8192)

    def test_epoch_stamps_cannot_be_stale_partial_or_float(self):
        stamps = torch.full((234,), 2, dtype=torch.int64)
        saved.validate_stamps(stamps, 2)
        for bad in (stamps[:233], stamps.float(), stamps - 1):
            with self.assertRaises(ValueError):
                saved.validate_stamps(bad, 2)

    def test_exit_and_report_byte_binding_cannot_be_replaced_by_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            exit_receipt = write_receipts(directory)
            _, reports, _ = saved.capture_receipts(directory, exit_receipt)
            self.assertEqual([report["rank"] for report in reports], list(range(16)))
            completed = json.loads(exit_receipt.read_text())
            exit_receipt.write_text(json.dumps({**completed, "returncode": 137}))
            with self.assertRaisesRegex(ValueError, "exit zero"):
                saved.capture_receipts(directory, exit_receipt)
            exit_receipt.write_text(json.dumps(completed))
            with (directory / "rank-7.json").open("a") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                saved.capture_receipts(directory, exit_receipt)

    def test_failed_capture_and_noninteger_rank_cannot_upgrade(self):
        for mutation in ("failed", "rank"):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                exit_receipt = write_receipts(directory)
                path = directory / "summary.json"
                summary = json.loads(path.read_text())
                if mutation == "failed":
                    summary["status"] = "captured_with_failures"
                else:
                    summary["ranks"][0]["rank"] = False
                path.write_text(json.dumps(summary))
                with self.assertRaises(ValueError):
                    saved.capture_receipts(directory, exit_receipt)

    def test_removed_prefix_checks_cannot_hide_behind_other_checks(self):
        prefixes = [{"latent": torch.ones(1)}]
        checks = [
            {"name": phase + ".prefix.0.latent", "passed": True, "exact": True}
            for phase in ("serial", "wide", "reset")
        ]
        saved.validate_prefix_checks({"checks": checks}, prefixes)
        with self.assertRaises(ValueError):
            saved.validate_prefix_checks({"checks": checks[:2]}, prefixes)


if __name__ == "__main__":
    unittest.main()
