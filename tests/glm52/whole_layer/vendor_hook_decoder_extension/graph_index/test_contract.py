"""Check exact reproduction and the failure-save observation lifetime on CPU."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent.parent)]
if (HERE / "decoder_graph_index_diagnostic.py").exists():
    from decoder_graph_index_diagnostic import (
        accepted_evidence, expected_numerical_failure, observe_failure_saves, reproduction)
else:
    from diagnostic import (
        accepted_evidence, expected_numerical_failure, observe_failure_saves, reproduction)


def fixture():
    return {"plan": {"index_slots": torch.tensor([1021], dtype=torch.int32)},
            "expected": {"output": torch.zeros(1, 4, dtype=torch.bfloat16)},
            "actual": {"output": torch.ones(1, 4, dtype=torch.bfloat16)}}


def case():
    return {"rows": 1, "initial_residual": "provided", "iteration": 2,
            "phase": "native_graph_replay", "inputs": {"hidden_sha256": "abc"}}


class GraphIndexContracts(unittest.TestCase):
    def test_stale_retained_key_or_scale_rejects_otherwise_reproduced_failure(self):
        inputs = {name: {"bitwise_equal": True} for name in ("hidden", "slots")}
        producers = {name: {"bitwise_equal": True} for name in (
            "vendor_key", "native_key", "vendor_scale", "native_scale")}
        self.assertTrue(accepted_evidence({"pass": True}, inputs, producers))
        for name in producers:
            changed = deepcopy(producers)
            changed[name]["bitwise_equal"] = False
            self.assertFalse(accepted_evidence({"pass": True}, inputs, changed))

    def test_only_accepted_exact_original_exception_is_handled(self):
        message = "The complete decoder graph differs from its eager oracle"
        accepted = {"intermediate_evidence_accepted": True}
        self.assertTrue(expected_numerical_failure(RuntimeError(message), accepted))
        for error, report in ((RuntimeError("device failure"), accepted),
                              (ValueError(message), accepted), (RuntimeError(message), None),
                              (RuntimeError(message), {"intermediate_evidence_accepted": False})):
            self.assertFalse(expected_numerical_failure(error, report))

    def test_reproduction_rejects_a_different_slot_byte_or_signed_zero(self):
        self.assertTrue(reproduction(torch, fixture(), fixture(), case(), case())["pass"])
        for collection, key in (("plan", "index_slots"), ("expected", "output"),
                                ("actual", "output")):
            current = fixture()
            value = current[collection][key]
            value.flatten()[0] = -0.0 if collection == "expected" else 7
            with self.subTest(collection=collection):
                self.assertFalse(reproduction(torch, fixture(), current, case(), case())["pass"])

    def test_reproduction_rejects_changed_input_witness_or_tensor_set(self):
        changed = deepcopy(case())
        changed["inputs"]["hidden_sha256"] = "def"
        self.assertFalse(reproduction(torch, fixture(), fixture(), case(), changed)["pass"])
        changed = fixture()
        changed["actual"]["extra"] = torch.zeros(1)
        self.assertFalse(reproduction(torch, fixture(), changed, case(), case())["pass"])

    def test_callback_reads_saved_failure_before_caller_teardown(self):
        original = torch.save
        events = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failure-native.pt"

            def callback(value, destination, save):
                self.assertTrue(path.is_file())
                self.assertIs(save, original)
                self.assertEqual(events, ["live_graph"])
                saved = torch.load(destination, weights_only=True)
                self.assertTrue(torch.equal(saved["actual"]["output"], value["actual"]["output"]))
                events.append("read_before_teardown")

            with observe_failure_saves(torch, callback):
                try:
                    events.append("live_graph")
                    torch.save(fixture(), path)
                    raise RuntimeError("original numerical failure")
                except RuntimeError as error:
                    self.assertEqual(str(error), "original numerical failure")
                finally:
                    events.append("graph_teardown")
        self.assertIs(torch.save, original)
        self.assertEqual(events, ["live_graph", "read_before_teardown", "graph_teardown"])

    def test_unrelated_saves_do_not_observe_and_errors_restore_original(self):
        original = torch.save
        calls = []

        def fail(*args):
            calls.append(args)
            raise RuntimeError("callback failure")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "callback failure"):
                with observe_failure_saves(torch, fail):
                    torch.save(fixture(), Path(directory) / "failure-vendor.pt")
                    self.assertEqual(calls, [])
                    torch.save(fixture(), Path(directory) / "failure-native.pt")
        self.assertEqual(len(calls), 1)
        self.assertIs(torch.save, original)


if __name__ == "__main__":
    unittest.main()
