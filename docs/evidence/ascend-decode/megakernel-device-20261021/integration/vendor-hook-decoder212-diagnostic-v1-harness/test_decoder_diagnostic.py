"""CPU tests of diagnostic interception, persistence, and storage ownership."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decoder_diagnostic import DiagnosticTrace, count_contract, observe_original_calls


class OriginalScheme:
    pass


def original_dispose(value):
    value.set_(torch.empty(0, dtype=value.dtype))


class DiagnosticContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = torch.library.Library("decoder_diagnostic_cpu", "DEF")
        cls.library.define(
            "grouped_matmul_swiglu_quant_v2(Tensor x, Tensor[] weight, "
            "Tensor[] weight_scale, Tensor x_scale, Tensor group_list, "
            "int group_list_type=1) -> (Tensor, Tensor)")
        cls.order = []

        def operation(x, weight, weight_scale, x_scale, group_list, group_list_type=1):
            cls.order.append("original")
            return x.clone(), x_scale.clone()

        cls.library.impl("grouped_matmul_swiglu_quant_v2", operation, "CPU")

    def make_trace(self, output, synchronize):
        return DiagnosticTrace(torch, None, [], Path(output), synchronize, lambda: 123)

    def invoke(self, trace, namespace):
        values = {
            "x": torch.arange(8 * 32, dtype=torch.int8).reshape(8, 32),
            "weight": [torch.ones(256, 32, 4, dtype=torch.int32)],
            "weight_scale": [torch.ones(256, 32, dtype=torch.int64)],
            "x_scale": torch.ones(8),
            "group_list": torch.tensor([0, 3, 0, 5] + [0] * 252, dtype=torch.int64),
            "group_list_type": 1,
        }
        result = namespace.grouped_matmul_swiglu_quant_v2(**values)
        return values, result

    def test_counts_and_prefixes_expose_out_of_bounds(self):
        self.assertTrue(count_contract([0, 3, 0, 5], 1, 8)["valid"])
        self.assertTrue(count_contract([0, 3, 3, 8], 0, 8)["valid"])
        self.assertFalse(count_contract([0, 3, 0, 8], 1, 8)["valid"])
        self.assertFalse(count_contract([0, -1, 9], 1, 8)["valid"])
        self.assertEqual(count_contract([0, 3, 0, 5], 1, 8)["cumulative_ends"],
                         [0, 3, 3, 8])

    def test_real_torch_dispatch_is_fenced_without_changing_arguments(self):
        self.order.clear()
        namespace = torch.ops.decoder_diagnostic_cpu
        original = namespace.grouped_matmul_swiglu_quant_v2
        scheme = SimpleNamespace(dispose_tensor=original_dispose,
                                 AscendW4A8DynamicFusedMoEMethod=OriginalScheme)
        with tempfile.TemporaryDirectory() as output:
            trace = self.make_trace(output, lambda: self.order.append("sync"))
            with observe_original_calls(trace, [], namespace, scheme):
                values, result = self.invoke(trace, namespace)
                self.assertTrue(torch.equal(values["x"], result[0]))
                scheme.dispose_tensor(values["x"])
            self.assertEqual(self.order, ["sync", "original", "sync"])
            self.assertIs(namespace.grouped_matmul_swiglu_quant_v2, original)
            self.assertIs(scheme.dispose_tensor, original_dispose)
            self.assertEqual(values["x"].numel(), 0)
            self.assertEqual(trace.observed[0][1]().numel(), 0)
            snapshot = torch.load(Path(output) / "gmm1-call-000-inputs.pt", weights_only=True)
            self.assertEqual(tuple(snapshot["x"].shape), (8, 32))
            self.assertEqual(trace.events[-1]["tensor"]["storage_bytes"], 0)

    def test_before_failure_persists_and_never_submits(self):
        self.order.clear()
        namespace = torch.ops.decoder_diagnostic_cpu
        original = namespace.grouped_matmul_swiglu_quant_v2
        scheme = SimpleNamespace(dispose_tensor=original_dispose,
                                 AscendW4A8DynamicFusedMoEMethod=OriginalScheme)

        def fail():
            raise RuntimeError("queued producer failed")

        with tempfile.TemporaryDirectory() as output:
            trace = self.make_trace(output, fail)
            with self.assertRaisesRegex(RuntimeError, "queued producer"):
                with observe_original_calls(trace, [], namespace, scheme):
                    self.invoke(trace, namespace)
            self.assertEqual(self.order, [])
            self.assertEqual(trace.events[-1]["event"], "gmm1_producers_failed")
            self.assertIs(namespace.grouped_matmul_swiglu_quant_v2, original)
            self.assertIn("gmm1_producers_failed",
                          (Path(output) / "decoder-diagnostic.json").read_text())

    def test_after_failure_is_before_caller_disposal(self):
        self.order.clear()
        namespace = torch.ops.decoder_diagnostic_cpu
        scheme = SimpleNamespace(dispose_tensor=original_dispose,
                                 AscendW4A8DynamicFusedMoEMethod=OriginalScheme)

        def fail_second_sync():
            self.order.append("sync")
            if self.order.count("sync") == 2:
                raise RuntimeError("original kernel failed")

        with tempfile.TemporaryDirectory() as output:
            trace = self.make_trace(output, fail_second_sync)
            with self.assertRaisesRegex(RuntimeError, "original kernel"):
                with observe_original_calls(trace, [], namespace, scheme):
                    self.invoke(trace, namespace)
            self.assertEqual(self.order, ["sync", "original", "sync"])
            self.assertEqual(trace.events[-1]["event"], "gmm1_completion_failed")
            self.assertFalse(any(event["event"] == "dispose_enter" for event in trace.events))


if __name__ == "__main__":
    unittest.main()
