"""CPU checks for explicit TP cut routing; these do not validate vendor execution."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decoder_forward import TPCuts, decoder_cuts, tensor_witness


class CutContract(unittest.TestCase):
    def test_external_input_and_local_output_are_distinct_contracts(self):
        external = torch.full((2, 6144), 3, dtype=torch.bfloat16)
        partial = torch.full_like(external, 7)
        cut = TPCuts(torch, external)
        cut.phase = "attention"
        result = cut.reduce(partial)
        self.assertTrue(torch.equal(result, external))
        self.assertNotEqual(result.data_ptr(), external.data_ptr())
        partial.fill_(11)
        self.assertTrue(bool((cut.values[0] == 7).all()))
        cut.phase = "ffn"
        self.assertIs(cut.reduce(partial), partial)
        cut.phase = None
        cut.finish()
        self.assertEqual([item["action"] for item in cut.trace],
                         ["external_post_ar1", "local_pre_ar2"])

    def test_missing_early_and_extra_reductions_are_rejected(self):
        value = torch.ones(1, 6144, dtype=torch.bfloat16)
        cut = TPCuts(torch, value)
        with self.assertRaises(RuntimeError):
            cut.finish()
        cut.phase = "ffn"
        with self.assertRaises(RuntimeError):
            cut.reduce(value)
        cut.phase = "attention"
        cut.reduce(value)
        with self.assertRaises(RuntimeError):
            cut.reduce(value)
        cut.phase = "ffn"
        cut.reduce(value)
        with self.assertRaises(RuntimeError):
            cut.reduce(value)

    def test_boundary_geometry_and_dtype_are_checked(self):
        for value in (torch.ones(1, 6144), torch.ones(0, 6144, dtype=torch.bfloat16),
                      torch.ones(2, 512, dtype=torch.bfloat16)):
            with self.subTest(shape=tuple(value.shape), dtype=value.dtype):
                with self.assertRaises(ValueError):
                    TPCuts(torch, value)
        cut = TPCuts(torch, torch.ones(1, 6144, dtype=torch.bfloat16))
        cut.phase = "attention"
        with self.assertRaises(RuntimeError):
            cut.reduce(torch.ones(1, 6144))

    def test_context_restores_method_and_hooks_after_failure(self):
        original = lambda value: value
        group = SimpleNamespace(all_reduce=original)
        layer = SimpleNamespace(self_attn=torch.nn.Identity(), mlp=torch.nn.Identity())
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with decoder_cuts(torch, group, layer, torch.ones(1, 6144, dtype=torch.bfloat16)):
                raise RuntimeError("fixture failure")
        self.assertIs(group.all_reduce, original)
        for module in (layer.self_attn, layer.mlp):
            self.assertFalse(module._forward_pre_hooks)
            self.assertFalse(module._forward_hooks)

    def test_raw_witness_distinguishes_signed_zero(self):
        positive = torch.tensor([0.0], dtype=torch.bfloat16)
        negative = torch.tensor([-0.0], dtype=torch.bfloat16)
        self.assertNotEqual(tensor_witness(torch, positive)["sha256"],
                            tensor_witness(torch, negative)["sha256"])


if __name__ == "__main__":
    unittest.main()
