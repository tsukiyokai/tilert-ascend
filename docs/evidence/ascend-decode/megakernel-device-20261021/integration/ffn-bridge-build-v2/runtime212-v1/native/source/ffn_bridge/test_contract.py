"""Check adversarial fixture meaning and the pinned vendor capacity change."""

from pathlib import Path
import tempfile
import unittest

import torch

from fixtures import bridge_case, epilogue_reference, logits_cases
from prepare_vendor import NORM, TOPK, prepare


class BridgeContract(unittest.TestCase):
    def test_finite_extremes_and_boundary_tie(self):
        cases = {name: (value, bias) for name, value, bias in logits_cases(torch, 32)}
        self.assertEqual(len(cases), 8)
        for values, bias in cases.values():
            self.assertEqual(values.shape, (32, 256))
            self.assertTrue(torch.isfinite(values).all())
            self.assertTrue(torch.isfinite(bias).all())
        tied, _ = cases["eighth_ninth_tie"]
        scores = tied.sigmoid().sort(descending=True, stable=True).values
        self.assertTrue(torch.equal(scores[:, 7], scores[:, 8]))
        values, correction = cases["negative_correction"]
        self.assertTrue((correction < 0).all())
        self.assertFalse(torch.equal(values.argmax(-1),
                                    (values.sigmoid() + correction).argmax(-1)))

    def test_separate_bf16_epilogue_is_observable(self):
        generator = torch.Generator().manual_seed(99)
        routed = torch.randn(4096, generator=generator).bfloat16()
        shared = torch.randn(4096, generator=generator).bfloat16()
        _, expected = epilogue_reference(routed, shared)
        collapsed = (routed.float() * 2.5 + shared.float()).bfloat16()
        self.assertGreater(int((expected != collapsed).sum()), 0)
        negative_zero = torch.tensor([-0.0], dtype=torch.bfloat16)
        scaled, combined = epilogue_reference(negative_zero, negative_zero)
        self.assertTrue(torch.signbit(scaled).all())
        self.assertTrue(torch.signbit(combined).all())

    def test_zero_and_cancelled_residuals(self):
        for iteration in (4, 5):
            hidden, residual, *_ = bridge_case(torch, 2, iteration)
            self.assertEqual(int(torch.count_nonzero(hidden + residual)), 0)

    def test_overlay_changes_only_two_capacities(self):
        original = Path("/tmp/vllm-ascend-clone/va")
        if not original.is_dir():
            self.skipTest("Pinned vendor source is not available")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            records = prepare(original, destination)
            self.assertEqual(sum(item["changed"] for item in records), 1)
            source = (original / TOPK / "moe_gating_top_k_without_group.h").read_text()
            actual = (destination / "moe_gating_top_k_without_group.h").read_text()
            source_lines, actual_lines = source.splitlines(), actual.splitlines()
            self.assertEqual(len(source_lines), len(actual_lines))
            changed = [(a, b) for a, b in zip(source_lines, actual_lines) if a != b]
            self.assertEqual(len(changed), 2)
            self.assertTrue(all("expertCountAlign_ * sizeof(int32_t)" in b
                                for _, b in changed))
            for name in ("add_rms_norm_bias_single_n.h", "rms_norm_base.h", "reduce_common.h"):
                self.assertEqual((original / NORM / name).read_bytes(),
                                 (destination / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
