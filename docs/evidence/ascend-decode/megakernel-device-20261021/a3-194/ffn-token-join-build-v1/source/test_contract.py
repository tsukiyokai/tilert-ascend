"""Check ownership, rounding boundaries and the frozen token-output seams."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from prepare import BODY, BODY_MARKER, STAGE_FIVE, bind_body, bind_kernel, prepare


def raw(tensor):
    return tensor.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8)


def ordered_sum(routes, probabilities):
    """Unpermute order: slot zero seeds only when its probability is nonzero."""
    output = torch.zeros_like(routes[:, 0], dtype=torch.float32)
    for slot in range(8):
        for token in range(routes.shape[0]):
            probability = probabilities[token, slot].item()
            if probability == 0.0:
                continue
            term = routes[token, slot].float() * probability
            if slot == 0:
                output[token] = term
            else:
                output[token] = output[token] + term
    return output.to(torch.bfloat16)


def narrowed_tail(routed, shared):
    scaled = (routed.float() * 2.5).to(torch.bfloat16)
    return scaled, (scaled.float() + shared.float()).to(torch.bfloat16)


class TokenOutputContract(unittest.TestCase):
    parent = Path(os.environ.get("FFN_TOKEN_OUTPUT_PARENT", str(
        Path(__file__).resolve().parents[4]
        / "docs/evidence/ascend-decode/megakernel-device-20261021"
        / "a3-194/ffn-composed-build-v2")))

    def test_cpp_ownership_and_compact_dot_addresses(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "ownership"
            subprocess.run([
                "g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                str(Path(__file__).with_name("ownership_probe.cpp")),
                "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_all_preceding_math_and_six_body_barriers_are_unchanged(self):
        original = (self.parent / BODY).read_text()
        candidate = bind_body(original)
        prefix = candidate.split(BODY_MARKER, 1)[0].replace(
            '\n#include "../../../source/output_contract.h"', "").replace(
            '#include "../../../source/output_native.h"\n\n', "")
        self.assertEqual(prefix, original.split(BODY_MARKER, 1)[0])

        def body_prefix(value):
            start = value.index("\n{\n", value.index(BODY_MARKER)) + 3
            return value[start:value.index(STAGE_FIVE) + len(STAGE_FIVE)]

        self.assertEqual(body_prefix(candidate), body_prefix(original))
        self.assertEqual(candidate.count("timing.Barrier("), 5)
        for name in ("routed_int4_front.h", "routed_int4_dispatch.h"):
            self.assertEqual((self.parent / Path(BODY).with_name(name)).read_text().count(
                "timing.Barrier(0);"), 1)
        kernel = (self.parent / "source/kernel.cpp").read_text()
        patched = bind_kernel(kernel)
        self.assertEqual(patched.count("ffn_bridge::device::EndPhase(pipe);"), 2)
        self.assertEqual(patched.count("SetSysWorkspace("), 2)
        self.assertNotIn("ffn_bridge::device::Epilogue(", patched)

    def test_public_validators_scratch_and_projection_helpers_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            report = prepare(self.parent, output)
            for name in ("source/contract.h", "source/binding.cpp",
                         "body/source/routed_int4/host_validate.h",
                         "bridge/source/ffn_bridge/host_validate.h",
                         "body/source/routed_int4/routed_int4_front.h",
                         "body/source/routed_int4/routed_int4_dispatch.h",
                         "body/source/routed_int4/routed_int4_pipeline_native.h"):
                expected = (self.parent / name).read_text().replace(
                    "ffn_composed", "ffn_token_output").replace(
                    "FFN_COMPOSED", "FFN_TOKEN_OUTPUT")
                self.assertEqual((output / name).read_text(), expected)
            self.assertEqual(report["public_scratch_count"], 32)
            self.assertEqual(report["additional_gm_bytes"], 0)
            self.assertEqual(report["additional_ub_bytes"], 3072)
            self.assertEqual(report["maximum_resident_ub_bytes"], 115424)
            for name, expected in report["sources"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected)
            binary = Path(directory) / "public-contract-ownership"
            subprocess.run([
                "g++", "-std=c++17", "-O2", "-include",
                str(output / "source/contract.h"),
                str(Path(__file__).with_name("ownership_probe.cpp")),
                "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_slot_zero_negative_zero_and_zero_probability_nan(self):
        values = torch.full((1, 8, 16), -0.0, dtype=torch.bfloat16)
        probabilities = torch.full((1, 8), 0.125, dtype=torch.float32)
        result = ordered_sum(values, probabilities)
        negative = torch.full_like(result, -0.0)
        positive = torch.zeros_like(result)
        self.assertTrue(torch.equal(raw(result), raw(negative)))
        probabilities[:, 1:] = 0.0
        self.assertTrue(torch.equal(raw(ordered_sum(values, probabilities)), raw(negative)))
        probabilities[0, 0] = 0.0
        self.assertTrue(torch.equal(raw(ordered_sum(values, probabilities)), raw(positive)))
        values[0, 0] = float("nan")
        self.assertTrue(torch.equal(raw(ordered_sum(values, probabilities)), raw(positive)))
        probabilities[0, 0] = 0.125
        self.assertTrue(torch.isnan(ordered_sum(values, probabilities)).all())

    def test_routed_rounding_precedes_2_point_5_scaling(self):
        accumulator = torch.tensor([1.00390625, -1.00390625], dtype=torch.float32)
        retained = (accumulator.bfloat16().float() * 2.5).bfloat16()
        collapsed = (accumulator * 2.5).bfloat16()
        expected = torch.tensor([16416, -16352], dtype=torch.int16).view(torch.bfloat16)
        self.assertTrue(torch.equal(raw(retained), raw(expected)))
        self.assertFalse(torch.equal(raw(retained), raw(collapsed)))

    def test_ub_budget_retains_shared_store_allocations(self):
        span = 2048 + 64
        routed = [span, 4 * span, 4 * span, 4 * span, 2 * span,
                  span // 2 + 64, 4 * span, 4 * span, 4 * span,
                  32 * 4, 32 * 4, 32 * 128 * 4, 32 * 8 * 4,
                  *([256 * 4] * 6)]
        shared = [1024, 256 * 4, 256 * 4, (256 + 8) * 4,
                  256 * 4, 256 * 4, 8 * 4, 128 * 2, 128]
        shared_store = [8 * 256 * 4, 8 * 256 * 4, 8 * 256 * 2,
                        256 * 4, 256 * 4, 8 * 8 * 4, 256 * 4]
        metadata = [256 * 4] * 3
        allocations = routed + shared + shared_store + metadata
        self.assertTrue(all(size % 32 == 0 for size in allocations))
        self.assertEqual(sum(routed), 81952)
        self.assertEqual(sum(shared), 6592)
        self.assertEqual(sum(shared_store), 23808)
        self.assertEqual(sum(allocations), 115424)

    def test_route_and_both_epilogue_bfloat16_boundaries_are_required(self):
        generator = torch.Generator().manual_seed(20261022)
        before_route = torch.randn((2, 8, 1536), generator=generator)
        probabilities = torch.rand((2, 8), generator=generator).to(torch.bfloat16).float()
        routes = before_route.to(torch.bfloat16)
        routed = ordered_sum(routes, probabilities)
        shared = torch.randn((2, 1536), generator=generator).to(torch.bfloat16)
        scaled, combined = narrowed_tail(routed, shared)
        no_route_rounding = ordered_sum(before_route, probabilities)
        no_scaled_rounding = (routed.float() * 2.5 + shared.float()).to(torch.bfloat16)
        self.assertFalse(torch.equal(raw(no_route_rounding), raw(routed)))
        self.assertFalse(torch.equal(raw(no_scaled_rounding), raw(combined)))
        self.assertEqual(scaled.dtype, torch.bfloat16)

    def test_token_tiles_match_independent_route_major_boundaries(self):
        generator = torch.Generator().manual_seed(37)
        rows, hidden = 12, 6144
        # Inputs to this check are the original FP16 partial outputs and correction.
        high = torch.randn((rows, 8, hidden), generator=generator).half().float()
        low = torch.randn((rows, 8, hidden), generator=generator).half().float()
        correction = torch.randn((rows, 8, hidden), generator=generator)
        activation = torch.rand((rows, 8, 1), generator=generator)
        probabilities = torch.rand((rows, 8), generator=generator).to(torch.bfloat16).float()
        probabilities[::2, 0] = 0.0
        probabilities[1::2, 1::2] = 0.0
        shared = torch.randn((rows, hidden), generator=generator).to(torch.bfloat16)
        route_reference = (((high * 16.0 + low) + correction) * activation).to(torch.bfloat16)
        routed_reference = ordered_sum(route_reference, probabilities)
        scaled_reference, final_reference = narrowed_tail(routed_reference, shared)
        actual_routes = torch.empty_like(route_reference)
        actual_routed = torch.empty_like(routed_reference)
        actual_scaled = torch.empty_like(scaled_reference)
        actual_final = torch.empty_like(final_reference)
        for token in range(rows):
            for column in range(0, hidden, 1536):
                columns = slice(column, column + 1536)
                accumulator = torch.zeros((1536,), dtype=torch.float32)
                for slot in range(8):
                    route = (((high[token, slot, columns] * 16.0
                        + low[token, slot, columns]) + correction[token, slot, columns])
                        * activation[token, slot]).to(torch.bfloat16)
                    actual_routes[token, slot, columns] = route
                    probability = probabilities[token, slot].item()
                    if probability == 0.0:
                        continue
                    term = route.float() * probability
                    accumulator = term if slot == 0 else accumulator + term
                routed = accumulator.to(torch.bfloat16)
                scaled, final = narrowed_tail(routed, shared[token, columns])
                actual_routed[token, columns] = routed
                actual_scaled[token, columns] = scaled
                actual_final[token, columns] = final
        for actual, expected in ((actual_routes, route_reference),
                                 (actual_routed, routed_reference),
                                 (actual_scaled, scaled_reference),
                                 (actual_final, final_reference)):
            self.assertTrue(torch.equal(raw(actual), raw(expected)))

    def test_invalid_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "parent"
            parent.mkdir()
            (parent / "manifest.json").write_text(json.dumps({
                "artifacts": {"ffn_composed.so": "unverified"}}))
            with self.assertRaisesRegex(ValueError, "validated uninstrumented"):
                prepare(parent, Path(directory) / "output")


if __name__ == "__main__":
    unittest.main()
