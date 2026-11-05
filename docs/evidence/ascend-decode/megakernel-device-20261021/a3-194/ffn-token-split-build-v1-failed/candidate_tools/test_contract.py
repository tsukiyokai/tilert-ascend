"""Check the token-owned split against the frozen expert-major contract."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from prepare import BODY, SPLIT_END, SPLIT_START, bind_body, digest, prepare, rename


def raw(tensor):
    return tensor.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8)


class TokenSplitContract(unittest.TestCase):
    parent = Path(os.environ.get("FFN_TOKEN_SPLIT_PARENT", str(
        Path(__file__).resolve().parents[4]
        / "docs/evidence/ascend-decode/megakernel-device-20261021/integration"
        / "ffn-active-token-join-build-v1/build-v1")))

    def test_actual_cpp_owner_and_exhaustive_int8_pair_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "ownership"
            subprocess.run([
                "g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                str(Path(__file__).with_name("ownership_probe.cpp")),
                "-o", str(binary)], check=True)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout), {
                "ownership_configurations": 2176, "packed_pairs": 65536, "pass": True})

    def test_split_integer_values_survive_existing_cast_boundaries(self):
        codes = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8)
        values = codes.half().float()
        high = torch.floor(values * (1.0 / 16.0))
        low = high * -16.0
        low = low + values
        low = low + -8.0
        expected_high = torch.div(codes.int(), 16, rounding_mode="floor").float()
        expected_low = ((codes.int() & 15) - 8).float()
        self.assertTrue(torch.equal(raw(high), raw(expected_high)))
        self.assertTrue(torch.equal(raw(low), raw(expected_low)))
        self.assertTrue(torch.equal(raw(high.half().float()), raw(high)))
        self.assertTrue(torch.equal(raw(low.half().float()), raw(low)))
        self.assertTrue(torch.equal(16 * high + low + 8, values))

    def test_only_phase_one_enumeration_changes(self):
        original = (self.parent / BODY).read_text()
        candidate = bind_body(original)
        stripped = candidate.replace('\n#include "../../../source/split_contract.h"', "")
        stripped = stripped.replace('\n#include "../../../source/split_native.h"', "")
        self.assertEqual(stripped.split(SPLIT_START)[0], original.split(SPLIT_START)[0])
        self.assertEqual(stripped.split(SPLIT_END)[1], original.split(SPLIT_END)[1])
        self.assertEqual(candidate.count("timing.Barrier("), 4)
        front = self.parent / "body/source/routed_int4/routed_int4_front.h"
        self.assertEqual(front.read_text().count("timing.Barrier(0);"), 1)
        self.assertEqual((self.parent / "source/kernel.cpp").read_text().count(
            "ffn_bridge::device::EndPhase(pipe);"), 1)

    def test_frozen_public_abi_descriptor_and_other_math_are_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            report = prepare(self.parent, output)
            manifest = json.loads((self.parent / "manifest.json").read_text())
            for name in manifest["sources"]:
                if (name == BODY or name.startswith("candidate_tools/")
                        or name in ("check.py", "parent-manifest.json", "source-transforms.json")):
                    continue
                expected = (self.parent / name).read_bytes()
                if Path(name).suffix in (".cpp", ".h"):
                    expected = rename(expected.decode()).encode()
                self.assertEqual((output / name).read_bytes(), expected, name)
            self.assertEqual((output / "check.py").read_text(),
                (self.parent / "check.py").read_text().replace(
                    "torch.ops.tilert_ffn_active_token_join", "torch.ops.tilert_ffn_token_split"))
            self.assertEqual(report["public_scratch_count"], 33)
            self.assertEqual(report["descriptor_bytes"], 408)
            for name, identity in report["sources"].items():
                self.assertEqual(digest(output / name), identity)
            binary = Path(directory) / "descriptor"
            subprocess.run([
                "g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                "-DROUTED_ACTIVE_LIST", "-DROUTED_ROW_BALANCED",
                "-DROUTED_SHARED_PIPELINE", "-DROUTED_INPUT_FRONT",
                "-I", str(output / "source"),
                str(output / "source/descriptor_probe.cpp"), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_existing_buffer_lifetimes_and_zero_probability_routes(self):
        source = Path(__file__).with_name("split_native.h").read_text()
        self.assertNotIn("InitBuffer", source)
        self.assertNotIn("Reset", source)
        self.assertNotIn("probabilit", source)
        self.assertNotIn("original_weight", source)
        self.assertIn("slot < kTopK", source)
        self.assertLess(source.index("Fence<HardEvent::MTE2_S>();"),
                        source.index("inverse.SetValue"))
        self.assertLess(source.index("DataSyncBarrier<MemDsbT::UB>();"),
                        source.index("inverse.GetValue"))
        self.assertLess(source.index("Cast(low_packed"),
                        source.index("Fence<HardEvent::V_MTE3>();"))
        self.assertTrue(source.rstrip().endswith(
            "Fence<HardEvent::MTE3_V>();\n"
            "        Fence<HardEvent::MTE3_MTE2>();\n    }\n}\n\n#endif"))

    def test_ub_aliases_fit_without_growing_parent_footprint(self):
        span = 2048 + 64
        inverse_and_ids = 2 * 256 * 4
        self.assertLessEqual(inverse_and_ids, span * 4)
        self.assertLessEqual(2048 // 2, span)
        self.assertLessEqual(2048 // 2, span // 2 + 64)
        parent = json.loads((self.parent / "manifest.json").read_text())
        self.assertEqual(parent["maximum_resident_ub_bytes"], 116480)
        with tempfile.TemporaryDirectory() as directory:
            report = prepare(self.parent, Path(directory) / "candidate")
        self.assertEqual(report["additional_ub_bytes"], 0)
        self.assertEqual(report["additional_gm_bytes"], 0)
        self.assertEqual(report["maximum_resident_ub_bytes"], 116480)

    def test_work_reduction_does_not_claim_fewer_route_writes(self):
        rows, hidden, topk = 12, 6144, 8
        original_tiles = rows * topk * hidden // 2048
        token_tiles = rows * hidden // 2048
        self.assertEqual((original_tiles, token_tiles), (288, 36))
        self.assertEqual((rows * topk * hidden, rows * hidden), (589824, 73728))
        self.assertEqual(original_tiles * 2, token_tiles * topk * 2)
        self.assertEqual(rows * hidden // 1536 * topk * 2, 768)
        self.assertLess(token_tiles, 2 * 24)


if __name__ == "__main__":
    unittest.main()
