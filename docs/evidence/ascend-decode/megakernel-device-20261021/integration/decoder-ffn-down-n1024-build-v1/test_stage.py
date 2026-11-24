"""Check the original-Decoder oracle and complete prepared Down record."""

import ast
import hashlib
import json
from pathlib import Path
import unittest

import torch

from stage_decoder import check_down_tiling, check_original_functions


ROOT = Path("/tmp/tilert-megakernel-validation-20261021")
PARENT = ROOT / "ffn-dual-quant-decoder-parent/vendor-hook-decoder212-ffn-v2"
STAGED = ROOT / "vendor-hook-decoder212-ffn-down-n1024-v1"


class DecoderGateContract(unittest.TestCase):
    def test_original_reference_and_inputs_remain_frozen(self):
        original = (PARENT / "decoder_ffn.py").read_text()
        changed = (STAGED / "decoder_ffn.py").read_text()
        check_original_functions(original, changed)
        for line in (STAGED / "frozen-inputs.sha256").read_text().splitlines():
            expected, name = line.split("  ", 1)
            self.assertEqual(hashlib.sha256((STAGED / name).read_bytes()).hexdigest(),
                             expected, name)

    def test_candidate_namespace_and_immediate_consumers_are_explicit(self):
        source = (STAGED / "decoder_ffn.py").read_text()
        self.assertIn("torch.ops.tilert_ffn_down_n1024", source)
        self.assertNotIn("torch.ops.tilert_ffn_down_n512", source)
        self.assertIn("passed = len(checks) == 11", source)
        self.assertIn("consumers[name].copy_(value)", source)
        self.assertIn("body[9], body[19], *consumers.values()", source)
        self.assertEqual(source.count("**check_down_tiling("), 2)
        ast.parse(source)

    def test_entire_record_rejects_parent_and_boundary_mutations(self):
        expected = json.loads(Path(__file__).with_name(
            "expected-tilings.json").read_text())
        record = expected["candidate"]["1"]
        tensor = torch.tensor(list(bytes.fromhex(record["raw_hex"])),
                              dtype=torch.uint8)
        self.assertTrue(check_down_tiling(torch, tensor, record)["pass"])
        for index in (0, 199):
            mutated = tensor.clone()
            mutated[index] ^= 1
            with self.assertRaisesRegex(RuntimeError, "host tiler"):
                check_down_tiling(torch, mutated, record)
        parent = torch.tensor(list(bytes.fromhex(expected["parent"]["1"]["raw_hex"])),
                              dtype=torch.uint8)
        with self.assertRaisesRegex(RuntimeError, "host tiler"):
            check_down_tiling(torch, parent, record)


if __name__ == "__main__":
    unittest.main()
