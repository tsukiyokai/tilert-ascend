"""CPU checks for diagnostic layouts and preservation of a failure witness."""

import unittest

import torch

from indexer_front_check import layout
from vendor_hook_index_diagnostic import decode_front, reproduction_checks
from vendor_hook_index_diagnostic import rope_reference


class IndexDiagnosticTests(unittest.TestCase):
    def test_accumulator_tiles_ignore_padding_and_preserve_column_order(self):
        for rows in (2, 32):
            for quantized in (False, True):
                capacity = 16 if rows <= 16 else 32
                front, size = layout(rows)
                prefix = 2 << 20
                offsets = {"capacity": capacity, "front_begin": prefix}
                offsets.update({"front_" + name: prefix + offset
                                for name, (offset, unused_size) in front.items()})
                cursor = 0
                for name, count, element_bytes in (
                        ("query_norm", rows * 2048, 1),
                        ("query_norm_scale", rows, 4),
                        ("query", rows * 4096, 1 if quantized else 2),
                        ("key", rows * 128, 1 if quantized else 2),
                        ("weights", rows * 32, 2),
                        ("query_scale", rows * 32, 2), ("key_scale", rows, 2)):
                    offsets[name] = cursor
                    cursor = (cursor + count * element_bytes + 511) // 512 * 512
                self.assertLess(cursor, prefix)
                raw = torch.zeros(prefix + size, dtype=torch.uint8)
                q = torch.arange(capacity * 4096, dtype=torch.int32).reshape(capacity, 4096)
                k = torch.arange(capacity * 160, dtype=torch.float32).reshape(capacity, 160)
                for name, value, tiles, columns in (
                        ("query_acc", q, 16, 256), ("key_acc", k, 5, 32)):
                    packed = value.reshape(capacity, tiles, columns).permute(1, 0, 2).contiguous()
                    offset, count = front[name]
                    raw[prefix + offset:prefix + offset + count].copy_(
                        packed.reshape(-1).view(torch.uint8))
                values = decode_front(torch, raw, offsets, rows, quantized)
                self.assertTrue(torch.equal(values["query_accumulators"], q[:rows]))
                self.assertTrue(torch.equal(values["key_accumulators"], k[:rows]))
                self.assertEqual(tuple(values["query_rope"].shape), (rows, 32, 128))

    def test_rope_interleaves_and_preserves_non_rope_bits(self):
        source = torch.arange(128, dtype=torch.bfloat16).reshape(1, 1, 128)
        source[0, 0, 100] = -0.0
        output = rope_reference(source, torch.zeros(1, 64, dtype=torch.bfloat16),
                                torch.ones(1, 64, dtype=torch.bfloat16))
        self.assertTrue(torch.equal(output[..., :64:2], -source[..., 1:64:2]))
        self.assertTrue(torch.equal(output[..., 1:64:2], source[..., :64:2]))
        self.assertTrue(torch.equal(output[..., 64:].view(torch.int16),
                                    source[..., 64:].view(torch.int16)))

    def test_reproduction_requires_exact_inputs_and_both_arms(self):
        def state():
            return {"hidden": torch.ones(1, 2, dtype=torch.bfloat16),
                    "metadata": {"slots": torch.tensor([1], dtype=torch.int32)},
                    "baseline": {"output": torch.tensor([0.0], dtype=torch.bfloat16)},
                    "native": {"output": torch.tensor([1.0], dtype=torch.bfloat16)}}

        parent, current = state(), state()
        self.assertTrue(reproduction_checks(torch, parent, current)["pass"])
        current["baseline"]["output"][0] = -0.0
        self.assertFalse(reproduction_checks(torch, parent, current)["pass"])
        current = state()
        current["metadata"]["slots"][0] = 2
        self.assertFalse(reproduction_checks(torch, parent, current)["pass"])


if __name__ == "__main__":
    unittest.main()
