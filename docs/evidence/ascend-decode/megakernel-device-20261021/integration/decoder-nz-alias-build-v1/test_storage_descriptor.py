"""CPU checks of NZ carrier units, exact byte layout, and retained dump parsing."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_exception import fields, shape


def nz(value, columns_per_block):
    experts, rows, columns = value.shape
    return (value.reshape(experts, rows // 16, 16,
                          columns // columns_per_block, columns_per_block)
            .permute(0, 3, 1, 2, 4).contiguous())


class StorageContract(unittest.TestCase):
    def test_int8_nz_bytes_are_canonical_int32_nz_bytes(self):
        for inner, columns in ((16, 64), (128, 256), (32, 6144)):
            values = torch.arange(3 * inner * columns // 2, dtype=torch.int64)
            packed = (values % 256).to(torch.uint8).reshape(3, inner, columns // 2)
            old_bytes = nz(packed, 32)
            carrier = packed.view(torch.int32)
            canonical = nz(carrier, 8)
            self.assertTrue(torch.equal(old_bytes.flatten(),
                                        canonical.view(torch.uint8).flatten()))

    def test_real_gu_and_down_spans_use_carrier_units(self):
        for inner, columns, expected in ((6144, 256, 201326592),
                                         (128, 6144, 100663296)):
            physical = 256 * (columns // 64) * (inner // 16) * 16
            self.assertEqual(physical * 8 * 4, expected)
            self.assertEqual(physical * 64 // 2, expected)
            self.assertEqual(physical * 256 // 2, 4 * expected)
            self.assertEqual((columns // 64) * 256, 4 * columns)

    def test_shape_proto_supports_packed_and_unpacked_dimensions(self):
        self.assertEqual(shape(bytes.fromhex("0a0700028003801008")), [0, 2, 384, 2048, 8])
        self.assertEqual(shape(bytes.fromhex("08020803")), [2, 3])

    def test_truncated_proto_rejected(self):
        with self.assertRaises(ValueError):
            fields(b"\x0a\x05\x00")

if __name__ == "__main__":
    unittest.main()
