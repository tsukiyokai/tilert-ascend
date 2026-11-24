"""Check the exact packed-scale representation used at the FFN boundary."""

import unittest

import numpy as np
import torch

from decoder_ffn import unpack_scale


class ScaleTests(unittest.TestCase):
    def test_round_trip_all_finite_bit_patterns(self):
        generator = np.random.default_rng(73023)
        bits = generator.integers(0, 1 << 32, (256, 256), dtype=np.uint32)
        bits[(bits & 0x7f800000) == 0x7f800000] = 0
        bits[0, :8] = [0, 0x80000000, 1, 0x80000001,
                       0x7f7fffff, 0xff7fffff, 0x3f800000, 0xbf800000]
        packed = torch.from_numpy(bits.astype(np.int64))
        before = packed.clone()
        for carrier in (packed, packed[:, None, :]):
            output = unpack_scale(torch, carrier, 256)
            np.testing.assert_array_equal(output.numpy().view(np.uint32), bits)
        self.assertTrue(torch.equal(before, packed))

    def test_nonzero_high_word_is_rejected(self):
        packed = torch.zeros((256, 256), dtype=torch.int64)
        packed[3, 7] = 1 << 32
        with self.assertRaises(ValueError):
            unpack_scale(torch, packed, 256)

    def test_nonfinite_scale_is_rejected(self):
        packed = torch.zeros((256, 256), dtype=torch.int64)
        packed[4, 9] = 0x7f800000
        with self.assertRaises(ValueError):
            unpack_scale(torch, packed, 256)

    def test_float_carrier_is_rejected(self):
        with self.assertRaises(ValueError):
            unpack_scale(torch, torch.zeros((256, 256)), 256)


if __name__ == "__main__":
    unittest.main()
