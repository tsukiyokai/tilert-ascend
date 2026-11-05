"""Check the packing probe's byte-level oracle and persistent output contract."""

import unittest

import torch

import pack_check


class PackingProbeContract(unittest.TestCase):
    def test_direct_out_preserves_bf16_segments_and_storage(self):
        latent, rope = pack_check.fixture(torch, 2, 20261021)
        packed = torch.empty(2, 128, 576, dtype=torch.bfloat16)
        pointer = packed.data_ptr()
        result = pack_check.pack_into(torch, latent, rope, packed)
        self.assertEqual(result.data_ptr(), pointer)
        self.assertTrue(torch.equal(packed[..., :512].view(torch.int16),
                                    latent.squeeze(2).view(torch.int16)))
        self.assertTrue(torch.equal(packed[..., 512:].view(torch.int16),
                                    rope.squeeze(2).view(torch.int16)))
        self.assertEqual(packed[0, 0, :5].view(torch.int16).tolist(),
                         [0, -32768, 32640, -128, 32705])

    def test_raw_gate_rejects_signed_zero_and_nan_payload_changes(self):
        words = torch.tensor([0, -32768, 32705], dtype=torch.int16)
        expected = words.view(torch.bfloat16)
        same = expected.clone()
        self.assertTrue(pack_check.compare_bytes(torch, same, expected)["pass"])
        same.view(torch.int16)[1] = 0
        same.view(torch.int16)[2] = 32706
        result = pack_check.compare_bytes(torch, same, expected)
        self.assertFalse(result["pass"])
        self.assertEqual(result["bitwise_different_bytes"], 2)


if __name__ == "__main__":
    unittest.main()
