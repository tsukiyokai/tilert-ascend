"""Check exact BF16 payloads and the probe's raw-byte comparison."""

import unittest

import torch

from check import compare, fixtures, poison, raw_bytes


class ProbeFixtures(unittest.TestCase):
    def test_special_payloads_survive_fixture_construction(self):
        cases = dict(fixtures(torch, 12))
        self.assertEqual(len(cases), 12)
        for value in cases.values():
            self.assertEqual(tuple(value.shape), (12, 6144))
        self.assertEqual(int(cases["negative_zero"].view(torch.int16)[0, 0]), -32768)
        self.assertEqual(int(cases["distinct_scale_witness"].view(torch.int16)[0, 0]),
                         0x3C2F)
        self.assertEqual(int(cases["signaling_nan_payloads"].view(torch.int16)[0, 0]),
                         0x7F81)

    def test_byte_metric_distinguishes_signed_zero_and_nan_payload(self):
        first = torch.tensor([0, 0x7FC1], dtype=torch.int16).view(torch.bfloat16)
        second = torch.tensor([-32768, 0x7FC2], dtype=torch.int16).view(torch.bfloat16)
        self.assertFalse(compare(torch, first, second)["bitwise_equal"])
        self.assertTrue(compare(torch, first, first.clone())["bitwise_equal"])
        self.assertEqual(raw_bytes(torch, first).numel(), 4)

    def test_two_poison_passes_reject_a_missing_write(self):
        values = {"codes": torch.empty(12, 6144, dtype=torch.int8),
                  "scale": torch.empty(12, 1, dtype=torch.float32)}
        poison(torch, values, 0)
        old = {key: value.clone() for key, value in values.items()}
        poison(torch, values, 1)
        self.assertTrue(all(not compare(torch, value, old[key])["bitwise_equal"]
                            for key, value in values.items()))


if __name__ == "__main__":
    unittest.main()
