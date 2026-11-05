"""Check raw output preservation before a failed device arithmetic gate."""

from pathlib import Path
import tempfile
import unittest

import torch

from kshift_check import retain_raw_outputs


class RawOutputs(unittest.TestCase):
    def test_nan_payload_and_all_guard_bytes_survive(self):
        acc = torch.empty(16, 160, dtype=torch.float32)
        projection = torch.empty(16, 160, dtype=torch.bfloat16)
        for value in (acc, projection):
            value.view(torch.uint8).fill_(0xa5)
            value[0].zero_()
        acc.view(torch.int32)[0, 17] = 0x7fc00123
        projection.view(torch.int16)[0, 25] = 0x7fc1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.pt"
            report = retain_raw_outputs(torch, acc, projection, path)
            saved = torch.load(path, weights_only=True)
        for name, expected in (("accumulators", acc), ("projection", projection)):
            self.assertTrue(torch.equal(saved[name].view(torch.uint8),
                                        expected.view(torch.uint8)))
            self.assertEqual(report["outputs"][name]["guard_different_bytes"], 0)
            self.assertEqual(tuple(saved[name].shape), (16, 160))
        self.assertEqual(report["outputs"]["accumulators"]
                         ["nonfinite_active_coordinates"], [[0, 17]])
        self.assertEqual(report["outputs"]["projection"]
                         ["nonfinite_active_coordinates"], [[0, 25]])

    def test_guard_damage_is_reported_without_discarding_output(self):
        acc = torch.empty(16, 160, dtype=torch.float32)
        projection = torch.empty(16, 160, dtype=torch.bfloat16)
        acc.view(torch.uint8).fill_(0xa5)
        projection.view(torch.uint8).fill_(0xa5)
        acc.view(torch.uint8)[15, 639] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.pt"
            report = retain_raw_outputs(torch, acc, projection, path)
            self.assertTrue(path.exists())
        self.assertEqual(report["outputs"]["accumulators"]
                         ["guard_different_bytes"], 1)
        self.assertEqual(report["outputs"]["projection"]
                         ["guard_different_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
