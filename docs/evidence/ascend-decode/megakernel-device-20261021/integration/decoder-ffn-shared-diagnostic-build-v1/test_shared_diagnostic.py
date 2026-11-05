"""Check observation restoration and the frozen shared-payload layout."""

from types import SimpleNamespace
import unittest

import torch

from decoder_ffn_diagnostic import replace_attribute, shared_payload


class SharedDiagnosticTests(unittest.TestCase):
    def test_replacement_restores_after_failure(self):
        original = object()
        replacement = object()
        owner = SimpleNamespace(operator=original)
        with self.assertRaisesRegex(RuntimeError, "inner failure"):
            with replace_attribute(owner, "operator", replacement):
                self.assertIs(owner.operator, replacement)
                raise RuntimeError("inner failure")
        self.assertIs(owner.operator, original)

    def test_payload_uses_token_order_and_sum_of_six_k_parts(self):
        generator = torch.Generator().manual_seed(41213)
        parts = torch.randint(-100, 101, (6, 4, 32, 64), generator=generator,
                              dtype=torch.int32)
        middle = torch.randint(-127, 128, (32, 128), generator=generator,
                               dtype=torch.int8)
        scales = torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8)
        down = torch.arange(24 * 32 * 256, dtype=torch.int32).reshape(24, 32, 256)
        payload = torch.cat((torch.zeros(32 * 6144, dtype=torch.uint8),
                             parts.view(torch.uint8).reshape(-1),
                             middle.view(torch.uint8).reshape(-1),
                             scales.view(torch.uint8).reshape(-1),
                             down.view(torch.uint8).reshape(-1)))
        bridge = [None] * 12
        body = [None] * 33
        body[17] = payload
        actual = shared_payload(torch, bridge, body)
        self.assertTrue(torch.equal(actual["middle"], middle))
        self.assertTrue(torch.equal(actual["middle_scale"], scales[:, 0]))
        for row in (0, 20, 31):
            for column in (0, 63, 64, 127, 128, 255):
                expected = sum(int(parts[k, column // 64, row, column % 64])
                               for k in range(6))
                self.assertEqual(int(actual["gate_int32"][row, column]), expected)
            for column in (0, 255, 256, 2893, 5220, 6143):
                expected = down[column // 256, row, column % 256]
                self.assertEqual(int(actual["down_int32"][row, column]), int(expected))


if __name__ == "__main__":
    unittest.main()
