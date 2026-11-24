"""Check token transfers with CPU tensors and independent scalar targets."""

from itertools import product
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from check_mtp_round import make_executor  # noqa: E402
from check_round_invariance import (  # noqa: E402
    DRAFTERS, OUTPUT_TOKENS, PROMPT, RULES, WIDTHS, ScalarTarget, decode,
    serial_decode,
)
from tilert.models.glm_5_2_ascend.cann_model import (  # noqa: E402
    CannModel, VOCAB_SIZE, device_tokens,
)


class PredictionOnlyTarget(ScalarTarget):
    def host_tokens(self, *groups):
        if len(groups) != 1:
            raise AssertionError("Host candidates were sent back to device")
        return super().host_tokens(*groups)


class TokenStagingChecks(unittest.TestCase):
    def test_host_tokens_use_one_tensor_factory(self):
        for width in (1, *WIDTHS):
            values = list(range(width))
            with (
                self.subTest(width=width),
                patch.object(torch, "tensor", wraps=torch.tensor) as factory,
                patch.object(torch, "as_tensor",
                             wraps=torch.as_tensor) as scalar,
                patch.object(torch, "stack", wraps=torch.stack) as stack,
            ):
                result = device_tokens(values, "cpu")
                self.assertEqual(result.tolist(), values)
                self.assertEqual(result.dtype, torch.int64)
                self.assertEqual(factory.call_count, 1)
                self.assertEqual(scalar.call_count, 0)
                self.assertEqual(stack.call_count, 0)

    def test_mixed_device_scalars_keep_their_values_and_order(self):
        values = [3, torch.tensor(7, dtype=torch.int32),
                  torch.tensor(11, dtype=torch.int64), 17]
        self.assertEqual(device_tokens(values, "cpu").tolist(), [3, 7, 11, 17])
        self.assertEqual(device_tokens(23, "cpu").tolist(), [23])

    def test_existing_int64_tensor_keeps_its_storage(self):
        values = torch.tensor([[3, 7]], dtype=torch.int64)
        result = device_tokens(values, "cpu")
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.data_ptr(), values.data_ptr())
        self.assertEqual(device_tokens(values.int(), "cpu").tolist(), [3, 7])

    def test_invalid_host_values_fail_before_allocation(self):
        for value in (-1, VOCAB_SIZE, True, 1.5, "3"):
            for prefix in ([], [torch.tensor(1)]):
                with (
                    self.subTest(value=value, mixed=bool(prefix)),
                    patch.object(torch, "tensor") as factory,
                    patch.object(torch, "as_tensor") as scalar,
                ):
                    with self.assertRaises(ValueError):
                        device_tokens([*prefix, value], "cpu")
                    factory.assert_not_called()
                    scalar.assert_not_called()

    def test_one_prediction_group_needs_no_concatenation(self):
        model = SimpleNamespace(device="cpu")
        values = torch.tensor([3, 7, 11], dtype=torch.int64)
        with patch.object(torch, "cat", side_effect=AssertionError):
            self.assertEqual(CannModel.host_tokens(model, values),
                             ((3, 7, 11),))
        self.assertEqual(CannModel.host_tokens(model, [1, 2], values),
                         ((1, 2), (3, 7, 11)))

    def test_host_drafters_read_only_predictions_and_match_serial(self):
        for rule, name, width in product(RULES, DRAFTERS, WIDTHS):
            with self.subTest(rule=rule.__name__, drafter=name, width=width):
                output, _ = decode(rule, name, width, PredictionOnlyTarget)
                self.assertEqual(output,
                                 serial_decode(rule, PROMPT, OUTPUT_TOKENS))

    def test_model_drafter_still_reads_both_device_groups(self):
        executor, _, _ = make_executor(4, 128, False, "accept")
        original = executor.target.host_tokens
        with patch.object(executor.target, "host_tokens",
                          wraps=original) as read:
            executor.advance()
        self.assertEqual(len(read.call_args.args), 2)


if __name__ == "__main__":
    unittest.main()
