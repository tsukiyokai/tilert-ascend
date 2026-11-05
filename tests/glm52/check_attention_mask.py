"""CPU checks for selection-mask sharing and single-token fallback."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from tilert.models.glm_5_2_ascend import cann_attention as attention


def expected_mask(indices, positions, capacity):
    return torch.tensor([
        [column not in row.tolist() or column > position
         for column in range(capacity)]
        for row, position in zip(indices, positions.tolist())
    ])


class AttentionMaskChecks(unittest.TestCase):
    def test_selection_and_causal_boundaries(self):
        indices = torch.tensor([[0, 3, 6], [1, 4, 7]], dtype=torch.int32)
        columns = torch.arange(8).view(1, 8)
        for positions in ([3, 6], [2, 4], [3, 6]):
            positions = torch.tensor(positions)
            actual = attention.selection_mask(
                indices[:, None], columns > positions[:, None],
            )
            self.assertTrue(torch.equal(
                actual, expected_mask(indices, positions, 8),
            ))

    def test_producers_refresh_consumers_reuse_and_reset(self):
        shared = attention.SharedKVAttention(2, 8, "cpu")
        roles = (False, False, False, True, True, True)
        impls = [SimpleNamespace(
            skip_topk=role, scale=0.25,
            _execute_sparse_flash_attention_process=Mock(),
        ) for role in roles]
        for impl in impls:
            shared.bind(impl)
        positions = torch.empty(2, dtype=torch.int32)
        metadata = SimpleNamespace(slot_mapping=positions)
        snapshots = []
        with patch.object(
            attention, "shared_kv_fia",
            side_effect=lambda q, qr, kv, mask, scale: mask.clone(),
        ) as fia, patch.object(
            attention, "selection_mask", wraps=attention.selection_mask,
        ) as build:
            for shift in (0, 1, 0):
                positions.copy_(torch.tensor([5 - shift, 6 - shift]))
                outputs = []
                for layer, impl in enumerate(impls):
                    if not impl.skip_topk:
                        indices = (
                            torch.tensor([[0, 3, 6], [1, 4, 7]])
                            + layer + shift
                        ).remainder(8).int()
                    cache = (object(), object())
                    q = torch.full((2, 4, 512), float(layer))
                    qr = torch.full((2, 4, 64), float(layer + 1))
                    output = impl._execute_sparse_flash_attention_process(
                        q, qr, cache, indices.clone(), metadata, None, None,
                    )
                    self.assertIs(fia.call_args.args[0], q)
                    self.assertIs(fia.call_args.args[1], qr)
                    self.assertIs(fia.call_args.args[2], cache)
                    self.assertTrue(torch.equal(
                        output, expected_mask(indices, positions, 8),
                    ))
                    outputs.append(output)
                snapshots.append(outputs)
            self.assertEqual(build.call_count, 9)
        self.assertTrue(all(torch.equal(a, b)
                            for a, b in zip(snapshots[0], snapshots[2])))

    def test_m1_preserves_original_attention(self):
        for width in (8, 16, 32):
            with self.subTest(width=width):
                self.check_m1_fallback(width)

    def check_m1_fallback(self, width):
        original = Mock(return_value="original")
        impl = SimpleNamespace(
            skip_topk=False, scale=0.25,
            _execute_sparse_flash_attention_process=original,
        )
        shared = attention.SharedKVAttention(width, 8, "cpu")
        shared.bind(impl)
        output = impl._execute_sparse_flash_attention_process(
            torch.empty(1, 4, 512), None, None, None, None, None, None,
        )
        self.assertEqual(output, "original")
        self.assertIsNone(shared.mask)
        original.assert_called_once()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
