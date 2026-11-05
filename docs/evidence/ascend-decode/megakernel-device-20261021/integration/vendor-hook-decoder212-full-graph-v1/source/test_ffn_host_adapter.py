"""Check FFN buffer ownership and post-norm/MLP handoff without device arithmetic."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffn_host_adapter import LayerWorkspacePool, PendingFFN, Scratch


def scratch(rows=2):
    bridge = [torch.empty(rows, 6144, dtype=torch.bfloat16)
              for _ in range(12)]
    return Scratch(bridge, [], 17)


class PendingContracts(unittest.TestCase):
    def test_retained_and_normalized_keep_their_real_owners(self):
        owner = scratch()
        pending = PendingFFN()
        normalized, retained = pending.begin(owner, (0, 19, 2))
        self.assertIs(normalized, owner.bridge[0])
        self.assertIs(retained, owner.bridge[1])
        self.assertIs(pending.consume(normalized, (0, 19, 2)), owner)
        pending.require_empty()

    def test_equal_valued_clone_is_not_the_pending_normalized_owner(self):
        owner = scratch()
        pending = PendingFFN()
        normalized, _ = pending.begin(owner, (0, 19, 2))
        with self.assertRaises(RuntimeError):
            pending.consume(normalized.clone(), (0, 19, 2))
        self.assertIs(pending.consume(normalized, (0, 19, 2)), owner)

    def test_interleaved_post_norm_cannot_overwrite_pending_output(self):
        pending = PendingFFN()
        owner = scratch()
        normalized, _ = pending.begin(owner, (0, 19, 2))
        with self.assertRaises(RuntimeError):
            pending.begin(scratch(), (0, 19, 2))
        self.assertIs(pending.consume(normalized, (0, 19, 2)), owner)

    def test_mlp_cannot_consume_another_stream_or_consume_twice(self):
        pending = PendingFFN()
        owner = scratch()
        normalized, _ = pending.begin(owner, (0, 19, 2))
        with self.assertRaises(RuntimeError):
            pending.consume(normalized, (0, 23, 2))
        pending.consume(normalized, (0, 19, 2))
        with self.assertRaises(RuntimeError):
            pending.consume(normalized, (0, 19, 2))


class PoolContracts(unittest.TestCase):
    def setUp(self):
        self.prepared = []
        self.reference = torch.empty(2, 6144, dtype=torch.bfloat16)

    def prepare(self, value, blocks, layout):
        self.assertEqual((blocks, layout), (24, 1))
        output = scratch(value.shape[0])
        self.prepared.append(output)
        return output.bridge, output.body, output.sync

    def state(self, stream, capturing, rows=2):
        return patch("ffn_host_adapter.stream_state",
                     return_value=(0, stream, rows, capturing))

    def test_warmup_preserves_capture_reserve(self):
        pool = LayerWorkspacePool(object(), self.prepare)
        with self.state(7, False):
            reserved = pool.reserve(self.reference)
            warmed, _ = pool.get(self.reference)
        self.assertIsNot(warmed, reserved)
        with self.state(29, True):
            captured, key = pool.get(self.reference)
            repeated, _ = pool.get(self.reference)
        self.assertIs(captured, reserved)
        self.assertIs(repeated, reserved)
        self.assertEqual(pool.claims, [key])
        self.assertEqual(len(self.prepared), 2)

    def test_two_layers_do_not_alias_retained_residual_banks(self):
        pools = [LayerWorkspacePool(object(), self.prepare) for _ in range(2)]
        with self.state(7, False):
            for pool in pools:
                pool.reserve(self.reference)
        with self.state(29, True):
            banks = [pool.get(self.reference)[0] for pool in pools]
        addresses = [{value.data_ptr() for value in bank.bridge} for bank in banks]
        self.assertTrue(addresses[0].isdisjoint(addresses[1]))

    def test_capture_refuses_unreserved_exact_width(self):
        pool = LayerWorkspacePool(object(), self.prepare)
        with self.state(7, False):
            pool.reserve(self.reference)
        with self.state(29, True, rows=12):
            with self.assertRaises(RuntimeError):
                pool.get(torch.empty(12, 6144, dtype=torch.bfloat16))
        self.assertEqual(len(self.prepared), 1)

    def test_reservation_cannot_allocate_during_capture(self):
        pool = LayerWorkspacePool(object(), self.prepare)
        with self.state(29, True):
            with self.assertRaises(RuntimeError):
                pool.reserve(self.reference)
        self.assertFalse(self.prepared)


if __name__ == "__main__":
    unittest.main()
