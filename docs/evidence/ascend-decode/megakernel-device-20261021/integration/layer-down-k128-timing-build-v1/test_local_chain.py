"""CPU contracts for live local-chain producers and defined observations."""

import ast
from pathlib import Path
import unittest

import torch

import local_chain
from local_chain_views import attention_values, poison, strict_checks, tensor_view


def byte_metric(unused_torch, actual, expected):
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    same = (actual.dtype == expected.dtype and actual.shape == expected.shape
            and torch.equal(actual.contiguous().view(torch.uint8),
                            expected.contiguous().view(torch.uint8)))
    return {"pass": finite and same, "bitwise_equal": same}


class LocalChainContracts(unittest.TestCase):
    def test_attention_packet_preserves_producers_and_unused_slots(self):
        groups = [[object() for _ in range(size)] for size in (8, 9, 6, 3, 6, 6, 4)]
        norm, main, front, tail, metadata, index, caches = groups
        work, tiling, output = object(), object(), object()
        full = local_chain.attention_addresses(*groups, work, tiling, output)
        expected = [norm[3], norm[6], norm[7], *main, *front, *tail, None,
                    *metadata, *index, *caches, work, tiling, output]
        self.assertEqual(full, expected)
        shared = local_chain.attention_addresses(
            norm, main, [], tail, metadata, None, caches[:2], work, tiling, output)
        self.assertEqual(len(shared), 41)
        self.assertEqual(shared[12:18], [None] * 6)
        self.assertEqual(shared[28:34], [None] * 6)
        self.assertEqual(shared[36:38], [None] * 2)

    def test_successor_retains_two_original_owners(self):
        bridge = [torch.empty(1) for _ in range(12)]
        hidden, residual = local_chain.successor_inputs(bridge)
        self.assertIs(hidden, bridge[10])
        self.assertIs(residual, bridge[1])
        bridge[1] = bridge[10]
        with self.assertRaisesRegex(ValueError, "distinct"):
            local_chain.successor_inputs(bridge)

    def test_topk_refresh_preserves_owner_and_unused_rows(self):
        owner = torch.full((32, 2048), -1, dtype=torch.int32)
        pointer = owner.data_ptr()
        for iteration in range(3):
            producer = torch.full((12, 1, 2048), iteration, dtype=torch.int32)
            local_chain.publish_topk(producer, owner, 12)
            self.assertTrue(torch.equal(owner[:12], producer.view(12, 2048)))
        self.assertEqual(owner.data_ptr(), pointer)
        self.assertTrue(bool((owner[12:] == -1).all()))
        with self.assertRaises(ValueError):
            local_chain.publish_topk(producer, owner, 2)

    def test_shared_views_never_expose_the_dead_prolog_arena(self):
        for rows in (1, 2, 12, 32):
            scratch = torch.full((2 << 20,), 37, dtype=torch.uint8)
            values = attention_values(torch, scratch, {}, rows, "shared")
            self.assertEqual(set(values), {"latent", "tail_values", "tail_quantized",
                                           "tail_scales", "tail_accumulators"})
            self.assertEqual(values["tail_values"].shape, (16, rows, 64))
            self.assertEqual(values["tail_accumulators"].shape, (24, rows, 256))
            poison(values)
            # The last byte lies outside every observed region.
            self.assertEqual(int(scratch[-1]), 37)

    def test_view_rejects_out_of_owner_access(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            tensor_view(torch, torch.empty(16, dtype=torch.uint8), 8,
                        (4,), torch.float32)

    def test_poison_rejects_empty_or_partial_execution(self):
        actual = {"normalized": torch.zeros(12, 6144),
                  "bridge.ids": torch.zeros(12, 8, dtype=torch.int32),
                  "body.active_list": torch.zeros(264, dtype=torch.int32)}
        expected = {name: value.clone() for name, value in actual.items()}
        poison(actual)
        self.assertFalse(strict_checks(torch, actual, expected, byte_metric)[1])
        actual["normalized"][:6].zero_()
        self.assertFalse(strict_checks(torch, actual, expected, byte_metric)[1])
        actual["normalized"].zero_()
        self.assertTrue(strict_checks(torch, actual, expected, byte_metric)[1])

    def test_byte_gate_distinguishes_signed_zero_and_shape(self):
        self.assertFalse(strict_checks(
            torch, {"x": torch.tensor([-0.0])}, {"x": torch.tensor([0.0])},
            byte_metric)[1])
        with self.assertRaisesRegex(RuntimeError, "boundary sets"):
            strict_checks(torch, {}, {"x": torch.ones(1)}, byte_metric)

    def test_distinct_integer_poison_rejects_unwritten_codes(self):
        arms = [{"codes": torch.zeros(12, 6144, dtype=torch.int8),
                 "bridge.ids": torch.empty(12, 8, dtype=torch.int32)}
                for _ in range(2)]
        for ordinal, values in enumerate(arms, 1):
            poison(values, ordinal)
            self.assertTrue(bool((values["bridge.ids"] == 0).all()))
        self.assertFalse(strict_checks(torch, *arms, byte_metric)[1])

    def test_actual_metadata_updates_int64_slots_in_place(self):
        from vendor_hook_forward import page_plan
        from vendor_hook_graph import copy_fields, validate_page_plan

        for rows in (1, 2, 12, 32):
            first = page_plan(torch, rows, 0, 8192, 66, 66)
            destinations = {name: value.clone() for name, value in first.items()}
            slots = [torch.empty(rows, dtype=torch.int64) for _ in range(2)]
            pointers = [value.data_ptr() for value in slots]
            for iteration in range(8):
                plan = page_plan(torch, rows, iteration, 8192, 66, 66)
                validate_page_plan(torch, plan, rows)
                copy_fields(destinations, plan)
                for destination, key in zip(slots, ("main_slots", "index_slots")):
                    destination.copy_(destinations[key])
                    self.assertTrue(torch.equal(destination, plan[key].long()))
            self.assertEqual(pointers, [value.data_ptr() for value in slots])
            self.assertFalse(torch.equal(slots[0], slots[1]))

    def test_reference_calls_parents_and_no_external_ar1(self):
        source = Path(local_chain.__file__).read_text()
        tree = ast.parse(source)
        cls = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "LayerCase")
        separate = next(node for node in cls.body
                        if isinstance(node, ast.FunctionDef) and node.name == "separate")
        calls = [ast.unparse(node.func) for node in ast.walk(separate)
                 if isinstance(node, ast.Call)]
        for name in ("self.input_norm", "self.torch_npu.npu_dynamic_quant",
                     "self.torch.ops.tilert.indexed_attention_out",
                     "self.torch.ops.tilert.shared_full_attention_out",
                     "self.ffn.native.forward_out"):
            self.assertIn(name, calls)
        self.assertNotIn("post_ar1", source)
        self.assertNotIn("all_reduce(", source)


if __name__ == "__main__":
    unittest.main()
