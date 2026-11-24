"""CPU contracts for persistent graph producers, not device graph acceptance."""

import unittest

import torch

from vendor_hook_forward import page_plan
from vendor_hook_graph import copy_fields, signature, validate_page_plan


class PersistentMetadataTests(unittest.TestCase):
    def test_changing_page_plans_keep_geometry_and_independent_mapping(self):
        for rows in (1, 2, 12, 32):
            prior = None
            for iteration in range(8):
                plan = page_plan(torch, rows, iteration, 8192, 66, 66)
                validate_page_plan(torch, plan, rows)
                self.assertFalse(torch.equal(plan["main_slots"], plan["index_slots"]))
                if prior is not None:
                    self.assertNotEqual(plan["lengths"].item(), prior["lengths"].item())
                    self.assertFalse(torch.equal(plan["main_slots"], prior["main_slots"]))
                    self.assertFalse(torch.equal(plan["index_slots"], prior["index_slots"]))
                prior = plan

    def test_stale_slot_mapping_is_rejected(self):
        plan = page_plan(torch, 12, 1, 8192, 66, 66)
        previous = page_plan(torch, 12, 0, 8192, 66, 66)
        plan["main_slots"] = previous["main_slots"]
        with self.assertRaisesRegex(ValueError, "main slots"):
            validate_page_plan(torch, plan, 12)

    def test_live_update_preserves_storage_and_signed_zero(self):
        destinations = {"slots": torch.empty(12, dtype=torch.int32),
                        "rope": torch.empty(12, 64, dtype=torch.bfloat16)}
        pointers = {key: signature(value) for key, value in destinations.items()}
        slots_i64 = torch.empty(12, dtype=torch.int64)
        for iteration in range(4):
            sources = {"slots": torch.arange(12, dtype=torch.int32) + iteration * 128,
                       "rope": torch.full((12, 64), iteration / 8, dtype=torch.bfloat16)}
            sources["rope"][0, 0] = -0.0
            copy_fields(destinations, sources)
            slots_i64.copy_(destinations["slots"])
            self.assertEqual(pointers, {key: signature(value)
                                        for key, value in destinations.items()})
            self.assertTrue(torch.equal(slots_i64, sources["slots"].long()))
            self.assertTrue(torch.equal(destinations["rope"].view(torch.int16),
                                        sources["rope"].view(torch.int16)))

    def test_shape_change_is_rejected_before_any_producer_is_written(self):
        destinations = {"length": torch.tensor([8192], dtype=torch.int32),
                        "table": torch.zeros(1, 64, dtype=torch.int32)}
        sources = {"length": torch.tensor([8064], dtype=torch.int32),
                   "table": torch.ones(1, 63, dtype=torch.int32)}
        with self.assertRaisesRegex(ValueError, "shape or dtype"):
            copy_fields(destinations, sources)
        self.assertEqual(destinations["length"].item(), 8192)
        self.assertEqual(torch.count_nonzero(destinations["table"]), 0)


if __name__ == "__main__":
    unittest.main()
