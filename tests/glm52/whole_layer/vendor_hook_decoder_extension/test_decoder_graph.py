"""CPU checks for replay witnesses and exact local Decoder boundary checks."""

from pathlib import Path
import sys
import unittest

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from decoder_graph import DecoderObservation, validate_observation
from vendor_hook_forward import page_plan
from vendor_hook_graph import copy_fields, signature, validate_page_plan


def boundary_fixture():
    boundaries = [torch.full((2, 6144), value, dtype=torch.bfloat16)
                  for value in (0.0, 3.0)]
    values = {}
    selected = torch.arange(2048, dtype=torch.int32).repeat(32, 1)
    for number, boundary in zip((6, 7), boundaries):
        prefix = f"layer{number}"
        values[prefix + ".post_ar1"] = boundary.clone()
        values[prefix + ".output"] = torch.full_like(boundary, number)
        values[prefix + ".pre_ar2"] = values[prefix + ".output"].clone()
        values[prefix + ".residual"] = torch.full_like(boundary, -number)
        values[prefix + ".post_norm.1"] = values[prefix + ".residual"].clone()
        values[prefix + ".topk"] = selected.clone()
    result = values["layer7.output"].clone(), values["layer7.residual"].clone()
    return values, boundaries, result


class DecoderGraphContracts(unittest.TestCase):
    def test_exact_boundary_gate_rejects_stale_cut_and_residual(self):
        validate_observation(torch, *boundary_fixture())
        for name in ("layer6.post_ar1", "layer6.pre_ar2", "layer7.post_norm.1"):
            with self.subTest(name=name):
                values, boundaries, result = boundary_fixture()
                values[name][0, 0] = 19
                with self.assertRaises(RuntimeError):
                    validate_observation(torch, values, boundaries, result)

    def test_gate_keeps_signed_zero_and_ordered_topk(self):
        for name in ("layer6.post_ar1", "layer7.topk"):
            with self.subTest(name=name):
                values, boundaries, result = boundary_fixture()
                if values[name].is_floating_point():
                    values[name][0, 0] = -0.0
                else:
                    values[name][0, :2] = values[name][0, :2].flip(0)
                with self.assertRaises(RuntimeError):
                    validate_observation(torch, values, boundaries, result)

    def test_empty_replay_cannot_reuse_valid_observation(self):
        values, boundaries, result = boundary_fixture()
        observation = DecoderObservation(torch, (), None, None, ())
        observation.values = values
        pointers = {name: value.data_ptr() for name, value in values.items()}
        observation.poison()
        self.assertEqual(pointers, {name: value.data_ptr()
                                    for name, value in values.items()})
        with self.assertRaises(RuntimeError):
            validate_observation(torch, values, boundaries, result)

    def test_actual_lengths_and_independent_maps_change_at_fixed_geometry(self):
        for rows in (1, 2, 12, 32):
            plans = [page_plan(torch, rows, iteration, 8192, 66, 66)
                     for iteration in range(8)]
            lengths = set()
            for plan in plans:
                validate_page_plan(torch, plan, rows)
                lengths.add(int(plan["lengths"][0]))
                self.assertFalse(torch.equal(plan["main_table"], plan["index_table"]))
            self.assertEqual(len(lengths), 8)
            destinations = {name: value.clone() for name, value in plans[0].items()}
            addresses = {name: signature(value) for name, value in destinations.items()}
            for plan in plans[1:]:
                copy_fields(destinations, plan)
                self.assertEqual(addresses, {name: signature(value)
                                            for name, value in destinations.items()})
                for name in plan:
                    self.assertTrue(torch.equal(destinations[name], plan[name]))
            changed = {**plans[-1], "positions": torch.arange(rows + 1)}
            with self.assertRaises(ValueError):
                copy_fields(destinations, changed)


if __name__ == "__main__":
    unittest.main()
