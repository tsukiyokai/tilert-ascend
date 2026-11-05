"""Reject a false hot-expert gate and preserve the original router call."""

import ast
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

try:
    from hot_routing import (check_bias_owner, configure_hot_bias,
                             expected_routes, observe_router, poison_routes)
except ModuleNotFoundError:
    from routing import (check_bias_owner, configure_hot_bias, expected_routes,
                         observe_router, poison_routes)


def raw(value):
    return value.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8)


def case(rows):
    ids = torch.stack([torch.arange(8).roll(row % 8) for row in range(rows)])
    probabilities = torch.arange(1, 9).float().expand(rows, 8).clone() / 36
    return {"route_ids": ids, "route_probabilities": probabilities}


class Router:
    routed_scaling_factor = 1.0

    def __init__(self, bias):
        self.e_score_correction_bias = bias
        self.calls = 0
        self.result = (torch.ones(1, 8), torch.arange(8).reshape(1, 8))

    def _select_experts(self, *args, **kwargs):
        self.calls += 1
        return self.result


def layer_with_router():
    bias = torch.zeros(256)
    router = Router(bias)
    layer = SimpleNamespace(mlp=SimpleNamespace(
        gate=SimpleNamespace(e_score_correction_bias=bias),
        experts=SimpleNamespace(routed_experts=SimpleNamespace(router=router))))
    return layer, router


class HotGateContract(unittest.TestCase):
    def test_c17_and_c32_force_two_m_tiles_and_full_metadata(self):
        for rows in (17, 32):
            metadata, summary = expected_routes(torch, case(rows), rows)
            self.assertTrue(summary["pass"])
            self.assertEqual(summary["maximum_count"], rows)
            self.assertEqual(summary["down_m_tiles_per_expert"], 2)
            counts = torch.zeros(256, dtype=torch.int32)
            counts[:8] = rows
            active = torch.zeros(264, dtype=torch.int32)
            active[0] = 8
            active[8:16] = torch.arange(8)
            self.assertTrue(torch.equal(raw(metadata["route_counts"]), raw(counts)))
            self.assertTrue(torch.equal(raw(metadata["active_experts"]), raw(active)))

    def test_duplicate_or_nonhot_ids_cannot_claim_boundary_coverage(self):
        for mutation in ("duplicate", "other"):
            values = case(17)
            values["route_ids"][0, 0] = 1 if mutation == "duplicate" else 8
            unused_metadata, summary = expected_routes(torch, values, 17)
            self.assertFalse(summary["pass"])

    def test_bias_uses_shared_owner_and_separates_sigmoid_scores(self):
        layer, router = layer_with_router()
        identity = configure_hot_bias(torch, layer)
        check_bias_owner(torch, layer, identity["address"])
        logits = torch.linspace(-1000, 1000, 256).flip(0)
        scores = logits.sigmoid() + router.e_score_correction_bias
        self.assertTrue(torch.equal(scores.topk(8).indices.sort().values,
                                    torch.arange(8)))
        router.e_score_correction_bias = router.e_score_correction_bias.clone()
        with self.assertRaisesRegex(RuntimeError, "owner"):
            check_bias_owner(torch, layer, identity["address"])

    def test_router_observer_returns_same_objects_and_restores_method(self):
        layer, router = layer_with_router()
        with observe_router(torch, layer) as values:
            result = router._select_experts("sentinel")
            self.assertIs(result, router.result)
            result[0].zero_()
            self.assertTrue(bool((values["route_probabilities"] == 1).all()))
        self.assertEqual(router.calls, 1)
        self.assertNotIn("_select_experts", vars(router))
        with self.assertRaisesRegex(RuntimeError, "one original"):
            with observe_router(torch, layer):
                router._select_experts()
                router._select_experts()
        self.assertNotIn("_select_experts", vars(router))

    def test_poison_rejects_missing_active_padding_writes(self):
        bridge = [None] * 12
        body = [None] * 21
        bridge[4] = torch.empty(17, 8)
        bridge[5] = torch.empty(17, 8, dtype=torch.int32)
        body[12] = torch.empty(256, dtype=torch.int32)
        body[20] = torch.empty(264, dtype=torch.int32)
        poison_routes(torch, bridge, body)
        metadata, unused_summary = expected_routes(torch, case(17), 17)
        body[20][0] = 8
        body[20][8:16] = metadata["active_experts"][8:16]
        self.assertFalse(torch.equal(raw(body[20]), raw(metadata["active_experts"])))
        self.assertTrue(bool(torch.isnan(bridge[4]).all()))
        self.assertTrue(bool((body[12] == 0).all()))

    def test_frozen_gate_preserves_original_math_and_checks_all_outputs(self):
        staged = Path(sys.argv[0]).resolve().parent
        if not (staged / "decoder_ffn.py").exists():
            staged = Path("/tmp/tilert-megakernel-validation-20261021/") / (
                "vendor-hook-decoder212-ffn-down-n1024-hot-v1")
        source = (staged / "decoder_ffn.py").read_text()
        functions = {node.name: ast.dump(node, include_attributes=False)
                     for node in ast.parse(source).body
                     if isinstance(node, ast.FunctionDef)}
        original = (staged / "parent-decoder_ffn.py").read_text()
        parent_functions = {node.name: ast.dump(node, include_attributes=False)
                            for node in ast.parse(original).body
                            if isinstance(node, ast.FunctionDef)}
        for name in ("unpack_scale", "shared_weights", "clone_output",
                     "observe_original", "check_down_tiling"):
            self.assertEqual(functions[name], parent_functions[name], name)
        for token in ("len(checks) == 15", 'hot_case["pass"]',
                      '"route_counts": body[12]', '"active_experts": body[20]',
                      "consumers[name].copy_(value)", "args.native_blocks"):
            self.assertIn(token, source)
        for line in (staged / "frozen-inputs.sha256").read_text().splitlines():
            expected, name = line.split("  ", 1)
            self.assertEqual(hashlib.sha256((staged / name).read_bytes()).hexdigest(),
                             expected, name)


if __name__ == "__main__":
    unittest.main()
