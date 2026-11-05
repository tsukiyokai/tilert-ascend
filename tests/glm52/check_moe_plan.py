"""Check MoE packing and scalar combination using only the standard library."""

import argparse
from dataclasses import FrozenInstanceError
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import platform
import random
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.models.glm_5_2_ascend.moe_plan import (  # noqa: E402
    build_plan,
    combine,
    pack_routes,
    pack_tokens,
    restore_routes,
    stable_top8,
)


PARTIAL_ROUTES = (
    (255, 1, 11, 20, 31, 40, 50, 60),
    (1, 3, 11, 21, 31, 41, 51, 61),
    (255, 3, 12, 21, 32, 42, 50, 62),
    (0, 1, 12, 22, 31, 43, 52, 63),
)


def route_fixtures():
    random_source = random.Random(1759)
    for count in (1, 2, 4):
        yield f"shared.m{count}", ((255, 0, 9, 7, 5, 3, 1, 2),) * count
        yield f"disjoint.m{count}", tuple(
            tuple(range(token * 8 + 7, token * 8 - 1, -1))
            for token in range(count)
        )
        yield f"overlap.m{count}", PARTIAL_ROUTES[:count]
        for trial in range(32):
            yield f"random.m{count}.{trial}", tuple(
                tuple(random_source.sample(range(256), 8))
                for _ in range(count)
            )


def expert_output(token, expert, width):
    return tuple(
        (expert - 127) * (channel + 1) + (token + 1) * (-1) ** channel
        for channel in range(width)
    )


def scalar_combine(routes, weights, width):
    """Compute each original token/slot directly with exact rational sums."""
    result = []
    for token, experts in enumerate(routes):
        row = []
        for channel in range(width):
            total = Fraction(0)
            for slot, expert in enumerate(experts):
                value = expert_output(token, expert, width)[channel]
                total += Fraction(value) * Fraction(weights[token][slot])
            row.append(float(total))
        result.append(tuple(row))
    return tuple(result)


class MoePlanChecks(unittest.TestCase):
    def test_every_token_slot_occurs_once(self):
        for name, routes in route_fixtures():
            with self.subTest(name=name):
                plan = build_plan(routes)
                self.assertEqual(
                    sorted(zip(plan.token_ids, plan.slot_ids)),
                    [(token, slot) for token in range(len(routes))
                     for slot in range(8)],
                )

    def test_grouping_is_stable_and_expert_sorted(self):
        for name, routes in route_fixtures():
            with self.subTest(name=name):
                plan = build_plan(routes)
                expected = sorted(
                    (expert, token, slot)
                    for token, row in enumerate(routes)
                    for slot, expert in enumerate(row)
                )
                actual = []
                for group, expert in enumerate(plan.expert_ids):
                    for packed in range(*plan.group_offsets[group:group + 2]):
                        actual.append((expert, plan.token_ids[packed],
                                       plan.slot_ids[packed]))
                self.assertEqual(actual, expected)

    def test_inverse_is_a_permutation(self):
        for name, routes in route_fixtures():
            with self.subTest(name=name):
                plan = build_plan(routes)
                self.assertEqual(
                    sorted(plan.inverse), list(range(8 * len(routes)))
                )

    def test_inverse_preserves_original_slot_labels(self):
        plan = build_plan(PARTIAL_ROUTES)
        self.assertEqual(
            [(plan.token_ids[position], plan.slot_ids[position])
             for position in plan.inverse],
            [(token, slot) for token in range(4) for slot in range(8)],
        )

    def test_shared_experts_have_four_rows_per_group(self):
        plan = build_plan((tuple(range(8)),) * 4)
        self.assertEqual(
            (plan.expert_ids, plan.group_offsets, plan.group_sizes),
            (tuple(range(8)), tuple(range(0, 33, 4)), (4,) * 8),
        )

    def test_disjoint_experts_have_one_row_per_group(self):
        routes = tuple(tuple(range(i * 8, i * 8 + 8)) for i in range(4))
        plan = build_plan(routes)
        self.assertEqual(
            (plan.expert_ids, plan.group_offsets, plan.group_sizes),
            (tuple(range(32)), tuple(range(33)), (1,) * 32),
        )

    def test_route_payload_roundtrip_is_identity(self):
        for name, routes in route_fixtures():
            with self.subTest(name=name):
                plan = build_plan(routes)
                values = tuple(
                    tuple((token, slot, object()) for slot in range(8))
                    for token in range(len(routes))
                )
                self.assertEqual(
                    restore_routes(plan, pack_routes(plan, values)), values
                )

    def test_restoring_then_packing_is_identity(self):
        plan = build_plan(PARTIAL_ROUTES)
        values = tuple(object() for _ in range(32))
        self.assertEqual(
            pack_routes(plan, restore_routes(plan, values)), values
        )

    def test_a8_rows_and_scales_keep_their_token_owner(self):
        plan = build_plan(PARTIAL_ROUTES)
        scales = (1 / 128, 2.0, 8.0, 1 / 1024)
        for width in (128, 6144):
            with self.subTest(width=width):
                rows = tuple(
                    tuple((13 * token + channel) % 255 - 127
                          for channel in range(width))
                    for token in range(4)
                )
                self.assertEqual(
                    tuple(zip(pack_tokens(plan, rows),
                              pack_tokens(plan, scales))),
                    tuple((rows[token], scales[token])
                          for token in plan.token_ids),
                )

    def test_token_payloads_retain_identity(self):
        plan = build_plan(PARTIAL_ROUTES)
        values = [object() for _ in range(4)]
        self.assertTrue(all(
            value is values[token]
            for value, token in zip(pack_tokens(plan, values), plan.token_ids)
        ))

    def test_weighted_combine_matches_independent_scalar_oracle(self):
        for name, routes in route_fixtures():
            for width in (1, 7):
                with self.subTest(name=name, width=width):
                    plan = build_plan(routes)
                    weights = tuple(
                        tuple((token + slot) / 64 for slot in range(8))
                        for token in range(len(routes))
                    )
                    outputs = tuple(
                        expert_output(token, routes[token][slot], width)
                        for token, slot in zip(plan.token_ids, plan.slot_ids)
                    )
                    self.assertEqual(
                        combine(plan, outputs, weights),
                        scalar_combine(routes, weights, width),
                    )

    def test_combine_does_not_normalize_weights(self):
        plan = build_plan((tuple(range(8)),))
        self.assertEqual(combine(plan, ((2,),) * 8, ((2,) * 8,)), ((32.0,),))

    def test_plan_leaves_route_input_unchanged(self):
        routes = [list(row) for row in PARTIAL_ROUTES]
        expected = [row[:] for row in routes]
        build_plan(routes)
        self.assertEqual(routes, expected)

    def test_plan_is_frozen(self):
        plan = build_plan(PARTIAL_ROUTES)
        with self.assertRaises(FrozenInstanceError):
            plan.num_tokens = 1

    def test_selection_ties_use_ascending_expert_id(self):
        self.assertEqual(
            stable_top8(((0,) * 256,) * 4), (tuple(range(8)),) * 4
        )

    def test_selection_respects_strict_scores_before_ties(self):
        scores = [0] * 256
        for expert in (255, 64, 7, 4):
            scores[expert] = 2
        self.assertEqual(
            stable_top8((scores,)), ((4, 7, 64, 255, 0, 1, 2, 3),)
        )

    def test_selection_is_independent_for_each_token(self):
        scores = (tuple(range(256)), tuple(-i for i in range(256)))
        self.assertEqual(
            stable_top8(scores),
            (tuple(range(255, 247, -1)), tuple(range(8))),
        )

    def test_invalid_token_counts_are_rejected(self):
        for count in (0, 3, 5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                build_plan((tuple(range(8)),) * count)

    def test_incorrect_topk_width_is_rejected(self):
        for count in (0, 7, 9):
            with self.subTest(count=count), self.assertRaises(ValueError):
                build_plan((tuple(range(count)),))

    def test_duplicate_expert_within_a_token_is_rejected(self):
        for token in range(4):
            rows = [list(row) for row in PARTIAL_ROUTES]
            rows[token][-1] = rows[token][0]
            with self.subTest(token=token), self.assertRaises(ValueError):
                build_plan(rows)

    def test_invalid_expert_ids_are_rejected(self):
        for value in (-1, 256, True, 1.0, "7", None):
            row = list(range(8))
            row[-1] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_plan((row,))

    def test_invalid_score_rows_are_rejected(self):
        invalid = ((), ((0,) * 255,), ((0,) * 257,), ((0,) * 256,) * 3)
        for rows in invalid:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                stable_top8(rows)

    def test_nonfinite_or_nonnumeric_scores_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf"), "1", True):
            row = [0] * 256
            row[-1] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                stable_top8((row,))

    def test_payload_shape_mismatches_are_rejected(self):
        plan = build_plan(PARTIAL_ROUTES)
        cases = (
            lambda: pack_tokens(plan, (1, 2)),
            lambda: pack_routes(plan, ((1,) * 8,)),
            lambda: pack_routes(plan, ((1,) * 7,) * 4),
            lambda: restore_routes(plan, (1,) * 31),
            lambda: combine(plan, ((1,),) * 31, ((1,) * 8,) * 4),
            lambda: combine(plan, ((1,),) * 32, ((1,) * 7,) * 4),
            lambda: combine(plan, ((),) * 32, ((1,) * 8,) * 4),
            lambda: combine(plan, ((1,),) * 31 + ((1, 2),), ((1,) * 8,) * 4),
        )
        for case, operation in enumerate(cases):
            with self.subTest(case=case), self.assertRaises(ValueError):
                operation()

    def test_nonfinite_combine_inputs_are_rejected(self):
        plan = build_plan((tuple(range(8)),))
        for value in (float("nan"), float("inf"), "1", True):
            with self.subTest(output=value), self.assertRaises(ValueError):
                combine(plan, ((value,),) * 8, ((1,) * 8,))
            with self.subTest(weight=value), self.assertRaises(ValueError):
                combine(plan, ((1,),) * 8, ((value,) * 8,))

    def test_import_does_not_load_torch(self):
        self.assertFalse(any(
            name == "torch" or name.startswith("torch.")
            for name in sys.modules
        ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(MoePlanChecks)
    )
    print(stream.getvalue(), end="")
    files = (
        Path(__file__),
        ROOT / "tilert/models/glm_5_2_ascend/moe_plan.py",
        ROOT / "tilert/__init__.py",
        ROOT / "tilert/tilert_init.py",
        ROOT / "tilert/models/glm_5_2_ascend/__init__.py",
    )
    report = {
        "status": "pass" if result.wasSuccessful() else "fail",
        "scope": "stdlib route packing and scalar arithmetic; no device run",
        "hostname": platform.node(), "python": platform.python_version(),
        "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "route_fixtures": sum(1 for _ in route_fixtures()),
        "token_counts": [1, 2, 4], "num_experts": 256, "top_k": 8,
        "a8_row_widths": [128, 6144],
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in files
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(not result.wasSuccessful())


if __name__ == "__main__":
    raise SystemExit(main())
