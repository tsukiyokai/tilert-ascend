"""Stable top-8 MoE packing for one request with 1, 2, or 4 token rows.

The plan groups equal experts across tokens while retaining every token's
slot order through an inverse map. Packing gathers whole token rows, so an
A8 activation row and its scale keep the same token owner. Numeric helpers
use Python floats; they do not emulate CANN or BF16 rounding.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Real
from operator import index
from typing import TypeVar


NUM_EXPERTS = 256
TOP_K = 8
TOKEN_COUNTS = (1, 2, 4)
Value = TypeVar("Value")


@dataclass(frozen=True)
class MoePlan:
    """Expert groups and inverse permutation produced by ``build_plan``.

    ``expert_ids`` is ascending and contains only selected experts. Group
    ``g`` occupies ``[group_offsets[g], group_offsets[g + 1])``. The final
    offset equals ``num_tokens * TOP_K``; ``group_offsets[1:]`` contains
    cumulative group ends, not group sizes.

    Packed row ``p`` belongs to ``(token_ids[p], slot_ids[p])``. Conversely,
    ``inverse[token * TOP_K + slot]`` gives that route's packed row. Within
    an expert, rows remain in token-major, then slot-major order.
    """

    num_tokens: int
    expert_ids: tuple[int, ...]
    group_offsets: tuple[int, ...]
    token_ids: tuple[int, ...]
    slot_ids: tuple[int, ...]
    inverse: tuple[int, ...]

    @property
    def num_routes(self) -> int:
        """Return the number of selected token/expert pairs."""
        return self.num_tokens * TOP_K

    @property
    def group_sizes(self) -> tuple[int, ...]:
        """Return the number of token rows assigned to each expert."""
        return tuple(
            right - left for left, right in zip(
                self.group_offsets, self.group_offsets[1:]
            )
        )


def build_plan(routes: Sequence[Sequence[int]]) -> MoePlan:
    """Group explicit top-8 IDs, preserving the caller's token/slot labels.

    Each token must select eight distinct experts in ``[0, 256)``. The same
    expert may occur in different tokens. Scores and routing weights do
    not participate in the permutation.
    """
    _check_token_count(len(routes))
    entries = []
    for token, row in enumerate(routes):
        if len(row) != TOP_K:
            raise ValueError("Each token must select eight experts")
        experts = tuple(_expert_id(value) for value in row)
        if len(set(experts)) != TOP_K:
            raise ValueError("A token cannot select an expert twice")
        entries.extend(
            (expert, token, slot) for slot, expert in enumerate(experts)
        )
    entries.sort(key=lambda entry: entry[0])
    expert_ids, group_offsets = [], [0]
    token_ids, slot_ids = [], []
    inverse = [0] * len(entries)
    for packed, (expert, token, slot) in enumerate(entries):
        if not expert_ids or expert != expert_ids[-1]:
            if expert_ids:
                group_offsets.append(packed)
            expert_ids.append(expert)
        token_ids.append(token)
        slot_ids.append(slot)
        inverse[token * TOP_K + slot] = packed
    group_offsets.append(len(entries))
    return MoePlan(
        len(routes), tuple(expert_ids), tuple(group_offsets),
        tuple(token_ids), tuple(slot_ids), tuple(inverse),
    )


def stable_top8(
    scores: Sequence[Sequence[Real]],
) -> tuple[tuple[int, ...], ...]:
    """Select by descending finite score, breaking ties by expert ID.

    This is a scalar selection oracle. Inputs are the final selection
    scores, after any router correction; probability normalization and
    expert output weighting remain separate operations.
    """
    _check_token_count(len(scores))
    routes = []
    for row in scores:
        if len(row) != NUM_EXPERTS:
            raise ValueError("Each token needs 256 expert scores")
        values = tuple(_finite_float(value) for value in row)
        experts = sorted(
            range(NUM_EXPERTS), key=lambda expert: (-values[expert], expert)
        )
        routes.append(tuple(experts[:TOP_K]))
    return tuple(routes)


def pack_tokens(plan: MoePlan, values: Sequence[Value]) -> tuple[Value, ...]:
    """Gather one payload per token into expert order without changing it.

    A payload may be a complete K-channel activation row or its A8 scale.
    Returned payloads retain their original identity; this helper neither
    mutates them nor joins channels from different tokens.
    """
    if len(values) != plan.num_tokens:
        raise ValueError("Token payload count differs from the plan")
    return tuple(values[token] for token in plan.token_ids)


def pack_routes(
    plan: MoePlan, values: Sequence[Sequence[Value]],
) -> tuple[Value, ...]:
    """Gather a payload for every ``[token][slot]`` into expert order."""
    _check_route_values(plan, values)
    return tuple(
        values[token][slot]
        for token, slot in zip(plan.token_ids, plan.slot_ids)
    )


def restore_routes(
    plan: MoePlan, packed: Sequence[Value],
) -> tuple[tuple[Value, ...], ...]:
    """Restore packed route payloads to ``[token][slot]`` without a sum."""
    if len(packed) != plan.num_routes:
        raise ValueError("Packed payload count differs from the plan")
    return tuple(
        tuple(packed[plan.inverse[token * TOP_K + slot]]
              for slot in range(TOP_K))
        for token in range(plan.num_tokens)
    )


def combine(
    plan: MoePlan,
    packed_outputs: Sequence[Sequence[Real]],
    route_weights: Sequence[Sequence[Real]],
) -> tuple[tuple[float, ...], ...]:
    """Weight expert outputs and sum the eight slots of each token.

    Outputs have shape ``[M * 8, H]`` in packed order and weights have shape
    ``[M, 8]`` in original slot order. Each weight is applied once without
    normalization. Python-float products are summed with ``math.fsum``;
    device dtype casts and TP reduction are outside this scalar oracle.
    """
    _check_route_values(plan, route_weights)
    if len(packed_outputs) != plan.num_routes:
        raise ValueError("Packed output count differs from the plan")
    width = len(packed_outputs[0])
    if width < 1 or any(len(row) != width for row in packed_outputs):
        raise ValueError("Expert output rows need one common positive width")
    outputs = tuple(tuple(_finite_float(value) for value in row)
                    for row in packed_outputs)
    weights = tuple(tuple(_finite_float(value) for value in row)
                    for row in route_weights)
    return tuple(
        tuple(math.fsum(
            outputs[plan.inverse[token * TOP_K + slot]][channel]
            * weights[token][slot]
            for slot in range(TOP_K)
        ) for channel in range(width))
        for token in range(plan.num_tokens)
    )


def _check_token_count(count: int) -> None:
    if count not in TOKEN_COUNTS:
        raise ValueError("One request must contain 1, 2, or 4 token rows")


def _expert_id(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("Expert IDs must be integers")
    try:
        expert = index(value)
    except TypeError as error:
        raise ValueError("Expert IDs must be integers") from error
    if not 0 <= expert < NUM_EXPERTS:
        raise ValueError("Expert IDs must lie in [0, 256)")
    return expert


def _check_route_values(plan: MoePlan, values: Sequence[Sequence]) -> None:
    if len(values) != plan.num_tokens or any(
        len(row) != TOP_K for row in values
    ):
        raise ValueError("Route payloads must have shape [M, 8]")


def _finite_float(value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Scores, weights, and outputs must be real numbers")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(
            "Scores, weights, and outputs must be finite"
        ) from error
    if not math.isfinite(result):
        raise ValueError("Scores, weights, and outputs must be finite")
    return result
