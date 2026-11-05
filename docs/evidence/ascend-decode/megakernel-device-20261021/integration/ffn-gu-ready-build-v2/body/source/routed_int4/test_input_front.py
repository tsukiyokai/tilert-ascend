"""Routing-front oracles preserve original slots across expert row orders."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location(
    "input_front_check", HERE / "check_input_front.py",
)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def test_sparse_metadata_keeps_original_slots_and_probability_rounding():
    choices = torch.tensor([
        [255, 0, 7, 3, 6, 2, 5, 1],
        [1, 5, 2, 6, 3, 7, 0, 255],
    ], dtype=torch.int32)
    probabilities = torch.arange(1, 17).reshape(2, 8).float() / 32
    probabilities[0, 0] = -0.0
    probabilities[0, 1] = 1 + 2 ** -8
    counts, starts, sources, slots, weights = CHECK.metadata_reference(
        torch, choices, probabilities,
    )
    original = torch.tensor([1, 14, 7, 8, 5, 10, 3, 12,
                             6, 9, 4, 11, 2, 13, 0, 15])
    assert torch.equal(sources[:16] * 8 + slots[:16], original)
    assert counts.nonzero().flatten().tolist() == [0, 1, 2, 3, 5, 6, 7, 255]
    assert counts[counts != 0].tolist() == [2] * 8
    assert starts[[0, 1, 2, 3, 4, 5, 6, 7, 8, 255]].tolist() == [
        0, 2, 4, 6, 8, 8, 10, 12, 14, 14,
    ]
    assert weights[0].item() == 1.0
    assert torch.signbit(weights[14])
    assert sources[16:].count_nonzero() == slots[16:].count_nonzero() == 0
    assert weights[16:].count_nonzero() == 0


@pytest.mark.parametrize("width", [1, 2, 12, 32])
@pytest.mark.parametrize("iteration", [0, 3, 6, 7])
def test_expert_permutation_preserves_original_high_low_rows(width, iteration):
    _, choices, probabilities = CHECK.check.route_case(
        torch, width, iteration, 20261021,
    )
    counts, starts, sources, slots, weights = CHECK.metadata_reference(
        torch, choices, probabilities,
    )
    routes = width * 8
    original = sources[:routes].long() * 8 + slots[:routes]
    assert counts.max() <= width
    assert original.sort().values.tolist() == list(range(routes))
    # Reversing each expert's rows changes the inverse map while retaining
    # rowwise projections and both high/low MSD halves in original order.
    permuted = original.clone()
    for expert in counts.nonzero().flatten().tolist():
        begin, count = int(starts[expert]), int(counts[expert])
        permuted[begin:begin + count] = original[begin:begin + count].flip(0)
    compact = torch.empty((2 * routes, 2), dtype=torch.int32)
    for expert in counts.nonzero().flatten().tolist():
        begin, count = int(starts[expert]), int(counts[expert])
        group = permuted[begin:begin + count].to(torch.int32)
        for part in range(2):
            values = group + 1000 * part
            compact[2 * begin + part * count:2 * begin + (part + 1) * count] = (
                torch.stack((values, -values), -1)
            )
    metadata = counts, starts, permuted // 8, permuted % 8, weights
    restored = CHECK.check.ordered_partials(torch, compact, metadata, width, 2)
    expected = torch.arange(routes, dtype=torch.int32)
    assert torch.equal(restored[0, :, 0], expected)
    assert torch.equal(restored[1, :, 0], expected + 1000)
    assert torch.equal(restored[:, :, 1], -restored[:, :, 0])


def test_all_256_experts_have_one_route_and_no_padding():
    choices = torch.arange(255, -1, -1, dtype=torch.int32).reshape(32, 8)
    probabilities = torch.full((32, 8), 0.125)
    counts, starts, sources, slots, weights = CHECK.metadata_reference(
        torch, choices, probabilities,
    )
    assert counts.tolist() == [1] * 256
    assert starts.tolist() == list(range(256))
    assert (sources * 8 + slots).tolist() == list(range(255, -1, -1))
    assert weights.tolist() == [0.125] * 256


def test_timing_requires_30_independent_event_samples(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "check_input_front.py", "--device", "npu:0", "--output", "/tmp/out.json",
        "--timing-samples", "6", "--iterations", "300",
    ])
    with pytest.raises(SystemExit):
        CHECK.arguments()
