"""Independent shared inputs and tile-major diagnostics for composition."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location(
    "pipeline_front_check", HERE / "check_pipeline_front.py",
)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


@pytest.mark.parametrize("iteration", range(8))
def test_shared_input_has_distinct_values_at_every_edge_case(iteration):
    routed, shared, choices, probabilities = CHECK.input_case(
        torch, 12, iteration, 20261021,
    )
    assert routed.shape == shared.shape == (12, 6144)
    assert routed.dtype == shared.dtype == torch.bfloat16
    assert not torch.equal(routed, shared)
    assert choices.shape == probabilities.shape == (12, 8)


def test_shared_products_omit_padding_in_every_column_tile():
    capacity, width = 16, 2
    products = torch.full((24, capacity, 256), -999, dtype=torch.int32)
    for tile in range(24):
        products[tile, 0] = torch.arange(tile * 256, (tile + 1) * 256)
        products[tile, 1] = -products[tile, 0]
    prefix = torch.zeros(512, dtype=torch.uint8)
    payload = torch.cat((prefix, products.reshape(-1).view(torch.uint8)))
    actual = CHECK.shared_down_products(torch, payload, 512, width, capacity)
    expected = torch.arange(6144, dtype=torch.int32)
    assert torch.equal(actual, torch.stack((expected, -expected)))
