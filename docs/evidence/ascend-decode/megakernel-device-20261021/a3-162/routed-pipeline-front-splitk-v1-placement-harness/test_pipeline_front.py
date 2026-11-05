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


@pytest.mark.parametrize("iteration", range(8))
def test_shared_experts_preserve_inputs_and_slot_probabilities(iteration):
    mixed = CHECK.input_case(torch, 12, iteration, 20261021)
    shared = CHECK.input_case(torch, 12, iteration, 20261021, "shared8")
    for index in (0, 1, 3):
        assert torch.equal(mixed[index].view(torch.uint8),
                           shared[index].view(torch.uint8))
    assert shared[2].unique().numel() == 8
    assert torch.equal(shared[2][0], mixed[2][0])
    assert torch.equal(shared[2], shared[2][0].expand_as(shared[2]))


def test_shared_timing_storage_preserves_library_tiling_owners():
    first = [torch.full((4,), index, dtype=torch.int32) for index in range(20)]
    second = [value.clone() for value in first]
    owned, (shared, sync) = CHECK.shared_timing_arenas(
        torch, (first, 123), (second, 123),
    )
    assert sync == 123
    assert all(left is right for left, right in zip(owned[0], first))
    for index, value in enumerate(shared):
        assert value is (second[index] if index in (6, 7, 18) else first[index])
    first[0].fill_(17)
    assert shared[0].tolist() == [17] * 4
    assert second[0].tolist() == [0] * 4
    second[6].fill_(29)
    assert shared[6].tolist() == [29] * 4
    assert first[6].tolist() == [6] * 4


def test_shared_timing_storage_rejects_different_prepared_tilings():
    first = [torch.zeros(4, dtype=torch.int32) for _ in range(20)]
    second = [value.clone() for value in first]
    second[18][0] = 1
    with pytest.raises(ValueError, match="byte-identical prepared tilings"):
        CHECK.shared_timing_arenas(torch, (first, 123), (second, 123))


@pytest.mark.parametrize("owner_planes", [1, 3])
def test_shared_gate_planes_preserve_prefix_and_library_tilings(owner_planes):
    rows = 16
    owner = [torch.zeros(4, dtype=torch.int32) for _ in range(20)]
    recipient = [value.clone() for value in owner]
    owner[1] = torch.full((owner_planes * rows, 256), -11, dtype=torch.int32)
    recipient[1] = torch.full(((4 - owner_planes) * rows, 256), -29,
                              dtype=torch.int32)
    owned, shared = CHECK.shared_timing_arenas(
        torch, (owner, 123), (recipient, 123),
    )
    large, small = sorted((owned[0][1], shared[0][1]),
                          key=lambda tensor: tensor.numel(), reverse=True)
    assert large.shape == (3 * rows, 256)
    assert small.shape == (rows, 256)
    assert large.data_ptr() == small.data_ptr()
    assert large.stride() == small.stride() == (256, 1)
    assert large.storage_offset() == small.storage_offset() == 0
    assert large.untyped_storage().nbytes() == 3 * rows * 256 * 4
    large.fill_(23)
    small.fill_(7)
    assert torch.equal(large[:rows], torch.full_like(small, 7))
    assert torch.all(large[rows:] == 23)
    for index in (6, 7, 18):
        assert owned[0][index] is owner[index]
        assert shared[0][index] is recipient[index]
    if owner_planes == 3:
        assert large.data_ptr() == owner[1].data_ptr()
    else:
        assert large.data_ptr() != owner[1].data_ptr()
        assert torch.all(owner[1] == -11)


def test_shared_gate_planes_reject_other_shape_changes():
    owner = [torch.zeros(4, dtype=torch.int32) for _ in range(20)]
    recipient = [value.clone() for value in owner]
    owner[1] = torch.zeros((16, 256), dtype=torch.int32)
    recipient[1] = torch.zeros((32, 256), dtype=torch.int32)
    with pytest.raises(ValueError, match="scratch contracts"):
        CHECK.shared_timing_arenas(torch, (owner, 123), (recipient, 123))
