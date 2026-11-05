"""CPU checks for the packed operands consumed by the MLA tail kernel."""

import pytest
import torch

from tilert.models.glm_5_2_ascend.mla_tail import (
    pack_output_weights,
    pack_value_weights,
)


@pytest.mark.parametrize("rows", [1, 12, 32])
def test_value_tiles_preserve_head_and_channel_order(rows):
    generator = torch.Generator().manual_seed(7891)
    latent = torch.randint(-4, 5, (rows, 4, 512), generator=generator).float()
    weights = torch.randint(
        -4, 5, (4, 512, 256), generator=generator,
    ).bfloat16()
    packed = pack_value_weights(weights)
    expected = torch.bmm(latent.transpose(0, 1), weights.float())
    actual = torch.empty(4, rows, 256)
    for tile in range(16):
        head, channel_tile = divmod(tile, 4)
        first = channel_tile * 64
        actual[head, :, first:first + 64] = (
            latent[:, head] @ packed[tile].float()
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_output_tiles_preserve_integer_projection():
    generator = torch.Generator().manual_seed(8912)
    inputs = torch.randint(-128, 128, (3, 1024), generator=generator)
    weights = torch.randint(
        -128, 128, (1024, 6144), dtype=torch.int8, generator=generator,
    )
    packed = pack_output_weights(weights)
    actual = torch.cat([inputs @ tile.long() for tile in packed], dim=-1)
    torch.testing.assert_close(actual, inputs @ weights.long(), rtol=0, atol=0)
