"""CPU contract for the TP16 MLA tail; no NPU or torch import is required."""

from dataclasses import dataclass

import numpy as np


HEADS = 4
LATENT = 512
VALUE = 256
HIDDEN = 6144
VALUE_TILE = 64
OUTPUT_TILE = 256
PROJECTION_INPUT = HEADS * VALUE
SYSTEM_WORKSPACE_BYTES = 16 << 20


@dataclass(frozen=True)
class Region:
    name: str
    offset: int
    size: int


def workspace_layout(capacity, heads=HEADS):
    """Return disjoint byte ranges, including every padded cube row."""
    if capacity not in (16, 32) or heads not in (4, 8):
        raise ValueError("capacity must be 16|32 and heads must be 4|8")
    projection_input = heads * VALUE
    sizes = (
        ("heads", capacity * heads * LATENT * 2),
        ("values", capacity * projection_input * 4),
        ("quantized", capacity * projection_input),
        ("token_scales", capacity * 8 * 4),
        ("accumulators", capacity * HIDDEN * 4),
        ("system", SYSTEM_WORKSPACE_BYTES),
    )
    regions = []
    end = 0
    for name, size in sizes:
        offset = (end + 511) // 512 * 512
        regions.append(Region(name, offset, size))
        end = offset + size
    return tuple(regions)


def resident_blocks(cubes, vectors, requested=0):
    """The mixed 1:2 launch must fit concurrently in both core populations."""
    limit = min(cubes, vectors // 2)
    blocks = min(8, limit) if requested == 0 else requested
    if blocks <= 0 or blocks > limit:
        raise ValueError("block count exceeds mixed-core residency")
    return blocks


def bf16_round(values):
    """Round finite FP32 values to BF16, returning their FP32 representation."""
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("the tail arithmetic contract requires finite values")
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def quantize_rows(values):
    """Ideal FP32 dynamic A8 formula; device vector-Div rounding is separate."""
    values = np.asarray(values, dtype=np.float32)
    maximum = np.max(np.abs(values), axis=-1)
    scale = maximum * np.float32(1 / 127)
    denominator = np.where(maximum == 0, np.float32(1), maximum)
    factor = np.float32(127) / denominator
    codes = np.rint(values * factor[..., None]).clip(-127, 127)
    return codes.astype(np.int8), scale


def pack_weights(value_weight, output_weight):
    """Pack logical ND W_UV[H,512,256] and o_proj[256*H,6144]."""
    if (value_weight.ndim != 3 or value_weight.shape[0] not in (4, 8)
            or value_weight.shape[1:] != (LATENT, VALUE)):
        raise ValueError("value_weight must be [4|8,512,256]")
    heads = value_weight.shape[0]
    projection_input = heads * VALUE
    value_tile = projection_input // 16
    if output_weight.shape != (projection_input, HIDDEN):
        raise ValueError(f"output_weight must be [{projection_input},6144]")
    value_tiles = value_weight.reshape(
        heads, LATENT, VALUE // value_tile, value_tile,
    )
    value_tiles = value_tiles.transpose(0, 2, 1, 3).copy()
    output_tiles = output_weight.reshape(projection_input, 24, OUTPUT_TILE)
    output_tiles = output_tiles.transpose(1, 0, 2).copy()
    return value_tiles.reshape(16, LATENT, value_tile), output_tiles


def dequantize(accumulator, weight_scale, token_scale, bias=None):
    """CANN's channel-scale, token-scale, floating-bias, BF16 order."""
    if bias is not None and bias.dtype == np.int32:
        accumulator = accumulator + bias
    result = accumulator.astype(np.float32) * weight_scale
    result = result * token_scale[:, None]
    if bias is not None and bias.dtype != np.int32:
        result = result + bias.astype(np.float32)
    return bf16_round(result)


def reference(latent, value_weight, output_weight, weight_scale, bias=None):
    """The original two projections with the intervening BF16/A8 boundary."""
    projection_input = latent.shape[1] * VALUE
    expanded = np.einsum("mhd,hdv->mhv", latent, value_weight, optimize=True)
    expanded = bf16_round(expanded).reshape(latent.shape[0], projection_input)
    quantized, token_scale = quantize_rows(expanded)
    accumulator = quantized.astype(np.int32) @ output_weight.astype(np.int32)
    return dequantize(accumulator, weight_scale, token_scale, bias)


def tiled_reference(latent, value_tiles, output_tiles, weight_scale,
                    capacity, blocks, bias=None):
    """Execute the kernel's tile ownership and padded scratch on the CPU."""
    if not 0 < latent.shape[0] <= capacity or capacity not in (16, 32):
        raise ValueError("invalid row capacity")
    if blocks <= 0:
        raise ValueError("blocks must be positive")
    rows, head_count, _ = latent.shape
    value_tile = (head_count * VALUE) // 16
    tiles_per_head = VALUE // value_tile
    heads = np.zeros((head_count, capacity, LATENT), dtype=np.float32)
    heads[:, :rows] = latent.transpose(1, 0, 2)
    values = np.full((16, capacity, value_tile), np.nan, dtype=np.float32)
    for core in range(blocks):
        for tile in range(core, 16, blocks):
            values[tile] = heads[tile // tiles_per_head] @ value_tiles[tile]
    expanded = bf16_round(values.transpose(1, 0, 2))
    quantized, token_scale = quantize_rows(expanded.reshape(capacity, -1))
    accumulator = np.empty((24, capacity, OUTPUT_TILE), dtype=np.int32)
    for core in range(blocks):
        for tile in range(core, 24, blocks):
            accumulator[tile] = (
                quantized.astype(np.int32)
                @ output_tiles[tile].astype(np.int32)
            )
    accumulator = accumulator.transpose(1, 0, 2).reshape(capacity, HIDDEN)
    return dequantize(accumulator[:rows], weight_scale,
                      token_scale[:rows], bias)
