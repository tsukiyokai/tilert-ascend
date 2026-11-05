"""A3 vector GEMV that reads two signed W4 output channels per byte."""

import torch_npu  # noqa: F401
import tilelang
from tilelang import language as T
from tilelang.jit import JITKernel
from tvm.tir import PrimFunc


OUTPUT_PAIRS = 8


def compile_w4a8_gemv(out_features: int, in_features: int) -> JITKernel:
    return tilelang.compile(
        _w4a8_prim_func(out_features, in_features),
        out_idx=[],
        target="ascendc",
        platform="A3",
        execution_backend="cython",
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
        },
    )


def _w4a8_prim_func(out_features: int, in_features: int) -> PrimFunc:
    block_k = min(256, in_features)
    block_n = 2 * OUTPUT_PAIRS
    vector_blocks = out_features // block_n
    tile_elements = OUTPUT_PAIRS * block_k

    @T.prim_func
    def w4a8_gemv_row(
        x: T.Tensor((1, in_features), "int8"),
        packed: T.Tensor((out_features // 2, in_features), "uint8"),
        weight_scale: T.Tensor((out_features,), "float32"),
        token_scale: T.Tensor((1,), "float32"),
        out: T.Tensor((1, out_features), "bfloat16"),
    ):
        with T.Kernel(vector_blocks // 2, is_npu=True) as (block, vector):
            worker = 2 * block + vector
            pair_start = worker * OUTPUT_PAIRS
            output_start = worker * block_n

            packed_ub = T.alloc_ub((OUTPUT_PAIRS, block_k), "uint8")
            weight_half = T.alloc_ub((OUTPUT_PAIRS, block_k), "float16")
            weight_integer = T.alloc_ub((OUTPUT_PAIRS, block_k), "int16")
            nibble = T.alloc_ub((OUTPUT_PAIRS, block_k), "int16")
            signed_nibble = T.alloc_ub((OUTPUT_PAIRS, block_k), "int16")
            low_mask = T.alloc_ub((OUTPUT_PAIRS, block_k), "int16")
            sign_mask = T.alloc_ub((OUTPUT_PAIRS, block_k), "int16")
            product = T.alloc_ub((OUTPUT_PAIRS, block_k), "float32")
            x_integer = T.alloc_ub((block_k,), "int8")
            x_half = T.alloc_ub((block_k,), "float16")
            x_float = T.alloc_ub((block_k,), "float32")
            partial = T.alloc_ub((OUTPUT_PAIRS,), "float32")
            low_sum = T.alloc_ub((OUTPUT_PAIRS,), "float32")
            high_sum = T.alloc_ub((OUTPUT_PAIRS,), "float32")
            scale = T.alloc_ub((block_n,), "float32")
            row_scale = T.alloc_ub((1,), "float32")
            output_float = T.alloc_ub((block_n,), "float32")
            output_bfloat = T.alloc_ub((block_n,), "bfloat16")

            T.tile.fill(low_sum, 0.0)
            T.tile.fill(high_sum, 0.0)
            T.tile.fill(low_mask, 15)
            T.tile.fill(sign_mask, 8)

            for tile in T.serial(in_features // block_k):
                start = tile * block_k
                T.copy(x[0, start:start + block_k], x_integer)
                T.copy(
                    packed[pair_start:pair_start + OUTPUT_PAIRS,
                           start:start + block_k],
                    packed_ub,
                )
                T.tile.cast(x_half, x_integer, "CAST_NONE", block_k)
                T.tile.cast(x_float, x_half, "CAST_NONE", block_k)
                T.tile.cast(weight_half, packed_ub, "CAST_NONE", tile_elements)
                T.tile.cast(
                    weight_integer, weight_half, "CAST_RINT", tile_elements
                )

                # AscendC Xor requires a destination separate from its inputs.
                # (u4 XOR 8) - 8 restores signed nibbles without a branch.
                T.tile.bitwise_and(nibble, weight_integer, low_mask)
                T.tile.bitwise_xor(signed_nibble, nibble, sign_mask)
                T.tile.cast(
                    weight_half, signed_nibble, "CAST_NONE", tile_elements
                )
                T.tile.cast(product, weight_half, "CAST_NONE", tile_elements)
                T.tile.add(product, product, -8.0)
                for pair in T.serial(OUTPUT_PAIRS):
                    T.tile.mul(product[pair, :], product[pair, :], x_float)
                T.reduce_sum(product, partial, dim=-1)
                T.tile.add(low_sum, low_sum, partial)

                T.tile.bitwise_rshift(nibble, weight_integer, 4)
                T.tile.bitwise_xor(signed_nibble, nibble, sign_mask)
                T.tile.cast(
                    weight_half, signed_nibble, "CAST_NONE", tile_elements
                )
                T.tile.cast(product, weight_half, "CAST_NONE", tile_elements)
                T.tile.add(product, product, -8.0)
                for pair in T.serial(OUTPUT_PAIRS):
                    T.tile.mul(product[pair, :], product[pair, :], x_float)
                T.reduce_sum(product, partial, dim=-1)
                T.tile.add(high_sum, high_sum, partial)

            for pair in T.serial(OUTPUT_PAIRS):
                output_float[2 * pair] = low_sum[pair]
                output_float[2 * pair + 1] = high_sum[pair]

            T.copy(weight_scale[output_start:output_start + block_n], scale)
            T.copy(token_scale, row_scale)
            T.tile.mul(output_float, output_float, scale)
            T.tile.mul(output_float, output_float, row_scale[0])
            T.tile.cast(output_bfloat, output_float, "CAST_RINT", block_n)
            T.copy(output_bfloat, out[0, output_start:output_start + block_n])

    return w4a8_gemv_row
