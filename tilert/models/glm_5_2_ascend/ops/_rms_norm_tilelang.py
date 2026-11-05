"""A3 vector RMSNorm for tilelang-ascend's ascendc_pto backend."""

import torch_npu  # noqa: F401
import tilelang
from tilelang import language as T
from tilelang.jit import JITKernel
from tvm.tir import PrimFunc


def compile_rms_norm(hidden_size: int, eps: float) -> JITKernel:
    return tilelang.compile(
        _rms_norm_prim_func(hidden_size, eps),
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


def _rms_norm_prim_func(hidden_size: int, eps: float) -> PrimFunc:
    @T.prim_func
    def rms_norm_row(
        x: T.Tensor((1, hidden_size), "bfloat16"),
        weight: T.Tensor((hidden_size,), "float32"),
        out: T.Tensor((1, hidden_size), "bfloat16"),
    ):
        with T.Kernel(1, threads=1, is_npu=True) as block:
            x_bfloat = T.alloc_ub((hidden_size,), "bfloat16")
            x_float = T.alloc_ub((hidden_size,), "float32")
            weight_float = T.alloc_ub((hidden_size,), "float32")
            squares = T.alloc_ub((hidden_size,), "float32")
            inverse_rms = T.alloc_ub((1,), "float32")
            variance = T.alloc_ub((1,), "float32")
            correction = T.alloc_ub((1,), "float32")

            T.copy(x[block, :], x_bfloat)
            T.copy(weight, weight_float)
            T.tile.cast(x_float, x_bfloat, "CAST_NONE", hidden_size)
            T.tile.mul(squares, x_float, x_float)
            T.reduce_sum(squares, inverse_rms, dim=0)
            T.tile.mul(inverse_rms, inverse_rms, 1.0 / hidden_size)
            T.tile.add(inverse_rms, inverse_rms, eps)
            T.tile.mul(variance, inverse_rms, 1.0)
            T.tile.rsqrt(inverse_rms, inverse_rms)
            # Refine the A3 Rsqrt approximation before BF16 rounding.
            for iteration in T.serial(2):
                T.tile.mul(correction, inverse_rms, inverse_rms)
                T.tile.mul(correction, correction, variance)
                T.tile.mul(correction, correction, -0.5)
                T.tile.add(correction, correction, 1.5)
                T.tile.mul(inverse_rms, inverse_rms, correction)
            T.tile.mul(x_float, x_float, inverse_rms[0])
            T.tile.cast(x_bfloat, x_float, "CAST_RINT", hidden_size)
            T.tile.cast(x_float, x_bfloat, "CAST_NONE", hidden_size)
            T.tile.mul(x_float, x_float, weight_float)
            T.tile.cast(x_bfloat, x_float, "CAST_RINT", hidden_size)
            T.copy(x_bfloat, out[block, :])

    return rms_norm_row
