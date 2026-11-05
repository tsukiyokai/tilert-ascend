"""Single-row TileLang W4A8 GEMV and a separate CPU reference."""

from functools import lru_cache
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tilelang.jit import JITKernel


SUPPORTED_SHAPES = ((256, 6144), (6144, 128))


def w4a8_gemv(
    x: torch.Tensor,
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    token_scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute BF16 ``[1, N]`` from A8 ``x[1, K]`` and packed W4.

    ``packed[N/2, K]`` stores output ``2*i`` in the low nibble and
    ``2*i+1`` in the high nibble of row ``i``. Scales are FP32 ``[N]``
    and ``[1]``. Inputs are contiguous NPU tensors and remain unchanged.
    Signed integer products accumulate in FP32 before either scale is
    applied. Direct signed W4 arithmetic needs no MSD ``scale_bias``.
    """
    shape = _validate_inputs(x, packed, weight_scale, token_scale, "npu")
    if out is None:
        out = torch.empty((1, shape[0]), dtype=torch.bfloat16, device=x.device)
    else:
        _validate_output(out, shape[0], (x, packed, weight_scale, token_scale))

    kernel = build_kernel(*shape)
    with torch.npu.device(x.device):
        kernel(x, packed.view(torch.uint8), weight_scale, token_scale, out)
    return out


def w4a8_gemv_reference(
    x: torch.Tensor,
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    token_scale: torch.Tensor,
) -> torch.Tensor:
    """Compute signed integer products and final BF16 rounding on CPU."""
    out_features, in_features = _validate_inputs(
        x, packed, weight_scale, token_scale, "cpu"
    )
    values = packed.to(torch.int16)
    low = ((values & 15) ^ 8) - 8
    high = (((values >> 4) & 15) ^ 8) - 8
    weight = torch.stack((low, high), dim=1).reshape(out_features, in_features)
    product = x.to(torch.int32) @ weight.to(torch.int32).T
    return (product.float() * weight_scale * token_scale).to(torch.bfloat16)


@lru_cache(maxsize=2)
def build_kernel(out_features: int, in_features: int) -> "JITKernel":
    """Compile an A3 specialization without allocating NPU tensors."""
    _validate_shape(out_features, in_features)
    from ._w4a8_tilelang import compile_w4a8_gemv

    return compile_w4a8_gemv(out_features, in_features)


def _validate_shape(out_features: int, in_features: int) -> None:
    if (out_features, in_features) not in SUPPORTED_SHAPES:
        raise ValueError(f"(N, K) must be one of {SUPPORTED_SHAPES}")


def _validate_inputs(
    x: torch.Tensor,
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    token_scale: torch.Tensor,
    device_type: str,
) -> tuple[int, int]:
    tensors = (x, packed, weight_scale, token_scale)
    if any(tensor.device.type != device_type for tensor in tensors):
        raise ValueError(f"all inputs must be on {device_type}")
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("all inputs must be on the same device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all inputs must be contiguous")
    if x.ndim != 2 or x.shape[0] != 1 or packed.ndim != 2:
        raise ValueError("x must have shape [1, K] and packed shape [N/2, K]")
    shape = (2 * packed.shape[0], x.shape[1])
    _validate_shape(*shape)
    if packed.shape[1] != shape[1]:
        raise ValueError("packed weight and activation K must match")
    if weight_scale.shape != (shape[0],) or token_scale.shape != (1,):
        raise ValueError("weight_scale must have shape [N], token_scale [1]")
    if x.dtype != torch.int8 or packed.dtype not in (torch.int8, torch.uint8):
        raise TypeError("x must be INT8 and packed must be INT8 or UINT8")
    if weight_scale.dtype != torch.float32 or token_scale.dtype != torch.float32:
        raise TypeError("both scales must be FP32")
    return shape


def _validate_output(
    out: torch.Tensor, out_features: int, inputs: tuple[torch.Tensor, ...]
) -> None:
    if out.shape != (1, out_features) or out.dtype != torch.bfloat16:
        raise ValueError("out must be BF16 with shape [1, N]")
    if out.device != inputs[0].device or not out.is_contiguous():
        raise ValueError("out must be contiguous and on the input device")
    input_storage = {tensor.untyped_storage().data_ptr() for tensor in inputs}
    if out.untyped_storage().data_ptr() in input_storage:
        raise ValueError("out must have separate storage from every input")
