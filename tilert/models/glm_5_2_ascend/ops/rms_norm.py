"""Single-row RMSNorm with separate NPU and CPU reference entry points."""

import math
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tilelang.jit import JITKernel


SUPPORTED_HIDDEN_SIZES = (64, 256, 512, 2048, 6144)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run TileLang Ascend RMSNorm on contiguous BF16 ``x[1, H]``.

    ``weight[H]`` is FP32 with BF16-representable values. Normalization is
    rounded to BF16 before applying weight, matching the HF GLM RMSNorm.
    An explicit output must have separate storage from
    both inputs. Execution uses the input device's current NPU stream.
    """
    hidden_size = _validate_inputs(x, weight, eps, "npu")
    if out is None:
        out = torch.empty_like(x)
    else:
        _validate_output(out, x, weight)

    kernel = build_kernel(hidden_size, eps)
    with torch.npu.device(x.device):
        kernel(x, weight, out)
    return out


def rms_norm_reference(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Compute the HF GLM normalization and weight rounding on CPU."""
    _validate_inputs(x, weight, eps, "cpu")
    x_float = x.float()
    inverse_rms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
    normalized = (x_float * inverse_rms).to(torch.bfloat16)
    return (normalized.float() * weight).to(torch.bfloat16)


@lru_cache(maxsize=32)
def build_kernel(hidden_size: int, eps: float = 1e-5) -> "JITKernel":
    """Compile an A3 kernel without allocating tensors or launching it.

    Requires tilelang-ascend's ``ascendc_pto`` compiler and CANN Bisheng.
    Compilation is explicit; importing this module does not load TileLang.
    """
    _validate_specialization(hidden_size, eps)
    from ._rms_norm_tilelang import compile_rms_norm

    return compile_rms_norm(hidden_size, float(eps))


def _validate_specialization(hidden_size: int, eps: float) -> None:
    if hidden_size not in SUPPORTED_HIDDEN_SIZES:
        raise ValueError(f"hidden_size must be one of {SUPPORTED_HIDDEN_SIZES}")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")


def _validate_inputs(
    x: torch.Tensor, weight: torch.Tensor, eps: float, device_type: str
) -> int:
    if x.device.type != device_type or weight.device.type != device_type:
        raise ValueError(f"x and weight must be on {device_type}")
    if x.device != weight.device:
        raise ValueError("x and weight must be on the same device")
    if x.ndim != 2 or x.shape[0] != 1:
        raise ValueError("x must have shape [1, hidden_size]")
    hidden_size = x.shape[1]
    _validate_specialization(hidden_size, eps)
    if weight.shape != (hidden_size,):
        raise ValueError("weight must have shape [hidden_size]")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.float32:
        raise TypeError("x must be BF16 and weight must be FP32")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x and weight must be contiguous")
    return hidden_size


def _validate_output(
    out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
) -> None:
    if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device:
        raise ValueError("out must match x's shape, dtype, and device")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    output_storage = out.untyped_storage().data_ptr()
    if output_storage in (
        x.untyped_storage().data_ptr(),
        weight.untyped_storage().data_ptr(),
    ):
        raise ValueError("out must have separate storage from x and weight")
