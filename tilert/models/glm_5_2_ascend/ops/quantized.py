"""CANN baselines for the checkpoint's W8A8 and W4A8 projections."""

from dataclasses import dataclass

import torch

from ..weights import QuantizedWeight


def quantize_a8_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric INT8 with FP32 scales and nearest-even rounding."""
    values = x.float()
    maximum = values.abs().amax(dim=-1)
    scale = maximum * (1 / 127)
    divisor = torch.where(maximum == 0, torch.ones_like(maximum), maximum)
    factor = torch.div(torch.full_like(divisor, 127), divisor)
    quantized = (values * factor[:, None]).round().clamp(-127, 127)
    return quantized.to(torch.int8), scale


@dataclass
class NpuQuantizedLinear:
    """Prepacked single projection; input is BF16 [M, input]."""

    weight: torch.Tensor
    scale: torch.Tensor
    bits: int
    scale_bias: torch.Tensor | None

    @classmethod
    def from_checkpoint(
        cls, source: QuantizedWeight, device: str = "npu:0"
    ) -> "NpuQuantizedLinear":
        import torch_npu

        weight = source.values.T.contiguous()
        scale = source.scale
        bias = source.scale_bias
        if source.bits == 4:
            # INT32 stores eight output nibbles in ND order.
            weight = weight.view(torch.int32).unsqueeze(0).to(device)
            # CANN A8W4 reads FP32 scale bits from the low word of INT64.
            scale = (scale.view(torch.int32).to(torch.int64) & 0xFFFFFFFF)
            scale = scale.reshape(1, 1, -1)
            bias = bias.reshape(1, -1).to(device)
        else:
            torch.npu.config.allow_internal_format = True
            weight = torch_npu.npu_format_cast(weight.to(device), 29)
        return cls(weight, scale.to(device), source.bits, bias)

    def apply_quantized(
        self, activation: torch.Tensor, activation_scale: torch.Tensor
    ) -> torch.Tensor:
        """Project INT8 rows without modifying the caller's activation."""
        import torch_npu

        if self.bits == 8:
            return torch_npu.npu_quant_matmul(
                activation,
                self.weight,
                self.scale,
                pertoken_scale=activation_scale,
                output_dtype=torch.bfloat16,
            )
        return torch_npu.npu_grouped_matmul(
            # CANN's A8W4 MSD preprocess overwrites its input with INT4 parts.
            x=[activation.clone()],
            weight=[self.weight],
            scale=[self.scale],
            bias=[self.scale_bias],
            per_token_scale=[activation_scale],
            split_item=2,
            group_type=0,
            group_list=torch.tensor(
                [activation.shape[0]], dtype=torch.int64, device=activation.device
            ),
            group_list_type=0,
            output_dtype=torch.bfloat16,
        )[0]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        import torch_npu

        activation, scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
        return self.apply_quantized(activation, scale)
