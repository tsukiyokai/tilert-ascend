"""One TP16 W4A8 expert partition with explicit BF16 SwiGLU boundaries."""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from ..weights import ExpertWeights, QuantizedWeight
from .quantized import NpuQuantizedLinear


@dataclass(frozen=True)
class ExpertTrace:
    input_quantized: torch.Tensor
    input_scale: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    silu: torch.Tensor
    intermediate: torch.Tensor
    down_quantized: torch.Tensor
    down_scale: torch.Tensor
    output: torch.Tensor


@dataclass
class NpuExpert:
    """Rank-local output before TP reduction and routing-weight application."""

    gate_up: NpuQuantizedLinear
    down: NpuQuantizedLinear
    hidden_size: int
    intermediate_size: int

    @classmethod
    def from_checkpoint(
        cls, source: ExpertWeights, device: str = "npu:0"
    ) -> "NpuExpert":
        gate, up, down = source.gate, source.up, source.down
        if any(weight.bits != 4 for weight in (gate, up, down)):
            raise ValueError("The expert requires three W4A8 projections")
        if any(weight.scale_bias is None for weight in (gate, up, down)):
            raise ValueError("W4A8 projections require scale_bias")
        if (
            gate.in_features != up.in_features
            or gate.out_features != up.out_features
            or gate.out_features != down.in_features
            or gate.in_features != down.out_features
        ):
            raise ValueError("The expert projection dimensions do not match")
        merged = QuantizedWeight(
            values=torch.cat((gate.values, up.values), dim=0),
            scale=torch.cat((gate.scale, up.scale)),
            bits=4,
            scale_bias=torch.cat((gate.scale_bias, up.scale_bias)),
        )
        return cls(
            NpuQuantizedLinear.from_checkpoint(merged, device),
            NpuQuantizedLinear.from_checkpoint(down, device),
            gate.in_features,
            gate.out_features,
        )

    def trace(self, x: torch.Tensor) -> ExpertTrace:
        """Return intermediates without modifying x or either INT8 input."""
        import torch_npu

        if x.dtype != torch.bfloat16 or x.ndim != 2:
            raise ValueError("Expert input must be BF16 [tokens, hidden]")
        if x.shape[1] != self.hidden_size:
            raise ValueError("Expert input hidden dimension does not match")
        activation, scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
        projected = self.gate_up.apply_quantized(activation, scale)
        gate, up = projected.split(self.intermediate_size, dim=-1)
        silu = functional.silu(gate)
        intermediate = silu * up
        down_activation, down_scale = torch_npu.npu_dynamic_quant(
            intermediate, dst_type=torch.int8
        )
        output = self.down.apply_quantized(down_activation, down_scale)
        return ExpertTrace(
            activation, scale, gate, up, silu, intermediate,
            down_activation, down_scale, output,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.trace(x).output
