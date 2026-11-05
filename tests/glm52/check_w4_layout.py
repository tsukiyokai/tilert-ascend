"""Locate W4 layout errors with exact integer one-hot projections."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu

from tilert.models.glm_5_2_ascend.ops.quantized import NpuQuantizedLinear
from tilert.models.glm_5_2_ascend.weights import ModelSlimCheckpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    if args.checkpoint:
        source = ModelSlimCheckpoint(args.checkpoint).expert(3, 0, 0).gate
        weights = source.unpack()
        packed = source.values
        weight_scale = source.scale
        weight_bias = source.scale_bias
    else:
        generator = torch.Generator().manual_seed(1234)
        weights = torch.randint(-8, 8, (128, 128), generator=generator).to(torch.int8)
        packed = (weights[0::2] & 15) | (weights[1::2] << 4)
        weight_scale = torch.ones(128)
        weight_bias = 8 * weights.sum(dim=1).float()
    nd = packed.T.contiguous().unsqueeze(0).npu()
    variants = {
        "nd_int32": nd.view(torch.int32),
        "int8_nz_then_int32": torch_npu.npu_format_cast(nd, 29).view(torch.int32),
        "int32_then_nz": torch_npu.npu_format_cast(nd.view(torch.int32), 29),
    }
    positions = torch.tensor([0, 1, 31, 32, 63, 64, 127])
    x = torch.zeros(len(positions), weights.shape[1], dtype=torch.int8)
    x[torch.arange(len(positions)), positions] = 1
    expected = (weights[:, positions].T.float() * weight_scale).to(torch.bfloat16)
    scale = weight_scale.view(torch.int32).to(torch.int64).reshape(1, 1, -1).npu()
    bias = weight_bias.reshape(1, -1).npu()
    x_npu = x.npu()
    x_scale = torch.ones(len(positions)).npu()
    results = {}
    for name, weight in variants.items():
        operation = NpuQuantizedLinear(weight, scale, 4, bias)
        actual = operation.apply_quantized(x_npu.clone(), x_scale).cpu()
        zero = operation.apply_quantized(torch.zeros_like(x_npu), x_scale).cpu()
        no_bias = NpuQuantizedLinear(weight, scale, 4, torch.zeros_like(bias))
        zero_bias_actual = no_bias.apply_quantized(x_npu.clone(), x_scale).cpu()
        results[name] = {
            "format": torch_npu.get_npu_format(weight),
            "shape": list(weight.shape),
            "storage_bytes": weight.untyped_storage().nbytes(),
            "exact": torch.equal(actual, expected),
            "max_abs": (actual.float() - expected.float()).abs().max().item(),
            "zero_input_max": zero.float().abs().max().item(),
            "zero_bias_max_error": (
                zero_bias_actual.float() - expected.float()
            ).abs().max().item(),
            "max_error_after_subtract_zero": (
                actual.float() - zero.float() - expected.float()
            ).abs().max().item(),
            "expected_first16": expected[:, :16].tolist(),
            "actual_first16": actual[:, :16].tolist(),
        }
    report = {"positions": positions.tolist(), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {k: v for k, v in row.items() if "first16" not in k}
                      for name, row in results.items()}, indent=2))


if __name__ == "__main__":
    main()
