"""Record A3 dynamic-quant rounding and C8 rotation across fixed shapes."""

import argparse
import hashlib
import json
import math
import platform
import struct
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from tilert.models.glm_5_2_ascend.ops.indexer import hadamard128
from tilert.models.glm_5_2_ascend.ops.quantized import quantize_a8_reference


SHAPES = (
    (1, 128), (4, 128), (32, 128), (8192, 128), (8193, 128),
    (1, 6144), (4, 6144),
)
SAMPLE_LIMIT = 16


def float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(value):
    word = struct.unpack("<I", struct.pack("<f", value))[0]
    return f"0x{word:08x}"


def differences(actual, expected):
    mask = actual != expected
    positions = mask.nonzero(as_tuple=False)
    return {
        "unequal": int(mask.sum()),
        "total": actual.numel(),
        "positions": positions[:SAMPLE_LIMIT].tolist(),
    }


def scale_comparison(actual, expected):
    result = differences(actual.view(torch.int32), expected.view(torch.int32))
    result["samples"] = [
        {
            "row": row,
            "actual": actual[row].item(),
            "expected": expected[row].item(),
            "actual_bits": bits(actual[row].item()),
            "expected_bits": bits(expected[row].item()),
        }
        for row, in result["positions"]
    ]
    return result


def candidate_parameters(maximum):
    denominator = torch.where(maximum == 0, 1.0, maximum)
    full127 = torch.full_like(maximum, 127.0)
    inverse127 = torch.tensor(float32(1.0 / 127.0), dtype=torch.float32)
    scalar_div_scale = maximum / 127.0
    tensor_div_scale = maximum / full127
    multiply_scale = maximum * inverse127

    def reciprocal(scale):
        return torch.where(scale == 0, 1.0, scale).reciprocal()

    return {
        "tensor_127_div_amax": (full127 / denominator, multiply_scale),
        "literal_127_rdiv_amax": (127.0 / denominator, multiply_scale),
        "reciprocal_scalar_div_scale": (
            reciprocal(scalar_div_scale), scalar_div_scale,
        ),
        "reciprocal_tensor_div_scale": (
            reciprocal(tensor_div_scale), tensor_div_scale,
        ),
        "reciprocal_multiply_scale": (
            reciprocal(multiply_scale), multiply_scale,
        ),
    }


def scalar_parameters(maximum):
    """Round each scalar operation to FP32, independently of Torch kernels."""
    divided = [float32(value / 127.0) for value in maximum.tolist()]
    multiplied = [
        float32(value * float32(1.0 / 127.0)) for value in maximum.tolist()
    ]
    direct = [float32(127.0 / value) if value else 127.0
              for value in maximum.tolist()]
    inverse_then_multiply = [
        float32(float32(1.0 / value) * 127.0) if value else 127.0
        for value in maximum.tolist()
    ]
    inverse_divided = [float32(1.0 / value) if value else 1.0
                       for value in divided]
    inverse_multiplied = [float32(1.0 / value) if value else 1.0
                          for value in multiplied]
    return {
        "tensor_127_div_amax": (direct, multiplied),
        "literal_127_rdiv_amax": (inverse_then_multiply, multiplied),
        "reciprocal_scalar_div_scale": (inverse_divided, divided),
        "reciprocal_tensor_div_scale": (inverse_divided, divided),
        "reciprocal_multiply_scale": (inverse_multiplied, multiplied),
    }


def quantization_details(rows, actual, scale):
    values = rows.float()
    maximum = values.abs().amax(dim=-1)
    scalar = scalar_parameters(maximum)
    expected, expected_scale = quantize_a8_reference(rows)
    metadata = {
        "strict_reference_int8": differences(actual, expected),
        "strict_reference_scale": scale_comparison(scale, expected_scale),
        "candidates": {},
    }
    tensors = {
        "input_bf16": rows,
        "npu_int8": actual,
        "npu_scale_fp32": scale,
        "npu_scale_fp32_bits": scale.view(torch.int32),
        "amax_fp32": maximum,
        "amax_fp32_bits": maximum.view(torch.int32),
        "candidates": {},
    }
    candidates = candidate_parameters(maximum)
    for name, (factor, candidate_scale) in candidates.items():
        normalized = values * factor[:, None]
        candidate = normalized.round().clamp(-127, 127).to(torch.int8)
        comparison = differences(actual, candidate)
        samples = []
        sample_positions = {
            tuple(position) for position in comparison["positions"]
        }
        sample_positions.update((0, column) for column in range(5))
        for row, column in sorted(sample_positions):
            samples.append({
                "row": row, "column": column,
                "input": values[row, column].item(),
                "amax_bits": bits(maximum[row].item()),
                "factor_bits": bits(factor[row].item()),
                "before_round": normalized[row, column].item(),
                "before_round_bits": bits(normalized[row, column].item()),
                "npu_int8": actual[row, column].item(),
                "candidate_int8": candidate[row, column].item(),
            })
        scalar_factor = torch.tensor(scalar[name][0], dtype=torch.float32)
        scalar_scale = torch.tensor(scalar[name][1], dtype=torch.float32)
        metadata["candidates"][name] = {
            "int8": comparison,
            "scale": scale_comparison(scale, candidate_scale),
            "scale_fp16": differences(scale.half(), candidate_scale.half()),
            "factor_vs_scalar_steps": scale_comparison(factor, scalar_factor),
            "scale_vs_scalar_steps": scale_comparison(
                candidate_scale, scalar_scale
            ),
            "samples": samples,
        }
        tensors["candidates"][name] = {
            "int8": candidate,
            "before_round_fp32": normalized,
            "before_round_fp32_bits": normalized.view(torch.int32),
            "factor_fp32": factor,
            "factor_fp32_bits": factor.view(torch.int32),
            "scale_fp32": candidate_scale,
            "scale_fp32_bits": candidate_scale.view(torch.int32),
            "scalar_factor_fp32": scalar_factor,
            "scalar_scale_fp32": scalar_scale,
            "scalar_factor_fp32_bits": scalar_factor.view(torch.int32),
            "scalar_scale_fp32_bits": scalar_scale.view(torch.int32),
        }
    return metadata, tensors


def reference_hadamard():
    indices = torch.arange(128, dtype=torch.int32)
    products = indices[:, None] & indices[None, :]
    parity = torch.zeros_like(products)
    for shift in range(7):
        parity ^= (products >> shift) & 1
    return ((1 - 2 * parity).float() / math.sqrt(128)).to(torch.bfloat16)


def input_patterns(rows, columns, random_inputs):
    patterns = {}
    for name, maximum in (
        ("half_integer_3_75", 3.75),
        ("rounding_split_2_234375", 2.234375),
    ):
        pattern = torch.tensor(
            [maximum, -maximum, maximum / 2, -maximum / 2, 0.0],
            dtype=torch.bfloat16,
        )
        vector = pattern.repeat(math.ceil(columns / len(pattern)))[:columns]
        patterns[name] = vector.repeat(rows, 1)
    patterns["random"] = random_inputs[columns][:rows].clone()
    return patterns


def compare_shapes(label, shape, actual, scale, baselines):
    first_row = actual[0].clone()
    first_scale = scale[:1].clone()
    if label not in baselines:
        baselines[label] = (shape, first_row, first_scale)
    baseline_shape, baseline_values, baseline_scale = baselines[label]
    return {
        "baseline_shape": baseline_shape,
        "first_row_int8": differences(first_row, baseline_values),
        "first_row_scale": scale_comparison(first_scale, baseline_scale),
    }


def record_stage(label, device_rows, baselines):
    import torch_npu

    before = device_rows.cpu().clone()
    quantized, scale = torch_npu.npu_dynamic_quant(
        device_rows, dst_type=torch.int8
    )
    quantized, scale = quantized.cpu(), scale.cpu()
    metadata, tensors = quantization_details(before, quantized, scale)
    metadata["input_preserved"] = torch.equal(device_rows.cpu(), before)
    metadata["same_row_across_m"] = compare_shapes(
        label, list(before.shape), quantized, scale, baselines
    )
    pattern, _, stage = label
    if pattern != "random" and stage == "direct":
        metadata["same_values_across_k"] = compare_shapes(
            (pattern, "shared_prefix"), list(before.shape),
            quantized[:, :128], scale, baselines,
        )
    metadata["input_shape"] = list(before.shape)
    return metadata, tensors


def run_case(pattern, rows, args, baselines, report):
    count, columns = rows.shape
    name = f"m{count}-k{columns}-{pattern}"
    device_rows = rows.to(args.device)
    direct, direct_tensors = record_stage(
        (pattern, columns, "direct"), device_rows, baselines,
    )
    result = {"name": name, "direct": direct}
    tensors = {"input_bf16": rows, "direct": direct_tensors}
    if columns == 128:
        matrix = hadamard128()
        expected_matrix = reference_hadamard()
        rotated = device_rows @ matrix.to(args.device)
        cpu_rotated_fp32 = rows.float() @ expected_matrix.float()
        cpu_rotated = cpu_rotated_fp32.to(torch.bfloat16)
        rotated_cpu = rotated.cpu()
        rotation, rotation_tensors = record_stage(
            (pattern, columns, "rotated"), rotated, baselines,
        )
        end_to_end, reference_tensors = quantization_details(
            cpu_rotated, rotation_tensors["npu_int8"],
            rotation_tensors["npu_scale_fp32"],
        )
        result["hadamard_equal"] = torch.equal(matrix, expected_matrix)
        result["rotation_bf16"] = differences(rotated_cpu, cpu_rotated)
        result["rotation_bf16_bits"] = differences(
            rotated_cpu.view(torch.int16), cpu_rotated.view(torch.int16)
        )
        result["quantize_same_npu_rotation"] = rotation
        result["compare_cpu_rotation_end_to_end"] = end_to_end
        tensors.update({
            "hadamard_bf16": matrix,
            "npu_rotated_bf16": rotated_cpu,
            "cpu_rotated_bf16": cpu_rotated,
            "cpu_rotated_fp32": cpu_rotated_fp32,
            "quantize_same_npu_rotation": rotation_tensors,
            "compare_cpu_rotation_end_to_end": reference_tensors,
        })
    artifact = args.artifacts / f"{name}.pt"
    torch.save(tensors, artifact)
    result["artifact"] = str(artifact)
    result["artifact_sha256"] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    report["cases"].append(result)
    print(json.dumps({"case": name, "artifact": str(artifact)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    args.artifacts = args.output.parent / (args.output.stem + "-tensors")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    sources = (
        Path(__file__),
        root / "tilert/models/glm_5_2_ascend/ops/indexer.py",
        root / "tilert/models/glm_5_2_ascend/ops/quantized.py",
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "error", "hostname": platform.node(),
        "command": [sys.executable, *sys.argv],
        "git_head": args.source_commit, "device": args.device,
        "seed": args.seed, "shapes": SHAPES,
        "scope": "arithmetic diagnosis; existing strict checks unchanged",
        "c8_scoring_scope": (
            "encoder only; the QLI FP16 scoring boundary is checked "
            "separately by check_indexer.py"
        ),
        "scalar_literal_rdiv_model": "FP32(FP32(1/amax)*127)",
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest() for path in sources
        },
        "versions": {"python": platform.python_version(),
                     "torch": torch.__version__},
        "cases": [],
    }
    try:
        import torch_npu

        report["versions"]["torch_npu"] = torch_npu.__version__
        torch.set_num_threads(4)
        torch.npu.set_device(args.device)
        generator = torch.Generator().manual_seed(args.seed)
        random_inputs = {
            128: torch.randn(8193, 128, generator=generator).to(
                torch.bfloat16
            ),
            6144: torch.randn(4, 6144, generator=generator).to(torch.bfloat16),
        }
        baselines = {}
        with torch.inference_mode():
            for rows, columns in SHAPES:
                for pattern, values in input_patterns(
                    rows, columns, random_inputs
                ).items():
                    run_case(pattern, values, args, baselines, report)
        report["status"] = "complete"
    except Exception:
        report["error"] = traceback.format_exc()
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.write_text(payload)
    print(json.dumps({"status": report["status"],
                      "cases": len(report["cases"])}))
    return int(report["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
