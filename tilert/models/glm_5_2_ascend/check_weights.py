"""Validate real ModelSlim packing, TP16 slices, and CANN projections."""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from .ops.quantized import NpuQuantizedLinear, quantize_a8_reference
from .reference_weights import quantize_a8, w4_msd_product
from .weights import ModelSlimCheckpoint


def compare(name, actual, expected, checks, *, exact=True):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        checks.append({
            "name": name,
            "passed": False,
            "shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "dtype": str(actual.dtype),
            "expected_dtype": str(expected.dtype),
        })
        return
    left, right = actual.float(), expected.float()
    error = (left - right).norm().item()
    reference_norm = right.norm().item()
    nrmse = error / max(reference_norm, 1e-12)
    if reference_norm == 0:
        cosine = float(torch.count_nonzero(left).item() == 0)
    else:
        cosine = torch.nn.functional.cosine_similarity(
            left.flatten(), right.flatten(), dim=0
        ).item()
    passed = torch.equal(actual, expected) if exact else (
        nrmse <= 0.01 and cosine >= 0.999
    )
    checks.append({
        "name": name,
        "passed": passed,
        "exact_required": exact,
        "shape": list(actual.shape),
        "max_abs": (left - right).abs().max().item(),
        "nrmse": nrmse,
        "cosine": cosine,
    })


def unpack_int32_layout(weight):
    packed = weight.values.T.contiguous().view(torch.int32).to(torch.int64)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    nibbles = (packed[..., None] >> shifts) & 15
    signed = ((nibbles ^ 8) - 8).to(torch.int8)
    return signed.reshape(weight.in_features, weight.out_features).T.contiguous()


def check_rounding_arithmetic(checks):
    codes = torch.arange(-128, 128, dtype=torch.int32)
    high = torch.div(codes, 16, rounding_mode="floor")
    low = torch.bitwise_and(codes, 15) - 8
    compare("msd.int8_decomposition", 16 * high + low + 8, codes, checks)

    # This exact binary scale exposes rounding hidden by the signed-dot law.
    integer = -torch.ones(1, 128, dtype=torch.int32)
    integer[0, -1] = 0
    scale = torch.tensor([23 * 2**-24])
    correction = torch.tensor([-23368 * 2**-24])
    actual = w4_msd_product(
        torch.zeros(1, 128, dtype=torch.int32), integer, scale, correction,
        torch.ones(1, 1),
    )
    expected = torch.tensor([[-2**-21]], dtype=torch.bfloat16)
    compare("msd.fp16_partial_rounding", actual, expected, checks)

    x = torch.tensor([[211 * 2**-16, 211 * 2**-17, -211 * 2**-17]],
                     dtype=torch.bfloat16)
    expected_codes = torch.tensor([[127, 64, -64]], dtype=torch.int8)
    expected_scale = torch.tensor([2.5351217118441127e-05])
    for name, quantize in (("reader", quantize_a8),
                           ("operator", quantize_a8_reference)):
        codes, scale = quantize(x)
        compare(f"{name}.a8_half_integer", codes, expected_codes, checks)
        compare(f"{name}.a8_scale", scale, expected_scale, checks)


def check_checkpoint(checkpoint, layer, expert, checks):
    prefix = f"model.layers.{layer}.mlp.experts.{expert}"
    projections = ("gate_proj", "up_proj", "down_proj")
    whole = {name: checkpoint.linear(prefix + "." + name) for name in projections}
    decoded = {name: weight.unpack() for name, weight in whole.items()}
    for name, weight in whole.items():
        compare(name + ".int32_packing", unpack_int32_layout(weight), decoded[name], checks)
        bias = 8 * decoded[name].sum(dim=1).float() * weight.scale
        compare(name + ".scale_bias", weight.scale_bias, bias, checks)

    for rank in range(16):
        shard = checkpoint.expert(layer, expert, rank)
        channels = slice(rank * 128, (rank + 1) * 128)
        for name, weight in zip(projections, (shard.gate, shard.up, shard.down)):
            expected = (
                decoded[name][:, channels] if name == "down_proj"
                else decoded[name][channels]
            )
            label = f"rank{rank}.{name}"
            compare(label + ".weights", weight.unpack(), expected, checks)
            bias = 8 * expected.sum(dim=1).float() * weight.scale
            compare(label + ".scale_bias", weight.scale_bias, bias, checks)
    return {
        name: {
            "packed_shape": list(weight.values.shape),
            "logical_shape": [weight.out_features, weight.in_features],
            "scale_bias_shape": list(
                checkpoint.shape(prefix + "." + name + ".scale_bias")
            ),
            "sha256": hashlib.sha256(weight.values.numpy().tobytes()).hexdigest(),
        }
        for name, weight in whole.items()
    }


def check_projection(name, source, device, generator, checks):
    import torch_npu

    operation = NpuQuantizedLinear.from_checkpoint(source, device)
    expected_format = 2 if source.bits == 4 else 29
    checks.append({
        "name": name + ".weight_format",
        "passed": torch_npu.get_npu_format(operation.weight) == expected_format,
        "format": torch_npu.get_npu_format(operation.weight),
        "expected_format": expected_format,
    })
    for rows in (1, 4):
        x = torch.randn(rows, source.in_features, generator=generator).to(torch.bfloat16)
        activation, scale = torch_npu.npu_dynamic_quant(x.to(device), dst_type=torch.int8)
        expected_activation, expected_scale = quantize_a8_reference(x)
        label = f"{name}.M{rows}"
        compare(label + ".activation", activation.cpu(), expected_activation, checks)
        compare(
            label + ".activation_scale", scale.cpu(), expected_scale, checks,
            exact=False,
        )
        original_activation = activation.cpu()
        original_scale = scale.cpu()
        expected = source.reference(original_activation, original_scale).to(torch.bfloat16)
        actual = operation.apply_quantized(activation, scale).cpu()
        compare(label + ".activation_preserved", activation.cpu(), original_activation, checks)
        compare(label + ".projection", actual, expected, checks, exact=False)
        if source.bits == 4:
            msd = w4_msd_product(
                original_activation.int(), source.unpack().int(), source.scale,
                source.scale_bias, original_scale[:, None],
            )
            compare(label + ".msd_projection", actual, msd, checks)

    zeros = torch.zeros(1, source.in_features, dtype=torch.bfloat16, device=device)
    actual = operation(zeros).cpu()
    compare(name + ".zero", actual, torch.zeros_like(actual), checks)


def check_npu(checkpoint, layer, expert, device, checks):
    import torch_npu  # Registers NPU with TORCH_DEVICE_BACKEND_AUTOLOAD=0.

    torch.npu.set_device(device)
    torch.npu.reset_peak_memory_stats()
    generator = torch.Generator().manual_seed(1234)
    for rank in (0, 15):
        shard = checkpoint.expert(layer, expert, rank)
        for name, weight in (("gate", shard.gate), ("up", shard.up), ("down", shard.down)):
            check_projection(f"rank{rank}.{name}", weight, device, generator, checks)
    source = checkpoint.linear("model.layers.0.self_attn.q_a_proj")
    check_projection("q_a_proj", source, device, generator, checks)
    torch.npu.synchronize()
    return {
        "torch_npu": torch_npu.__version__,
        "peak_tensor_bytes": torch.npu.max_memory_allocated(),
        "peak_allocator_bytes": torch.npu.max_memory_reserved(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "npu:0"), default="cpu")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=0)
    args = parser.parse_args()
    checkpoint = ModelSlimCheckpoint(args.checkpoint)
    checks = []
    check_rounding_arithmetic(checks)
    weights = check_checkpoint(checkpoint, args.layer, args.expert, checks)
    npu = None
    if args.device != "cpu":
        npu = check_npu(checkpoint, args.layer, args.expert, args.device, checks)
    failed = sum(not check["passed"] for check in checks)
    root = Path(__file__).resolve().parents[3]
    sources = [
        Path(__file__), Path(__file__).with_name("weights.py"),
        Path(__file__).with_name("reference_weights.py"),
        Path(__file__).with_name("ops") / "quantized.py",
    ]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if failed == 0 else "fail",
        "checkpoint": str(args.checkpoint),
        "layer": args.layer,
        "expert": args.expert,
        "device": args.device,
        "hostname": platform.node(),
        "command": [sys.executable, *sys.argv],
        "git_head": args.source_commit,
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "weights": weights,
        "npu": npu,
        "checks_total": len(checks),
        "checks_failed": failed,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        key: report[key] for key in ("status", "checks_total", "checks_failed")
    }))
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
