"""Check one real TP16 expert partition against CPU integer arithmetic."""

import argparse
import hashlib
import json
import platform
import sys
import traceback
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as functional

from tilert.models.glm_5_2_ascend.ops.expert import NpuExpert
from tilert.models.glm_5_2_ascend.ops.quantized import quantize_a8_reference
from tilert.models.glm_5_2_ascend.weights import ModelSlimCheckpoint


def integer_projection(weight, activation, scale):
    """Decode raw nibbles independently of the production packing helper."""
    packed = weight.values.to(torch.int32) & 255
    nibbles = torch.stack((packed & 15, packed >> 4), dim=1)
    signed = torch.where(nibbles >= 8, nibbles - 16, nibbles)
    matrix = signed.reshape(weight.out_features, weight.in_features)
    product = activation.to(torch.int32) @ matrix.T
    return (product.float() * weight.scale * scale[:, None]).to(torch.bfloat16)


def reference(source, x):
    activation, scale = quantize_a8_reference(x)
    gate = integer_projection(source.gate, activation, scale)
    up = integer_projection(source.up, activation, scale)
    silu = functional.silu(gate.float()).to(torch.bfloat16)
    intermediate = (silu.float() * up.float()).to(torch.bfloat16)
    down_activation, down_scale = quantize_a8_reference(intermediate)
    output = integer_projection(source.down, down_activation, down_scale)
    return dict(
        input_quantized=activation, input_scale=scale, gate=gate, up=up,
        silu=silu, intermediate=intermediate,
        down_quantized=down_activation, down_scale=down_scale, output=output,
    )


def compare(checks, name, actual, expected, mode="numerical"):
    result = {
        "name": name, "mode": mode, "passed": False,
        "shape": list(actual.shape), "expected_shape": list(expected.shape),
        "dtype": str(actual.dtype), "expected_dtype": str(expected.dtype),
    }
    checks.append(result)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return
    left, right = actual.float(), expected.float()
    if not (torch.isfinite(left).all() and torch.isfinite(right).all()):
        result["nonfinite"] = True
        return
    error = (left - right).norm().item()
    norm = right.norm().item()
    nrmse = error / max(norm, 1e-12)
    cosine = (
        functional.cosine_similarity(
            left.flatten(), right.flatten(), dim=0
        ).item() if norm else float(torch.count_nonzero(left).item() == 0)
    )
    result.update(
        max_abs=(left - right).abs().max().item(), nrmse=nrmse,
        cosine=cosine, unequal=int(torch.count_nonzero(actual != expected)),
    )
    if mode == "exact":
        result["passed"] = torch.equal(actual, expected)
    elif mode == "scale":
        result["passed"] = torch.allclose(
            actual, expected, rtol=1e-6, atol=1e-8
        )
    else:
        result["passed"] = nrmse <= 0.01 and cosine >= 0.999


def check_case(operation, source, x, device, label, checks):
    expected = reference(source, x.clone())
    original = x.clone()
    device_input = x.to(device)
    trace = operation.trace(device_input)
    actual = {
        field.name: getattr(trace, field.name).detach().cpu().clone()
        for field in fields(trace)
    }
    compare(checks, label + ".input_preserved", device_input.cpu(), original,
            "exact")
    zero_rows = original.abs().amax(dim=1) == 0
    if zero_rows.any():
        output = actual["output"][zero_rows]
        compare(checks, label + ".zero_output", output,
                torch.zeros_like(output), "exact")
    for name in ("input_quantized", "input_scale", "gate", "up", "silu",
                 "intermediate", "down_quantized", "down_scale", "output"):
        mode = "numerical"
        if name == "input_quantized":
            mode = "exact"
        elif name == "input_scale":
            mode = "scale"
        compare(checks, label + ".chain." + name,
                actual[name], expected[name], mode)

    activation, scale = quantize_a8_reference(actual["intermediate"])
    compare(checks, label + ".local.down_input_preserved",
            actual["down_quantized"], activation, "exact")
    compare(checks, label + ".local.down_scale",
            actual["down_scale"], scale, "scale")
    silu = functional.silu(actual["gate"].float()).to(torch.bfloat16)
    compare(checks, label + ".local.silu", actual["silu"], silu)
    intermediate = (
        actual["silu"].float() * actual["up"].float()
    ).to(torch.bfloat16)
    compare(checks, label + ".local.multiply", actual["intermediate"],
            intermediate, "exact")
    for name, weight, qname, sname in (
        ("gate", source.gate, "input_quantized", "input_scale"),
        ("up", source.up, "input_quantized", "input_scale"),
        ("output", source.down, "down_quantized", "down_scale"),
    ):
        projected = integer_projection(weight, actual[qname], actual[sname])
        compare(checks, label + ".local." + name, actual[name], projected)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, report):
    import torch_npu

    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    checkpoint = ModelSlimCheckpoint(args.checkpoint)
    generator = torch.Generator().manual_seed(args.seed)
    inputs = {
        rows: torch.randn(rows, 6144, generator=generator).to(torch.bfloat16)
        for rows in (1, 4)
    }
    inputs[4][-1].zero_()
    report["versions"]["torch_npu"] = torch_npu.__version__
    report["weight_sha256"] = {}
    for rank in (0, 15):
        source = checkpoint.expert(args.layer, args.expert, rank)
        report["weight_sha256"][str(rank)] = {
            name: {
                part: hashlib.sha256(
                    getattr(weight, part).numpy().tobytes()
                ).hexdigest()
                for part in ("values", "scale", "scale_bias")
            }
            for name, weight in (
                ("gate", source.gate), ("up", source.up), ("down", source.down)
            )
        }
        operation = NpuExpert.from_checkpoint(source, args.device)
        for rows, x in inputs.items():
            check_case(operation, source, x, args.device,
                       f"rank{rank}.M{rows}", report["checks"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    model = root / "tilert/models/glm_5_2_ascend"
    sources = [Path(__file__), model / "weights.py",
               model / "ops/quantized.py", model / "ops/expert.py"]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "error", "checkpoint": str(args.checkpoint),
        "layer": args.layer, "expert": args.expert, "tp_size": 16,
        "ranks": [0, 15], "rows": [1, 4], "device": args.device,
        "seed": args.seed, "hostname": platform.node(),
        "command": [sys.executable, *sys.argv], "git_head": args.source_commit,
        "scope": "rank-local expert; no routing weight or TP reduction",
        "arithmetic": (
            "merged gate/up; BF16 projection, SiLU and product; A8; down"
        ),
        "reference": (
            "CPU integer dots with separate gate/up and BF16 boundaries"
        ),
        "zero_input_rows": {"M1": [], "M4": [3]},
        "thresholds": {"nrmse": 0.01, "cosine": 0.999,
                       "scale_rtol": 1e-6, "scale_atol": 1e-8},
        "versions": {"python": platform.python_version(),
                     "torch": torch.__version__},
        "source_sha256": {
            str(path.relative_to(root)): digest(path) for path in sources
        },
        "checks": [],
    }
    try:
        report["metadata_sha256"] = {
            name: digest(args.checkpoint / name) for name in (
                "config.json", "quant_model_description.json",
                "quant_model_weights.safetensors.index.json",
            )
        }
        with torch.inference_mode():
            run(args, report)
        report["status"] = (
            "pass" if all(item["passed"] for item in report["checks"])
            else "fail"
        )
    except Exception:
        report["error"] = traceback.format_exc()
    report["checks_total"] = len(report["checks"])
    report["checks_failed"] = sum(
        not item["passed"] for item in report["checks"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.write_text(payload)
    print(json.dumps({key: report[key] for key in (
        "status", "checks_total", "checks_failed"
    )}))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
