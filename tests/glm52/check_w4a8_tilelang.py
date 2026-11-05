"""Validate A3 packed-W4 GEMV against integer dots on real TP16 weights."""

import argparse
import hashlib
import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_npu

from tilert.models.glm_5_2_ascend.check_weights import compare
from tilert.models.glm_5_2_ascend.ops.quantized import quantize_a8_reference
from tilert.models.glm_5_2_ascend.ops.w4a8 import w4a8_gemv
from tilert.models.glm_5_2_ascend.weights import ModelSlimCheckpoint, QuantizedWeight


def merge_gate_up(source):
    return QuantizedWeight(
        torch.cat((source.gate.values, source.up.values)),
        torch.cat((source.gate.scale, source.up.scale)),
        4,
    )


def check_projection(weight, label, generator, checks):
    values = torch.randn(1, weight.in_features, generator=generator).to(torch.bfloat16)
    quantized, token_scale = quantize_a8_reference(values)
    packed = weight.values.npu()
    patterns = {
        "random": quantized,
        "extreme": torch.full_like(quantized, -127),
        "zero": torch.zeros_like(quantized),
    }
    for pattern, activation in patterns.items():
        for scale_name in ("checkpoint", "unit"):
            scale = weight.scale if scale_name == "checkpoint" else torch.ones_like(weight.scale)
            per_token = token_scale if scale_name == "checkpoint" else torch.ones_like(token_scale)
            reference_weight = QuantizedWeight(weight.values, scale, 4)
            expected = reference_weight.reference(activation, per_token).to(torch.bfloat16)
            x, channel_scale, row_scale = (
                activation.npu(), scale.npu(), per_token.npu()
            )
            output = torch.empty(
                (1, weight.out_features), dtype=torch.bfloat16, device="npu:0"
            )
            result = w4a8_gemv(x, packed, channel_scale, row_scale, out=output)
            name = f"{label}.{pattern}.{scale_name}"
            compare(
                name + ".projection", result.cpu(), expected, checks,
                exact=scale_name == "unit",
            )
            compare(name + ".input", x.cpu(), activation, checks)
            compare(name + ".weight", packed.cpu(), weight.values, checks)
            checks.append({
                "name": name + ".output_storage",
                "passed": result.data_ptr() == output.data_ptr(),
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    model = root / "tilert/models/glm_5_2_ascend"
    sources = [
        Path(__file__), model / "weights.py", model / "ops/w4a8.py",
        model / "ops/_w4a8_tilelang.py", model / "ops/quantized.py",
    ]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "command": [sys.executable, *sys.argv],
        "git_head": args.source_commit,
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "layer": 3, "expert": 0, "tp_size": 16, "ranks": [0, 15],
        "rows": 1, "seed": 1234,
        "checks": [], "status": "error",
        "scope": "single projection correctness; no performance timing",
    }
    try:
        torch.set_num_threads(4)
        torch.npu.set_device(0)
        torch.npu.reset_peak_memory_stats()
        checkpoint = ModelSlimCheckpoint(args.checkpoint)
        generator = torch.Generator().manual_seed(1234)
        with torch.inference_mode():
            for rank in (0, 15):
                source = checkpoint.expert(3, 0, rank)
                check_projection(
                    merge_gate_up(source), f"rank{rank}.gate_up",
                    generator, report["checks"],
                )
                check_projection(
                    source.down, f"rank{rank}.down",
                    generator, report["checks"],
                )
        torch.npu.synchronize()
        report["peak_tensor_bytes"] = torch.npu.max_memory_allocated()
        report["peak_allocator_bytes"] = torch.npu.max_memory_reserved()
        report["status"] = (
            "pass" if all(check["passed"] for check in report["checks"]) else "fail"
        )
    except Exception:
        report["error"] = traceback.format_exc()
    report["checks_total"] = len(report["checks"])
    report["checks_failed"] = sum(not item["passed"] for item in report["checks"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        key: report[key] for key in ("status", "checks_total", "checks_failed")
    }))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
