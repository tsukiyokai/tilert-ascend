"""Explain the saved separate norm/quant result using its BF16 boundaries."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(args.inputs, map_location="cpu", weights_only=True)
    inputs, records = saved["inputs"], []
    total = inputs["x"].float() + inputs["residual"].float()
    rounded_total = total.bfloat16().float()
    rstd = torch.rsqrt(rounded_total.square().mean(-1, keepdim=True) + 1e-5)
    normalized = (rounded_total * rstd).bfloat16().float()
    norm = (normalized * inputs["gamma"].float()
            + inputs["beta"].float()).bfloat16()
    scale = norm.float().abs().amax(-1, keepdim=True) / 127
    quant = (norm.float() / scale).round().to(torch.int8)
    for width, case in saved["outputs"].items():
        repeats = int(width)
        records.append({
            "width": repeats,
            "three_roundings_norm_differences": int((
                norm.repeat(repeats, 1) != case["separate_norm_bf16"]).sum()),
            "three_roundings_quant_differences": int((
                quant.repeat(repeats, 1) != case["separate_q"]).sum()),
            "final_rounding_only_norm_differences": int((
                case["cpu_fp32_norm_cast_bf16"]
                != case["separate_norm_bf16"]).sum()),
        })
    report = {"cases": records, "input_sha256": hashlib.sha256(
        args.inputs.read_bytes()).hexdigest(), "device_execution": False}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
