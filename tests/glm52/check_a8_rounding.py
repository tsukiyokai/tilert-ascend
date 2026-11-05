"""Compare operation order in CANN dynamic quantization on fixed BF16 rows."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(1234)
    torch.randn(1, 6144, generator=generator)
    x = torch.randn(4, 6144, generator=generator).to(torch.bfloat16)
    observed, npu_scale = torch_npu.npu_dynamic_quant(x.npu(), dst_type=torch.int8)
    observed = observed.cpu()
    values = x.float()
    maximum = values.abs().amax(dim=-1, keepdim=True)
    scale = maximum / 127.0
    candidates = {
        "divide_scale": values / scale,
        "multiply_reciprocal": values * (1.0 / scale),
        "multiply_127_over_max": values * (127.0 / maximum),
        "divide_max_multiply_127": values / maximum * 127.0,
    }
    report = {
        "shape": list(x.shape),
        "seed": 1234,
        "scale": scale.flatten().tolist(),
        "npu_scale": npu_scale.cpu().flatten().tolist(),
        "candidates": {},
    }
    for name, candidate in candidates.items():
        expected = candidate.round().clamp(-127, 127).to(torch.int8)
        indices = (expected != observed).nonzero()
        report["candidates"][name] = {
            "mismatches": len(indices),
            "examples": [
                {
                    "row": row, "column": column,
                    "x": x[row, column].item(),
                    "before_round": candidate[row, column].item(),
                    "expected": expected[row, column].item(),
                    "observed": observed[row, column].item(),
                }
                for row, column in indices[:16].tolist()
            ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
