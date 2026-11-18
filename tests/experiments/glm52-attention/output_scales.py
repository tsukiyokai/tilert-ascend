import argparse
import json
from pathlib import Path

import torch
import torch_npu


def bits(value):
    return value.cpu().contiguous().view(torch.uint16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight-format", choices=("nd", "nz"), default="nd")
    parser.add_argument("--scale-dtypes", nargs="+", default=["bfloat16", "float32"],
                        choices=("bfloat16", "float32"))
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    cases = []
    for seed in (42, 43, 44):
        generator = torch.Generator().manual_seed(seed)
        x = torch.randint(-127, 128, (32, 2048), generator=generator,
                          dtype=torch.int8)
        weight = torch.randint(-127, 128, (2048, 6144), generator=generator,
                               dtype=torch.int8)
        token = torch.rand(32, generator=generator) / 64 + .01
        channel = (torch.rand(6144, generator=generator) + .25) / 128
        accumulator = x.int() @ weight.int()
        xn, wn, tn = (value.to(args.device) for value in (x, weight, token))
        if args.weight_format == "nz":
            wn = torch_npu.npu_format_cast(wn, 29)
        for name in args.scale_dtypes:
            dtype = getattr(torch, name)
            scale = channel.to(dtype)
            actual = bits(torch_npu.npu_quant_matmul(
                xn, wn, scale.to(args.device), pertoken_scale=tn,
                output_dtype=torch.bfloat16))
            references = {
                "channel_then_token": (accumulator.float() * scale.float())
                * token[:, None],
                "combined_scales": accumulator.float()
                * (scale.float()[None, :] * token[:, None]),
            }
            comparisons = {}
            for name, value in references.items():
                expected = bits(value.bfloat16())
                coordinates = (actual != expected).nonzero()
                comparisons[name] = {
                    "different": len(coordinates),
                    "coordinates": coordinates[:8].tolist(),
                    "actual": [int(actual[tuple(index)])
                               for index in coordinates[:8].tolist()],
                    "expected": [int(expected[tuple(index)])
                                 for index in coordinates[:8].tolist()],
                }
            cases.append({"seed": seed, "channel_dtype": str(dtype),
                          "weight_format": torch_npu.get_npu_format(wn),
                          "weight_storage_size": torch_npu.get_storage_size(wn),
                          "comparisons": comparisons})
    report = {"status": "measured", "shape": [32, 6144, 2048],
              "scope": "TP8 output quant matmul scale order; no model TPOT",
              "weight_format": args.weight_format,
              "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
