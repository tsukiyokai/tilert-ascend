import argparse
import hashlib
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
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    torch.npu.matmul.allow_hf32 = False
    cases = []
    for seed in (42, 43, 44):
        generator = torch.Generator().manual_seed(seed)
        hidden = (torch.randn((32, 6144), generator=generator) * .1).bfloat16()
        weight = (torch.randn((160, 6144), generator=generator) * .05).bfloat16()
        x = hidden.to(args.device)
        nk = weight.to(args.device)
        kn = weight.T.contiguous().to(args.device)
        nz = torch_npu.npu_format_cast(weight.to(args.device), 29)
        outputs = {
            "mm_kn": bits(torch.mm(x, kn)),
            "linear_nk": bits(torch.nn.functional.linear(x, nk)),
            "mm_nk_view": bits(torch.mm(x, nk.T)),
            "linear_nz": bits(torch.nn.functional.linear(x, nz)),
        }
        reference = outputs["linear_nk"]
        comparisons = {}
        for name, value in outputs.items():
            coordinates = (value != reference).nonzero()
            comparisons[name] = {
                "different": len(coordinates),
                "coordinates": coordinates[:8].tolist(),
                "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            }
        cases.append({"seed": seed, "comparisons": comparisons,
                      "formats": {"nk": torch_npu.get_npu_format(nk),
                                  "kn": torch_npu.get_npu_format(kn),
                                  "nz": torch_npu.get_npu_format(nz)}})
    report = {"status": "measured", "shape": [32, 160, 6144],
              "scope": "BF16 WK weight layout differential; no model TPOT",
              "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
