import argparse
import json
from pathlib import Path

import torch
import torch_npu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    cases = []
    for rows, columns in ((1, 128), (32, 128), (32, 2048)):
        for exponent in (-1, 0, 1):
            values = torch.zeros(rows, columns, dtype=torch.bfloat16)
            values[:, 0] = 254 * 2.0 ** exponent
            values[:, 1:127] = torch.arange(-125, 127, 2) * 2.0 ** exponent
            codes, scales = torch_npu.npu_dynamic_quant(values.to(args.device))
            codes, scales = codes.cpu(), scales.cpu()
            expected = torch.round(values.float() / (2.0 ** (exponent + 1)))
            mask = codes != expected.to(torch.int8)
            cases.append({"shape": [rows, columns], "exponent": exponent,
                          "different_from_nearest_even": int(mask.sum()),
                          "coordinates": mask.nonzero()[:12].tolist(),
                          "actual": codes[0, :16].tolist(),
                          "nearest_even": expected[0, :16].tolist(),
                          "scale": float(scales[0])})
    report = {"status": "measured", "scope": "Exact-half INT8 quantization witness",
              "cases": cases}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
