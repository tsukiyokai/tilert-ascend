import argparse
import json
from pathlib import Path
import re

import torch


def difference(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    shape = (*actual.shape, actual.element_size())
    mask = (actual.contiguous().view(torch.uint8).reshape(shape)
            != expected.contiguous().view(torch.uint8).reshape(shape)).any(dim=-1)
    delta = actual.float() - expected.float()
    coordinates = mask.nonzero()[:8].tolist()
    return {
        "different": int(mask.sum()),
        "per_row": mask.reshape(mask.shape[0], -1).sum(dim=1).tolist(),
        "max_abs": float(delta.abs().max()),
        "nrmse": float(delta.norm() / expected.float().norm()),
        "coordinates": coordinates,
        "actual": [float(actual[tuple(index)]) for index in coordinates],
        "expected": [float(expected[tuple(index)]) for index in coordinates],
    }


def cache_rows(value, metadata, length):
    table = metadata["block_table"].reshape(-1)[:(length + 127) // 128].long()
    return value[table].reshape(-1, *value.shape[2:])[:length]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, candidate = [torch.load(path, map_location="cpu", weights_only=True)
                           for path in (args.baseline, args.candidate)]
    assert baseline["position"] == candidate["position"]
    assert baseline["active_rows"] == candidate["active_rows"]
    rows = baseline["active_rows"]
    assert baseline["frames"].keys() == candidate["frames"].keys()
    layers = {}
    order = sorted(baseline["frames"], key=lambda name:
                   int(re.search(r"(?:^|\.)layers\.(\d+)\.", name)[1]))
    for name in order:
        layers[name] = {field: difference(candidate["frames"][name][field][:rows],
                                          baseline["frames"][name][field][:rows])
                        for field in ("input", "output")}
    first = next((name for name in order
                  if not layers[name]["input"]["different"]
                  and layers[name]["output"]["different"]), None)
    report = {"position": baseline["position"], "active_rows": rows, "layers": layers,
              "first_equal_input_different_output": first,
              "scope": "Actual TP attention boundaries; diagnostic copies, no TPOT"}
    left, right = candidate["detail"], baseline["detail"]
    if left["layer"] == right["layer"]:
        report["detail_layer"] = left["layer"]
        a, b = left.get("phases", {}), right.get("phases", {})
        report["phases"] = {name: difference(a[name][:rows], b[name][:rows])
                            for name in sorted(a.keys() & b.keys())}
        report["phase_missing"] = {
            "baseline": right.get("phase_missing", []),
            "candidate": left.get("phase_missing", [])}
        length = baseline["position"] + rows
        report["caches"] = {
            field: [difference(cache_rows(x, left[metadata], length),
                               cache_rows(y, right[metadata], length))
                    for x, y in zip(left[field], right[field])]
            for field, metadata in (("main_cache", "main_metadata"),
                                    ("index_cache", "index_metadata"))
            if (left.get(field) and right.get(field) and
                left.get(metadata) and right.get(metadata))}
        if "topk" in a and "topk" in b:
            report["topk_sets"] = []
            for row in range(rows):
                x, y = [set(value[row].reshape(-1).tolist())
                        for value in (a["topk"], b["topk"])]
                report["topk_sets"].append({
                    "row": row, "candidate_only": sorted(x - y),
                    "baseline_only": sorted(y - x)})
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"position": report["position"], "layers": len(layers),
                      "first_equal_input_different_output": first}))


if __name__ == "__main__":
    main()
