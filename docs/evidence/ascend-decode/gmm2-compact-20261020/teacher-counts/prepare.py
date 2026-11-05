"""Freeze five GMM2 count cases and check the existing tile address formulas."""

import hashlib
import json
import os
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch

torch.set_num_threads(1)
ROOT = Path("/workspace/experiments/gmm2-compact-20261020")
OUT = ROOT / "teacher-counts-screen"
ROUTES = Path("/workspace/experiments/teacher-input-router-20261020/teacher-routes.pt")
OUT.mkdir(exist_ok=True)
data = torch.load(ROUTES, map_location="cpu", weights_only=True)
assert data["routes"].shape == (75, 32, 8)
all_counts = {
    layer: torch.bincount(route.flatten().long(), minlength=256).tolist()
    for layer, route in zip(data["layers"], data["routes"])
}
cases = []
for layer in (29, 60, 21, 17):
    cases.append({"name": f"teacher-layer-{layer}", "layer": layer,
                  "counts": all_counts[layer]})
cases.append({"name": "shared8", "layer": None,
              "counts": [32 if expert in range(0, 256, 32) else 0
                         for expert in range(256)]})


def geometry(counts, base_m, base_n):
    intervals = [[] for _ in range(256)]
    tasks = 0
    offset = 0
    weight_bytes = 0
    for expert, count in enumerate(counts):
        assert 0 <= count <= 32
        for m_index in range((2 * count + base_m - 1) // base_m):
            first = offset + m_index * base_m // 2
            valid = min(base_m // 2, offset + count - first)
            assert 0 < valid <= base_m // 2
            for n_start in range(0, 6144, base_n):
                tasks += 1
                weight_start = (expert * 6144 + n_start) * 128 // 2
                length = base_n * 128 // 2
                assert 0 <= weight_start < weight_start + length <= 256 * 393216
                weight_bytes += length
                for row in range(first, first + valid):
                    intervals[row].append((n_start, n_start + base_n))
        offset += count
    assert offset == 256
    for row in intervals:
        previous = 0
        for start, end in sorted(row):
            assert start == previous
            previous = end
        assert previous == 6144
    return {"tasks": tasks, "logical_weight_bytes": weight_bytes}


for case in cases:
    counts = case["counts"]
    assert len(counts) == 256 and sum(counts) == 256
    case.update(active=sum(c > 0 for c in counts), hot=sum(c > 16 for c in counts),
                maximum=max(counts), compact=geometry(counts, 64, 512),
                wide=geometry(counts, 32, 1024))
fallback = [c for c in all_counts.values() if max(c) > 16]
active = sum(sum(c > 0 for c in counts) for counts in all_counts.values())
fallback_active = sum(sum(c > 0 for c in counts) for counts in fallback)
hot = sum(sum(c > 16 for c in counts) for counts in fallback)
assert (len(fallback), fallback_active, hot, active) == (35, 3198, 49, 7452)
result = {"scope": data["scope"], "npu_launched": False,
          "routes_sha256": hashlib.sha256(ROUTES.read_bytes()).hexdigest(),
          "cases": cases, "fallback_layers": len(fallback),
          "extra_logical_weight_bytes": hot * 393216,
          "extra_fraction_of_fallback_weight": hot / fallback_active,
          "extra_fraction_of_all75_weight": hot / active}
(OUT / "counts.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "cases"}))
for case in cases:
    print(json.dumps({k: v for k, v in case.items() if k != "counts"}))
