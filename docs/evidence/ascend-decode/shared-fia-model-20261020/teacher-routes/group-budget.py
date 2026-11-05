"""Count expert-local GMM2 tile geometry on the reconstructed M1 teacher prior."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import torch

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)
data = torch.load(ROOT / "teacher-routes.pt", weights_only=True, map_location="cpu")
routes = data["routes"]
assert routes.shape == (75, 32, 8) and routes.dtype == torch.int32
assert data["layers"] == list(range(3, 78))
assert ((routes >= 0) & (routes < 256)).all()
assert (routes.sort(dim=2).values.diff(dim=2) > 0).all()
rows = []
for layer, selected in zip(data["layers"], routes):
    counts = torch.bincount(selected.flatten().long(), minlength=256)
    active = counts[counts > 0].tolist()
    assert sum(active) == 256 and max(active) <= 32
    hot = [count for count in active if count > 16]
    old = sum(12 * ((count + 31) // 32) for count in active)
    per_expert = sum(6 if count <= 16 else 12 * ((count + 31) // 32)
                     for count in active)
    current = old if hot else 6 * len(active)
    rows.append({"layer": layer, "active_experts": len(active),
                 "hot_experts_gt16": len(hot), "hot_counts": sorted(hot),
                 "old_tiles": old, "current_global_guard_tiles": current,
                 "per_expert_guard_tiles": per_expert})
fallback = [row for row in rows if row["hot_experts_gt16"]]
assert len(fallback) == 35
total_active = sum(row["active_experts"] for row in fallback)
total_hot = sum(row["hot_experts_gt16"] for row in fallback)
old = sum(row["old_tiles"] for row in fallback)
candidate = sum(row["per_expert_guard_tiles"] for row in fallback)
current_all = sum(row["current_global_guard_tiles"] for row in rows)
candidate_all = sum(row["per_expert_guard_tiles"] for row in rows)
result = {"scope": "Current M1-kernel reconstructed teacher prior only; not original routes, live M32 routes, rejected lookup tails, or timing.",
          "formula": {"old": "12*ceil(c/32)",
                      "per_expert": "6 if 1<=c<=16 else 12*ceil(c/32)",
                      "counts_range": "0<=c<=32; only active c>0 counted"},
          "fallback_layers": len(fallback), "fallback_active_experts": total_active,
          "fallback_hot_experts_gt16": total_hot,
          "fallback_hot_expert_fraction": total_hot / total_active,
          "fallback_old_tiles": old, "fallback_per_expert_tiles": candidate,
          "fallback_tile_reduction_fraction": (old - candidate) / old,
          "all75_current_global_guard_tiles": current_all,
          "all75_per_expert_guard_tiles": candidate_all,
          "all75_tile_reduction_fraction": (current_all - candidate_all) / current_all,
          "fallback_layer_rows": fallback,
          "input_sha256": hashlib.sha256((ROOT / "teacher-routes.pt").read_bytes()).hexdigest(),
          "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(ROOT / "group-budget.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: value for key, value in result.items()
                  if key not in ("fallback_layer_rows", "scope", "formula")}))
