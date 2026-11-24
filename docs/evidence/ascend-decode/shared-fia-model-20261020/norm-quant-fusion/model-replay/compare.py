"""Measure propagation of the three dense norm/quant substitutions."""

import hashlib
import json
import os
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch

torch.set_num_threads(1)
root = Path(__file__).resolve().parent
control = root.parents[1] / "two-rounds/full-78/output/rank-0"
candidate = root / "on-full-78-02/output/rank-0"
rows = [0, 1, 2, 8]


def metrics(actual, expected):
    a, b = actual.double().reshape(-1), expected.double().reshape(-1)
    return {
        "nrmse": float(torch.linalg.vector_norm(a - b)
                       / torch.linalg.vector_norm(b)),
        "max_abs": float((a - b).abs().max()),
        "cosine": float(torch.dot(a, b)
                        / (torch.linalg.vector_norm(a)
                           * torch.linalg.vector_norm(b))),
    }


curve = []
for layer in range(78):
    filename = f"layer-{layer:02d}.pt"
    a = torch.load(candidate / filename, map_location="cpu", weights_only=True)
    b = torch.load(control / filename, map_location="cpu", weights_only=True)
    curve.append({
        "layer": layer,
        **{field: metrics(a[field][rows], b[field][rows])
           for field in ("hidden", "residual", "merged")},
    })
a = torch.load(candidate / "final.pt", map_location="cpu", weights_only=True)
b = torch.load(control / "final.pt", map_location="cpu", weights_only=True)
result = {
    "scope": "Rank0, trusted rows only, same eager target and frozen rounds",
    "control": str(control), "candidate": str(candidate), "rows": rows,
    "control_predictions": [b["predicted_tokens"][i] for i in rows],
    "candidate_predictions": [a["predicted_tokens"][i] for i in rows],
    "logits": metrics(a["logits"][rows], b["logits"][rows]),
    "curve": curve,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(root / "on-full-78-02/propagation.json").write_text(
    json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "curve"}))
