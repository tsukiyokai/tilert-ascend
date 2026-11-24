"""Audit saved intervention outputs and the unchanged preparation control."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1] / "two-rounds"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return (left.shape == right.shape and left.dtype == right.dtype
                and torch.equal(left.contiguous().view(torch.uint8),
                                right.contiguous().view(torch.uint8)))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            equal(a, b) for a, b in zip(left, right)
        )
    return left == right


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("case")
parser.add_argument("--layers", type=int, required=True)
parser.add_argument("--compare-control", action="store_true")
args = parser.parse_args()
torch.set_num_threads(1)
root = ROOT / args.case
spec = importlib.util.spec_from_file_location("round_audit", BASE / "audit.py")
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)
status = auditor.main(SimpleNamespace(run=root / "output",
                                     output=root / "audit.json",
                                     layers=args.layers))
counts = {"bound_norm_artifacts": 0, "norm_calls": 0,
          "layer_control_comparisons": 0}
for rank in range(16):
    r = json.loads((root / f"output/norm-intervention-rank-{rank}.json")
                   .read_text())
    assert r["script_sha256"] == digest(ROOT / "replay_intervention.py")
    assert r["runner_sha256"] == digest(BASE / "source/tests/glm52/"
                                       "run_cann_layer_major.py")
    assert [v["layer"] for v in r["records"]] == list(range(min(3, args.layers)))
    for record in r["records"]:
        path = Path(record["path"])
        assert digest(path) == record["sha256"]
        values = torch.load(path, map_location="cpu", weights_only=True)
        assert len(values) == record["calls"] == 2
        counts["bound_norm_artifacts"] += 1
        counts["norm_calls"] += 2
    if args.compare_control:
        assert r["arm"] == "off"
        for layer in range(args.layers):
            suffix = f"output/rank-{rank}/layer-{layer:02d}.pt"
            left = torch.load(root / suffix, map_location="cpu", weights_only=True)
            right = torch.load(BASE / "full-78" / suffix, map_location="cpu",
                               weights_only=True)
            for name in ("input_hidden", "input_residual", "input_merged",
                         "hidden", "residual", "merged", "indices",
                         "round_calls"):
                assert equal(left[name], right[name]), (rank, layer, name)
                counts["layer_control_comparisons"] += 1
result = {"status": "pass", "target_audit_returncode": status,
          "counts": counts, "all_control_values_equal": args.compare_control,
          "auditor_sha256": digest(Path(__file__)),
          "runner_auditor_sha256": digest(BASE / "audit.py")}
(root / "intervention-audit.json").write_text(
    json.dumps(result, indent=2) + "\n",
)
print(json.dumps(result))
