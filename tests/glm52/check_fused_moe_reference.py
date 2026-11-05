"""Compare the CPU fused-MoE reference with frozen CANN expert outputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from tilert.models.glm_5_2_ascend.reference_weights import (  # noqa: E402
    ReferenceWeights,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected):
    left, right = actual.float().flatten(), expected.float().flatten()
    nrmse = float((left - right).norm() / right.norm().clamp_min(1e-30))
    cosine = float(torch.nn.functional.cosine_similarity(left, right, dim=0))
    return {"passed": nrmse <= 0.01 and cosine >= 0.999,
            "exact": torch.equal(actual, expected),
            "unequal": int((actual != expected).sum()),
            "nrmse": nrmse, "cosine": cosine}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--rank", type=int, default=13)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    captured = json.loads(args.capture.with_suffix(".json").read_text())
    assert digest(args.capture) == captured["capture_sha256"]
    weights = ReferenceWeights(args.checkpoint)
    assert weights.contract["metadata_sha256"] == captured["checkpoint_sha256"]
    rows = torch.load(args.capture, map_location="cpu", weights_only=True)
    assert len(rows) >= 2
    checks = []
    for position, row in enumerate(rows[:2]):
        experts = (row["groups"] != 0).nonzero().flatten().tolist()
        assert len(experts) == 8
        assert all(int(row["groups"][expert]) == 1 for expert in experts)
        partials = {
            expert: weights.fused_expert_partials(
                f"model.layers.{args.layer}.mlp.experts.{expert}",
                row["input"], ranks=(args.rank,),
            )[0] for expert in experts
        }
        down = torch.cat([partials[expert] for expert in experts])
        combined = torch.stack([
            partials[expert].float() * probability.float()
            for expert, probability in zip(row["ids"][0].tolist(),
                                           row["probs_bf16"][0])
        ]).sum(0).bfloat16()
        checks.append({"position": position,
                       "down": compare(row["down"], down),
                       "combine": compare(row["combine"], combined)})
    arithmetic = ROOT / "tilert/models/glm_5_2_ascend/reference_weights.py"
    report = {
        "scope": "Two frozen routed-MoE rows; CPU reference versus CANN",
        "status": "pass" if all(c[name]["passed"] for c in checks
                                  for name in ("down", "combine")) else "fail",
        "capture_sha256": digest(args.capture),
        "capture_report_sha256": digest(args.capture.with_suffix(".json")),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (
            Path(__file__), arithmetic,
        )},
        "checkpoint_sha256": weights.contract["metadata_sha256"],
        "layer": args.layer, "rank": args.rank, "checks": checks,
    }
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    raise SystemExit(report["status"] != "pass")


if __name__ == "__main__":
    main()
