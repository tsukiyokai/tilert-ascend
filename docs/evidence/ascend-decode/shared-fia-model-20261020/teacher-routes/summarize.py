"""Summarize current-kernel M1 teacher-route prior without device calls."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def distribution(values):
    data = torch.tensor(values, dtype=torch.float64)
    return {"min": int(data.min()), "mean": float(data.mean()),
            "p50": float(data.quantile(0.5)), "p95": float(data.quantile(0.95)),
            "max": int(data.max()),
            "histogram": {str(value): values.count(value) for value in sorted(set(values))}}


def main():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    torch.set_num_threads(1)
    receipt = json.loads((ROOT / "routes-result.json").read_text())
    norm = json.loads((ROOT / "norm-result.json").read_text())
    gate = json.loads((ROOT / "cpu-gate-result.json").read_text())
    assert receipt["status"] == norm["status"] == gate["status"] == "pass"
    assert receipt["binding"] == norm["binding"] == gate["binding"]
    assert receipt["binding"]["script_sha256"] == sha(ROOT / "collect.py")
    data = torch.load(ROOT / "teacher-routes.pt", weights_only=True, map_location="cpu")
    raw, logits = data["routes"], data["router_logits"]
    assert data["layers"] == list(range(3, 78))
    assert raw.shape == (75, 32, 8) and raw.dtype == torch.int32
    assert logits.shape == (75, 32, 256) and logits.dtype == torch.float32
    assert torch.isfinite(logits).all() and ((raw >= 0) & (raw < 256)).all()
    assert (raw.sort(dim=2).values.diff(dim=2) > 0).all()
    unions = json.loads((ROOT / "teacher-unions.json").read_text())
    widths = {}
    for width in (8, 16, 32):
        sizes, maxima, misses = [], [], []
        records = iter(unions["widths"][str(width)])
        for index, layer in enumerate(data["layers"]):
            for start in range(0, 32, width):
                counts = torch.bincount(raw[index, start:start + width].flatten().long(), minlength=256)
                record = next(records)
                size, maximum = int((counts > 0).sum()), int(counts.max())
                assert record["layer"] == layer and record["teacher_start"] == start
                assert record["expert_row_counts"] == counts.tolist()
                assert record["union"] == size and record["max_expert_rows"] == maximum
                assert int(counts.sum()) == 8 * width and maximum <= width
                sizes.append(size)
                maxima.append(maximum)
                if maximum > 16:
                    misses.append({"layer": layer, "teacher_start": start,
                                   "max_expert_rows": maximum})
        assert next(records, None) is None
        widths[str(width)] = {
            "layer_windows": len(sizes), "union": distribution(sizes),
            "max_expert_rows": distribution(maxima),
            "gmm2_count_predicate_le16": {"hits": len(sizes) - len(misses),
                                         "total": len(sizes),
                                         "rate": 1 - len(misses) / len(sizes)},
            "gmm2_count_predicate_misses": misses,
        }
    result = {"status": "pass", "cpu_only_summary": True,
              "scope": "Frozen residual reconstructed with current same-shape M1 kernels: teacher-route prior, not historical original routes, live M32 routes, or rejected lookup tails.",
              "gmm2_scope": "Count predicate only; actual adaptive dispatch was not measured.",
              "historical_native_identity_verified": False, "widths": widths,
              "input_sha256": {name: sha(ROOT / name) for name in (
                  "collect.py", "routes-result.json", "norm-result.json",
                  "teacher-routes.pt", "teacher-unions.json", "cpu-gate-result.json")},
              "summary_source_sha256": sha(Path(__file__))}
    (ROOT / "teacher-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({width: {"union_mean": info["union"]["mean"],
                              "max_rows_mean": info["max_expert_rows"]["mean"],
                              "le16": info["gmm2_count_predicate_le16"]}
                      for width, info in widths.items()}))


if __name__ == "__main__":
    main()
