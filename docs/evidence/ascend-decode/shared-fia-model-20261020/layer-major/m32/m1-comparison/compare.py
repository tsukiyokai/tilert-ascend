"""Compare frozen M1 and M32 layer artifacts on CPU, without a tolerance gate."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    assert bool(torch.isfinite(left).all() & torch.isfinite(right).all())
    different = left.contiguous().view(torch.uint8) != right.contiguous().view(torch.uint8)
    a, b = left.double().flatten(), right.double().flatten()
    delta, norm_a, norm_b = (a - b).norm().item(), a.norm().item(), b.norm().item()
    result = {
        "shape": list(left.shape), "dtype": str(left.dtype),
        "bitexact": not bool(different.any()),
        "different_bytes": int(different.sum()),
        "different_values": int((a != b).sum()),
        "nrmse": delta / norm_a if norm_a else (0.0 if norm_b == 0 else None),
        "cosine": a.dot(b).item() / (norm_a * norm_b) if norm_a * norm_b else float(norm_a == norm_b),
        "maxabs": (a - b).abs().max().item(),
    }
    changed = (a != b).nonzero()
    if changed.numel():
        offset = int(changed[0])
        result["firstdiff"] = {"flat_offset": offset, "row": offset // left.shape[-1],
                               "column": offset % left.shape[-1],
                               "m1": a[offset].item(), "m32": b[offset].item()}
    return result


def compare(args):
    roots = [args.m1.resolve(), args.m32.resolve()]
    reports = [json.loads((root / "rank-0.json").read_text()) for root in roots]
    assert [r["query_width"] for r in reports] == [1, 32]
    for report in reports:
        assert report["model_tp_reduction"] == "float32"
        assert report["steps"] == 32 and report["attention_kernel"] == "sfa"
    same = {name: reports[0][name] == reports[1][name]
            for name in ("checkpoint_sha256", "inputs", "teacher_tokens", "expected_tokens", "packages")}
    assert all(same.values()), same
    teacher_fields = ("sha256", "baseline_request_id", "prompt_file_sha256", "prompt_identity",
                      "teacher_tokens", "expected_next_tokens", "scope")
    assert all(reports[0]["teacher_source"][key] == reports[1]["teacher_source"][key]
               for key in teacher_fields)
    result = {"scope": "rank0, layers0..7, 32 frozen teacher rows; diagnostic cross-width comparison",
              "same_report_fields": same, "reports": [], "layers": []}
    for root, report in zip(roots, reports):
        result["reports"].append({"path": str(root / "rank-0.json"),
                                  "sha256": digest(root / "rank-0.json"),
                                  "query_width": report["query_width"],
                                  "capacity": report["capacity"],
                                  "environment": report["environment"]})
    previous = [None, None]
    for layer in range(8):
        data, paths = [], []
        for arm, (root, report) in enumerate(zip(roots, reports)):
            path = root / "rank-0" / f"layer-{layer:02d}.pt"
            sha = digest(path)
            assert sha == next(value for key, value in report["artifact_sha256"].items()
                               if key.endswith(f"/rank-0/layer-{layer:02d}.pt"))
            value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            assert value["layer"] == layer and value["rank"] == 0
            assert value["positions"] == list(range(8192, 8224))
            assert all(value[key] == report[key] for key in ("checkpoint_sha256", "inputs", "teacher_source", "teacher_tokens"))
            for field in ("hidden", "residual", "merged", "input_hidden", "input_merged"):
                assert value[field].shape == (32, 6144) and value[field].dtype == torch.bfloat16
            if layer:
                assert torch.equal(value["input_hidden"], previous[arm]["hidden"])
                assert torch.equal(value["input_residual"], previous[arm]["residual"])
            else:
                assert value["input_residual"] is None
            previous[arm] = value
            data.append(value)
            paths.append({"path": str(path), "sha256": sha})
        fields = ("input_hidden", "input_merged", "hidden", "residual", "merged")
        row = {"layer": layer, "artifacts": paths,
               "tensors": {key: metrics(data[0][key], data[1][key]) for key in fields},
               "cache_rows": {key: metrics(data[0]["cache_rows"][key], data[1]["cache_rows"][key])
                              for key in data[0]["cache_rows"]}}
        if layer:
            row["tensors"]["input_residual"] = metrics(data[0]["input_residual"], data[1]["input_residual"])
        left, right = [value["indices"] for value in data]
        assert left.shape == right.shape == (32, 2048)
        overlaps = [len(set(a) & set(b)) / 2048 for a, b in zip(left.tolist(), right.tolist())]
        row["indices"] = {"bitexact": torch.equal(left, right),
                          "same_set_every_row": torch.equal(left.sort(1).values, right.sort(1).values),
                          "overlap_per_row": overlaps, "mean_overlap": sum(overlaps) / 32,
                          "minimum_overlap": min(overlaps)}
        changed = (left != right).nonzero()
        row["indices"]["different_entries"] = len(changed)
        if len(changed):
            query, column = changed[0].tolist()
            row["indices"]["firstdiff"] = {
                "query": query, "column": column,
                "m1": int(left[query, column]), "m32": int(right[query, column]),
            }
        result["layers"].append(row)
        print(json.dumps({"layer": layer, **{key: row["tensors"][key] for key in ("hidden", "residual")},
                          "index_overlap": row["indices"]["mean_overlap"]}), flush=True)
    result["torch_npu_imported"] = "torch_npu" in sys.modules
    assert not result["torch_npu_imported"]
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("m1", "m32", "output"):
        parser.add_argument(name, type=Path)
    torch.set_num_threads(1)
    compare(parser.parse_args())
