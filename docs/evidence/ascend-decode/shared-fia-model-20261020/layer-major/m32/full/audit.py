"""CPU audit of paired TP16 M32 layer-major artifacts; cross-arm errors are diagnostics."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import torch

QUOTA = 2 * 1024 ** 3
PRODUCERS = {0, 1, 2, *range(6, 78, 4)}
INT8 = set(range(6, 67, 4))


def require(ok, label):
    if not ok:
        raise ValueError(label)


@lru_cache(None)
def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gates(checks, required, label):
    names = {item["name"] for item in checks}
    require(len(names) == len(checks) and required <= names
            and all(item["passed"] is True for item in checks), label)


def reports(root, kernel, count, teacher, teacher_sha):
    summary = json.loads((root / "summary.json").read_text())
    require(summary["status"] == "pass" and len(summary["ranks"]) == 16
            and {x["rank"] for x in summary["ranks"]} == set(range(16)), f"{root}: ranks")
    rows = [json.loads((root / f"rank-{rank}.json").read_text()) for rank in range(16)]
    for rank, report in enumerate(rows):
        label, paths = f"{kernel}/rank{rank}", report["artifact_sha256"]
        receipt = next(x for x in summary["ranks"] if x["rank"] == rank)
        require(receipt["status"] == report["status"] == "pass"
                and receipt["report_sha256"] == digest(root / f"rank-{rank}.json"), label + ": receipt")
        require([report[x] for x in ("rank", "layers", "steps", "query_width", "capacity", "position")]
                == [rank, count, 32, 32, 9344, 8224]
                and report["attention_kernel"] == kernel and report["model_tp_reduction"] == "float32"
                and report["torch_quota_bytes"] == QUOTA, label + ": execution contract")
        require(report["inputs"]["initial_position"] == teacher["prompt_tokens"] == 8192
                and report["inputs"]["first_token_id"] == teacher["first_token_id"] == teacher["teacher_tokens"][0]
                and report["teacher_tokens"] == teacher["teacher_tokens"]
                and report["expected_tokens"] == teacher["expected_next_tokens"]
                and report["teacher_source"]["sha256"] == teacher_sha, label + ": teacher")
        require(report["full_model_executed"] is (count == 78)
                and report["predicted_tokens"] == (teacher["expected_next_tokens"] if count == 78 else [])
                and report["greedy_correct"] == report["greedy_total"] == (32 if count == 78 else 0), label + ": final labels")
        expected = {root / f"rank-{rank}" / f"layer-{n:02d}.pt" for n in range(count)}
        expected |= {root / f"rank-{rank}" / "final.pt"} if count == 78 else set()
        require({Path(p).resolve() for p in paths} == expected, label + ": artifact coverage")
        for path, sha in {**report["source_sha256"], **paths}.items():
            require(digest(Path(path)) == sha, label + ": hash " + path)
        require(bool(report["source_sha256"]), label + ": missing sources")
        gates(report["checks"], {"rank_predictions_agree"}, label + ": checks")
        require(len(report["layers_report"]) == count, label + ": layers")
        for n, layer in enumerate(report["layers_report"]):
            fields = {"latent", "rope"} | ({"index_keys"} if n in PRODUCERS else set())
            fields |= {"index_scales"} if n in INT8 else set()
            required = {f"{phase}.prefix.0.{f}" for phase in ("before", "after") for f in fields}
            required |= {"owners_released", "fp32_tp_reductions", "teacher_row_shapes"}
            required |= {f"token{i}.causal_indices" for i in range(32)}
            required |= {"shared_indices_restored_per_token"} if n not in PRODUCERS else set()
            gates(layer["checks"], required, f"{label}/layer{n}: checks")
            require(layer["layer"] == n and 0 < layer["peak_allocated"] <= QUOTA
                    and 0 <= layer["allocated_after_release"] <= layer["peak_allocated"]
                    and all(x.get("exact") is True for x in layer["checks"] if ".prefix." in x["name"])
                    and next(x for x in layer["checks"] if x["name"] == "owners_released")["alive"] == [], label + ": residency/prefix")
    return rows


def tensor(value, shape, dtype, label):
    require(value.shape == shape and value.dtype == dtype
            and bool(torch.isfinite(value).all()), label)


def layer_data(root, report, n, previous, shared):
    rank = report["rank"]
    data = torch.load(root / f"rank-{rank}" / f"layer-{n:02d}.pt", map_location="cpu", weights_only=True)
    label = f"{root.name}/rank{rank}/layer{n}"
    require(data["rank"] == rank and data["layer"] == n
            and data["teacher_tokens"] == report["teacher_tokens"] and data["positions"] == list(range(8192, 8224))
            and all(data[k] == report[k] for k in ("checkpoint_sha256", "inputs", "teacher_source")), label + ": identity")
    for field in ("input_hidden", "input_merged", "hidden", "residual", "merged"):
        tensor(data[field], (32, 6144), torch.bfloat16, label + ": " + field)
    if n:
        tensor(data["input_residual"], (32, 6144), torch.bfloat16, label + ": input_residual")
        require(all(torch.equal(data[f"input_{k}"], previous[k]) for k in ("hidden", "residual")), label + ": handoff")
    else:
        require(data["input_residual"] is None and torch.equal(data["input_hidden"], data["input_merged"]), label + ": initial norm")
    ids = data["indices"]
    require(ids.shape == (32, 2048) and ids.dtype == torch.int32
            and bool(((ids >= 0) & (ids <= torch.arange(8192, 8224)[:, None])).all())
            and bool((ids.sort(dim=1).values.diff(dim=1) > 0).all()), label + ": indices")
    require(data["skip_topk"] is (n not in PRODUCERS)
            and (n in PRODUCERS or torch.equal(ids, shared)), label + ": producer reuse")
    return data, ids if n in PRODUCERS else shared


def difference(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    na, nb = a.norm().item(), b.norm().item()
    return {"nrmse": (a - b).norm().item() / na if na else (0.0 if nb == 0 else None),
            "cosine": torch.dot(a, b).item() / (na * nb) if na * nb else float(na == nb)}


def audit(args):
    roots = [p.resolve() for p in (args.sfa, args.shared_fia)]
    teacher = json.loads(args.teacher.read_text())
    require(len(teacher["teacher_tokens"]) == len(teacher["expected_next_tokens"]) == 32
            and teacher["expected_next_tokens"][:-1] == teacher["teacher_tokens"][1:]
            and all(type(x) is int and 0 <= x < 154880 for x in teacher["teacher_tokens"] + teacher["expected_next_tokens"]), "teacher chain")
    runs = [reports(p, k, args.layers, teacher, digest(args.teacher)) for p, k in zip(roots, ("sfa", "shared_fia"))]
    stats = [{"layer": n, "hidden": [], "residual": [], "index_overlap": []} for n in range(args.layers)]
    for rank in range(16):
        a, b = runs[0][rank], runs[1][rank]
        require(all(a[k] == b[k] for k in ("checkpoint_sha256", "inputs", "teacher_source", "fp32_head"))
                and all(b["source_sha256"][p] == sha for p, sha in a["source_sha256"].items() if p in b["source_sha256"]), f"rank{rank}: arms differ")
        for arm in range(2):
            require(all(runs[arm][rank][k] == runs[arm][0][k] for k in ("checkpoint_sha256", "teacher_source", "source_sha256"))
                    and all(a["inputs"][k] == runs[0][0]["inputs"][k] for k in ("request_id", "generation", "prompt_sha256", "capture_manifest_sha256")), f"rank{rank}: rank identity")
        previous, shared = [None, None], [None, None]
        for n, stat in enumerate(stats):
            data = []
            for arm in range(2):
                value, shared[arm] = layer_data(roots[arm], runs[arm][rank], n, previous[arm], shared[arm])
                previous[arm] = value
                data.append(value)
            for field in ("hidden", "residual"):
                stat[field].append(difference(data[0][field], data[1][field]))
            left, right = [x["indices"].tolist() for x in data]
            stat["index_overlap"].append(sum(len(set(x) & set(y)) for x, y in zip(left, right)) / (32 * 2048))
        if args.layers == 78:
            for arm, root in enumerate(roots):
                final = torch.load(root / f"rank-{rank}" / "final.pt", map_location="cpu", weights_only=True)
                tensor(final["normalized"], (32, 6144), torch.bfloat16, "final normalized")
                tensor(final["logits"], (32, 154880), torch.float32 if runs[arm][rank]["fp32_head"] else torch.bfloat16, "final logits")
                require(final["teacher_tokens"] == teacher["teacher_tokens"] and final["predicted_tokens"] == final["logits"].argmax(-1).tolist() == teacher["expected_next_tokens"], "final raw teacher labels")
    for stat in stats:
        for field in ("hidden", "residual"):
            stat[field] = {"max_nrmse": max((x["nrmse"] for x in stat[field] if x["nrmse"] is not None), default=None), "min_cosine": min(x["cosine"] for x in stat[field]), "zero_reference_nonzero_candidate": sum(x["nrmse"] is None for x in stat[field])}
        stat["index_overlap"] = {"minimum_rank_mean": min(stat["index_overlap"]), "mean": sum(stat["index_overlap"]) / 16}
    return {"status": "teacher_pass" if args.layers == 78 else "partial_execution_pass", "layers": args.layers,
            "teacher_passed": True if args.layers == 78 else None, "teacher_labels_per_rank": 32 if args.layers == 78 else 0,
            "reported_layer_peak_bytes": {root.name: max(x["peak_allocated"] for r in rows for x in r["layers_report"]) for root, rows in zip(roots, runs)},
            "diagnostics_only": stats, "native_overlay_bound": False, "native_overlay_note": "Python hashes and OPP search paths do not identify loaded host libraries or executed device ELF.",
            "inputs": {str(root): digest(root / "summary.json") for root in roots}, "teacher_sha256": digest(args.teacher)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sfa", "shared_fia", "teacher", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--layers", type=int, choices=(8, 78), default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        result = audit(args)
    except Exception as error:
        result = {"status": "fail", "error": f"{type(error).__name__}: {error}"}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    raise SystemExit(result["status"] == "fail")
