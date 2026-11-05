"""Audit a completed failed arm without changing the teacher acceptance gate."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch

BASE = Path("/workspace/experiments/m32-full-teacher-20261020")
HELPER = Path("/tmp/layer-major-m32-20261020/audit.py")
HELPER_SHA = "ada3a95aa8e677197a4a3ea22149823416e1d5145f07347e17dc4c5b17996cec"
TEACHER = Path("/workspace/experiments/shared-fia-20261020/teacher-32.json")
TEACHER_SHA = "82d011962a288b30b777fc4964d5c96c6b9784cb13b97f509322f603b263b655"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    assert digest(HELPER) == HELPER_SHA
    assert digest(TEACHER) == TEACHER_SHA
    spec = importlib.util.spec_from_file_location("audit", HELPER)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    roots = [BASE / "sfa-78", BASE / "shared/shared-fia-78"]
    teacher = json.loads(TEACHER.read_text())
    labels = teacher["expected_next_tokens"]
    tokens = teacher["teacher_tokens"]
    h.require(len(tokens) == len(labels) == 32 and labels[:-1] == tokens[1:], "teacher chain")
    final_reference, arms, reports, rank0_layers = [], [], [], []
    identities = ("checkpoint_sha256", "inputs", "teacher_source", "fp32_head")
    for arm, root in enumerate(roots):
        summary = json.loads((root / "summary.json").read_text())
        h.require(summary["status"] == ("pass" if arm == 0 else "fail")
                  and {r["rank"] for r in summary["ranks"]} == set(range(16)), "summary")
        rows, finals, layers0 = [], [], []
        for rank in range(16):
            report = json.loads((root / f"rank-{rank}.json").read_text())
            rows.append(report)
            receipt = next(r for r in summary["ranks"] if r["rank"] == rank)
            h.require(receipt["report_sha256"] == digest(root / f"rank-{rank}.json")
                      and receipt["status"] == report["status"] == summary["status"], "report receipt")
            h.require([report[k] for k in ("rank", "layers", "steps", "query_width", "capacity", "position")]
                      == [rank, 78, 32, 32, 9344, 8224]
                      and report["attention_kernel"] == ("sfa", "shared_fia")[arm]
                      and report["model_tp_reduction"] == "float32"
                      and report["torch_quota_bytes"] == h.QUOTA
                      and report["fp32_head"] is True, "run contract")
            h.require(report["teacher_tokens"] == tokens and report["expected_tokens"] == labels
                      and report["teacher_source"]["sha256"] == TEACHER_SHA, "teacher identity")
            h.require(all(report[k] == rows[0][k] for k in identities if k != "inputs")
                      and all(report["inputs"][k] == rows[0]["inputs"][k] for k in
                              ("request_id", "generation", "prompt_sha256", "prompt_length",
                               "last_prompt_token", "first_token_id", "initial_position",
                               "capture_manifest_sha256")), "rank identities")
            if arm:
                h.require(all(report[k] == reports[0][rank][k] for k in identities), "paired identity")
            expected = {root / f"rank-{rank}" / f"layer-{n:02d}.pt" for n in range(78)}
            expected.add(root / f"rank-{rank}" / "final.pt")
            h.require({Path(p) for p in report["artifact_sha256"]} == expected, "artifact coverage")
            for path, sha in {**report["source_sha256"], **report["artifact_sha256"]}.items():
                h.require(h.digest(path) == sha, "hash " + path)
            failed = [c["name"] for c in report["checks"] if c["passed"] is not True]
            h.require(failed == ([] if arm == 0 else ["greedy_matches_frozen_trajectory"]), "unexpected gate failure")
            h.gates([c for c in report["checks"] if c["name"] != "greedy_matches_frozen_trajectory"],
                    {"logits_shape_and_dtype", "final_normalized_shape_and_dtype",
                     "final_outputs_finite", "rank_predictions_agree"}, "other checks")
            h.require(len(report["layers_report"]) == 78, "layer coverage")
            previous, shared = None, None
            for n, layer in enumerate(report["layers_report"]):
                fields = {"latent": ((32, 512), torch.bfloat16), "rope": ((32, 64), torch.bfloat16)}
                if n in h.PRODUCERS:
                    fields["index_keys"] = ((32, 128), torch.int8 if n in h.INT8 else torch.bfloat16)
                if n in h.INT8:
                    fields["index_scales"] = ((32,), torch.float16)
                required = {f"{phase}.prefix.0.{f}" for phase in ("before", "after") for f in fields}
                required |= {"owners_released", "fp32_tp_reductions", "teacher_row_shapes"}
                required |= {f"token{i}.causal_indices" for i in range(32)}
                if n not in h.PRODUCERS:
                    required.add("shared_indices_restored_per_token")
                h.gates(layer["checks"], required, f"rank{rank}/layer{n}")
                h.require(layer["layer"] == n and 0 < layer["peak_allocated"] <= h.QUOTA
                          and 0 <= layer["allocated_after_release"] <= layer["peak_allocated"]
                          and all(c.get("exact") is True for c in layer["checks"] if ".prefix." in c["name"])
                          and next(c for c in layer["checks"] if c["name"] == "owners_released")["alive"] == [], "residency/prefix")
                data, shared = h.layer_data(root, report, n, previous, shared)
                h.require(set(data["cache_rows"]) == set(fields), "KV fields")
                for name, (shape, dtype) in fields.items():
                    h.tensor(data["cache_rows"][name], shape, dtype, name)
                previous = data
                if rank == 0:
                    layers0.append({k: data[k] for k in ("hidden", "residual", "merged", "indices")})
            final = torch.load(root / f"rank-{rank}/final.pt", map_location="cpu", weights_only=True)
            h.tensor(final["normalized"], (32, 6144), torch.bfloat16, "normalized")
            h.tensor(final["logits"], (32, 154880), torch.float32, "logits")
            predicted = final["logits"].argmax(-1).tolist()
            correct = sum(a == b for a, b in zip(predicted, labels))
            h.require(final["teacher_tokens"] == tokens and final["predicted_tokens"]
                      == report["predicted_tokens"] == predicted
                      and report["greedy_correct"] == correct and report["greedy_total"] == 32,
                      "raw logits versus report")
            if rank == 0:
                reference = final
            h.require(torch.equal(final["logits"].view(torch.uint8), reference["logits"].view(torch.uint8)), "cross-rank logits")
            finals.append({"rank": rank, "greedy_correct": correct, "wrong_rows": [i for i, (a, b) in enumerate(zip(predicted, labels)) if a != b]})
        reports.append(rows)
        rank0_layers.append(layers0)
        final_reference.append(reference)
        arms.append({"kernel": ("sfa", "shared_fia")[arm], "summary_sha256": digest(root / "summary.json"),
                     "all_final_logits_bytes_equal": True, "ranks": finals,
                     "peak_layer_bytes": max(x["peak_allocated"] for r in rows for x in r["layers_report"])})
    curves = []
    for n, (a, b) in enumerate(zip(*rank0_layers)):
        stat = {"layer": n}
        for key in ("hidden", "residual", "merged"):
            stat[key] = h.difference(a[key], b[key])
            stat[key]["per_query"] = [h.difference(x, y) for x, y in zip(a[key], b[key])]
        stat["index_overlap_count"] = [len(set(x) & set(y)) for x, y in zip(a["indices"].tolist(), b["indices"].tolist())]
        curves.append(stat)
    a, b = [f["logits"].double() for f in final_reference]
    query_stats = []
    for i, label in enumerate(labels):
        row = {"query": i, "teacher": label, "max_abs_difference": (a[i] - b[i]).abs().max().item()}
        for name, logits in (("sfa", a), ("shared_fia", b)):
            values, ids = logits[i].topk(2)
            row[name] = {"top2_ids": ids.tolist(), "top2_scores": values.tolist(),
                         "top1_margin": (values[0] - values[1]).item(),
                         "teacher_score": logits[i, label].item(),
                         "max_abs_logit": logits[i].abs().max().item()}
        la, lb = a[i].log_softmax(-1), b[i].log_softmax(-1)
        row["kl_sfa_to_shared"] = (la.exp() * (la - lb)).sum().item()
        query_stats.append(row)
    out = {"status": "teacher_failure_audited", "teacher_passed": False, "performance_measured": False,
           "artifact_hashes_validated": 2528, "layer_handoffs_checked": 2464,
           "teacher_sha256": TEACHER_SHA, "helper_sha256": HELPER_SHA, "auditor_sha256": digest(__file__),
           "arms": arms, "rank0_layer_diagnostics": curves, "final_query_diagnostics": query_stats,
           "final_normalized": h.difference(final_reference[0]["normalized"], final_reference[1]["normalized"]),
           "scope": "Original teacher gate remains failed. Layer metrics diagnose; no gate is relaxed."}
    (BASE / "shared/paired-diagnostic.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: out[k] for k in ("status", "artifact_hashes_validated", "final_normalized")}))


if __name__ == "__main__":
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()
