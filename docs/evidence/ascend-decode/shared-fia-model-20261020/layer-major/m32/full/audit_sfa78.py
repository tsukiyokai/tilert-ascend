"""Audit completed TP16 M32 SFA78 teacher artifacts on CPU."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch

HELPER = Path("/tmp/layer-major-m32-20261020/audit.py")
HELPER_SHA = "ada3a95aa8e677197a4a3ea22149823416e1d5145f07347e17dc4c5b17996cec"
TEACHER_SHA = "82d011962a288b30b777fc4964d5c96c6b9784cb13b97f509322f603b263b655"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def helper():
    if digest(HELPER) != HELPER_SHA:
        raise ValueError("Existing layer-major auditor source differs")
    spec = importlib.util.spec_from_file_location("layer_major_audit", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(args):
    h, root = helper(), args.run.resolve()
    h.require(digest(args.teacher) == TEACHER_SHA, "frozen teacher source")
    teacher = json.loads(args.teacher.read_text())
    tokens, labels = teacher["teacher_tokens"], teacher["expected_next_tokens"]
    h.require(len(tokens) == len(labels) == 32 and labels[:-1] == tokens[1:]
              and all(type(x) is int and 0 <= x < 154880 for x in tokens + labels), "teacher chain")
    reports = h.reports(root, "sfa", 78, teacher, TEACHER_SHA)
    identities = ("request_id", "generation", "prompt_sha256", "prompt_length",
                  "last_prompt_token", "first_token_id", "initial_position", "capture_manifest_sha256")
    final_hashes, reference, float_tensors, float_values = {}, None, 0, 0
    for rank, report in enumerate(reports):
        h.require(report["fp32_head"] is True, f"rank{rank}: FP32 head required")
        h.require(all(report[k] == reports[0][k] for k in
                      ("checkpoint_sha256", "teacher_source", "source_sha256"))
                  and all(report["inputs"][k] == reports[0]["inputs"][k]
                          for k in identities), f"rank{rank}: execution identity")
        h.gates(report["checks"], {"greedy_matches_frozen_trajectory", "logits_shape_and_dtype",
                "final_normalized_shape_and_dtype", "final_outputs_finite", "rank_predictions_agree"},
                f"rank{rank}: final gates")
        previous, shared = None, None
        for number in range(78):
            data, shared = h.layer_data(root, report, number, previous, shared)
            fields = {"latent": ((32, 512), torch.bfloat16), "rope": ((32, 64), torch.bfloat16)}
            if number in h.PRODUCERS:
                fields["index_keys"] = ((32, 128), torch.int8 if number in h.INT8 else torch.bfloat16)
            if number in h.INT8:
                fields["index_scales"] = ((32,), torch.float16)
            h.require(set(data["cache_rows"]) == set(fields), f"rank{rank}/layer{number}: KV fields")
            for name, (shape, dtype) in fields.items():
                value = data["cache_rows"][name]
                h.tensor(value, shape, dtype, f"rank{rank}/layer{number}: new KV {name}")
                if value.is_floating_point():
                    float_tensors += 1
                    float_values += value.numel()
            previous = data
        final = torch.load(root / f"rank-{rank}" / "final.pt", map_location="cpu", weights_only=True)
        h.tensor(final["normalized"], (32, 6144), torch.bfloat16, f"rank{rank}: normalized")
        logits = final["logits"]
        h.tensor(logits, (32, 154880), torch.float32, f"rank{rank}: logits")
        h.require(final["teacher_tokens"] == tokens and final["predicted_tokens"]
                  == logits.argmax(-1).tolist() == labels, f"rank{rank}: raw final labels")
        raw = logits.contiguous().view(torch.uint8)
        if reference is None:
            reference = raw.clone()
        h.require(torch.equal(raw, reference), f"rank{rank}: final logits bytes differ from rank0")
        final_hashes[str(rank)] = hashlib.sha256(raw.numpy()).hexdigest()
    return {"status": "teacher_pass", "teacher_passed": True, "ranks": 16, "layers": 78,
            "query_width": 32, "fp32_head": True, "teacher_labels_per_rank": 32,
            "all_final_logits_bytes_equal": True, "new_float_kv_tensors_checked": float_tensors,
            "new_float_kv_values_checked": float_values, "layer_handoffs_checked": 16 * 77,
            "artifact_hashes_validated": 16 * 79, "final_logit_tensor_sha256": final_hashes,
            "reported_layer_peak_bytes": max(x["peak_allocated"] for r in reports for x in r["layers_report"]),
            "summary_sha256": digest(root / "summary.json"), "teacher_sha256": TEACHER_SHA,
            "helper_sha256": HELPER_SHA, "auditor_sha256": digest(__file__),
            "precision_verified": False, "performance_measured": False, "native_overlay_bound": False,
            "scope": "One frozen teacher, eager layer-major M32. Native ELF identity, token-major Graph/reset and request TPOT require separate evidence."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "teacher", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    torch.set_num_threads(1)
    try:
        result = audit(args)
    except FileNotFoundError as error:
        result = {"status": "incomplete", "teacher_passed": None, "error": str(error)}
    except Exception as error:
        result = {"status": "fail", "teacher_passed": False, "error": f"{type(error).__name__}: {error}"}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    raise SystemExit({"teacher_pass": 0, "fail": 1, "incomplete": 2}[result["status"]])
