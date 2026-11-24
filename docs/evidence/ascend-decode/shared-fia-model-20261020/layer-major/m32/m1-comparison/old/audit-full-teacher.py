"""Independently audit frozen layer-major tensors on the server CPU."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bits_equal(left, right):
    return (left.dtype == right.dtype and left.shape == right.shape
            and torch.equal(left.contiguous().view(torch.uint8),
                            right.contiguous().view(torch.uint8)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    reports, source_hashes, final_reference = [], {}, None
    state_pairs = elements = artifact_count = indices_checked = 0
    for rank in range(16):
        path = args.directory / f"rank-{rank}.json"
        report = json.loads(path.read_text())
        assert (report["rank"], report["layers"], report["steps"],
                report["query_width"], report["position"]) == (
                    rank, 78, 32, 1, 8224)
        assert report["full_model_executed"]
        assert report["model_tp_reduction"] == "float32"
        assert not report["precision_verified"]
        assert not report["performance_measured"]
        teacher = report["teacher_source"]
        assert sha256(teacher["path"]) == teacher["sha256"]
        assert sha256(teacher["prompt_path"]) == teacher["prompt_file_sha256"]
        assert report["teacher_tokens"] == teacher["teacher_tokens"]
        assert report["expected_tokens"] == teacher["expected_next_tokens"]
        for name, digest in report["source_sha256"].items():
            if name not in source_hashes:
                source_hashes[name] = sha256(name)
            assert source_hashes[name] == digest, name
        for name, digest in report["artifact_sha256"].items():
            assert sha256(name) == digest, name
            artifact_count += 1
        failed = [row["name"] for row in report["checks"]
                  if not row["passed"]]
        assert set(failed) <= {"greedy_matches_frozen_trajectory"}, failed
        previous, selections = None, None
        for number in range(78):
            path = args.directory / f"rank-{rank}/layer-{number:02d}.pt"
            data = torch.load(path, map_location="cpu", weights_only=True,
                              mmap=True)
            assert data["rank"] == rank and data["layer"] == number
            assert data["teacher_tokens"] == report["teacher_tokens"]
            assert data["positions"] == list(range(8192, 8224))
            assert data["checkpoint_sha256"] == report["checkpoint_sha256"]
            assert data["inputs"] == report["inputs"]
            assert data["teacher_source"] == teacher
            for name in ("input_hidden", "hidden", "residual", "merged"):
                value = data[name]
                assert value.shape == (32, 6144)
                assert value.dtype == torch.bfloat16
                assert bool(torch.isfinite(value).all())
            if previous is None:
                assert data["input_residual"] is None
            else:
                for actual, expected in (
                    (data["input_hidden"], previous[0]),
                    (data["input_residual"], previous[1]),
                ):
                    assert bits_equal(actual, expected), (rank, number)
                    state_pairs += 1
                    elements += actual.numel()
            indices = data["indices"]
            assert indices.shape == (32, 2048)
            ordered = indices.sort(dim=1).values
            assert bool((ordered[:, 1:] != ordered[:, :-1]).all())
            assert bool((indices >= 0).all())
            assert bool((indices <= torch.arange(8192, 8224)[:, None]).all())
            indices_checked += indices.numel()
            if data["skip_topk"]:
                assert bits_equal(indices, selections), (rank, number)
            else:
                selections = indices.clone()
            previous = data["hidden"], data["residual"]
        final = torch.load(args.directory / f"rank-{rank}/final.pt",
                           map_location="cpu", weights_only=True)
        logits = final["logits"]
        assert logits.shape == (32, 154880)
        assert logits.dtype == torch.bfloat16
        assert bool(torch.isfinite(logits).all())
        predictions = logits.argmax(dim=-1).tolist()
        assert predictions == report["predicted_tokens"]
        assert predictions == final["predicted_tokens"]
        assert final["teacher_tokens"] == report["teacher_tokens"]
        if final_reference is None:
            final_reference = logits.clone()
        assert bits_equal(logits, final_reference), rank
        mismatch = []
        top_two = logits.float().topk(2, dim=-1).values
        for step, (actual, expected) in enumerate(zip(
                predictions, report["expected_tokens"])):
            if actual != expected:
                mismatch.append({
                    "step": step, "position": 8192 + step,
                    "teacher_input": report["teacher_tokens"][step],
                    "predicted": actual, "expected": expected,
                    "predicted_logit": float(logits[step, actual]),
                    "expected_logit": float(logits[step, expected]),
                    "top1_margin": float(top_two[step, 0]
                                         - top_two[step, 1]),
                })
        assert report["greedy_correct"] == 32 - len(mismatch)
        reports.append({
            "rank": rank, "report_sha256": sha256(path.parent.parent
                                                   / f"rank-{rank}.json"),
            "execution_checks": len(report["checks"]) - 1,
            "greedy_correct": 32 - len(mismatch), "greedy_total": 32,
            "mismatches": mismatch,
            "peak_allocated_bytes": max(row["peak_allocated"]
                                        for row in report["layers_report"]),
        })
        print(json.dumps({"rank": rank, "greedy_correct": 32 - len(mismatch),
                          "mismatches": len(mismatch)}), flush=True)
    assert "torch_npu" not in __import__("sys").modules
    result = {
        "status": "execution_valid",
        "scope": "One frozen 8K prompt, 78 layers x32 M1 teacher rows, TP16",
        "precision_verified": False, "performance_measured": False,
        "torch_npu_imported": False,
        "state_component_bit_pairs": state_pairs,
        "state_component_elements": elements,
        "unique_causal_indices_checked": indices_checked,
        "artifacts_hashed": artifact_count,
        "source_files_hashed": len(source_hashes),
        "final_logits_identical_across_ranks": True,
        "all_teacher_successors_match": all(row["greedy_correct"] == 32
                                           for row in reports),
        "ranks": reports,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print("audit_sha256", sha256(args.output), flush=True)


if __name__ == "__main__":
    main()
