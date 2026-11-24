"""Independently inspect frozen two-round layer artifacts on server CPU."""
import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch


@lru_cache(None)
def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main(args):
    root = args.run.resolve()
    summary = json.loads((root / "summary.json").read_text())
    assert len(summary["ranks"]) == 16
    assert {r["rank"] for r in summary["ranks"]} == set(range(16))
    baseline, finals, peaks, report_hashes = [], [], [], {}
    positions = list(range(8192, 8200)) + list(range(8195, 8203))
    labels = [382, 262, 576, 2524]
    for rank in range(16):
        report_path = root / f"rank-{rank}.json"
        r = json.loads(report_path.read_text())
        report_hashes[str(rank)] = digest(report_path)
        assert next(x for x in summary["ranks"] if x["rank"] == rank)["report_sha256"] == digest(report_path)
        assert r["rank"] == rank and r["layers"] == args.layers
        assert r["steps"] == 16 and r["query_width"] == 8 and r["position"] == 8196
        assert r["attention_kernel"] == "sfa" and r["fp32_head"] is True
        assert r["scored_rows"] == [0, 1, 2, 8] and r["expected_tokens"] == labels
        assert r["round_trace"]["sha256"] == "3aa6cce191de24a06a1131f45db3c93f30716ffe5024ad0c23ca4f8c7524d671"
        assert r["target_graph"] is False and r["conditioned_mtp"] is False
        assert all(c["passed"] for c in r["checks"] if c["name"] != "trusted_successor_labels")
        paths = {root / f"rank-{rank}/layer-{n:02d}.pt" for n in range(args.layers)}
        if args.layers == 78:
            paths.add(root / f"rank-{rank}/final.pt")
        assert {Path(p) for p in r["artifact_sha256"]} == paths
        for path, sha in {**r["source_sha256"], **r["artifact_sha256"]}.items():
            assert digest(path) == sha, path
        previous, shared = None, None
        assert len(r["layers_report"]) == args.layers
        for n, layer in enumerate(r["layers_report"]):
            assert layer["layer"] == n and all(c["passed"] for c in layer["checks"])
            assert 0 < layer["peak_allocated"] <= 2 * 1024**3
            assert next(c for c in layer["checks"] if c["name"] == "owners_released")["alive"] == []
            peaks.append(layer["peak_allocated"])
            d = torch.load(root / f"rank-{rank}/layer-{n:02d}.pt", map_location="cpu", weights_only=True)
            assert d["positions"] == positions and d["candidate_tokens"] == r["candidate_tokens"]
            assert d["inputs"] == r["inputs"] and d["checkpoint_sha256"] == r["checkpoint_sha256"]
            for field in ("input_hidden", "input_merged", "hidden", "residual", "merged"):
                assert d[field].shape == (16, 6144) and d[field].dtype == torch.bfloat16
                assert bool(torch.isfinite(d[field]).all())
            if previous is None:
                assert d["input_residual"] is None
            else:
                assert torch.equal(d["input_hidden"], previous["hidden"])
                assert torch.equal(d["input_residual"], previous["residual"])
            ids = d["indices"]
            assert ids.shape == (16, 2048) and ids.dtype == torch.int32
            assert bool(((ids >= 0) & (ids <= torch.tensor(positions)[:, None])).all())
            assert bool((ids.sort(1).values.diff() > 0).all())
            if d["skip_topk"]:
                assert torch.equal(ids, shared)
            else:
                shared = ids
            assert len(d["round_calls"]) == 2
            for call, (start, retained) in zip(d["round_calls"], ((8192, 8195), (8195, 8196))):
                assert [call[k] for k in ("position", "written", "retained")] == [start, start + 8, retained]
                assert call["positions"] == call["slots"] == list(range(start, start + 8))
                assert call["seq_lens"] == [start + 8]
                for md in call["attention"].values():
                    assert md["query"] == [8] and md["key"] == [start + 8]
                    pages = (start + 8 + 127) // 128
                    assert torch.equal(md["block_table"][:, :pages], torch.arange(pages)[None])
                for value in call["cache_rows"].values():
                    assert value.shape[0] == 8 and bool(torch.isfinite(value).all())
            replicated = {k: d[k] for k in ("hidden", "residual", "indices")}
            if rank == 0:
                baseline.append(replicated)
            else:
                assert all(torch.equal(v, baseline[n][k]) for k, v in replicated.items())
            previous = d
        if args.layers == 78:
            f = torch.load(root / f"rank-{rank}/final.pt", map_location="cpu", weights_only=True)
            logits = f["logits"]
            assert logits.shape == (16, 154880) and logits.dtype == torch.float32
            assert bool(torch.isfinite(logits).all())
            predicted = logits.argmax(-1).tolist()
            assert predicted == f["predicted_tokens"] == r["predicted_tokens"]
            assert [predicted[i] for i in r["scored_rows"]] == r["scored_predictions"]
            assert r["greedy_correct"] == sum(predicted[i] == y for i, y in zip(r["scored_rows"], labels))
            rows = []
            for i, expected in zip(r["scored_rows"], labels):
                scores, ids = logits[i].topk(2)
                rows.append({"row": i, "expected": expected, "top2_ids": ids.tolist(),
                             "top2_scores": scores.tolist(), "margin": (scores[0] - scores[1]).item(),
                             "score_2524": logits[i, 2524].item(), "score_6500": logits[i, 6500].item()})
            finals.append({"rank": rank, "correct": r["greedy_correct"], "rows": rows})
    passed = all(x["correct"] == 4 for x in finals) if finals else None
    result = {"status": "partial_execution_pass" if not finals else ("target_labels_pass" if passed else "target_labels_fail"),
              "layers": args.layers, "ranks": 16, "labels_passed": passed, "performance_measured": False,
              "artifact_hashes_validated": 16 * (args.layers + bool(finals)),
              "all_layer_hidden_residual_indices_equal_across_ranks": True,
              "peak_layer_bytes": max(peaks), "finals": finals, "report_sha256": report_hashes,
              "auditor_sha256": digest(__file__), "scope": "Frozen eager target calls. Finite KV is not a complete write-coverage proof; original Graph and conditioned MTP remain separate."}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("finals", "report_sha256")}))
    return 1 if passed is False else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", type=int, choices=(8, 78), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    raise SystemExit(main(args))
