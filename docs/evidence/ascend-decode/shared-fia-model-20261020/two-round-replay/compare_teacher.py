"""Compare only causally identical rows with the accepted M32 teacher."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(actual, expected):
    left, right = actual.double().flatten(), expected.double().flatten()
    delta = left - right
    return {
        "equal": torch.equal(actual, expected),
        "nrmse": (delta.norm() / right.norm()).item(),
        "max_abs": delta.abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(
            left, right, dim=0,
        ).item(),
    }


def main(args):
    roots = [args.replay, args.teacher]
    reports = [json.loads((p / "rank-0.json").read_text()) for p in roots]
    actual, reference = reports
    assert actual["checkpoint_sha256"] == reference["checkpoint_sha256"]
    assert actual["inputs"] == reference["inputs"]
    assert actual["scored_rows"] == [0, 1, 2, 8]
    assert actual["expected_tokens"] == reference["expected_tokens"][:4]
    assert reference["greedy_correct"] == reference["greedy_total"] == 32
    actual_rows, reference_rows = [0, 1, 2, 8], [0, 1, 2, 3]
    first, layers, hashes = {}, [], {}
    for number in range(actual["layers"]):
        paths = [p / f"rank-0/layer-{number:02d}.pt" for p in roots]
        for path, report in zip(paths, reports):
            hashes[str(path)] = digest(path)
            assert hashes[str(path)] == report["artifact_sha256"][str(path)]
        a, b = [torch.load(p, map_location="cpu", weights_only=True)
                for p in paths]
        assert [a["candidate_tokens"][i] for i in actual_rows] == b[
            "teacher_tokens"
        ][:4]
        assert [a["positions"][i] for i in actual_rows] == b["positions"][:4]
        rows = []
        for ai, bi in zip(actual_rows, reference_rows):
            result = {"replay_row": ai, "teacher_row": bi}
            for field in ("input_hidden", "input_residual", "input_merged",
                          "hidden", "residual", "merged"):
                if a[field] is None:
                    assert b[field] is None
                    continue
                result[field] = metrics(a[field][ai], b[field][bi])
                if not result[field]["equal"]:
                    first.setdefault(field, {"layer": number, "row": ai})
            left, right = a["indices"][ai], b["indices"][bi]
            result["index_order_equal"] = torch.equal(left, right)
            result["index_overlap"] = torch.isin(left, right).sum().item()
            for key, same in (("index_order", result["index_order_equal"]),
                              ("index_set", result["index_overlap"] == 2048)):
                if not same:
                    first.setdefault(key, {"layer": number, "row": ai})
            call, row = (0, ai) if ai < 8 else (1, ai - 8)
            result["new_cache"] = {}
            for name, value in a["round_calls"][call]["cache_rows"].items():
                same = torch.equal(value[row], b["cache_rows"][name][bi])
                result["new_cache"][name] = same
                if not same:
                    first.setdefault("cache." + name,
                                     {"layer": number, "row": ai})
            rows.append(result)
        layers.append({"layer": number, "rows": rows})
    result = {
        "scope": "Rank0 trusted causal rows, frozen eager M8 versus M32",
        "layers": layers, "first_difference": first,
        "artifact_sha256": hashes, "script_sha256": digest(Path(__file__)),
        "report_sha256": {str(p / "rank-0.json"): digest(p / "rank-0.json")
                          for p in roots},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"layers": len(layers), "first_difference": first}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("output", type=Path)
    torch.set_num_threads(1)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    main(parser.parse_args())
