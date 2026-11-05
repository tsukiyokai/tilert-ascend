"""Count exact sparse-sharing structure from frozen M32 producer indices."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch


LEVELS = (32, 16, 8, 4, 1)
METHODS = ("per_query", "union_mask", "cascade", "full_cache_reference")
Q_HEADS = 4
TP_SIZE = 16


@dataclass
class Segment:
    start: int
    width: int
    keys: set[int]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha256(value):
    text = json.dumps(value, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def round_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def cascade(selected):
    segments = []

    def extract(start, residual, depth):
        width = LEVELS[depth]
        assert len(residual) == width
        common = set.intersection(*residual)
        if common:
            segments.append(Segment(start, width, common))
        if width == 1:
            return
        remainder = [row - common for row in residual]
        child_width = LEVELS[depth + 1]
        for offset in range(0, width, child_width):
            extract(start + offset,
                    remainder[offset:offset + child_width], depth + 1)

    extract(0, selected, 0)
    return segments


def restore(segments):
    rows = [[] for _ in range(32)]
    for segment in segments:
        for query in range(segment.start, segment.start + segment.width):
            rows[query].extend(segment.keys)
    return rows


def assert_exact(selected, restored):
    for expected, actual in zip(selected, restored, strict=True):
        assert len(actual) == len(set(actual)) == 2048
        assert set(actual) == expected


def count_work(segments):
    active = [segment for segment in segments if segment.keys]
    pairs = sum(segment.width * len(segment.keys) for segment in active)
    padded = sum(round_up(Q_HEADS * segment.width, 16)
                 * round_up(len(segment.keys), 128) for segment in active)
    return {
        "logical_kv_rows": sum(len(segment.keys) for segment in active),
        "qk_pairs_no_padding": pairs,
        "pv_pairs_no_padding": pairs,
        "qk_pairs_four_heads_no_padding": Q_HEADS * pairs,
        "pv_pairs_four_heads_no_padding": Q_HEADS * pairs,
        "qk_pairs_four_heads_m16_kv128": padded,
        "pv_pairs_four_heads_m16_kv128": padded,
        "tiles_m16_kv128": padded // (16 * 128),
        "segments": len(active),
    }


def producer(args, report, layer):
    path = args.report.parent / "rank-0" / f"layer-{layer:02d}.pt"
    digest = sha256(path)
    expected = [value for name, value in report["artifact_sha256"].items()
                if name.endswith(f"/rank-0/layer-{layer:02d}.pt")]
    assert expected == [digest]
    data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    assert data["rank"] == 0 and data["layer"] == layer
    assert not data["skip_topk"]
    assert data["positions"] == list(range(8192, 8224))
    for key in ("checkpoint_sha256", "inputs", "teacher_source",
                "teacher_tokens"):
        assert data[key] == report[key]
    indices = data["indices"]
    assert indices.shape == (32, 2048) and indices.dtype == torch.int32
    original = indices.tolist()
    selected = [set(row) for row in original]
    assert all(len(row) == 2048 for row in selected)
    assert all(min(row) >= 0 and max(row) <= 8192 + query
               for query, row in enumerate(selected))
    union = set.union(*selected)
    individual = [Segment(query, 1, row) for query, row in enumerate(selected)]
    shared = cascade(selected)
    assert_exact(selected, restore(individual))
    assert_exact(selected, [sorted(union & row) for row in selected])
    assert_exact(selected, restore(shared))
    plans = {
        "per_query": individual,
        "union_mask": [Segment(0, 32, union)],
        "cascade": shared,
        "full_cache_reference": [
            Segment(0, 32, set(range(report["capacity"]))),
        ],
    }
    models = {name: count_work(plan) for name, plan in plans.items()}
    assert models["per_query"]["qk_pairs_no_padding"] == 32 * 2048
    assert models["cascade"]["qk_pairs_no_padding"] == 32 * 2048
    levels = []
    for width in LEVELS:
        row = count_work([segment for segment in shared
                          if segment.width == width])
        levels.append({"query_group": width, **row})
    return {
        "layer": layer, "artifact_path": str(path), "artifact_sha256": digest,
        "selected_sets_sha256": canonical_sha256(
            [sorted(row) for row in selected]),
        "selected_pairs": 32 * 2048, "union_rows": len(union),
        "intersection_32_rows": len(set.intersection(*selected)),
        "models": models, "cascade_by_group": levels,
        "exact_query_sets_checked": 32 * 3,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = json.loads(args.report.read_text())
    assert report["status"] == "pass" and report["rank"] == 0
    assert report["query_width"] == 32 and report["layers"] == 78
    assert report["greedy_correct"] == report["greedy_total"] == 32
    assert report["attention_kernel"] == "sfa" and report["capacity"] == 9344
    checkpoint_arg = report["command"].index("--checkpoint") + 1
    config_path = Path(report["command"][checkpoint_arg]) / "config.json"
    assert sha256(config_path) == report["checkpoint_sha256"]["config.json"]
    config = json.loads(config_path.read_text())
    assert config["num_attention_heads"] == TP_SIZE * Q_HEADS
    producers = [row["layer"] for row in report["layers_report"]
                 if any(".indexer." in name
                        for name in row["coverage"]["cache_specs"])]
    assert len(producers) == 21
    rows = [producer(args, report, layer) for layer in producers]
    totals = {method: {key: sum(row["models"][method][key] for row in rows)
                       for key in rows[0]["models"][method]}
              for method in METHODS}
    result = {
        "scope": "rank 0, frozen M32/SFA trajectory, 21 producer selections",
        "report_path": str(args.report), "report_sha256": sha256(args.report),
        "script_sha256": sha256(Path(__file__)),
        "teacher_sha256": report["teacher_source"]["sha256"],
        "inputs": report["inputs"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "query_count": 32, "top_k": 2048, "cache_capacity": 9344,
        "q_heads_per_rank": Q_HEADS, "tensor_parallel_size": TP_SIZE,
        "config_path": str(config_path), "config_sha256": sha256(config_path),
        "selected_pairs_per_head": sum(row["selected_pairs"] for row in rows),
        "selected_pairs_four_heads": Q_HEADS * sum(
            row["selected_pairs"] for row in rows),
        "cascade_levels": list(LEVELS),
        "work_model": {
            "logical_unit": ("query-KV pairs per attention head; "
                             "QK and PV counted separately"),
            "logical_no_padding": ("sum(segment_query_count "
                                   "* segment_KV_count)"),
            "four_heads_no_padding": "4 * logical query-KV pairs",
            "four_heads_m16_kv128": "sum(ceil(4*g/16)*16 * ceil(N/128)*128)",
            "logical_reads": ("one KV row occurrence per nonempty segment, "
                              "shared by 4 Q heads; K/V not double-counted"),
            "mask": ("union and full-cache rectangles compute "
                     "masked pairs as dense work"),
            "effective_selected_pairs": ("65536 per producer "
                                         "for all exact sparse plans"),
            "physical_measurement": False,
        },
        "paper": {
            "url": "https://arxiv.org/html/2610.11134v1#S3.SS3",
            "mechanism": ("recursive intersection removal; "
                          "no clipping or tail push-down"),
        },
        "totals_over_21_producers": totals,
        "exact_query_sets_checked": sum(
            row["exact_query_sets_checked"] for row in rows),
        "producers": rows,
        "cpu": {
            "torch": str(torch.__version__),
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "backend_autoload": os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"],
        },
        "torch_npu_imported": "torch_npu" in sys.modules,
    }
    assert not result["torch_npu_imported"]
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "producers": producers, "totals": totals,
        "exact_query_sets_checked": result["exact_query_sets_checked"],
    }))


if __name__ == "__main__":
    main()
