import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def summarize(directory, reference, golden, require_warm_head=False,
              candidate_variant=1):
    assert candidate_variant > 0
    variants = (0, candidate_variant)
    expected = golden["output_token_ids"]
    digest = hashlib.sha256(json.dumps(expected).encode()).hexdigest()
    series = {variant: [] for variant in variants}
    runs = []
    heads_prepared = []
    for pair in range(5):
        for variant in variants:
            folder = directory / f"pair-{pair}-variant-{variant}"
            starts = []
            rows = []
            for rank in range(8):
                row = read(folder / f"mtp-rank-{rank}.json")["mtp_w8"]
                state = read(reference / f"rank-{rank}.json")
                assert row["tokens"] == expected, (pair, variant, rank, "tokens")
                assert row["output_sha256"] == digest
                assert state["main_matches"]
                assert row["main_state"] == state["main_state"], (pair, variant, rank, "main")
                assert row["draft_state"] == state["serial_draft_state"], (
                    pair, variant, rank, "draft")
                assert row["synchronized_start"]
                head = row.get("head_execution_before_timing", {})
                graph_width = head.get("active_graph_width", 32)
                ready = head.get("wrapped", False) or graph_width in head.get("graph_widths", [])
                assert not require_warm_head or ready, (pair, variant, rank, "cold head")
                heads_prepared.append(ready)
                assert sum(row["accepted_per_round"]) == len(expected) - 1
                assert row["rounds"] == len(row["accepted_per_round"])
                starts.append(row["start_monotonic_ns"])
                rows.append(row)
            row = rows[0]
            assert all(peer["accepted_per_round"] == row["accepted_per_round"] for peer in rows)
            elapsed = row["elapsed_ms"] / (len(expected) - 1)
            assert abs(elapsed - row["tpot_ms"]) < 0.006
            series[variant].append(elapsed)
            runs.append({"pair": pair, "variant": variant,
                         "rank0_tpot_ms": elapsed, "rounds": row["rounds"],
                         "start_spread_ms": (max(starts) - min(starts)) / 1e6,
                         "all_8_rank_states_exact": True})
    control, candidate = (statistics.median(series[v]) for v in variants)
    changes = [new - old for old, new in zip(series[0], series[candidate_variant])]
    return {"control_samples_ms": series[0], "candidate_samples_ms": series[candidate_variant],
            "control_median_ms": control, "candidate_median_ms": candidate,
            "candidate_change_percent": (candidate / control - 1) * 100,
            "paired_changes_ms": changes,
            "all_pairs_faster": all(change < 0 for change in changes),
            "max_start_spread_ms": max(run["start_spread_ms"] for run in runs),
            "all_80_rank_runs_tokens_main_serial_draft_exact": True,
            "all_main_heads_prepared_before_timing": all(heads_prepared),
            "golden_sha256": digest, "runs": runs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("golden", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-warm-head", action="store_true")
    parser.add_argument("--candidate-variant", type=int, default=1)
    args = parser.parse_args()
    result = summarize(args.directory, args.reference, read(args.golden),
                       args.require_warm_head, args.candidate_variant)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
