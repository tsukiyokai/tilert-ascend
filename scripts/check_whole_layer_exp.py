import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def history(request):
    fields = ("width", "candidates", "accepted_tokens", "natural_accepted",
              "matched_drafts", "before", "written", "retained",
              "draft_retained")
    return [[row[field] for field in fields] for row in request["rounds"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("golden", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = read(args.evidence / "summary.json")
    golden = read(args.golden)
    ranks = [read(args.evidence / f"rank-{rank}.json") for rank in range(8)]
    requests = ranks[0]["requests"]
    expected_tokens = golden["requests"][-1]["tokens"]
    expected_history = history(golden["rank_requests"][-1])
    checks = {
        "completed": all(rank["status"] == "completed" and rank["closed"]
                         for rank in ranks),
        "rank_order": [rank["rank"] for rank in ranks] == list(range(8)),
        "summary": (summary["status"] == "measured"
                    and summary["all_rank_state_agreement"]
                    and summary["world_size"] == 8
                    and summary["expert_parallel"]),
        "rank_errors": not any(rank["errors"] for rank in summary["ranks"]),
        "request_count": (
            len(requests) == 6
            and sum(not row["warmup"] for row in requests) == 3
            and all(len(rank["rank_requests"]) == 6 for rank in ranks)),
        "tokens": all(row["tokens"] == expected_tokens for row in requests),
        "output_hash": all(
            hashlib.sha256(json.dumps(row["tokens"], separators=(",", ":"))
                           .encode("ascii")).hexdigest()
            == row["output_sha256"] for row in requests),
        "protocol": (ranks[0]["width"] == 32
                     and ranks[0]["max_tokens"] == 1024
                     and not any(row["profiled"] for row in requests)),
        "trajectory": all(history(row) == expected_history
                          for rank in ranks for row in rank["rank_requests"]),
        "checkpoint": all(
            rank["checkpoint_sha256"] == golden["checkpoint_sha256"]
            for rank in ranks),
        "drafter": all(
            rank["host_drafter"][key] == golden["host_drafter"][key]
            for rank in ranks
            for key in ("module", "class", "kind", "file_sha256")),
        "prompt": all(
            rank["inputs"]["prompt_sha256"]
            == golden["inputs"]["prompt_sha256"] for rank in ranks),
    }
    timings = [(row["timestamps_ns"][-1] - row["timestamps_ns"][0]) /
               (len(row["tokens"]) - 1) / 1e6
               for row in requests if not row["warmup"]]
    measured = [row for row in requests if not row["warmup"]]
    checks["timing"] = all(abs(value - row["tpot_ms"]) < 1e-9
                           for value, row in zip(timings, measured))
    report = {
        "pass": all(checks.values()), "checks": checks,
        "scope": ("case-00 TP8+EP8 baseline token and acceptance trajectory; "
                  "excludes HF precision"),
        "golden_sha256": hashlib.sha256(args.golden.read_bytes()).hexdigest(),
        "output_sha256": [row["output_sha256"] for row in requests],
        "rounds": [row["session"]["completed_rounds"] for row in requests],
        "tpot_ms": {"samples": timings, "median": statistics.median(timings),
                    "min": min(timings), "max": max(timings)},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    raise SystemExit(not report["pass"])


if __name__ == "__main__":
    main()
