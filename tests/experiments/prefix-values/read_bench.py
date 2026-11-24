"""Recompute the three-arm A3 report from every measured output and clock."""

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def readback(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    samples = report["samples"]
    orders = Counter(tuple(row["order"]) for row in samples)
    assert report["status"] == "measured" and len(samples) == 60
    assert len(orders) == 6 and set(orders.values()) == {10}
    arms = ("native", "expanded", "cached")
    roots = [report["workers"][arm]["cache_root"] for arm in arms]
    assert len(set(roots)) == 3
    for arm in arms:
        worker = report["workers"][arm]
        assert worker["identity"]["inputs_unchanged"]
        assert worker["identity"]["prefix_unchanged"]
        assert all(
            row[arm]["output_sha256"] == worker["expected_sha256"]
            for row in samples
        )
        for clock in ("wall_us", "event_us"):
            values = sorted(row[arm][clock] for row in samples)
            assert all(math.isfinite(value) and value > 0 for value in values)
            expected = {
                "p50": statistics.median(values),
                "p90": values[round(0.90 * 59)],
                "p95": values[round(0.95 * 59)],
                "p99": values[round(0.99 * 59)],
                "std": statistics.pstdev(values),
            }
            assert expected == report["timings"][arm][clock]
    times = report["timings"]
    projection = times["cached"]["wall_us"]["p50"] <= (
        0.8 * times["expanded"]["wall_us"]["p50"]
    )
    native = {
        stat: all(
            times["cached"][clock][stat] <= 0.95 * times["native"][clock][stat]
            for clock in ("wall_us", "event_us")
        )
        for stat in ("p50", "p95")
    }
    assert projection == report["projection_prediction_met"]
    assert native["p50"] == report["native_performance_adoption_met"]
    return {
        "status": "pass",
        "scope": "Independent readback of180 measured outputs and clocks; one frozen actual input",
        "source_report_sha256": hashlib.sha256(raw).hexdigest(),
        "readback_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "sample_triplets": len(samples),
        "orders": [
            {"order": list(key), "count": count}
            for key, count in orders.items()
        ],
        "timings": times,
        "paired_wins_cached": {
            arm: sum(
                row["cached"]["wall_us"] < row[arm]["wall_us"]
                for row in samples
            )
            for arm in ("native", "expanded")
        },
        "cached_wall_p50_ratio": {
            arm: times["cached"]["wall_us"]["p50"]
            / times[arm]["wall_us"]["p50"]
            for arm in ("native", "expanded")
        },
        "projection_prediction_met": projection,
        "native_p50_prediction_met": native["p50"],
        "native_p95_condition_met": native["p95"],
        "formal_precision_gate_passed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = readback(args.report)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "sample_triplets",
                    "projection_prediction_met",
                    "native_p50_prediction_met",
                    "native_p95_condition_met",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
