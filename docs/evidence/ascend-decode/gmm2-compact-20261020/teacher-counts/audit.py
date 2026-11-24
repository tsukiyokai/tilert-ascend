"""Recompute the completed paired GMM2 screen from its frozen raw records."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    ordered = sorted(values)
    return {"p50": statistics.median(values),
            "p90": ordered[round((len(values) - 1) * .9)],
            "p99": ordered[round((len(values) - 1) * .99)],
            "std": statistics.stdev(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.root
    report = json.loads((root / "device-result.json").read_text())
    counts = json.loads((root / "counts.json").read_text())
    access = json.loads((root / "device-access.json").read_text())
    build = json.loads((root / "manifest.json").read_text())
    for name, identity in build["files"].items():
        assert digest(root / name) == identity["sha256"], name
    for field, path in (("counts_sha256", root / "counts.json"),
                        ("probe_sha256", root / "bench_counts.py"),
                        ("validate_sha256", root.parent / "validate.py")):
        assert report[field] == digest(path)
    binary = Path(access["binary"])
    assert digest(binary) == access["sha256"] == build["elf_sha256"]
    assert json.loads(binary.with_suffix(".json").read_text())["sha256"] == digest(binary)
    assert access["opened"] and access["accessed"] and access["returncode"] == 0
    assert json.loads((root / "screen-01-exit.json").read_text())["returncode"] == 0
    assert report["status"] == "pass"
    assert (report["calls_per_graph"], report["banks"], report["warmup_pairs"],
            report["abba_samples"]) == (10, 8, 10, 30)
    assert report["peak_allocated_bytes"] <= report["peak_reserved_bytes"] <= report["quota_bytes"]
    assert len(report["cases"]) == len(counts["cases"]) == 5
    zero_hash = hashlib.sha256(bytes(256 * 6144 * 2)).hexdigest()
    records = []
    case_hashes = []
    for actual, frozen in zip(report["cases"], counts["cases"]):
        assert actual["distribution"] == frozen["name"]
        assert actual["group_counts"] == frozen["counts"]
        dense = frozen["counts"]
        assert len(dense) == 256 and sum(dense) == 256
        active, hot = sum(c > 0 for c in dense), sum(c > 16 for c in dense)
        assert (actual["active"], actual["hot"], actual["maximum"]) == (active, hot, max(dense))
        compact_tasks = sum(12 * ((c + 31) // 32) for c in dense)
        wide_tasks = sum(6 * ((c + 15) // 16) for c in dense)
        assert (actual["compact_tasks"], actual["wide_tasks"]) == (compact_tasks, wide_tasks)
        raw = actual["raw_us"]
        assert len(raw) == 30
        for pair in raw:
            assert set(pair) == {"base512", "wide1024"}
            assert all(len(v) == 2 for v in pair.values())
            assert all(math.isfinite(v) and v > 0
                       for values in pair.values() for v in values)
        for arm in ("base512", "wide1024"):
            assert stats([v for p in raw for v in p[arm]]) == actual[arm + "_us"]
        deltas = [statistics.fmean(p["wide1024"]) - statistics.fmean(p["base512"])
                  for p in raw]
        assert stats(deltas) == actual["paired_delta_us"]
        wins = sum(d < 0 for d in deltas)
        assert wins == actual["wide1024_wins"]
        hashes = actual["output_sha256"]
        assert len(hashes["base512"]) == len(hashes["wide1024"]) == 10
        assert hashes["base512"] == hashes["wide1024"]
        assert all(len(v) == 64 and v != zero_hash for values in hashes.values() for v in values)
        assert actual["outputs_exact"]
        case_hashes.append(hashes["base512"][0])
        base, wide = actual["base512_us"]["p50"], actual["wide1024_us"]["p50"]
        records.append({"case": frozen["name"], "active": active, "hot": hot,
                        "maximum": max(dense), "compact_tasks": compact_tasks,
                        "wide_tasks": wide_tasks,
                        "extra_logical_weight_fraction": hot / active,
                        "compact_p50_us": base, "wide_p50_us": wide,
                        "arm_median_reduction_fraction": (base - wide) / base,
                        "paired_delta_p50_us": statistics.median(deltas),
                        "paired_delta_min_us": min(deltas),
                        "paired_delta_max_us": max(deltas), "wins": wins,
                        "paired_deltas_us": deltas})
    assert len(set(case_hashes)) == 5
    routes = [list(range(8)) for _ in range(32)]
    for row in range(16):
        routes[row][row // 2] = 8 + row // 2
    assert all(len(set(row)) == 8 for row in routes)
    counter = Counter(expert for row in routes for expert in row)
    adversarial = [counter[expert] for expert in range(256)]
    assert adversarial == [30] * 8 + [2] * 8 + [0] * 240
    result = {"status": "pass", "cpu_only": True,
              "output_evidence": "50 matching output-hash pairs and the bound device script's byte comparisons; output tensors were not retained.",
              "source_scope": report["data_scope"],
              "elf_sha256": digest(binary), "build_files_rehashed": len(build["files"]),
              "cases": records,
              "peak_allocated_bytes": report["peak_allocated_bytes"],
              "peak_reserved_bytes": report["peak_reserved_bytes"],
              "input_sha256": {n: digest(root / n) for n in
                               ["device-result.json", "device-access.json", "counts.json", "bench_counts.py", "manifest.json"]},
              "auditor_sha256": digest(Path(__file__)),
              "unmeasured_counterexample": {"counts": adversarial, "active": 16,
                                             "hot": 8, "maximum": 30,
                                             "compact_tasks": 192, "wide_tasks": 144,
                                             "extra_logical_weight_fraction": .5}}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("cases", "unmeasured_counterexample")}, indent=2))
    for case in records:
        print(json.dumps({k: v for k, v in case.items() if k != "paired_deltas_us"}))


if __name__ == "__main__":
    main()
