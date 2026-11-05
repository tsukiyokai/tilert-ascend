"""Read frozen no-retention samples; report plain metrics and paired wins."""

import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = [Path("/tmp/tilert-lookup-arena-noretention-bench-" + start + "-20261028.json")
             for start in ("fresh", "carry")]
    sources = {str(Path(__file__).resolve()): digest(Path(__file__))}
    results = []
    for path in paths:
        sources[str(path)] = digest(path)
        report = json.loads(path.read_text())
        raw = Path(report["raw_samples"])
        sources[str(raw)] = digest(raw)
        assert report["status"] == "pass" and report["source_stable"]
        assert sources[str(raw)] == report["raw_samples_sha256"]
        assert report["raw_sample_count"] == 1152
        pairs = {}
        with raw.open() as stream:
            for line in stream:
                sample = json.loads(line)
                if sample["warmup"] or sample["mode"] != "gc_on_plain":
                    continue
                pairs.setdefault((sample["workload"], sample["sample"]), {})[
                    sample["arm"]] = {field: sample[field + "_ns"]
                                      for field in ("setup", "steady", "total")}
        for item in report["results"]:
            if item["mode"] != "gc_on_plain":
                continue
            output = {"gc_start": report["gc_start"],
                      "workload": item["workload"], "arms": {}}
            for arm, values in item["arms"].items():
                row = {
                    "setup_p50_ms": values["setup_ms"]["p50"],
                    "steady_p50_ms": values["steady_ms"]["p50"],
                    "total_p50_ms": values["total_ms"]["p50"],
                    "commit_p99_us": values["commit_us"]["p99"],
                    "steady_p99_ms": values["steady_ms"]["p99"],
                }
                if arm != "production":
                    matching = [pair for (name, _), pair in pairs.items()
                                if name == item["workload"]]
                    assert len(matching) == 30
                    row["paired_wins_over_production"] = {
                        field: sum(pair[arm][field] < pair["production"][field]
                                   for pair in matching)
                        for field in ("setup", "steady", "total")
                    }
                output["arms"][arm] = row
            results.append(output)
            print(report["gc_start"], item["workload"])
            for arm, row in output["arms"].items():
                print(" ", arm, "setup/steady/total p50 ms",
                      "/".join(f"{row[field + '_p50_ms']:.3f}"
                               for field in ("setup", "steady", "total")),
                      "commit p99 us", f"{row['commit_p99_us']:.3f}",
                      "steady p99 ms", f"{row['steady_p99_ms']:.3f}",
                      "wins", row.get("paired_wins_over_production", {}))
    output = {"status": "pass", "source_sha256": sources, "results": results,
              "scope": "Fixed-oracle no-retention CPU control; input allocation "
                       "also changes and is not index-only attribution"}
    path = Path("/tmp/tilert-lookup-arena-noretention-plain-summary-20261028.json")
    path.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
