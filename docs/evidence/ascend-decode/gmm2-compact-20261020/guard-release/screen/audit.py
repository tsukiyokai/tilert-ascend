"""Audit the four production GMM2 arms from their raw JSON records."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

ARMS = ("A1", "B1", "B2", "A2")


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
    fixture = json.loads((root / "counts.json").read_text())
    identities = json.loads((root / "identity.json").read_text())
    records, arm_info, input_hashes = {}, {}, {}
    for arm in ARMS:
        record = json.loads((root / (arm + ".json")).read_text())
        access = json.loads((root / (arm + "-access.json")).read_text())
        assert json.loads((root / (arm + "-exit.json")).read_text())["returncode"] == 0
        expected = identities[arm[0]]
        binary = Path(expected["binary"])
        assert record["elf_path"] == access["binary"] == str(binary)
        assert record["elf_sha256"] == access["sha256"] == expected["elf_sha256"] == digest(binary)
        assert access["opened"] and access["accessed"] and access["returncode"] == 0
        assert record["descriptor_sha256"] == digest(binary.with_suffix(".json"))
        for path, sha in expected["sources"].items():
            assert digest(Path(path)) == sha
        assert record["status"] == "pass" and record["arm"] == arm
        assert (record["banks"], record["calls_per_graph"], record["warmup"], record["samples"]) == (8, 10, 10, 30)
        assert len(set(record["weight_pointers"])) == 8
        assert record["probe_sha256"] == digest(root / "bench.py")
        assert record["fixture_sha256"] == digest(root / "counts.json")
        assert record["bank_builder_sha256"] == digest(root.parent / "validate.py")
        assert record["peak_allocated_bytes"] <= record["peak_reserved_bytes"] <= record["quota_bytes"] == 2 * 1024**3
        assert len(record["cases"]) == len(fixture["cases"]) == 8
        records[arm] = record
        preflight = json.loads((root / (arm + "-preflight.json")).read_text())
        assert len(preflight["samples"]) == 2
        assert all(s["aicore"] == 0 and s["free_mib"] >= 6000 for s in preflight["samples"])
        arm_info[arm] = {"elf_sha256": record["elf_sha256"], "preflight": preflight,
                         "peak_allocated_bytes": record["peak_allocated_bytes"],
                         "peak_reserved_bytes": record["peak_reserved_bytes"]}
        for name in (arm + ".json", arm + "-access.json", arm + "-exit.json",
                     arm + "-preflight.json"):
            input_hashes[name] = digest(root / name)
    cases = []
    for index, frozen in enumerate(fixture["cases"]):
        values = {arm: records[arm]["cases"][index] for arm in ARMS}
        count_bytes = struct.pack("<256q", *frozen["counts"])
        expected_count = hashlib.sha256(count_bytes).hexdigest()
        expected_route = hashlib.sha256(count_bytes * 10).hexdigest()
        first = values["A1"]
        for arm, case in values.items():
            assert case["name"] == frozen["name"]
            for field in ("active", "hot", "maximum"):
                assert case[field] == frozen[field]
            mode = "new_expected_wide" if arm[0] == "B" else "old_expected_wide"
            assert case["source_predicted_wide"] == frozen[mode]
            assert case["counts_sha256"] == expected_count
            assert case["route_sha256"] == expected_route
            for field in ("x_sha256", "token_scale_sha256", "output_sha256", "output_concat_sha256"):
                assert case[field] == first[field], (arm, frozen["name"], field)
            assert len(case["output_sha256"]) == 10 and case["eager_graph_first_exact"]
            for field, summary in (("raw_us", "us"), ("raw_host_us", "host_us")):
                assert len(case[field]) == 30
                assert all(math.isfinite(v) and v > 0 for v in case[field])
                assert stats(case[field]) == case[summary]
        medians = {arm: values[arm]["us"]["p50"] for arm in ARMS}
        deltas = [(values["B1"]["raw_us"][i] + values["B2"]["raw_us"][i]
                   - values["A1"]["raw_us"][i] - values["A2"]["raw_us"][i]) / 2
                  for i in range(30)]
        forward = medians["B1"] - medians["A1"]
        reverse = medians["B2"] - medians["A2"]
        cases.append({"name": frozen["name"], "arm_p50_us": medians,
                      "arm_host_p50_us": {arm: values[arm]["host_us"]["p50"] for arm in ARMS},
                      "old_expected_wide": frozen["old_expected_wide"],
                      "new_expected_wide": frozen["new_expected_wide"],
                      "pooled_old_us": stats(values["A1"]["raw_us"] + values["A2"]["raw_us"]),
                      "pooled_new_us": stats(values["B1"]["raw_us"] + values["B2"]["raw_us"]),
                      "balanced_arm_median_delta_us": (forward + reverse) / 2,
                      "B1_minus_A1_us": forward, "B2_minus_A2_us": reverse,
                      "A2_minus_A1_us": medians["A2"] - medians["A1"],
                      "B2_minus_B1_us": medians["B2"] - medians["B1"],
                      "adjacent_arm_sign_conflict": forward * reverse < 0,
                      "ordinal_balanced_delta_us": stats(deltas),
                      "ordinal_balanced_wins": sum(d < 0 for d in deltas),
                      "ordinal_balanced_raw_us": deltas})
    rejected = {}
    for directory, arm in (("rejected-01", "A1"), ("rejected-02", "B2")):
        path = root / directory / (arm + "-preflight.json")
        rejected[directory] = json.loads(path.read_text())
        assert any(s["aicore"] != 0 or s["free_mib"] < 6000
                   for s in rejected[directory]["samples"])
        assert not (path.parent / (arm + ".json")).exists()
        input_hashes[str(path.relative_to(root))] = digest(path)
    result = {"status": "pass", "cpu_only": True,
              "timing_scope": "Four-process ABBA; ordinal samples are not adjacent in time; all raw samples retained.",
              "output_scope": "80 corresponding call positions have equal hashes in all four arms; raw output tensors were not retained.",
              "arms": arm_info, "cases": cases, "rejected_preflights": rejected,
              "input_sha256": input_hashes,
              "auditor_sha256": digest(Path(__file__)),
              "source_sha256": {n: digest(root / n) for n in
                                ("bench.py", "counts.json", "identity.json", "run-arm.sh", "launch-abba.py", "resume-abba.py")}}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for case in cases:
        print(json.dumps({k: v for k, v in case.items() if k != "ordinal_balanced_raw_us"}))


if __name__ == "__main__":
    main()
