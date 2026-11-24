import gzip
import hashlib
import json
import lzma
from pathlib import Path
import shutil
import statistics
import subprocess


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


target = Path("docs/evidence/ascend-decode/lookup-coordinate-20261028")
target.mkdir(exist_ok=True)
sources = target / "sources"
sources.mkdir(exist_ok=True)
baseline = subprocess.check_output([
    "git", "show", "019c97b:tilert/models/glm_5_2_ascend/lookup_draft.py",
])
baseline_hash = hashlib.sha256(baseline).hexdigest()
for name in ("tilert-lookup-local-variants-20261028.py",
             "tilert-lookup-local-audit-20261028.py",
             "tilert-lookup-local-bench-20261028.py"):
    shutil.copyfile(Path("/tmp") / name, sources / name)
shutil.copyfile("/tmp/tilert-lookup-local-audit-20261028.json",
                target / "prototype-audit.json")
panels, artifacts, tails = [], {}, []
for kind, original_stem, arms in (
    ("prototype", "tilert-lookup-local-bench", 3),
    ("paired", "tilert-coordinate-probe", 2),
):
    for policy in ("fresh", "carry"):
        original = Path("/tmp") / f"{original_stem}-{policy}-20261028.json"
        report = json.loads(original.read_text())
        assert report["status"] == "pass" and report["source_stable"]
        for name, expected in report["source_sha256"].items():
            if name.endswith("tilert/models/glm_5_2_ascend/lookup_draft.py"):
                assert expected == baseline_hash or digest(Path(name)) == expected
            else:
                assert digest(Path(name)) == expected, name
        if kind == "paired":
            assert report["head_stable"]
            identity = report["implementation_sources"]
            assert identity["baseline_source_sha256"] == baseline_hash
            assert digest(Path(identity["candidate_loaded_path"])) == (
                identity["candidate_source_sha256"])
            actual = hashlib.sha256(json.dumps(
                report["parameters"], sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest()
            assert actual == report["parameters_sha256"]
        raw = original.with_suffix(".samples.jsonl")
        assert digest(raw) == report["raw_samples_sha256"]
        expected = {case["name"]: case["oracle_signature_sha256"]
                    for case in report["workloads"]}
        counts, totals, pairs = {}, {}, {}
        warmup = measured = 0
        with raw.open() as stream:
            for line in stream:
                row = json.loads(line)
                rounds = row["rounds"]
                actual = hashlib.sha256(json.dumps([
                    (r["position"], r["proposal"], r["accepted"])
                    for r in rounds
                ]).encode()).hexdigest()
                assert actual == row["signature_sha256"] == expected[
                    row["workload"]]
                steady = sum(r["propose_ns"] + r["commit_ns"] for r in rounds)
                assert steady == row["steady_ns"]
                assert row["total_ns"] == row["setup_ns"] + steady
                delta = [after["collections"] - before["collections"]
                         for before, after in zip(row["gc_stats_before"],
                                                  row["gc_stats_after"])]
                if row["mode"] != "gc_on_plain":
                    assert delta == [sum(c["generation"] == g
                                         for c in row["gc_collections"])
                                     for g in range(3)]
                if row["mode"] == "gc_off_observed":
                    assert delta == [0, 0, 0]
                if row["warmup"]:
                    warmup += 1
                    continue
                measured += 1
                key = row["workload"], row["mode"], row["arm"]
                counts[key] = counts.get(key, 0) + 1
                totals.setdefault(key, []).append(row["total_ns"] / 1e6)
                pair = pairs.setdefault((row["workload"], row["mode"],
                                         row["sample"]), {})
                pair[row["arm"]] = row["total_ns"]
                if (kind == "paired" and policy == "carry"
                        and row["workload"] == "copy_5pct_edits"
                        and row["arm"] == "candidate" and (
                            (row["mode"] == "gc_on_observed"
                             and row["sample"] == 18)
                            or (row["mode"] == "gc_on_plain"
                                and row["sample"] == 24))):
                    position = 686 if row["mode"] == "gc_on_observed" else 863
                    interval = next(value for value in row["phase_intervals"]
                                    if value[:2] == ["commit", position])
                    call = next(value for value in rounds
                                if value["position"] == position)
                    overlap = sum(max(0, min(interval[3], c["stop_ns"]) -
                                      max(interval[2], c["start_ns"]))
                                  for c in row["gc_collections"])
                    assert interval[3] - interval[2] == call["commit_ns"]
                    tails.append({
                        "panel": f"{kind}-{policy}", "mode": row["mode"],
                        "ordinal": row["ordinal"], "sample": row["sample"],
                        "position": position, "accepted": call["accepted"],
                        "call_interval_ns": interval[2:],
                        "commit_ns": call["commit_ns"],
                        "observed_gc_overlap_ns": overlap,
                        "gc_stats_collection_delta": delta,
                        "collections": [c for c in row["gc_collections"]
                                        if min(interval[3], c["stop_ns"]) >
                                        max(interval[2], c["start_ns"])],
                        "plain_call_cause_known": False,
                    })
        modes = len(report["modes"])
        assert measured == 4 * modes * arms * 30
        assert warmup == 4 * modes * arms * 2
        assert set(counts.values()) == {30}
        for result in report["results"]:
            for arm, data in result["arms"].items():
                key = result["workload"], result["mode"], arm
                assert statistics.median(totals[key]) == data["total_ms"]["p50"]
            if kind == "paired":
                selected = [value for key, value in pairs.items()
                            if key[:2] == (result["workload"], result["mode"])]
                wins = sum(pair["candidate"] < pair["baseline"]
                           for pair in selected)
                assert wins == result["paired_wins"]["total"]["wins"]
        panels.append({"name": f"{kind}-{policy}", "measured": measured,
                       "warmup": warmup, "all_signatures_match": True,
                       "all_time_sums_and_total_p50_match": True,
                       "all_observed_collection_counts_match_stats": True,
                       "gc_off_has_no_collection": True})
        for path, suffix in ((original, ".json.gz"),
                             (raw, ".samples.jsonl.xz")):
            destination = target / (f"{kind}-{policy}" + suffix)
            with path.open("rb") as input_stream, destination.open("wb") as out:
                compressor = (lzma.LZMAFile(out, mode="wb", preset=9)
                              if path == raw else gzip.GzipFile(
                                  fileobj=out, mode="wb", filename="", mtime=0))
                with compressor as compressed:
                    shutil.copyfileobj(input_stream, compressed)
            payload = hashlib.sha256()
            opener = lzma.open if path == raw else gzip.open
            with opener(destination, "rb") as restored:
                for block in iter(lambda: restored.read(1024**2), b""):
                    payload.update(block)
            assert payload.hexdigest() == digest(path)
            artifacts[destination.name] = {"sha256": digest(destination),
                                           "payload_sha256": digest(path)}
        print(kind, policy, "all samples verified and saved", flush=True)
(target / "independent-audit.json").write_text(json.dumps({
    "status": "pass", "panels": panels, "tail_observations": tails,
    "artifacts": artifacts,
}, indent=2) + "\n")
shutil.copyfile(__file__, sources / Path(__file__).name)
