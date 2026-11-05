import gzip
import hashlib
import json
import lzma
from pathlib import Path
import shutil
import statistics


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


target = Path("docs/evidence/ascend-decode/lookup-arena-20261028")
target.mkdir(exist_ok=True)
sources = target / "sources"
sources.mkdir(exist_ok=True)
names = (
    "tilert-lookup-arena-prototype-20261028.py",
    "tilert-lookup-arena-tuple-prototype-20261028.py",
    "tilert-lookup-arena-bench-20261028.py",
    "tilert-lookup-arena-noretention-bench-20261028.py",
    "tilert-lookup-arena-summary-20261028.py",
    "tilert-lookup-arena-noretention-summary-20261028.py",
)
for name in names:
    shutil.copyfile(Path("/tmp") / name, sources / name)
shutil.copyfile("tilert/models/glm_5_2_ascend/lookup_draft.py",
                sources / "production-radix.py")
artifacts = {}
panels = []
for prefix in ("bench", "noretention-bench"):
    for policy in ("fresh", "carry"):
        stem = f"tilert-lookup-arena-{prefix}-{policy}-20261028"
        original = Path("/tmp") / (stem + ".json")
        report = json.loads(original.read_text())
        assert report["status"] == "pass" and report["source_stable"]
        for name, expected in report["source_sha256"].items():
            assert digest(Path(name)) == expected, name
        raw = original.with_suffix(".samples.jsonl")
        assert digest(raw) == report["raw_samples_sha256"]
        expected = {case["name"]: case["oracle_signature_sha256"]
                    for case in report["workloads"]}
        counts = {}
        totals = {}
        measured = warmup = 0
        generations = {}
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
                observed = [sum(c["generation"] == g
                                for c in row["gc_collections"])
                            for g in range(3)]
                if row["mode"] != "gc_on_plain":
                    assert observed == delta
                if row["mode"] == "gc_off_observed":
                    assert delta == [0, 0, 0]
                assert all(value >= 0 for value in row["gc_overlap_ns"].values())
                if row["warmup"]:
                    warmup += 1
                    continue
                measured += 1
                key = row["workload"], row["mode"], row["arm"]
                counts[key] = counts.get(key, 0) + 1
                totals.setdefault(key, []).append(row["total_ns"] / 1e6)
                for collection in row["gc_collections"]:
                    label = (row["mode"], row["arm"], collection["phase"],
                             collection["generation"])
                    generations[label] = generations.get(label, 0) + 1
        assert measured == 1080 and warmup == 72
        assert len(counts) == 36 and set(counts.values()) == {30}
        for item in report["results"]:
            for arm, result in item["arms"].items():
                key = item["workload"], item["mode"], arm
                assert statistics.median(totals[key]) == result[
                    "total_ms"]["p50"]
        panels.append({
            "name": stem, "measured": measured, "warmup": warmup,
            "signature_and_time_sums_verified": True,
            "all_observed_generation_counts_match_gc_stats": True,
            "gc_off_has_no_collection": True,
            "all_total_p50_readbacks_match": True,
            "observed_collections": [
                {"mode": key[0], "arm": key[1], "phase": key[2],
                 "generation": key[3], "count": value}
                for key, value in sorted(generations.items())
            ],
        })
        for path in (original, raw):
            suffix = ".xz" if path == raw else ".gz"
            destination = target / (path.name + suffix)
            with path.open("rb") as input_stream, destination.open("wb") as out:
                compressor = (lzma.LZMAFile(out, mode="wb", preset=9)
                              if path == raw else gzip.GzipFile(
                                  fileobj=out, mode="wb", filename="", mtime=0))
                with compressor as compressed:
                    shutil.copyfileobj(input_stream, compressed)
            artifacts[destination.name] = {"sha256": digest(destination),
                                           "payload_sha256": digest(path)}
        print(stem, "verified and saved", flush=True)
for name in ("tilert-lookup-arena-correctness-20261028.json",
             "tilert-lookup-arena-tuple-correctness-20261028.json",
             "tilert-lookup-arena-plain-summary-20261028.json",
             "tilert-lookup-arena-noretention-plain-summary-20261028.json"):
    shutil.copyfile(Path("/tmp") / name, target / name)
(target / "independent-audit.json").write_text(json.dumps({
    "status": "pass", "panels": panels, "artifacts": artifacts,
    "stored_bytes": sum(path.stat().st_size for path in target.rglob("*")
                        if path.is_file()),
}, indent=2) + "\n")
shutil.copyfile(__file__, sources / Path(__file__).name)
