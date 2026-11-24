"""Compare exact packed-token rfind with length-indexed lookup on CPU."""

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
from time import perf_counter_ns


class PackedFind:
    def __init__(self, prompt, longest=32, shortest=2):
        if shortest < 1 or longest < shortest:
            raise ValueError("Invalid tail lengths")
        self.context = []
        self.data = bytearray()
        self.longest = longest
        self.shortest = shortest
        self.commit(prompt)

    def commit(self, tokens):
        added = [int(token) for token in tokens]
        self.context.extend(added)
        self.data.extend(array("I", added).tobytes())

    def propose(self, count):
        maximum = min(self.longest, len(self.context) - 1)
        for size in range(maximum, self.shortest - 1, -1):
            pattern = bytes(self.data[-size * 4:])
            stop = len(self.data) - 4
            while True:
                start = self.data.rfind(pattern, 0, stop)
                if start < 0:
                    break
                if start % 4 == 0:
                    end = start // 4 + size
                    return self.context[end:end + count]
                # Retain earlier occurrences overlapping the unaligned hit.
                stop = start + len(pattern) - 1
        return []


class Reverse32:
    def __init__(self, prompt, longest=32, shortest=2):
        self.context = list(prompt)
        self.longest = longest
        self.shortest = shortest

    def commit(self, tokens):
        self.context.extend(tokens)

    def propose(self, count):
        from ngram_acceptance import lookup

        proposals = lookup(self.context, count, self.longest, self.shortest)
        return proposals[0] if proposals else []


def workloads():
    rng = random.Random(419)
    base = [rng.randrange(1000, 10000) for _ in range(256)]
    copied = base * 4
    edited = copied[:]
    for index in rng.sample(range(1024), 51):
        edited[index] = rng.randrange(1000, 10000)
    ambiguous_prompt = [value for index in range(2730)
                        for value in (10000 + index, 5, 6)] + [500, 501]
    ambiguous_tokens = [value for index in range(342)
                        for value in (60000 + index, 5, 6)][:1024]
    return {
        "exact_copy": (base * 32, copied),
        "copy_5pct_edits": (base * 32, edited),
        "ambiguous_bigram": (ambiguous_prompt, ambiguous_tokens),
        "random_no_match": (
            [rng.randrange(1000, 50000) for _ in range(8192)],
            rng.sample(range(70000, 150000), 1024),
        ),
    }


def check_equivalence():
    from ngram_acceptance import lookup

    rng = random.Random(20261028)
    checks = 0
    for shortest, longest in ((1, 1), (1, 8), (2, 32), (4, 12), (32, 32)):
        context = [rng.randrange(7) for _ in range(40)]
        candidate = PackedFind(context, longest, shortest)
        for _ in range(80):
            for count in (0, 1, 7, 31):
                proposals = lookup(context, count, longest, shortest)
                assert candidate.propose(count) == (
                    proposals[0] if proposals else [])
                checks += 1
            added = [rng.randrange(7) for _ in range(rng.randrange(9))]
            context.extend(added)
            candidate.commit(added)
    overlap = PackedFind([0, 0, 256, 7, 0, 0], longest=2)
    assert overlap.data.rfind(bytes(8), 0, len(overlap.data) - 4) == 1
    assert overlap.propose(1) == [256]
    unaligned = PackedFind([256, 65536, 0, 1, 256], longest=2)
    assert unaligned.propose(31) == []
    return {"incremental_proposal_checks": checks,
            "unaligned_false_hit_rejected": True,
            "earlier_aligned_overlap_preserved": True}


def stats(values):
    values = sorted(values)
    return {"p50": statistics.median(values),
            "p95": values[math.ceil(len(values) * 0.95) - 1],
            "p99": values[math.ceil(len(values) * 0.99) - 1],
            "std": statistics.pstdev(values)}


def raw_sample(value, sample, order):
    times = [[row["propose_ns"], row["commit_ns"]]
             for row in value["rounds"]]
    return {"sample": sample, "order": order,
            "setup_ns": value["setup_ns"],
            "steady_ns": sum(sum(pair) for pair in times),
            "total_ns": value["total_ns"],
            "signature_sha256": value["signature_sha256"],
            "round_propose_commit_ns": times}


def summarize(samples):
    return {
        "setup_ms": stats([item["setup_ns"] / 1e6 for item in samples]),
        "steady_request_cpu_ms": stats([
            item["steady_ns"] / 1e6 for item in samples]),
        "total_request_cpu_ms": stats([
            item["total_ns"] / 1e6 for item in samples]),
        "propose_us": stats([
            pair[0] / 1e3 for item in samples
            for pair in item["round_propose_commit_ns"]]),
        "commit_us": stats([
            pair[1] / 1e3 for item in samples
            for pair in item["round_propose_commit_ns"]]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root), str(root / "tests/glm52")]
    from bench_lookup import replay
    from tilert.models.glm_5_2_ascend.lookup_draft import LookupDrafter

    assert array("I").itemsize == 4
    sources = [Path(__file__), root / "tests/glm52/bench_lookup.py",
               root / "tests/glm52/ngram_acceptance.py",
               root / "tilert/models/glm_5_2_ascend/lookup_draft.py"]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sources}
    report = {
        "status": "running", "command": [sys.executable, *sys.argv],
        "python": platform.python_version(), "machine": platform.machine(),
        "hostname": platform.node(), "width": 32,
        "longest": 32, "shortest": 2,
        "warmup": args.warmup, "samples": args.samples,
        "scope": "Local CPU proposer microprobe; synthetic token workloads, "
                 "no target inference or NPU/TPOT conclusion",
        "token_domain": "Valid GLM token IDs, stored as native uint32",
        "clock": "perf_counter_ns; nanoseconds in raw samples",
        "control": "Independent reverse-scan oracle is outside timed arms; "
                   "alternate current/rfind order each sample",
        "environment_control": "No CPU affinity or clock locking",
        "source_sha256": hashes, "equivalence": check_equivalence(),
        "results": [],
    }
    factories = {"current": LookupDrafter, "rfind": PackedFind}
    for name, (prompt, tokens) in workloads().items():
        print(json.dumps({"workload": name, "phase": "reverse_scan_oracle"}),
              flush=True)
        started = perf_counter_ns()
        oracle = replay(Reverse32, prompt, tokens, 32)
        item = {
            "name": name, "prompt_token_ids": prompt,
            "output_token_ids": tokens, "oracle": oracle,
            "oracle_elapsed_ns": perf_counter_ns() - started,
            "arms": {label: [] for label in factories},
        }
        for sample in range(-args.warmup, args.samples):
            names = list(factories)
            if sample % 2:
                names.reverse()
            for order, label in enumerate(names):
                value = replay(factories[label], prompt, tokens, 32)
                assert value["signature_sha256"] == oracle["signature_sha256"]
                if sample >= 0:
                    item["arms"][label].append(raw_sample(value, sample, order))
        item["summary"] = {label: summarize(samples)
                           for label, samples in item["arms"].items()}
        item["all_round_signatures_equal_independent_reverse_scan"] = True
        report["results"].append(item)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"name": name, "rounds": len(oracle["rounds"]),
                          "summary": item["summary"]}), flush=True)
    report["source_stable"] = all(
        hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha
        for name, sha in hashes.items())
    report["status"] = "pass" if report["source_stable"] else "source_changed"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
