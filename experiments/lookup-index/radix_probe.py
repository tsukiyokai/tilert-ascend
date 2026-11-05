"""Compare the current exact lookup with a frozen Git baseline on CPU."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from types import ModuleType

from rfind_probe import Reverse32, raw_sample, summarize, workloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-ref", default="1ef8a8d")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root), str(root / "tests/glm52")]
    from bench_lookup import replay
    from tilert.models.glm_5_2_ascend.lookup_draft import LookupDrafter

    relative = "tilert/models/glm_5_2_ascend/lookup_draft.py"
    baseline_ref = subprocess.check_output(
        ["git", "rev-parse", args.baseline_ref + "^{commit}"],
        cwd=root, text=True,
    ).strip()
    baseline_source = subprocess.check_output(
        ["git", "show", baseline_ref + ":" + relative], cwd=root,
    )
    baseline = ModuleType("baseline_lookup")
    exec(compile(baseline_source, baseline_ref + ":" + relative, "exec"),
         baseline.__dict__)
    sources = [Path(__file__), Path(__file__).with_name("rfind_probe.py"),
               root / "tests/glm52/bench_lookup.py",
               root / "tests/glm52/ngram_acceptance.py", root / relative]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sources}
    collision = [30000, 30712, 30000, 32048, 30503]
    other = [30306, 30000, 30004, 30000, 30000]
    context = collision + [111] + other + [222] + collision[-4:]
    context += [333] + collision
    expected = Reverse32(context, longest=5).propose(1)
    actual = LookupDrafter(context, longest=5).propose(1)
    assert expected == actual == [111]
    report = {
        "status": "running", "command": [sys.executable, *sys.argv],
        "scope": "Local CPU proposer calls only; synthetic token streams, "
                 "no target inference, orchestration, I/O or NPU/TPOT claim",
        "python": platform.python_version(), "machine": platform.machine(),
        "hostname": platform.node(), "width": 32,
        "longest": 32, "shortest": 2,
        "warmup": args.warmup, "samples": args.samples,
        "clock": "perf_counter_ns; nanoseconds in raw samples",
        "control": "Alternate baseline/radix order; independent reverse "
                   "scan is outside timed arms",
        "environment_control": "No CPU affinity or clock locking",
        "garbage_collection": {"enabled": gc.isenabled(),
                               "thresholds": list(gc.get_threshold())},
        "baseline_ref": baseline_ref,
        "baseline_source_sha256": hashlib.sha256(
            baseline_source).hexdigest(),
        "source_sha256": hashes,
        "collision_case": {
            "context": context, "longest": 5, "expected": expected,
            "candidate": actual,
            "baseline": baseline.LookupDrafter(context, longest=5).propose(1),
        },
        "results": [],
    }
    factories = {"baseline": baseline.LookupDrafter, "radix": LookupDrafter}
    for name, (prompt, tokens) in workloads().items():
        print(json.dumps({"name": name, "phase": "reverse_scan_oracle"}),
              flush=True)
        oracle = replay(Reverse32, prompt, tokens, 32)
        item = {"name": name, "prompt_token_ids": prompt,
                "output_token_ids": tokens, "oracle": oracle,
                "arms": {label: [] for label in factories}}
        for sample in range(-args.warmup, args.samples):
            names = list(factories)
            if sample % 2:
                names.reverse()
            for order, label in enumerate(names):
                before = gc.get_stats()
                value = replay(factories[label], prompt, tokens, 32)
                after = gc.get_stats()
                assert value["signature_sha256"] == oracle["signature_sha256"]
                if sample >= 0:
                    raw = raw_sample(value, sample, order)
                    raw["gc_collections"] = [
                        right["collections"] - left["collections"]
                        for left, right in zip(before, after)
                    ]
                    item["arms"][label].append(raw)
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
