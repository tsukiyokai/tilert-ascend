"""Pair unchanged Python and bound native lookup on frozen CPU workloads."""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

from check import digest, load, load_build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--gc-start", choices=("fresh", "carry"),
                        default="fresh")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0:
        parser.error("Positive samples and nonnegative warmup required")
    args.root = args.root.resolve()
    sys.path.insert(0, str(args.root))
    if gc.callbacks:
        raise RuntimeError("Use a process without existing GC callbacks")
    checks = json.loads(args.checks.read_text())
    if (checks["status"] != "pass"
            or checks["build_receipt_sha256"] != digest(args.build)):
        raise ValueError("Native audit and build receipt must match")
    native, build, identity = load_build(args.build, args.root)
    lookup = args.root / "tilert/models/glm_5_2_ascend/lookup_draft.py"
    python = load("lookup_python_baseline", lookup)
    sampler = load("lookup_coordinate_sampler", args.root /
                   "experiments/lookup-index/coordinate_probe.py")
    paths = [Path(__file__).resolve(), Path(__file__).with_name("check.py"),
             Path(sampler.__file__), lookup, lookup.with_suffix(".pxd"),
             Path(build["binary"]), args.build.resolve(),
             args.checks.resolve(),
             args.reference.resolve(), args.root /
             "tilert/pd_vllm/ascend_cache.py", args.root / sampler.NGRAM]
    hashes = {str(path): digest(path) for path in paths}
    reference, decoded_hash = sampler.read_reference(args.reference)
    scorer = [sha for path, sha in reference["source_sha256"].items()
              if path.endswith(sampler.NGRAM)]
    if (reference["status"] != "pass"
            or scorer != [digest(args.root / sampler.NGRAM)]):
        raise ValueError("Frozen oracle and current reverse scan differ")
    cases = [{name: case[name] for name in (
        "name", "prompt_token_ids", "output_token_ids", "oracle",
    )} for case in reference["results"]]
    del reference
    if [case["name"] for case in cases] != list(sampler.CASE_NAMES):
        raise ValueError("Expected the four fixed 8K+1024 workloads")
    for case in cases:
        sampler.validate_case(case)
    initial_collected = gc.collect()
    args.modes = list(sampler.MODES)
    raw_path = args.output.with_suffix(".samples.jsonl")
    if args.output.exists() or raw_path.exists():
        raise FileExistsError("Choose new immutable output paths")
    parameters = {
        "width": sampler.WIDTH, "longest": sampler.LONGEST,
        "shortest": sampler.SHORTEST, "prompt_length": sampler.PROMPT_LENGTH,
        "output_length": sampler.OUTPUT_LENGTH, "samples": args.samples,
        "warmup": args.warmup, "gc_start": args.gc_start,
        "modes": args.modes, "gc_thresholds": gc.get_threshold(),
    }
    report = {
        "status": "running", "scope": "CPU setup+propose+commit only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "source_sha256": hashes,
        "git_head": sampler.git_commit(args.root, "HEAD"),
        "hostname": platform.node(), "python": sys.version,
        "machine": platform.machine(), "native": identity,
        "baseline": {"kind": "python", "file": str(lookup),
                     "file_sha256": digest(lookup)},
        "same_algorithm_source": digest(lookup) ==
        identity["build"]["source_sha256"][lookup.name],
        "parameters": parameters,
        "parameters_sha256": hashlib.sha256(json.dumps(
            parameters, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest(),
        "reference_decoded_sha256": decoded_hash,
        "initial_plan_collected": initial_collected,
        "environment_control": "No affinity or clock locking",
        "p99_scope": "Nearest-rank p99 of 30 requests is their maximum",
        "gc_start": args.gc_start, "modes": args.modes,
        "warmup": args.warmup, "samples": args.samples,
        "arm_permutations": sampler.ORDERS,
        "raw_samples": str(raw_path), "teardown_in_total": False,
        "driver_retains_round_objects_during_measurement": False,
        "commit_inputs_prepared_as_tuples": True,
        "acceptance_replayed_before_trials": True,
        "actual_proposals_checked_and_buffered": True,
        "callback_changes_execution": True,
        "gc_stats_scope": "Setup, calls, validation, teardown, counter reads",
        "workloads": [{"name": case["name"], "prompt_token_ids":
                       case["prompt_token_ids"], "output_token_ids":
                       case["output_token_ids"], "oracle_signature_sha256":
                       case["oracle_signature_sha256"]} for case in cases],
        "npu_executed": False, "performance_measured": False, "results": [],
    }
    assert report["same_algorithm_source"]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        if args.check_only:
            report["status"] = "checked"
        else:
            factories = {"baseline": python.LookupDrafter,
                         "candidate": native.LookupDrafter}
            sampler.measure(args, cases, factories, raw_path)
            report["results"] = sampler.summarize(raw_path, args.modes)
            report["status"] = "pass"
        from tilert.pd_vllm.ascend_cache import drafter_metadata

        assert drafter_metadata("lookup") == identity
        assert "torch" not in sys.modules and "Cython" not in sys.modules
        assert list(gc.get_threshold()) == list(parameters["gc_thresholds"])
    except Exception:
        report.update(status="fail", error=traceback.format_exc())
    report["source_stable"] = all(digest(Path(path)) == sha
                                  for path, sha in hashes.items())
    report["head_stable"] = (
        sampler.git_commit(args.root, "HEAD") == report["git_head"])
    if not report["source_stable"] or not report["head_stable"]:
        report["status"] = "fail"
    if raw_path.exists():
        report["raw_samples_sha256"] = digest(raw_path)
        with raw_path.open() as stream:
            report["raw_sample_count"] = sum(1 for _ in stream)
        report["performance_measured"] = report["raw_sample_count"] > 0
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "source_stable": report["source_stable"]}), flush=True)
    return int(report["status"] not in ("pass", "checked"))


if __name__ == "__main__":
    raise SystemExit(main())
