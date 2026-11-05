"""Pair tracked and acyclic native lookup nodes on frozen CPU workloads."""

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import sys
import traceback

from verify import ROOT, digest, load, load_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--baseline-build", type=Path, required=True)
    parser.add_argument("--candidate-build", type=Path, required=True)
    parser.add_argument("--candidate-checks", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--gc-start", choices=("fresh", "carry"),
                        default="fresh")
    parser.add_argument("--modes", nargs="+", default=[
        "gc_on_plain", "gc_on_observed", "gc_off_observed"])
    args = parser.parse_args()
    args.root = args.root.resolve()
    sys.path.insert(0, str(args.root))
    sampler = load("lookup_gc_sampler",
                   args.root / "experiments/lookup-index/coordinate_probe.py")
    if (args.samples < 1 or args.warmup < 0
            or len(set(args.modes)) != len(args.modes)
            or any(mode not in sampler.MODES for mode in args.modes)):
        parser.error("Positive samples, nonnegative warmup, unique GC modes")
    if gc.callbacks:
        raise RuntimeError("Use a process without existing GC callbacks")
    checks = json.loads(args.candidate_checks.read_text())
    if (checks["status"] != "pass" or checks["build_receipt_sha256"]
            != digest(args.candidate_build)):
        raise ValueError("Candidate checks and build receipt differ")
    builder = load("lookup_native_builder",
                   args.root / "experiments/lookup-native/build.py")
    modules, builds = {}, {}
    for arm in ("baseline", "candidate"):
        modules[arm], builds[arm] = load_native(
            getattr(args, arm + "_build"), builder)
        assert gc.is_tracked(modules[arm]._Node(0, 0, -1)) == (
            arm == "baseline")
        assert gc.is_tracked(modules[arm].LookupDrafter([1, 2, 1, 2]))
    same_source = (builds["baseline"]["native_identity"]["source_sha256"] ==
                   builds["candidate"]["native_identity"]["source_sha256"])
    assert same_source
    for field in ("compiler_version", "compiler_config",
                  "compiler_environment"):
        assert builds["baseline"][field] == builds["candidate"][field]
    for field in ("cython", "setuptools", "python_abi", "python", "machine",
                  "directives"):
        assert builds["baseline"]["native_identity"][field] == (
            builds["candidate"]["native_identity"][field])
    paths = [Path(__file__), Path(__file__).with_name("verify.py"),
             Path(sampler.__file__), Path(builder.__file__),
             args.reference.resolve(), args.candidate_checks.resolve(),
             args.root / sampler.NGRAM]
    for arm, build in builds.items():
        paths.extend((getattr(args, arm + "_build").resolve(),
                      Path(build["binary"]), Path(build["generated_c"]),
                      Path(build["generated_c"]).with_suffix(".py")))
        paths.extend(Path(build["binary"]).with_name(name) for name in
                     build["native_identity"]["source_sha256"])
        lowering = build["native_identity"].get("lowering_sha256")
        if lowering is not None:
            path = Path(__file__).with_name("build.py")
            assert digest(path) == lowering
            paths.append(path)
    hashes = {str(path.resolve()): digest(path) for path in paths}
    reference, decoded_hash = sampler.read_reference(args.reference)
    scorer = [sha for name, sha in reference["source_sha256"].items()
              if name.endswith(sampler.NGRAM)]
    assert reference["status"] == "pass"
    assert scorer == [digest(args.root / sampler.NGRAM)]
    cases = [{name: case[name] for name in (
        "name", "prompt_token_ids", "output_token_ids", "oracle",
    )} for case in reference["results"]]
    del reference
    assert [case["name"] for case in cases] == list(sampler.CASE_NAMES)
    for case in cases:
        sampler.validate_case(case)
    initial_collected = gc.collect()
    raw_path = args.output.with_suffix(".samples.jsonl")
    if args.output.exists() or raw_path.exists():
        raise FileExistsError("Choose new immutable output paths")
    threshold, enabled = gc.get_threshold(), gc.isenabled()
    report = {
        "status": "running", "command": [sys.executable, *sys.argv],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Native node GC ablation; CPU setup+propose+commit only",
        "hostname": platform.node(), "python": sys.version,
        "machine": platform.machine(), "source_sha256": hashes,
        "git_head": sampler.git_commit(args.root, "HEAD"),
        "initial_plan_collected": initial_collected,
        "reference_decoded_sha256": decoded_hash,
        "same_python_and_pxd": same_source,
        "native_builds": {arm: {
            "receipt_sha256": digest(getattr(args, arm + "_build")),
            "binary": build["binary"], "binary_sha256": build["binary_sha256"],
            "loaded_identity": modules[arm]._NATIVE_BUILD,
            "node_tracked": arm == "baseline", "drafter_tracked": True,
        } for arm, build in builds.items()},
        "parameters": {"width": 32, "longest": 32, "shortest": 2,
                       "prompt_length": 8192, "output_length": 1024,
                       "samples": args.samples, "warmup": args.warmup,
                       "gc_start": args.gc_start, "modes": args.modes,
                       "gc_thresholds": threshold},
        "environment_control": "No affinity or clock locking",
        "screening_only": args.samples < 30,
        "gc_start": args.gc_start, "modes": args.modes,
        "warmup": args.warmup, "samples": args.samples,
        "arm_permutations": sampler.ORDERS, "raw_samples": str(raw_path),
        "driver_retains_round_objects_during_measurement": False,
        "teardown_in_total": False, "actual_proposals_checked_and_buffered":
        True, "commit_inputs_prepared_as_tuples": True,
        "callback_changes_execution": True, "npu_executed": False,
        "workloads": [{"name": case["name"], "prompt_token_ids":
                       case["prompt_token_ids"], "output_token_ids":
                       case["output_token_ids"], "oracle_signature_sha256":
                       case["oracle_signature_sha256"]} for case in cases],
        "results": [],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        factories = {arm: module.LookupDrafter for arm, module in
                     modules.items()}
        sampler.measure(args, cases, factories, raw_path)
        report["results"] = sampler.summarize(raw_path, args.modes)
        assert gc.get_threshold() == threshold and gc.isenabled() == enabled
        assert "torch" not in sys.modules and "Cython" not in sys.modules
        report["status"] = "pass"
    except Exception:
        report.update(status="fail", error=traceback.format_exc())
    report["source_stable"] = all(digest(Path(path)) == sha for path, sha in
                                  hashes.items())
    report["head_stable"] = (
        sampler.git_commit(args.root, "HEAD") == report["git_head"])
    if not report["source_stable"] or not report["head_stable"]:
        report["status"] = "fail"
    if raw_path.exists():
        report["raw_samples_sha256"] = digest(raw_path)
        with raw_path.open() as stream:
            report["raw_sample_count"] = sum(1 for _ in stream)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "error": report.get("error")}), flush=True)
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
