"""Audit a bound lookup binary through oracle, session and HTTP entries."""

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
import weakref


MODULE = "tilert.models.glm_5_2_ascend.lookup_draft"
BASIC = (
    "check_lookup", "check_lookup_benchmark", "check_round_invariance",
    "check_mtp_round", "check_mtp_session", "check_mtp_tp",
    "check_cann_lookup_threads",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_build(path, root):
    report = json.loads(path.read_text())
    if report["status"] != "pass":
        raise ValueError("A passing native build receipt is required")
    binary = Path(report["binary"])
    if digest(binary) != report["binary_sha256"]:
        raise ValueError("Native binary differs from its build receipt")
    for name, expected in report["native_identity"]["source_sha256"].items():
        source = root / "tilert/models/glm_5_2_ascend" / name
        if digest(source) != expected:
            raise ValueError("Current lookup sources differ from the build")
    module = load(MODULE, binary)
    if ("__pyx_vtable__" not in vars(module.LookupDrafter)
            or "context" not in vars(module.LookupDrafter)):
        raise ValueError("Native build did not augment the lookup classes")
    import tilert.models.glm_5_2_ascend as package

    package.lookup_draft = module
    from tilert.pd_vllm.ascend_cache import drafter_metadata

    identity = drafter_metadata("lookup")
    assert identity["build"] == report["isolated_load"]
    assert identity["file"] == str(binary.resolve())
    return module, report, identity


def boundary_checks(cls):
    checks = []
    for bound in (32, 2**100):
        drafter = cls([2**100, -1, 3, 2**100, -1], bound, 1)
        assert drafter.propose(2**100) == [3, 2**100, -1]
        checks.append("arbitrary_integers_" + str(bound))
    assert cls([1, 2, 3], 2**100, 2**100).propose(1) == []
    assert cls([1, 2, 3, 1, 2], 3, 1).propose(-1) == []
    checks.extend(("unreachable_huge_tail", "negative_count"))
    drafter = cls([1, 2, 3])
    drafter.annotation = "request-local"
    assert drafter.__dict__["annotation"] == "request-local"
    drafter.cycle = drafter
    reference = weakref.ref(drafter)
    del drafter
    gc.collect()
    assert reference() is None
    checks.extend(("dynamic_attributes", "weakref_cycle_collected"))
    return checks


def run_suite(args):
    sys.path[:0] = [str(args.root), str(args.root / "tests/glm52")]
    module, build, identity = load_build(args.build, args.root)
    cls = module.LookupDrafter
    paths = [Path(__file__).resolve(), args.build,
             *(args.root / "tilert/models/glm_5_2_ascend").glob("*.py"),
             *(args.root / "tilert/pd_vllm").glob("*.py"),
             *(args.root / "tests/glm52" / (name + ".py")
               for name in (*BASIC, "check_cann_lookup_pd", "bench_lookup",
                            "ngram_acceptance", "serve_cann_pd",
                            "bench_cann_mtp"))]
    hashes = {str(path): digest(path) for path in paths}
    report = {
        "status": "running", "command": [sys.executable, *sys.argv],
        "python": platform.python_version(), "native": identity,
        "build_receipt_sha256": digest(args.build),
        "checker_sha256": digest(Path(__file__)), "suite": args.suite,
        "source_sha256": hashes,
        "boundary_checks": boundary_checks(cls), "npu_executed": False,
    }
    if args.suite == "basic":
        audit = load("lookup_oracle_audit",
                     args.root / "experiments/lookup-index/audit.py")
        report.update(
            bad_compression=audit.check_bad_compression(cls),
            proposals=audit.check_proposals(cls),
            sessions=audit.check_sessions(cls),
            oracle_auditor_sha256=digest(Path(audit.__file__)),
        )
    names = BASIC if args.suite == "basic" else ("check_cann_lookup_pd",)
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    if args.suite == "basic":
        assert sys.modules["check_lookup"].LookupDrafter is cls
        assert "torch" not in sys.modules
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    from tilert.pd_vllm.ascend_cache import drafter_metadata

    assert drafter_metadata("lookup") == identity
    report["source_stable"] = all(digest(Path(path)) == expected
                                  for path, expected in hashes.items())
    assert report["source_stable"]
    report.update(tests_run=result.testsRun,
                  status="pass" if result.wasSuccessful() else "fail")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"],
                      "suite": args.suite, "tests": result.testsRun}))
    return int(not result.wasSuccessful())


def guard_checks(args, build):
    script = '''import importlib.util, pathlib, sys
root, filename, mutation = sys.argv[1:]
sys.path.insert(0, root)
name = "tilert.models.glm_5_2_ascend.lookup_draft"
path = pathlib.Path(filename)
spec = importlib.util.spec_from_file_location(name, filename)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
if mutation.startswith("before_"):
    source = path.with_name("lookup_draft." + mutation[7:])
    source.write_bytes(source.read_bytes() + b"\\n# changed\\n")
    try:
        spec.loader.exec_module(module)
    except ImportError:
        print("rejected")
    else:
        raise AssertionError("Stale extension imported")
else:
    spec.loader.exec_module(module)
    from tilert.pd_vllm.ascend_cache import drafter_metadata
    assert drafter_metadata("lookup")["build"] == module._NATIVE_BUILD
    changed = path if mutation == "after_binary" else path.with_name(
        "lookup_draft." + mutation[6:])
    replacement = changed.with_name(changed.name + ".replacement")
    replacement.write_bytes(changed.read_bytes() + b"\\n# changed\\n")
    replacement.replace(changed)
    try:
        drafter_metadata("lookup")
    except RuntimeError:
        print("rejected")
    else:
        raise AssertionError("Changed imported file accepted")
assert "torch" not in sys.modules and "Cython" not in sys.modules
'''
    results = []
    for mutation in ("before_py", "before_pxd", "after_py", "after_pxd",
                     "after_binary"):
        with tempfile.TemporaryDirectory(prefix="lookup-guard-") as temp:
            directory = Path(temp)
            binary = Path(build["binary"])
            for path in (binary, binary.with_name("lookup_draft.py"),
                         binary.with_name("lookup_draft.pxd")):
                shutil.copyfile(path, directory / path.name)
            process = subprocess.run(
                [sys.executable, "-I", "-S", "-c", script, str(args.root),
                 str(directory / binary.name), mutation],
                capture_output=True, text=True, check=True,
            )
            assert process.stdout.strip() == "rejected"
            results.append({"mutation": mutation, "rejected": True})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("basic", "http"))
    args = parser.parse_args()
    args.root, args.build = args.root.resolve(), args.build.resolve()
    if args.suite:
        return run_suite(args)
    args.output.mkdir(parents=True, exist_ok=False)
    build = json.loads(args.build.read_text())
    report = {
        "status": "running", "scope": "Native CPU lookup; no NPU proof",
        "command": [sys.executable, *sys.argv], "suites": [],
        "build_receipt_sha256": digest(args.build),
        "checker_sha256": digest(Path(__file__)),
        "guards": guard_checks(args, build),
    }
    for suite in ("basic", "http"):
        output = args.output / (suite + ".json")
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--root", str(args.root), "--build", str(args.build),
                   "--output", str(output), "--suite", suite]
        with (args.output / (suite + ".log")).open("w") as stream:
            process = subprocess.run(command, stdout=stream,
                                     stderr=subprocess.STDOUT)
        report["suites"].append({
            "suite": suite, "returncode": process.returncode,
            "report": str(output),
            "report_sha256": digest(output) if output.exists() else None,
        })
    report["source_stable"] = (
        digest(Path(__file__)) == report["checker_sha256"])
    passed = report["source_stable"] and all(
        item["returncode"] == 0 and json.loads(
            Path(item["report"]).read_text())["checker_sha256"] ==
        report["checker_sha256"] for item in report["suites"])
    report["status"] = "pass" if passed else "fail"
    (args.output / "checks.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"],
                      "output": str(args.output)}))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
