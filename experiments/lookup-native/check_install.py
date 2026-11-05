"""Exercise immutable installation and normal imports in an isolated tree."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from build import DIRECTORY, MODULE
from check import digest


def invoke(root, args, label):
    receipt = args.output / (label + ".json")
    command = [sys.executable, str(Path(__file__).with_name("install.py")),
               "--root", str(root), "--build", str(args.build),
               "--checks", str(args.checks), "--output", str(receipt)]
    process = subprocess.run(command, capture_output=True, text=True)
    (args.output / (label + ".stdout")).write_text(process.stdout)
    (args.output / (label + ".stderr")).write_text(process.stderr)
    return process.returncode, json.loads(receipt.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    build = json.loads(args.build.read_text())
    binary = Path(build["binary"])
    report = {
        "status": "running", "command": [sys.executable, *sys.argv],
        "build_receipt_sha256": digest(args.build),
        "checks_sha256": digest(args.checks), "npu_executed": False,
        "source_sha256": {str(path): digest(path) for path in (
            Path(__file__), Path(__file__).with_name("install.py"),
            Path(__file__).with_name("build.py"),
            Path(__file__).with_name("check.py"),
        )}, "checks": [],
    }
    with TemporaryDirectory(prefix="lookup-install-") as temporary:
        root = Path(temporary)
        for source in (args.root / "tilert").rglob("*.py"):
            target = root / source.relative_to(args.root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        shutil.copyfile(args.root / DIRECTORY / "lookup_draft.pxd",
                        root / DIRECTORY / "lookup_draft.pxd")
        destination = root / DIRECTORY / binary.name
        destination.write_bytes(b"different native artifact")
        conflict_hash = digest(destination)
        code, rejected = invoke(root, args, "conflict")
        assert code == 1 and rejected["status"] == "fail"
        assert "FileExistsError" in rejected["error"]
        assert digest(destination) == conflict_hash
        report["checks"].append("conflicting_file_preserved")
        destination.unlink()
        code, installed = invoke(root, args, "installed")
        assert code == 0 and installed["status"] == "pass"
        assert digest(destination) == build["binary_sha256"]
        script = '''import importlib, json, pathlib, sys
root, name, filename = sys.argv[1:]
sys.path.insert(0, root)
module = importlib.import_module(name)
loaded = pathlib.Path(module.__file__).resolve()
assert loaded == pathlib.Path(filename).resolve()
assert "__pyx_vtable__" in vars(module.LookupDrafter)
assert module.LookupDrafter([1,2,7,9,2,8,2],3,1).propose(1) == [8]
from tilert.pd_vllm.ascend_cache import drafter_metadata
identity = drafter_metadata("lookup")
assert identity["kind"] == "cython"
assert "torch" not in sys.modules and "Cython" not in sys.modules
print(json.dumps(identity, sort_keys=True))
'''
        process = subprocess.run(
            [sys.executable, "-I", "-S", "-c", script, str(root), MODULE,
             str(destination)], capture_output=True, text=True,
        )
        (args.output / "normal-import.stdout").write_text(process.stdout)
        (args.output / "normal-import.stderr").write_text(process.stderr)
        process.check_returncode()
        report["normal_import"] = json.loads(process.stdout)
        assert report["normal_import"]["build"] == installed["installed_load"]
        report["checks"].append("normal_import_without_runtime_cython")
        before = destination.stat()
        code, repeated = invoke(root, args, "repeated")
        after = destination.stat()
        assert code == 0 and repeated["status"] == "pass"
        assert digest(destination) == build["binary_sha256"]
        assert (before.st_ino, before.st_mtime_ns) == (
            after.st_ino, after.st_mtime_ns)
        report["checks"].append("same_artifact_preserves_inode_and_mtime")
    report["source_stable"] = all(
        digest(Path(path)) == sha
        for path, sha in report["source_sha256"].items())
    assert report["source_stable"]
    report["status"] = "pass"
    (args.output / "checks.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"],
                      "checks": report["checks"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
