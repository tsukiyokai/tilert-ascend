"""Freeze the accepted serial probe, private tree and unmodified root driver."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


DRIVER_MANIFEST_SHA = "6e7c6ef717e53c6d967b3f41a5c18a62ae8e3d218c1aaeada11eb9d51d51cf50"
BUILD_MANIFEST_SHA = "22d408d22afcccfce6941a0a9cd5c2cdc3e38c53110401f21cb46832b4b79716"
BUILD_REPORT_SHA = "b857a54c47aa61deb5429e81ee84651fdec7e05f840343d7251f64c41b6961b3"
PARENT_SHA = "1be5fa375d0d6e7a2a6dcf7f5265cf886317108398aeec940e2b1ce0de04e22f"
CANDIDATE_SHA = "a0bd2462e2f999280720fa5060466159a2eab13ce5541639ddc7afef35a45ebf"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(path, expected):
    if digest(path) != expected:
        raise ValueError(f"A frozen input changed: {path}")


def stage(driver, build, parent, output):
    require(driver / "manifest.json", DRIVER_MANIFEST_SHA)
    require(build / "manifest.json", BUILD_MANIFEST_SHA)
    require(build / "results/build.json", BUILD_REPORT_SHA)
    require(parent, PARENT_SHA)
    require(build / "results/communication_tree_probe.so", CANDIDATE_SHA)
    driver_manifest = json.loads((driver / "manifest.json").read_text())
    build_manifest = json.loads((build / "manifest.json").read_text())
    report = json.loads((build / "results/build.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    for name, expected in driver_manifest["files_sha256"].items():
        require(driver / name, expected)
        shutil.copy2(driver / name, output / name)
    shutil.copy2(driver / "manifest.json", output / "driver-manifest.json")
    for name, expected in build_manifest["sources"].items():
        require(build / name, expected)
        target = output / "build" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(build / name, target)
    shutil.copy2(build / "manifest.json", output / "build/manifest.json")
    for name in ("build.json", "commands.jsonl", "compiler-version.log",
                 "host-compiler-version.log", "python-contract.log",
                 "plan-contract.log", "exports.log", "device-compile.log",
                 "host-compile.log", "probe-host-compile.log", "link.log"):
        shutil.copy2(build / "results" / name, output / "build" / name)
    shutil.copy2(parent, output / "parent_probe.so")
    shutil.copy2(build / "results/communication_tree_probe.so", output / "candidate_probe.so")
    for name in ("stage_device.py", "device_cpu_check.py"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    manifest = {
        "scope": "Immutable three-rank serial/tree/FP32-HCCL device fixture; not executed at freeze",
        "driver_manifest_sha256": DRIVER_MANIFEST_SHA,
        "build_manifest_sha256": BUILD_MANIFEST_SHA,
        "build_report_sha256": BUILD_REPORT_SHA,
        "parent_sha256": PARENT_SHA, "candidate_sha256": CANDIDATE_SHA,
        "private_c_prefix": "communication_tree_probe_",
        "shmem_lifetime": "Only parent initialize/finalize; per-arm prepare/destroy with distinct windows",
        "shmem_commit": report["shmem_commit"],
        "shmem_header_manifest_sha256": report["shmem_header_manifest_sha256"],
        "shmem_libraries": report["shmem_libraries"],
        "chunk_elements": 2048, "fp32_stack_slots": 5, "phase_ub_bytes": 45056,
        "build_source_count": 22, "compile_dependency_count": 859,
        "files_sha256": {str(path.relative_to(output)): digest(path)
                         for path in sorted(output.rglob("*")) if path.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    files = dict(manifest["files_sha256"])
    files["manifest.json"] = digest(output / "manifest.json")
    (output / "frozen-inputs.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in sorted(files.items())))
    print(json.dumps({"fixture": str(output), "manifest_sha256": files["manifest.json"],
                      "frozen_input_count": len(files)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("driver", "build", "parent", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    stage(*(getattr(args, name).resolve() for name in
            ("driver", "build", "parent", "output")))
