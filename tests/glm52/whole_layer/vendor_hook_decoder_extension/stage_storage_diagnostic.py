#!/usr/bin/env python3
"""Freeze the descriptor-only correction beside the original GMM diagnostic."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--alias-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = args.parent / "frozen-inputs.sha256"
    files = {}
    for line in frozen.read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(args.parent / name) != expected:
            raise RuntimeError(f"The frozen diagnostic changed: {name}")
        files[name] = expected
    if digest(args.parent / "decoder_diagnostic.py") != "2042c77a5466ac6122e0d7678eb6ccc268ec0d5508d7bdd7c4b1c2616e956538":
        raise RuntimeError("Use the original synchronized diagnostic-v1")
    build = json.loads((args.alias_build / "manifest.json").read_text())
    if (not build["completed"] or build["npu_initialized"]
            or digest(args.alias_build / "nz_alias.so") != build["library_sha256"]):
        raise RuntimeError("The isolated alias helper did not pass its actual CPU ABI gate")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in files:
        if name == "manifest.json":
            continue
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.parent / name, target)
    source = Path(__file__).parent
    for name in ("decoder_storage_diagnostic.py", "stage_storage_diagnostic.py",
                 "nz_alias.cpp", "test_storage_descriptor.py", "decode_exception.py"):
        shutil.copy2(source / name, args.output / name)
    shutil.copy2(args.alias_build / "nz_alias.so", args.output / "nz_alias.so")
    shutil.copy2(args.alias_build / "manifest.json", args.output / "nz-alias-manifest.json")
    driver = (args.parent / "device_forward_diagnostic.py").read_text()
    old = "from decoder_diagnostic import run_decoder_forward"
    if driver.count(old) != 1:
        raise RuntimeError("Unexpected diagnostic entry import")
    (args.output / "device_forward_storage.py").write_text(driver.replace(
        old, "from decoder_storage_diagnostic import run_decoder_forward"))
    launcher = (args.parent / "device_diagnostic_run.sh").read_text()
    launcher = launcher.replace("vendor-hook-decoder212-diagnostic-v1",
                                "vendor-hook-decoder212-storage-v1")
    launcher = launcher.replace("device_forward_diagnostic.py", "device_forward_storage.py")
    (args.output / "device_storage_run.sh").write_text(launcher)
    identities = {str(path.relative_to(args.output)): digest(path)
                  for path in args.output.rglob("*") if path.is_file()}
    report = {
        "scope": "Original diagnostic with one-time storage-preserving NZ INT32 descriptor aliases",
        "parent": str(args.parent), "parent_manifest_sha256": digest(args.parent / "manifest.json"),
        "parent_frozen_sha256": digest(frozen), "alias_build": build,
        "driver_change": "Only the helper import changes; the original diagnostic and decoder bodies remain unchanged",
        "official_python_and_extension": "Unchanged from diagnostic-v1",
        "new_hbm_bytes": 0, "files_sha256": identities,
    }
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    identities["manifest.json"] = digest(manifest)
    (args.output / "frozen-inputs.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in sorted(identities.items())))
    print(json.dumps({"path": str(args.output), "file_count": len(identities),
                      "manifest_sha256": digest(manifest),
                      "frozen_inputs_sha256": digest(args.output / "frozen-inputs.sha256"),
                      "library_sha256": build["library_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
