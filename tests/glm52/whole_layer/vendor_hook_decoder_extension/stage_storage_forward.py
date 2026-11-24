#!/usr/bin/env python3
"""Remove diagnostic fences while retaining the accepted NZ descriptor repair."""

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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = args.parent / "frozen-inputs.sha256"
    files = {}
    for line in frozen.read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(args.parent / name) != expected:
            raise RuntimeError(f"The accepted storage diagnostic changed: {name}")
        files[name] = expected
    if digest(args.parent / "decoder_forward.py") != "c802ae9371103bb75c1f8f5df39e3c9b2e49d77fd3a86ef23ad5d58781fcb207":
        raise RuntimeError("The original asynchronous forward changed")
    source = Path(__file__).parent
    original = (args.parent / "decoder_storage_diagnostic.py").read_text()
    replacements = {
        '"""Normalize only W4A8 carrier descriptors before the original diagnostic."""':
        '"""Normalize W4A8 carrier descriptors before the original asynchronous forward."""',
        "from decoder_diagnostic import run_decoder_forward as original_diagnostic":
        "from decoder_forward import run_decoder_forward as original_forward",
        "One-time INT32 NZ carrier descriptors before the unchanged original decoder/GMM diagnostic":
        "One-time INT32 NZ carrier descriptors before the unchanged original asynchronous decoder",
        "return original_diagnostic(torch, layers, config, state, args)":
        "return original_forward(torch, layers, config, state, args)",
    }
    expected_helper = original
    for old, new in replacements.items():
        if expected_helper.count(old) != 1:
            raise RuntimeError("Unexpected accepted helper source")
        expected_helper = expected_helper.replace(old, new)
    if (source / "decoder_storage_forward.py").read_text() != expected_helper:
        raise RuntimeError("The alias helper changed beyond the original-forward delegation")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in files:
        if name == "manifest.json":
            continue
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.parent / name, target)
    for name in ("decoder_storage_forward.py", "stage_storage_forward.py"):
        shutil.copy2(source / name, args.output / name)
    driver = (args.parent / "device_forward_storage.py").read_text()
    old = "from decoder_storage_diagnostic import run_decoder_forward"
    if driver.count(old) != 1:
        raise RuntimeError("Unexpected diagnostic entry import")
    (args.output / "device_forward_storage_async.py").write_text(driver.replace(
        old, "from decoder_storage_forward import run_decoder_forward"))
    launcher = (args.parent / "device_storage_run.sh").read_text()
    launcher = launcher.replace("vendor-hook-decoder212-storage-v1",
                                "vendor-hook-decoder212-storage-forward-v2")
    launcher = launcher.replace("device_forward_storage.py", "device_forward_storage_async.py")
    (args.output / "device_storage_forward_run.sh").write_text(launcher)
    identities = {str(path.relative_to(args.output)): digest(path)
                  for path in args.output.rglob("*") if path.is_file()}
    report = {
        "scope": "Accepted one-time NZ descriptor repair with original asynchronous DecoderLayer forward",
        "parent": str(args.parent), "parent_manifest_sha256": digest(args.parent / "manifest.json"),
        "parent_frozen_sha256": digest(frozen),
        "driver_change": "Helper import only",
        "helper_changes": replacements,
        "official_python_extension_alias_library": "Unchanged from storage-v1",
        "gmm_hooks_and_extra_pre_post_fences": False,
        "original_forward_sha256": digest(args.output / "decoder_forward.py"),
        "new_hbm_bytes": 0, "files_sha256": identities,
    }
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    identities["manifest.json"] = digest(manifest)
    (args.output / "frozen-inputs.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in sorted(identities.items())))
    print(json.dumps({"path": str(args.output), "file_count": len(identities),
                      "manifest_sha256": digest(manifest),
                      "frozen_inputs_sha256": digest(args.output / "frozen-inputs.sha256")}, indent=2))


if __name__ == "__main__":
    main()
