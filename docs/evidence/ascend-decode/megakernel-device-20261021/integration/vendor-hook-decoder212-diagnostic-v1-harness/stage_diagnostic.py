#!/usr/bin/env python3
"""Freeze a synchronized diagnostic beside the unchanged decoder fixture."""

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
    for line in frozen.read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(args.parent / name) != expected:
            raise RuntimeError(f"The original decoder fixture changed: {name}")
    expected_forward = "c802ae9371103bb75c1f8f5df39e3c9b2e49d77fd3a86ef23ad5d58781fcb207"
    expected_device = "9bf415fe02963ddf87e57ed00a1825c9e97f5cdce0016bb1389b3a7c86505b79"
    if (digest(args.parent / "decoder_forward.py") != expected_forward
            or digest(args.parent / "device_forward.py") != expected_device):
        raise RuntimeError("Use the original failed forward-v1 source")
    source = Path(__file__).parent
    original = (args.parent / "device_forward.py").read_text()
    diagnostic = (source / "device_forward_diagnostic.py").read_text()
    if diagnostic.replace("from decoder_diagnostic import run_decoder_forward",
                          "from decoder_forward import run_decoder_forward") != original:
        raise RuntimeError("The diagnostic driver changed more than the helper import")
    shutil.copytree(args.parent, args.output)
    for name in ("decoder_diagnostic.py", "device_forward_diagnostic.py",
                 "device_diagnostic_run.sh", "stage_diagnostic.py",
                 "test_decoder_diagnostic.py"):
        shutil.copy2(source / name, args.output / name)
    files = {str(path.relative_to(args.output)): digest(path)
             for path in args.output.rglob("*")
             if path.is_file() and path.name not in ("manifest.json", "frozen-inputs.sha256")}
    report = {
        "scope": "Original forward-v1 with GMM1 synchronization and persistent argument diagnostics",
        "parent": str(args.parent), "parent_frozen_sha256": digest(frozen),
        "parent_manifest_sha256": digest(args.parent / "manifest.json"),
        "original_forward_sha256": expected_forward,
        "original_device_sha256": expected_device,
        "device_change": "Only the run_decoder_forward import selects the diagnostic wrapper",
        "official_python_and_extension": "Unchanged from forward-v1",
        "files_sha256": files,
    }
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    files["manifest.json"] = digest(manifest)
    frozen_output = args.output / "frozen-inputs.sha256"
    frozen_output.write_text("".join(
        f"{sha}  {name}\n" for name, sha in sorted(files.items())))
    print(json.dumps({"path": str(args.output), "file_count": len(files),
                      "manifest_sha256": digest(manifest),
                      "frozen_inputs_sha256": digest(frozen_output)}, indent=2))


if __name__ == "__main__":
    main()
