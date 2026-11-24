#!/usr/bin/env python3
"""Create an isolated decoder overlay without changing the accepted runtime."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = "acac5e7220b51bdd24e41352ffc52f88aac5f58dd9d262b0a43aad2ca9efd0fc"
    if digest(args.extension) != expected:
        raise RuntimeError("The official extension differs from the frozen CPU build")
    parent = args.runtime / "vendor-hook-pinned/ascend-python-build/vllm_ascend"
    args.output.mkdir(parents=True, exist_ok=False)
    package = args.output / "python/vllm_ascend"
    shutil.copytree(parent, package)
    original = {str(p.relative_to(parent)): digest(p) for p in parent.rglob("*") if p.is_file()}
    shutil.copy2(args.extension, package / args.extension.name)
    for name in ("device_forward.py", "decoder_forward.py", "cpu_constructor.py",
                 "device_forward_run.sh", "stage_forward.py"):
        shutil.copy2(Path(__file__).with_name(name), args.output / name)
    actual = {str(p.relative_to(args.output)): digest(p)
              for p in args.output.rglob("*") if p.is_file()}
    for name, sha in original.items():
        if actual["python/vllm_ascend/" + name] != sha:
            raise RuntimeError(f"The overlay changed an official Python source: {name}")
    report = {"scope": "Unchanged official Python package plus the independently built official decoder host extension",
              "runtime": str(args.runtime),
              "runtime_frozen_sha256": digest(args.runtime / "frozen-inputs.sha256"),
              "official_python_file_count": len(original),
              "extension_sha256": expected, "files_sha256": actual}
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    actual["manifest.json"] = digest(args.output / "manifest.json")
    (args.output / "frozen-inputs.sha256").write_text(
        "".join(f"{sha}  {name}\n" for name, sha in sorted(actual.items())))
    print(json.dumps({"path": str(args.output), "file_count": len(actual),
                      "manifest_sha256": actual["manifest.json"],
                      "frozen_inputs_sha256": digest(args.output / "frozen-inputs.sha256")}, indent=2))


if __name__ == "__main__":
    main()
