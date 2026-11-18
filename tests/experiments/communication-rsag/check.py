"""Compile and run actual private layouts/tree plans and schedule negatives."""

import argparse
import json
from pathlib import Path
import subprocess

from prepare import HERE, PARENT, digest, prepare


def check(parent, output):
    manifest = prepare(parent, output)
    command = ["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
               "-Wno-sign-compare", "-fno-fast-math", "-ffp-contract=off",
               "-I", str(output), str(HERE / "contract.cpp"),
               "-o", str(output / "contract")]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = subprocess.run([str(output / "contract")], check=True,
                            capture_output=True, text=True)
    report = {
        "scope": "compiled C++ layout/tree and abstract protocol schedule",
        "command": command, "result": result.stdout.strip(),
        "contract_sha256": digest(HERE / "contract.cpp"),
        "binary_sha256": digest(output / "contract"),
        "manifest_sha256": digest(output / "manifest.json"),
        "device_validation": "pending",
    }
    expected = manifest["parent_sha256"]["communication_tree/reduce.inc"]
    for arm in ("control", "candidate"):
        if digest(output / arm / "source/reduce.inc") != expected:
            raise ValueError("The accepted tree reducer identity changed")
    (output / "cpu-check.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.parent.resolve(), args.output.resolve()),
                     indent=2))


if __name__ == "__main__":
    main()
