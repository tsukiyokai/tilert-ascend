"""Validate and run the unchanged original-FFN Decoder gates with M1 WK."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(manifest, mode, device, report_name):
    if not re.fullmatch(r"[0-9]+", device):
        raise ValueError("Use an integer physical device")
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", report_name):
        raise ValueError("Use a simple report name")
    result = list(manifest["commands"][mode])
    result[2], result[3] = device, report_name
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("eager", "graph"))
    parser.add_argument("device")
    parser.add_argument("report_name")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report = json.loads(Path(__file__).with_name("manifest.json").read_text())
    for path, expected in report["required_sha256"].items():
        if digest(Path(path)) != expected:
            raise RuntimeError(f"Frozen gate input changed: {path}")
    invocation = command(report, args.mode, args.device, args.report_name)
    print(json.dumps({"mode": args.mode, "command": invocation,
                      "verified_files": len(report["required_sha256"]),
                      "executes_device": args.execute}), flush=True)
    if args.execute:
        os.execvp(invocation[0], invocation)


if __name__ == "__main__":
    main()
