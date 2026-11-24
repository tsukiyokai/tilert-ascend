"""Record native indexer sources, vendor headers, SDK and artifacts."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--cann", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    vendor = args.vendor.resolve()
    headers = []
    for name in ("lightning_indexer", "quant_lightning_indexer"):
        headers.extend((vendor / "attention" / name / "op_kernel").rglob(
            "*.h"
        ))
    paths = sorted(source.glob("indexer_*.*"))
    paths += [source / "attention_host.h", source / "attention_contract.h"]
    paths += sorted(args.output.parent.glob("*.o"))
    paths += sorted(args.output.parent.glob("*.so"))
    revision = subprocess.run(
        ["git", "-C", str(vendor), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    versions = {}
    for name in ("version.cfg", "version.info"):
        for path in args.cann.rglob(name):
            if path.is_file() and path.stat().st_size < 65536:
                versions[str(path.relative_to(args.cann))] = path.read_text(
                    errors="replace"
                )
    report = {
        "vendor_root": str(vendor),
        "vendor_revision": revision.stdout.strip() or None,
        "vendor_headers": {
            str(path.relative_to(vendor)): sha256(path)
            for path in sorted(set(headers))
        },
        "sources_and_artifacts": {str(path): sha256(path) for path in paths},
        "cann_root": str(args.cann.resolve()),
        "cann_versions": versions,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
