"""Preserve the source and SDK identity of the compiled indexer frontend."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cann", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    paths = sorted(here.glob("indexer_front*.*"))
    paths += [here / name for name in
              ("attention_host.h", "attention_contract.h")]
    paths += [here.parent / "mla_tail" / name for name in
              ("mla_tail.h", "mla_tail_native.h")]
    snapshot = args.output.parent / "sources"
    snapshot.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, snapshot / path.name)
    compiler = subprocess.run([str(args.cann / "bin/bisheng"), "--version"],
                              text=True, capture_output=True, check=True)
    layernorm = args.cann / (
        "opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/dynamic/"
        "layer_norm_v3.py"
    )
    versions = {}
    for name in ("version.cfg", "version.info"):
        for path in args.cann.rglob(name):
            if path.is_file() and path.stat().st_size < 65536:
                versions[str(path.relative_to(args.cann))] = path.read_text(
                    errors="replace"
                )
    report = {
        "sources": {str(path): digest(path) for path in paths},
        "source_snapshot": str(snapshot),
        "artifacts": {str(path): digest(path)
                      for path in args.output.parent.glob("*.so")},
        "cann_root": str(args.cann.resolve()),
        "compiler": compiler.stdout + compiler.stderr,
        "cann_versions": versions,
        "layernorm_reference": {str(layernorm): digest(layernorm)}
            if layernorm.is_file() else None,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
