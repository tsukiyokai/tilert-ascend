"""Compare a saved TP16 decode trace with a regenerated HF oracle on CPU."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from run_native_decode import (
    compare_trace, digest, validate_reference_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)

    native = json.loads((args.native_run / "rank-0.json").read_text())
    manifest = args.reference / "manifest.json"
    reference = json.loads(manifest.read_text())
    validate_reference_manifest(reference, native["metadata_sha256"])
    checks, files = [], {}
    for rank in range(16):
        path = args.native_run / f"rank-{rank}.json"
        report = json.loads(path.read_text())
        if "error" in report:
            raise ValueError(f"Native rank {rank} did not finish")
        for item in report["checks"]:
            if not item["name"].startswith("hf."):
                checks.append({**item, "rank": rank})
        files[str(path)] = digest(path)

    for name, expected_digest in sorted(reference["trace_sha256"].items()):
        expected_path = args.reference / name
        actual_path = args.native_run / "trace" / name
        if digest(expected_path) != expected_digest:
            raise ValueError(f"HF trace hash differs: {name}")
        actual, expected = (
            torch.load(path, map_location="cpu", weights_only=True)
            for path in (actual_path, expected_path)
        )
        compare_trace(f"hf.step{expected['position']}", actual, expected,
                      checks)
        files[str(actual_path)] = digest(actual_path)
        files[str(expected_path)] = expected_digest

    failed = sum(not item["passed"] for item in checks)
    source = Path(__file__)
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU comparison of saved native tensors; no new NPU run",
        "native_run": str(args.native_run),
        "native_source_sha256": native["source_sha256"],
        "reference": str(args.reference),
        "reference_manifest_sha256": digest(manifest),
        "weight_arithmetic": reference["weight_arithmetic"],
        "source_sha256": {
            source.name: digest(source),
            "run_native_decode.py": digest(
                source.with_name("run_native_decode.py")
            ),
        },
        "artifact_sha256": files, "timed": False, "profiled": False,
        "checks": checks, "checks_total": len(checks),
        "checks_failed": failed, "status": "pass" if not failed else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({name: output[name] for name in (
        "status", "checks_total", "checks_failed",
    )}))
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
