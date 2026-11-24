"""Record the exact native source, vendor headers and build artifacts."""

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
    parser.add_argument("--norm-vendor", type=Path)
    parser.add_argument("--with-indexer", action="store_true")
    parser.add_argument("--indexed-profile", action="store_true")
    parser.add_argument("--indexed-parallel", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    vendor = args.vendor.resolve()
    vendor_paths = []
    for relative in (
        "attention/mla_prolog/op_kernel",
        "attention/sparse_flash_attention/op_kernel",
    ):
        vendor_paths.extend((vendor / relative).rglob("*.h"))
    if args.with_indexer:
        for name in ("lightning_indexer", "quant_lightning_indexer"):
            vendor_paths.extend((vendor / "attention" / name /
                                 "op_kernel").rglob("*.h"))
    revision = subprocess.run(
        ["git", "-C", str(vendor), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False,
    )
    paths = sorted(source.glob("attention_*.*"))
    if args.with_indexer:
        paths.extend(source / name for name in (
            "indexer_native.h", "indexer_contract.h", "indexer_tiling.h",
            "indexer_front_native.h", "indexer_front_contract.h",
            "indexer_front_tiling.cpp",
        ))
        schedule = source / "indexer_front_schedule.h"
        if schedule.exists():
            paths.append(schedule)
    paths.extend(source.parent / "mla_tail" / name for name in (
        "mla_tail.h", "mla_tail_native.h", "mla_tail_tiling.cpp",
    ) if (source.parent / "mla_tail" / name).exists())
    paths.extend(sorted(args.output.parent.glob("*.o")))
    paths.extend(sorted(args.output.parent.glob("*.so")))
    norm_paths = []
    if args.norm_vendor:
        paths.extend(source / name for name in (
            "norm_native.h", "norm_contract.h",
        ))
        norm_directory = (
            args.norm_vendor / "csrc/moe/add_rms_norm_bias/op_kernel"
        )
        norm_paths = [norm_directory / name for name in (
            "add_rms_norm_bias_single_n.h", "rms_norm_base.h",
            "reduce_common.h",
        )]
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
            for path in sorted(set(vendor_paths))
        },
        "sources_and_artifacts": {str(path): sha256(path) for path in paths},
        "norm_vendor_headers": {
            str(path): sha256(path) for path in norm_paths
        },
        "cann_root": str(args.cann.resolve()),
        "cann_versions": versions,
    }
    if args.indexed_profile:
        report["indexed_profile"] = {
            "namespace": "tilert_indexed_profile",
            "config_namespace": "whole_layer_indexed_profile",
            "launch": "whole_layer_indexed_attention_profile_launch",
            "kernel": "whole_layer_indexed_attention_profile",
            "timer": "GetSystemCycle, A3 SYS_CNT 50 MHz",
            "trace": "six local timestamps per worker; existing barriers retained",
        }
    if args.indexed_parallel:
        variant = "profile" if args.indexed_profile else "parallel"
        report["indexed_qk_parallel"] = {
            "namespace": f"tilert_indexed_{variant}",
            "config_namespace": f"whole_layer_indexed_{variant}",
            "launch": f"whole_layer_indexed_attention_{variant}_launch",
            "kernel": f"whole_layer_indexed_attention_{variant}",
            "flag": "INDEXER_FRONT_PARALLEL_PROJECTIONS",
            "schedule": "Q tiles 0..15, K tiles 16..20 when groups >= 21",
            "fallback": "original sequential Q/K assignment below 21 groups",
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
