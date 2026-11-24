"""Record the compiled MLA diagnostic and the SDK used to build it."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cann", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=["profile", "dequant"],
                        default="profile")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    names = ["mla_tail.h", "mla_tail_native.h", "build_profile.sh",
             "profile_provenance.py", f"mla_tail_{args.kind}.h",
             f"mla_tail_{args.kind}_kernel.cpp"]
    names += (["profile_stages.py"] if args.kind == "profile" else
              ["build_dequant.sh", "dequant_check.py"])
    sources = [source / name for name in names]
    snapshot = args.output.parent / "sources"
    snapshot.mkdir(parents=True, exist_ok=True)
    for path in sources:
        shutil.copy2(path, snapshot / path.name)
    arch = args.cann / (platform.machine() + "-linux")
    sdk = [arch / path for path in (
        "asc/impl/basic_api/dav_c220/kernel_operator_sys_var_impl.h",
        "asc/impl/basic_api/dav_c220/kernel_operator_sync_impl.h",
        "asc/include/basic_api/kernel_operator_cache_intf.h",
        "asc/include/adv_api/matmul/matmul_intf.h",
    )]
    compiler = subprocess.run([str(args.compiler), "--version"],
                              capture_output=True, text=True, check=True)
    versions = {}
    for name in ("version.cfg", "version.info"):
        for path in args.cann.rglob(name):
            if path.is_file() and path.stat().st_size < 65536:
                versions[str(path.relative_to(args.cann))] = path.read_text(
                    errors="replace"
                )
    report = {
        "kind": args.kind,
        "cann_root": str(args.cann.resolve()),
        "compiler": compiler.stdout + compiler.stderr,
        "cann_versions": versions,
        "sources": {str(path): digest(path) for path in sources},
        "source_snapshot": str(snapshot),
        "sdk_headers": {str(path): digest(path) for path in sdk
                        if path.is_file()},
        "artifacts": {str(path): digest(path)
                      for path in sorted(args.output.parent.glob("*.so"))},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
