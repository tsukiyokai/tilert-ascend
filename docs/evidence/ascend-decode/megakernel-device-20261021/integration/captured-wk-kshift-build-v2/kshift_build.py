#!/usr/bin/env python3
"""Build the isolated BF16 Mmad K-order probe with one resident L0C tile."""

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
    args.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    for name in ("kshift_kernel.cpp", "kshift_check.py", "kshift_build.py", "check.py"):
        shutil.copy2(here / name, args.output / name)
    cann = args.cann.resolve()
    arch = cann / "aarch64-linux"
    includes = ["-I" + str(cann / "include")]
    for relative in (
        "asc", "asc/include", "asc/include/adv_api", "asc/include/basic_api",
        "asc/include/c_api", "asc/include/simt_api", "asc/include/utils",
        "asc/include/basic_api/reg_compute", "asc/impl/adv_api",
        "asc/impl/basic_api", "asc/impl/c_api", "asc/impl/simt_api",
        "asc/impl/utils", "asc/impl/basic_api/reg_compute", "tikcpp/tikcfw",
        "tikcpp/tikcfw/lib", "tikcpp/tikcfw/impl", "tikcpp/tikcfw/interface",
    ):
        if (arch / relative).is_dir():
            includes.append("-I" + str(arch / relative))
    compiler = next(path for path in (
        cann / "bin/bisheng", cann / "compiler/ccec_compiler/bin/bisheng"
    ) if path.is_file())
    commands = [
        [str(compiler), "-xcce", "--npu-arch=dav-2201", "-std=c++17", "-O2",
         "-fPIC", *includes, "-c", str(args.output / "kshift_kernel.cpp"),
         "-o", str(args.output / "kshift_kernel.o")],
        ["g++", "-shared", str(args.output / "kshift_kernel.o"), "-o",
         str(args.output / "wk_kshift.so"), "-L" + str(arch / "lib64"),
         "-lascendcl", "-lruntime", "-lunified_dlog", "-lc_sec", "-lprofapi"],
    ]
    manifest = {
        "scope": "M16 N16 K128 BF16 Mmad; natural and MatMulV2-230097 K256 order; one L0C across all 48 mads",
        "commands": commands, "cann_root": str(cann),
        "compiler_sha256": digest(compiler), "build_complete": False,
        "sources": {path.name: digest(path) for path in args.output.iterdir()
                    if path.suffix in (".cpp", ".py")},
        "sdk_sources": {},
    }
    for parent in (arch / "asc/include/basic_api", arch / "asc/impl/basic_api/dav_c220"):
        for path in parent.rglob("*.h"):
            if any(part in path.name for part in ("mmad", "data_copy", "data_load", "fixpipe", "struct")):
                manifest["sdk_sources"][str(path.relative_to(cann))] = digest(path)
    try:
        for index, command in enumerate(commands):
            with (args.output / f"command-{index}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest["artifacts"] = {path.name: digest(path) for path in args.output.iterdir()
                                 if path.suffix in (".o", ".so")}
        manifest["build_complete"] = True
    finally:
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
