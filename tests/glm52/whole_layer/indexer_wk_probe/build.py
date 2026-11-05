#!/usr/bin/env python3
"""Build the isolated WK geometry probe from accepted frozen headers."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


PARENT_SOURCES = {
    "tests/glm52/whole_layer/indexer_front_native.h":
        "772a591f78b5276013f67970e106566119c566e14a272f3ee5703a675865a184",
    "tests/glm52/whole_layer/indexer_front_contract.h":
        "b3cdf1e6f08a106da03c7039202ee34ed7faecb03c6d485cb5fc3bfef62d5db4",
    "tests/glm52/mla_tail/mla_tail_native.h":
        "dafafe680633a71fd3eded30c77c3aa44145c10154a89b321fca963ccc1e472b",
}
TILING_SHA = "fe1c6036ea231cd950bc8f3d30e332af7f74314a328fffd7be89d58bdffbc045"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--cann", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent_source = args.parent / "source"
    for name, expected in PARENT_SOURCES.items():
        if digest(parent_source / name) != expected:
            raise RuntimeError(f"Accepted source hash mismatch: {name}")
    original = args.parent / "indexer_front_tiling.o"
    if digest(original) != TILING_SHA:
        raise RuntimeError("Accepted tiling object hash mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "source"
    shutil.copytree(parent_source, snapshot)
    here = Path(__file__).resolve().parent
    for name in ("native_kernel.cpp", "native_tiling.cpp", "build.py"):
        shutil.copy2(here / name, args.output / name)
    shutil.copy2(original, args.output / original.name)
    cann = args.cann.resolve()
    arch = cann / "aarch64-linux"
    includes = ["-I" + str(snapshot / "tests/glm52/whole_layer"),
                "-I" + str(cann / "include")]
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
         "-fPIC", *includes, "-c", str(args.output / "native_kernel.cpp"),
         "-o", str(args.output / "native_kernel.o")],
        ["g++", "-std=c++17", "-O2", "-fPIC", *includes, "-c",
         str(args.output / "native_tiling.cpp"), "-o",
         str(args.output / "native_tiling.o")],
        ["g++", "-shared", str(args.output / "native_kernel.o"),
         str(args.output / "native_tiling.o"),
         str(args.output / "indexer_front_tiling.o"), "-o",
         str(args.output / "wk_geometry.so"), "-L" + str(arch / "lib64"),
         "-lascendcl", "-lruntime", "-lunified_dlog", "-lc_sec", "-lregister",
         "-lprofapi", "-ltiling_api", "-lplatform"],
    ]
    manifest = {
        "scope": "Accepted Project<BF16,FP32> template; N32 original tiling control and N160 geometry candidate",
        "parent": str(args.parent), "original_tiling_sha256": TILING_SHA,
        "source_hashes": {str(p.relative_to(snapshot)): digest(p)
                          for p in sorted(snapshot.rglob("*")) if p.is_file()},
        "commands": commands, "cann_root": str(cann),
        "compiler_sha256": digest(compiler), "build_complete": False,
    }
    manifest_path = args.output / "manifest.json"
    try:
        for index, command in enumerate(commands):
            with (args.output / f"command-{index}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest["artifacts"] = {p.name: digest(p) for p in args.output.iterdir()
                                 if p.is_file() and p.suffix in (".cpp", ".py", ".o", ".so")}
        manifest["build_complete"] = True
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
