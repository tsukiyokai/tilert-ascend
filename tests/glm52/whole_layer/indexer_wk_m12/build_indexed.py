#!/usr/bin/env python3
"""Add the measured M12 WK traversal to the frozen Torch212 indexed library."""

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


PARENT_SHA = "e957650c7def45dff8b918b74c987a8ce44eb0b2cca816e7246cbbff8a52378e"
SHARED_SHA = "64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c"
FRONT = Path("tests/glm52/whole_layer/indexer_front_native.h")
HEADER = FRONT.with_name("indexer_wk_m12_native.h")
INCLUDE = '#include "indexer_wk_m32_native.h"'
OLD_BRANCH = """        if (c.rows == 32) {
            indexer_wk_m32::device::Project(pipe, c, args.hidden,"""
NEW_BRANCH = """        if (c.rows == 12) {
            indexer_wk_m12::device::Project(pipe, c, args.hidden,
                args.key_weight, args.scratch + c.key_acc);
        } else if (c.rows == 32) {
            indexer_wk_m32::device::Project(pipe, c, args.hidden,"""


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(root):
    return {str(path.relative_to(root)): digest(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def verify(path, expected):
    if digest(path) != expected:
        raise RuntimeError(f"Frozen identity changed: {path}")


def transform(source):
    if source.count(INCLUDE) != 1 or source.count(OLD_BRANCH) != 1:
        raise RuntimeError("Expected exactly one frozen M32 WK dispatch")
    return source.replace(INCLUDE, INCLUDE + '\n#include "indexer_wk_m12_native.h"').replace(
        OLD_BRANCH, NEW_BRANCH)


def prepare(args):
    parent_manifest = args.parent / "manifest.json"
    parent = json.loads(parent_manifest.read_text())
    original = next(value for value in parent["builds"] if value["mode"] == "indexed")
    if not parent["completed"] or original["library_sha256"] != PARENT_SHA:
        raise RuntimeError("Use the frozen e957 Torch212 indexed parent")
    indexed = args.parent / "indexed"
    verify(indexed / "whole_layer_indexed_attention.so", PARENT_SHA)
    if hashes(indexed / "source") != original["source_hashes"]:
        raise RuntimeError("The parent source snapshot changed")
    verify(args.parent / "shared/whole_layer_full_attention.so", SHARED_SHA)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(indexed / "source", args.output / "source")
    before = (indexed / "source" / FRONT).read_text()
    after = transform(before)
    (args.output / "source" / FRONT).write_text(after)
    shutil.copy2(Path(__file__).with_name("native.h"), args.output / "source" / HEADER)
    unchanged = dict(original["source_hashes"])
    unchanged.pop(str(FRONT))
    for relative, expected in unchanged.items():
        verify(args.output / "source" / relative, expected)
    objects = {name: sha for name, sha in original["reused_objects"].items()
               if name != "attention_indexed_kernel.o"}
    objects["attention_indexed_binding.o"] = original["binding_sha256"]
    for name, sha in objects.items():
        verify(indexed / name, sha)
        shutil.copy2(indexed / name, args.output / name)
    for name in ("build_indexed.py", "native.h", "test_indexed.py"):
        shutil.copy2(Path(__file__).with_name(name), args.output / name)
    shutil.copy2(parent_manifest, args.output / "parent-manifest.json")
    shutil.copy2(args.provenance, args.output / "parent-provenance.json")
    (args.output / "front.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile=f"e957/{FRONT}", tofile=f"m12/{FRONT}")))
    write(args.output / "manifest.json", {
        "scope": "Only rows==12 WK uses measured K512 order; M32, other widths and all epilogues unchanged",
        "parent": str(args.parent), "parent_library_sha256": PARENT_SHA,
        "shared_library": str(args.parent / "shared/whole_layer_full_attention.so"),
        "shared_library_sha256": SHARED_SHA,
        "parent_manifest_sha256": digest(parent_manifest),
        "source_hashes": hashes(args.output / "source"),
        "unchanged_parent_sources": unchanged, "reused_objects": objects,
        "parent_link_command": original["link_command"], "schemas": parent["schemas"],
        "additional_gm_bytes": 0,
        "local_bytes": {"L1": 8192, "L0A": 4096, "L0B": 4096, "L0C": 1024},
        "cpu_binding_recompiled": False, "device_execution": False,
    })


def kernel(args):
    path = args.output / "manifest.json"
    report = json.loads(path.read_text())
    if hashes(args.output / "source") != report["source_hashes"]:
        raise RuntimeError("Candidate sources changed")
    provenance = json.loads((args.output / "parent-provenance.json").read_text())
    for relative, expected in provenance["vendor_headers"].items():
        verify(args.vendor / relative, expected)
    command = ["bash", str(args.output / "source/tests/glm52/whole_layer/attention_build.sh"),
               str(args.output)]
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith("ATTENTION_") and name != "SANITIZE"}
    settings = {"CANN_ROOT": str(args.cann), "OPS_TRANSFORMER_ROOT": str(args.vendor),
                "ATTENTION_KERNEL_ONLY": "1", "ATTENTION_INDEXED": "1", "PYTHON_BIN": "python3"}
    environment.update(settings)
    with (args.output / "kernel-build.log").open("w") as output:
        subprocess.run(command, env=environment, stdout=output, stderr=subprocess.STDOUT, check=True)
    report.update(kernel_build_command=command, kernel_build_environment=settings,
                  kernel_sha256=digest(args.output / "attention_indexed_kernel.o"))
    write(path, report)


def link(args):
    import torch
    import torch_npu

    if list(Path("/dev").glob("davinci*")) or torch.npu.is_initialized():
        raise RuntimeError("CPU link/load requires no exposed or initialized NPU")
    if torch.__version__ != "2.12.0+cu130" or torch_npu.__version__ != "2.12.0":
        raise RuntimeError("Use the frozen Torch212 host ABI")
    path = args.output / "manifest.json"
    report = json.loads(path.read_text())
    for name, expected in report["reused_objects"].items():
        verify(args.output / name, expected)
    verify(args.output / "attention_indexed_kernel.o", report["kernel_sha256"])
    command = [str(args.output / Path(value).name) if value.endswith((".o", ".so")) else value
               for value in report["parent_link_command"]]
    with (args.output / "link.log").open("w") as output:
        subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
    library = args.output / "whole_layer_indexed_attention.so"
    shared = Path(report["shared_library"])
    verify(shared, SHARED_SHA)
    torch.ops.load_library(str(library))
    torch.ops.load_library(str(shared))
    schemas = {name: str(getattr(torch.ops.tilert, name).default._schema)
               for name in report["schemas"]}
    if schemas != report["schemas"] or torch.npu.is_initialized():
        raise RuntimeError("The candidate changed schemas or initialized NPU")
    report.update(link_command=command, library_sha256=digest(library),
                  cpu_load={"torch": torch.__version__, "torch_npu": torch_npu.__version__,
                            "cxx11_abi": torch.compiled_with_cxx11_abi(),
                            "npu_initialized": False, "schemas": schemas})
    write(path, report)
    print(json.dumps({"library_sha256": report["library_sha256"], "cpu_load": report["cpu_load"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "kernel", "link"))
    parser.add_argument("--parent", type=Path, default=Path("/workspace/logs/runtime212-build-v1/attention"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cann", type=Path, default=Path("/opt/ascend/cann-9.2.0"))
    parser.add_argument("--vendor", type=Path, default=Path("/workspace/attention_vendor"))
    parser.add_argument("--provenance", type=Path, default=Path(
        "/workspace/tests/glm52/whole_layer/attention_wk_kshift_build-v1/parent-provenance.json"))
    args = parser.parse_args()
    {"prepare": prepare, "kernel": kernel, "link": link}[args.stage](args)


if __name__ == "__main__":
    main()
