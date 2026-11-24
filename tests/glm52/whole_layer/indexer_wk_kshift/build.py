#!/usr/bin/env python3
"""Build the M32 WK K-order candidate from the frozen O-combined library."""

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


FRONT = Path("tests/glm52/whole_layer/indexer_front_native.h")
HEADER = FRONT.with_name("indexer_wk_m32_native.h")
PARENT_SHA = "9aa71e6a385752824050368930f76a1d9d8d4989a484dbbccf8080ec02b8546e"
SHARED_SHA = "dd66e68b9c8f46c4fdee987d7b7bb29a062e3dcae5481237c45bc21544d346b9"
OLD_PROJECT = """        Project<bfloat16_t, float>(pipe, c, args.hidden, args.key_weight,
            args.scratch + c.key_acc, args.tiling + kTilingStride,
            kKeyTile, kHidden, kKeyColumns / kKeyTile);"""
NEW_PROJECT = """        if (c.rows == 32) {
            indexer_wk_m32::device::Project(pipe, c, args.hidden,
                args.key_weight, args.scratch + c.key_acc);
        } else {
            Project<bfloat16_t, float>(pipe, c, args.hidden, args.key_weight,
                args.scratch + c.key_acc, args.tiling + kTilingStride,
                kKeyTile, kHidden, kKeyColumns / kKeyTile);
        }"""


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def verify(path, expected):
    if digest(path) != expected:
        raise RuntimeError(f"SHA-256 mismatch: {path}")


def source_hashes(root):
    return {str(p.relative_to(root)): digest(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def prepare(args):
    parent_path = args.parent / "manifest.json"
    parent = json.loads(parent_path.read_text())
    original = next(b for b in parent["builds"] if b["mode"] == "indexed")
    if original["library_sha256"] != PARENT_SHA:
        raise RuntimeError("Expected the frozen O-combined indexed parent")
    old_output = args.parent / "indexed"
    verify(old_output / original["library_name"], PARENT_SHA)
    old_source = old_output / "source"
    if source_hashes(old_source) != original["source_hashes"]:
        raise RuntimeError("Frozen parent sources changed")
    before = (old_source / FRONT).read_text()
    include = '#include "indexer_front_contract.h"'
    if before.count(OLD_PROJECT) != 1 or before.count(include) != 1:
        raise RuntimeError("Unexpected front-end projection or include structure")
    after = before.replace(include, include + '\n#include "indexer_wk_m32_native.h"')
    after = after.replace(OLD_PROJECT, NEW_PROJECT)
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "source"
    shutil.copytree(old_source, snapshot)
    (snapshot / FRONT).write_text(after)
    shutil.copyfile(Path(__file__).with_name("native.h"), snapshot / HEADER)
    for name in ("build.py", "native.h", "test_contract.py"):
        shutil.copyfile(Path(__file__).with_name(name), args.output / name)
    shutil.copyfile(parent_path, args.output / "parent-manifest.json")
    shutil.copyfile(old_output / "parent-provenance.json",
                    args.output / "parent-provenance.json")
    for name, expected in original["reused_objects"].items():
        verify(old_output / name, expected)
        shutil.copyfile(old_output / name, args.output / name)
    patch = difflib.unified_diff(before.splitlines(keepends=True),
                                after.splitlines(keepends=True),
                                fromfile=f"parent/{FRONT}", tofile=f"candidate/{FRONT}")
    (args.output / "front.patch").write_text("".join(patch))
    shared = args.parent / "shared/whole_layer_full_attention.so"
    verify(shared, SHARED_SHA)
    report = {
        "scope": "M32 BF16 WK K-order only; M1..31 use the original projection",
        "parent": str(args.parent), "parent_library_sha256": PARENT_SHA,
        "parent_manifest_sha256": digest(parent_path),
        "shared_library": str(shared), "shared_library_sha256": SHARED_SHA,
        "source_hashes": source_hashes(snapshot),
        "parent_source_hashes": original["source_hashes"],
        "reused_objects": original["reused_objects"],
        "parent_link_command": original["link_command"],
        "library_name": original["library_name"], "schemas": parent["schemas"],
        "builder_sha256": digest(Path(__file__)),
        "added_gm_workspace_bytes": 0,
        "wk_local_bytes": {"L1": 8192, "L0A": 4096, "L0B": 4096, "L0C": 1024},
        "output_abi_changed": False,
    }
    save(args.output / "manifest.json", report)


def compile_kernel(args):
    path = args.output / "manifest.json"
    report = json.loads(path.read_text())
    snapshot = args.output / "source"
    if source_hashes(snapshot) != report["source_hashes"]:
        raise RuntimeError("Candidate source identity changed")
    provenance = json.loads((args.output / "parent-provenance.json").read_text())
    for relative, expected in provenance["vendor_headers"].items():
        verify(args.vendor / relative, expected)
    command = ["bash", str(snapshot / "tests/glm52/whole_layer/attention_build.sh"),
               str(args.output)]
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("ATTENTION_") and key != "SANITIZE"}
    specific = {"CANN_ROOT": str(args.cann),
                "OPS_TRANSFORMER_ROOT": str(args.vendor),
                "ATTENTION_KERNEL_ONLY": "1", "ATTENTION_INDEXED": "1",
                "PYTHON_BIN": "python3"}
    env.update(specific)
    with (args.output / "kernel-build.log").open("w") as log:
        subprocess.run(command, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    report["kernel_build_command"] = command
    report["kernel_build_env"] = specific
    report["kernel_sha256"] = digest(args.output / "attention_indexed_kernel.o")
    save(path, report)


def link_and_load(args):
    import torch
    import torch_npu

    devices = [str(p) for p in Path("/dev").glob("davinci*")]
    if devices or torch.npu.is_initialized():
        raise RuntimeError("CPU link check requires no exposed or initialized NPU")
    if torch.__version__ != "2.10.0+cpu" or torch_npu.__version__ != "2.10.0.post2":
        raise RuntimeError("Use the recorded Torch 2.10 image")
    path = args.output / "manifest.json"
    report = json.loads(path.read_text())
    for name, expected in report["reused_objects"].items():
        verify(args.output / name, expected)
    verify(args.output / "attention_indexed_kernel.o", report["kernel_sha256"])
    command = [str(args.output / Path(value).name)
               if value.endswith((".o", ".so")) else value
               for value in report["parent_link_command"]]
    with (args.output / "link.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    library = args.output / report["library_name"]
    shared = Path(report["shared_library"])
    verify(shared, SHARED_SHA)
    torch.ops.load_library(str(library))
    torch.ops.load_library(str(shared))
    schemas = {name: str(getattr(torch.ops.tilert, name).default._schema)
               for name in report["schemas"]}
    if schemas != report["schemas"]:
        raise RuntimeError("Candidate schemas differ from the parent schemas")
    report["link_command"] = command
    report["library_sha256"] = digest(library)
    report["cpu_load"] = {"torch": torch.__version__,
                          "torch_npu": torch_npu.__version__,
                          "npu_initialized": torch.npu.is_initialized(),
                          "device_nodes": devices, "schemas": schemas}
    if report["cpu_load"]["npu_initialized"]:
        raise RuntimeError("CPU check initialized the NPU")
    save(path, report)
    print(json.dumps({"library_sha256": report["library_sha256"],
                      "cpu_load": report["cpu_load"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "kernel", "link"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parent", type=Path, default=Path(
        "/workspace/tests/glm52/whole_layer/attention_o_combined_build-v1"))
    parser.add_argument("--cann", type=Path,
                        default=Path("/opt/ascend/cann-9.2.0"))
    parser.add_argument("--vendor", type=Path,
                        default=Path("/workspace/attention_vendor"))
    args = parser.parse_args()
    {"prepare": prepare, "kernel": compile_kernel, "link": link_and_load}[args.stage](args)


if __name__ == "__main__":
    main()
