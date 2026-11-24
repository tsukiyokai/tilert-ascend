#!/usr/bin/env python3
"""Relink accepted attention objects against the selected Torch installation."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


ACCEPTED = (
    ("indexed", "attention_indexed_build-scale-order", "source",
     "attention_indexed", "whole_layer_indexed_attention",
     "6588e1dcde9dd843c10479ce1276ae069d89a9e60c16bc296f20ba4afb7c1c34",
     ("attention_indexed_kernel.o", "indexer_front_tiling.o", "mla_tail_tiling.o")),
    ("shared", "attention_full_build-dequant", "sources",
     "attention_full", "whole_layer_full_attention",
     "dce289d8fa391921348b50505f5eaf96a496546f0f359d7b38c39ee3ab4b0d06",
     ("attention_full_kernel.o", "mla_tail_tiling.o")),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cann", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    args = parser.parse_args()
    import torch
    import torch_npu

    if torch.npu.is_initialized():
        raise RuntimeError("Binding rebuild must start without NPU initialization")
    torch_root = Path(torch.__file__).resolve().parent
    npu_root = Path(importlib.util.find_spec("torch_npu").origin).resolve().parent
    arch = args.cann / f"{platform.machine()}-linux"
    compiler = os.environ.get("CXX", "g++")
    includes = [args.cann / "include", arch / "include",
                args.workspace / "attention_vendor", torch_root / "include",
                torch_root / "include/torch/csrc/api/include", npu_root / "include"]
    for relative in ("asc", "asc/include", "asc/include/adv_api",
                     "asc/include/basic_api", "asc/include/utils",
                     "asc/impl/adv_api", "asc/impl/basic_api"):
        if (arch / relative).is_dir():
            includes.append(arch / relative)
    report = {
        "scope": "Torch binding rebuild; accepted device and tiling objects reused",
        "script_sha256": digest(Path(__file__)), "image_id": args.image_id,
        "python": sys.version, "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "compiler": subprocess.check_output([compiler, "--version"], text=True),
        "cann_root": str(args.cann), "cann_version_files": {}, "builds": [],
    }
    for pattern in ("**/ascend_toolkit_install.info", "**/version.info"):
        for version in args.cann.glob(pattern):
            if version.is_file() and version.stat().st_size < 16384:
                report["cann_version_files"][str(version)] = {
                    "sha256": digest(version), "text": version.read_text(errors="replace"),
                }
    args.output.mkdir(parents=True, exist_ok=False)
    for mode, directory, snapshot, stem, library, accepted, object_names in ACCEPTED:
        original = args.workspace / "tests/glm52/whole_layer" / directory
        old_library = original / f"{library}.so"
        if digest(old_library) != accepted:
            raise RuntimeError(f"Accepted library identity changed: {old_library}")
        source_root = original / snapshot
        source = source_root / "tests/glm52/whole_layer"
        output = args.output / mode
        output.mkdir()
        copied = []
        identities = {}
        for name in object_names:
            path = original / name
            target = output / name
            shutil.copyfile(path, target)
            copied.append(str(target))
            identities[name] = {"path": str(path), "sha256": digest(path)}
        binding = output / f"{stem}_binding.o"
        command = [compiler, "-std=c++17", "-O2", "-fPIC",
                   f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"]
        for include in [source, *includes]:
            command.extend(["-I", str(include)])
        command.extend(["-c", str(source / f"{stem}_binding.cpp"), "-o", str(binding)])
        subprocess.run(command, check=True)
        result = output / f"{library}.so"
        link = [compiler, "-shared", *copied, str(binding), "-o", str(result)]
        for directory in (arch / "lib64", args.cann / "lib64",
                          torch_root / "lib", npu_root / "lib"):
            link.extend(["-L", str(directory)])
        link.extend(f"-l{name}" for name in (
            "ascendcl", "runtime", "unified_dlog", "c_sec", "register",
            "profapi", "tiling_api", "platform", "torch", "torch_cpu",
            "c10", "torch_npu"))
        subprocess.run(link, check=True)
        torch.ops.load_library(str(result))
        report["builds"].append({
            "mode": mode, "accepted_library": str(old_library),
            "accepted_library_sha256": accepted, "reused_objects": identities,
            "source_root": str(source_root),
            "source_hashes": {str(path.relative_to(source_root)): digest(path)
                              for path in sorted(source_root.rglob("*"))
                              if path.is_file()},
            "compile_command": command, "link_command": link,
            "binding_sha256": digest(binding), "library": str(result),
            "library_sha256": digest(result),
        })
    report["schemas"] = {
        name: str(getattr(torch.ops.tilert, name).default._schema)
        for name in ("indexed_attention_prepare", "indexed_attention_out",
                     "shared_full_attention_prepare", "shared_full_attention_out")
    }
    report["npu_initialized"] = torch.npu.is_initialized()
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["npu_initialized"]:
        raise RuntimeError("CPU binding audit unexpectedly initialized the NPU")
    print(json.dumps({"builds": report["builds"], "schemas": report["schemas"]}, indent=2))


if __name__ == "__main__":
    main()
