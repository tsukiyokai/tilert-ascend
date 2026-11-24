#!/usr/bin/env python3
"""Rebind the accepted attention device objects to the recorded Torch 2.12 ABI."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


INDEXED_SHA = "20095457af00d9299fcff884d806b03400669adfe3304750ca9ae97983a40ca8"
SHARED_SHA = "dd66e68b9c8f46c4fdee987d7b7bb29a062e3dcae5481237c45bc21544d346b9"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(root):
    return {str(path.relative_to(root)): digest(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def verify(path, expected):
    if digest(path) != expected:
        raise RuntimeError(f"Frozen identity changed: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cann", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    args = parser.parse_args()
    import torch
    import torch_npu

    if list(Path("/dev").glob("davinci*")) or torch.npu.is_initialized():
        raise RuntimeError("The binding build requires no exposed or initialized NPU")
    if (torch.__version__, torch_npu.__version__) != ("2.12.0+cu130", "2.12.0"):
        raise RuntimeError("Use the recorded Torch 2.12 installation")
    base = args.workspace / "tests/glm52/whole_layer"
    indexed_path = base / "attention_wk_kshift_build-v1"
    shared_parent = base / "attention_o_combined_build-v1"
    indexed = json.loads((indexed_path / "manifest.json").read_text())
    combined = json.loads((shared_parent / "manifest.json").read_text())
    shared = next(item for item in combined["builds"] if item["mode"] == "shared")
    parents = (
        ("indexed", indexed_path, indexed, "attention_indexed", INDEXED_SHA,
         ("attention_indexed_kernel.o", "indexer_front_tiling.o", "mla_tail_tiling.o")),
        ("shared", shared_parent / "shared", shared, "attention_full", SHARED_SHA,
         ("attention_full_kernel.o", "mla_tail_tiling.o")),
    )
    torch_root = Path(torch.__file__).resolve().parent
    npu_root = Path(torch_npu.__file__).resolve().parent
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
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "Torch 2.12 host bindings; unchanged CANN 9.2 device and tiling objects",
        "script_sha256": digest(Path(__file__)), "image_id": args.image_id,
        "python": sys.version, "torch": torch.__version__,
        "torch_npu": torch_npu.__version__, "torch_root": str(torch_root),
        "torch_npu_root": str(npu_root),
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "compiler": subprocess.check_output([compiler, "--version"], text=True),
        "cann_root": str(args.cann), "builds": [], "completed": False,
    }
    for mode, parent_path, parent, stem, expected, names in parents:
        verify(parent_path / parent["library_name"], expected)
        if parent["library_sha256"] != expected:
            raise RuntimeError("The parent manifest describes another library")
        source = parent_path / "source"
        if hashes(source) != parent["source_hashes"]:
            raise RuntimeError("The frozen parent source tree changed")
        output = args.output / mode
        output.mkdir()
        snapshot = output / "source"
        shutil.copytree(source, snapshot)
        expected_objects = dict(parent["reused_objects"])
        expected_objects[f"{stem}_kernel.o"] = parent["kernel_sha256"]
        reused = {}
        for name in names:
            verify(parent_path / name, expected_objects[name])
            shutil.copyfile(parent_path / name, output / name)
            reused[name] = digest(output / name)
            undefined = subprocess.check_output(
                ["nm", "-u", "-C", str(output / name)], text=True)
            (output / f"{name}.undefined.txt").write_text(undefined)
            if any(token in undefined for token in ("at::", "c10::", "torch::")):
                raise RuntimeError(f"A reused object unexpectedly depends on Torch: {name}")
        binding_source = snapshot / f"tests/glm52/whole_layer/{stem}_binding.cpp"
        binding = output / f"{stem}_binding.o"
        compile_command = [compiler, "-std=c++17", "-O2", "-fPIC",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"]
        for include in [binding_source.parent, *includes]:
            compile_command.extend(["-I", str(include)])
        compile_command.extend(["-c", str(binding_source), "-o", str(binding)])
        with (output / "compile.log").open("w") as log:
            subprocess.run(compile_command, stdout=log, stderr=subprocess.STDOUT, check=True)
        library = output / parent["library_name"]
        link_command = [compiler, "-shared", *(str(output / name) for name in names),
                        str(binding), "-o", str(library)]
        for directory in (arch / "lib64", args.cann / "lib64",
                          torch_root / "lib", npu_root / "lib"):
            link_command.extend(["-L", str(directory)])
        link_command.extend(f"-l{name}" for name in (
            "ascendcl", "runtime", "unified_dlog", "c_sec", "register", "profapi",
            "tiling_api", "platform", "torch", "torch_cpu", "c10", "torch_npu"))
        with (output / "link.log").open("w") as log:
            subprocess.run(link_command, stdout=log, stderr=subprocess.STDOUT, check=True)
        torch.ops.load_library(str(library))
        report["builds"].append({
            "mode": mode, "parent": str(parent_path), "parent_library_sha256": expected,
            "source_hashes": hashes(snapshot), "reused_objects": reused,
            "binding_source_sha256": digest(binding_source),
            "binding_sha256": digest(binding), "compile_command": compile_command,
            "link_command": link_command, "library": str(library),
            "library_sha256": digest(library),
        })
        (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    report["schemas"] = {
        name: str(getattr(torch.ops.tilert, name).default._schema)
        for name in indexed["schemas"]
    }
    if report["schemas"] != indexed["schemas"]:
        raise RuntimeError("The rebuilt libraries changed the recorded schemas")
    report["npu_initialized"] = torch.npu.is_initialized()
    if report["npu_initialized"]:
        raise RuntimeError("The CPU build initialized the NPU")
    report["completed"] = True
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"completed": True, "builds": [
        {"mode": item["mode"], "library_sha256": item["library_sha256"]}
        for item in report["builds"]], "npu_initialized": False}), flush=True)


if __name__ == "__main__":
    main()
