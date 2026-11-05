#!/usr/bin/env python3
"""Build the pinned Torch oracle adapter and rebind a frozen bridge kernel."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import sysconfig


ADAPTER_SOURCES = (
    "csrc/moe/moe_gating_top_k/moe_gating_top_k_torch_adpt.h",
    "csrc/moe/add_rms_norm_bias/add_rms_norm_bias_torch_adpt.h",
    "csrc/aclnn_torch_adapter/op_api_common.h",
    "csrc/aclnn_torch_adapter/NPUBridge.h",
    "csrc/aclnn_torch_adapter/NPUBridge.cpp",
    "csrc/aclnn_torch_adapter/NPUStorageImpl.h",
    "csrc/aclnn_torch_adapter/NPUStorageImpl.cpp",
    "csrc/torch_binding.cpp",
)
OPERATORS = ("moe_gating_top_k", "npu_add_rms_norm_bias")
API_NAMES = ("aclnnMoeGatingTopK", "aclnnAddRmsNormBias")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registration(text, operator):
    pattern = (r'    ops\.def\(\n        "' + operator + r'\(.*?'
               r'    ops\.impl\("' + operator + r'", torch::kPrivateUse1,\s*'
               r'&vllm_ascend::' + operator + r'\);')
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one pinned registration for {operator}")
    match = matches[0]
    return match.group(), [text.count("\n", 0, match.start()) + 1,
                           text.count("\n", 0, match.end()) + 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--native-build", type=Path, required=True)
    parser.add_argument("--oracle-prefix", type=Path, required=True)
    parser.add_argument("--cann", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    args = parser.parse_args()
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(args.oracle_prefix.resolve())
    import torch
    import torch_npu

    if torch.npu.is_initialized():
        raise RuntimeError("CPU binding build requires an uninitialized NPU runtime")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("build_runtime.py", "check.py", "fixtures.py"):
        shutil.copyfile(Path(__file__).with_name(name), args.output / name)
    snapshot = args.output / "source"
    identities = {}
    for relative in ADAPTER_SOURCES:
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.vendor / relative, target)
        identities[relative] = digest(target)
    text = (snapshot / "csrc/torch_binding.cpp").read_text()
    registrations = {name: registration(text, name) for name in OPERATORS}
    unit = args.output / "oracle_registration.cpp"
    unit.write_text('''#include <dlfcn.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "aclnn_torch_adapter/op_api_common.h"
#include "moe/moe_gating_top_k/moe_gating_top_k_torch_adpt.h"
#include "moe/add_rms_norm_bias/add_rms_norm_bias_torch_adpt.h"
TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
''' + "\n".join(value[0] for value in registrations.values()) + '''
}
extern "C" const char* ffn_bridge_oracle_api_library(const char* name) {
    void* address = GetOpApiFuncAddr(name);
    Dl_info information{};
    if (address == nullptr || dladdr(address, &information) == 0) return nullptr;
    return information.dli_fname;
}
''')
    native = args.output / "native"
    native.mkdir()
    parent_manifest = json.loads((args.native_build / "manifest.json").read_text())
    shutil.copytree(args.native_build / "source", native / "source")
    reused = {}
    for name in ("kernel.o", "tiling.o"):
        original = args.native_build / name
        expected = parent_manifest["artifacts"][name]
        if digest(original) != expected:
            raise RuntimeError(f"Frozen bridge object changed: {original}")
        shutil.copyfile(original, native / name)
        reused[name] = {"source": str(original), "sha256": expected}
    compiler = os.environ.get("CXX", "g++")
    arch = args.cann / (platform.machine() + "-linux")
    torch_root = Path(torch.__file__).resolve().parent
    npu_root = Path(torch_npu.__file__).resolve().parent
    includes = [snapshot / "csrc", Path(sysconfig.get_paths()["include"]),
                native / "source/ffn_bridge", torch_root / "include",
                torch_root / "include/torch/csrc/api/include", npu_root / "include",
                args.cann / "include", arch / "include"]
    compile_flags = [compiler, "-std=c++17", "-O2", "-fPIC",
                     f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"]
    for directory in includes:
        compile_flags.extend(["-I", str(directory)])
    link_flags = []
    for directory in (arch / "lib64", args.cann / "lib64",
                      torch_root / "lib", npu_root / "lib"):
        link_flags.extend(["-L", str(directory)])
    link_flags.extend("-l" + name for name in (
        "torch", "torch_cpu", "torch_python", "c10", "torch_npu", "ascendcl",
        "runtime", "unified_dlog", "c_sec", "register", "profapi",
        "tiling_api", "platform", "dl"))
    report = {"scope": "Pinned official adapters; frozen native device/tiling objects",
              "vendor_commit": "8237ee6d690d1399bc6eb48ef5977ac2bd470f3f",
              "image_id": args.image_id, "python": sys.version,
              "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "source_sha256": identities, "reused_objects": reused,
              "parent_manifest": parent_manifest, "commands": [],
              "registration_lines": {k: v[1] for k, v in registrations.items()},
              "registration_sha256": digest(unit), "complete": False}
    manifest = args.output / "manifest.json"

    def execute(label, command):
        report["commands"].append({"step": label, "argv": command})
        manifest.write_text(json.dumps(report, indent=2) + "\n")
        with (args.output / (label + ".log")).open("w") as log:
            status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if status.returncode:
            raise RuntimeError(f"CPU build failed: {label}")

    objects = []
    for path in (unit, snapshot / "csrc/aclnn_torch_adapter/NPUBridge.cpp",
                 snapshot / "csrc/aclnn_torch_adapter/NPUStorageImpl.cpp"):
        target = args.output / (path.stem + ".o")
        execute(path.stem, [*compile_flags, "-c", str(path), "-o", str(target)])
        objects.append(str(target))
    oracle = args.output / "official_bridge_oracle.so"
    execute("oracle-link", [compiler, "-shared", *objects, "-o", str(oracle), *link_flags])
    binding = native / "binding.o"
    execute("native-binding", [*compile_flags, "-c",
                               str(native / "source/ffn_bridge/binding.cpp"),
                               "-o", str(binding)])
    bridge = native / "ffn_bridge.so"
    execute("native-link", [compiler, "-shared", str(native / "kernel.o"),
                            str(native / "tiling.o"), str(binding),
                            "-o", str(bridge), *link_flags])
    report.update({"oracle_sha256": digest(oracle), "native_sha256": digest(bridge),
                   "npu_initialized": torch.npu.is_initialized()})
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    torch.ops.load_library(str(oracle))
    torch.ops.load_library(str(bridge))
    resolver = ctypes.CDLL(str(oracle)).ffn_bridge_oracle_api_library
    resolver.argtypes, resolver.restype = [ctypes.c_char_p], ctypes.c_char_p
    api = {}
    for name in API_NAMES:
        for symbol in (name, name + "GetWorkspaceSize"):
            resolved = resolver(symbol.encode())
            if resolved is None:
                raise RuntimeError(f"The pinned oracle package does not resolve {symbol}")
            path = Path(resolved.decode()).resolve()
            if args.oracle_prefix.resolve() not in path.parents:
                raise RuntimeError(f"Oracle unexpectedly resolved outside its package: {path}")
            api[symbol] = {"path": str(path), "sha256": digest(path)}
    report.update({"api_libraries": api, "npu_initialized": torch.npu.is_initialized(),
                   "oracle_sha256": digest(oracle), "native_sha256": digest(bridge),
                   "schemas": {name: str(getattr(torch.ops._C_ascend, name).default._schema)
                               for name in OPERATORS}, "complete": True})
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    if report["npu_initialized"]:
        raise RuntimeError("The CPU build initialized the NPU")
    print(json.dumps({k: report[k] for k in (
        "oracle_sha256", "native_sha256", "api_libraries", "npu_initialized")}))


if __name__ == "__main__":
    main()
