#!/usr/bin/env python3
"""Build the pinned decoder's required official host operators and Python entry."""

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig


OPERATORS = {
    "npu_add_rms_norm_bias": (
        "moe/add_rms_norm_bias/add_rms_norm_bias_torch_adpt.h", "aclnnAddRmsNormBias"),
    "moe_gating_top_k": (
        "moe/moe_gating_top_k/moe_gating_top_k_torch_adpt.h", "aclnnMoeGatingTopK"),
    "grouped_matmul_swiglu_quant_v2": (
        "gmm/grouped_matmul_swiglu_quant_v2/grouped_matmul_swiglu_quant_v2_torch_adpt.h",
        "aclnnGroupedMatmulSwigluQuantWeightNzV2"),
}
FILES = (
    "torch_binding.cpp", "torch_binding_meta.cpp", "camem_allocator.cpp",
    "aclnn_torch_adapter/op_api_common.h", "aclnn_torch_adapter/NPUBridge.h",
    "aclnn_torch_adapter/NPUBridge.cpp", "aclnn_torch_adapter/NPUStorageImpl.h",
    "aclnn_torch_adapter/NPUStorageImpl.cpp",
    *(item[0] for item in OPERATORS.values()),
)
BINDING_SHA = "ed1fe9f751f4c6c786c758ce9ff2731a9a0d89e14a81ff80438cdee96b4fc9e7"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_registration(text, name):
    pattern = (r'    ops\.def\(\n        "' + name + r'\(.*?'
               r'    ops\.impl\("' + name + r'", torch::kPrivateUse1,\s*'
               r'&vllm_ascend::' + name + r'\);')
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one original registration: {name}")
    match = matches[0]
    return match.group(), [text.count("\n", 0, match.start()) + 1,
                           text.count("\n", 0, match.end()) + 1]


def original_meta(text, name):
    pattern = r"std::tuple<[^\n]+> " + name + r"_meta\("
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one original Meta function: {name}")
    start = matches[0].start()
    body = text.index("{", matches[0].end())
    depth = 0
    end = None
    for index in range(body, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            end = index + 1
            break
    if end is None:
        raise RuntimeError(f"Unterminated original Meta function: {name}")
    line = f'    ops.impl("{name}", &vllm_ascend::meta::{name}_meta);'
    if text.count(line) != 1:
        raise RuntimeError(f"Expected one original Meta registration: {name}")
    return text[start:end], line, [text.count("\n", 0, start) + 1,
                                  text.count("\n", 0, end) + 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--oracle-prefix", type=Path, required=True)
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument("--sfa-library", type=Path, required=True)
    parser.add_argument("--cann", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    args = parser.parse_args()
    if list(Path("/dev").glob("davinci*")):
        raise RuntimeError("No NPU may be exposed to the host-extension build")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(args.oracle_prefix.resolve())
    import torch
    import torch_npu

    if (torch.__version__, torch_npu.__version__) != ("2.12.0+cu130", "2.12.0"):
        raise RuntimeError("Use the recorded Torch/TorchNPU 2.12 build image")
    if torch.npu.is_initialized():
        raise RuntimeError("The host build must not initialize an NPU")
    if digest(args.vendor / "csrc/torch_binding.cpp") != BINDING_SHA:
        raise RuntimeError("The official Torch registration source changed")
    expected = json.loads(args.oracle_manifest.read_text())["installed_sha256"]
    package = args.oracle_prefix.parents[2]
    for relative, sha in expected.items():
        path = package / relative
        if not path.is_file() or digest(path) != sha:
            raise RuntimeError(f"The official oracle package changed: {path}")
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "source"
    identities = {}
    for name in FILES:
        path = source / "csrc" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.vendor / "csrc" / name, path)
        identities["csrc/" + name] = digest(path)
    binding = (source / "csrc/torch_binding.cpp").read_text()
    meta = (source / "csrc/torch_binding_meta.cpp").read_text()
    registrations = {name: original_registration(binding, name) for name in OPERATORS}
    meta_functions = {name: original_meta(meta, name) for name in OPERATORS}
    unit = args.output / "registration.cpp"
    unit.write_text(
        '#include <dlfcn.h>\n#include <torch/extension.h>\n#include <torch/library.h>\n'
        '#include "aclnn_torch_adapter/op_api_common.h"\n'
        + "".join(f'#include "{header}"\n' for header, unused in OPERATORS.values())
        + 'TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {\n'
        + "\n".join(value[0] for value in registrations.values()) + '\n}\n'
        + '''extern "C" const char* decoder_official_api_library(const char* name) {
    void* address = GetOpApiFuncAddr(name);
    Dl_info information{};
    if (address == nullptr || dladdr(address, &information) == 0) return nullptr;
    return information.dli_fname;
}
''')
    meta_unit = args.output / "meta.cpp"
    meta_unit.write_text(
        '#include <torch/extension.h>\n#include <torch/library.h>\n'
        'namespace vllm_ascend { namespace meta {\n'
        + "\n".join(value[0] for value in meta_functions.values()) + '\n}}\n'
        'TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops) {\n'
        + "\n".join(value[1] for value in meta_functions.values()) + '\n}\n')
    torch_root = Path(torch.__file__).resolve().parent
    npu_root = Path(torch_npu.__file__).resolve().parent
    compiler = os.environ.get("CXX", "g++")
    flags = [compiler, "-std=c++17", "-O2", "-fPIC",
             f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"]
    for directory in (source / "csrc", Path(sysconfig.get_paths()["include"]),
                      torch_root / "include", torch_root / "include/torch/csrc/api/include",
                      npu_root / "include", args.cann / "include"):
        flags.extend(["-I", str(directory)])
    report = {
        "scope": "Official camem Python entry and three unmodified decoder host/Meta operators",
        "vendor_commit": "8237ee6d690d1399bc6eb48ef5977ac2bd470f3f",
        "image_id": args.image_id, "python": sys.version, "torch": torch.__version__,
        "torch_npu": torch_npu.__version__, "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
        "source_sha256": identities, "script_sha256": digest(Path(__file__)),
        "registration_lines": {name: value[1] for name, value in registrations.items()},
        "registration_sha256": {name: hashlib.sha256(value[0].encode()).hexdigest()
                                for name, value in registrations.items()},
        "meta_lines": {name: value[2] for name, value in meta_functions.items()},
        "meta_sha256": {name: hashlib.sha256(value[0].encode()).hexdigest()
                        for name, value in meta_functions.items()},
        "oracle_manifest_sha256": digest(args.oracle_manifest),
        "oracle_prefix": str(args.oracle_prefix), "oracle_file_count": len(expected),
        "generated_sha256": {path.name: digest(path) for path in (unit, meta_unit)},
        "commands": [], "completed": False,
    }
    manifest = args.output / "manifest.json"

    def execute(label, command):
        report["commands"].append({"label": label, "argv": command})
        manifest.write_text(json.dumps(report, indent=2) + "\n")
        with (args.output / (label + ".log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

    objects = []
    dependencies = {}
    units = [unit, meta_unit, source / "csrc/camem_allocator.cpp",
             source / "csrc/aclnn_torch_adapter/NPUBridge.cpp",
             source / "csrc/aclnn_torch_adapter/NPUStorageImpl.cpp"]
    for path in units:
        obj = args.output / (path.stem + ".o")
        dep = obj.with_suffix(".d")
        execute(path.stem, [*flags, "-MD", "-MF", str(dep), "-c", str(path), "-o", str(obj)])
        objects.append(obj)
        for name in dep.read_text().replace("\\\n", " ").split()[1:]:
            dependency = Path(name)
            if dependency.is_file():
                dependencies[str(dependency)] = digest(dependency)
    extension = args.output / ("vllm_ascend_C" + sysconfig.get_config_var("EXT_SUFFIX"))
    link = [compiler, "-shared", *(str(path) for path in objects), "-o", str(extension)]
    for directory in (args.cann / "lib64", torch_root / "lib", npu_root / "lib"):
        link.extend(["-L", str(directory)])
    link.extend("-l" + name for name in (
        "torch", "torch_cpu", "torch_python", "c10", "torch_npu", "ascendcl", "dl"))
    execute("link", link)
    spec = importlib.util.spec_from_file_location("vllm_ascend_C", extension)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    methods = ("init_module", "python_create_and_map", "python_unmap_and_release")
    if not all(callable(getattr(module, name, None)) for name in methods):
        raise RuntimeError("The actual official Python allocator entry is missing")
    torch.ops.load_library(str(args.sfa_library))
    resolver = ctypes.CDLL(str(extension)).decoder_official_api_library
    resolver.argtypes, resolver.restype = [ctypes.c_char_p], ctypes.c_char_p
    api = {}
    for name, (unused_header, function) in OPERATORS.items():
        for symbol in (function, function + "GetWorkspaceSize"):
            resolved = resolver(symbol.encode())
            if resolved is None:
                raise RuntimeError(f"The official API does not resolve: {symbol}")
            path = Path(resolved.decode()).resolve()
            required = args.cann if name == "grouped_matmul_swiglu_quant_v2" else args.oracle_prefix
            if required.resolve() not in path.parents:
                raise RuntimeError(f"The official API resolved outside its required package: {path}")
            api[symbol] = {"path": str(path), "sha256": digest(path)}
    x = torch.empty(12, 6144, dtype=torch.bfloat16, device="meta")
    gamma = torch.empty(6144, dtype=torch.bfloat16, device="meta")
    norm = torch.ops._C_ascend.npu_add_rms_norm_bias(x, x, gamma, None, 1e-5)
    if [list(value.shape) for value in norm] != [[12, 6144], [12, 1], [12, 6144]]:
        raise RuntimeError("The original Meta norm contract changed")
    report.update({
        "library": str(extension), "library_sha256": digest(extension),
        "python_entry_methods": list(methods), "api_libraries": api,
        "dependency_sha256": dependencies,
        "schemas": {name: str(getattr(torch.ops._C_ascend, name).default._schema)
                    for name in OPERATORS},
        "meta_registered": {name: torch._C._dispatch_has_kernel_for_dispatch_key(
            "_C_ascend::" + name, "Meta") for name in OPERATORS},
        "sfa_coload_sha256": digest(args.sfa_library),
        "npu_initialized": torch.npu.is_initialized(), "completed": True,
    })
    if report["npu_initialized"]:
        raise RuntimeError("The CPU build initialized an NPU")
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "completed", "library_sha256", "python_entry_methods", "api_libraries",
        "npu_initialized")}), flush=True)


if __name__ == "__main__":
    main()
