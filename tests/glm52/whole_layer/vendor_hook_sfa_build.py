#!/usr/bin/env python3
"""Build the pinned vendor SFA adapter and schema without its other operators."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig


SOURCES = (
    "csrc/attention/sparse_flash_attention/sparse_flash_attention_torch_adpt.h",
    "csrc/aclnn_torch_adapter/op_api_common.h",
    "csrc/aclnn_torch_adapter/NPUBridge.h",
    "csrc/aclnn_torch_adapter/NPUBridge.cpp",
    "csrc/aclnn_torch_adapter/NPUStorageImpl.h",
    "csrc/aclnn_torch_adapter/NPUStorageImpl.cpp",
    "csrc/torch_binding.cpp",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cann", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    args = parser.parse_args()
    import torch
    import torch_npu

    if torch.npu.is_initialized():
        raise RuntimeError("The host adapter build requires an uninitialized NPU runtime")
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "source"
    identities = {}
    for relative in SOURCES:
        source = args.vendor / relative
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        identities[relative] = digest(target)

    binding_text = (snapshot / "csrc/torch_binding.cpp").read_text()
    pattern = (r'    ops\.def\(\n        "npu_sparse_flash_attention\(.*?'
               r'    ops\.impl\("npu_sparse_flash_attention", torch::kPrivateUse1, '
               r'&vllm_ascend::npu_sparse_flash_attention\);')
    matches = list(re.finditer(pattern, binding_text, re.DOTALL))
    if len(matches) != 1:
        raise RuntimeError("The pinned SFA registration block is not uniquely identifiable")
    match = matches[0]
    registration = match.group(0)
    translation_unit = args.output / "sfa_registration.cpp"
    translation_unit.write_text('''#include <dlfcn.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "aclnn_torch_adapter/op_api_common.h"
#include "attention/sparse_flash_attention/sparse_flash_attention_torch_adpt.h"

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
''' + registration + '''
}

extern "C" const char* vendor_hook_sfa_api_library(const char* name) {
    void* address = GetOpApiFuncAddr(name);
    Dl_info information{};
    if (address == nullptr || dladdr(address, &information) == 0) {
        return nullptr;
    }
    return information.dli_fname;
}
''')
    torch_root = Path(torch.__file__).resolve().parent
    npu_root = Path(torch_npu.__file__).resolve().parent
    arch = args.cann / f"{platform.machine()}-linux"
    includes = [snapshot / "csrc", Path(sysconfig.get_paths()["include"]),
                torch_root / "include", torch_root / "include/torch/csrc/api/include",
                args.cann / "include", arch / "include", npu_root / "include"]
    compiler = os.environ.get("CXX", "g++")
    base = [compiler, "-std=c++17", "-O2", "-fPIC",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"]
    for include in includes:
        base.extend(["-I", str(include)])
    sources = [translation_unit, *(snapshot / relative for relative in SOURCES
                                   if relative.endswith(".cpp")
                                   and not relative.endswith("torch_binding.cpp"))]
    commands, objects, dependencies = [], [], set()
    for source in sources:
        target = args.output / f"{source.stem}.o"
        dependency = target.with_suffix(".d")
        command = [*base, "-MD", "-MF", str(dependency), "-c", str(source),
                   "-o", str(target)]
        commands.append(command)
        subprocess.run(command, check=True)
        objects.append(target)
        dependency_text = dependency.read_text().replace("\\\n", " ")
        dependencies.update(Path(value) for value in
                            shlex.split(dependency_text.split(":", 1)[1]))
    library = args.output / "vendor_sfa_adapter.so"
    link = [compiler, "-shared", *(str(path) for path in objects), "-o", str(library)]
    for directory in (arch / "lib64", args.cann / "lib64",
                      torch_root / "lib", npu_root / "lib"):
        link.extend(["-L", str(directory)])
    link.extend(f"-l{name}" for name in ("torch", "torch_cpu", "torch_python", "c10",
                                        "torch_npu", "ascendcl", "dl"))
    subprocess.run(link, check=True)
    torch.ops.load_library(str(library))
    manifest = {
        "scope": "standalone build of the official SFA C++ adapter and registration",
        "vendor_commit": "8237ee6d690d1399bc6eb48ef5977ac2bd470f3f",
        "image_id": args.image_id, "python": sys.version,
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "script_sha256": digest(Path(__file__)), "source_sha256": identities,
        "registration_source_lines": [binding_text.count("\n", 0, match.start()) + 1,
                                      binding_text.count("\n", 0, match.end()) + 1],
        "registration_sha256": hashlib.sha256(registration.encode()).hexdigest(),
        "generated_unit_sha256": digest(translation_unit),
        "compile_commands": commands, "link_command": link,
        "dependency_sha256": {str(path): digest(path) for path in sorted(dependencies)},
        "library": str(library), "library_sha256": digest(library),
        "schema": str(torch.ops._C_ascend.npu_sparse_flash_attention.default._schema),
        "npu_initialized": torch.npu.is_initialized(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["npu_initialized"]:
        raise RuntimeError("The host adapter build initialized the NPU runtime")
    print(json.dumps({key: manifest[key] for key in (
        "scope", "library", "library_sha256", "schema", "npu_initialized")}), flush=True)


if __name__ == "__main__":
    main()
