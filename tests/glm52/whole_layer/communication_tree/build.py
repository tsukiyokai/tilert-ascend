"""Compile the private reducer without mapped NPU devices."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys


SHMEM_COMMIT = "24c15d21c26972404b0bb0e9cfd08d03930d15c6"
SHMEM_HEADERS_SHA = "0590362bf3a7faaff067a639f3e4e6f12846ebc7ff2b7eb71cec97acef1c35d2"
SHMEM_LIBRARIES = {
    "libshmem.so": "cf9b853bb7612ce1e44853b1d16851016c4a2c279bdc9bb32da0640bc411f532",
    "libshmem_utils.so": "743ea9661dbc6707fba5030967a22c59ebfdd0daed4884fdff39da12ae008ac0",
    "aclshmem_bootstrap_config_store.so":
        "ef9160f3706a7eac573ff0a83160ae2d38579550dd923f45b330f309a33d6541",
}
PARENT_LIBRARY_SHA = "1be5fa375d0d6e7a2a6dcf7f5265cf886317108398aeec940e2b1ce0de04e22f"
EXPORTS = ("initialize", "prepare", "execute", "destroy", "finalize", "last_error")
INCLUDE_PATHS = (
    "asc/impl/adv_api", "asc/impl/basic_api", "asc/impl/c_api",
    "asc/impl/basic_api/reg_compute", "asc/impl/simt_api", "asc/impl/utils", "asc",
    "asc/include", "asc/include/adv_api", "asc/include/basic_api",
    "asc/include/c_api", "asc/include/basic_api/reg_compute",
    "asc/include/simt_api", "asc/include/utils", "tikcpp/tikcfw",
    "tikcpp/tikcfw/impl", "tikcpp/tikcfw/interface", "tikcpp/tikcfw/lib",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, files):
    for name, expected in files.items():
        if digest(root / name) != expected:
            raise ValueError(f"Frozen input changed: {root / name}")


def run(output, label, command, env=None):
    with (output / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps({"step": label, "argv": list(map(str, command))}) + "\n")
    with (output / (label + ".log")).open("w") as stream:
        subprocess.run(command, check=True, env=env, stdout=stream,
                       stderr=subprocess.STDOUT)


def dependency_files(output):
    paths = set()
    for file in output.glob("*.d"):
        body = file.read_text().replace("\\\n", " ").split(":", 1)[1]
        paths.update(Path(name).resolve() for name in shlex.split(body))
    if not paths:
        raise RuntimeError("The compiler did not retain its dependency paths")
    return {str(path): digest(path) for path in sorted(paths)}


def build(output, cann, shmem, libraries, headers, control):
    import torch
    import torch_npu

    if (not torch.__version__.startswith("2.12.0")
            or torch_npu.__version__ != "2.12.0"
            or torch.npu.is_initialized()
            or list(Path("/dev").glob("davinci[0-9]*"))):
        raise RuntimeError("Use the recorded Torch212 image without mapped NPU devices")
    source = output / "source"
    manifest = json.loads((output / "manifest.json").read_text())
    verify(output, manifest["sources"])
    if digest(control) != PARENT_LIBRARY_SHA:
        raise ValueError("The accepted control DSO changed")
    if ((shmem / "SOURCE_COMMIT").read_text().strip() != SHMEM_COMMIT
            or digest(headers) != SHMEM_HEADERS_SHA):
        raise ValueError("The SHMEM revision or frozen header manifest changed")
    shmem_headers = {}
    for line in headers.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        relative = str(Path(name).relative_to("/workspace/communication-deps/shmem"))
        shmem_headers[relative] = expected
    verify(shmem, shmem_headers)
    verify(libraries, SHMEM_LIBRARIES)

    environment = dict(os.environ)
    environment["COMMUNICATION_TREE_PARENT"] = str(output / "parent")
    environment["COMMUNICATION_TREE_CONTROL"] = str(control)
    run(output, "python-contract", [sys.executable, str(source / "test_prepare.py")],
        environment)
    run(output, "plan-contract-build", ["c++", "-std=c++17", "-O2", "-Wall",
        "-Wextra", "-Werror", "-fno-fast-math", "-ffp-contract=off",
        str(source / "contract_test.cpp"), "-o", str(output / "plan-contract")])
    run(output, "plan-contract", [str(output / "plan-contract")])
    compiler = cann / "bin/bisheng"
    target = cann / (platform.machine() + "-linux")
    includes = ["-I", str(source), "-I", str(cann / "include"),
                "-I", str(shmem / "include"), "-I", str(shmem / "src/device")]
    for relative in INCLUDE_PATHS:
        if (target / relative).is_dir():
            includes.extend(["-I", str(target / relative)])
    run(output, "compiler-version", [str(compiler), "--version"])
    run(output, "host-compiler-version", ["c++", "--version"])
    run(output, "device-compile", [str(compiler), "-xcce", "--npu-arch=dav-2201",
        "-std=c++17", "-O2", "-ffp-contract=off", "-fPIC", *includes,
        "-MD", "-MF", str(output / "device.d"), "-c",
        str(source / "communication_probe.cpp"), "-o", str(output / "device.o")])
    run(output, "host-compile", ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
        "-Wno-missing-field-initializers", "-fPIC", "-I", str(cann / "include"),
        "-I", str(shmem / "include"), "-MD", "-MF", str(output / "host.d"),
        "-c", str(source / "communication_host.cpp"), "-o", str(output / "host.o")])
    run(output, "probe-host-compile", ["c++", "-std=c++17", "-O2", "-fPIC", *includes,
        "-MD", "-MF", str(output / "probe-host.d"), "-c",
        str(source / "communication_probe_host.cpp"),
        "-o", str(output / "probe-host.o")])
    library = output / "communication_tree_probe.so"
    run(output, "link", ["c++", "-shared", str(output / "device.o"),
        str(output / "host.o"), str(output / "probe-host.o"), "-o", str(library),
        str(libraries / "libshmem.so"), "-L", str(target / "lib64"),
        "-L", str(cann / "lib64"), "-ltiling_api", "-lplatform", "-lascendcl",
        "-lruntime", "-lunified_dlog", "-lc_sec", "-lregister", "-lprofapi"])
    run(output, "exports", ["nm", "-D", "--defined-only", str(library)])
    symbols = {line.split()[-1] for line in (output / "exports.log").read_text().splitlines()}
    for name in EXPORTS:
        if "communication_tree_probe_" + name not in symbols:
            raise RuntimeError(f"Missing private C ABI: {name}")
        if "communication_probe_" + name in symbols:
            raise RuntimeError(f"Parent C ABI leaked into private DSO: {name}")
    handles = [ctypes.CDLL(str(control)), ctypes.CDLL(str(library))]
    addresses = {}
    for handle, prefix in zip(handles, ("communication_probe_", "communication_tree_probe_")):
        addresses[prefix] = {
            name: ctypes.cast(getattr(handle, prefix + name), ctypes.c_void_p).value
            for name in EXPORTS}
        last_error = getattr(handle, prefix + "last_error")
        last_error.argtypes = []
        last_error.restype = ctypes.c_char_p
        if last_error() != b"":
            raise RuntimeError("Loading a fresh probe unexpectedly set its error state")
    for name in EXPORTS:
        if addresses["communication_probe_"][name] == addresses["communication_tree_probe_"][name]:
            raise RuntimeError("Private C ABI resolved to the parent implementation")
    if torch.npu.is_initialized():
        raise RuntimeError("The no-device co-load initialized the NPU")
    dependencies = dependency_files(output)
    mapped = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = Path(line.split()[-1])
        if (path.is_absolute() and path.is_file()
                and (str(path).startswith(str(cann))
                     or str(path).startswith(str(libraries)))):
            mapped[str(path)] = digest(path)
    verify(output, manifest["sources"])
    verify(shmem, shmem_headers)
    verify(libraries, SHMEM_LIBRARIES)
    report = {
        "pass": True, "manifest_sha256": digest(output / "manifest.json"),
        "parent_library_sha256": digest(control), "library_sha256": digest(library),
        "namespace": "whole_layer_tree", "c_prefix": "communication_tree_probe_",
        "exports": EXPORTS, "symbol_addresses": addresses,
        "plan_contract": json.loads((output / "plan-contract.log").read_text()),
        "python_contract_tests": 5,
        "shmem_commit": SHMEM_COMMIT, "shmem_header_manifest_sha256": digest(headers),
        "shmem_headers": shmem_headers, "shmem_libraries": SHMEM_LIBRARIES,
        "compiler_sha256": digest(compiler.resolve()),
        "host_compiler_sha256": digest(Path(shutil.which("c++")).resolve()),
        "compile_dependencies": dependencies, "mapped_dependency_libraries": mapped,
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "npu_initialized_before": False, "npu_initialized_after": False,
        "mapped_davinci_devices": [], "device_execution": False,
        "co_load": True, "shmem_initialize_called": False,
        "artifacts": {path.name: digest(path) for path in sorted(output.iterdir())
                      if path.is_file() and path.suffix in (".o", ".so", ".log", ".d", ".jsonl")},
    }
    (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("pass", "library_sha256", "plan_contract", "device_execution")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    for name in ("cann", "shmem", "libraries", "headers", "control"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    build(*(getattr(args, name).resolve() for name in
            ("output", "cann", "shmem", "libraries", "headers", "control")))
