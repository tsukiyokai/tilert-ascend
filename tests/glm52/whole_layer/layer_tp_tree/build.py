"""Build the private TP tree layer with the frozen C2 compiler configuration."""

import argparse
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from prepare import EXPORTS, PARENT_LIBRARY, verify


SHMEM_COMMIT = "24c15d21c26972404b0bb0e9cfd08d03930d15c6"
SHMEM_HEADERS_SHA = "0590362bf3a7faaff067a639f3e4e6f12846ebc7ff2b7eb71cec97acef1c35d2"
SHMEM_LIBRARIES = {
    "libshmem.so": "cf9b853bb7612ce1e44853b1d16851016c4a2c279bdc9bb32da0640bc411f532",
    "libshmem_utils.so": "743ea9661dbc6707fba5030967a22c59ebfdd0daed4884fdff39da12ae008ac0",
    "aclshmem_bootstrap_config_store.so":
        "ef9160f3706a7eac573ff0a83160ae2d38579550dd923f45b330f309a33d6541",
}
DEFINITIONS = [
    "-DROUTED_ROW_BALANCED=1", "-DROUTED_SHARED_PIPELINE=1",
    "-DROUTED_INPUT_FRONT=1", "-DSHARED_W8_DYNAMIC_PERFORMANCE_QUANT=1",
    "-DROUTED_ACTIVE_LIST=1",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output, label, command):
    with (output / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps({"step": label, "argv": command}) + "\n")
    with (output / (label + ".log")).open("w") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)


def dependency_files(output):
    paths = set()
    for path in output.glob("*.d"):
        body = path.read_text().replace("\\\n", " ").split(":", 1)[1]
        paths.update(Path(name).resolve() for name in shlex.split(body))
    if not paths:
        raise RuntimeError("The compilers did not retain dependency paths")
    return {str(path): digest(path) for path in sorted(paths)}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def co_load(output, control):
    candidate = output / "layer_tp_tree_chain.so"
    modules = [load_module(output / "parent/host.py", "parent_tp_host"),
               load_module(output / "source/host.py", "private_tp_tree_host")]
    libraries = [module.Library(path, digest(path))
                 for module, path in zip(modules, (control, candidate))]
    prefixes = ("layer_tp_", "layer_tp_tree_")
    addresses = {}
    for library, prefix in zip(libraries, prefixes):
        handle = library.handle
        addresses[prefix] = {
            name: ctypes.cast(getattr(handle, prefix + name), ctypes.c_void_p).value
            for name in EXPORTS}
        sizes = [getattr(handle, prefix + "record_bytes")(index) for index in range(4)]
        if sizes != [664, 928, 48, 0]:
            raise RuntimeError(f"Record ABI changed: {sizes}")
        if getattr(handle, prefix + "last_error")() != b"":
            raise RuntimeError("A newly loaded layer already has an error state")
    for name in EXPORTS:
        if addresses[prefixes[0]][name] == addresses[prefixes[1]][name]:
            raise RuntimeError(f"Private export resolves to the parent: {name}")
    for index, (library, prefix) in enumerate(zip(libraries, prefixes)):
        if getattr(library.handle, prefix + "finalize")() == 0:
            raise RuntimeError("An uninitialized SHMEM context was accepted")
        if not getattr(library.handle, prefix + "last_error")():
            raise RuntimeError("The rejected context did not retain its error")
        if index == 0 and libraries[1].handle.layer_tp_tree_last_error() != b"":
            raise RuntimeError("The parent changed the private host error registry")
    return {"exports": addresses, "record_bytes": [664, 928, 48, 0],
            "separate_host_error_registries": True,
            "uninitialized_context_rejected": True,
            "shmem_initialize_called": False}


def build(output, cann, vendor, shmem, libraries, headers, control):
    import torch
    import torch_npu

    if (not torch.__version__.startswith("2.12.0")
            or torch_npu.__version__ != "2.12.0"
            or not torch.compiled_with_cxx11_abi()
            or torch.npu.is_initialized()
            or list(Path("/dev").glob("davinci[0-9]*"))):
        raise RuntimeError("Use the recorded Torch212 ABI without mapped NPU devices")
    if digest(control) != PARENT_LIBRARY:
        raise ValueError("The accepted C2 control library changed")
    manifest = json.loads((output / "manifest.json").read_text())
    verify(output, manifest["sources"])
    provenance = json.loads((output / "local/attention-provenance.json").read_text())
    verify(vendor, provenance["vendor_headers"])
    if ((shmem / "SOURCE_COMMIT").read_text().strip() != SHMEM_COMMIT
            or digest(headers) != SHMEM_HEADERS_SHA):
        raise ValueError("The recorded SHMEM revision or header manifest changed")
    shmem_headers = {}
    for line in headers.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        relative = str(Path(name).relative_to("/workspace/communication-deps/shmem"))
        shmem_headers[relative] = expected
    verify(shmem, shmem_headers)
    verify(libraries, SHMEM_LIBRARIES)
    parent_commands = {record["step"]: record["argv"] for record in
                       map(json.loads, (output / "parent/commands.jsonl")
                           .read_text().splitlines())}
    replacements = {"/candidate": str(output), "/opt/ascend/cann-9.2.0": str(cann),
                    "/vendor": str(vendor), "/shmem/": str(shmem) + "/",
                    "/shmem-lib": str(libraries)}

    def original_command(label):
        command = list(parent_commands[label])
        for before, after in replacements.items():
            command = [part.replace(before, after) for part in command]
        return command

    if [part for part in original_command("kernel") if part.startswith("-D")] != DEFINITIONS:
        raise RuntimeError("The parent's routing or Matmul compile definitions changed")
    run(output, "python-contract", [sys.executable, str(output / "source/test_contract.py")])
    run(output, "host-contract-build", original_command("host-contract-build"))
    run(output, "host-contract", [str(output / "contract-test")])
    run(output, "integration-contract-build", ["c++", "-std=c++17", "-O2", "-Wall",
        "-Wextra", "-Werror", *DEFINITIONS, str(output / "source/integration_contract.cpp"),
        "-o", str(output / "integration-contract")])
    run(output, "integration-contract", [str(output / "integration-contract")])
    run(output, "tree-contract-build", ["c++", "-std=c++17", "-O2", "-Wall",
        "-Wextra", "-Werror", "-fno-fast-math", "-ffp-contract=off",
        str(output / "communication/contract_test.cpp"), "-o", str(output / "tree-contract")])
    run(output, "tree-contract", [str(output / "tree-contract")])
    run(output, "compiler-version", [str(cann / "bin/bisheng"), "--version"])
    run(output, "host-compiler-version", ["c++", "--version"])
    command_contract = {}
    for label in ("kernel", "host", "communication-host"):
        command = original_command(label)
        compiled = command + ["-MD", "-MF", str(output / (label + ".d"))]
        run(output, label, compiled)
        command_contract[label] = {"parent_options_preserved": compiled[:-3] == command,
                                   "only_added_options": compiled[-3:]}
    link = original_command("link")
    link[link.index("-o") + 1] = str(output / "layer_tp_tree_chain.so")
    run(output, "link", link)
    expected_link = original_command("link")
    restored_link = list(link)
    restored_link[restored_link.index("-o") + 1] = expected_link[expected_link.index("-o") + 1]
    if restored_link != expected_link:
        raise RuntimeError("Link options changed beyond the library name")
    library = output / "layer_tp_tree_chain.so"
    run(output, "exports", ["nm", "-D", "--defined-only", str(library)])
    symbols = {line.split()[-1] for line in (output / "exports.log").read_text().splitlines()}
    for name in EXPORTS:
        if "layer_tp_tree_" + name not in symbols or "layer_tp_" + name in symbols:
            raise RuntimeError(f"The private C ABI is missing or exposes a parent symbol: {name}")
    loaded = co_load(output, control)
    if torch.npu.is_initialized():
        raise RuntimeError("CPU co-loading initialized the NPU")
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
        "namespace": "layer_tp_tree_chain", "c_prefix": "layer_tp_tree_",
        "python_contract_tests": 7,
        "host_contract": json.loads((output / "host-contract.log").read_text()),
        "integration_contract": json.loads((output / "integration-contract.log").read_text()),
        "tree_contract": json.loads((output / "tree-contract.log").read_text()),
        "kernel_definitions": DEFINITIONS, "compiler_command_contract": command_contract,
        "link_options_preserved": True, "co_load": loaded,
        "shmem_commit": SHMEM_COMMIT, "shmem_header_manifest_sha256": digest(headers),
        "shmem_headers": shmem_headers, "shmem_libraries": SHMEM_LIBRARIES,
        "compiler_sha256": digest((cann / "bin/bisheng").resolve()),
        "host_compiler_sha256": digest(Path(shutil.which("c++")).resolve()),
        "compile_dependencies": dependencies, "mapped_dependency_libraries": mapped,
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "cxx11_abi": torch.compiled_with_cxx11_abi(),
        "npu_initialized_before": False, "npu_initialized_after": False,
        "mapped_davinci_devices": [], "device_execution": False,
        "artifacts": {path.name: digest(path) for path in sorted(output.iterdir())
                      if path.is_file() and path.suffix in (".o", ".so", ".log", ".d", ".jsonl")},
    }
    (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("pass", "library_sha256", "integration_contract", "device_execution")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    for name in ("cann", "vendor", "shmem", "libraries", "headers", "control"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    build(*(getattr(args, name).resolve() for name in
            ("output", "cann", "vendor", "shmem", "libraries", "headers", "control")))
