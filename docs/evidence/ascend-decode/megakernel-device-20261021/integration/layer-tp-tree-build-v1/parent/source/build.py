"""Build the frozen TP layer with CANN 9.2 and the recorded SHMEM dependency."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess


SHMEM_COMMIT = "24c15d21c26972404b0bb0e9cfd08d03930d15c6"
SHMEM_HEADERS_SHA = "0590362bf3a7faaff067a639f3e4e6f12846ebc7ff2b7eb71cec97acef1c35d2"
SHMEM_LIBRARIES = {
    "libshmem.so": "cf9b853bb7612ce1e44853b1d16851016c4a2c279bdc9bb32da0640bc411f532",
    "libshmem_utils.so": "743ea9661dbc6707fba5030967a22c59ebfdd0daed4884fdff39da12ae008ac0",
    "aclshmem_bootstrap_config_store.so": "ef9160f3706a7eac573ff0a83160ae2d38579550dd923f45b330f309a33d6541",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, files):
    for name, expected in files.items():
        if digest(root / name) != expected:
            raise ValueError(f"Frozen input changed: {root / name}")


def run(output, label, command):
    with (output / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps({"step": label, "argv": command}) + "\n")
    with (output / (label + ".log")).open("w") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)


def build(output, cann, vendor, shmem, libraries, headers):
    import torch
    import torch_npu

    if (not torch.__version__.startswith("2.12.0")
            or torch_npu.__version__ != "2.12.0"
            or not torch.compiled_with_cxx11_abi()
            or torch.npu.is_initialized()
            or list(Path("/dev").glob("davinci[0-9]*"))):
        raise RuntimeError("Use the recorded Torch212 ABI with no mapped NPU")
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
    commands = [json.loads(line) for line in
                (output / "local/ffn-commands.jsonl").read_text().splitlines()]
    parent = next(item["argv"] for item in commands if item["step"] == "kernel")
    parent_root = str(Path(parent[parent.index("-o") + 1]).parent)
    command = [value.replace(parent_root, str(output / "local/ffn")).replace(
        "/opt/ascend/cann-9.2.0", str(cann)) for value in parent]
    command[command.index("-c") + 1] = str(output / "source/kernel.cpp")
    command[command.index("-o") + 1] = str(output / "kernel.o")
    command.extend(["-I", str(vendor), "-I", str(shmem / "include"),
                    "-I", str(shmem / "src/device")])
    definitions = [value for value in command if value.startswith("-D")]
    run(output, "host-contract-build", ["c++", "-std=c++17", "-O2", "-Wall",
        "-Wextra", "-Werror", *definitions,
        str(output / "source/contract_test.cpp"), "-o", str(output / "contract-test")])
    run(output, "host-contract", [str(output / "contract-test")])
    run(output, "kernel", command)
    torch_root = Path(torch.__file__).parent
    npu_root = Path(torch_npu.__file__).parent
    for label, source in (("host", "source/host.cpp"),
                          ("communication-host", "communication/communication_host.cpp")):
        run(output, label, ["c++", "-std=c++17", "-O2", "-fPIC",
            "-D_GLIBCXX_USE_CXX11_ABI=1", *definitions,
            "-I", str(torch_root / "include"),
            "-I", str(torch_root / "include/torch/csrc/api/include"),
            "-I", str(npu_root / "include"), "-I", str(cann / "include"),
            "-I", str(shmem / "include"), "-c", str(output / source),
            "-o", str(output / (label + ".o"))])
    library = output / "layer_tp_chain.so"
    run(output, "link", ["c++", "-shared", "-Wl,--no-undefined",
        *(str(output / (name + ".o")) for name in
          ("kernel", "host", "communication-host")), "-o", str(library),
        "-L", str(libraries), "-lshmem", "-lshmem_utils",
        "-L", str(cann / "aarch64-linux/lib64"), "-L", str(cann / "lib64"),
        "-lascendcl", "-lruntime", "-lunified_dlog", "-lc_sec", "-lregister",
        "-lprofapi", "-ltiling_api", "-lplatform",
        "-L", str(torch_root / "lib"), "-L", str(npu_root / "lib"),
        "-ltorch", "-ltorch_cpu", "-lc10", "-ltorch_npu"])
    run(output, "exports", ["nm", "-D", "--defined-only", str(library)])
    symbols = (output / "exports.log").read_text()
    exports = ("indexed_launch", "shared_launch", "last_error", "record_bytes",
               "initialize", "window_create", "window_info", "submit",
               "window_destroy", "finalize")
    for name in exports:
        if not any(line.split()[-1] == "layer_tp_" + name for line in symbols.splitlines()):
            raise RuntimeError(f"Missing TP layer export: {name}")
    loaded = ctypes.CDLL(str(library))
    loaded.layer_tp_record_bytes.argtypes = [ctypes.c_int]
    loaded.layer_tp_record_bytes.restype = ctypes.c_size_t
    sizes = [loaded.layer_tp_record_bytes(index) for index in range(4)]
    if sizes != [664, 928, 48, 0]:
        raise RuntimeError(f"Loaded TP record ABI differs: {sizes}")
    loaded.layer_tp_finalize.argtypes = []
    loaded.layer_tp_finalize.restype = ctypes.c_int
    loaded.layer_tp_last_error.argtypes = []
    loaded.layer_tp_last_error.restype = ctypes.c_char_p
    if loaded.layer_tp_finalize() == 0 or not loaded.layer_tp_last_error():
        raise RuntimeError("An uninitialized SHMEM context was accepted")
    if torch.npu.is_initialized():
        raise RuntimeError("Loading the TP host shim initialized the NPU")
    verify(output, manifest["sources"])
    verify(shmem, shmem_headers)
    verify(libraries, SHMEM_LIBRARIES)
    report = {
        "parent_manifest_sha256": digest(output / "manifest.json"),
        "library_sha256": digest(library), "exports": list(exports),
        "host_contract": json.loads((output / "host-contract.log").read_text()),
        "shmem_commit": SHMEM_COMMIT, "shmem_headers": shmem_headers,
        "shmem_libraries": SHMEM_LIBRARIES,
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "cxx11_abi": torch.compiled_with_cxx11_abi(),
        "npu_initialized_before": False, "npu_initialized_after": False,
        "mapped_davinci_devices": [], "dlopen_and_abi": True,
        "uninitialized_context_rejected": True, "device_execution": False,
        "artifacts": {path.name: digest(path) for path in output.iterdir()
                      if path.is_file() and path.suffix in (".o", ".so", ".log", ".jsonl")},
    }
    (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"library_sha256": report["library_sha256"],
                      "host_contract": report["host_contract"],
                      "device_execution": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    for name in ("cann", "vendor", "shmem", "libraries", "headers"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    build(*(getattr(args, name).resolve() for name in
            ("output", "cann", "vendor", "shmem", "libraries", "headers")))
