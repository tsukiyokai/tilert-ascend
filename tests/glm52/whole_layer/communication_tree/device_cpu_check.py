"""Load both frozen probe ABIs and the unchanged driver without an NPU."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root, libraries, output):
    import torch
    import torch_npu
    from communication_tree_devicecheck import NativeProbe, TreeProbe, cpu_contract

    if (not torch.__version__.startswith("2.12.0")
            or torch_npu.__version__ != "2.12.0"
            or torch.npu.is_initialized()
            or list(Path("/dev").glob("davinci[0-9]*"))):
        raise RuntimeError("The CPU gate requires Torch212 without mapped or initialized NPU")
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Frozen fixture changed: {name}")
    for name, expected in manifest["shmem_libraries"].items():
        if digest(libraries / name) != expected:
            raise ValueError(f"The SHMEM dependency changed: {name}")
    build = json.loads((root / "build/build.json").read_text())
    parent = NativeProbe(root / "parent_probe.so")
    candidate = TreeProbe(root / "candidate_probe.so")
    addresses = {}
    for probe, prefix in ((parent, "communication_probe_"),
                          (candidate, "communication_tree_probe_")):
        addresses[prefix] = {
            name: ctypes.cast(getattr(probe.library, prefix + name), ctypes.c_void_p).value
            for name in build["exports"]}
        error = getattr(probe.library, prefix + "last_error")
        error.argtypes = []
        error.restype = ctypes.c_char_p
        if error() != b"":
            raise RuntimeError("A newly loaded probe has a nonempty error state")
    for name in build["exports"]:
        if addresses["communication_probe_"][name] == addresses["communication_tree_probe_"][name]:
            raise RuntimeError("The two C ABIs resolved to the same symbol")
    shared = [ctypes.cast(getattr(probe.library, "aclshmemx_init_status"),
                          ctypes.c_void_p).value for probe in (parent, candidate)]
    if not shared[0] or shared[0] != shared[1]:
        raise RuntimeError("The probes do not resolve the same SHMEM instance")
    oracle = cpu_contract(torch)
    mapped = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = Path(line.split()[-1])
        if path.is_absolute() and path.is_file() and (
                str(path).startswith("/opt/ascend/cann-9.2.0/")
                or str(path).startswith(str(libraries))):
            mapped[str(path)] = digest(path)
    expected_mapped = {
        name.replace("/shmem-lib/", str(libraries) + "/"): sha
        for name, sha in build["mapped_dependency_libraries"].items()}
    for name, sha in mapped.items():
        if name not in expected_mapped or expected_mapped[name] != sha:
            raise RuntimeError(f"A runtime dependency differs from the CPU build: {name}")
    if torch.npu.is_initialized():
        raise RuntimeError("The CPU contract initialized the NPU")
    report = {
        "pass": True, "scope": "No-device dual-probe ABI and original driver arithmetic",
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "fixture_manifest_sha256": digest(root / "manifest.json"),
        "driver_sha256": digest(root / "communication_tree_devicecheck.py"),
        "parent_sha256": digest(root / "parent_probe.so"),
        "candidate_sha256": digest(root / "candidate_probe.so"),
        "frozen_file_count": len(manifest["files_sha256"]),
        "frozen_build_source_count": len(json.loads((root / "build/manifest.json").read_text())["sources"]),
        "compile_dependency_count": len(build["compile_dependencies"]),
        "shmem_libraries": manifest["shmem_libraries"],
        "c_abi_addresses": addresses, "shared_shmem_status_function_addresses": shared,
        "same_shmem_instance": True, "shmem_initialize_calls": 0,
        "driver_cpu_contract": oracle, "mapped_dependencies": mapped,
        "npu_initialized_before": False, "npu_initialized_after": False,
        "mapped_davinci_devices": [], "device_execution": False,
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("pass", "fixture_manifest_sha256", "parent_sha256",
                       "candidate_sha256", "same_shmem_instance", "device_execution")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--libraries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.fixture.resolve(), args.libraries.resolve(), args.output.resolve())
