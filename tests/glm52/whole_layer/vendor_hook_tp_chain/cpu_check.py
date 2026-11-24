"""Verify the eager TP fixture in the frozen Torch212 environment without NPUs."""

import argparse
import ctypes
import hashlib
import importlib
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    lines = (root / "frozen-inputs.sha256").read_text().splitlines()
    for line in lines:
        expected, name = line.split("  ", 1)
        if digest(root / name) != expected:
            raise RuntimeError(f"A frozen input changed: {name}")
    return len(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("decoder", "runtime", "indexed", "shared", "shmem", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if list(Path("/dev").glob("davinci*")):
        raise RuntimeError("The CPU gate must not map a davinci device")
    counts = {"decoder": verify(args.decoder), "runtime": verify(args.runtime)}
    manifest = json.loads((args.decoder / "manifest.json").read_text())
    shmem = json.loads((args.decoder / "shmem-libraries.json").read_text())
    for name, expected in shmem.items():
        if digest(args.shmem / name) != expected:
            raise RuntimeError(f"A required SHMEM library changed: {name}")
    sys.path[:0] = [str(args.decoder), str(args.runtime / "tests/glm52/whole_layer")]
    import torch
    import torch_npu
    import local_chain
    import tp_chain
    import tp_constructor
    import tp_device
    import tp_groups
    from local_chain_host import Library
    from tp_host import Library as TPLibrary
    from vllm.distributed import parallel_state
    from vllm_ascend.patch.worker.patch_distributed import GroupCoordinatorPatch
    from vllm_ascend.utils import create_hccl_pg_options

    if (not torch.__version__.startswith("2.12.0") or torch_npu.__version__ != "2.12.0"
            or torch.npu.is_initialized()):
        raise RuntimeError("Use the recorded Torch212 ABI without NPU initialization")
    for path, key in ((args.indexed, "indexed_library_sha256"),
                      (args.shared, "shared_attention_library_sha256")):
        if digest(path) != manifest[key]:
            raise RuntimeError("A separate attention parent changed")
        torch.ops.load_library(str(path))
    local_chain.load_ffn(args.decoder / "complete_ffn.so")
    local = Library(args.decoder / "layer_local_chain.so", local_chain.CHAIN_SHA)
    tp = TPLibrary(args.decoder / "layer_tp_chain.so", tp_chain.TP_SHA)
    acl = ctypes.CDLL("libascendcl.so")
    for name in ("aclrtGetPhyDevIdByLogicDevId", "aclrtDeviceCanAccessPeer"):
        getattr(acl, name)
    log = io.StringIO()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(
        importlib.import_module(name)) for name in ("test_local_chain", "test_tp_contract"))
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError(log.getvalue())
    from tp_nonmember_cpu import run as run_nonmember
    nonmember = run_nonmember()
    command = [sys.executable, str(args.decoder / "tp_device.py"),
               "--runtime", str(args.runtime), "--output", "/tmp/not-started",
               "--extension-sha256", "unused", "--physical-devices", "6", "7",
               "--adapter", str(args.runtime / "tilert/models/glm_5_2_ascend/attention_native.py"),
               "--indexed-library", str(args.indexed), "--shared-library", str(args.shared),
               "--sfa-library", str(args.runtime / "native/sfa/vendor_sfa_adapter.so")]
    controls = []
    for extra, code, message in (
            (["--help"], 0, "--tp-library"),
            (["--physical-devices", "7", "6"], 2, "ordered distinct devices"),
            (["--forward-widths", "3"], 2, "ordered distinct devices"),
            (["--forward-replays", "0"], 2, "ordered distinct devices"),
            (["--graph-residual-modes", "none"], 2, "invalid choice")):
        run = subprocess.run(command + extra, text=True, capture_output=True, check=False)
        if run.returncode != code or message not in run.stdout + run.stderr:
            raise RuntimeError(f"Unexpected CLI result: {run}")
        controls.append({"arguments": extra, "exit_code": run.returncode,
                         "stdout": run.stdout, "stderr": run.stderr})
    shell = subprocess.run(["bash", "-n", str(args.decoder / "device_tp_run.sh")],
                           text=True, capture_output=True, check=True)
    if torch.npu.is_initialized() or Path("/tmp/not-started").exists():
        raise RuntimeError("The CPU gate entered a device worker")
    sources = [tp_constructor.run_device_rank, tp_device.worker,
               tp_groups.create_communication_groups,
               parallel_state.init_model_parallel_group, GroupCoordinatorPatch,
               create_hccl_pg_options]
    report = {
        "scope": "Actual Torch212 and five-DSO co-load; CPU contracts plus real three-rank Gloo nonmember probe; no HCCL group or NPU created",
        "python": sys.version, "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "torch_path": torch.__file__, "torch_npu_path": torch_npu.__file__,
        "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
        "npu_initialized_before": False, "npu_initialized_after": False,
        "mapped_davinci_devices": [], "device_execution": False,
        "files_verified": counts, "manifest_sha256": digest(args.decoder / "manifest.json"),
        "actual_sources": {inspect.getfile(value): digest(inspect.getfile(value)) for value in sources},
        "local_record_bytes": [local.config_bytes, local.address_bytes],
        "tp_record_bytes": [tp.handle.layer_tp_record_bytes(index) for index in range(3)],
        "contracts_run": result.testsRun, "contract_log": log.getvalue(),
        "actual_nonmember_probe": nonmember,
        "cli_controls": controls, "launcher_syntax_exit": shell.returncode,
        "libraries": {str(path): digest(path) for path in (
            args.indexed, args.shared, args.decoder / "complete_ffn.so",
            args.decoder / "layer_local_chain.so", args.decoder / "layer_tp_chain.so")},
        "shmem_libraries": shmem,
        "script_sha256": digest(Path(__file__)), "pass": True,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("contracts_run", "npu_initialized_after", "pass")}), flush=True)


if __name__ == "__main__":
    main()
