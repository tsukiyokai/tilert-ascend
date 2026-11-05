"""Check the frozen Decoder graph fixture in its actual Torch image without devices."""

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs(root):
    checked = {}
    for line in (root / "frozen-inputs.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        actual = digest(root / name)
        if actual != expected:
            raise RuntimeError(f"Frozen input changed: {root / name}")
        checked[name] = actual
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--indexed", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if list(Path("/dev").glob("davinci*")):
        raise RuntimeError("The CPU gate must not have any davinci device mapping")
    decoder_hashes = verify_inputs(args.decoder)
    runtime_hashes = verify_inputs(args.runtime)
    sys.path[:0] = [str(args.decoder), str(args.runtime / "tests/glm52/whole_layer")]
    import torch
    import torch_npu

    initialized_before = torch.npu.is_initialized()
    if initialized_before:
        raise RuntimeError("Torch import initialized an NPU")
    spec = importlib.util.spec_from_file_location(
        "frozen_decoder_graph_contract", args.decoder / "test_decoder_graph.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(module))
    if not result.wasSuccessful():
        raise RuntimeError(log.getvalue())
    for library in (args.indexed, args.shared):
        torch.ops.load_library(str(library))
    schemas = sorted(str(value) for value in torch._C._jit_get_all_schemas()
                     if str(value).startswith(("tilert::indexed_attention",
                                               "tilert::shared_full_attention")))
    if len(schemas) != 4:
        raise RuntimeError(f"Unexpected attention schemas: {schemas}")
    base = [sys.executable, str(args.decoder / "device_graph.py"),
            "--runtime", str(args.runtime), "--output", "/tmp/unused-decoder-gate",
            "--extension-sha256", "unused",
            "--adapter", str(args.runtime / "tilert/models/glm_5_2_ascend/attention_native.py"),
            "--indexed-library", str(args.indexed), "--shared-library", str(args.shared),
            "--sfa-library", str(args.runtime / "native/sfa/vendor_sfa_adapter.so")]
    controls = []
    for extra, expected_code, expected_text in (
            (["--help"], 0, "--graph-residual-modes"),
            (["--forward-widths", "3"], 2, "Use widths 1/2/12/32"),
            (["--forward-replays", "0"], 2, "Use widths 1/2/12/32"),
            (["--graph-residual-modes", "none", "none"], 2,
             "Each initial residual branch needs one distinct capture")):
        completed = subprocess.run(base + extra, text=True, capture_output=True, check=False)
        if (completed.returncode != expected_code
                or expected_text not in completed.stdout + completed.stderr):
            raise RuntimeError(f"Unexpected CLI result: {completed}")
        controls.append({"arguments": extra, "exit_code": completed.returncode,
                         "stdout": completed.stdout, "stderr": completed.stderr})
    initialized_after = torch.npu.is_initialized()
    if initialized_after:
        raise RuntimeError("The CPU fixture check initialized an NPU")
    report = {
        "scope": "Frozen actual Torch212 imports, CPU contracts, CLI rejection and native schema co-load",
        "script_sha256": digest(Path(__file__)), "python": sys.version,
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "torch_path": torch.__file__, "torch_npu_path": torch_npu.__file__,
        "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
        "npu_initialized_before": initialized_before,
        "npu_initialized_after": initialized_after,
        "mapped_davinci_devices": [], "device_execution": False,
        "decoder_manifest_sha256": digest(args.decoder / "manifest.json"),
        "decoder_files_verified": len(decoder_hashes),
        "runtime_files_verified": len(runtime_hashes),
        "original_forward_sha256": decoder_hashes["decoder_forward.py"],
        "graph_source_sha256": decoder_hashes["decoder_graph.py"],
        "contracts_run": result.testsRun, "contract_log": log.getvalue(),
        "libraries": {str(path): digest(path) for path in (args.indexed, args.shared)},
        "schemas": schemas, "cli_controls": controls, "pass": True,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "torch", "torch_npu", "contracts_run", "decoder_files_verified",
        "runtime_files_verified", "npu_initialized_after", "pass")}), flush=True)


if __name__ == "__main__":
    main()
