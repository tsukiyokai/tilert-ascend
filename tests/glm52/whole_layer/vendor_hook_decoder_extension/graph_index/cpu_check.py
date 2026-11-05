"""Verify the frozen graph-index observation in Torch212 without device access."""

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


def verify(root):
    entries = {}
    for line in (root / "frozen-inputs.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(root / name) != expected:
            raise RuntimeError(f"Frozen input changed: {root / name}")
        entries[name] = expected
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--indexed", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--failure-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if list(Path("/dev").glob("davinci*")):
        raise RuntimeError("No davinci device may be mapped into this CPU gate")
    decoder, runtime = verify(args.decoder), verify(args.runtime)
    manifest = json.loads((args.decoder / "manifest.json").read_text())
    for path, expected in (
            (args.indexed, manifest["required_indexed_library_sha256"]),
            (args.shared, manifest["required_shared_library_sha256"]),
            (args.failure, manifest["failure_sha256"]),
            (args.failure_report, manifest["failure_report_sha256"])):
        if digest(path) != expected:
            raise RuntimeError(f"An external frozen input changed: {path}")
    sys.path[:0] = [str(args.decoder), str(args.runtime / "tests/glm52/whole_layer")]
    import torch
    import torch_npu
    before = torch.npu.is_initialized()
    if before:
        raise RuntimeError("Torch import initialized an NPU")
    log, suite = io.StringIO(), unittest.TestSuite()
    for name in ("test_decoder_graph.py", "test_graph_index_diagnostic.py"):
        spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), args.decoder / name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError(log.getvalue())
    for path in (args.indexed, args.shared):
        torch.ops.load_library(str(path))
    schemas = sorted(str(value) for value in torch._C._jit_get_all_schemas()
                     if str(value).startswith(("tilert::indexed_attention", "tilert::shared_full_attention")))
    if len(schemas) != 4:
        raise RuntimeError(f"Unexpected native attention schemas: {schemas}")
    base = [sys.executable, str(args.decoder / "device_graph_index.py"),
            "--runtime", str(args.runtime), "--output", "/tmp/unused-graph-index-gate",
            "--extension-sha256", "unused",
            "--adapter", str(args.runtime / "tilert/models/glm_5_2_ascend/attention_native.py"),
            "--indexed-library", str(args.indexed), "--shared-library", str(args.shared),
            "--sfa-library", str(args.runtime / "native/sfa/vendor_sfa_adapter.so"),
            "--graph-index-parent", str(args.failure),
            "--graph-index-parent-report", str(args.failure_report)]
    controls = []
    for extra, code, message in (
            (["--help"], 0, "--graph-index-parent-report"),
            (["--forward-widths", "12", "--forward-replays", "3", "--graph-residual-modes", "provided"],
             2, "Use C8 M1, provided residual, and exactly three replays"),
            (["--forward-widths", "1", "--forward-replays", "2", "--graph-residual-modes", "provided"],
             2, "Use C8 M1, provided residual, and exactly three replays"),
            (["--forward-widths", "1", "--forward-replays", "3", "--graph-residual-modes", "none"],
             2, "Use C8 M1, provided residual, and exactly three replays")):
        completed = subprocess.run(base + extra, text=True, capture_output=True, check=False)
        if completed.returncode != code or message not in completed.stdout + completed.stderr:
            raise RuntimeError(f"Unexpected CLI result: {completed}")
        controls.append({"arguments": extra, "exit_code": completed.returncode,
                         "stdout": completed.stdout, "stderr": completed.stderr})
    after = torch.npu.is_initialized()
    if after:
        raise RuntimeError("The CPU gate initialized an NPU")
    report = {
        "scope": "Actual Torch212 CPU imports, failure-lifetime contracts, CLI refusal and accepted DSO schema co-load",
        "script_sha256": digest(Path(__file__)), "python": sys.version,
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "torch_path": torch.__file__, "torch_npu_path": torch_npu.__file__,
        "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
        "npu_initialized_before": before, "npu_initialized_after": after,
        "mapped_davinci_devices": [], "device_execution": False,
        "decoder_files_verified": len(decoder), "runtime_files_verified": len(runtime),
        "manifest_sha256": digest(args.decoder / "manifest.json"),
        "original_graph_sha256": decoder["decoder_graph.py"],
        "contracts_run": result.testsRun, "contract_log": log.getvalue(),
        "libraries": {str(path): digest(path) for path in (args.indexed, args.shared)},
        "schemas": schemas, "cli_controls": controls, "pass": True,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: report[name] for name in (
        "torch", "torch_npu", "contracts_run", "decoder_files_verified",
        "runtime_files_verified", "npu_initialized_after", "pass")}), flush=True)


if __name__ == "__main__":
    main()
