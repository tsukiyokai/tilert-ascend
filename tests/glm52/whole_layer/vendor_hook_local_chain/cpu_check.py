"""Load the frozen local-chain fixture under Torch212 without NPU access."""

import argparse
import ast
import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_inputs(directory):
    count = 0
    for line in (directory / "frozen-inputs.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(directory / name) != expected:
            raise RuntimeError(f"A frozen input changed: {name}")
        count += 1
    return count


def unchanged_storage_contract(directory):
    parent = ast.parse((directory / "decoder_storage_full_graph.py").read_text())
    candidate_text = (directory / "decoder_storage_local_chain.py").read_text()
    old = "One-time INT32 NZ carrier descriptors before original Decoder calls through the official ACLGraphWrapper"
    new = "One-time INT32 NZ carrier descriptors before the eager local component chain"
    if candidate_text.count(new) != 1 or old in candidate_text:
        raise RuntimeError("The storage report must describe the eager local chain")
    candidate = ast.parse(candidate_text.replace(new, old))
    definitions = lambda source: [ast.dump(node) for node in source.body
                                  if isinstance(node, ast.FunctionDef)]
    if definitions(parent) != definitions(candidate):
        raise RuntimeError("The one-time storage correction changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--indexed", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if list(Path("/dev").glob("davinci*")):
        raise RuntimeError("The CPU gate must not map a davinci device")
    decoder_count = verify_inputs(args.decoder)
    runtime_count = verify_inputs(args.runtime)
    unchanged_storage_contract(args.decoder)
    sys.path[:0] = [str(args.decoder),
                   str(args.runtime / "tests/glm52/whole_layer")]
    import torch
    import torch_npu
    import local_chain
    from local_chain_host import Library

    if (not torch.__version__.startswith("2.12.0")
            or torch_npu.__version__ != "2.12.0"
            or torch.npu.is_initialized()):
        raise RuntimeError("Use the actual Torch212 runtime without NPU initialization")
    manifest = json.loads((args.decoder / "manifest.json").read_text())
    for path, key in ((args.indexed, "indexed_library_sha256"),
                      (args.shared, "shared_attention_library_sha256")):
        if digest(path) != manifest[key]:
            raise RuntimeError(f"Use the accepted separate attention library: {path}")
        torch.ops.load_library(str(path))
    ffn = local_chain.load_ffn(args.decoder / "complete_ffn.so")
    chain = Library(args.decoder / "layer_local_chain.so", local_chain.CHAIN_SHA)
    tests = importlib.import_module("test_local_chain")
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests))
    if not result.wasSuccessful():
        raise RuntimeError(log.getvalue())
    schemas = sorted(str(value) for value in torch._C._jit_get_all_schemas()
                     if str(value).startswith(("tilert::indexed_attention",
                                               "tilert::shared_full_attention",
                                               "tilert_ffn_original_shared_gu_static::")))
    if len(schemas) != 6 or ffn.forward_out is None:
        raise RuntimeError("The accepted separate schemas did not co-load")
    command = [sys.executable, str(args.decoder / "device_local_chain.py"),
               "--runtime", str(args.runtime), "--output", "/tmp/not-started",
               "--extension-sha256", "unused",
               "--adapter", str(args.runtime / "tilert/models/glm_5_2_ascend/attention_native.py"),
               "--indexed-library", str(args.indexed),
               "--shared-library", str(args.shared),
               "--sfa-library", str(args.runtime / "native/sfa/vendor_sfa_adapter.so")]
    controls = []
    for extra, code, text in (
            (["--help"], 0, "--chain-library"),
            (["--graph-residual-modes", "none"], 2, "requires a provided residual"),
            (["--forward-widths", "3"], 2, "Use widths 1/2/12/32"),
            (["--forward-replays", "0"], 2, "Use widths 1/2/12/32")):
        run = subprocess.run(command + extra, capture_output=True, text=True,
                             check=False)
        if run.returncode != code or text not in run.stdout + run.stderr:
            raise RuntimeError(f"Unexpected CLI result: {run}")
        controls.append({"arguments": extra, "exit_code": run.returncode,
                         "stdout": run.stdout, "stderr": run.stderr})
    if torch.npu.is_initialized():
        raise RuntimeError("The CPU fixture check initialized an NPU")
    report = {
        "scope": "Actual Torch212 import and four-library co-load; CPU address-group, successor, metadata, poison, view and CLI contracts",
        "python": sys.version, "torch": torch.__version__,
        "torch_npu": torch_npu.__version__, "torch_path": torch.__file__,
        "torch_npu_path": torch_npu.__file__,
        "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
        "npu_initialized_before": False, "npu_initialized_after": False,
        "mapped_davinci_devices": [], "device_execution": False,
        "decoder_files_verified": decoder_count,
        "runtime_files_verified": runtime_count,
        "decoder_manifest_sha256": digest(args.decoder / "manifest.json"),
        "storage_correction_functions_ast_unchanged_except_scope_literal": True,
        "chain_record_bytes": [chain.config_bytes, chain.address_bytes],
        "contracts_run": result.testsRun, "contract_log": log.getvalue(),
        "schemas": schemas, "cli_controls": controls,
        "libraries": {str(path): digest(path) for path in (
            args.indexed, args.shared, args.decoder / "complete_ffn.so",
            args.decoder / "layer_local_chain.so")},
        "script_sha256": digest(Path(__file__)), "pass": True,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: report[name] for name in
                      ("contracts_run", "decoder_files_verified",
                       "npu_initialized_after", "pass")}), flush=True)


if __name__ == "__main__":
    main()
