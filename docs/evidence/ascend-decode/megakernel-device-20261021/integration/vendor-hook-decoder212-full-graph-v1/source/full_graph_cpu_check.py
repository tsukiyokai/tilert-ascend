"""Check the complete-FFN Decoder overlay in Torch212 without NPU devices."""

import argparse
import ast
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


def observation_contract(root):
    sources = [ast.parse((root / name).read_text()) for name in
               ("decoder_graph.py", "decoder_full_graph.py")]
    names = ("DecoderInputs", "DecoderObservation", "validate_observation")
    for name in names:
        definitions = [next(node for node in source.body
                            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                            and node.name == name) for source in sources]
        if ast.dump(definitions[0]) != ast.dump(definitions[1]):
            raise RuntimeError(f"The original graph observation changed: {name}")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--indexed", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if list(Path("/dev").glob("davinci*")):
        raise RuntimeError("The CPU gate must not expose a davinci device")
    decoder_hashes = verify_inputs(args.decoder)
    runtime_hashes = verify_inputs(args.runtime)
    manifest = json.loads((args.decoder / "manifest.json").read_text())
    for path, key in ((args.indexed, "indexed_library_sha256"),
                      (args.shared, "shared_attention_library_sha256")):
        if digest(path) != manifest[key]:
            raise RuntimeError(f"Use the graph-validated attention library: {path}")
    observations = observation_contract(args.decoder)
    sys.path[:0] = [str(args.decoder),
                   str(args.runtime / "tests/glm52/whole_layer")]
    import torch
    import torch_npu

    if torch.npu.is_initialized():
        raise RuntimeError("Torch import initialized an NPU")
    tests = [importlib.import_module(name) for name in
             ("test_decoder_graph", "test_ffn_host_adapter",
              "test_ffn_runner_protocol")]
    import ffn_host_adapter
    from vllm.model_executor.layers.fused_moe.runner import moe_runner
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner

    if digest(Path(inspect.getfile(moe_runner))) != ffn_host_adapter.LOOKUP_SOURCE_SHA:
        raise RuntimeError("The original MoE lookup source changed")
    if digest(Path(inspect.getfile(AscendMoERunner))) != ffn_host_adapter.RUNNER_SOURCE_SHA:
        raise RuntimeError("The original Ascend reduction source changed")
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromModule(module) for module in tests))
    if not result.wasSuccessful():
        raise RuntimeError(log.getvalue())
    libraries = [args.indexed, args.shared, args.decoder / "complete_ffn.so"]
    for library in libraries[:2]:
        torch.ops.load_library(str(library))
    ffn_host_adapter.load(libraries[2])
    prefixes = ("tilert::indexed_attention", "tilert::shared_full_attention",
                "tilert_ffn_original_shared_gu_static::")
    schemas = sorted(str(value) for value in torch._C._jit_get_all_schemas()
                     if str(value).startswith(prefixes))
    if len(schemas) != 6:
        raise RuntimeError(f"Unexpected native schemas: {schemas}")
    base = [sys.executable, str(args.decoder / "device_full_graph.py"),
            "--runtime", str(args.runtime), "--output", "/tmp/unused-decoder-gate",
            "--extension-sha256", "unused",
            "--adapter", str(args.runtime / "tilert/models/glm_5_2_ascend/attention_native.py"),
            "--indexed-library", str(args.indexed), "--shared-library", str(args.shared),
            "--sfa-library", str(args.runtime / "native/sfa/vendor_sfa_adapter.so")]
    controls = []
    for extra, code, message in (
            (["--help"], 0, "--ffn-library"),
            (["--forward-widths", "3"], 2, "Use widths 1/2/12/32"),
            (["--forward-replays", "0"], 2, "Use widths 1/2/12/32"),
            (["--graph-residual-modes", "none", "none"], 2,
             "Each initial residual branch needs one distinct capture")):
        completed = subprocess.run(base + extra, text=True,
                                   capture_output=True, check=False)
        if (completed.returncode != code
                or message not in completed.stdout + completed.stderr):
            raise RuntimeError(f"Unexpected CLI result: {completed}")
        controls.append({"arguments": extra, "exit_code": completed.returncode,
                         "stdout": completed.stdout, "stderr": completed.stderr})
    if torch.npu.is_initialized():
        raise RuntimeError("The CPU check initialized an NPU")
    report = {
        "scope": "Actual Torch212 host protocol, CPU ownership/observation contracts, CLI and native schema co-load",
        "runner_fixture_scope": "Actual classes and helper functions; registry entries bypass constructors; collective replaced by a recorded CPU cut",
        "python": sys.version, "torch": torch.__version__,
        "torch_npu": torch_npu.__version__, "torch_path": torch.__file__,
        "torch_npu_path": torch_npu.__file__,
        "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
        "mapped_davinci_devices": [], "device_execution": False,
        "npu_initialized_before": False, "npu_initialized_after": False,
        "script_sha256": digest(Path(__file__)),
        "decoder_manifest_sha256": digest(args.decoder / "manifest.json"),
        "decoder_files_verified": len(decoder_hashes),
        "runtime_files_verified": len(runtime_hashes),
        "unchanged_observation_definitions": observations,
        "contracts_run": result.testsRun, "contract_log": log.getvalue(),
        "use_layername": moe_runner._USE_LAYERNAME,
        "helper_source_sha256": {inspect.getfile(value): digest(Path(inspect.getfile(value)))
                                 for value in (moe_runner, AscendMoERunner, ffn_host_adapter)},
        "libraries": {str(path): digest(path) for path in libraries},
        "schemas": schemas, "cli_controls": controls, "pass": True,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: report[name] for name in
                      ("contracts_run", "use_layername", "decoder_files_verified",
                       "npu_initialized_after", "pass")}), flush=True)


if __name__ == "__main__":
    main()
