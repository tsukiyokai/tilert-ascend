"""Check the local ACL graph fixture in the real Torch212 CPU environment."""

import argparse
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
    count = 0
    for line in (root / "frozen-inputs.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(root / name) != expected:
            raise RuntimeError(f"A frozen input changed: {name}")
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--indexed", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if list(Path("/dev").glob("davinci*")):
        raise RuntimeError("The graph CPU gate must not map a davinci device")
    counts = {"decoder": verify(args.decoder), "runtime": verify(args.runtime)}
    manifest = json.loads((args.decoder / "manifest.json").read_text())
    for name, key in (("local_chain.py", "eager_component_source_sha256"),
                      ("local_chain_views.py", "eager_views_sha256")):
        if digest(args.decoder / name) != manifest[key]:
            raise RuntimeError("The accepted eager fixture changed")
    sys.path[:0] = [str(args.decoder), str(args.runtime / "tests/glm52/whole_layer")]
    import torch
    import torch_npu
    import local_chain
    import local_chain_graph
    from local_chain_host import Library
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    from vllm_ascend.compilation.updatable_graph import UpdatableGraph

    if (not torch.__version__.startswith("2.12.0") or torch_npu.__version__ != "2.12.0"
            or torch.npu.is_initialized()):
        raise RuntimeError("Use the original Torch212 ABI without NPU initialization")
    for path, key in ((args.indexed, "indexed_library_sha256"),
                      (args.shared, "shared_attention_library_sha256")):
        if digest(path) != manifest[key]:
            raise RuntimeError("The separate attention parent changed")
        torch.ops.load_library(str(path))
    local_chain.load_ffn(args.decoder / "complete_ffn.so")
    chain = Library(args.decoder / "layer_local_chain.so", local_chain.CHAIN_SHA)
    log = io.StringIO()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(
        importlib.import_module(name)) for name in ("test_local_chain", "test_local_graph", "test_topk_contract"))
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError(log.getvalue())
    command = [sys.executable, str(args.decoder / "device_local_graph.py"),
               "--runtime", str(args.runtime), "--output", "/tmp/not-started",
               "--extension-sha256", "unused",
               "--adapter", str(args.runtime / "tilert/models/glm_5_2_ascend/attention_native.py"),
               "--indexed-library", str(args.indexed), "--shared-library", str(args.shared),
               "--sfa-library", str(args.runtime / "native/sfa/vendor_sfa_adapter.so")]
    controls = []
    for extra, code, message in (
            (["--help"], 0, "--chain-library"),
            (["--graph-residual-modes", "none"], 2, "requires a provided residual"),
            (["--forward-widths", "3"], 2, "Use widths 1/2/12/32"),
            (["--forward-replays", "0"], 2, "Use widths 1/2/12/32")):
        run = subprocess.run(command + extra, text=True, capture_output=True, check=False)
        if run.returncode != code or message not in run.stdout + run.stderr:
            raise RuntimeError(f"Unexpected graph CLI result: {run}")
        controls.append({"arguments": extra, "exit_code": run.returncode,
                         "stdout": run.stdout, "stderr": run.stderr})
    if torch.npu.is_initialized():
        raise RuntimeError("The CPU graph check initialized an NPU")
    report = {
        "scope": "Actual Torch212 imports and four-library co-load; CPU lifecycle/metadata/poison/CLI contracts; no graph executed",
        "python": sys.version, "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "torch_path": torch.__file__, "torch_npu_path": torch_npu.__file__,
        "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
        "npu_initialized_before": False, "npu_initialized_after": False,
        "mapped_davinci_devices": [], "device_execution": False,
        "files_verified": counts, "manifest_sha256": digest(args.decoder / "manifest.json"),
        "actual_graph_classes": {inspect.getfile(cls): digest(inspect.getfile(cls))
                                 for cls in (ACLGraphWrapper, UpdatableGraph)},
        "chain_record_bytes": [chain.config_bytes, chain.address_bytes],
        "contracts_run": result.testsRun, "contract_log": log.getvalue(),
        "cli_controls": controls,
        "libraries": {str(path): digest(path) for path in (
            args.indexed, args.shared, args.decoder / "complete_ffn.so",
            args.decoder / "layer_local_chain.so")},
        "script_sha256": digest(Path(__file__)), "pass": True,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("contracts_run", "npu_initialized_after", "pass")}), flush=True)


if __name__ == "__main__":
    main()
