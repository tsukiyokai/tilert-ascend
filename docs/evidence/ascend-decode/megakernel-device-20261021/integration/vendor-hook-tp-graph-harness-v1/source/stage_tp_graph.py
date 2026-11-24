"""Freeze an official ACL graph variant without changing the eager TP parent."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil


PARENT_SHA = "4d608c5a7a382a37002035026871b4cb3a64c20c2947242f15a38692d635507f"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Expected one source anchor: {old!r}")
    return source.replace(old, new)


def graph_source(source):
    value = once(source, "from tp_parent import separate", "from tp_parent import separate\n"
                 "from tp_graph_runtime import CapturedChain, prepare_graph_registry, reserve_capture_stream")
    value = once(value, "def run_tp_chain(", "def run_tp_graph(")
    value = once(value, "source_identity(run_tp_chain)", "source_identity(run_tp_graph)")
    value = once(value,
                 '"scope": "Two TP16-shaped local shards: separate parents plus HCCL2 versus one launch per layer with SHMEM2",',
                 '"scope": "Official ACL graph of two TP16-shaped local shards with SHMEM2; eager separate parents plus HCCL2 oracle",')
    value = once(value, '"blocks": BLOCKS, "eager": True, "graph_capture": False, "timing": False,',
                 '"blocks": BLOCKS, "eager": False, "graph_capture": True, "timing": False,')
    value = once(value, '"cases": [], "pass": False,', '"captures": [], "cases": [], "pass": False,')
    value = once(value, 'args.output / "tp-chain.json"', 'args.output / "tp-chain-graph.json"')
    value = once(value, "    window = None\n    initialized = False",
                 "    window = None\n    initialized = False\n    current_graph = None\n    capture_owner = None")
    value = once(value, "        stream = torch.npu.Stream(device=topk.device)",
                 "        prepare_graph_registry(args.forward_widths)\n"
                 "        stream, capture_owner = reserve_capture_stream(torch)")
    value = once(value, '"epoch_contract": "Four device increments per two-layer candidate case; never reset by host"}',
                 '"epoch_contract": "Four device increments per explicit two-layer replay; never reset by host",\n'
                 '                                "capture_manager_retained": True}')
    value = once(value, "                for iteration in range(args.forward_replays):",
                 "                current_graph = CapturedChain(\n"
                 "                    torch, config, layers, inputs, cases, window, stream,\n"
                 "                    execute, consumers, values)\n"
                 "                for iteration in range(args.forward_replays):")
    value = once(value,
                 "                    reset(iteration, 2)\n                    torch.npu.synchronize()\n"
                 "                    torch.distributed.barrier(group=args.oracle_cpu)\n"
                 "                    if rank == iteration % 2:\n                        time.sleep(0.01)\n"
                 "                    execute(True, [])",
                 "                    reset(iteration, 2)\n"
                 "                    if iteration == 0:\n"
                 "                        report[\"captures\"].append(current_graph.capture())\n"
                 "                        persist()\n"
                 "                    torch.npu.synchronize()\n"
                 "                    torch.distributed.barrier(group=args.oracle_cpu)\n"
                 "                    if rank == iteration % 2:\n                        time.sleep(0.01)\n"
                 "                    current_graph.replay()")
    value = once(value, '"rank_witnesses": peers, "inputs": inputs.witness(),',
                 '"rank_witnesses": peers, "inputs": inputs.witness(),\n'
                 '                              "explicit_replay": current_graph.replays,\n'
                 '                              "python_calls_during_replay": 0,\n'
                 '                              "capture_stream": current_graph.capture_stream.npu_stream,')
    value = once(value, 'args.output / f"tp-chain-m{rows}-r{iteration}.pt"',
                 'args.output / f"tp-chain-graph-m{rows}-r{iteration}.pt"')
    value = once(value, 'raise RuntimeError("The TP layer differs from separate parents and HCCL2")',
                 'raise RuntimeError("The TP graph differs from eager parents and HCCL2")')
    value = once(value, "                del cases, first, second, values, consumers, topk_copies, inputs",
                 "                current_graph.close()\n                current_graph = None\n"
                 "                del cases, first, second, values, consumers, topk_copies, inputs")
    value = once(value, "    finally:\n        if initialized:",
                 "    finally:\n        if current_graph is not None:\n            current_graph.close()\n"
                 "        if initialized:")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deploy-name", default="vendor-hook-tp-graph-v1")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", args.deploy_name):
        raise ValueError("Use a simple deployment name")
    if digest(args.parent / "manifest.json") != PARENT_SHA:
        raise RuntimeError("The accepted eager TP parent changed")
    files = {}
    for line in (args.parent / "frozen-inputs.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(args.parent / name) != expected:
            raise RuntimeError(f"A parent input changed: {name}")
        files[name] = expected
    args.output.mkdir(parents=True, exist_ok=False)
    for name in files:
        if name == "manifest.json":
            continue
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.parent / name, target)
    directory = Path(__file__).resolve().parent
    for source, target in (("runtime.py", "tp_graph_runtime.py"), ("stage.py", "stage_tp_graph.py"),
                           ("test_graph.py", "test_tp_graph.py"), ("cpu_check.py", "tp_graph_cpu_check.py")):
        shutil.copy2(directory / source, args.output / target)
    source = (args.parent / "tp_chain.py").read_text()
    generated = graph_source(source)
    (args.output / "tp_graph.py").write_text(generated)
    (args.output / "graph.patch").write_text("".join(difflib.unified_diff(
        source.splitlines(keepends=True), generated.splitlines(keepends=True),
        fromfile="tp_chain.py", tofile="tp_graph.py")))
    storage = once((args.parent / "decoder_storage_tp_chain.py").read_text(),
                   "from tp_chain import run_tp_chain as original_forward",
                   "from tp_graph import run_tp_graph as original_forward")
    storage = storage.replace("before the two-rank eager component chain", "before the two-rank component graph")
    (args.output / "decoder_storage_tp_graph.py").write_text(storage)
    constructor = once((args.parent / "tp_constructor.py").read_text(),
                       "from decoder_storage_tp_chain import run_decoder_forward",
                       "from decoder_storage_tp_graph import run_decoder_forward")
    constructor = once(constructor, 'stage("actual_two_rank_component_chain",',
                       'stage("actual_two_rank_component_graph",')
    constructor = once(constructor, 'args.output / "tp-chain.json"', 'args.output / "tp-chain-graph.json"')
    constructor = once(constructor, '"two_rank_component_chain_executed": True,',
                       '"two_rank_component_graph_executed": True,')
    (args.output / "tp_graph_constructor.py").write_text(constructor)
    device = once((args.parent / "tp_device.py").read_text(),
                  "from tp_constructor import run_device_rank", "from tp_graph_constructor import run_device_rank")
    (args.output / "tp_graph_device.py").write_text(device)
    launcher = (args.parent / "device_tp_run.sh").read_text().replace(
        "vendor-hook-tp-chain-v2", args.deploy_name)
    launcher = once(launcher, "python3 /decoder/tp_device.py", "python3 /decoder/tp_graph_device.py")
    (args.output / "device_tp_graph_run.sh").write_text(launcher)
    parent = json.loads((args.parent / "manifest.json").read_text())
    files = {str(path.relative_to(args.output)): digest(path)
             for path in args.output.rglob("*") if path.is_file()}
    report = {key: value for key, value in parent.items()
              if key.endswith("_sha256") and key != "parent_manifest_sha256"}
    report.update(parent_manifest_sha256=PARENT_SHA, deployment_directory=args.deploy_name,
                  scope="Official ACLGraphWrapper with SHMEM2; eager HCCL2 parent oracle",
                  device_execution=False, files_sha256=files)
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    files["manifest.json"] = digest(args.output / "manifest.json")
    (args.output / "frozen-inputs.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in sorted(files.items())))
    print(json.dumps({"directory": str(args.output), "files": len(files),
                      "manifest_sha256": files["manifest.json"]}, indent=2))


if __name__ == "__main__":
    main()
