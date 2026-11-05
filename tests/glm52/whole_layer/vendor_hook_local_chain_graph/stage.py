"""Freeze the official ACL graph fixture above the accepted eager local chain."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil


PARENT_SHA = "2ebf8e0b80a708de354376ab7f182f5853986e879c767773b4016b9ebc901543"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one frozen source anchor: {old!r}")
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deploy-name", default="vendor-hook-local-graph-v1")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", args.deploy_name):
        raise ValueError("Use a simple deployment directory name")
    if digest(args.parent / "manifest.json") != PARENT_SHA:
        raise RuntimeError("Use the frozen local-chain eager v3 fixture")
    hashes = {}
    for line in (args.parent / "frozen-inputs.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(args.parent / name) != expected:
            raise RuntimeError(f"The frozen parent changed: {name}")
        hashes[name] = expected
    args.output.mkdir(parents=True, exist_ok=False)
    for name in hashes:
        if name == "manifest.json":
            continue
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.parent / name, target)
    for source, target in (("graph.py", "local_chain_graph.py"),
                           ("test_graph.py", "test_local_graph.py"),
                           ("cpu_check.py", "local_graph_cpu_check.py"),
                           ("stage.py", "stage_local_graph.py")):
        shutil.copy2(Path(__file__).with_name(source), args.output / target)
    storage = once((args.parent / "decoder_storage_local_chain.py").read_text(),
                   "from local_chain import run_local_chain as original_forward",
                   "from local_chain_graph import run_local_graph as original_forward")
    storage = once(storage, "before the local component chain.",
                   "before the local component graph.")
    storage = once(storage, "before the eager local component chain",
                   "before the official ACL local component graph")
    (args.output / "decoder_storage_local_graph.py").write_text(storage)
    original = (args.parent / "device_local_chain.py").read_text()
    driver = once(original, "from decoder_storage_local_chain import run_decoder_forward",
                  "from decoder_storage_local_graph import run_decoder_forward")
    driver = once(driver, 'stage("actual_local_component_chain",',
                  'stage("actual_local_component_graph",')
    driver = once(driver, 'args.output / "local-chain.json"',
                  'args.output / "local-chain-graph.json"')
    driver = once(driver, '"local_component_chain_executed": True,',
                  '"local_component_graph_executed": True,')
    driver = once(driver,
                  '"scope": "Actual weights and metadata; separate native components versus one local layer launch; both TP reductions identity",',
                  '"scope": "Official ACL graph of two local launches against eager separate native components; both TP reductions identity",')
    driver = once(driver,
                  '"scope": "Actual Decoder weights; separate components versus a local layer launch; no TP reduction",',
                  '"scope": "Actual Decoder weights; official ACL local-component graph versus eager separate components; no TP reduction",')
    (args.output / "device_local_graph.py").write_text(driver)
    (args.output / "local-graph-driver.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), driver.splitlines(keepends=True),
        fromfile="parent/device_local_chain.py", tofile="candidate/device_local_graph.py")))
    launcher = (args.parent / "device_local_chain_run.sh").read_text().replace(
        "vendor-hook-local-chain-v3", args.deploy_name).replace(
        "device_local_chain.py", "device_local_graph.py")
    (args.output / "device_local_graph_run.sh").write_text(launcher)
    files = {str(path.relative_to(args.output)): digest(path)
             for path in args.output.rglob("*") if path.is_file()}
    parent = json.loads((args.parent / "manifest.json").read_text())
    report = {name: value for name, value in parent.items()
              if name.endswith("_sha256") and name != "parent_manifest_sha256"}
    report.update({
        "scope": "Official ACLGraphWrapper local6-to-7 chain; eager separate parents as oracle",
        "parent_manifest_sha256": PARENT_SHA,
        "deployment_directory": args.deploy_name,
        "eager_component_source_sha256": files["local_chain.py"],
        "eager_views_sha256": files["local_chain_views.py"],
        "original_decoder_forward_called": False,
        "tp_collectives_executed": 0,
        "input_residual": "provided BF16 only",
        "topk": "Explicit layer6 persistent-owner copy inside graph before shared layer7",
        "immediate_consumers": "Two preallocated strong outputs copied after the final local launch",
        "device_execution": False, "timing": False,
        "files_sha256": files,
    })
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    files["manifest.json"] = digest(manifest)
    (args.output / "frozen-inputs.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in sorted(files.items())))
    print(json.dumps({"directory": str(args.output), "files": len(files),
                      "manifest_sha256": digest(manifest)}, indent=2))


if __name__ == "__main__":
    main()
