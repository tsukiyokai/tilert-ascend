"""Capture the 609/N1024 local layer with its original persistent TopK copy."""

import argparse
import difflib
import json
from pathlib import Path
import re
import shutil

from stage import digest, once


EAGER_SHA = "228ca66512b1ad477bf2e018260d2c37d3fad6fa46958bebceb3f0a2c86ae0ea"
GRAPH_SHA = "48f8d549dbb2748e9cbaf0777346cf8091147eb56f5f24863207ced51d0b846c"
GRAPH_FILES = (
    "local_chain_graph.py", "local_graph_cpu_check.py", "test_local_graph.py",
    "decoder_storage_local_graph.py", "device_local_graph.py",
    "device_local_graph_run.sh",
)


def verify(directory, expected):
    if digest(directory / "manifest.json") != expected:
        raise ValueError(f"Unexpected frozen fixture: {directory}")
    files = {}
    for line in (directory / "frozen-inputs.sha256").read_text().splitlines():
        sha, name = line.split("  ", 1)
        if digest(directory / name) != sha:
            raise ValueError(f"A frozen input changed: {name}")
        files[name] = sha
    return files


def graph_source(source):
    source = once(source, "from local_chain_views import poison, strict_checks\n",
                  "from local_chain_views import poison, strict_checks\n"
                  "from down_tiling import check_down_tiling\n")
    source = once(source,
                  '        "timing": False, "captures": [], "cases": [], "pass": False,\n',
                  '        "timing": False, "captures": [], "cases": [], "pass": False,\n'
                  '        "down_tiling_checks": [],\n')
    source = once(source, '                    "workspace_preallocated": True,\n',
                  '                    "workspace_preallocated": True,\n'
                  '                    "down_tiling": [case.down_tiling_before\n'
                  '                                    for case in observation.cases],\n')
    for arm, anchor in (("separate", "expected = collect(torch, observation)"),
                        ("combined", "actual = collect(torch, observation)")):
        source = once(source, "                    " + anchor + "\n",
                      "                    " + anchor + "\n"
                      '                    report["down_tiling_checks"].append({\n'
                      f'                        "rows": rows, "iteration": iteration, "arm": "{arm}",\n'
                      '                        "layers": [check_down_tiling(torch, case.body[7])\n'
                      '                                   for case in observation.cases]})\n')
    compile(source, "local_chain_graph.py", "exec")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deploy-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", args.deploy_name):
        raise ValueError("Use a simple deployment name")
    eager = verify(args.parent, EAGER_SHA)
    verify(args.graph, GRAPH_SHA)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in eager:
        if name == "manifest.json":
            continue
        destination = args.output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.parent / name, destination)
    for name in GRAPH_FILES:
        shutil.copy2(args.graph / name, args.output / name)
    original = (args.graph / "local_chain_graph.py").read_text()
    candidate = graph_source(original)
    (args.output / "local_chain_graph.py").write_text(candidate)
    (args.output / "n1024-graph.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), candidate.splitlines(keepends=True),
        fromfile="parent/local_chain_graph.py", tofile="candidate/local_chain_graph.py")))
    shutil.copy2(args.graph / "local_chain_graph.py",
                 args.output / "parent-local_chain_graph.py")
    shutil.copy2(Path(__file__).with_name("test_graph_contract.py"),
                 args.output / "test_graph_contract.py")
    cpu = args.output / "local_graph_cpu_check.py"
    cpu.write_text(once(cpu.read_text(), '("test_local_chain", "test_local_graph")',
                        '("test_local_chain", "test_local_graph",\n'
                        '                                                   "test_contract", "test_graph_contract")'))
    launcher = args.output / "device_local_graph_run.sh"
    launcher.write_text(once(launcher.read_text(), "vendor-hook-local-graph-v3",
                             args.deploy_name))
    shutil.copy2(Path(__file__), args.output / "n1024_tools/graph_stage.py")
    parent = json.loads((args.parent / "manifest.json").read_text())
    files = {str(path.relative_to(args.output)): digest(path)
             for path in args.output.rglob("*") if path.is_file()}
    parent.update({
        "scope": "Official ACL graph of 609 local layers with N1024 FFN preparation",
        "parent_fixture_manifest_sha256": EAGER_SHA,
        "graph_fixture_manifest_sha256": GRAPH_SHA,
        "deployment_directory": args.deploy_name,
        "eager_component_source_sha256": files["local_chain.py"],
        "eager_views_sha256": files["local_chain_views.py"],
        "topk": "Both arms retain the explicit layer6-to-persistent-owner copy",
        "graph_execution": False, "timing": False, "files_sha256": files,
    })
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(parent, indent=2) + "\n")
    files["manifest.json"] = digest(manifest)
    (args.output / "frozen-inputs.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in sorted(files.items())))
    print(json.dumps({"directory": str(args.output), "files": len(files),
                      "manifest_sha256": digest(manifest)}, indent=2))


if __name__ == "__main__":
    main()
