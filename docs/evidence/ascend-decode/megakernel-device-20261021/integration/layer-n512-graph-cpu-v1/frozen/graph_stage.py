"""Capture the N512 persistent-top-k layer through the accepted ACL wrapper."""

import argparse
import difflib
import json
from pathlib import Path
import re
import shutil

from stage import digest, once


EAGER_SHA = "cedb2922e5bbe9d77f09323918523d9212c588affccd44fc8b82c6bb0bb1c437"
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
    source = once(source, "        self.calls = 0\n",
                  "        self.calls = 0\n        self.topk_copy_calls = 0\n")
    source = once(source, "            if number == 6:\n",
                  "            if number == 6 and not combined:\n")
    source = once(source,
                  "                publish_topk(case.selected, self.topk, self.inputs.rows)\n",
                  "                publish_topk(case.selected, self.topk, self.inputs.rows)\n"
                  "                self.topk_copy_calls += 1\n")
    source = once(source, "        poison(self.topk_copies, integer)\n",
                  "        if integer == 2:\n"
                  "            self.cases[0].selected.fill_(2)\n"
                  "        poison(self.topk_copies, integer)\n")
    source = once(source,
                  "two local layer launches plus persistent top-k copy and immediate consumers",
                  "two N512 local layer launches with direct persistent top-k and immediate consumers")
    source = once(source,
                  '        "timing": False, "captures": [], "cases": [], "pass": False,\n',
                  '        "timing": False, "captures": [], "cases": [], "pass": False,\n'
                  '        "down_tiling_checks": [],\n')
    source = once(source, "                reset(0, observation, 2)\n",
                  "                reset(0, observation, 2)\n"
                  "                copies_before_capture = observation.topk_copy_calls\n")
    source = once(source,
                  '                    raise RuntimeError("The official wrapper did not capture the two local launches once")\n',
                  '                    raise RuntimeError("The official wrapper did not capture the two local launches once")\n'
                  '                if observation.topk_copy_calls != copies_before_capture:\n'
                  '                    raise RuntimeError("The candidate graph copied its top-k output")\n')
    source = once(source, '                    "workspace_preallocated": True,\n',
                  '                    "workspace_preallocated": True,\n'
                  '                    "candidate_topk_copy_calls": 0,\n'
                  '                    "down_tiling": [case.down_tiling_before\n'
                  '                                    for case in observation.cases],\n')
    source = once(source, "                    expected = collect(torch, observation)\n",
                  "                    expected = collect(torch, observation)\n"
                  '                    report["down_tiling_checks"].append({\n'
                  '                        "rows": rows, "iteration": iteration, "arm": "separate",\n'
                  '                        "layers": [check_down_tiling(torch, case.body[7])\n'
                  '                                   for case in observation.cases]})\n')
    source = once(source, "                    actual = collect(torch, observation)\n",
                  "                    actual = collect(torch, observation)\n"
                  "                    internal_unchanged = bool((first.selected == 2).all())\n"
                  "                    inactive_unchanged = bool((topk[rows:] == -1).all())\n"
                  "                    if not internal_unchanged or not inactive_unchanged:\n"
                  '                        raise RuntimeError("Replay wrote outside the active persistent top-k owner")\n'
                  '                    report["down_tiling_checks"].append({\n'
                  '                        "rows": rows, "iteration": iteration, "arm": "combined",\n'
                  '                        "layers": [check_down_tiling(torch, case.body[7])\n'
                  '                                   for case in observation.cases]})\n')
    source = once(source, '                        "python_calls_during_replay": 0,\n',
                  '                        "python_calls_during_replay": 0,\n'
                  '                        "candidate_topk_copy_calls": 0,\n'
                  '                        "unused_internal_topk_unchanged": internal_unchanged,\n'
                  '                        "inactive_persistent_rows_unchanged": inactive_unchanged,\n')
    compile(source, "local_chain_graph.py", "exec")
    return source


DIRECT_OWNER_TEST = '''    def test_combined_reuses_direct_owner_without_publishing_copy(self):
        from unittest.mock import patch

        observation, first, second, topk = self.fixture()
        first.selected.fill_(2)

        def direct():
            first.bridge[10].copy_(first.norm[0] + 1)
            first.bridge[1].copy_(first.norm[1] + 2)
            topk[:2].fill_(17)

        first.combined = direct
        with patch.object(local_chain_graph, "publish_topk",
                          side_effect=AssertionError("Unexpected candidate copy")):
            observation.execute(True)
        self.assertTrue(bool((second.selected == 17).all()))
        self.assertTrue(bool((first.selected == 2).all()))
        self.assertTrue(bool((topk[2:] == -1).all()))
        self.assertEqual(observation.topk_copy_calls, 0)

'''


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
    (args.output / "n512-graph.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), candidate.splitlines(keepends=True),
        fromfile="parent/local_chain_graph.py", tofile="candidate/local_chain_graph.py")))
    tests = args.output / "test_local_graph.py"
    anchor = "    def test_preallocated_consumers_survive_source_poison(self):\n"
    tests.write_text(once(tests.read_text(), anchor, DIRECT_OWNER_TEST + anchor))
    cpu = args.output / "local_graph_cpu_check.py"
    cpu.write_text(once(cpu.read_text(), '("test_local_chain", "test_local_graph")',
                        '("test_local_chain", "test_local_graph", "test_topk_contract")'))
    launcher = args.output / "device_local_graph_run.sh"
    launcher.write_text(once(launcher.read_text(), "vendor-hook-local-graph-v3",
                             args.deploy_name))
    shutil.copy2(Path(__file__), args.output / "n512_tools/graph_stage.py")
    parent = json.loads((args.parent / "manifest.json").read_text())
    files = {str(path.relative_to(args.output)): digest(path)
             for path in args.output.rglob("*") if path.is_file()}
    parent.update({
        "scope": "Official ACL graph of N512 local layers with direct persistent top-k",
        "parent_fixture_manifest_sha256": EAGER_SHA,
        "graph_fixture_manifest_sha256": GRAPH_SHA,
        "deployment_directory": args.deploy_name,
        "eager_component_source_sha256": files["local_chain.py"],
        "eager_views_sha256": files["local_chain_views.py"],
        "topk": "Candidate writes the retained owner directly; eager reference copies",
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
