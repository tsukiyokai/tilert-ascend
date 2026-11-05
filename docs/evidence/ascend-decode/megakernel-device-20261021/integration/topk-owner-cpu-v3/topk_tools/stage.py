"""Freeze the accepted two-layer fixture with a persistent top-k producer."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil

from prepare import replace_once


FIXTURE_MANIFEST = "2ebf8e0b80a708de354376ab7f182f5853986e879c767773b4016b9ebc901543"
FIXTURE_SCRIPT = "92e8671cdea507fd1096741d18b6a2191ddb94c6f4403461f093c6ada3822fb1"
PARENT_LIBRARY = "609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform_fixture(source, library_sha):
    source = replace_once(source, PARENT_LIBRARY, library_sha)
    source = replace_once(
        source, '            self.selected if self.mode == "shared" else None, geometry,\n',
        '            topk[:self.rows].view(self.rows, 1, 2048), geometry,\n')
    source = replace_once(
        source, '    def separate(self):\n',
        '        self.persistent_topk = topk[:self.rows].view(self.rows, 1, 2048)\n'
        '        if self.mode != "shared":\n'
        '            self.values["attention.topk"] = self.persistent_topk\n\n'
        '    def separate(self):\n')
    source = replace_once(
        source, '                "attention_mode": self.mode, "sync": self.sync}\n',
        '                "persistent_topk": signature(self.persistent_topk),\n'
        '                "internal_topk": (None if self.mode == "shared"\n'
        '                                  else signature(self.selected)),\n'
        '                "attention_mode": self.mode, "sync": self.sync}\n')
    source = replace_once(
        source, '"topk_contract": "Layer6 selected indices copy into the actual persistent owner before shared layer7",',
        '"topk_contract": "The indexed candidate writes directly into the persistent owner; only the separate parent copies its selected indices",')
    source = replace_once(
        source, '        "timing": False,\n',
        '        "timing": False,\n'
        '        "topk_write_guards": "Candidate internal slice stays at 2; inactive persistent rows stay at -1",\n')
    source = replace_once(
        source,
        '"Floating NaN in both arms; non-address integers 1/2; bounded indices/counts zero"',
        '"Floating NaN; non-address integers 1/2; bounded indices/counts zero except active_list fills 1 with E=0"')
    source = replace_once(
        source, '                    topk_copies = {}\n',
        '                    if arm == "combined":\n'
        '                        first.selected.fill_(2)\n'
        '                    topk_copy_calls = 0\n'
        '                    topk_copies = {}\n')
    source = replace_once(
        source, '                        case.norm[5].fill_(float("nan"))\n',
        '                        case.norm[5].fill_(float("nan"))\n'
        '                        case.body[20].fill_(1)\n'
        '                        case.body[20][0].zero_()\n')
    source = replace_once(
        source, '                        if number == 6:\n',
        '                        if number == 6 and arm == "separate":\n')
    source = replace_once(
        source, '                            publish_topk(case.selected, topk, rows)\n',
        '                            publish_topk(case.selected, topk, rows)\n'
        '                            topk_copy_calls += 1\n')
    source = replace_once(
        source, '                    values = {f"layer{number}.{name}": snapshot(torch, value)\n',
        '                    inactive_unchanged = bool((topk[rows:] == -1).all())\n'
        '                    if not inactive_unchanged:\n'
        '                        raise RuntimeError("The selector overwrote inactive persistent rows")\n'
        '                    if arm == "combined":\n'
        '                        internal_unchanged = bool((first.selected == 2).all())\n'
        '                        if not internal_unchanged:\n'
        '                            raise RuntimeError("The candidate still writes the internal top-k slice")\n'
        '                    values = {f"layer{number}.{name}": snapshot(torch, value)\n')
    source = replace_once(
        source, '                          "inputs": inputs.witness(), "candidate_diagnostics": diagnostics,\n',
        '                          "inputs": inputs.witness(), "candidate_diagnostics": diagnostics,\n'
        '                          "unused_internal_topk_unchanged": internal_unchanged,\n'
        '                          "inactive_persistent_rows_unchanged": inactive_unchanged,\n'
        '                          "candidate_topk_copy_calls": topk_copy_calls,\n')
    compile(source, "local_chain.py", "exec")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deploy-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", args.deploy_name):
        raise ValueError("Use a simple deployment name")
    if (digest(args.parent / "manifest.json") != FIXTURE_MANIFEST
            or digest(args.parent / "local_chain.py") != FIXTURE_SCRIPT):
        raise ValueError("Use the accepted two-layer eager v3 fixture")
    frozen = {}
    for line in (args.parent / "frozen-inputs.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(args.parent / name) != expected:
            raise ValueError(f"A frozen fixture file changed: {name}")
        frozen[name] = expected
    build_manifest = json.loads((args.build / "manifest.json").read_text())
    build = json.loads((args.build / "results/build.json").read_text())
    if build["parent_manifest_sha256"] != digest(args.build / "manifest.json"):
        raise ValueError("The build does not match its frozen source")
    for name, expected in build_manifest["sources"].items():
        if digest(args.build / name) != expected:
            raise ValueError(f"A frozen candidate file changed: {name}")
    library = args.build / "results/layer_topk_owner.so"
    library_sha = digest(library)
    if library_sha != build["library_sha256"]:
        raise ValueError("The candidate library differs from the CPU build")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in frozen:
        if name == "manifest.json":
            continue
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.parent / name, target)
    shutil.copy2(library, args.output / "layer_local_chain.so")
    shutil.copy2(args.build / "source/host.py", args.output / "local_chain_host.py")
    for name in ("topk_contract.py", "test_topk_contract.py"):
        shutil.copy2(args.build / "source" / name, args.output / name)
    tool_directory = args.output / "topk_tools"
    tool_directory.mkdir()
    for name in ("stage.py", "prepare.py", "topk_contract.py",
                 "test_topk_contract.py", "README.md"):
        shutil.copy2(Path(__file__).with_name(name), tool_directory / name)
    cpu_check = args.output / "local_chain_cpu_check.py"
    cpu_text = replace_once(
        cpu_check.read_text(),
        '    tests = importlib.import_module("test_local_chain")\n',
        '    tests = unittest.TestSuite(\n'
        '        unittest.defaultTestLoader.loadTestsFromModule(\n'
        '            importlib.import_module(name))\n'
        '        for name in ("test_local_chain", "test_topk_contract"))\n')
    cpu_text = replace_once(
        cpu_text, '        unittest.defaultTestLoader.loadTestsFromModule(tests))\n',
        '        tests)\n')
    cpu_check.write_text(cpu_text)
    original = (args.parent / "local_chain.py").read_text()
    candidate = transform_fixture(original, library_sha)
    (args.output / "local_chain.py").write_text(candidate)
    (args.output / "topk-fixture.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), candidate.splitlines(keepends=True),
        fromfile="parent/local_chain.py", tofile="candidate/local_chain.py")))
    launcher = args.output / "device_local_chain_run.sh"
    launcher.write_text(replace_once(launcher.read_text(),
                                   "vendor-hook-local-chain-v3", args.deploy_name))
    report = json.loads((args.parent / "manifest.json").read_text())
    report.update({
        "scope": "Two local layers with direct persistent top-k output versus accepted separate parents and explicit copy",
        "parent_fixture_manifest_sha256": FIXTURE_MANIFEST,
        "candidate_build_manifest_sha256": digest(args.build / "manifest.json"),
        "candidate_build_report_sha256": digest(args.build / "results/build.json"),
        "deployment_directory": args.deploy_name,
        "chain_library_sha256": library_sha,
        "chain_host_sha256": digest(args.output / "local_chain_host.py"),
        "topk_contract": "No candidate copy; inactive owner rows and unused internal slice are checked",
        "files_sha256": {str(path.relative_to(args.output)): digest(path)
                         for path in args.output.rglob("*") if path.is_file()},
    })
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    identities = {**report["files_sha256"], "manifest.json": digest(manifest)}
    (args.output / "frozen-inputs.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in sorted(identities.items())))
    print(json.dumps({"output": str(args.output), "files": len(identities),
                      "manifest_sha256": digest(manifest)}, indent=2))


if __name__ == "__main__":
    main()
