"""Use N1024 preparation in the frozen 609 local layer with TopK copy."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil


FIXTURE_SHA = "2ebf8e0b80a708de354376ab7f182f5853986e879c767773b4016b9ebc901543"
N256_MANIFEST = "ee4b590ff1b67e58081e2e9cc6b8cda80a466aa213992e217c25fa3ccec79eee"
N1024_MANIFEST = "f5cae0c1f94708f56e84b018d751c496c2e85ea8fc718494667224b1f3dcc9ab"
CHAIN_MANIFEST = "31fffff64ecda3526c53a031e1f41baf01e56657af184d4704f41ba40dfdd0f2"
N256_LIBRARY = "381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2"
N1024_LIBRARY = "bbd52e425a1e94bb199e464e3bf6001c4420e8238afe993e8741e38af50b77ef"
N256_NAMESPACE = "ffn_original_shared_gu_static"
N1024_NAMESPACE = "ffn_down_n1024"
TILING_SOURCE = "body/source/routed_int4/routed_int4_tiling.cpp"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Expected one frozen source anchor: {before!r}")
    return text.replace(before, after)


def verify_build(directory, expected):
    if digest(directory / "manifest.json") != expected:
        raise ValueError(f"Unexpected frozen build: {directory}")
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, sha in manifest["sources"].items():
        if digest(directory / name) != sha:
            raise ValueError(f"A frozen source changed: {directory / name}")
    return manifest


def verify_device_sources(n256, n1024, chain):
    parent = verify_build(n256, N256_MANIFEST)
    candidate = verify_build(n1024, N1024_MANIFEST)
    verify_build(chain, CHAIN_MANIFEST)
    same, changed, embedded = [], [], []
    for name in sorted(parent["sources"].keys() & candidate["sources"].keys()):
        if Path(name).suffix not in (".cpp", ".h"):
            continue
        before = (n256 / name).read_text()
        after = (n1024 / name).read_text().replace(
            N1024_NAMESPACE, N256_NAMESPACE).replace(
                N1024_NAMESPACE.upper(), N256_NAMESPACE.upper())
        (same if before == after else changed).append(name)
        if not name.startswith("candidate_tools/"):
            if (chain / "ffn" / name).read_bytes() != (n256 / name).read_bytes():
                raise ValueError(f"The embedded FFN differs from its parent: {name}")
            embedded.append(name)
    if changed != [TILING_SOURCE] or len(same) != 49 or len(embedded) != 49:
        raise ValueError("The source comparison exceeds the host-tiling change")
    return {"normalized_unchanged_files": same, "changed_files": changed,
            "embedded_ffn_files_identical": embedded,
            "n256_manifest_sha256": N256_MANIFEST,
            "n1024_manifest_sha256": N1024_MANIFEST,
            "chain_manifest_sha256": CHAIN_MANIFEST}


def fixture_source(source):
    source = once(source, "from decoder_forward import bind_caches, metadata_for_layers\n",
                  "from decoder_forward import bind_caches, metadata_for_layers\n"
                  "from down_tiling import check_down_tiling\n")
    source = once(source, "            self.partial, BLOCKS, 1)\n",
                  "            self.partial, BLOCKS, 1)\n"
                  "        self.down_tiling_before = check_down_tiling(torch, self.body[7])\n")
    source = once(source, "completeFFN381", "completeFFNbbd-N1024")
    source = once(source,
                  "                        case.norm[5].fill_(float(\"nan\"))\n",
                  "                        case.norm[5].fill_(float(\"nan\"))\n"
                  "                        case.body[20].fill_(1)\n"
                  "                        case.body[20][0].zero_()\n")
    source = once(source, '        "prepared": [], "cases": [], "pass": False,\n',
                  '        "prepared": [], "cases": [], "pass": False,\n'
                  '        "down_tiling_checks": [],\n')
    source = once(source,
                  '            report["prepared"].append({"rows": rows, "owners": owners})\n',
                  '            report["prepared"].append({\n'
                  '                "rows": rows, "owners": owners,\n'
                  '                "down_tiling": [case.down_tiling_before for case in cases]})\n')
    source = once(source,
                  '                    values = {f"layer{number}.{name}": snapshot(torch, value)\n',
                  '                    report["down_tiling_checks"].append({\n'
                  '                        "rows": rows, "iteration": iteration, "arm": arm,\n'
                  '                        "layers": [check_down_tiling(torch, case.body[7])\n'
                  '                                   for case in cases]})\n'
                  '                    values = {f"layer{number}.{name}": snapshot(torch, value)\n')
    compile(source, "local_chain.py", "exec")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--n256-build", type=Path, required=True)
    parser.add_argument("--n1024-build", type=Path, required=True)
    parser.add_argument("--chain-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deploy-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", args.deploy_name):
        raise ValueError("Use a simple deployment name")
    if digest(args.parent / "manifest.json") != FIXTURE_SHA:
        raise ValueError("Use the frozen 609 local-chain v3 fixture")
    frozen = {}
    for line in (args.parent / "frozen-inputs.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(args.parent / name) != expected:
            raise ValueError(f"A frozen fixture file changed: {name}")
        frozen[name] = expected
    compatibility = verify_device_sources(
        args.n256_build, args.n1024_build, args.chain_build)
    library = args.n1024_build / "ffn_down_n1024.so"
    if digest(library) != N1024_LIBRARY:
        raise ValueError("Use the accepted N1024 FFN library")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in frozen:
        if name == "manifest.json":
            continue
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.parent / name, target)
    shutil.copy2(library, args.output / "complete_ffn.so")
    for name in ("down_tiling.py", "test_contract.py", "stage.py", "README.md"):
        target = (args.output / name if name in ("down_tiling.py", "test_contract.py")
                  else args.output / "n1024_tools" / name)
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(Path(__file__).with_name(name), target)
    packing = args.output / "ffn_host_adapter.py"
    text = once(packing.read_text(), N256_LIBRARY, N1024_LIBRARY)
    text = once(text, N256_NAMESPACE, N1024_NAMESPACE)
    text = once(text, "Use the original-shared complete FFN library",
                "Use the accepted N1024 complete FFN library")
    packing.write_text(text)
    cpu = args.output / "local_chain_cpu_check.py"
    cpu_source = once(cpu.read_text(), N256_NAMESPACE, N1024_NAMESPACE)
    cpu_source = once(cpu_source,
                     "        unittest.defaultTestLoader.loadTestsFromModule(tests))",
                     "        unittest.TestSuite([\n"
                     "            unittest.defaultTestLoader.loadTestsFromModule(tests),\n"
                     "            unittest.defaultTestLoader.loadTestsFromModule(\n"
                     "                importlib.import_module(\"test_contract\"))]))")
    cpu.write_text(cpu_source)
    original = (args.parent / "local_chain.py").read_text()
    candidate = fixture_source(original)
    (args.output / "local_chain.py").write_text(candidate)
    (args.output / "n1024-fixture.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), candidate.splitlines(keepends=True),
        fromfile="parent/local_chain.py", tofile="candidate/local_chain.py")))
    shutil.copy2(args.parent / "local_chain.py", args.output / "parent-local_chain.py")
    (args.output / "n1024-source-compatibility.json").write_text(
        json.dumps(compatibility, indent=2) + "\n")
    launcher = args.output / "device_local_chain_run.sh"
    launcher.write_text(once(launcher.read_text(), "vendor-hook-local-chain-v3",
                             args.deploy_name))
    report = json.loads((args.parent / "manifest.json").read_text())
    report.update({
        "scope": "609 local layer and explicit TopK copy with N1024 FFN preparation",
        "parent_fixture_manifest_sha256": FIXTURE_SHA,
        "complete_ffn_library_sha256": N1024_LIBRARY,
        "deployment_directory": args.deploy_name,
        "down_tiling_checks": "Full 200 bytes after preparation and each arm",
        "topk": "609 internal selector output and explicit retained-owner copy remain",
        "active_poison": "Complete table fill1 then E=0 before each arm",
        "n1024_source_compatibility": compatibility,
        "device_execution": False,
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
