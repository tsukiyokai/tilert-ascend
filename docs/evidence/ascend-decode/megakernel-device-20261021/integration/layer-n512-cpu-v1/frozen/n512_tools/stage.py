"""Use accepted N512 preparation in the persistent-top-k local layer fixture."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil


FIXTURE_SHA = "23ad17150e52b2e3644305d547dcbcc4a618e202a1180c724fd676f03279aae9"
N256_MANIFEST = "ee4b590ff1b67e58081e2e9cc6b8cda80a466aa213992e217c25fa3ccec79eee"
N512_MANIFEST = "6a8780321d41b5d834e1d268ff08c30b675bf39558367a10134c53e28bebf613"
CHAIN_MANIFEST = "f5ca5f534022fd2d9c9d108a3dc74755add5b1514b5c3df95f3081dc68578c9c"
N256_LIBRARY = "381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2"
N512_LIBRARY = "c90b3666a1697070eda19d5b3e133e9ed58f36fe429abefceae6ffe102a6a032"
N256_NAMESPACE = "ffn_original_shared_gu_static"
N512_NAMESPACE = "ffn_down_n512"
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


def verify_device_sources(n256, n512, chain):
    parent = verify_build(n256, N256_MANIFEST)
    candidate = verify_build(n512, N512_MANIFEST)
    verify_build(chain, CHAIN_MANIFEST)
    same, changed, embedded = [], [], []
    for name in sorted(parent["sources"].keys() & candidate["sources"].keys()):
        if Path(name).suffix not in (".cpp", ".h"):
            continue
        before = (n256 / name).read_text()
        after = (n512 / name).read_text().replace(
            N512_NAMESPACE, N256_NAMESPACE).replace(
                N512_NAMESPACE.upper(), N256_NAMESPACE.upper())
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
            "n512_manifest_sha256": N512_MANIFEST,
            "chain_manifest_sha256": CHAIN_MANIFEST}


def fixture_source(source):
    source = once(source, "from decoder_forward import bind_caches, metadata_for_layers\n",
                  "from decoder_forward import bind_caches, metadata_for_layers\n"
                  "from down_tiling import check_down_tiling\n")
    source = once(source, "            self.partial, BLOCKS, 1)\n",
                  "            self.partial, BLOCKS, 1)\n"
                  "        self.down_tiling_before = check_down_tiling(torch, self.body[7])\n")
    source = once(source, "completeFFN381", "completeFFNc90-N512")
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
    parser.add_argument("--n512-build", type=Path, required=True)
    parser.add_argument("--chain-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deploy-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", args.deploy_name):
        raise ValueError("Use a simple deployment name")
    if digest(args.parent / "manifest.json") != FIXTURE_SHA:
        raise ValueError("Use the frozen persistent-top-k v3 fixture")
    frozen = {}
    for line in (args.parent / "frozen-inputs.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(args.parent / name) != expected:
            raise ValueError(f"A frozen fixture file changed: {name}")
        frozen[name] = expected
    compatibility = verify_device_sources(
        args.n256_build, args.n512_build, args.chain_build)
    library = args.n512_build / "ffn_down_n512.so"
    if digest(library) != N512_LIBRARY:
        raise ValueError("Use the accepted N512 FFN library")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in frozen:
        if name == "manifest.json":
            continue
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.parent / name, target)
    shutil.copy2(library, args.output / "complete_ffn.so")
    for name in ("down_tiling.py", "stage.py", "README.md"):
        target = (args.output / name if name == "down_tiling.py"
                  else args.output / "n512_tools" / name)
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(Path(__file__).with_name(name), target)
    packing = args.output / "ffn_host_adapter.py"
    text = once(packing.read_text(), N256_LIBRARY, N512_LIBRARY)
    text = once(text, N256_NAMESPACE, N512_NAMESPACE)
    text = once(text, "Use the original-shared complete FFN library",
                "Use the accepted N512 complete FFN library")
    packing.write_text(text)
    cpu = args.output / "local_chain_cpu_check.py"
    cpu.write_text(once(cpu.read_text(), N256_NAMESPACE, N512_NAMESPACE))
    original = (args.parent / "local_chain.py").read_text()
    candidate = fixture_source(original)
    (args.output / "local_chain.py").write_text(candidate)
    (args.output / "n512-fixture.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), candidate.splitlines(keepends=True),
        fromfile="parent/local_chain.py", tofile="candidate/local_chain.py")))
    (args.output / "n512-source-compatibility.json").write_text(
        json.dumps(compatibility, indent=2) + "\n")
    launcher = args.output / "device_local_chain_run.sh"
    launcher.write_text(once(launcher.read_text(), "vendor-hook-topk-owner-v3",
                             args.deploy_name))
    report = json.loads((args.parent / "manifest.json").read_text())
    report.update({
        "scope": "Persistent-top-k local layer with accepted N512 FFN preparation",
        "parent_fixture_manifest_sha256": FIXTURE_SHA,
        "complete_ffn_library_sha256": N512_LIBRARY,
        "deployment_directory": args.deploy_name,
        "down_tiling_checks": "Full 200 bytes after preparation and each arm",
        "n512_source_compatibility": compatibility,
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
