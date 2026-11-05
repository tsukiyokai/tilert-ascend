"""Freeze the existing 609 paired fixture with K128 candidate preparation."""

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil


FIXTURE_SHA = "a0fa94aa2ff3829ef5e95b949e42fa95f4ca3addf9b2b4fc90cf283bd0595cd8"
BASE_MANIFEST = "ee4b590ff1b67e58081e2e9cc6b8cda80a466aa213992e217c25fa3ccec79eee"
K128_MANIFEST = "98336396e67bae92c1ec4d9eb72b2575d2aacf729d06f092daba85644b59384e"
CHAIN_MANIFEST = "31fffff64ecda3526c53a031e1f41baf01e56657af184d4704f41ba40dfdd0f2"
OLD_LIBRARY = "bbd52e425a1e94bb199e464e3bf6001c4420e8238afe993e8741e38af50b77ef"
K128_LIBRARY = "0842cd0f3a946c39aefb3e926e01eb870a61d6d1ad8689aebcf52a1b3daa2b65"
K128_TILING = "c2b73af9d605ad8e64ea551f288d188a0cf636828a5a0471ea7c63d616f50999"
TILING_SOURCE = "body/source/routed_int4/routed_int4_tiling.cpp"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Expected one frozen anchor: {before!r}")
    return text.replace(before, after)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def verify_build(directory, expected):
    if digest(directory / "manifest.json") != expected:
        raise ValueError(f"Unexpected build manifest: {directory}")
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, sha in manifest["sources"].items():
        if digest(directory / name) != sha:
            raise ValueError(f"A frozen source changed: {directory / name}")
    return manifest


def verify_fixture(directory):
    if digest(directory / "manifest.json") != FIXTURE_SHA:
        raise ValueError("Use the frozen N512/N1024 paired fixture")
    files = {}
    for line in (directory / "frozen-inputs.sha256").read_text().splitlines():
        sha, name = line.split("  ", 1)
        if digest(directory / name) != sha:
            raise ValueError(f"A frozen fixture file changed: {name}")
        files[name] = sha
    manifest = json.loads((directory / "manifest.json").read_text())
    if files != {**manifest["files_sha256"], "manifest.json": FIXTURE_SHA}:
        raise ValueError("The fixture manifest and full file list differ")
    return files, manifest


def source_contract(base, candidate, chain):
    parent = verify_build(base, BASE_MANIFEST)
    child = verify_build(candidate, K128_MANIFEST)
    verify_build(chain, CHAIN_MANIFEST)
    same, changed, embedded = [], [], []
    for name in sorted(parent["sources"].keys() & child["sources"].keys()):
        if (Path(name).suffix not in (".cpp", ".h")
                or name.startswith("candidate_tools/")):
            continue
        before = (base / name).read_text()
        after = (candidate / name).read_text().replace(
            "ffn_down_k128", "ffn_original_shared_gu_static").replace(
                "FFN_DOWN_K128", "FFN_ORIGINAL_SHARED_GU_STATIC")
        (same if before == after else changed).append(name)
        if (chain / "ffn" / name).read_bytes() != (base / name).read_bytes():
            raise ValueError(f"The 609 embedded source changed: {name}")
        embedded.append(name)
    if changed != [TILING_SOURCE] or len(same) != 48 or len(embedded) != 49:
        raise ValueError("The common device sources exceed the host-tiling change")
    return {
        "normalized_unchanged_files": same, "changed_files": changed,
        "embedded_ffn_files_identical": embedded,
        "base_manifest_sha256": BASE_MANIFEST,
        "k128_manifest_sha256": K128_MANIFEST,
        "chain_manifest_sha256": CHAIN_MANIFEST,
        "new_host_plan_sources": {
            name: sha for name, sha in child["sources"].items()
            if name in ("source/down_plan.h", "source/static_down_plan.cpp")},
    }, child


def candidate_tiling(manifest):
    records = [value for value in manifest["compiled_tilings"]["candidate"]
               if value["projection"] == 1]
    if len(records) != 2 or {value["layout"] for value in records} != {0, 1}:
        raise ValueError("Both compiled ND and NZ Down records are required")
    for value in records:
        raw = bytes.fromhex(value["raw_hex"])
        if (value["status"] != 0 or len(raw) != 200
                or hashlib.sha256(raw).hexdigest() != K128_TILING
                or value["raw_sha256"] != K128_TILING
                or [value[key] for key in ("baseM", "baseN", "baseK")]
                != [32, 1024, 128]):
            raise ValueError("The compiled K128 host record changed")
    if records[0]["raw_hex"] != records[1]["raw_hex"]:
        raise ValueError("The ND and NZ byte oracles differ")
    return records


def transform_files(output, old_tiling, new_tiling, deploy_name):
    replacements = {
        "ffn_host_adapter.py": [
            (OLD_LIBRARY, K128_LIBRARY),
            ("tilert_ffn_down_n1024", "tilert_ffn_down_k128"),
            ("accepted N1024 complete FFN library", "accepted K128 complete FFN library")],
        "down_tiling.py": [
            (old_tiling["hex"], new_tiling["hex"]),
            (old_tiling["sha256"], new_tiling["sha256"]),
            ("accepted N1024 host tiler", "accepted K128 host tiler"),
            ("from the N1024 host tiler", "from the K128 host tiler")],
        "tiling_packets.py": [(OLD_LIBRARY, K128_LIBRARY)],
        "local_chain_cpu_check.py": [
            ("tilert_ffn_down_n1024::", "tilert_ffn_down_k128::")],
        "local_chain.py": [
            ("completeFFNbbd-N1024", "completeFFN0842-K128"),
            ("N512 versus N1024 Down tiling", "N512 versus N1024/K128 Down tiling")],
        "layer_pair_timing.py": [
            ("N512 versus N1024 Down tiling", "N512 versus N1024/K128 Down tiling")],
        "test_contract.py": [
            ("n1024-source-compatibility.json", "k128-source-compatibility.json"),
            ('len(compatibility["normalized_unchanged_files"]), 49',
             'len(compatibility["normalized_unchanged_files"]), 48')],
        "device_local_chain_run.sh": [
            ("vendor-hook-layer-down-n1024-timing-v1", deploy_name)],
    }
    patch = []
    for name, edits in replacements.items():
        path = output / name
        before = path.read_text()
        after = before
        for old, new in edits:
            after = once(after, old, new)
        if path.suffix == ".py":
            compile(after, name, "exec")
        if name in ("local_chain.py", "layer_pair_timing.py"):
            restored = after
            for old, new in reversed(edits):
                restored = once(restored, new, old)
            if ast.dump(ast.parse(restored)) != ast.dump(ast.parse(before)):
                raise ValueError(f"The executable fixture changed: {name}")
        path.write_text(after)
        patch.extend(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f"n1024-timing-v1/{name}", tofile=f"k128-timing-v1/{name}"))
    (output / "k128-fixture.patch").write_text("".join(patch))
    return sorted(replacements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--base-build", type=Path, required=True)
    parser.add_argument("--candidate-build", type=Path, required=True)
    parser.add_argument("--chain-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deploy-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", args.deploy_name):
        raise ValueError("Use a simple deployment name")
    frozen, report = verify_fixture(args.parent)
    compatibility, build = source_contract(
        args.base_build, args.candidate_build, args.chain_build)
    records = candidate_tiling(build)
    library = args.candidate_build / "ffn_down_k128.so"
    if digest(library) != K128_LIBRARY:
        raise ValueError("Use the accepted K128 FFN library")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in frozen:
        if name == "manifest.json":
            continue
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.parent / name, target)
    shutil.copy2(library, args.output / "complete_ffn.so")
    paired = json.loads((args.output / "paired-tilings.json").read_text())
    candidate = {"hex": records[0]["raw_hex"], "sha256": K128_TILING,
                 "geometry": [32, 1024, 128], "library_sha256": K128_LIBRARY}
    changed = transform_files(
        args.output, paired["candidate"], candidate, args.deploy_name)
    paired["candidate"] = candidate
    write_json(args.output / "paired-tilings.json", paired)
    write_json(args.output / "k128-source-compatibility.json", compatibility)
    write_json(args.output / "k128-compiled-tilings.json", {
        "manifest_sha256": K128_MANIFEST, "records": records,
        "source": "Actual compiled K128 DSO host tiler, both ND and NZ",
        "device_execution": False})
    tool_output = args.output / "k128_tools"
    tool_output.mkdir()
    for name in ("stage.py", "README.md"):
        shutil.copy2(Path(__file__).with_name(name), tool_output / name)
    report["inherited_n1024_source_compatibility"] = report.pop(
        "n1024_source_compatibility")
    report.update({
        "scope": "Same-address 609 plus TopK copy: N512/K128 versus N1024/K128 Down",
        "parent_fixture_manifest_sha256": FIXTURE_SHA,
        "candidate_build_manifest_sha256": K128_MANIFEST,
        "complete_ffn_library_sha256": K128_LIBRARY,
        "deployment_directory": args.deploy_name,
        "k128_source_compatibility": compatibility,
        "changes_from_frozen_timing": changed + [
            "complete_ffn.so", "paired-tilings.json"],
        "executable_timing_ast_unchanged_except_report_labels": True,
        "device_execution": False,
        "files_sha256": {
            str(path.relative_to(args.output)): digest(path)
            for path in sorted(args.output.rglob("*")) if path.is_file()},
    })
    manifest = args.output / "manifest.json"
    write_json(manifest, report)
    files = {**report["files_sha256"], "manifest.json": digest(manifest)}
    (args.output / "frozen-inputs.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in sorted(files.items())))
    print(json.dumps({"output": str(args.output), "files": len(files),
                      "manifest_sha256": digest(manifest)}, indent=2))


if __name__ == "__main__":
    main()
