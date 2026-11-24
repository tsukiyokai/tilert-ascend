"""Freeze the accepted TP layer with the accepted private tree reducer."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil


HERE = Path(__file__).resolve().parent
PARENT_MANIFEST = "e69649149c455134bc0f9757775ac19deb9331f1dfd6da3cb08b8c344024dcf1"
PARENT_LIBRARY = "c2b72e469c3ee80a5fe7d9df4a3cceedc0e65cca95e6f426aea762c213392462"
PARENT_BUILD = "956a08e6040977a47590e7ed369acc3019b24999839c4f87e390a35034ffd7fa"
PARENT_COMMANDS = "563b676ca94d74f7e1c8c9611db9ed8491e96f69eb997c1cfeebda581ad56ef9"
HOST_SHA = "c3caf00eaf135c963c5ef5e990ae309531e9802aa03574edbbdc0fd310534493"
TREE_MANIFEST = "22d408d22afcccfce6941a0a9cd5c2cdc3e38c53110401f21cb46832b4b79716"
TREE_BUILD = "b857a54c47aa61deb5429e81ee84651fdec7e05f840343d7251f64c41b6961b3"
TREE_LIBRARY = "a0bd2462e2f999280720fa5060466159a2eab13ce5541639ddc7afef35a45ebf"
EXPORTS = (
    "indexed_launch", "shared_launch", "last_error", "record_bytes",
    "initialize", "window_create", "window_info", "submit",
    "window_destroy", "finalize",
)
RENAMED_FILES = (
    "source/contract.h", "source/contract_test.cpp",
    "source/host.cpp", "source/kernel.cpp",
)
TREE_FILES = (
    "communication.h", "communication_device.h", "communication_host.cpp",
    "tree_plan.h", "reduce.inc", "contract_test.cpp",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_digest(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Frozen input changed: {path}")


def verify(root, files):
    for name, expected in files.items():
        require_digest(root / name, expected)


def private_names(source):
    source = re.sub(r"\blayer_tp_chain\b", "layer_tp_tree_chain", source)
    source = re.sub(r"\bwhole_layer\b", "whole_layer_tree", source)
    source = source.replace("TILERT_LAYER_TP_CHAIN_", "TILERT_LAYER_TP_TREE_CHAIN_")
    for name in EXPORTS:
        source = re.sub(r"\blayer_tp_" + name + r"\b",
                        "layer_tp_tree_" + name, source)
    return source


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare(parent, tree, host, output):
    require_digest(parent / "manifest.json", PARENT_MANIFEST)
    require_digest(parent / "results/build.json", PARENT_BUILD)
    require_digest(parent / "results/commands.jsonl", PARENT_COMMANDS)
    require_digest(parent / "results/layer_tp_chain.so", PARENT_LIBRARY)
    require_digest(tree / "manifest.json", TREE_MANIFEST)
    require_digest(tree / "results/build.json", TREE_BUILD)
    require_digest(tree / "results/communication_tree_probe.so", TREE_LIBRARY)
    require_digest(host, HOST_SHA)
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    tree_manifest = json.loads((tree / "manifest.json").read_text())
    verify(parent, parent_manifest["sources"])
    verify(tree, tree_manifest["sources"])
    output.mkdir(parents=True, exist_ok=False)
    for name in parent_manifest["sources"]:
        copy(parent / name, output / name)
    for name in RENAMED_FILES + ("source/build.py",):
        copy(parent / name, output / "parent" / name)
    for name in TREE_FILES[:3]:
        copy(parent / "communication" / name,
             output / "parent/communication" / name)
    for root, label in ((parent, "parent"), (tree, "tree")):
        copy(root / "manifest.json", output / label / "manifest.json")
        copy(root / "results/build.json", output / label / "build.json")
        copy(root / "results/commands.jsonl", output / label / "commands.jsonl")
    copy(host, output / "parent/host.py")
    changes = []
    for name in RENAMED_FILES:
        original = (parent / name).read_text()
        transformed = private_names(original)
        (output / name).write_text(transformed)
        changes.extend(difflib.unified_diff(
            original.splitlines(keepends=True), transformed.splitlines(keepends=True),
            fromfile="parent/" + name, tofile="candidate/" + name))
    for name in TREE_FILES:
        copy(tree / "source" / name, output / "communication" / name)
    for name in TREE_FILES[:3]:
        changes.extend(difflib.unified_diff(
            (parent / "communication" / name).read_text().splitlines(keepends=True),
            (output / "communication" / name).read_text().splitlines(keepends=True),
            fromfile="parent/communication/" + name,
            tofile="candidate/communication/" + name))
    original_host = host.read_text()
    private_host = original_host.replace("layer_tp_", "layer_tp_tree_")
    (output / "source/host.py").write_text(private_host)
    changes.extend(difflib.unified_diff(
        original_host.splitlines(keepends=True), private_host.splitlines(keepends=True),
        fromfile="parent/host.py", tofile="candidate/source/host.py"))
    for name in ("prepare.py", "build.py", "test_contract.py",
                 "integration_contract.cpp", "README.md"):
        copy(HERE / name, output / "source" / name)
    (output / "tree-integration.patch").write_text("".join(changes))
    source_hashes = {str(path.relative_to(output)): digest(path)
                     for path in sorted(output.rglob("*")) if path.is_file()}
    report = {
        "scope": "Private TP layer using the accepted FP32 postorder reducer",
        "parent_manifest_sha256": PARENT_MANIFEST,
        "parent_library_sha256": PARENT_LIBRARY,
        "parent_build_sha256": PARENT_BUILD,
        "tree_manifest_sha256": TREE_MANIFEST,
        "tree_probe_library_sha256": TREE_LIBRARY,
        "tree_build_sha256": TREE_BUILD,
        "host_parent_sha256": HOST_SHA,
        "namespace": "layer_tp_tree_chain", "c_prefix": "layer_tp_tree_",
        "kernel_argument_bytes": 1648,
        "record_bytes": [664, 928, 48],
        "communication_chunk_elements": 2048,
        "communication_ub_bytes": 45056,
        "ranks": [2, 16],
        "device_execution": False,
        "sources": source_hashes,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    frozen = {**source_hashes, "manifest.json": digest(output / "manifest.json")}
    (output / "frozen-inputs.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in sorted(frozen.items())))
    print(json.dumps({"files": len(source_hashes),
                      "manifest_sha256": frozen["manifest.json"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "tree", "host", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(*(getattr(args, name).resolve()
              for name in ("parent", "tree", "host", "output")))
