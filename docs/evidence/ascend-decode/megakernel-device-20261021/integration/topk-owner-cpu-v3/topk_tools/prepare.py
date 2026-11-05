"""Freeze a local layer that writes selected indices to their retained owner."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


PARENT_SHA = "609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f"
PARENT_MANIFEST = "31fffff64ecda3526c53a031e1f41baf01e56657af184d4704f41ba40dfdd0f2"
HOST_SHA = "70bfd7393e7ceb2a6921100be6057f0cea3f922c52d6e3df2e02ea398c66e6f9"
INDEXED_HEADER = (
    "attention/source/tests/glm52/whole_layer/attention_indexed_native.h")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one frozen source anchor: {old!r}")
    return text.replace(old, new)


def transform_indexed(text):
    text = replace_once(
        text, "const DeviceArgs& a, int bias_kind)\n",
        "const DeviceArgs& a, int bias_kind,\n"
        "                               GM_ADDR retained_topk = nullptr)\n")
    anchor = "    GM_ADDR memory = a.scratch;\n"
    text = replace_once(text, anchor, anchor +
                        "    GM_ADDR topk = retained_topk != nullptr\n"
                        "        ? retained_topk : memory + c.topk;\n")
    text = replace_once(
        text, "a.caches[0], a.caches[1], memory + c.topk,\n",
        "a.caches[0], a.caches[1], topk,\n")
    return replace_once(
        text, "memory + c.indexer_offset, memory + c.topk};\n",
        "memory + c.indexer_offset, topk};\n")


def rename(text):
    return text.replace("layer_local_chain", "layer_topk_owner").replace(
        "layer_local_", "layer_topk_").replace(
        "LAYER_LOCAL_CHAIN", "LAYER_TOPK_OWNER")


def prepare(parent, parent_host, output):
    if (digest(parent / "manifest.json") != PARENT_MANIFEST
            or digest(parent / "results/layer_local_chain.so") != PARENT_SHA):
        raise ValueError("Use the accepted local-layer v4 source and library")
    if digest(parent_host) != HOST_SHA:
        raise ValueError("Use the separately validated local-layer Python host")
    manifest = json.loads((parent / "manifest.json").read_text())
    for relative, expected in manifest["sources"].items():
        if digest(parent / relative) != expected:
            raise ValueError(f"The frozen local-layer source changed: {relative}")
    output.mkdir(parents=True, exist_ok=False)
    for relative in manifest["sources"]:
        if relative == "source/host_cpu_check.py":
            continue
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / relative, target)
    for name in ("attention-provenance.json", "attention-manifest.json",
                 "ffn-manifest.json", "ffn-commands.jsonl"):
        shutil.copy2(parent / name, output / name)
    shutil.copy2(parent / "manifest.json", output / "parent-manifest.json")
    header = output / INDEXED_HEADER
    header.write_text(transform_indexed(header.read_text()))
    own = output / "source"
    shutil.copy2(parent_host, own / "host.py")
    for path in own.iterdir():
        if path.suffix in (".cpp", ".h", ".py"):
            path.write_text(rename(path.read_text()))
    kernel = own / "kernel.cpp"
    kernel.write_text(replace_once(
        kernel.read_text(), "                                    config.bias_kind);\n",
        "                                    config.bias_kind,\n"
        "                                    (GM_ADDR)addresses.shared_topk);\n"))
    contract = own / "contract.h"
    contract.write_text(replace_once(
        contract.read_text(), "          addresses.attention.output)) {\n",
        "          addresses.attention.output && addresses.shared_topk &&\n"
        "          addresses.shared_topk % 32 == 0)) {\n"))
    test = own / "contract_test.cpp"
    text = replace_once(
        test.read_text(), "    addresses.post_gamma = 0xc000;\n",
        "    addresses.shared_topk = 0xf000;\n"
        "    addresses.post_gamma = 0xc000;\n")
    text = replace_once(
        text, "            auto wrong = addresses;\n",
        "            auto wrong = addresses;\n"
        "            wrong.shared_topk = 0;\n"
        "            assert(!Valid(config, wrong));\n"
        "            wrong.shared_topk = addresses.shared_topk + 4;\n"
        "            assert(!Valid(config, wrong));\n"
        "            wrong = addresses;\n")
    test.write_text(text)
    host = own / "host.py"
    host.write_text(replace_once(
        host.read_text(),
        "        if not indexed and shared_topk is None:\n"
        "            raise ValueError(\"Shared-indexer attention requires its retained top-k owner\")\n",
        "        from topk_contract import validate_topk\n"
        "        validate_topk(shared_topk, geometry[0], norm[0].device, groups)\n"))
    for name in ("prepare.py", "topk_contract.py", "test_topk_contract.py",
                 "README.md"):
        shutil.copy2(Path(__file__).with_name(name), own / name)
    report = {
        "scope": "Local input norm, attention and complete FFN; indexed selection writes directly to the persistent top-k owner",
        "parent_library_sha256": PARENT_SHA,
        "parent_manifest_sha256": PARENT_MANIFEST,
        "parent_host_sha256": HOST_SHA,
        "record_bytes": {"config": 664, "addresses": 928},
        "topk": "Existing shared_topk slot is mandatory in both variants; indexed producer and SFA consumer use the same address",
        "changes": ["Selector output and current-layer SFA input address",
                    "Require an aligned retained INT32 ND top-k owner",
                    "Private library namespace and C exports"],
        "synchronization": "Parent selector publication and mixed-core join are unchanged",
        "scratch": "Parent arenas and all offsets remain unchanged; the internal top-k slice is no longer written by the indexed layer",
        "tp_contract": "Both reductions remain identity",
        "device_execution": False,
        "sources": {str(path.relative_to(output)): digest(path)
                    for path in sorted(output.rglob("*")) if path.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--host", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.parent.resolve(), args.host.resolve(),
                     args.output.resolve())
    print(json.dumps({"output": str(args.output),
                      "sources": len(result["sources"])}))
