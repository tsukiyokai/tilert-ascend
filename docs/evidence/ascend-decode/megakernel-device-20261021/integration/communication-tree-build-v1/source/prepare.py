"""Freeze a private tree-reduction probe from the accepted serial probe."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


HERE = Path(__file__).resolve().parent
PARENT_LIBRARY_SHA = "1be5fa375d0d6e7a2a6dcf7f5265cf886317108398aeec940e2b1ce0de04e22f"
PARENT_DEVICE_SHA = "cf9e9a843c82ff77a4a4c2d6e4ff65024107cbd3c45044280c6502d65be8bde7"
PARENT_FILES = (
    "communication.h", "communication_device.h", "communication_host.cpp",
    "communication_probe.h", "communication_probe.cpp",
    "communication_probe_host.cpp", "communication_build.sh",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Expected exactly one source anchor: {before!r}")
    return text.replace(before, after, 1)


def private_names(text):
    text = re.sub(r"\bwhole_layer\b", "whole_layer_tree", text)
    text = text.replace("TESTS_GLM52_WHOLE_LAYER_",
                        "TESTS_GLM52_COMMUNICATION_TREE_")
    text = text.replace("communication_probe_", "communication_tree_probe_")
    text = text.replace("void communication_probe(",
                        "void communication_tree_probe(")
    return text.replace("communication_probe<<<", "communication_tree_probe<<<")


def transform(name, text):
    if name.endswith(".sh"):
        return text
    if name == "communication.h":
        text = replace_once(text, "kCommunicationChunk * 10;",
                            "kCommunicationChunk * 22;")
    if name == "communication_device.h":
        text = replace_once(text, '#include "shmem.h"',
                            '#include "shmem.h"\n#include "tree_plan.h"')
        text = replace_once(text, "with 20 KiB available", "with 44 KiB available")
        text = replace_once(
            text, "pipe.InitBuffer(accumulator_buffer_, kCommunicationChunk * 4);",
            "pipe.InitBuffer(accumulator_buffer_, kCommunicationChunk * 4 * 5);")
        text = replace_once(
            text, "        pipe.InitBuffer(term_buffer_, kCommunicationChunk * 4);\n", "")
        text = replace_once(text, "term = term_buffer_.Get<float>();",
                            "term = accumulator[kCommunicationChunk];")
        text = replace_once(
            text, "    AscendC::TBuf<AscendC::TPosition::VECCALC> term_buffer_;\n", "")
        begin = text.index("__aicore__ inline void ReduceCommunicationPayload(")
        end = text.index("// All resident AIC and AIV cores", begin)
        text = text[:begin] + '#include "reduce.inc"\n\n' + text[end:]
    return private_names(text)


def prepare(parent, output):
    manifest = json.loads((parent / "manifest.json").read_text())
    for name in PARENT_FILES:
        if digest(parent / name) != manifest["files_sha256"][name]:
            raise ValueError(f"Accepted parent source changed: {name}")
    if digest(parent / "communication_probe.so") != PARENT_LIBRARY_SHA:
        raise ValueError("Accepted serial library changed")
    if digest(parent / "communication_device.h") != PARENT_DEVICE_SHA:
        raise ValueError("Accepted serial reduction changed")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for name in PARENT_FILES:
        original = (parent / name).read_text()
        (source / name).write_text(transform(name, original))
    for name in ("tree_plan.h", "reduce.inc", "contract_test.cpp",
                 "prepare.py", "build.py", "test_prepare.py", "README.md"):
        shutil.copyfile(HERE / name, source / name)
    original = output / "parent"
    original.mkdir()
    for name in PARENT_FILES:
        shutil.copyfile(parent / name, original / name)
    shutil.copyfile(parent / "manifest.json", original / "manifest.json")
    result = {
        "parent": str(parent), "parent_library_sha256": PARENT_LIBRARY_SHA,
        "parent_device_sha256": PARENT_DEVICE_SHA,
        "parent_manifest_sha256": digest(parent / "manifest.json"),
        "scope": "Only local FP32 tree reduction and its phase-local UB allocation",
        "namespace": "whole_layer_tree", "c_prefix": "communication_tree_probe_",
        "chunk_elements": 2048, "fp32_stack_slots": 5, "ub_bytes": 45056,
        "sources": {str(p.relative_to(output)): digest(p)
                    for p in sorted(output.rglob("*")) if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.parent.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
