"""Freeze private tree-control and five-phase RSAG probe sources."""

import argparse
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
PARENT = HERE.parents[1] / "tests/glm52/whole_layer"
PROTOCOL_SHA256 = "17cf70bfc88dfc03afeb8bb260f512196982cbff5b335bbba34db2071377ca75"
PARENT_HASHES = {
    "communication.h":
        "be3b248287e3cb51c288b68f3f466fcb4d4750979f36eee2b7ab9378ff0efff8",
    "communication_device.h":
        "cf9e9a843c82ff77a4a4c2d6e4ff65024107cbd3c45044280c6502d65be8bde7",
    "communication_probe.h":
        "7b60c5589b1b428bf8cb791b59b20eaba964402b540a700a8299b97b298dadfe",
    "communication_host.cpp":
        "0c1cc5abde6fec3812b844833b7631958a4ae965f2b13fcc7fef1f93167e4701",
    "communication_probe.cpp":
        "83ae9cf423836bfc5b466073190da3d98a79b60953b7c07faf8ef280677a5225",
    "communication_probe_host.cpp":
        "e530a7d66813f26d06c74a6c64f1b6c3dde85cf6385ff48b240c3eb3ff606121",
    "communication_build.sh":
        "2453d0bb7372249fd17333dcedc530ed5ef5b1f30c517f8eb826a03e22199cc6",
    "communication_tree/tree_plan.h":
        "7546e8a03c5c500a64e9739c46c62f758a45e5d619ea548808595652ed225af9",
    "communication_tree/reduce.inc":
        "f3011d78ab8eb10e1197221122098e95f1f4f4259ba26907a73a85fd61d11a0f",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Changed source anchor: {before!r}")
    return text.replace(before, after, 1)


def private_names(text, arm):
    namespace = "trt_rsag_" + arm
    text = text.replace("rsag_contract", namespace + "_contract")
    text = text.replace("TRT_RSAG_LAYOUT_H", namespace.upper() + "_LAYOUT_H")
    text = re.sub(r"\bwhole_layer(?:_tree)?\b", namespace, text)
    text = text.replace("TESTS_GLM52_WHOLE_LAYER_", namespace.upper() + "_")
    text = text.replace("TESTS_GLM52_COMMUNICATION_TREE_",
                        namespace.upper() + "_TREE_")
    text = text.replace("communication_probe_", namespace + "_probe_")
    text = text.replace("void communication_probe(",
                        "void " + namespace + "_probe(")
    return text.replace("communication_probe<<<", namespace + "_probe<<<")


def tree_buffers(text):
    text = replace_once(text, "with 20 KiB available", "with 44 KiB available")
    text = replace_once(
        text, "pipe.InitBuffer(accumulator_buffer_, kCommunicationChunk * 4);",
        "pipe.InitBuffer(accumulator_buffer_, kCommunicationChunk * 4 * 5);")
    text = replace_once(text,
        "        pipe.InitBuffer(term_buffer_, kCommunicationChunk * 4);\n", "")
    text = replace_once(text, "term = term_buffer_.Get<float>();",
                        "term = accumulator[kCommunicationChunk];")
    return replace_once(text,
        "    AscendC::TBuf<AscendC::TPosition::VECCALC> term_buffer_;\n", "")


def transform(name, text, arm):
    if name.endswith(".sh"):
        return text
    if name == "communication.h":
        text = replace_once(text, "#include <cstdint>",
                            '#include <cstdint>\n#include "layout.h"')
        text = replace_once(text, "kCommunicationChunk * 10;",
                            "kCommunicationChunk * 22;")
        if arm == "control":
            text = replace_once(
                text, "layout.ready + 2 * ranks * kCommunicationCacheLine;",
                "rsag_contract::Align512(layout.ready + 2 * ranks * kCommunicationCacheLine);")
            text = text.replace("12585024", "12585472")
        else:
            start = text.index("struct CommunicationLayout {")
            end = text.index("// One window belongs", start)
            text = text[:start] + """using CommunicationLayout = rsag_contract::Layout;

WHOLE_LAYER_COMMUNICATION_DEVICE constexpr CommunicationLayout
MakeCommunicationLayout(int64_t capacity, int32_t ranks)
{
    return rsag_contract::MakeLayout(capacity, ranks);
}

""" + text[end:]
            text = text.replace("12585024", "1577472")
    elif name == "communication_device.h":
        text = tree_buffers(text)
        text = replace_once(text, '#include "shmem.h"',
                            '#include "shmem.h"\n#include "tree_plan.h"')
        if arm == "control":
            start = text.index("__aicore__ inline void ReduceCommunicationPayload(")
            end = text.index("// All resident AIC and AIV cores", start)
            text = text[:start] + '#include "reduce.inc"\n\n' + text[end:]
        else:
            start = text.index("__aicore__ inline __gm__ int32_t* CommunicationReady(")
            end = text.index("}  // namespace whole_layer", start)
            text = text[:start] + '#include "rsag_device.inc"\n\n' + text[end:]
    elif name == "communication_host.cpp":
        start = text.index("    // calloc zeroes")
        end = text.index("    return window;", start)
        text = text[:start] + """    window.address = aclshmem_align(512, window.bytes);
    if (window.address == nullptr) {
        throw std::runtime_error("SHMEM communication allocation failed");
    }
    if (!rsag_contract::AlignedPayload(reinterpret_cast<uintptr_t>(window.address))
        || aclrtMemset(window.address, window.bytes, 0, window.bytes) != ACL_SUCCESS
        || aclrtSynchronizeDevice() != ACL_SUCCESS) {
        aclshmem_free(window.address);
        throw std::runtime_error("aligned SHMEM zero/synchronization failed");
    }
    // The driver must rendezvous all ranks after prepare, before any execute.
""" + text[end:]
    elif name == "communication_probe_host.cpp":
        text = replace_once(text,
            '          "communication probe scratch allocation failed");',
            '          "communication probe scratch allocation failed");\n'
            '    Check(rsag_contract::AlignedPayload(reinterpret_cast<uintptr_t>(*address)),\n'
            '          "probe scratch requires 512-byte alignment");')
        text = replace_once(text,
            '              "communication probe received a null launch argument");',
            '              "communication probe received a null launch argument");\n'
            '        Check(rsag_contract::AlignedPayload(reinterpret_cast<uintptr_t>(partial))\n'
            '                  && rsag_contract::AlignedPayload(reinterpret_cast<uintptr_t>(weight))\n'
            '                  && rsag_contract::AlignedPayload(reinterpret_cast<uintptr_t>(first_sum))\n'
            '                  && rsag_contract::AlignedPayload(reinterpret_cast<uintptr_t>(output)),\n'
            '              "probe tensors require 512-byte alignment");')
    return private_names(text, arm)


def prepare(parent, output):
    if digest(parent / "communication_tree/rsag_protocol.md") != PROTOCOL_SHA256:
        raise ValueError("Frozen RSAG protocol changed")
    for name, expected in PARENT_HASHES.items():
        if digest(parent / name) != expected:
            raise ValueError(f"Frozen parent changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    report = {"parent": str(parent), "parent_sha256": PARENT_HASHES,
              "protocol_sha256": PROTOCOL_SHA256,
              "sources": {}, "arms": {}}
    for arm in ("control", "candidate"):
        source = output / arm / "source"
        source.mkdir(parents=True)
        for name in PARENT_HASHES:
            destination = Path(name).name
            if name.startswith("communication_tree/"):
                body = (parent / name).read_text()
                if destination == "tree_plan.h":
                    body = private_names(body, arm)
            else:
                body = transform(name, (parent / name).read_text(), arm)
            (source / destination).write_text(body)
        (source / "layout.h").write_text(
            private_names((HERE / "layout.h").read_text(), arm))
        if arm == "candidate":
            (source / "rsag_device.inc").write_text(
                private_names((HERE / "rsag_device.inc").read_text(), arm))
        report["arms"][arm] = {"namespace": "trt_rsag_" + arm,
                               "c_prefix": "trt_rsag_" + arm + "_probe_",
                               "mixed_joins_per_sum": 3 if arm == "control" else 5}
    report["sources"] = {str(path.relative_to(output)): digest(path)
                         for path in sorted(output.rglob("*")) if path.is_file()}
    report["experiment_sources"] = {
        name: digest(HERE / name) for name in (
            "prepare.py", "layout.h", "rsag_device.inc", "contract.cpp",
            "check.py", "build.py", "device_check.py", "README.md")}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.parent.resolve(), args.output.resolve()),
                     indent=2))


if __name__ == "__main__":
    main()
