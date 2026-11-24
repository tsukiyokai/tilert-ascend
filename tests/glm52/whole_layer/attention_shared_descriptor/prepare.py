"""Bind unchanged shared-attention arithmetic to host-captured addresses."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil


PARENT_LIBRARY = "64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c"
LAYER = "tests/glm52/whole_layer/"
FIELDS = (
    "token_x token_scale weight_dq weight_uq_qr weight_uk weight_dkv_kr "
    "scale_dq scale_uq_qr scale_dkv_kr gamma_cq gamma_ckv sin cos slots "
    "kv_cache rope_cache topk query_lengths kv_lengths page_table "
    "value_weight output_weight output_scale bias scratch tail_tiling output"
).split()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one frozen source seam: {before!r}")
    return source.replace(before, after, 1)


def bind_native(source):
    start = source.index("struct DeviceArgs {")
    end = source.index("\n};", start) + len("\n};")
    original = source[start:end]
    fields = re.findall(r"^\s+Address ([^;]+);", original, flags=re.MULTILINE)
    names = [name.strip() for group in fields for name in group.split(",")]
    if names != FIELDS[:20] + ["scratch", "output"]:
        raise ValueError("Unexpected original DeviceArgs field order")
    return (source[:start]
            + "using DeviceArgs = shared_attention_arguments::Attention<GM_ADDR>;\n"
            + "using FullArgs = shared_attention_arguments::Full<GM_ADDR>;\n"
            + "static_assert(sizeof(FullArgs) == 232);"
            + source[end:]).replace(
                '#include "attention_contract.h"',
                '#include "attention_contract.h"\n#include "descriptor.h"', 1)


def bind_full(source):
    source = replace_once(source,
        "    AscendC::TPipe& pipe, const FullConfig& config, DeviceArgs attention,\n"
        "    GM_ADDR value_weight, GM_ADDR output_weight, GM_ADDR output_scale,\n"
        "    GM_ADDR bias, int bias_kind, GM_ADDR scratch, GM_ADDR tail_tiling,\n"
        "    GM_ADDR output)",
        "    AscendC::TPipe& pipe, const FullConfig& config,\n"
        "    const FullArgs& arguments, int bias_kind)")
    source = replace_once(source,
        "    GM_ADDR arena = scratch + config.arena_offset;\n"
        "    attention.scratch = arena;\n"
        "    attention.output = scratch;\n"
        "    Forward(pipe, config.attention, attention);",
        "    Forward(pipe, config.attention, arguments.attention);")
    start = source.index("    mla_tail::device::Forward(")
    end = source.index("    EndPhase(pipe);", start)
    tail = source[start:end]
    mapping = {name: "arguments." + name for name in FIELDS[20:]}
    mapping["arena"] = "arguments.attention.scratch"
    for name in sorted(mapping, key=len, reverse=True):
        tail = re.sub(r"\b" + name + r"\b", mapping[name], tail)
    return source[:start] + tail + source[end:]


def bind_kernel(source):
    start = source.index("void whole_layer_full_attention(")
    body = source.index("\n{", start)
    finish = source.index("\n}\n", body) + 2
    original = source[body:finish]
    for statement in (
        "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);",
        "InitSocState();", "SetSyncBaseAddr(sync_base);", "TPipe pipe;",
    ):
        if original.count(statement) != 1:
            raise ValueError("Unexpected kernel initialization")
    replacement = """void whole_layer_full_attention(
    uint64_t sync_base, FullConfig config, FullArgs arguments, int bias_kind)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);
    TPipe pipe;
    FullForward(pipe, config, arguments, bias_kind);
}"""
    source = source[:start] + replacement + source[finish:]
    launch = source.index("    whole_layer_full_attention<<<")
    end = source.index("\n}", launch)
    capture = ",\n        ".join(", ".join("(GM_ADDR)" + name for name in FIELDS[index:index + 4])
                                  for index in range(0, len(FIELDS), 4))
    body = ("    const auto arguments = shared_attention_arguments::Capture<GM_ADDR>(\n"
            f"        {capture}, config.arena_offset);\n"
            "    whole_layer_full_attention<<<config.attention.blocks, nullptr, stream>>>(\n"
            "        sync_base, config, arguments, bias_kind);")
    return source[:launch] + body + source[end:]


def private_names(source):
    return (source.replace("whole_layer_attention", "whole_layer_attention_descriptor")
            .replace("whole_layer_full_attention", "whole_layer_full_attention_descriptor")
            .replace("namespace mla_tail {", "namespace mla_tail_shared_descriptor {")
            .replace("mla_tail::", "mla_tail_shared_descriptor::")
            .replace("mla_tail_build_tiling", "mla_tail_shared_descriptor_build_tiling")
            .replace("TORCH_LIBRARY_FRAGMENT(tilert,", "TORCH_LIBRARY_FRAGMENT(tilert_shared_descriptor,")
            .replace("TORCH_LIBRARY_IMPL(tilert,", "TORCH_LIBRARY_IMPL(tilert_shared_descriptor,"))


def prepare(parent, output):
    manifest = json.loads((parent / "manifest.json").read_text())
    shared = next(item for item in manifest["builds"] if item["mode"] == "shared")
    if shared["library_sha256"] != PARENT_LIBRARY:
        raise ValueError("Use the frozen runtime212 shared-attention parent")
    parent_source = parent / "shared/source"
    for name, expected in shared["source_hashes"].items():
        if digest(parent_source / name) != expected:
            raise ValueError(f"Frozen parent source changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    source_root = output / "source"
    for name in shared["source_hashes"]:
        target = source_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent_source / name, target)
    changes = {}
    for name, transform in (("attention_native.h", bind_native),
                            ("attention_full_native.h", bind_full),
                            ("attention_full_kernel.cpp", bind_kernel)):
        target = source_root / LAYER / name
        old = target.read_text()
        new = transform(old)
        target.write_text(new)
        changes[name] = "".join(difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile="parent/" + name, tofile="candidate/" + name))
    for path in source_root.rglob("*"):
        if path.suffix in (".cpp", ".h"):
            path.write_text(private_names(path.read_text()))
    shutil.copy2(Path(__file__).with_name("descriptor.h"), source_root / LAYER / "descriptor.h")
    (output / "descriptor.patch").write_text("".join(changes.values()))
    (output / "parent-manifest.json").write_bytes((parent / "manifest.json").read_bytes())
    report = {
        "parent_library_sha256": PARENT_LIBRARY,
        "parent_manifest_sha256": digest(parent / "manifest.json"),
        "descriptor_bytes": 232,
        "additional_gm_bytes": 0,
        "additional_ub_bytes": 0,
        "sources": {str(path.relative_to(source_root)): digest(path)
                    for path in sorted(source_root.rglob("*")) if path.is_file()},
    }
    (output / "source-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.parent, args.output)
