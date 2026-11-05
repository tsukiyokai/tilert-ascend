"""Freeze a shared quantization prefix from the validated N512 FFN."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT_LIBRARY = "c90b3666a1697070eda19d5b3e133e9ed58f36fe429abefceae6ffe102a6a032"
OLD_NAMESPACE = "ffn_down_n512"
NAMESPACE = "ffn_dual_quant"
BRIDGE = "bridge/source/ffn_bridge/body_native.h"
FRONT = "body/source/routed_int4/routed_int4_front.h"
BODY = "body/source/routed_int4/routed_body_native.h"
KERNEL = "source/kernel.cpp"
HELPER = "dual_quant_native.h"
TOOLS = ("prepare.py", "build.py", "test_prepare.py", "abi_probe.cpp",
         "DESIGN.md", "audit_contract.py", HELPER)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one frozen source seam: {before!r}")
    return source.replace(before, after, 1)


def rename(source):
    return source.replace(OLD_NAMESPACE, NAMESPACE).replace(
        OLD_NAMESPACE.upper(), NAMESPACE.upper())


def transform_bridge(source):
    source = once(source, '#include "native.h"\n',
                  '#include "native.h"\n'
                  '#include "../../../source/dual_quant_native.h"\n')
    source = once(source,
                  "    Addresses a, int64_t rows, int64_t blocks, float epsilon)",
                  "    Addresses a, GM_ADDR routed_codes, GM_ADDR routed_scales,\n"
                  "    int64_t rows, int64_t blocks, float epsilon)")
    source = once(source, "    device::Quantize(pipe,", 
                  "    ffn_dual_quant::device::Quantize(pipe,")
    return once(source,
                "        (__gm__ float*)a.values[kSharedScales], rows);",
                "        (__gm__ float*)a.values[kSharedScales],\n"
                "        (__gm__ int8_t*)routed_codes,\n"
                "        (__gm__ float*)routed_scales, rows);")


def quant_block(source):
    start = source.index("        auto values = input_buf.Get<float>();")
    end = source.index("\n    }\n    PipeBarrier<PIPE_ALL>();", start)
    return source[start:end]


def transform_front(source):
    source = once(source, "template <class Timing>\n",
                  "template <bool input_quantized = false, class Timing>\n")
    block = quant_block(source)
    wrapped = "        if constexpr (!input_quantized) {\n" + "\n".join(
        "    " + line if line else line for line in block.splitlines())
    return once(source, block, wrapped + "\n        }")


def transform_body(source):
    source = once(source,
                  "template <CubeFormat weight_format, bool from_dispatch>\n"
                  "__aicore__ inline void Forward(",
                  "template <CubeFormat weight_format, bool from_dispatch,\n"
                  "          bool input_quantized = false>\n"
                  "__aicore__ inline void Forward(")
    source = once(source, "    Timing timing;\n",
                  "    static_assert(!input_quantized || from_dispatch,\n"
                  '                  "prequantized input requires front preparation");\n'
                  "    Timing timing;\n")
    return once(source, "            front::Prepare(pipe,",
                "            front::Prepare<input_quantized>(pipe,")


def transform_kernel(source):
    source = once(source,
                  "                        bridge, arguments.rows, arguments.blocks, epsilon);",
                  "                        bridge,\n"
                  "                        (GM_ADDR)arguments.scratch[kRestoredRows],\n"
                  "                        (GM_ADDR)arguments.scratch[kRestoredScale],\n"
                  "                        arguments.rows, arguments.blocks, epsilon);")
    return once(source, "    routed_body::Forward<format, true>(pipe, arguments,",
                "    routed_body::Forward<format, true, true>(pipe, arguments,")


TRANSFORMS = {BRIDGE: transform_bridge, FRONT: transform_front,
              BODY: transform_body, KERNEL: transform_kernel}


def prepare(parent, output):
    manifest = json.loads((parent / "manifest.json").read_text())
    if manifest["artifacts"][OLD_NAMESPACE + ".so"] != PARENT_LIBRARY:
        raise ValueError("Use the validated N512 parent")
    if digest(parent / (OLD_NAMESPACE + ".so")) != PARENT_LIBRARY:
        raise ValueError("The frozen N512 library changed")
    for name, expected in manifest["sources"].items():
        if digest(parent / name) != expected:
            raise ValueError(f"The frozen source changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest["sources"]:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, target)
    patches = []
    for name, transform in TRANSFORMS.items():
        original = (parent / name).read_text()
        changed = transform(original)
        (output / name).write_text(changed)
        patches.extend(difflib.unified_diff(
            original.splitlines(keepends=True), changed.splitlines(keepends=True),
            fromfile="parent/" + name, tofile="candidate/" + name))
    (output / "dual-quant.patch").write_text("".join(patches))
    for path in output.rglob("*"):
        if path.suffix in (".h", ".cpp"):
            path.write_text(rename(path.read_text()))
    shutil.copy2(Path(__file__).with_name(HELPER), output / "source" / HELPER)
    for name in TOOLS:
        shutil.copy2(Path(__file__).with_name(name), output / "candidate_tools" / name)
    shutil.copy2(parent / "manifest.json", output / "parent-manifest.json")
    report = {
        "parent_library_sha256": PARENT_LIBRARY,
        "scope": "One input load/widen/Abs/ReduceMax; independent original quant suffixes",
        "validation_oracle": "Original DecoderLayer norm/MLP and direct quant byte comparison",
        "descriptor_bytes": 408, "public_scratch_count": 33,
        "maximum_resident_ub_bytes": 195584,
        "quant_phase_ub_bytes": 92192, "routed_body_ub_bytes": 116480,
        "additional_gm_bytes": 0, "additional_ub_bytes": 0,
        "body_barriers": 5, "final_outer_end_phases": 0,
        "source_changes_before_namespace": sorted(TRANSFORMS),
        "additional_device_sources": ["source/" + HELPER],
        "sources": {str(path.relative_to(output)): digest(path)
                    for path in sorted(output.rglob("*")) if path.is_file()},
    }
    (output / "source-transforms.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.parent.resolve(), args.output.resolve())
