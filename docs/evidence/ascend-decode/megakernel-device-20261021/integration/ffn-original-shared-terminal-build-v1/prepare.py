"""Bind the complete FFN to the installed shared-activation dispatch."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT_LIBRARY = "276352762139244a1ddfc8184065a856c2ae75ee2ccad5961af2af93f0dd5e00"
PARENT_CHECKER = "c5b5586376f9f412b1f9da2789089766a37231a682c8ef16c3c0f6f384e56d80"
NATIVE = "body/source/shared_w8/shared_w8_native.h"
QUANT_FLAG = "SHARED_W8_DYNAMIC_PERFORMANCE_QUANT"
UP_BIAS = """            if constexpr (kGroupedQuant) {
                Adds(held.values[kIntermediate], held.values[kIntermediate],
                     0.0F, kIntermediate);
                PipeBarrier<PIPE_V>();
            }
"""
QUANT_START = "            if constexpr (kGroupedQuant) {\n"
QUANT_END = """                Cast(held.codes, held.halves, RoundMode::CAST_NONE, kIntermediate);
            }
"""
DYNAMIC_QUANT = """            Fence<HardEvent::V_S>();
            // CANN 9.2 key 30013: scalar scale and reciprocal multiplication.
            const float scale = held.maximum.GetValue(0) / 127.0F;
            held.maximum.SetValue(0, scale);
            const float factor = scale == 0.0F ? 1.0F : 1.0F / scale;
            Fence<HardEvent::S_V>();
            Muls(held.values, held.values, factor, kIntermediate);
            PipeBarrier<PIPE_V>();
            auto integers = held.sum.ReinterpretCast<int16_t>();
            Cast(integers, held.values, RoundMode::CAST_RINT, kIntermediate);
            PipeBarrier<PIPE_V>();
            Cast(held.halves, integers, RoundMode::CAST_NONE, kIntermediate);
            PipeBarrier<PIPE_V>();
            Cast(held.codes, held.halves, RoundMode::CAST_NONE, kIntermediate);
"""


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one frozen source seam: {old!r}")
    return text.replace(old, new, 1)


def bind_activation(source):
    source = once(source, UP_BIAS, "")
    source = once(
        source,
        "Cast(held.values, held.sum, RoundMode::CAST_RINT, kGateColumns);",
        "Cast(held.values, held.sum, RoundMode::CAST_NONE, kGateColumns);")
    if source.count(QUANT_START) != 1 or source.count(QUANT_END) != 1:
        raise ValueError("The frozen shared quantizer changed")
    start = source.index(QUANT_START)
    end = source.index(QUANT_END, start) + len(QUANT_END)
    return source[:start] + DYNAMIC_QUANT + source[end:]


def rename(source):
    return (source.replace("ffn_token_terminal", "ffn_original_shared_terminal")
            .replace("FFN_TOKEN_TERMINAL", "FFN_ORIGINAL_SHARED_TERMINAL")
            .replace("SHARED_W8_GROUPED_QUANT", QUANT_FLAG)
            .replace("kGroupedQuant", "kDynamicPerformanceQuant")
            .replace("shared_w8_grouped_quant", "shared_w8_dynamic_quant")
            .replace("the shared pipeline requires the source grouped-quant contract",
                     "the shared pipeline requires CANN dynamic-performance quantization"))


def prepare(parent, output):
    manifest = json.loads((parent / "manifest.json").read_text())
    if manifest["artifacts"]["ffn_token_terminal.so"] != PARENT_LIBRARY:
        raise ValueError("Use the frozen token-terminal parent")
    if digest(parent / "check.py") != PARENT_CHECKER:
        raise ValueError("The parent checker changed")
    for name, expected in manifest["sources"].items():
        if digest(parent / name) != expected:
            raise ValueError(f"The frozen source changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest["sources"]:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, target)
    original = (parent / NATIVE).read_text()
    changed = bind_activation(original)
    (output / NATIVE).write_text(changed)
    (output / "activation.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile="parent/" + NATIVE, tofile="candidate/" + NATIVE)))
    for path in output.rglob("*"):
        if path.suffix in (".h", ".cpp", ".py", ".sh"):
            path.write_text(rename(path.read_text()))
    tools = output / "candidate_tools"
    tools.mkdir(exist_ok=True)
    for name in ("prepare.py", "build.py", "test_prepare.py", "README.md"):
        shutil.copy2(Path(__file__).with_name(name), tools / name)
    shutil.copy2(parent / "manifest.json", output / "parent-manifest.json")
    report = {
        "parent_library_sha256": PARENT_LIBRARY,
        "parent_checker_sha256": PARENT_CHECKER,
        "scope": "Shared SwiGLU and INT8 quantization for CANN 9.2 tiling key 30013",
        "quantization": "scalar amax/127, guarded reciprocal, Muls, I16 RINT/half NONE/I8 NONE",
        "source_dispatch": "DequantSwigluQuant_int32_high_performance_30013",
        "descriptor_bytes": 408, "public_scratch_count": 33,
        "maximum_resident_ub_bytes": 116480,
        "additional_gm_bytes": 0, "additional_ub_bytes": 0,
        "body_barriers": 5, "final_outer_end_phases": 0,
        "validation_oracle": "Original DecoderLayer post-attention norm and MLP",
        "sources": {
            str(path.relative_to(output)): digest(path)
            for path in sorted(output.rglob("*")) if path.is_file()},
    }
    (output / "source-transforms.json").write_text(
        json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.parent.resolve(), args.output.resolve())
