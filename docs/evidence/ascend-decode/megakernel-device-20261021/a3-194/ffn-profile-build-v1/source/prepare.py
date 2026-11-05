"""Add per-worker phase timestamps to the frozen complete FFN."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


PARENT_LIBRARY = "3673c2a73b2f4b5a4a93b3c2543a4aca35b45773d6a53fc043f27e59a2482655"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one profile seam: {before!r}")
    return source.replace(before, after, 1)


def instrument_bridge(source):
    source = replace_once(source, "float epsilon)",
                          "float epsilon, ffn_trace::Clock& clock)")
    marker = "    device::EndPhase(pipe);"
    if source.count(marker) != 3:
        raise ValueError("Expected the frozen three bridge barriers")
    parts = source.split(marker)
    result = parts[0]
    for index, part in enumerate(parts[1:], 1):
        result += marker + f"\n    clock.Mark({index});" + part
    return result


def instrument_body(source):
    source = replace_once(source, "    __gm__ int64_t* trace\n",
                          "    __gm__ int64_t* trace, ffn_trace::Clock& clock\n")
    return replace_once(source, "    Timing timing;",
                        "    ffn_trace::BodyTrace timing(clock);")


def instrument_kernel(source):
    source = replace_once(source, '#include "contract.h"',
                          '#include "contract.h"\n#include "trace.h"')
    source = replace_once(source, "    float epsilon)\n",
                          "    float epsilon, GM_ADDR trace)\n")
    source = replace_once(source, "    TPipe pipe;",
                          "    TPipe pipe;\n    ffn_trace::Clock clock;\n"
                          "    clock.Mark(0);")
    source = replace_once(source, "bridge, rows, blocks, epsilon);",
                          "bridge, rows, blocks, epsilon, clock);")
    source = replace_once(source, "dispatch, nullptr, shared);",
                          "dispatch, nullptr, clock, shared);")
    marker = "    ffn_bridge::device::EndPhase(pipe);"
    if source.count(marker) != 2:
        raise ValueError("Expected body and epilogue publication boundaries")
    before, middle, after = source.split(marker)
    source = (before + marker + "\n    clock.Mark(11);" + middle
              + marker + "\n    clock.Mark(12);\n"
              "    clock.Store((__gm__ int64_t*)trace, rows, blocks);" + after)
    source = replace_once(source, "int64_t weight_layout, float epsilon)",
                          "int64_t weight_layout, float epsilon, void* trace)")
    if source.count("shared, rows, blocks, epsilon);") != 2:
        raise ValueError("Expected the ND and NZ launch arguments")
    return source.replace("shared, rows, blocks, epsilon);",
                          "shared, rows, blocks, epsilon, (GM_ADDR)trace);")


def instrument_binding(source):
    source = replace_once(source,
        "    return {bridge_scratch, body_scratch, bridge_sync};",
        "    bridge_scratch.push_back(at::empty({3 * blocks, 32},\n"
        "        input.options().dtype(at::kLong)));\n"
        "    return {bridge_scratch, body_scratch, bridge_sync};")
    source = replace_once(source,
        "    const c10::OptionalDeviceGuard device_guard(input.device());",
        "    const c10::OptionalDeviceGuard device_guard(input.device());\n"
        "    TORCH_CHECK(bridge_scratch.size() == 13,\n"
        "                \"profile requires 12 bridge tensors and one trace\");\n"
        "    const auto trace = bridge_scratch.back();\n"
        "    bridge_scratch.pop_back();\n"
        "    bridge::CheckTensor(trace, input, at::kLong,\n"
        "                         {3 * blocks, 32}, \"trace\");\n"
        "    CheckCrossOverlap({trace}, bridge_scratch);\n"
        "    CheckCrossOverlap({trace}, body_scratch);\n"
        "    CheckCrossOverlap({trace}, routed_weights);\n"
        "    CheckCrossOverlap({trace}, shared_weights);\n"
        "    CheckCrossOverlap({trace}, {input, residual, gamma, gate, bias});")
    return replace_once(source, "        static_cast<float>(epsilon));",
                         "        static_cast<float>(epsilon), trace.data_ptr());")


def prepare(parent, output):
    manifest = json.loads((parent / "manifest.json").read_text())
    if manifest["artifacts"]["ffn_composed.so"] != PARENT_LIBRARY:
        raise ValueError("Use the accepted complete-FFN build-v2 parent")
    output.mkdir(parents=True, exist_ok=False)
    for name, expected in manifest["sources"].items():
        source = parent / name
        if digest(source) != expected:
            raise ValueError(f"Frozen parent source changed: {name}")
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    transformations = {
        "bridge/source/ffn_bridge/body_native.h": instrument_bridge,
        "body/source/routed_int4/routed_body_native.h": instrument_body,
        "source/kernel.cpp": instrument_kernel,
        "source/binding.cpp": instrument_binding,
    }
    for name, transform in transformations.items():
        path = output / name
        path.write_text(transform(path.read_text()))
    contract = output / "source/contract.h"
    contract.write_text(replace_once(contract.read_text(),
        "    int64_t weight_layout, float epsilon);",
        "    int64_t weight_layout, float epsilon, void* trace);"))
    for path in output.rglob("*"):
        if path.suffix in (".h", ".cpp"):
            path.write_text(path.read_text().replace("ffn_composed", "ffn_profile"))
    for name in ("trace.h", "prepare.py", "build.py", "decode.py", "README.md"):
        shutil.copy2(Path(__file__).with_name(name), output / "source" / name)
    report = {
        "parent_library_sha256": PARENT_LIBRARY,
        "parent_manifest_sha256": digest(parent / "manifest.json"),
        "trace": {"version": 1, "words": 32, "marks": 13, "stamp_offset": 8},
        "changes": ["SYS_CNT samples at existing phase boundaries",
                    "deferred trace stores after final epilogue publication",
                    "one additional disjoint trace tensor and private namespace"],
        "added_barriers": 0,
        "sources": {str(p.relative_to(output)): digest(p)
                    for p in sorted(output.rglob("*")) if p.is_file()},
    }
    (output / "source-transforms.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copy2(parent / "manifest.json", output / "parent-manifest.json")
    shutil.copy2(parent / "commands.jsonl", output / "parent-commands.jsonl")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.parent.resolve(), args.output.resolve())
