"""Overlap streamed down tiles with route dequantization in a frozen FFN parent."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT_LIBRARY = "3673c2a73b2f4b5a4a93b3c2543a4aca35b45773d6a53fc043f27e59a2482655"
BODY = "body/source/routed_int4/routed_body_native.h"
TILING = "body/source/routed_int4/routed_int4_tiling.cpp"
BODY_MARKER = "template <CubeFormat weight_format, bool from_dispatch>\n"
PHASE_FOUR = "    timing.Barrier(3);\n    timing.Begin(4);\n"
PHASE_FIVE = "    timing.Barrier(4);\n    timing.Begin(5);\n"
PHASE_SIX = "    timing.Barrier(5);\n    timing.Begin(6);\n"
DOWN_CALL = (
    "            Project(down_mm, Nibbles(middle, 2 * begin * kIntermediate),\n"
    "                    Nibbles(down, expert * kHidden * kIntermediate),\n"
    "                    wide + 2 * begin * kHidden,\n"
    "                    down_scale + expert * kHidden, kHidden, kIntermediate,\n"
    "                    counts.GetValue(expert));\n"
)
OVERLAP_CALL = (
    "            ProjectDownOverlap(down_mm,\n"
    "                    Nibbles(middle, 2 * begin * kIntermediate),\n"
    "                    Nibbles(down, expert * kHidden * kIntermediate),\n"
    "                    wide + 2 * begin * kHidden, kHidden, kIntermediate,\n"
    "                    counts.GetValue(expert), down_sequence);\n"
)
TILING_GUARD = (
    "    if (columns == ffn_down_overlap::kDownHidden\n"
    "            && inner == ffn_down_overlap::kDownInner\n"
    "            && !ffn_down_overlap::IsDownTiling(cube.baseM, cube.baseN,\n"
    "                cube.baseK, cube.singleCoreM, cube.singleCoreN, cube.singleCoreK)) {\n"
    "        return -2;\n"
    "    }\n"
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one frozen down seam: {before!r}")
    return source.replace(before, after, 1)


def rename(source):
    return source.replace("ffn_composed", "ffn_down_overlap").replace(
        "FFN_COMPOSED", "FFN_DOWN_OVERLAP")


def bind_body(source):
    source = replace_once(source, '#include "routed_int4_kernel.h"',
        '#include "routed_int4_kernel.h"\n#include "../../../source/overlap_contract.h"')
    source = replace_once(source, "}  // namespace\n\n" + BODY_MARKER,
        '#include "../../../source/overlap_native.h"\n\n'
        "}  // namespace\n\n" + BODY_MARKER)
    start = source.index(PHASE_FOUR) + len(PHASE_FOUR)
    end = source.index(PHASE_FIVE, start)
    phase = source[start:end]
    phase = replace_once(phase, "        int64_t active = 0;\n",
        "        int64_t down_sequence = 0;\n        int64_t active = 0;\n")
    phase = replace_once(phase, DOWN_CALL, OVERLAP_CALL)
    phase = replace_once(phase, "        Publish();\n    }\n",
        "        DrainDownCredits(down_sequence);\n        Publish();\n    }\n")
    phase += (
        "    if ASCEND_IS_AIV {\n"
        "        ConsumeDown(held, wide, down_bias, down_scale, middle_scale,\n"
        "                    route_output, experts, tokens, blocks);\n"
        "        Publish();\n    }\n"
    )
    source = source[:start] + phase + source[end:]
    start = source.index(PHASE_FIVE) + len(PHASE_FIVE)
    end = source.index(PHASE_SIX, start)
    return source[:start] + source[end:]


def bind_tiling(source):
    source = replace_once(source, '#include "routed_int4_kernel.h"',
        '#include "routed_int4_kernel.h"\n#include "../../../source/overlap_contract.h"')
    return replace_once(source, "    std::memcpy(buffer, &cube, sizeof(cube));\n",
        TILING_GUARD + "    std::memcpy(buffer, &cube, sizeof(cube));\n")


def prepare(parent, output):
    manifest = json.loads((parent / "manifest.json").read_text())
    if manifest["artifacts"]["ffn_composed.so"] != PARENT_LIBRARY:
        raise ValueError("Use the validated uninstrumented composed build-v2 parent")
    for name, expected in manifest["sources"].items():
        if digest(parent / name) != expected:
            raise ValueError(f"Frozen parent source changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest["sources"]:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, destination)
    originals = {name: (output / name).read_text() for name in (BODY, TILING)}
    (output / BODY).write_text(bind_body(originals[BODY]))
    (output / TILING).write_text(bind_tiling(originals[TILING]))
    patch = []
    for name, before in originals.items():
        patch.extend(difflib.unified_diff(before.splitlines(keepends=True),
            (output / name).read_text().splitlines(keepends=True),
            fromfile=f"parent/{name}", tofile=f"candidate/{name}"))
    (output / "down-overlap.patch").write_text("".join(patch))
    for path in output.rglob("*"):
        if path.suffix in (".h", ".cpp"):
            path.write_text(rename(path.read_text()))
    for name in ("overlap_contract.h", "overlap_native.h", "prepare.py", "build.py",
                 "test_contract.py", "test_protocol.py", "ownership_probe.cpp", "README.md",
                 "source-audit.json"):
        shutil.copy2(Path(__file__).with_name(name), output / "source" / name)
    checker = parent / "harness-v1/check.py"
    (output / "check.py").write_text(checker.read_text().replace(
        "torch.ops.tilert_ffn_composed", "torch.ops.tilert_ffn_down_overlap"))
    shutil.copy2(parent / "manifest.json", output / "parent-manifest.json")
    report = {
        "parent_library_sha256": PARENT_LIBRARY,
        "parent_manifest_sha256": digest(parent / "manifest.json"),
        "public_scratch_count": 32,
        "additional_gm_bytes": 0,
        "additional_ub_bytes": 0,
        "maximum_resident_ub_bytes": 112352,
        "body_barriers": 6,
        "outer_end_phases": 2,
        "down_base_shape": [64, 256, 128],
        "down_single_shape": [64, 6144, 128],
        "credit_window": 4,
        "ready_flag": 8,
        "free_flag": 9,
        "counter_mode": 2,
        "counter_tail": "All ready/free credits consumed before barrier 4",
        "diagnostic_outputs": "Original full down GM and all 30 retained boundaries",
        "system_workspace_behavior": "Original calls and initial bridge selection retained",
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
