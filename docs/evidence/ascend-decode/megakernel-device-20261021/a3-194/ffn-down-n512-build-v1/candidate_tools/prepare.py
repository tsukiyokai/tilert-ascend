"""Widen only the validated FFN's Down host tiling to N512."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT_LIBRARY = "381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2"
OLD_NAMESPACE = "ffn_original_shared_gu_static"
NAMESPACE = "ffn_down_n512"
TILING = "body/source/routed_int4/routed_int4_tiling.cpp"
CONTRACT = "down_tiling_contract.h"
TOOLS = ("prepare.py", "build.py", "test_prepare.py", "tiling_probe.cpp",
         "README.md", CONTRACT)
SET_SPLIT = """    const bool is_down = columns == routed_int4::kHidden &&
                         inner == routed_int4::kIntermediate;
    if (is_down) {
        if (rows != routed_int4::kMaxStackedRows || width != routed_int4::kHidden) {
            return -2;
        }
        const int split_status = tiling.SetFixSplit(-1, 512, -1);
        if (split_status != 0) {
            return split_status;
        }
    }

"""
CHECK_TILING = """    if (is_down && !ffn_down_n512::IsExpectedDownTiling(cube)) {
        return -3;
    }
"""
INCLUDE = '#include "../../../source/down_tiling_contract.h"\n'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one frozen source seam: {before!r}")
    return source.replace(before, after, 1)


def rename(source):
    return source.replace(OLD_NAMESPACE, NAMESPACE).replace(
        OLD_NAMESPACE.upper(), NAMESPACE.upper())


def transform(source):
    source = once(source, '#include "routed_int4_kernel.h"\n',
                  '#include "routed_int4_kernel.h"\n' + INCLUDE)
    source = once(source, "    AscendC::tiling::TCubeTiling cube;\n",
                  SET_SPLIT + "    AscendC::tiling::TCubeTiling cube;\n")
    return once(source, "    std::memcpy(buffer, &cube, sizeof(cube));\n",
                CHECK_TILING + "    std::memcpy(buffer, &cube, sizeof(cube));\n")


def prepare(parent, output):
    manifest = json.loads((parent / "manifest.json").read_text())
    if manifest["artifacts"][OLD_NAMESPACE + ".so"] != PARENT_LIBRARY:
        raise ValueError("Use the validated original-shared static-GU parent")
    for name, expected in manifest["sources"].items():
        if digest(parent / name) != expected:
            raise ValueError(f"The frozen source changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest["sources"]:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, target)
    original = (parent / TILING).read_text()
    changed = transform(original)
    (output / TILING).write_text(changed)
    (output / "down-n512.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile="parent/" + TILING, tofile="candidate/" + TILING)))
    for path in output.rglob("*"):
        if path.suffix in (".h", ".cpp"):
            path.write_text(rename(path.read_text()))
    shutil.copy2(Path(__file__).with_name(CONTRACT), output / "source" / CONTRACT)
    for name in TOOLS:
        shutil.copy2(Path(__file__).with_name(name), output / "candidate_tools" / name)
    shutil.copy2(parent / "manifest.json", output / "parent-manifest.json")
    report = {
        "parent_library_sha256": PARENT_LIBRARY,
        "scope": "Down host SetFixSplit(-1, 512, -1); device math and schedule unchanged",
        "validation_oracle": "Original DecoderLayer post-attention norm and MLP",
        "descriptor_bytes": 408, "public_scratch_count": 33,
        "maximum_resident_ub_bytes": 116480,
        "additional_gm_bytes": 0, "additional_ub_bytes": 0,
        "body_barriers": 5, "final_outer_end_phases": 0,
        "down_tiling": {"baseM": 64, "baseN": 512, "baseK": 128},
        "down_resources_bytes": {
            "L0A": 8192, "L0B": 65536, "L0C": 131072, "L1": 397312},
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
