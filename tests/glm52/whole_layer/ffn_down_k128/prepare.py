"""Keep N1024 device code and adapt six Down fields from the installed API."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT_LIBRARY = "bbd52e425a1e94bb199e464e3bf6001c4420e8238afe993e8741e38af50b77ef"
OLD_NAMESPACE = "ffn_down_n1024"
NAMESPACE = "ffn_down_k128"
TILING = "body/source/routed_int4/routed_int4_tiling.cpp"
CONTRACT = "source/down_tiling_contract.h"
TOOLS = ("prepare.py", "build.py", "test_prepare.py", "tiling_probe.cpp",
         "down_tiling_contract.h", "down_plan.h", "static_down_plan.cpp",
         "expected-tilings.json", "README.md")
OLD_CHECK = '''    if (is_down && !ffn_down_n1024::IsExpectedDownTiling(cube)) {
        return -3;
    }
'''
NEW_CHECK = '''    if (is_down) {
        if (!ffn_down_n1024::IsParentDownTiling(cube)) {
            return -3;
        }
        const auto& plan = weight_layout == routed_int4::kWeightNZ
            ? ffn_down_n1024_plan_nz : ffn_down_n1024_plan_nd;
        ffn_down_n1024::ApplyDownPlan(cube, plan);
        if (!ffn_down_n1024::IsExpectedDownTiling(cube)) {
            return -4;
        }
    }
'''
PLAN_INCLUDE = '#include "../../../source/down_plan.h"\n'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rename(source):
    return source.replace(OLD_NAMESPACE, NAMESPACE).replace(
        OLD_NAMESPACE.upper(), NAMESPACE.upper())


def transform(source):
    anchor = '#include "../../../source/down_tiling_contract.h"\n'
    if source.count(OLD_CHECK) != 1 or source.count(anchor) != 1:
        raise ValueError("The parent must have one complete Down record check")
    return source.replace(anchor, anchor + PLAN_INCLUDE, 1).replace(
        OLD_CHECK, NEW_CHECK, 1)


def prepare(parent, output):
    manifest = json.loads((parent / "manifest.json").read_text())
    if manifest["artifacts"][OLD_NAMESPACE + ".so"] != PARENT_LIBRARY:
        raise ValueError("Use the validated bbd Down N1024 parent")
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
    (output / "down-k128.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile="parent/" + TILING, tofile="candidate/" + TILING)))
    for path in output.rglob("*"):
        if path.suffix in (".h", ".cpp"):
            path.write_text(rename(path.read_text()))
    for name in ("down_tiling_contract.h", "down_plan.h", "static_down_plan.cpp"):
        shutil.copy2(Path(__file__).with_name(name), output / "source" / name)
    for name in TOOLS:
        shutil.copy2(Path(__file__).with_name(name), output / "candidate_tools" / name)
    shutil.copy2(parent / "manifest.json", output / "parent-manifest.json")

    unchanged = {}
    for name in manifest["sources"]:
        if (Path(name).suffix not in (".cpp", ".h") or
                name.startswith("candidate_tools/") or name in (TILING, CONTRACT)):
            continue
        if (output / name).read_text() != rename((parent / name).read_text()):
            raise ValueError(f"A non-tiling source changed: {name}")
        unchanged[name] = digest(parent / name)
    report = {
        "parent_library_sha256": PARENT_LIBRARY,
        "scope": "Six named Down host fields from installed GetMatmulApiTiling",
        "validation_oracle": "Original DecoderLayer post-attention norm and MLP",
        "descriptor_bytes": 408,
        "public_scratch_count": 33,
        "peak_ub_bytes": 195584,
        "routed_body_ub_bytes": 116480,
        "additional_gm_bytes": 0,
        "additional_ub_bytes": 0,
        "body_barriers": 5,
        "final_outer_end_phases": 0,
        "dynamic_expert_rows": "SetSingleShape(2 * count, 6144, 128), count in 1..32",
        "down_tiling": {"baseM": 32, "baseN": 1024, "baseK": 128},
        "down_resources_bytes": {
            "L0A_operand": 2048, "L0B_operand": 65536,
            "L0C": 131072, "L1": 397312},
        "effective_l0_a_b_buffers": 1,
        "down_tiling_bytes": 200,
        "down_tiling_sha256":
            "c2b73af9d605ad8e64ea551f288d188a0cf636828a5a0471ea7c63d616f50999",
        "named_field_changes": {
            "baseK": [64, 128], "depthA1": [4, 2], "depthB1": [12, 6],
            "stepKa": [2, 1], "stepKb": [2, 1], "dbL0B": [2, 1]},
        "normalized_unchanged_cpp_headers": unchanged,
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
