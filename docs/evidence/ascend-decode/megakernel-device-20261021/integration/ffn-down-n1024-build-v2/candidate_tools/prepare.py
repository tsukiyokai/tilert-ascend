"""Change only the c90 FFN Down tiling to M32/N1024/K64."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT_LIBRARY = "c90b3666a1697070eda19d5b3e133e9ed58f36fe429abefceae6ffe102a6a032"
OLD_NAMESPACE = "ffn_down_n512"
NAMESPACE = "ffn_down_n1024"
TILING = "body/source/routed_int4/routed_int4_tiling.cpp"
CONTRACT = "source/down_tiling_contract.h"
OLD_SPLIT = "tiling.SetFixSplit(-1, 512, -1)"
NEW_SPLIT = "tiling.SetFixSplit(32, 1024, 64)"
TOOLS = ("prepare.py", "build.py", "test_prepare.py", "tiling_probe.cpp",
         "down_tiling_contract.h", "expected-tilings.json", "README.md")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rename(source):
    return source.replace(OLD_NAMESPACE, NAMESPACE).replace(
        OLD_NAMESPACE.upper(), NAMESPACE.upper())


def transform(source):
    if source.count(OLD_SPLIT) != 1:
        raise ValueError("The parent must have one Down SetFixSplit call")
    return source.replace(OLD_SPLIT, NEW_SPLIT, 1)


def prepare(parent, output):
    manifest = json.loads((parent / "manifest.json").read_text())
    if manifest["artifacts"][OLD_NAMESPACE + ".so"] != PARENT_LIBRARY:
        raise ValueError("Use the validated c90 Down N512 parent")
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
    (output / "down-n1024.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile="parent/" + TILING, tofile="candidate/" + TILING)))
    for path in output.rglob("*"):
        if path.suffix in (".h", ".cpp"):
            path.write_text(rename(path.read_text()))
    shutil.copy2(Path(__file__).with_name("down_tiling_contract.h"),
                 output / CONTRACT)
    for name in TOOLS:
        shutil.copy2(Path(__file__).with_name(name),
                     output / "candidate_tools" / name)
    shutil.copy2(parent / "manifest.json", output / "parent-manifest.json")
    report = {
        "parent_library_sha256": PARENT_LIBRARY,
        "scope": "Down host SetFixSplit(32, 1024, 64); original device source retained",
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
        "down_tiling": {"baseM": 32, "baseN": 1024, "baseK": 64},
        "down_resources_bytes": {
            "L0A": 2048, "L0B": 65536, "L0C": 131072, "L1": 397312},
        "down_tiling_bytes": 200,
        "down_tiling_sha256":
            "c207230aafb11e2c2f825cd9a2ea70147ee7733046b8b2957593c9c2bc3e8009",
        "sources": {str(path.relative_to(output)): digest(path)
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
