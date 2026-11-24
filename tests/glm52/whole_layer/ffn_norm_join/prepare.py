"""Retain the FFN caller's norm publication and remove its preceding join."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT_LIBRARY = (
    "381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2"
)
NAMESPACE = "ffn_norm_join"
NORM = "bridge/source/norm_native.h"
CALLER = "bridge/source/ffn_bridge/body_native.h"
TOOLS = ("prepare.py", "build.py", "test_prepare.py", "README.md")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rename(source):
    return source.replace("ffn_original_shared_gu_static", NAMESPACE).replace(
        "FFN_ORIGINAL_SHARED_GU_STATIC", NAMESPACE.upper())


def transform(source):
    tail = "    SyncAll<false>();\n}\n\n}\n#endif\n"
    if source.count(tail) != 1 or source.count("SyncAll<false>()") != 1:
        raise ValueError("The frozen norm completion seam changed")
    return source.replace(tail, "}\n\n}\n#endif\n", 1)


def prepare(parent, output):
    manifest = json.loads((parent / "manifest.json").read_text())
    if manifest["artifacts"]["ffn_original_shared_gu_static.so"] != PARENT_LIBRARY:
        raise ValueError("Use the validated original-shared static-GU FFN")
    for name, expected in manifest["sources"].items():
        if digest(parent / name) != expected:
            raise ValueError(f"The frozen source changed: {name}")
    caller = (parent / CALLER).read_text()
    seam = (
        "        (__gm__ float*)a.values[kRstd], nullptr, nullptr, "
        "rows, blocks, epsilon);\n    device::EndPhase(pipe);"
    )
    calls = caller.count("whole_layer_norm::device::Forward(")
    if caller.count(seam) != 1 or calls != 1:
        raise ValueError("FFN norm must publish through its immediate caller")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest["sources"]:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, target)
    original = (parent / NORM).read_text()
    changed = transform(original)
    (output / NORM).write_text(changed)
    (output / "norm-join.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile="parent/" + NORM, tofile="candidate/" + NORM)))
    for path in output.rglob("*"):
        if path.suffix in (".h", ".cpp"):
            path.write_text(rename(path.read_text()))
    for name in TOOLS:
        shutil.copy2(Path(__file__).with_name(name), output / "candidate_tools" / name)
    shutil.copy2(parent / "manifest.json", output / "parent-manifest.json")
    report = {
        "parent_library_sha256": PARENT_LIBRARY,
        "scope": ("Remove the private norm's final SyncAll; "
                  "retain its caller's immediate EndPhase"),
        "validation_oracle": "Original DecoderLayer post-attention norm and MLP",
        "norm_internal_joins": 0,
        "norm_completion": ("Caller PIPE_ALL, DDR publication, "
                            "mixed SyncAll, TPipe reset"),
        "descriptor_bytes": 408, "public_scratch_count": 33,
        "maximum_resident_ub_bytes": 195584,
        "routed_body_ub_bytes": 116480,
        "additional_gm_bytes": 0, "additional_ub_bytes": 0,
        "body_barriers": 5, "final_outer_end_phases": 0,
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
