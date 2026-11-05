"""Overlap complete expert GU products with their paired AIV activation."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT_LIBRARY = "381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2"
NAMESPACE = "ffn_gu_static_ready"
BODY = "body/source/routed_int4/routed_body_native.h"
TOOLS = ("prepare.py", "build.py", "test_prepare.py", "README.md",
         "gu_ready_contract.h", "gu_ready_native.h")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one frozen source seam: {before!r}")
    return source.replace(before, after, 1)


def rename(source):
    return source.replace("ffn_original_shared_gu_static", NAMESPACE).replace(
        "FFN_ORIGINAL_SHARED_GU_STATIC", NAMESPACE.upper())


def transform(source):
    source = once(source, '#include "lib/matmul_intf.h"\n',
                  '#include "lib/matmul_intf.h"\n'
                  '#include "../../../source/gu_ready_contract.h"\n')
    source = once(source, '#include "../../../source/split_native.h"\n',
                  '#include "../../../source/split_native.h"\n'
                  '#include "../../../source/gu_ready_native.h"\n')
    source = once(source,
        "    // Every core reaches every barrier, including empty expert owners.\n",
        "    const bool overlap_gate = active_count >= arguments.blocks;\n\n"
        "    // Every core reaches every barrier, including empty expert owners.\n")
    source = once(source, "    shared.Activate();\n#endif\n",
        "    shared.Activate();\n#endif\n"
        "    if ASCEND_IS_AIV {\n"
        "        if (overlap_gate) {\n"
        "            ConsumeGate(held, arguments, active_count);\n"
        "        }\n"
        "    }\n")
    source = once(source, "        gate_mm.Init(&gate_cube, &pipe);\n",
                  "        gate_mm.Init(&gate_cube, &pipe);\n"
                  "        int64_t gate_sequence = 0;\n")
    source = once(source, """                    counts.GetValue(expert));
        }
        Publish();
    }
    timing.Barrier(2);
""", """                    counts.GetValue(expert));
            if (overlap_gate) {
                PublishGate(gate_sequence);
            }
        }
        if (overlap_gate) {
            DrainGate(gate_sequence);
        }
        Publish();
    }
    timing.Barrier(2);
""")
    begin = source.index("    if ASCEND_IS_AIV {", source.index("    timing.Barrier(2);"))
    end = source.index("    timing.Barrier(3);", begin)
    original = source[begin:end]
    if not original.endswith("    }\n"):
        raise ValueError("The original balanced activation scope changed")
    fallback = once(original, "    if ASCEND_IS_AIV {\n",
                    "    if ASCEND_IS_AIV {\n        if (!overlap_gate) {\n")
    fallback = fallback[:-6] + "        }\n    }\n"
    return source[:begin] + fallback + source[end:]


def prepare(parent, output):
    manifest = json.loads((parent / "manifest.json").read_text())
    if manifest["artifacts"]["ffn_original_shared_gu_static.so"] != PARENT_LIBRARY:
        raise ValueError("Use the validated original-shared static-GU FFN")
    for name, expected in manifest["sources"].items():
        if digest(parent / name) != expected:
            raise ValueError(f"The frozen source changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest["sources"]:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, target)
    original = (parent / BODY).read_text()
    changed = transform(original)
    (output / BODY).write_text(changed)
    (output / "gu-ready.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile="parent/" + BODY, tofile="candidate/" + BODY)))
    for path in output.rglob("*"):
        if path.suffix in (".h", ".cpp"):
            path.write_text(rename(path.read_text()))
    for name in ("gu_ready_contract.h", "gu_ready_native.h"):
        shutil.copy2(Path(__file__).with_name(name), output / "source" / name)
    if not (output / "parent-check.py").is_file():
        raise ValueError("The static-GU parent must retain its historical checker")
    for name in TOOLS:
        shutil.copy2(Path(__file__).with_name(name), output / "candidate_tools" / name)
    shutil.copy2(parent / "manifest.json", output / "parent-manifest.json")
    report = {
        "parent_library_sha256": PARENT_LIBRARY,
        "scope": "Per-expert GU publication and paired activation before unchanged join 2",
        "validation_oracle": "Original DecoderLayer post-attention norm and MLP",
        "overlap_condition": "active_count >= blocks; otherwise original balanced activation",
        "ready_flag": 8, "free_flag": 9, "mode": 2, "credits": 4,
        "descriptor_bytes": 408, "public_scratch_count": 33,
        "maximum_resident_ub_bytes": 116480,
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
