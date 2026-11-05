"""Freeze a GU-only constant-tiling variant of the terminal FFN."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT_LIBRARY = "276352762139244a1ddfc8184065a856c2ae75ee2ccad5961af2af93f0dd5e00"
PARENT_CHECKER = "c5b5586376f9f412b1f9da2789089766a37231a682c8ef16c3c0f6f384e56d80"
BODY = "body/source/routed_int4/routed_body_native.h"
OLD_ALIAS = """#ifdef ROUTED_GU_MDL
template <CubeFormat format>
using GateProjection = Matmul<APacked,
    MatmulType<TPosition::GM, format, int4b_t, false>, CProduct, BiasNone,
    CFG_MDL>;
#else
template <CubeFormat format>
using GateProjection = Projection<format>;
#endif
"""
NEW_ALIAS = """#if defined(ROUTED_GU_MDL) || defined(ROUTED_GU_PAD16) || defined(ROUTED_FIXPIPE_HALF)
#error "GU constant tiling preserves the parent INT32 Norm geometry"
#endif
template <CubeFormat format>
using GateProjection = Matmul<APacked,
    MatmulType<TPosition::GM, format, int4b_t, false>, CProduct, BiasNone,
    ffn_gu_static::kGateTiling>;
"""
INCLUDE = '#include "../../../source/gu_static_tiling.h"\n'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one frozen source seam: {before!r}")
    return source.replace(before, after, 1)


def rename(source):
    return source.replace("ffn_token_terminal", "ffn_gu_static").replace(
        "FFN_TOKEN_TERMINAL", "FFN_GU_STATIC")


def bind_body(source):
    source = replace_once(source, OLD_ALIAS, NEW_ALIAS)
    return replace_once(source, '#include "lib/matmul_intf.h"\n',
                        '#include "lib/matmul_intf.h"\n' + INCLUDE)


def prepare(parent, output):
    manifest = json.loads((parent / "manifest.json").read_text())
    if manifest["artifacts"]["ffn_token_terminal.so"] != PARENT_LIBRARY:
        raise ValueError("Use the frozen token-terminal parent")
    if digest(parent / "check.py") != PARENT_CHECKER:
        raise ValueError("The frozen numerical harness changed")
    for name, expected in manifest["sources"].items():
        if digest(parent / name) != expected:
            raise ValueError(f"The frozen source changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest["sources"]:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, target)
    original = (parent / BODY).read_text()
    changed = bind_body(original)
    (output / BODY).write_text(changed)
    (output / "gu-static.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile="parent/" + BODY, tofile="candidate/" + BODY)))
    for path in output.rglob("*"):
        if path.suffix in (".h", ".cpp"):
            path.write_text(rename(path.read_text()))
    shutil.copy2(Path(__file__).with_name("gu_static_tiling.h"),
                 output / "source/gu_static_tiling.h")
    (output / "check.py").write_text(rename((parent / "check.py").read_text()))
    for name in ("prepare.py", "build.py", "test_prepare.py", "README.md",
                 "gu_static_tiling.h", "gate-tiling.jsonl"):
        shutil.copy2(Path(__file__).with_name(name),
                     output / "candidate_tools" / name)
    shutil.copy2(parent / "manifest.json", output / "parent-manifest.json")
    report = {
        "parent_library_sha256": PARENT_LIBRARY,
        "parent_checker_sha256": PARENT_CHECKER,
        "scope": "GU constant tiling only; unchanged Norm scheduler and runtime expert rows",
        "descriptor_bytes": 408,
        "public_scratch_count": 33,
        "maximum_resident_ub_bytes": 116480,
        "additional_gm_bytes": 0,
        "additional_ub_bytes": 0,
        "body_barriers": 5,
        "gate_scheduler": "CFG_NORM",
        "runtime_gate_shape": "[2*expert_count, 6144] x [6144, 256]",
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
