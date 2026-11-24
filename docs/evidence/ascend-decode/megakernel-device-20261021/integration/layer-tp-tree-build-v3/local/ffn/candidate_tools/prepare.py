"""Add the verified GU static tiling to the original shared-quantization FFN."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT_LIBRARY = "c39acf92728c8a3f09ee0b66988bed4cf0e312a15285a68c02d2594969a348af"
GATE_LIBRARY = "05b3427aa29630c9af16ea829e2ecec8833d31a32615e6a052f6f4e9867ca6b2"
COMMON_PARENT = "276352762139244a1ddfc8184065a856c2ae75ee2ccad5961af2af93f0dd5e00"
GATE_HEADER = "6d8021fc7ce3832d5064c1b7cf73487c554ac39632c90d02e3dcc1dff30a657f"
SHARED_HEADER = "c6fdab0eb2e2b36644c8c520b6ee060db6c5dc54870f66ca27b919e3636e3760"
BODY = "body/source/routed_int4/routed_body_native.h"
SHARED = "body/source/shared_w8/shared_w8_native.h"
NAMESPACE = "ffn_original_shared_gu_static"
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
GATE_ALIAS = """#if defined(ROUTED_GU_MDL) || defined(ROUTED_GU_PAD16) || defined(ROUTED_FIXPIPE_HALF)
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


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one frozen source seam: {before!r}")
    return source.replace(before, after, 1)


def rename(source):
    return source.replace("ffn_original_shared_terminal", NAMESPACE).replace(
        "FFN_ORIGINAL_SHARED_TERMINAL", NAMESPACE.upper())


def verified_manifest(path, library, expected):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["artifacts"][library] != expected:
        raise ValueError(f"Unexpected frozen parent library: {library}")
    if manifest["parent_library_sha256"] != COMMON_PARENT:
        raise ValueError("The two branches must have the same token-terminal parent")
    for name, expected_hash in manifest["sources"].items():
        if digest(path / name) != expected_hash:
            raise ValueError(f"The frozen source changed: {path / name}")
    return manifest


def verify_common_body(parent, gate):
    original = (parent / BODY).read_text()
    specialized = (gate / BODY).read_text()
    restored = once(specialized, GATE_ALIAS, OLD_ALIAS)
    restored = once(restored, INCLUDE, "")
    restored = restored.replace("ffn_gu_static", "ffn_token_terminal").replace(
        "FFN_GU_STATIC", "FFN_TOKEN_TERMINAL")
    common = original.replace("ffn_original_shared_terminal", "ffn_token_terminal").replace(
        "FFN_ORIGINAL_SHARED_TERMINAL", "FFN_TOKEN_TERMINAL")
    if common != restored:
        raise ValueError("The shared-math and GU branches do not have identical GU bodies")
    return hashlib.sha256(common.encode()).hexdigest()


def prepare(parent, gate, output):
    manifest = verified_manifest(
        parent, "ffn_original_shared_terminal.so", PARENT_LIBRARY)
    verified_manifest(gate, "ffn_gu_static.so", GATE_LIBRARY)
    common_body = verify_common_body(parent, gate)
    if digest(gate / "source/gu_static_tiling.h") != GATE_HEADER:
        raise ValueError("The verified static tiling changed")
    if digest(parent / SHARED) != SHARED_HEADER:
        raise ValueError("The original shared-activation arithmetic changed")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest["sources"]:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, target)
    original = (parent / BODY).read_text()
    changed = once(original, OLD_ALIAS, GATE_ALIAS.replace("ffn_gu_static", NAMESPACE))
    changed = once(changed, '#include "lib/matmul_intf.h"\n',
                   '#include "lib/matmul_intf.h"\n' + INCLUDE)
    (output / BODY).write_text(changed)
    (output / "gu-static.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile="parent/" + BODY, tofile="candidate/" + BODY)))
    for path in output.rglob("*"):
        if path.suffix in (".h", ".cpp"):
            path.write_text(rename(path.read_text()))
    header = (gate / "source/gu_static_tiling.h").read_text()
    header = header.replace("ffn_gu_static", NAMESPACE).replace(
        "FFN_GU_STATIC", NAMESPACE.upper())
    (output / "source/gu_static_tiling.h").write_text(header)
    # The inherited checker uses the historical shared oracle. Device acceptance
    # uses the unchanged original DecoderLayer norm/MLP harness instead.
    (output / "check.py").rename(output / "parent-check.py")
    for name in ("prepare.py", "build.py", "test_prepare.py", "README.md"):
        shutil.copy2(Path(__file__).with_name(name), output / "candidate_tools" / name)
    shutil.copy2(parent / "manifest.json", output / "parent-manifest.json")
    shutil.copy2(gate / "manifest.json", output / "gu-manifest.json")
    report = {
        "parent_library_sha256": PARENT_LIBRARY,
        "gu_library_sha256": GATE_LIBRARY,
        "common_parent_library_sha256": COMMON_PARENT,
        "common_body_sha256": common_body,
        "scope": "GU static tiling with the original CANN key30013 shared arithmetic",
        "validation_oracle": "Original DecoderLayer post-attention norm and MLP",
        "inherited_checker": "parent-check.py; historical shared oracle, not an acceptance gate",
        "descriptor_bytes": 408, "public_scratch_count": 33,
        "maximum_resident_ub_bytes": 116480,
        "additional_gm_bytes": 0, "additional_ub_bytes": 0,
        "body_barriers": 5, "final_outer_end_phases": 0,
        "runtime_gate_shape": "[2*expert_count, 6144] x [6144, 256]",
        "gate_scheduler": "CFG_NORM",
        "shared_header_sha256": digest(output / SHARED),
        "sources": {str(path.relative_to(output)): digest(path)
                    for path in sorted(output.rglob("*")) if path.is_file()},
    }
    (output / "source-transforms.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("gate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.parent.resolve(), args.gate.resolve(), args.output.resolve())
