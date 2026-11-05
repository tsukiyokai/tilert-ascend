"""Extract the selected routed/shared body with caller-owned pipe state."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


PARENT_KERNEL = (
    "186b700ccf392e8bf8232679b14d6b21f3b63566077f66e7eae88f9c866fa695"
)
INITIALIZE = """    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);
    SetSysWorkspace(workspace);
    TPipe pipe;
"""
CALL = """    routed_body::Forward<weight_format, from_dispatch>(
        pipe, rows, gate_up, gu_bias, gu_scale, down, down_bias, down_scale,
        token_scale, group_count, group_start, source_row, source_slot,
        row_weight, stacked, partial, middle, middle_scale, wide, route_output,
        out, gu_tiling, down_tiling, workspace, experts, tokens, blocks,
        dispatch_input, trace
#ifdef ROUTED_SHARED_PIPELINE
        , shared_input
#endif
        );
"""
RENAMES = {
    "routed_shared_front_build_routed_tiling":
        "routed_shared_body_build_routed_tiling",
    "routed_shared_front_build_tiling": "routed_shared_body_build_tiling",
    "routed_shared_front_launch": "routed_shared_body_launch",
    "routed_shared_front_fused": "routed_shared_body_fused",
    "tilert_pipeline_front": "tilert_pipeline_body",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(source):
    kernel_start = source.index(
        "template <CubeFormat weight_format, bool from_dispatch>\n"
        "__global__ __mix__(1, 2) __schedmode__(1)\n"
    )
    body_start = source.index("\n{\n", kernel_start) + 3
    suffix_start = source.index("\n#ifndef ROUTED_ENABLE_PROFILE\n", body_start)
    body_end = suffix_start - 2
    if source[body_end:suffix_start] != "}\n":
        raise ValueError("Unexpected selected kernel boundary")
    declaration = source[kernel_start:body_start]
    body = source[body_start:body_end]
    if not body.startswith(INITIALIZE) or body.count(INITIALIZE) != 1:
        raise ValueError("Unexpected selected kernel initialization")
    numerical = body[len(INITIALIZE):]
    callable_declaration = declaration.replace(
        "__global__ __mix__(1, 2) __schedmode__(1)\n", ""
    ).replace(
        "__aicore__ void routed_int4_fused(",
        "__aicore__ inline void Forward("
    ).replace("    uint64_t sync_base,", "    TPipe& pipe,")
    using_start = source.index("using namespace AscendC;\n")
    header = (
        "#ifndef TILERT_ROUTED_BODY_NATIVE_H\n"
        "#define TILERT_ROUTED_BODY_NATIVE_H\n\n"
        + source[:using_start]
        + "namespace routed_body {\n\n"
        + source[using_start:kernel_start]
        + callable_declaration + numerical + "}\n\n"
        + "}  // namespace routed_body\n\n#endif\n"
    )
    wrapper = (
        '#include "routed_body_native.h"\n\n'
        "using namespace AscendC;\n"
        "using namespace routed_int4;\n"
        "using namespace routed_body;\n\n"
        + declaration + INITIALIZE + CALL + "}\n"
        + source[suffix_start:]
    )
    if header.count(numerical) != 1:
        raise ValueError("The callable body must preserve every original statement")
    return header, wrapper, hashlib.sha256(numerical.encode()).hexdigest()


def prepare(parent, output):
    manifest = json.loads((parent / "build-manifest.json").read_text())
    if (manifest["namespace"] != "tilert_pipeline_front"
            or manifest["own_sources"]["routed_int4/routed_int4_kernel.cpp"]
            != PARENT_KERNEL):
        raise ValueError("Use the selected frozen routed/shared parent")
    output.mkdir(parents=True, exist_ok=False)
    for name, expected in manifest["own_sources"].items():
        source = parent / "source" / name
        if digest(source) != expected:
            raise ValueError(f"Parent source changed: {name}")
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    own = output / "source/routed_int4"
    kernel = own / "routed_int4_kernel.cpp"
    header, wrapper, body_hash = extract(kernel.read_text())
    (own / "routed_body_native.h").write_text(header)
    kernel.write_text(wrapper)
    for name in ("routed_int4_binding.cpp", "routed_body_native.h",
                 "routed_int4_kernel.h", "routed_int4_pipeline.h"):
        path = own / name
        content = path.read_text()
        for before, after in RENAMES.items():
            content = content.replace(before, after)
        path.write_text(content)
    for name in ("prepare.py", "build.sh", "README.md"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    report = {
        "parent_kernel_sha256": PARENT_KERNEL,
        "parent_numerical_body_sha256": body_hash,
        "callable_numerical_body_sha256": body_hash,
        "changes": ["caller-owned TPipe and device initialization",
                    "named helper scope", "separate host ABI and Torch namespace"],
        "barriers_inside_body": 6,
        "return_contract": "Caller must drain, publish, synchronize and reset "
                           "before a later phase consumes outputs or reuses UB",
        "symbol_renames": RENAMES,
        "sources": {str(p.relative_to(output)): digest(p)
                    for p in sorted((output / "source").rglob("*"))
                    if p.is_file()},
    }
    (output / "source-transforms.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (output / "parent-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.parent.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
