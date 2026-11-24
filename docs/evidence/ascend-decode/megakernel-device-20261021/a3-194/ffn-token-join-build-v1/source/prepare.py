"""Generate a token-owned FFN output candidate from the frozen composed body."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT_LIBRARY = "3673c2a73b2f4b5a4a93b3c2543a4aca35b45773d6a53fc043f27e59a2482655"
BODY = "body/source/routed_int4/routed_body_native.h"
BODY_MARKER = "template <CubeFormat weight_format, bool from_dispatch>\n"
STAGE_FIVE = "    timing.Barrier(4);\n    timing.Begin(5);\n"
STAGE_SIX = "    timing.Barrier(5);\n    timing.Begin(6);\n"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one frozen FFN seam: {before!r}")
    return source.replace(before, after, 1)


def bind_body(source):
    source = replace_once(source, '#include "routed_int4_kernel.h"',
        '#include "routed_int4_kernel.h"\n#include "../../../source/output_contract.h"')
    source = replace_once(source,
        "}  // namespace\n\n" + BODY_MARKER,
        '#include "../../../source/output_native.h"\n\n'
        "}  // namespace\n\n" + BODY_MARKER)
    source = replace_once(source,
        "    , pipeline::Input shared_input\n#endif\n    )",
        "    , pipeline::Input shared_input\n#endif\n"
        "    , __gm__ bfloat16_t* scaled_output, __gm__ bfloat16_t* combined_output\n    )")
    start = source.index(STAGE_FIVE) + len(STAGE_FIVE)
    end = source.index(STAGE_SIX, start)
    source = (source[:start]
        + "    if ASCEND_IS_AIV {\n"
        + "        TokenOutputs(pipe, held, wide, down_bias, down_scale, middle_scale,\n"
        + "            (__gm__ int32_t*)dispatch_input.expert_ids, route_output, out,\n"
        + "            (__gm__ bfloat16_t*)shared_input.output, scaled_output,\n"
        + "            combined_output, tokens, blocks);\n"
        + "        Publish();\n    }\n" + source[end:])
    return replace_once(source,
        "    if ASCEND_IS_AIV {\n"
        "        CombineTokens(held, route_output, out, tokens, blocks);\n    }\n", "")


def bind_kernel(source):
    source = replace_once(source, "        dispatch, nullptr, shared);",
        "        dispatch, nullptr, shared,\n"
        "        (__gm__ bfloat16_t*)bridge.values[ffn_bridge::kScaledRouted],\n"
        "        (__gm__ bfloat16_t*)bridge.values[ffn_bridge::kCombined]);")
    return replace_once(source,
        "    ffn_bridge::device::Epilogue(pipe, body.values[kOutput],\n"
        "        body.values[pipeline::kSharedOutput], bridge.values[ffn_bridge::kScaledRouted],\n"
        "        bridge.values[ffn_bridge::kCombined], rows, blocks);\n", "")


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
    before = {name: (output / name).read_text()
              for name in (BODY, "source/kernel.cpp")}
    (output / BODY).write_text(bind_body(before[BODY]))
    (output / "source/kernel.cpp").write_text(bind_kernel(before["source/kernel.cpp"]))
    patch = []
    for name, original in before.items():
        patch.extend(difflib.unified_diff(original.splitlines(keepends=True),
            (output / name).read_text().splitlines(keepends=True),
            fromfile=f"parent/{name}", tofile=f"candidate/{name}"))
    (output / "token-output.patch").write_text("".join(patch))
    for path in output.rglob("*"):
        if path.suffix in (".h", ".cpp"):
            path.write_text(path.read_text().replace(
                "ffn_composed", "ffn_token_output").replace(
                "FFN_COMPOSED", "FFN_TOKEN_OUTPUT"))
    for name in ("output_contract.h", "output_native.h", "prepare.py", "build.py",
                 "test_contract.py", "ownership_probe.cpp", "README.md"):
        shutil.copy2(Path(__file__).with_name(name), output / "source" / name)
    checker = parent / "harness-v1/check.py"
    if checker.is_file():
        (output / "check.py").write_text(checker.read_text().replace(
            "torch.ops.tilert_ffn_composed", "torch.ops.tilert_ffn_token_output"))
    preserved = before[BODY].split(STAGE_FIVE, 1)[0]
    numerical_start = preserved.index("\n{\n", preserved.index(BODY_MARKER)) + 3
    report = {
        "parent_library_sha256": PARENT_LIBRARY,
        "parent_manifest_sha256": digest(parent / "manifest.json"),
        "public_scratch_count": 32,
        "additional_gm_bytes": 0,
        "additional_ub_bytes": 3072,
        "maximum_resident_ub_bytes": 115424,
        "hidden_tile": 1536,
        "body_barriers": 6,
        "outer_end_phases": 2,
        "unchanged_body_prefix_sha256": hashlib.sha256(
            preserved[numerical_start:].encode()).hexdigest(),
        "preserved_helpers": ["LoadScale", "ScaledPartial", "Recombine"],
        "diagnostic_outputs": "Original route, routed, scaled and combined tensors retained",
        "system_workspace_behavior": "Original calls and initial bridge selection retained",
        "sources": {str(path.relative_to(output)): digest(path)
                    for path in sorted(output.rglob("*")) if path.is_file()},
    }
    (output / "source-transforms.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copy2(parent / "manifest.json", output / "parent-manifest.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.parent.resolve(), args.output.resolve())
