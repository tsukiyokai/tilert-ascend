"""Replace the frozen FFN body's wide arguments with a constant descriptor."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil


PARENT_LIBRARY = "3673c2a73b2f4b5a4a93b3c2543a4aca35b45773d6a53fc043f27e59a2482655"
BODY = "body/source/routed_int4/routed_body_native.h"
BODY_MARKER = "template <CubeFormat weight_format, bool from_dispatch>\n"
TOKEN = re.compile(
    r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"'
    r"|'(?:\\.|[^'\\])*'|[A-Za-z_][A-Za-z_0-9]*")


def pointer(dtype, expression):
    return f"((__gm__ {dtype}*){expression})"


BINDINGS = {
    "rows": pointer("int8_t", "arguments.scratch[kRestoredRows]"),
    "gate_up": pointer("int4b_t", "arguments.weights[0]"),
    "gu_bias": pointer("float", "arguments.weights[1]"),
    "gu_scale": pointer("ScaleStorage", "arguments.weights[2]"),
    "down": pointer("int4b_t", "arguments.weights[3]"),
    "down_bias": pointer("float", "arguments.weights[4]"),
    "down_scale": pointer("ScaleStorage", "arguments.weights[5]"),
    "token_scale": pointer("float", "arguments.scratch[kRestoredScale]"),
    "group_count": pointer("int32_t", "arguments.scratch[kGroupCount]"),
    "group_start": pointer("int32_t", "arguments.scratch[kGroupStart]"),
    "source_row": pointer("int32_t", "arguments.scratch[kSourceRow]"),
    "source_slot": pointer("int32_t", "arguments.scratch[kSourceSlot]"),
    "row_weight": pointer("float", "arguments.scratch[kRowWeight]"),
    "stacked": pointer("int4b_t", "arguments.scratch[kStacked]"),
    "partial": pointer("RoutedProduct", "arguments.scratch[kGateProduct]"),
    "middle": pointer("int4b_t", "arguments.scratch[kMiddle]"),
    "middle_scale": pointer("float", "arguments.scratch[kMiddleScale]"),
    "wide": pointer("RoutedProduct", "arguments.scratch[kDownProduct]"),
    "route_output": pointer("bfloat16_t", "arguments.scratch[kRouteOutput]"),
    "out": pointer("bfloat16_t", "arguments.scratch[kOutput]"),
    "gu_tiling": pointer("uint8_t", "arguments.scratch[kGateTiling]"),
    "down_tiling": pointer("uint8_t", "arguments.scratch[kDownTiling]"),
    "workspace": "((GM_ADDR)arguments.scratch[kMatmulWorkspace])",
    "experts": "ffn_bridge::kExperts",
    "tokens": "arguments.rows",
    "blocks": "arguments.blocks",
    "dispatch_input": "arguments.dispatch",
    "trace": "((__gm__ int64_t*)nullptr)",
    "shared_input": "arguments.shared",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one frozen FFN seam: {before!r}")
    return source.replace(before, after, 1)


def body_parts(source):
    start = source.index(BODY_MARKER)
    body_start = source.index("\n{\n", start) + 3
    end = source.index("\n}\n\n}  // namespace routed_body", body_start)
    return source[:start], source[start:body_start], source[body_start:end], source[end:]


def bind_body(source):
    prefix, declaration, numerical, suffix = body_parts(source)
    prefix = replace_once(prefix, '#include "routed_int4_kernel.h"',
        '#include "routed_int4_kernel.h"\n#include "../../../source/descriptor.h"')
    translated = TOKEN.sub(lambda match: BINDINGS.get(match[0], match[0]), numerical)
    restored = translated
    for name, expression in sorted(BINDINGS.items(), key=lambda item: -len(item[1])):
        restored = restored.replace(expression, name)
    if restored != numerical:
        raise ValueError("Descriptor field substitution changed numerical statements")
    signature = (BODY_MARKER + "__aicore__ inline void Forward(\n"
                 "    TPipe& pipe, const ffn_descriptor::BodyArguments& arguments)\n{\n")
    return prefix + signature + translated + suffix, numerical, declaration


def bind_kernel(source):
    source = replace_once(source, '#include "contract.h"',
                          '#include "contract.h"\n#include "descriptor.h"')
    structs = source.index("struct BodyAddresses {")
    end_structs = source.index("template <CubeFormat format>", structs)
    source = source[:structs] + source[end_structs:]
    source = replace_once(source,
        "    GM_ADDR gate, GM_ADDR bias, WeightAddresses w, ffn_bridge::Addresses bridge,\n"
        "    BodyAddresses body, pipeline::Input shared, int64_t rows, int64_t blocks,\n",
        "    GM_ADDR gate, GM_ADDR bias, ffn_bridge::Addresses bridge,\n"
        "    const ffn_descriptor::BodyArguments arguments,\n")
    source = replace_once(source, "                        bridge, rows, blocks, epsilon);",
        "                        bridge, arguments.rows, arguments.blocks, epsilon);")
    start = source.index("    SetSysWorkspace(body.values[kMatmulWorkspace]);")
    end = source.index("    // NoTrace::FinishFinal", start)
    source = (source[:start]
              + "    SetSysWorkspace((GM_ADDR)arguments.scratch[kMatmulWorkspace]);\n"
              + "    routed_body::Forward<format, true>(pipe, arguments);\n"
              + source[end:])
    source = replace_once(source,
        "    ffn_bridge::device::Epilogue(pipe, body.values[kOutput],\n"
        "        body.values[pipeline::kSharedOutput], bridge.values[ffn_bridge::kScaledRouted],\n"
        "        bridge.values[ffn_bridge::kCombined], rows, blocks);",
        "    ffn_bridge::device::Epilogue(pipe,\n"
        "        (GM_ADDR)arguments.scratch[kOutput],\n"
        "        (GM_ADDR)arguments.scratch[pipeline::kSharedOutput],\n"
        "        bridge.values[ffn_bridge::kScaledRouted],\n"
        "        bridge.values[ffn_bridge::kCombined], arguments.rows, arguments.blocks);")
    start = source.index("    ffn_composed_device::WeightAddresses weights{};")
    end = source.index("    if (weight_layout ==", start)
    source = (source[:start]
        + "    ffn_bridge::Addresses bridge_addresses{};\n"
        + "    for (int64_t i = 0; i < ffn_composed::kBridgeBuffers; ++i) {\n"
        + "        bridge_addresses.values[i] = (GM_ADDR)bridge[i];\n    }\n"
        + "    const auto arguments = ffn_descriptor::MakeBodyArguments(\n"
        + "        routed_weights, bridge, body, shared, rows, blocks);\n"
        + source[end:])
    before = ("            (GM_ADDR)bias, weights, bridge_addresses, body_addresses,\n"
              "            shared, rows, blocks, epsilon);")
    if source.count(before) != 2:
        raise ValueError("Expected both ND and NZ kernel launch arguments")
    return source.replace(before,
        "            (GM_ADDR)bias, bridge_addresses, arguments, epsilon);")


def prepare(parent, output):
    manifest = json.loads((parent / "manifest.json").read_text())
    if manifest["artifacts"]["ffn_composed.so"] != PARENT_LIBRARY:
        raise ValueError("Use the validated uninstrumented composed build-v2 parent")
    for name, expected in manifest["sources"].items():
        if digest(parent / name) != expected:
            raise ValueError(f"Frozen parent source changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest["sources"]:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, target)
    body_path = output / BODY
    body_before = body_path.read_text()
    translated, numerical, declaration = bind_body(body_before)
    body_path.write_text(translated)
    kernel = output / "source/kernel.cpp"
    kernel_before = kernel.read_text()
    kernel.write_text(bind_kernel(kernel_before))
    patches = []
    for name, before in ((BODY, body_before), ("source/kernel.cpp", kernel_before)):
        patches.extend(difflib.unified_diff(before.splitlines(keepends=True),
            (output / name).read_text().splitlines(keepends=True),
            fromfile=f"parent/{name}", tofile=f"candidate/{name}"))
    (output / "descriptor.patch").write_text("".join(patches))
    for path in output.rglob("*"):
        if path.suffix in (".h", ".cpp"):
            path.write_text(path.read_text().replace(
                "ffn_composed", "ffn_descriptor").replace(
                "FFN_COMPOSED", "FFN_DESCRIPTOR"))
    for name in ("descriptor.h", "prepare.py", "build.py", "test_prepare.py",
                 "descriptor_probe.cpp", "README.md"):
        shutil.copy2(Path(__file__).with_name(name), output / "source" / name)
    parent_checker = parent / "harness-v1/check.py"
    if parent_checker.is_file():
        (output / "check.py").write_text(parent_checker.read_text().replace(
            "torch.ops.tilert_ffn_composed", "torch.ops.tilert_ffn_descriptor"))
    report = {
        "parent_library_sha256": PARENT_LIBRARY,
        "parent_manifest_sha256": digest(parent / "manifest.json"),
        "descriptor_bytes": 392,
        "additional_kernel_parameter_bytes": 56,
        "additional_gm_bytes": 0,
        "public_scratch_count": 32,
        "body_parameter_bindings": BINDINGS,
        "body_declaration_sha256": hashlib.sha256(declaration.encode()).hexdigest(),
        "body_statement_bytes": len(numerical.encode()),
        "body_statement_sha256": hashlib.sha256(numerical.encode()).hexdigest(),
        "body_statement_equivalence": "Exact after reversing parameter field substitutions",
        "system_workspace_behavior": "Original calls and initial bridge selection retained",
        "sources": {str(p.relative_to(output)): digest(p)
                    for p in sorted(output.rglob("*")) if p.is_file()},
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
