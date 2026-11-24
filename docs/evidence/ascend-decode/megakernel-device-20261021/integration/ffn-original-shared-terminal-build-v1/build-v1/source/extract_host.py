"""Freeze the bridge and callable routed body with validation-only host seams."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


BRIDGE_LIBRARY = "140f157225906b9ade87f63a70253ab94a2711fc61ea1024bafe713ac463b1df"
BODY_LIBRARY = "6dcb8401cf722f9b3c3650d2432321ce6100d79aa98ff329d3a17a63a0f1d6c4"
BODY_STATEMENTS = "520e55acd1524ed2b1d41bf94b4b0da4cd2371e7aff8126780cb9a59de188cd8"
RENAMES = {
    "ffn_bridge_build_tiling": "ffn_composed_bridge_build_tiling",
    "routed_shared_body_build_routed_tiling": "ffn_composed_body_build_routed_tiling",
    "routed_shared_body_build_tiling": "ffn_composed_body_build_shared_tiling",
    "routed_shared_body_launch": "ffn_composed_unused_body_launch",
    "routed_shared_body_fused": "ffn_composed_unused_body_kernel",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_copy(parent, hashes, destination):
    for name, expected in hashes.items():
        source = parent / name
        if digest(source) != expected:
            raise ValueError(f"Frozen source changed: {source}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def remove_call(source, name):
    """Remove one complete direct-launch statement, retaining every validator."""
    marker = "\n    " + name + "("
    if source.count(marker) != 1:
        raise ValueError(f"Expected one direct launch: {name}")
    begin = source.index(marker) + 1
    cursor = source.index("(", begin)
    depth = 1
    cursor += 1
    while depth:
        depth += (source[cursor] == "(") - (source[cursor] == ")")
        cursor += 1
    if source[cursor] != ";":
        raise ValueError("Unexpected launch terminator")
    removed = source[begin:cursor + 1]
    return source[:begin] + source[cursor + 1:], removed


def host_seam(source, scope, launches):
    boundary = "\n}  // namespace\n"
    if source.count(boundary) != 1:
        raise ValueError("Expected one owning host scope")
    prefix = source[:source.index(boundary) + len(boundary)]
    if prefix.count("namespace {\n") != 1:
        raise ValueError("Expected one anonymous host scope")
    prefix = prefix.replace("namespace {\n", f"namespace {scope} {{\n", 1)
    removed = []
    for launch in launches:
        prefix, statement = remove_call(prefix, launch)
        removed.append(statement)
    if "getCurrentNPUStream" in prefix:
        raise ValueError("Validation-only seam still submits stream work")
    return prefix, removed


def bridge_body(source):
    begin = source.index("    whole_layer_norm::device::Forward(pipe,")
    end = source.index("\n}\n", begin)
    statements = source[begin:end]
    if statements.count("device::EndPhase(pipe);") != 3:
        raise ValueError("Expected the frozen three bridge boundaries")
    return (
        '#include "native.h"\n\n'
        "namespace ffn_bridge {\n"
        "struct Addresses { GM_ADDR values[kBuffers]; };\n"
        "__aicore__ inline void Forward(AscendC::TPipe& pipe, GM_ADDR input,\n"
        "    GM_ADDR residual, GM_ADDR gamma, GM_ADDR gate, GM_ADDR bias,\n"
        "    Addresses a, int64_t rows, int64_t blocks, float epsilon)\n"
        "{\n    using namespace AscendC;\n" + statements + "\n}\n}\n",
        hashlib.sha256(statements.encode()).hexdigest(),
    )


def prepare(bridge, body, output):
    bridge_manifest = json.loads((bridge / "manifest.json").read_text())
    body_manifest = json.loads((body / "build-manifest.json").read_text())
    if bridge_manifest["artifacts"]["ffn_bridge.so"] != BRIDGE_LIBRARY:
        raise ValueError("Use the frozen bridge build-v2 parent")
    if (body_manifest["library_sha256"] != BODY_LIBRARY
            or body_manifest["callable_numerical_body_sha256"] != BODY_STATEMENTS):
        raise ValueError("Use the validated callable routed-body build-v2 parent")
    output.mkdir(parents=True, exist_ok=False)
    checked_copy(bridge, bridge_manifest["sources"], output / "bridge")
    checked_copy(body, body_manifest["sources"], output / "body")
    source = output / "source"
    source.mkdir()
    for name in ("prepare.py", "contract.h", "binding.cpp", "kernel.cpp",
                 "build.sh", "README.md", "test_prepare.py", "contract_test.cpp"):
        shutil.copy2(Path(__file__).with_name(name), source / name)

    bridge_host = output / "bridge/source/ffn_bridge/binding.cpp"
    bridge_header, bridge_removed = host_seam(
        bridge_host.read_text(), "ffn_composed_bridge_host",
        ("ffn_bridge_launch", "ffn_bridge_topk_launch", "ffn_bridge_epilogue_launch"))
    body_host = output / "body/source/routed_int4/routed_int4_binding.cpp"
    body_header, body_removed = host_seam(
        body_host.read_text(), "ffn_composed_body_host", ("routed_int4_launch",))
    bridge_host.with_name("host_validate.h").write_text(bridge_header)
    body_host.with_name("host_validate.h").write_text(body_header)
    callable_bridge, numerical_hash = bridge_body(
        bridge_host.with_name("kernel.cpp").read_text())
    bridge_host.with_name("body_native.h").write_text(callable_bridge)
    for directory in (output / "bridge", output / "body"):
        for path in directory.rglob("*"):
            if path.suffix not in (".h", ".cpp"):
                continue
            content = path.read_text()
            for before, after in RENAMES.items():
                content = content.replace(before, after)
            path.write_text(content)
    report = {
        "parents": {"bridge": bridge_manifest, "body": body_manifest},
        "bridge_statement_sha256": numerical_hash,
        "routed_body_statement_sha256": BODY_STATEMENTS,
        "removed_launch_statements": {"bridge": bridge_removed, "body": body_removed},
        "symbol_renames": RENAMES,
        "scope": "32 disjoint caller-owned tensors; TP outside; one mixed launch",
        "sources": {str(p.relative_to(output)): digest(p)
                    for p in sorted(output.rglob("*")) if p.is_file()},
    }
    (output / "source-transforms.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bridge", type=Path)
    parser.add_argument("body", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.bridge.resolve(), args.body.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
