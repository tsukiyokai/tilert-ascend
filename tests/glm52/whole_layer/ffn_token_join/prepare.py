"""Remove the redundant token-output tail joins from a frozen parent."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT_LIBRARY = "e92a9cbe4d11307a8ea8beae20b5cbe845be10f9aba5280cbc3f438b049ed041"
BODY = "body/source/routed_int4/routed_body_native.h"
KERNEL = "source/kernel.cpp"
TRACE = "body/source/routed_int4/routed_int4_trace.h"
END_PHASE = "bridge/source/ffn_bridge/native.h"
PROFILE_GUARD = (
    "#ifdef ROUTED_ENABLE_PROFILE\n"
    '#error "Token tail join requires the uninstrumented NoTrace body"\n'
    "#endif\n\n"
)
EMPTY_TAIL = (
    "    timing.Barrier(5);\n"
    "    timing.Begin(6);\n"
    "    timing.FinishFinal(6);\n"
    "    timing.Store(trace, tokens, blocks);\n"
    "}\n\n}  // namespace routed_body\n\n#endif\n"
)
FINAL_JOINS = (
    "    ffn_bridge::device::EndPhase(pipe);\n"
    "    ffn_bridge::device::EndPhase(pipe);\n"
    "}\n}  // namespace ffn_token_output_device\n"
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one frozen tail seam: {before!r}")
    return source.replace(before, after, 1)


def rename(source):
    return source.replace("ffn_token_output", "ffn_token_join").replace(
        "FFN_TOKEN_OUTPUT", "FFN_TOKEN_JOIN")


def bind_body(source):
    source = replace_once(source, EMPTY_TAIL,
                          EMPTY_TAIL.replace("    timing.Barrier(5);\n", ""))
    return replace_once(source, "#define TILERT_ROUTED_BODY_NATIVE_H\n\n",
                        "#define TILERT_ROUTED_BODY_NATIVE_H\n\n" + PROFILE_GUARD)


def bind_kernel(source):
    return replace_once(source, FINAL_JOINS, FINAL_JOINS.replace(
        "    ffn_bridge::device::EndPhase(pipe);\n", "", 1))


def prepare(parent, output):
    manifest = json.loads((parent / "manifest.json").read_text())
    if manifest["artifacts"]["ffn_token_output.so"] != PARENT_LIBRARY:
        raise ValueError("Use the validated token-output build-v2 parent")
    for name, expected in manifest["sources"].items():
        if digest(parent / name) != expected:
            raise ValueError(f"Frozen parent source changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest["sources"]:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, destination)
    originals = {name: (output / name).read_text() for name in (BODY, KERNEL)}
    (output / BODY).write_text(bind_body(originals[BODY]))
    (output / KERNEL).write_text(bind_kernel(originals[KERNEL]))
    patch = []
    for name, before in originals.items():
        patch.extend(difflib.unified_diff(
            before.splitlines(keepends=True),
            (output / name).read_text().splitlines(keepends=True),
            fromfile=f"parent/{name}", tofile=f"candidate/{name}"))
    (output / "tail-join.patch").write_text("".join(patch))
    for path in output.rglob("*"):
        if path.suffix in (".h", ".cpp"):
            path.write_text(rename(path.read_text()))
    checker = parent / "check.py"
    (output / "check.py").write_text(checker.read_text().replace(
        "torch.ops.tilert_ffn_token_output", "torch.ops.tilert_ffn_token_join"))
    tools = output / "candidate_tools"
    tools.mkdir()
    for name in ("prepare.py", "build.py", "test_contract.py", "README.md"):
        shutil.copy2(Path(__file__).with_name(name), tools / name)
    shutil.copy2(parent / "manifest.json", output / "parent-manifest.json")
    report = {
        "parent_library_sha256": PARENT_LIBRARY,
        "parent_manifest_sha256": digest(parent / "manifest.json"),
        "public_scratch_count": 32,
        "additional_gm_bytes": 0,
        "additional_ub_bytes": 0,
        "maximum_resident_ub_bytes": 115424,
        "body_barriers": 5,
        "outer_end_phases": 1,
        "removed": ["Body barrier 5", "Second adjacent outer EndPhase"],
        "profile_supported": False,
        "system_workspace_behavior": "Original calls and initial bridge selection retained",
        "numeric_contract": "All token-output statements and 30 retained boundaries unchanged",
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
