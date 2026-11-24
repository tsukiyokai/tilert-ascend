"""Freeze token-owned MSD splitting from the validated active/token-tail FFN."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT_LIBRARY = "8430ef17d068ce5a1cdbe4915ba189ca38c666322d8726a42091f8acb3b36f84"
PARENT_CHECKER = "50efbec8168286e01a8e327db6ef1bc79dce345fb7d36a6bb4b4db77969f967c"
BODY = "body/source/routed_int4/routed_body_native.h"
SPLIT_START = "    // Every core reaches every barrier, including empty expert owners.\n"
SPLIT_END = "    timing.Barrier(1);\n"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one frozen split seam: {before!r}")
    return source.replace(before, after, 1)


def rename(source):
    return source.replace("ffn_active_token_join", "ffn_token_split").replace(
        "FFN_ACTIVE_TOKEN_JOIN", "FFN_TOKEN_SPLIT")


def bind_body(source):
    source = replace_once(source, '#include "../../../source/output_contract.h"',
        '#include "../../../source/output_contract.h"\n'
        '#include "../../../source/split_contract.h"')
    source = replace_once(source, '#include "../../../source/output_native.h"',
        '#include "../../../source/output_native.h"\n'
        '#include "../../../source/split_native.h"')
    begin = source.index(SPLIT_START)
    end = source.index(SPLIT_END, begin)
    return (source[:begin] + SPLIT_START
        + "    if ASCEND_IS_AIV {\n"
        + "        SplitTokens(held, (__gm__ int8_t*)arguments.scratch[kRestoredRows],\n"
        + "            (__gm__ int4b_t*)arguments.scratch[kStacked],\n"
        + "            (__gm__ int32_t*)arguments.dispatch.expert_ids,\n"
        + "            arguments.rows, arguments.blocks);\n"
        + "        Publish();\n"
        + "    }\n" + source[end:])


def prepare(parent, output):
    manifest = json.loads((parent / "manifest.json").read_text())
    if manifest["artifacts"]["ffn_active_token_join.so"] != PARENT_LIBRARY:
        raise ValueError("Use the frozen 8430 active-descriptor/token-join parent")
    if digest(parent / "check.py") != PARENT_CHECKER:
        raise ValueError("Frozen parent checker changed")
    for name, expected in manifest["sources"].items():
        if digest(parent / name) != expected:
            raise ValueError(f"Frozen parent source changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest["sources"]:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, target)
    original = (parent / BODY).read_text()
    changed = bind_body(original)
    (output / BODY).write_text(changed)
    (output / "token-split.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile="parent/" + BODY, tofile="candidate/" + BODY)))
    for path in output.rglob("*"):
        if path.suffix in (".cpp", ".h"):
            path.write_text(rename(path.read_text()))
    (output / "check.py").write_text((parent / "check.py").read_text().replace(
        "torch.ops.tilert_ffn_active_token_join", "torch.ops.tilert_ffn_token_split"))
    for name in ("split_contract.h", "split_native.h"):
        shutil.copy2(Path(__file__).with_name(name), output / "source" / name)
    (output / "candidate_tools").mkdir(exist_ok=True)
    for name in ("prepare.py", "build.py", "test_contract.py", "ownership_probe.cpp",
                 "README.md"):
        shutil.copy2(Path(__file__).with_name(name), output / "candidate_tools" / name)
    shutil.copy2(parent / "manifest.json", output / "parent-manifest.json")
    report = {
        "parent_library_sha256": PARENT_LIBRARY,
        "parent_manifest_sha256": digest(parent / "manifest.json"),
        "parent_checker_sha256": PARENT_CHECKER,
        "public_scratch_count": 33,
        "descriptor_bytes": 408,
        "additional_gm_bytes": 0,
        "additional_ub_bytes": 0,
        "maximum_resident_ub_bytes": 116480,
        "split_tile": 2048,
        "body_barriers": 5,
        "outer_end_phases": 1,
        "profile_supported": False,
        "reused_buffers": {
            "staging_first_2048_bytes": "inverse[256] and original expert IDs[256]",
            "input_first_1024_bytes": "packed low bits after A8 conversion completes",
            "packed_first_1024_bytes": "packed high bits",
        },
        "preserved_helpers": ["SplitHalves", "Project", "Activate", "TokenOutputs"],
        "system_workspace_behavior": "Original calls and initial bridge selection retained",
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
