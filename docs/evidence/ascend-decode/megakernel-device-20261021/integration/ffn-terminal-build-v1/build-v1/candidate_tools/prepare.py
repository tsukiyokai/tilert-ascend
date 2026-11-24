"""Drain the complete FFN locally at its standalone global kernel exit."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT_LIBRARY = "8430ef17d068ce5a1cdbe4915ba189ca38c666322d8726a42091f8acb3b36f84"
PARENT_CHECKER = "50efbec8168286e01a8e327db6ef1bc79dce345fb7d36a6bb4b4db77969f967c"
KERNEL = "source/kernel.cpp"
OLD_EXIT = """    // NoTrace::FinishFinal is empty. This is the required publication seam,
    // after every body-local TBuf and matrix object has left scope.
    ffn_bridge::device::EndPhase(pipe);
"""
NEW_EXIT = """    // All output consumers follow this global task on the same stream.
    // The callable body and its internal publication boundaries are unchanged.
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
"""


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one frozen source seam: {before!r}")
    return source.replace(before, after, 1)


def rename(source):
    return source.replace("ffn_active_token_join", "ffn_terminal").replace(
        "FFN_ACTIVE_TOKEN_JOIN", "FFN_TERMINAL")


def bind_kernel(source):
    return replace_once(source, OLD_EXIT, NEW_EXIT)


def bind_checker(source):
    source = source.replace("torch.ops.tilert_ffn_active_token_join", "torch.ops.tilert_ffn_terminal")
    source = replace_once(source, "    def check_values(name, actual, expected):\n",
        "    def check_values(name, actual, expected):\n"
        "        expected = {**expected, **{\n"
        "            'consumer.' + key: expected[key] for key in consumer_names}}\n")
    source = replace_once(source, "    def snapshot(values):\n",
        "    consumer_names = ('bridge.combined', 'bridge.scaled_routed',\n"
        "                      'body.routes_bf16', 'body.routed_output')\n\n"
        "    def snapshot(values):\n")
    source = replace_once(source, "                    def composed():\n",
        "                    terminal_outputs = (cb[10], cb[11], cr[5], cr[9])\n"
        "                    consumed = [torch.empty_like(value) for value in terminal_outputs]\n\n"
        "                    def composed():\n")
    source = replace_once(source,
        "                            routed_weights, shared_weights, cb, cr, 1e-5, blocks, layout, cs)\n",
        "                            routed_weights, shared_weights, cb, cr, 1e-5, blocks, layout, cs)\n"
        "                        for destination, value in zip(consumed, terminal_outputs):\n"
        "                            destination.copy_(value)\n")
    source = replace_once(source, "                        for graph in captures.values():\n",
        "                        for value in (*terminal_outputs, *consumed):\n"
        "                            value.fill_(float('nan'))\n"
        "                        for graph in captures.values():\n")
    source = replace_once(source, '                        actual["body.active_experts"] = cr[20]\n',
        '                        actual["body.active_experts"] = cr[20]\n'
        "                        actual.update({'consumer.' + name: value\n"
        "                                       for name, value in zip(consumer_names, consumed)})\n")
    return source


def prepare(parent, output):
    manifest = json.loads((parent / "manifest.json").read_text())
    if manifest["artifacts"]["ffn_active_token_join.so"] != PARENT_LIBRARY:
        raise ValueError("Use the validated active-descriptor/token-output composition")
    if digest(parent / "check.py") != PARENT_CHECKER:
        raise ValueError("Frozen checker changed")
    for name, expected in manifest["sources"].items():
        if digest(parent / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest["sources"]:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, target)
    original = (parent / KERNEL).read_text()
    changed = bind_kernel(original)
    (output / KERNEL).write_text(changed)
    (output / "terminal.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile="parent/" + KERNEL, tofile="candidate/" + KERNEL)))
    for path in output.rglob("*"):
        if path.suffix in (".h", ".cpp"):
            path.write_text(rename(path.read_text()))
    (output / "check.py").write_text(bind_checker((parent / "check.py").read_text()))
    for name in ("prepare.py", "build.py", "test_prepare.py", "README.md"):
        shutil.copy2(Path(__file__).with_name(name), output / "candidate_tools" / name)
    shutil.copy2(parent / "manifest.json", output / "parent-manifest.json")
    report = {
        "parent_library_sha256": PARENT_LIBRARY,
        "parent_checker_sha256": PARENT_CHECKER,
        "scope": "Terminal global entry only; no later callable consumer or resource reuse",
        "descriptor_bytes": 408, "public_scratch_count": 33,
        "maximum_resident_ub_bytes": 116480,
        "additional_gm_bytes": 0, "additional_ub_bytes": 0,
        "body_barriers": 5, "final_outer_end_phases": 0,
        "profile_supported": False,
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
