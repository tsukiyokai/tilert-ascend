"""Freeze a same-address GU constant-tiling comparison against 276352."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT = "7196f88a49e33060287a3b462b72c10672237326bc781f52882ccf4e74cff016"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one benchmark seam: {before!r}")
    return source.replace(before, after, 1)


def transform(source):
    replacements = (
        ("        self.base = self.libraries[-1].ffn_active_token_join_launch",
         "        self.base = self.libraries[-1].ffn_token_terminal_launch"),
        ("        self.fused = self.libraries[-1].ffn_token_terminal_launch",
         "        self.fused = self.libraries[-1].ffn_gu_static_launch"),
        ('"scope": "Matched-address token-split terminal-completion FFN versus its active-token-join parent, TP excluded",',
         '"scope": "Matched-address GU constant-tiling FFN versus token-terminal FFN, TP excluded",'),
        ('"arm_names": {"fused": "token_terminal", "separate": "active_token_join_parent"},',
         '"arm_names": {"fused": "gu_static", "separate": "token_terminal_parent"},'),
    )
    for before, after in replacements:
        source = once(source, before, after)
    return source


def prepare(parent, output):
    if digest(parent) != PARENT:
        raise ValueError("Use the frozen token-terminal timing-v1 benchmark")
    original = parent.read_text()
    source = transform(original)
    compile(source, "bench.py", "exec")
    output.mkdir(parents=True, exist_ok=False)
    (output / "bench.py").write_text(source)
    (output / "benchmark.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), source.splitlines(keepends=True),
        fromfile="token-terminal-timing-v1/bench.py",
        tofile="gu-static/bench.py")))
    for name in ("prepare_bench.py", "bench_abi_probe.cpp", "test_bench.py", "bench_run.sh"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    (output / "parent").mkdir()
    shutil.copy2(parent, output / "parent/bench.py")
    shutil.copy2(parent.parent / "parent/bench_abi_probe.cpp",
                 output / "parent/bench_abi_probe.cpp")
    report = {
        "parent_benchmark_sha256": PARENT,
        "control_library_sha256": "276352762139244a1ddfc8184065a856c2ae75ee2ccad5961af2af93f0dd5e00",
        "candidate_library_sha256": "05b3427aa29630c9af16ea829e2ecec8833d31a32615e6a052f6f4e9867ca6b2",
        "comparison": "Both C ABIs share all 33 tensors and weight addresses",
        "table_validation": "Both arms replace all 264 poisoned INT32 words",
        "sources": {str(path.relative_to(output)): digest(path)
                    for path in sorted(output.rglob("*")) if path.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.parent.resolve(), args.output.resolve())
