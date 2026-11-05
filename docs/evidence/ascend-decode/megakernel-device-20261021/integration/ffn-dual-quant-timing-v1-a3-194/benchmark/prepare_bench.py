"""Freeze the shared quantization prefix versus its N512 parent at 33 addresses."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT = "cd7d20dd06b768d9c9f4255a80c11dc82f811e5856895e6a16637e97a9374e14"
CONTROL = "c90b3666a1697070eda19d5b3e133e9ed58f36fe429abefceae6ffe102a6a032"
CANDIDATE = "715060c0c50a275855eb757cd85128a6c5586deeaa7757dc2945b92bee64d69b"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one frozen benchmark seam: {before!r}")
    return source.replace(before, after, 1)


def transform(source):
    replacements = (
        ("self.base = self.libraries[-1].ffn_original_shared_terminal_launch",
         "self.base = self.libraries[-1].ffn_down_n512_launch"),
        ("self.fused = self.libraries[-1].ffn_original_shared_gu_static_launch",
         "self.fused = self.libraries[-1].ffn_dual_quant_launch"),
        ('"scope": "Matched-address static-GU FFN with original shared quantization versus its token-terminal parent, TP excluded",',
         '"scope": "Shared input-quantization prefix versus the N512 complete-FFN parent at identical addresses, TP excluded",'),
        ('"arm_names": {"fused": "original_shared_gu_static", "separate": "original_shared_terminal_parent"},',
         '"arm_names": {"fused": "dual_quant", "separate": "down_n512_parent"},'),
        ("torch.ops.tilert_ffn_original_shared_terminal.forward_out(",
         "torch.ops.tilert_ffn_down_n512.forward_out("),
        ("torch.ops.tilert_ffn_original_shared_terminal.prepare(",
         "torch.ops.tilert_ffn_down_n512.prepare("),
    )
    for before, after in replacements:
        source = once(source, before, after)
    return source


def prepare(parent, output):
    if digest(parent) != PARENT:
        raise ValueError("Use the frozen original-shared static-GU timing-v1 harness")
    original = parent.read_text()
    source = transform(original)
    compile(source, "bench.py", "exec")
    output.mkdir(parents=True, exist_ok=False)
    (output / "bench.py").write_text(source)
    (output / "benchmark.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), source.splitlines(keepends=True),
        fromfile="original-shared-gu-static-timing-v1/bench.py",
        tofile="dual-quant/bench.py")))
    for name in ("prepare_bench.py", "bench_abi_probe.cpp", "test_bench.py",
                 "bench_run.sh"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    (output / "parent").mkdir()
    shutil.copy2(parent, output / "parent/bench.py")
    shutil.copy2(parent.parent / "parent/bench_abi_probe.cpp",
                 output / "parent/bench_abi_probe.cpp")
    report = {
        "parent_benchmark_sha256": PARENT,
        "control_library_sha256": CONTROL,
        "candidate_library_sha256": CANDIDATE,
        "comparison": "All 33 tensor addresses and weight pointers are common to both arms",
        "owner": "Both independent arenas come from the unchanged N512 parent's prepare; its Torch validator runs before C ABI checks",
        "immutable_tiling": "Both arms use the same N512 tiling tensor; no tiling update during replay",
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
