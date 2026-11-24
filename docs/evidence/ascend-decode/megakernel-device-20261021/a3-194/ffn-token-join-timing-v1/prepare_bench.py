"""Freeze a same-address tail-join versus token-output timing comparison."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT = "a159332ae3750a60ce93a030923fc13f2b71b90860c252733180e96f512d1429"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one benchmark seam: {before!r}")
    return source.replace(before, after, 1)


def transform(source):
    source = once(source,
        "        self.base = self.libraries[0].ffn_composed_launch",
        "        self.base = self.libraries[0].ffn_token_output_launch")
    source = once(source,
        "        self.fused = self.libraries[-1].ffn_token_output_launch",
        "        self.fused = self.libraries[-1].ffn_token_join_launch")
    source = source.replace("torch.ops.tilert_ffn_composed",
                            "torch.ops.tilert_ffn_token_output")
    source = once(source,
        "        candidate = str(getattr(torch.ops.tilert_ffn_token_output, operation).default._schema)",
        "        candidate = str(getattr(torch.ops.tilert_ffn_token_join, operation).default._schema)")
    source = once(source,
        '        if candidate != baseline.replace("ffn_composed", "ffn_token_output"):',
        '        if candidate != baseline.replace("ffn_token_output", "ffn_token_join"):')
    source = once(source, 'raise RuntimeError(f"Token-output schema changed: {operation}")',
                  'raise RuntimeError(f"Tail-join schema changed: {operation}")')
    source = once(source,
        '"scope": "Matched-address token-owned output FFN versus accepted one-launch FFN, TP excluded",',
        '"scope": "Matched-address FFN tail join versus its token-output parent, TP excluded",')
    return once(source,
        '"arm_names": {"fused": "token_output", "separate": "base_one_launch"},',
        '"arm_names": {"fused": "token_join", "separate": "token_output_parent"},')


def prepare(parent, output):
    if digest(parent) != PARENT:
        raise ValueError("Use the frozen token-output timing-v1 benchmark")
    original = parent.read_text()
    source = transform(original)
    compile(source, "bench.py", "exec")
    output.mkdir(parents=True, exist_ok=False)
    (output / "bench.py").write_text(source)
    (output / "benchmark.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), source.splitlines(keepends=True),
        fromfile="token-output/bench.py", tofile="token-join/bench.py")))
    for name in ("prepare_bench.py", "bench_abi_probe.cpp", "test_bench.py", "bench_run.sh"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    (output / "parent").mkdir()
    for name in ("bench.py", "bench_abi_probe.cpp"):
        shutil.copy2(parent.with_name(name), output / "parent" / name)
    (output / "grandparent").mkdir()
    shutil.copy2(parent.parent / "parent/bench_abi_probe.cpp",
                 output / "grandparent/bench_abi_probe.cpp")
    report = {
        "parent_benchmark_sha256": PARENT,
        "control_library_sha256": "e92a9cbe4d11307a8ea8beae20b5cbe845be10f9aba5280cbc3f438b049ed041",
        "candidate_library_sha256": "4417fd158e48a2da4d59e61fec48ebd0dd6e113022f13cbeb4f01d596eead757",
        "sources": {str(p.relative_to(output)): digest(p)
                    for p in output.rglob("*") if p.is_file()},
        "comparison": "Tail-join and token-output C ABIs share 32 addresses in each of two owning-Torch arenas",
        "extra_storage_bytes": 0,
        "preserved_gates": [
            "Finite CPU references and fresh-write poison before each checked replay",
            "Parent and component owning-Torch validators retain their TensorImpl metadata",
            "16 independent equal-valued weight banks; 30 alternating event pairs per arena",
            "Every bank checked before and after timing; three output checks after every event",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.parent.resolve(), args.output.resolve())
