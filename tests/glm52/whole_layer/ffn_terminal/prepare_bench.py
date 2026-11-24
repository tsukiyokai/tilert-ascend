"""Compare terminal completion against the combined complete-FFN parent."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT = "dd05d557d550dac139884eea73fb92a91855c2cf41ee8b2369e8ec3e2c0c33f5"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one benchmark seam: {before!r}")
    return source.replace(before, after, 1)


def transform(source):
    replacements = (
        ("        self.base = self.libraries[-1].ffn_active_descriptor_launch",
         "        self.base = self.libraries[-1].ffn_active_token_join_launch"),
        ("        self.fused = self.libraries[-1].ffn_active_token_join_launch",
         "        self.fused = self.libraries[-1].ffn_terminal_launch"),
        ('"scope": "Matched-address active-token-join FFN versus active-descriptor FFN, TP excluded",',
         '"scope": "Matched-address terminal FFN versus active-token-join FFN, TP excluded",'),
        ('"arm_names": {"fused": "active_token_join", "separate": "active_descriptor"},',
         '"arm_names": {"fused": "terminal", "separate": "active_token_join"},'),
    )
    for before, after in replacements:
        source = once(source, before, after)
    return source


def prepare(parent, output):
    if digest(parent) != PARENT:
        raise ValueError("Use the frozen active-token-join timing-v1 benchmark")
    original = parent.read_text()
    source = transform(original)
    compile(source, "bench.py", "exec")
    output.mkdir(parents=True, exist_ok=False)
    (output / "bench.py").write_text(source)
    (output / "benchmark.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), source.splitlines(keepends=True),
        fromfile="active-token-join-timing-v1/bench.py",
        tofile="terminal/bench.py")))
    for name in ("prepare_bench.py", "bench_abi_probe.cpp", "test_bench.py",
                 "bench_run.sh"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    original_probe = (Path(__file__).resolve().parent.parent
                      / "ffn_composed/bench_abi_probe.cpp")
    (output / "parent").mkdir()
    shutil.copy2(parent, output / "parent/bench.py")
    shutil.copy2(original_probe, output / "parent/bench_abi_probe.cpp")
    report = {
        "parent_benchmark_sha256": PARENT,
        "comparison": "Both C ABIs share all 33 tensors and weight addresses",
        "table_validation": "Both arms replace all 264 poisoned INT32 words",
        "sources": {str(p.relative_to(output)): digest(p)
                    for p in output.rglob("*") if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.parent.resolve(), args.output.resolve())
