"""Keep the measured N1024 protocol and replace only the two tiling variants."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT = "477b3172c12134c2e87545ca7195ed49ad0160d257e0bf8550c15ae66b2519db"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(source):
    source = source.replace("ffn_down_n1024", "ffn_down_k128")
    source = source.replace("ffn_down_n512", "ffn_down_n1024")
    expected = json.loads(Path(__file__).with_name("expected-tilings.json").read_text())
    prefix = "EXPECTED_DOWN_TILINGS = "
    begin = source.index(prefix)
    end = source.index("\n\n\nclass DispatchInput", begin)
    source = source[:begin] + prefix + repr(expected) + source[end:]
    changes = (
        ("Down M32/N1024/K64 versus M64/N512/K128 FFN", "Down M32/N1024/K128 single-buffer versus K64 FFN"),
        ('"arm_names": {"fused": "down_n1024", "separate": "down_n512_parent"}',
         '"arm_names": {"fused": "down_k128", "separate": "down_n1024_parent"}'),
    )
    for before, after in changes:
        if source.count(before) != 1:
            raise ValueError(f"Expected one frozen benchmark seam: {before!r}")
        source = source.replace(before, after, 1)
    return source


def prepare(parent, output):
    if digest(parent) != PARENT:
        raise ValueError("Use the frozen Down-N1024 timing-v1 benchmark")
    original = parent.read_text()
    source = transform(original)
    compile(source, "bench.py", "exec")
    output.mkdir(parents=True, exist_ok=False)
    (output / "bench.py").write_text(source)
    (output / "benchmark.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), source.splitlines(keepends=True),
        fromfile="down-n1024-timing-v1/bench.py", tofile="down-k128/bench.py")))
    for name in ("prepare_bench.py", "bench_abi_probe.cpp", "test_bench.py",
                 "bench_run.sh", "expected-tilings.json"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    (output / "parent").mkdir()
    shutil.copy2(parent, output / "parent/bench.py")
    shutil.copy2(parent.parent / "parent/bench_abi_probe.cpp",
                 output / "parent/bench_abi_probe.cpp")
    expected = json.loads((output / "expected-tilings.json").read_text())
    report = {
        "parent_benchmark_sha256": PARENT,
        "control_library_sha256": expected["arms"]["separate"]["library_sha256"],
        "candidate_library_sha256": expected["arms"]["fused"]["library_sha256"],
        "comparison": "32 common tensor addresses; body[7] retains each arm's prepared TensorImpl and complete 200B tiling",
        "tiling_oracle_sha256": digest(output / "expected-tilings.json"),
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
