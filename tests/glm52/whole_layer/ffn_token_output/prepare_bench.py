"""Freeze a token-output/base comparison from the validated FFN timing gate."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT = "eee6ab5549c727d20fd60c5bbf5495e60af84dedac061be0c2e757427df39f62"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one benchmark seam: {before!r}")
    return source.replace(before, after, 1)


def transform(source):
    source = once(source, "    def __init__(self, fused, bridge, body):",
                  "    def __init__(self, fused, bridge, body, candidate):")
    source = once(source,
        "        self.fused = self.libraries[0].ffn_composed_launch",
        "        self.base = self.libraries[0].ffn_composed_launch\n"
        "        self.libraries.append(ctypes.CDLL(str(candidate.resolve())))\n"
        "        self.fused = self.libraries[-1].ffn_token_output_launch")
    source = once(source,
        "        self.bridge = self.libraries[1].ffn_bridge_launch",
        "        self.base.argtypes = list(self.fused.argtypes)\n"
        "        self.base.restype = None\n"
        "        self.bridge = self.libraries[1].ffn_bridge_launch")
    begin = source.index("    def separate(self):\n")
    end = source.index("    def owning_torch(self, owner):\n", begin)
    source = source[:begin] + '''    def separate(self):
        stream = self.torch.npu.current_stream().npu_stream
        self.launchers.base(stream, self.sync, *self.input_arguments(), self.weight_ptrs,
            self.bridge_ptrs, self.body_ptrs, self.shared, self.rows, self.blocks, self.layout, 1e-5)

''' + source[end:]
    source = once(source,
        '    parser.add_argument("--device", default="npu:0")',
        '    parser.add_argument("--candidate-library", type=Path, required=True)\n'
        '    parser.add_argument("--device", default="npu:0")')
    source = once(source, "    for path in libraries:\n",
                  "    for path in (*libraries, args.candidate_library):\n")
    source = once(source, "    launchers = Launchers(*libraries)",
        "    launchers = Launchers(*libraries, args.candidate_library)\n"
        '    for operation in ("prepare", "forward_out"):\n'
        "        baseline = str(getattr(torch.ops.tilert_ffn_composed, operation).default._schema)\n"
        "        candidate = str(getattr(torch.ops.tilert_ffn_token_output, operation).default._schema)\n"
        '        if candidate != baseline.replace("ffn_composed", "ffn_token_output"):\n'
        '            raise RuntimeError(f"Token-output schema changed: {operation}")')
    source = once(source,
        '"scope": "Frozen C ABI graph replay: one FFN launch versus bridge+body+epilogue, TP excluded",',
        '"scope": "Matched-address token-owned output FFN versus accepted one-launch FFN, TP excluded",\n'
        '        "arm_names": {"fused": "token_output", "separate": "base_one_launch"},\n'
        '        "extra_storage": "None; both C ABIs use the same 32 tensors",')
    source = once(source,
        '"libraries": {str(path): digest(path) for path in libraries},',
        '"libraries": {str(path): digest(path) for path in (*libraries, args.candidate_library)},')
    return source


def prepare(parent, output):
    if digest(parent) != PARENT:
        raise ValueError("Use the frozen timing-v3 benchmark")
    original = parent.read_text()
    source = transform(original)
    compile(source, "bench.py", "exec")
    output.mkdir(parents=True, exist_ok=False)
    (output / "bench.py").write_text(source)
    (output / "benchmark.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), source.splitlines(keepends=True),
        fromfile="timing-v3/bench.py", tofile="token-output/bench.py")))
    for name in ("prepare_bench.py", "bench_abi_probe.cpp", "test_bench.py", "bench_run.sh"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    (output / "parent").mkdir()
    for name in ("bench.py", "bench_abi_probe.cpp"):
        shutil.copy2(parent.with_name(name), output / "parent" / name)
    report = {
        "parent_benchmark_sha256": PARENT,
        "sources": {str(p.relative_to(output)): digest(p)
                    for p in output.rglob("*") if p.is_file()},
        "comparison": "Two one-launch C ABIs share all 32 tensor addresses in each of two original owning-Torch arenas",
        "extra_storage_bytes": 0,
        "preserved_gates": [
            "Finite CPU reference and poisoned writable boundaries before each checked replay",
            "Separate original owning-Torch validators retain their TensorImpl metadata",
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
