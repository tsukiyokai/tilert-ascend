"""Freeze a same-address expert-ready GU comparison with original shared quantization."""

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
         "        self.base = self.libraries[-1].ffn_original_shared_gu_static_launch"),
        ("        self.fused = self.libraries[-1].ffn_active_token_join_launch",
         "        self.fused = self.libraries[-1].ffn_gu_static_ready_launch"),
        ('"scope": "Matched-address active-token-join FFN versus active-descriptor FFN, TP excluded",',
         '"scope": "Matched-address expert-ready GU FFN with original shared quantization versus its static-GU parent, TP excluded",'),
        ('"arm_names": {"fused": "active_token_join", "separate": "active_descriptor"},',
         '"arm_names": {"fused": "gu_static_ready", "separate": "original_shared_gu_static_parent"},'),
    )
    for before, after in replacements:
        source = once(source, before, after)
    begin = source.index("    def owning_torch(self, owner):")
    end = source.index("\n\ndef load_module", begin)
    source = source[:begin] + CORRECTED_OWNER + source[end:]
    source = once(source, ORIGINAL_PREPARE, CORRECTED_PREPARE)
    source = once(source,
        "Owning frozen Torch native chain, checked separately against production before this timing experiment",
        "Original-shared complete FFN Torch entry, independently compared with the original DecoderLayer norm/MLP")
    source = once(source,
        "Each arena's owning Torch chain checks shapes/tiling identity/aliasing before CABI comparison",
        "Original-shared Torch validator checks all 33 tensors in both independently allocated arenas before CABI comparison")
    return source


CORRECTED_OWNER = """    def owning_torch(self, owner):
        b, r, sync = self.arena
        torch, bank = self.torch, self.bank
        torch.ops.tilert_ffn_original_shared_gu_static.forward_out(
            self.hidden, self.residual, bank["gamma"], bank["gate"], bank["bias"],
            bank["routed"], bank["shared"], b, [*r, self.extra],
            1e-5, self.blocks, self.layout, sync)


def prepare_arena(torch, launchers, hidden, blocks):
    bridge, body, sync = torch.ops.tilert_ffn_original_shared_gu_static.prepare(
        hidden, blocks, 1)
    launchers.extra_by_arena[bridge[8].data_ptr()] = body[20]
    return bridge, body[:20], sync
"""


ORIGINAL_PREPARE = """            cb, cr, cs = torch.ops.tilert_ffn_composed.prepare(hidden, args.blocks, 1)
            sb, ss = torch.ops.tilert_ffn_bridge.prepare(hidden, args.blocks)
            sr, rs = torch.ops.tilert_pipeline_body.routed_shared_prepare(sb[6], 256, args.blocks, 1)
            if cs != ss or cs != rs:
"""
CORRECTED_PREPARE = """            cb, cr, cs = prepare_arena(torch, launchers, hidden, args.blocks)
            sb, sr, ss = prepare_arena(torch, launchers, hidden, args.blocks)
            if cs != ss:
"""


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
        tofile="gu-static-ready/bench.py")))
    for name in ("prepare_bench.py", "bench_abi_probe.cpp", "test_bench.py", "bench_run.sh"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    (output / "parent").mkdir()
    shutil.copy2(parent, output / "parent/bench.py")
    shutil.copy2(parent.parent / "parent/bench_abi_probe.cpp",
                 output / "parent/bench_abi_probe.cpp")
    report = {
        "parent_benchmark_sha256": PARENT,
        "control_library_sha256": "381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2",
        "candidate_library_sha256": "affc88ba3f4bdf30b569ec14cc58a6c3d67a32b23fbb1a58f959571a7890d488",
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
