"""Freeze a 32-common-address comparison with each arm's immutable Down tiling."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT = "cd7d20dd06b768d9c9f4255a80c11dc82f811e5856895e6a16637e97a9374e14"
TILINGS = "1342242f984c4972a0e31af5684d7fc9dc0ce07e7570633b8fad646de1ef50ba"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one benchmark seam: {before!r}")
    return source.replace(before, after, 1)


OWNER = '''    def owning_torch(self, owner):
        b, r, sync = self.arena
        torch, bank = self.torch, self.bank
        namespace = ("tilert_ffn_down_n512" if owner == "fused"
                     else "tilert_ffn_original_shared_gu_static")
        getattr(torch.ops, namespace).forward_out(
            self.hidden, self.residual, bank["gamma"], bank["gate"], bank["bias"],
            bank["routed"], bank["shared"], b, [*r, self.extra],
            1e-5, self.blocks, self.layout, sync)


def prepare_arena(torch, launchers, hidden, blocks, owner):
    namespace = ("tilert_ffn_down_n512" if owner == "fused"
                 else "tilert_ffn_original_shared_gu_static")
    bridge, body, sync = getattr(torch.ops, namespace).prepare(hidden, blocks, 1)
    launchers.extra_by_arena[bridge[8].data_ptr()] = body[20]
    launchers.down_tilings[owner] = body[7]
    launchers.down_owners[owner] = (bridge, body, sync)
    return bridge, body[:20], sync


def check_down_tilings(torch, launchers):
    records = {}
    for arm, expected in EXPECTED_DOWN_TILINGS["arms"].items():
        value = launchers.down_tilings[arm]
        if value is not launchers.down_owners[arm][1][7]:
            raise RuntimeError("Down tiling lost its original prepared Tensor owner")
        if value.dtype != torch.uint8 or value.ndim != 1 or value.numel() < 200:
            raise RuntimeError("Down tiling must be an explicit uint8 buffer of at least 200 bytes")
        payload = bytes(value[:200].detach().cpu().tolist())
        if payload.hex() != expected["layouts"]["1"]["hex"]:
            raise RuntimeError(f"Prepared Down tiling differs from the loaded host tiler: {arm}")
        records[arm] = {"address": value.data_ptr(), "bytes": 200,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "hex": payload.hex(), "pass": True}
    if records["fused"]["address"] == records["separate"]["address"]:
        raise RuntimeError("Distinct Down geometries cannot share their tiling storage")
    return records
'''


def transform(source):
    path = Path(__file__).with_name("expected-tilings.json")
    if digest(path) != TILINGS:
        raise ValueError("The loaded host-tiler byte oracle changed")
    expected = json.loads(path.read_text())
    source = once(source, "class DispatchInput(ctypes.Structure):",
                  "EXPECTED_DOWN_TILINGS = " + repr(expected) +
                  "\n\n\nclass DispatchInput(ctypes.Structure):")
    replacements = (
        ("self.base = self.libraries[-1].ffn_original_shared_terminal_launch",
         "self.base = self.libraries[-1].ffn_original_shared_gu_static_launch"),
        ("self.fused = self.libraries[-1].ffn_original_shared_gu_static_launch",
         "self.fused = self.libraries[-1].ffn_down_n512_launch"),
        ('"scope": "Matched-address static-GU FFN with original shared quantization versus its token-terminal parent, TP excluded",',
         '"scope": "Down N512 versus N256 FFN with 32 common scratch addresses and private prepared Down tilings, TP excluded",'),
        ('"arm_names": {"fused": "original_shared_gu_static", "separate": "original_shared_terminal_parent"},',
         '"arm_names": {"fused": "down_n512", "separate": "down_n256_parent"},'),
        ("All 33 tensors, including the INT32[264] active table, share exact addresses within each pair",
         "32 tensors share addresses; body[7] is each arm's immutable prepared Down tiling"),
        ("        self.extra_by_arena = {}",
         "        self.extra_by_arena = {}\n        self.down_tilings = {}\n        self.down_owners = {}"),
        ('            cb, cr, cs = prepare_arena(torch, launchers, hidden, args.blocks)',
         '            cb, cr, cs = prepare_arena(torch, launchers, hidden, args.blocks, "fused")'),
        ('            sb, sr, ss = prepare_arena(torch, launchers, hidden, args.blocks)',
         '            sb, sr, ss = prepare_arena(torch, launchers, hidden, args.blocks, "separate")'),
        ('            for slot in (6, 7, 18):', '            for slot in (6, 18):'),
    )
    for before, after in replacements:
        source = once(source, before, after)
    begin = source.index("    def owning_torch(self, owner):")
    end = source.index("\n\ndef load_module", begin)
    source = source[:begin] + OWNER + source[end:]
    source = once(source,
        '        self.candidate_body_ptrs = pointers([*r, self.extra]) if launchers.kind == "active" else self.body_ptrs',
        '''        self.body_by_arm = {arm: [*r, self.extra] for arm in ("fused", "separate")}
        for arm, values in self.body_by_arm.items():
            values[7] = launchers.down_tilings[arm]
        self.candidate_body_ptrs = pointers(self.body_by_arm["fused"])
        self.base_body_ptrs = pointers(self.body_by_arm["separate"])''')
    source = once(source,
        '''        self.launchers.base(stream, self.sync, *self.input_arguments(), self.weight_ptrs,
            self.bridge_ptrs, self.candidate_body_ptrs, self.shared, self.rows, self.blocks, self.layout, 1e-5)''',
        '''        self.launchers.base(stream, self.sync, *self.input_arguments(), self.weight_ptrs,
            self.bridge_ptrs, self.base_body_ptrs, self.shared, self.rows, self.blocks, self.layout, 1e-5)''')
    source = once(source, '                result["extra_checks"] = []',
        '''                result["extra_checks"] = []
                result["down_tiling_checks"] = [{"stage": "before_capture",
                    "values": check_down_tilings(torch, launchers)}]
                result["body_addresses_by_arm"] = {
                    arm: [value.data_ptr() for value in calls[0].body_by_arm[arm]]
                    for arm in ("fused", "separate")}
                for slot, (left, right) in enumerate(zip(
                        result["body_addresses_by_arm"]["fused"],
                        result["body_addresses_by_arm"]["separate"], strict=True)):
                    if (left == right) != (slot != 7):
                        raise RuntimeError("Only Down tiling may have distinct body addresses")''')
    source = once(source, '                for iteration in range(args.replays):',
        '''                result["down_tiling_checks"].append({"stage": "after_capture",
                    "values": check_down_tilings(torch, launchers)})
                for iteration in range(args.replays):''')
    source = once(source, '                expected_tail = {key: expected_banks[-1][key] for key in',
        '''                result["down_tiling_checks"].append({"stage": "after_timing_capture",
                    "values": check_down_tilings(torch, launchers)})
                expected_tail = {key: expected_banks[-1][key] for key in''')
    source = once(source, '                for arm in ("fused", "separate"):\n                    for index, call in enumerate(calls):',
        '''                result["down_tiling_checks"].append({"stage": "after_timing",
                    "values": check_down_tilings(torch, launchers)})
                for arm in ("fused", "separate"):
                    for index, call in enumerate(calls):''')
    source = once(source, '                result["pass"] = True',
        '''                result["down_tiling_checks"].append({"stage": "after_postchecks",
                    "values": check_down_tilings(torch, launchers)})
                result["pass"] = True''')
    return source


def prepare(parent, output):
    if digest(parent) != PARENT:
        raise ValueError("Use the frozen original-shared static-GU timing-v1 benchmark")
    source = transform(parent.read_text())
    compile(source, "bench.py", "exec")
    output.mkdir(parents=True, exist_ok=False)
    (output / "bench.py").write_text(source)
    (output / "benchmark.patch").write_text("".join(difflib.unified_diff(
        parent.read_text().splitlines(keepends=True), source.splitlines(keepends=True),
        fromfile="original-shared-gu-static-timing-v1/bench.py",
        tofile="down-n512/bench.py")))
    for name in ("prepare_bench.py", "bench_abi_probe.cpp", "test_bench.py",
                 "bench_run.sh", "expected-tilings.json"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    (output / "parent").mkdir()
    shutil.copy2(parent, output / "parent/bench.py")
    shutil.copy2(parent.parent / "parent/bench_abi_probe.cpp",
                 output / "parent/bench_abi_probe.cpp")
    report = {
        "parent_benchmark_sha256": PARENT,
        "control_library_sha256": "381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2",
        "candidate_library_sha256": "c90b3666a1697070eda19d5b3e133e9ed58f36fe429abefceae6ffe102a6a032",
        "comparison": "32 common tensor addresses; per-arm prepared body[7] Down tiling",
        "tiling_oracle_sha256": TILINGS,
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
