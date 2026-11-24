"""Derive matched-address variant comparisons from the frozen FFN timing gate."""

import argparse
import hashlib
import json
from pathlib import Path


PARENT = "eee6ab5549c727d20fd60c5bbf5495e60af84dedac061be0c2e757427df39f62"


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one benchmark seam: {before!r}")
    return source.replace(before, after, 1)


EXTRA_CHECK = '''
    def check_extra(arena, expected):
        extra = launchers.extra_by_arena[arena[0][8].data_ptr()]
        actual = extra.cpu()
        if args.candidate_kind == "active":
            ids = (expected["body.counts"].cpu() != 0).nonzero().reshape(-1).to(torch.int32)
            reference = torch.zeros(264, dtype=torch.int32)
            reference[0] = ids.numel()
            reference[8:8 + ids.numel()] = ids
            comparison = bridge.compare(torch, actual, reference)
            if not comparison["bitwise_equal"]:
                raise RuntimeError("Active-expert table did not match reference counts")
            return comparison
        records = actual.tolist()
        for slot, record in enumerate(records):
            kind = int(slot >= args.blocks)
            index = slot - args.blocks if kind else slot
            if record[:5] != [1, kind, index, args.rows, args.blocks]:
                raise RuntimeError(f"Profile identity differs at worker {slot}")
            marks = record[8:21]
            if marks[0] <= 0 or any(b < a for a, b in zip(marks, marks[1:])):
                raise RuntimeError(f"Profile clock order differs at worker {slot}")
        return {"valid_worker_traces": len(records)}
'''


def transform(source):
    source = once(source, '    def __init__(self, fused, bridge, body):',
                  '    def __init__(self, fused, bridge, body, candidate, kind):')
    source = once(source, '        self.fused = self.libraries[0].ffn_composed_launch',
        '        self.base = self.libraries[0].ffn_composed_launch\n'
        '        self.libraries.append(ctypes.CDLL(str(candidate.resolve())))\n'
        '        self.fused = getattr(self.libraries[-1], "ffn_" + kind + "_launch")\n'
        '        self.kind = kind\n'
        '        self.extra_by_arena = {}')
    source = once(source, '        self.bridge = self.libraries[1].ffn_bridge_launch',
        '        self.base.argtypes = list(self.fused.argtypes)\n'
        '        self.base.restype = None\n'
        '        if kind == "profile":\n'
        '            self.fused.argtypes = [*self.fused.argtypes, ctypes.c_void_p]\n'
        '        self.bridge = self.libraries[1].ffn_bridge_launch')
    source = once(source, '        self.weight_ptrs = pointers(bank["routed"])',
        '        self.weight_ptrs = pointers(bank["routed"])\n'
        '        key = b[8].data_ptr()\n'
        '        if key not in launchers.extra_by_arena:\n'
        '            shape = (264,) if launchers.kind == "active" else (3 * blocks, 32)\n'
        '            dtype = torch.int32 if launchers.kind == "active" else torch.int64\n'
        '            extra = torch.empty(shape, dtype=dtype, device=hidden.device)\n'
        '            if extra.data_ptr() in {t.data_ptr() for t in (*b, *r)}:\n'
        '                raise RuntimeError("Extra variant storage aliases the common arena")\n'
        '            launchers.extra_by_arena[key] = extra\n'
        '        self.extra = launchers.extra_by_arena[key]\n'
        '        self.candidate_body_ptrs = pointers([*r, self.extra]) if launchers.kind == "active" else self.body_ptrs')
    source = once(source,
        '            self.bridge_ptrs, self.body_ptrs, self.shared, self.rows, self.blocks, self.layout, 1e-5)\n',
        '            self.bridge_ptrs, self.candidate_body_ptrs, self.shared, self.rows, self.blocks, self.layout, 1e-5,\n'
        '            *([self.extra.data_ptr()] if self.launchers.kind == "profile" else []))\n')
    begin = source.index('    def separate(self):\n')
    end = source.index('    def owning_torch(self, owner):\n', begin)
    source = source[:begin] + '''    def separate(self):
        stream = self.torch.npu.current_stream().npu_stream
        self.launchers.base(stream, self.sync, *self.input_arguments(), self.weight_ptrs,
            self.bridge_ptrs, self.body_ptrs, self.shared, self.rows, self.blocks, self.layout, 1e-5)

''' + source[end:]
    source = once(source, '    parser.add_argument("--device", default="npu:0")',
        '    parser.add_argument("--candidate-library", type=Path, required=True)\n'
        '    parser.add_argument("--candidate-kind", choices=("active", "profile"), required=True)\n'
        '    parser.add_argument("--device", default="npu:0")')
    source = once(source, '    launchers = Launchers(*libraries)',
                  '    launchers = Launchers(*libraries, args.candidate_library, args.candidate_kind)')
    source = once(source,
        '"scope": "Frozen C ABI graph replay: one FFN launch versus bridge+body+epilogue, TP excluded",',
        '"scope": "Matched-address candidate FFN versus accepted one-launch FFN, TP excluded",\n'
        '        "arm_names": {"fused": args.candidate_kind, "separate": "base_one_launch"},\n'
        '        "candidate_library": {"path": str(args.candidate_library), "sha256": digest(args.candidate_library)},')
    source = once(source,
        '"libraries": {str(path): digest(path) for path in libraries},',
        '"libraries": {str(path): digest(path) for path in (*libraries, args.candidate_library)},')
    source = once(source,
        '        poison_outputs(checker.all_values(torch, arena[0], arena[1], args.rows))',
        '        poison_outputs(checker.all_values(torch, arena[0], arena[1], args.rows))\n'
        '        extra = launchers.extra_by_arena[arena[0][8].data_ptr()]\n'
        '        if args.candidate_kind == "active":\n'
        '            extra.fill_(1)\n'
        '            extra[0].zero_()\n'
        '        else:\n'
        '            extra.zero_()')
    source = once(source, '    def checked(actual, expected, label):',
                  EXTRA_CHECK + '\n    def checked(actual, expected, label):')
    source = once(source,
        '                            checks = checked(snapshot(arena), expected, label)',
        '                            checks = checked(snapshot(arena), expected, label)\n'
        '                            if arm == "fused":\n'
        '                                result["extra_checks"].append({"name": label,\n'
        '                                    "checks": check_extra(arena, expected)})')
    source = once(source,
        '                        checked(actual_tail, expected_tail, f"{owner}-sample{sample}-{arm}-tail")',
        '                        checked(actual_tail, expected_tail, f"{owner}-sample{sample}-{arm}-tail")\n'
        '                        if arm == "fused":\n'
        '                            result["extra_checks"].append({"name": f"sample{sample}-tail",\n'
        '                                "checks": check_extra(arena, expected_tail)})')
    source = once(source,
        '                        checks = checked(snapshot(arena), expected_banks[index], f"{owner}-{arm}-bank{index}-post")',
        '                        checks = checked(snapshot(arena), expected_banks[index], f"{owner}-{arm}-bank{index}-post")\n'
        '                        if arm == "fused":\n'
        '                            result["extra_checks"].append({"name": f"bank{index}-post",\n'
        '                                "checks": check_extra(arena, expected_banks[index])})')
    source = once(source,
        '                report["arenas"].append(result)',
        '                result["extra_checks"] = []\n'
        '                result["extra_storage"] = {"address": calls[0].extra.data_ptr(),\n'
        '                    "shape": list(calls[0].extra.shape), "dtype": str(calls[0].extra.dtype),\n'
        '                    "bytes": calls[0].extra.nbytes}\n'
        '                report["arenas"].append(result)')
    return source


def prepare(parent, output):
    data = parent.read_bytes()
    if hashlib.sha256(data).hexdigest() != PARENT:
        raise ValueError("Use the frozen timing-v3 benchmark")
    source = transform(data.decode())
    compile(source, "bench.py", "exec")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "bench.py"
    path.write_text(source)
    report = {
        "parent_benchmark_sha256": PARENT,
        "sources": {"bench.py": hashlib.sha256(path.read_bytes()).hexdigest()},
        "comparison": "Candidate versus base one-launch C ABI with identical 32 common tensor addresses in two owner arenas",
        "preserved_gates": ["finite reference and poisoned writable boundaries before each checked replay",
                            "16 independent equal-valued weight banks",
                            "30 alternating event pairs per arena",
                            "independent reference snapshots and every bank checked after timing"],
        "extra_storage": "One disjoint extra tensor per arena. Before replay: active table has count zero and remaining slots one; trace is zero. The full active table or worker trace is checked after candidate replay.",
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.parent.resolve(), args.output.resolve())
