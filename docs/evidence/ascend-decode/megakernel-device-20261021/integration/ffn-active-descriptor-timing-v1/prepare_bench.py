"""Compare combined and active FFN using one shared 33-buffer arena."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT = "15ed8ad01e71e34703fb3628403f067568a0aabcd3db08d6a74d5402a592eb58"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Expected one benchmark seam: {before!r}")
    return source.replace(before, after, 1)


def check_both_active_tables(source):
    lines = source.splitlines(keepends=True)
    result = []
    changed = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        if (line.strip() == 'if arm == "fused":'
                and 'result["extra_checks"].append' in lines[index + 1]):
            indentation = len(line) - len(line.lstrip())
            index += 1
            while index < len(lines):
                nested = lines[index]
                if len(nested) - len(nested.lstrip()) <= indentation:
                    break
                result.append(nested[4:])
                index += 1
            changed += 1
        else:
            result.append(line)
            index += 1
    if changed != 3:
        raise ValueError("Expected precheck, event-tail and postcheck table gates")
    return "".join(result).replace(
        'result["extra_checks"].append({"name":',
        'result["extra_checks"].append({"arm": arm, "name":')


def transform(source):
    source = once(source,
        "    def __init__(self, fused, bridge, body, candidate, kind):",
        "    def __init__(self, fused, bridge, body, candidate, kind, active):")
    source = once(source,
        "        self.base = self.libraries[0].ffn_composed_launch",
        "        self.libraries.append(ctypes.CDLL(str(active.resolve())))\n"
        "        self.base = self.libraries[-1].ffn_active_launch")
    source = once(source,
        '        self.fused = getattr(self.libraries[-1], "ffn_" + kind + "_launch")',
        "        self.fused = self.libraries[-1].ffn_active_descriptor_launch")
    source = once(source,
        "            self.bridge_ptrs, self.body_ptrs, self.shared, self.rows, self.blocks, self.layout, 1e-5)",
        "            self.bridge_ptrs, self.candidate_body_ptrs, self.shared, self.rows, self.blocks, self.layout, 1e-5)")
    source = once(source,
        '    parser.add_argument("--candidate-kind", choices=("active", "profile"), required=True)',
        '    parser.add_argument("--candidate-kind", choices=("active",), required=True)\n'
        '    parser.add_argument("--active-library", type=Path, required=True)')
    source = once(source, "    for path in libraries:\n",
        "    for path in (*libraries, args.active_library, args.candidate_library):\n")
    source = once(source,
        "    launchers = Launchers(*libraries, args.candidate_library, args.candidate_kind)",
        "    launchers = Launchers(*libraries, args.candidate_library,\n"
        "                          args.candidate_kind, args.active_library)")
    source = once(source,
        '"scope": "Matched-address candidate FFN versus accepted one-launch FFN, TP excluded",',
        '"scope": "Matched-address active-descriptor FFN versus active FFN, TP excluded",')
    source = once(source,
        '"arm_names": {"fused": args.candidate_kind, "separate": "base_one_launch"},',
        '"arm_names": {"fused": "active_descriptor", "separate": "active"},')
    source = once(source,
        '"scratch_matching": "All 32 tensors including system workspaces and immutable tilings share exact addresses within each pair",',
        '"scratch_matching": "All 33 tensors, including the INT32[264] active table, share exact addresses within each pair",')
    source = once(source,
        '"libraries": {str(path): digest(path) for path in (*libraries, args.candidate_library)},',
        '"libraries": {str(path): digest(path) for path in (*libraries, args.active_library, args.candidate_library)},')
    return check_both_active_tables(source)


def prepare(parent, output):
    if digest(parent) != PARENT:
        raise ValueError("Use the frozen FFN variant timing-v3 benchmark")
    original = parent.read_text()
    source = transform(original)
    compile(source, "bench.py", "exec")
    output.mkdir(parents=True, exist_ok=False)
    (output / "bench.py").write_text(source)
    (output / "benchmark.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), source.splitlines(keepends=True),
        fromfile="variant-timing-v3/bench.py",
        tofile="active-descriptor/bench.py")))
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
        "comparison": "All 33 tensors shared by active and combined C ABIs",
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
