"""Compare the actual embedded CANN device binaries without running a kernel."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def extract_kernel(path, output, objdump):
    data = path.read_bytes()
    offset = data.find(b"\x7fELF", 1)
    if offset < 0:
        raise ValueError("The host object has no embedded device ELF")
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", data, offset)
    if header[2] != 4137:
        raise ValueError("The first embedded ELF is not the device binary")
    end = offset + header[6] + header[11] * header[12]
    if end > len(data):
        raise ValueError("The embedded ELF section table is out of bounds")
    elf = data[offset:end]
    output.write_bytes(elf)
    disassembly = subprocess.check_output(
        [str(objdump), "-d", "--demangle", str(output)], text=True)
    output.with_suffix(".txt").write_text(disassembly)
    return parse(disassembly), {"object_sha256": sha256(data),
                                "device_elf_sha256": sha256(elf)}


def parse(disassembly):
    symbols = {}
    current = None
    for line in disassembly.splitlines():
        symbol = re.fullmatch(r"([0-9a-f]+) <(.+)>:", line)
        if symbol:
            name = symbol[2]
            if "KernelIL10CubeFormat" in name and name.endswith(("_mix_aic", "_mix_aiv")):
                layout = re.search(r"CubeFormat([01])", name)[1]
                current = ("nz" if layout == "1" else "nd") + (
                    "_aic" if name.endswith("_mix_aic") else "_aiv")
                symbols[current] = {"symbol": name, "instructions": []}
            else:
                current = None
            continue
        instruction = re.match(r"\s*([0-9a-f]+):\s+([0-9a-f]{8})\s+(\S+)\s*(.*)", line)
        if current and instruction:
            symbols[current]["instructions"].append({
                "address": instruction[1], "word": instruction[2],
                "opcode": instruction[3], "operands": instruction[4],
            })
    if set(symbols) != {"nd_aic", "nd_aiv", "nz_aic", "nz_aiv"}:
        raise ValueError(f"Unexpected mixed-kernel symbol set: {sorted(symbols)}")
    return symbols


def compare(parent, candidate):
    result = {}
    for name in parent:
        old = parent[name]["instructions"]
        new = candidate[name]["instructions"]
        old_counts = Counter(item["opcode"] for item in old)
        new_counts = Counter(item["opcode"] for item in new)
        result[name] = {
            "parent_instruction_count": len(old),
            "candidate_instruction_count": len(new),
            "instruction_delta": len(new) - len(old),
            "instruction_words_equal": [item["word"] for item in old]
            == [item["word"] for item in new],
            "opcode_delta": {op: new_counts[op] - old_counts[op]
                             for op in sorted(old_counts.keys() | new_counts.keys())
                             if old_counts[op] != new_counts[op]},
            "parent_opcode_counts": dict(sorted(old_counts.items())),
            "candidate_opcode_counts": dict(sorted(new_counts.items())),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--objdump", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    old, old_identity = extract_kernel(
        args.parent, args.output / "parent.elf", args.objdump)
    new, new_identity = extract_kernel(
        args.candidate, args.output / "candidate.elf", args.objdump)
    result = {"scope": "Static instructions in four complete FFN device variants; no timing",
              "parent": old_identity, "candidate": new_identity,
              "objdump_sha256": sha256(args.objdump.read_bytes()),
              "variants": compare(old, new)}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, value in result["variants"].items():
        print(name, value["parent_instruction_count"],
              value["candidate_instruction_count"], value["instruction_words_equal"])


if __name__ == "__main__":
    main()
