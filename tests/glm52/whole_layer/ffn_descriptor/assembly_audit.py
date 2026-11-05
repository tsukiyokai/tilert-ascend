"""Count static instructions in the four actual FFN device entry functions."""

import argparse
import hashlib
import json
from pathlib import Path
import re


FUNCTION = re.compile(r"(?m)^(?=[0-9a-f]{16} <)")
INSTRUCTION = re.compile(r"^\s*([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)(.*)$")


def inspect(path):
    source = path.read_text()
    result = {}
    for function in FUNCTION.split(source)[1:]:
        lines = function.splitlines()
        symbol = lines[0]
        key = ("nz" if "CubeFormat1" in symbol else "nd")
        key += "_aic" if symbol.endswith("_mix_aic>:") else "_aiv"
        if key in result or not symbol.endswith(("_mix_aic>:", "_mix_aiv>:")):
            raise ValueError(f"Unexpected device function: {symbol}")
        instructions = [match for line in lines[1:]
                        if (match := INSTRUCTION.match(line))]
        loads = lambda op: op.startswith(("LD.", "LDP."))
        stores = lambda op: op.startswith(("ST.", "STP.", "STI."))
        result[key] = {
            "symbol": symbol,
            "start": hex(int(instructions[0][1], 16)),
            "last_instruction": hex(int(instructions[-1][1], 16)),
            "instructions": len(instructions),
            "load_instructions": sum(loads(row[2]) for row in instructions),
            "store_instructions": sum(stores(row[2]) for row in instructions),
            "direct_x30_memory_instructions": sum(
                "[X30]" in row[3] and (loads(row[2]) or stores(row[2]))
                for row in instructions),
            "sys_cnt_reads": sum("SYS_CNT" in row[3] for row in instructions),
            "first_40_instructions": [row[0] for row in instructions[:40]],
        }
    if set(result) != {"nz_aic", "nd_aic", "nz_aiv", "nd_aiv"}:
        raise ValueError("Expected four mixed kernel entry functions")
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "functions": result}


def compare(parent, candidate):
    report = {"parent": inspect(parent), "candidate": inspect(candidate)}
    report["candidate_minus_parent"] = {}
    metrics = ("instructions", "load_instructions", "store_instructions",
               "direct_x30_memory_instructions", "sys_cnt_reads")
    for key, original in report["parent"]["functions"].items():
        actual = report["candidate"]["functions"][key]
        if original["sys_cnt_reads"] or actual["sys_cnt_reads"]:
            raise ValueError("The comparison must contain no clock instrumentation")
        report["candidate_minus_parent"][key] = {
            name: actual[name] - original[name] for name in metrics}
    report["scope"] = (
        "Static disassembly counts over each complete function, not executed instruction counts, "
        "memory traffic, stack frame size, spill latency, or a body-entry timing attribution")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare(args.parent, args.candidate), indent=2) + "\n")
