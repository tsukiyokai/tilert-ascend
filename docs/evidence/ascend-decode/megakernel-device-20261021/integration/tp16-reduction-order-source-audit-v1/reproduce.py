"""CPU order witnesses and a bounded-stack schedule for the pinned AIV tree."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct


def float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def describe(value):
    bits = float_bits(value)
    assert math.isfinite(value)
    return {"value": value, "fp32_bits": f"0x{bits:08x}",
            "exact_bf16": bits & 0xFFFF == 0,
            "bf16_bits_if_exact": f"0x{bits >> 16:04x}"
            if bits & 0xFFFF == 0 else None}


def lower_power(count):
    """Largest power of two strictly below count, including powers of two."""
    assert count > 1
    return 1 << ((count - 1).bit_length() - 1)


def source_rounds(values):
    slots = list(values)
    rounds = []
    while len(slots) > 1:
        power = lower_power(len(slots))
        updated = slots[:power]
        pairs = []
        for front in range(len(slots) - power):
            back = power + front
            updated[front] = float32(slots[front] + slots[back])
            pairs.append({"front": front, "back": back,
                          "left": describe(slots[front]),
                          "right": describe(slots[back]),
                          "result": describe(updated[front])})
        rounds.append({"current_blocks": len(slots), "power": power,
                       "pairs": pairs,
                       "unchanged_fronts": list(range(len(slots) - power,
                                                       power))})
        slots = updated
    return slots[0], rounds


def serial_sum(values):
    total = values[0]
    steps = []
    for rank, value in enumerate(values[1:], 1):
        before = total
        total = float32(total + value)
        steps.append({"rank": rank, "left": describe(before),
                      "right": describe(value), "result": describe(total)})
    return total, steps


def source_tree(ranks):
    slots = list(range(ranks))
    while len(slots) > 1:
        power = lower_power(len(slots))
        updated = slots[:power]
        for front in range(len(slots) - power):
            updated[front] = (slots[front], slots[power + front])
        slots = updated
    return slots[0]


def postorder(node):
    if isinstance(node, int):
        return [("load", node)]
    return postorder(node[0]) + postorder(node[1]) + [("add",)]


def slot_program(ranks):
    depth = 0
    peak = 0
    program = []
    for operation in postorder(source_tree(ranks)):
        if operation[0] == "load":
            program.append({"op": "load_bf16_as_fp32",
                            "rank": operation[1], "dst": depth})
            depth += 1
            peak = max(peak, depth)
        else:
            assert depth >= 2
            program.append({"op": "add_fp32", "dst": depth - 2,
                            "left": depth - 2, "right": depth - 1})
            depth -= 1
    assert depth == 1
    return program, peak


def execute_program(values, program):
    slots = [None] * 5
    live = set()
    for operation in program:
        destination = operation["dst"]
        if operation["op"] == "load_bf16_as_fp32":
            assert destination not in live
            slots[destination] = values[operation["rank"]]
            live.add(destination)
        else:
            left, right = operation["left"], operation["right"]
            assert left in live and right in live and left != right
            assert destination == left
            slots[left] = float32(slots[left] + slots[right])
            live.remove(right)
    assert live == {0}
    return slots[0]


def witness(values):
    assert all(describe(value)["exact_bf16"] for value in values)
    serial, steps = serial_sum(values)
    tree, rounds = source_rounds(values)
    assert float_bits(serial) != float_bits(tree)
    assert describe(serial)["exact_bf16"] and describe(tree)["exact_bf16"]
    return {"inputs": [describe(value) for value in values],
            "serial": describe(serial), "tree": describe(tree),
            "serial_steps": steps, "tree_rounds": rounds}


def schedule_checks():
    rng = random.Random(20261022)
    results = []
    for ranks in range(2, 17):
        program, peak = slot_program(ranks)
        loads = [op["rank"] for op in program
                 if op["op"] == "load_bf16_as_fp32"]
        assert sorted(loads) == list(range(ranks))
        assert sum(op["op"] == "add_fp32" for op in program) == ranks - 1
        assert peak <= 5
        for _ in range(256):
            values = [float32(rng.randint(-255, 255)
                              * 2.0 ** rng.randint(-30, 30))
                      for _ in range(ranks)]
            assert all(describe(value)["exact_bf16"] for value in values)
            expected, _ = source_rounds(values)
            actual = execute_program(values, program)
            assert float_bits(actual) == float_bits(expected)
        results.append({"ranks": ranks, "leaf_order": loads,
                        "fp32_live_vectors": peak, "program": program,
                        "random_bitwise_checks": 256,
                        "ub_bytes_at_2048": 4096 + peak * 2048 * 4})
    return results


def resource_counts():
    values = []
    for rows in (1, 12, 32):
        elements = rows * 6144
        for ranks in (2, 3, 16):
            # A directed payload is counted once, without link framing.
            off_rank = 2 * (ranks - 1) * elements
            values.append({
                "rows": rows, "ranks": ranks,
                "c2_submitted_bytes_per_rank_per_ar": 2 * ranks * elements,
                "c2_off_rank_bytes_per_rank_per_ar": off_rank,
                "c2_off_rank_bytes_all_ranks_per_ar": ranks * off_rank,
                "c2_window_bytes_per_rank_at_capacity_32": (
                    64 + 2 * ranks * 64 + 2 * ranks * 32 * 6144 * 2),
                "hccl_fp32_partition_gather_broadcast_off_rank_bytes": (
                    8 * elements * (ranks - 1) // ranks),
                "c2_to_hccl_payload_ratio": ranks / 4,
            })
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    epsilon = 2.0 ** -25
    rank3 = witness([1.0, epsilon, -1.0])
    rank16 = witness([1.0, epsilon, -1.0, epsilon] + [0.0] * 12)
    assert rank3["serial"]["value"] == 0.0
    assert rank3["tree"]["value"] == epsilon
    assert rank16["serial"]["value"] == epsilon
    assert rank16["tree"]["value"] == 2.0 ** -24
    schedules = schedule_checks()
    assert schedules[-1]["fp32_live_vectors"] == 5
    report = {
        "scope": "CPU IEEE FP32 order witnesses and schedule legality; no NPU",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pass": True, "witness_rank3": rank3, "witness_rank16": rank16,
        "schedule_checks": schedules,
        "random_bitwise_checks": 15 * 256,
        "tree_ub_2048_bytes": 4096 + 5 * 2048 * 4,
        "tree_ub_768_bytes": 4096 + 5 * 768 * 4,
        "resource_counts": resource_counts(),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": True, "rank3": rank3["tree"],
                      "rank16": rank16["tree"], "max_vectors": 5,
                      "checks": report["random_bitwise_checks"]}))


if __name__ == "__main__":
    main()
