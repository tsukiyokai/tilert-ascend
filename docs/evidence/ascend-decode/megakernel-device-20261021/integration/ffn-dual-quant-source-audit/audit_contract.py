"""Check address and reduction-tree obligations without emulating an NPU."""

import argparse
import hashlib
import json
import struct
from pathlib import Path


HIDDEN = 6144
WIDTHS = (1, 2, 12, 32)
GROUPS = (16, 24)


def float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def reduction_tree(alias):
    source = list(range(HIDDEN))
    work = source if alias else [None] * HIDDEN
    for repeat in range(96):
        inputs = tuple(source[64 * repeat:64 * (repeat + 1)])
        work[2 * repeat] = ("vcmax", inputs)
        work[2 * repeat + 1] = ("unused_index", repeat)
        assert 2 * repeat + 1 < 64 * (repeat + 1)
    for repeat in range(3):
        inputs = tuple(work[64 * repeat:64 * (repeat + 1):2])
        work[2 * repeat] = ("vcmax", inputs)
        work[2 * repeat + 1] = ("unused_index", repeat)
        assert 2 * repeat + 1 < 64 * (repeat + 1)
    return ("vcmax", tuple(work[0:6:2]))


def leaves(node):
    if isinstance(node, int):
        return [node]
    assert node[0] == "vcmax"
    return [leaf for child in node[1] for leaf in leaves(child)]


def check_reduction_tree():
    shared = reduction_tree(False)
    routed = reduction_tree(True)
    assert shared == routed
    assert leaves(shared) == list(range(HIDDEN))
    encoded = json.dumps(shared, separators=(",", ":")).encode()
    return {
        "operation": "symbolic lane-order comparison, not NaN emulation",
        "first_repeats": 96,
        "second_repeats": 3,
        "last_repeats": 1,
        "leaves": HIDDEN,
        "tree_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def check_row_stores():
    cases = []
    for groups in GROUPS:
        for rows in range(1, 33):
            owners = [
                (vector, token)
                for vector in range(2 * groups)
                for token in range(vector, rows, 2 * groups)
            ]
            assert sorted(token for _, token in owners) == list(range(rows))
            assert all(vector == token for vector, token in owners)
            scales = [byte for _, row in owners
                      for byte in range(4 * row, 4 * (row + 1))]
            assert sorted(scales) == list(range(4 * rows))
            codes = [(row * HIDDEN, (row + 1) * HIDDEN)
                     for _, row in owners]
            assert sorted(codes) == [(r * HIDDEN, (r + 1) * HIDDEN)
                                     for r in range(rows)]
            if rows in WIDTHS:
                cases.append({"rows": rows, "groups": groups,
                              "active_aiv": len(owners)})
    return {"all_geometry_cases": 64, "device_gate_cases": cases}


def check_ub():
    boundaries = (0, 2 * HIDDEN, 6 * HIDDEN, 10 * HIDDEN,
                  14 * HIDDEN, 15 * HIDDEN, 15 * HIDDEN + 32)
    assert all(value % 32 == 0 for value in boundaries)
    assert boundaries[-1] == 92192
    assert all(left < right for left, right in zip(boundaries, boundaries[1:]))
    assert boundaries[2] + 32 <= boundaries[3]
    return {
        "quant_ub_bytes": boundaries[-1],
        "quant_interval_boundaries": boundaries,
        "routed_scale_ub_bytes": [boundaries[2], boundaries[2] + 32],
        "body_phase0_ub_bytes": 6592 + 76320 + 1056,
        "routed_body_peak_ub_bytes": 116480,
        "complete_ffn_peak_ub_bytes": 195584,
    }


def check_distinct_suffixes():
    maximum = struct.unpack("<f", struct.pack("<I", 0x3C2F0000))[0]
    routed = float32(maximum / 127.0)
    shared = float32(maximum * float32(1.0 / 127.0))
    assert float_bits(routed) == 0x38B060C2
    assert float_bits(shared) == 0x38B060C1
    assert float_bits(abs(-0.0)) == 0
    return {
        "scope": "ideal IEEE FP32 scalar witness; vector Div is not modeled",
        "amax_bf16_bits": "0x3c2f",
        "amax": maximum,
        "routed_scale_bits": f"0x{float_bits(routed):08x}",
        "shared_scale_bits": f"0x{float_bits(shared):08x}",
        "input_negative_zero_bits": "0x8000",
        "absolute_zero_bits": "0x00000000",
        "zero_row_vector_division": "device differential required",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {
        "status": "passed",
        "scope": "CPU symbolic and address contracts only",
        "reduction": check_reduction_tree(),
        "ownership": check_row_stores(),
        "resources": check_ub(),
        "suffix_witness": check_distinct_suffixes(),
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
