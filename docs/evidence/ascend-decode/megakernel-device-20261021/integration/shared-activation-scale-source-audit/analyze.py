"""Reconstruct captured scale counterfactuals with exact FP32 rounding."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import torch


CAPTURE_SHA = "977b888e50c84f55d3dd358be8b25073741b9ebc5ad6823067b6b1920424bf52"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fraction_from_bits(bits):
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    if exponent == 255 or bits >> 31:
        raise ValueError("Require a finite, nonnegative FP32 value")
    if exponent:
        mantissa |= 1 << 23
        power = exponent - 150
    else:
        power = -149
    if power >= 0:
        return Fraction(mantissa << power)
    return Fraction(mantissa, 1 << -power)


def rounded_integer(numerator, denominator):
    whole, remainder = divmod(numerator, denominator)
    return whole + (2 * remainder > denominator or
                    (2 * remainder == denominator and whole & 1))


def round32_bits(value):
    """Round a nonnegative rational directly to FP32, ties to even."""
    if value < 0:
        raise ValueError("Require a nonnegative rational")
    if value == 0:
        return 0
    n, d = value.numerator, value.denominator
    exponent = n.bit_length() - d.bit_length()
    if (n < d << exponent if exponent >= 0 else n << -exponent < d):
        exponent -= 1
    if exponent < -126:
        return rounded_integer(n << 149, d)
    shift = 23 - exponent
    mantissa = (rounded_integer(n << shift, d) if shift >= 0 else
                rounded_integer(n, d << -shift))
    if mantissa == 1 << 24:
        mantissa >>= 1
        exponent += 1
    if exponent > 127:
        return 0x7F800000
    return ((exponent + 127) << 23) | (mantissa - (1 << 23))


FACTOR_BITS = round32_bits(Fraction(1, 127))
FACTOR = fraction_from_bits(FACTOR_BITS)


def multiply_scale_bits(amax_bits):
    return round32_bits(fraction_from_bits(amax_bits) * FACTOR)


def first_not_less(target):
    low, high = 0, 0x7F800000
    while low < high:
        middle = (low + high) // 2
        if multiply_scale_bits(middle) < target:
            low = middle + 1
        else:
            high = middle
    return low


def preimages(target):
    first = first_not_less(target)
    end = first_not_less(target + 1)
    return list(range(first, end))


def bits(tensor):
    return tensor.contiguous().view(torch.int32).reshape(-1).tolist()


def raw_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def quantizer_witness(scale_bits, maximum_bits):
    scale = fraction_from_bits(scale_bits)
    reciprocal = fraction_from_bits(round32_bits(1 / scale))
    maximum = fraction_from_bits(maximum_bits)
    for integer in range(127):
        midpoint = round32_bits(Fraction(2 * integer + 1, 2) * scale)
        for activation_bits in range(midpoint - 2, midpoint + 3):
            activation = fraction_from_bits(activation_bits)
            if activation > maximum:
                continue
            divided = round32_bits(activation / scale)
            multiplied = round32_bits(activation * reciprocal)
            div_value = fraction_from_bits(divided)
            mul_value = fraction_from_bits(multiplied)
            div_code = rounded_integer(div_value.numerator, div_value.denominator)
            mul_code = rounded_integer(mul_value.numerator, mul_value.denominator)
            if div_code != mul_code:
                return {
                    "activation_bits": f"{activation_bits:08x}",
                    "scale_bits": f"{scale_bits:08x}",
                    "reciprocal_bits": f"{round32_bits(1 / scale):08x}",
                    "quotient_bits": f"{divided:08x}",
                    "reciprocal_product_bits": f"{multiplied:08x}",
                    "rint_division_code": div_code,
                    "rint_reciprocal_multiply_code": mul_code,
                    "scope": "Constructed FP32 witness, not a captured activation; hardware Div may differ",
                }
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if digest(args.capture) != CAPTURE_SHA:
        raise ValueError("The retained capture changed")
    torch.set_num_threads(1)
    captured = torch.load(args.capture, weights_only=True, map_location="cpu")
    reference, native = captured["reference"], captured["native"]
    reference_bits = bits(reference["middle_scale"])
    native_bits = bits(native["middle_scale"])
    rows = []
    for row, (actual, expected) in enumerate(zip(native_bits, reference_bits)):
        inferred = preimages(actual)
        if len(inferred) != 1:
            raise ValueError(f"Native row {row} has {len(inferred)} preimages")
        amax = inferred[0]
        division = round32_bits(fraction_from_bits(amax) / 127)
        rows.append({
            "row": row, "native_scale_bits": f"{actual:08x}",
            "official_scale_bits": f"{expected:08x}",
            "inferred_amax_bits": f"{amax:08x}",
            "native_mul_preimages": len(inferred),
            "previous_amax_mul_scale_bits": f"{multiply_scale_bits(amax - 1):08x}",
            "next_amax_mul_scale_bits": f"{multiply_scale_bits(amax + 1):08x}",
            "same_amax_div127_bits": f"{division:08x}",
            "same_amax_div127_matches_official": division == expected,
            "official_mul_preimages": [f"{v:08x}" for v in preimages(expected)],
        })
    accumulators = reference["gate_int32"]
    converted = accumulators.float()
    dequantized = ((converted * reference["gate_scale"]) *
                   reference["input_scale"][:, None])
    gate, up = dequantized[:, :128], dequantized[:, 128:]
    algorithms = {
        "cpu_fp32_div_exp": gate / (1 + torch.exp(-gate)) * up,
        "cpu_fp32_sigmoid_mul": gate * torch.sigmoid(gate) * up,
        "cpu_fp32_silu": torch.nn.functional.silu(gate) * up,
        "cpu_fp64_activation_then_fp32": (
            gate.double() / (1 + torch.exp(-gate.double())) * up.double()).float(),
    }
    inferred_bits = [int(row["inferred_amax_bits"], 16) for row in rows]
    cpu_algorithms = {}
    for name, activation in algorithms.items():
        maximum, indices = activation.abs().max(-1)
        maximum_bits = bits(maximum)
        mul_scales = [multiply_scale_bits(v) for v in maximum_bits]
        div_scales = [round32_bits(fraction_from_bits(v) / 127)
                      for v in maximum_bits]
        cpu_algorithms[name] = {
            "amax_bits": [f"{v:08x}" for v in maximum_bits],
            "amax_channels": indices.tolist(),
            "inferred_amax_different_rows": [i for i, (a, b) in enumerate(
                zip(maximum_bits, inferred_bits)) if a != b],
            "mul_scale_different_from_native": [i for i, (a, b) in enumerate(
                zip(mul_scales, native_bits)) if a != b],
            "div_scale_different_from_official": [i for i, (a, b) in enumerate(
                zip(div_scales, reference_bits)) if a != b],
        }
    report = {
        "capture_sha256": CAPTURE_SHA,
        "script_sha256": digest(Path(__file__)),
        "torch": torch.__version__, "device_execution": False,
        "exact_arithmetic": "FP32 RNE implemented with integer/Fraction arithmetic; no FP64 intermediate rounding",
        "factor_bits": f"{FACTOR_BITS:08x}",
        "parent_reproduction": captured["parent_reproduced"],
        "observed_equal_boundaries": {name: raw_equal(
            native[name], reference[name].reshape(native[name].shape))
            for name in ("normalized", "codes", "input_scale", "gate_int32", "middle")},
        "gu_max_abs_int32": int(accumulators.abs().max()),
        "gu_int32_to_fp32_exact": torch.equal(converted.to(torch.int32), accumulators),
        "scale_different_rows": [i for i, (a, b) in enumerate(
            zip(native_bits, reference_bits)) if a != b],
        "all_32_same_amax_div127_matches_official": all(
            row["same_amax_div127_matches_official"] for row in rows),
        "inference_boundary": (
            "amax is uniquely inferred under native RNE multiply by the stored FP32 reciprocal, "
            "not observed. A different activation maximum with the same multiply also fits official "
            "scales. Dispatch and/or retained activation is required to choose the cause."),
        "cpu_activation_boundary": (
            "CPU Exp and division are discriminators, not an Ascend vector implementation oracle."),
        "rows": rows, "cpu_algorithms": cpu_algorithms,
        "coupled_quantizer_witnesses": {
            str(row): quantizer_witness(reference_bits[row], inferred_bits[row])
            for row in (0, 1, 20)},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "gu_max_abs_int32", "gu_int32_to_fp32_exact", "scale_different_rows",
        "all_32_same_amax_div127_matches_official")}))


if __name__ == "__main__":
    main()
