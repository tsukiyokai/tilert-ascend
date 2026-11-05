"""Quantify the quantised matmul's per-call fixed cost, and whether one call
beats two for the same bytes.

A batch-one projection costs 22 to 33 microseconds almost regardless of its
weight size, so most of the cube block is prologue rather than streaming. Three
measurements follow from that claim. A sweep over the reduction axis separates
the fixed intercept from the streaming slope. A pair of equal-byte cases, one
wide call against two narrow ones, says what collapsing a prologue is worth --
the fused kernel's whole case, without writing one. And a sweep over rows says
how far the cube is idle at decode widths.

Eager back-to-back is the instrument: a single-operator graph pays a replay
floor, while eager dispatch is already hidden behind the previous kernel. Run
on an idle die.
"""

import argparse
import itertools
import json

import torch
import torch_npu

FRACTAL_NZ = 29
CEILING_GIGABYTES = 1190.77


def weight_banks(inner, outer, banks):
    weights = []
    for _ in range(banks):
        weight = torch_npu.npu_format_cast(
            torch.randint(-8, 8, (inner, outer), dtype=torch.int8,
                          device="npu"), FRACTAL_NZ)
        if torch_npu.get_npu_format(weight) != FRACTAL_NZ:
            raise RuntimeError("npu_format_cast fell back to ND")
        weights.append((weight, torch.rand(outer, dtype=torch.float32,
                                           device="npu") * 0.01))
    return weights


def timed(call, iterations):
    for _ in range(20):
        call()
    torch.npu.synchronize()
    start, end = torch.npu.Event(True), torch.npu.Event(True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def quantised(rows, inner):
    activation = torch.randn(rows, inner, dtype=torch.bfloat16, device="npu")
    return torch_npu.npu_dynamic_quant(activation)


def matmul_case(rows, inner, outer, banks, iterations):
    weights = weight_banks(inner, outer, banks)
    quantized, pertoken = quantised(rows, inner)
    rotation = itertools.cycle(weights)

    def call():
        weight, scale = next(rotation)
        torch_npu.npu_quant_matmul(quantized, weight, scale,
                                   pertoken_scale=pertoken,
                                   output_dtype=torch.bfloat16)

    microseconds = timed(call, iterations)
    return microseconds, inner * outer


def least_squares(points):
    """Fit microseconds = intercept + slope * bytes over the sweep."""
    count = len(points)
    mean_x = sum(x for x, _ in points) / count
    mean_y = sum(y for _, y in points) / count
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in points)
    variance = sum((x - mean_x) ** 2 for x, _ in points)
    slope = covariance / variance
    return mean_y - slope * mean_x, slope


def inner_sweep(arguments):
    outer, rows = 6144, arguments.rows
    print(f"\n[1] fixed cost vs bytes — N={outer}, rows={rows}")
    print(f"{'K':>6}{'MiB':>8}{'µs':>9}{'GB/s':>9}")
    points, cases = [], []
    for inner in (256, 512, 1024, 2048, 4096):
        microseconds, weight_bytes = matmul_case(
            rows, inner, outer, arguments.banks, arguments.iterations)
        points.append((weight_bytes, microseconds))
        cases.append({"inner": inner, "outer": outer,
                      "weight_bytes": weight_bytes,
                      "microseconds": microseconds})
        print(f"{inner:6d}{weight_bytes / 2**20:8.1f}{microseconds:9.2f}"
              f"{weight_bytes / microseconds / 1000.0:9.1f}")
    intercept, slope = least_squares(points)
    marginal = 1.0 / slope / 1000.0
    print(f"  fit: {intercept:.2f} µs fixed + {marginal:.1f} GB/s marginal "
          f"({marginal / CEILING_GIGABYTES * 100:.0f}% of ceiling)")
    return {"cases": cases, "fixed_microseconds": intercept,
            "marginal_gigabytes_per_second": marginal}


def fusion_sweep(arguments):
    """One wide call against two narrow calls carrying the same bytes."""
    inner, rows = 2048, arguments.rows
    print(f"\n[2] one call vs two for equal bytes — K={inner}, rows={rows}")
    print(f"{'layout':>22}{'µs':>9}{'MiB':>8}")
    results = {}
    wide, wide_bytes = matmul_case(rows, inner, 12288, arguments.banks,
                                   arguments.iterations)
    print(f"{'one call N=12288':>22}{wide:9.2f}{wide_bytes / 2**20:8.1f}")
    narrow, narrow_bytes = matmul_case(rows, inner, 6144, arguments.banks,
                                       arguments.iterations)
    print(f"{'one call N=6144':>22}{narrow:9.2f}{narrow_bytes / 2**20:8.1f}")
    print(f"{'two calls N=6144':>22}{2 * narrow:9.2f}"
          f"{2 * narrow_bytes / 2**20:8.1f}")
    saving = 2 * narrow - wide
    print(f"  fusing two into one saves {saving:.2f} µs "
          f"({saving / (2 * narrow) * 100:.1f}%)")
    results.update(wide_microseconds=wide, narrow_microseconds=narrow,
                   saving_microseconds=saving)
    return results


def rows_sweep(arguments):
    inner, outer = 2048, 6144
    print(f"\n[3] cost vs rows — K={inner}, N={outer}")
    print(f"{'rows':>6}{'µs':>9}{'µs/row':>9}")
    cases = []
    for rows in (1, 4, 16, 64):
        microseconds, _ = matmul_case(rows, inner, outer, arguments.banks,
                                      arguments.iterations)
        cases.append({"rows": rows, "microseconds": microseconds})
        print(f"{rows:6d}{microseconds:9.2f}{microseconds / rows:9.2f}")
    return {"cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--banks", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--report")
    arguments = parser.parse_args()

    torch_npu.npu.config.allow_internal_format = True
    report = {
        "rows": arguments.rows, "banks": arguments.banks,
        "ceiling_gigabytes_per_second": CEILING_GIGABYTES,
        "inner_sweep": inner_sweep(arguments),
        "fusion": fusion_sweep(arguments),
        "rows_sweep": rows_sweep(arguments),
    }
    if arguments.report:
        with open(arguments.report, "w") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")


main()
