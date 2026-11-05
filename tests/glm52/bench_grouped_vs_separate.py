"""Compare one grouped matmul against the same work as separate matmuls.

A chain of separate quantised matmuls at four rows reaches 23 to 48 per cent
of the die's streaming rate and chain length does not change it, so the
inefficiency lives inside one call. The grouped operator presents the same
weight bytes to the mover as a single call with more independent tiles, and
whether it streams faster decides whether the dense projections should be
restructured or fused. Run on an idle die.
"""

import argparse
import itertools
import json

import torch
import torch_npu

CEILING_GIGABYTES = 1190.77
L2_BYTES = 201326592


def timed(call, device, iterations):
    for _ in range(10):
        call()
    torch.npu.synchronize(device)
    start, end = torch.npu.Event(True), torch.npu.Event(True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    torch.npu.synchronize(device)
    return start.elapsed_time(end) * 1000.0 / iterations


def build_banks(groups, inner, outer, banks, device):
    """One bank holds every group's weight, so a rotation defeats L2."""
    return [[(torch.randint(-8, 8, (inner, outer), dtype=torch.int8,
                            device=device),
              torch.rand(outer, dtype=torch.float32, device=device) * 0.01)
             for _ in range(groups)] for _ in range(banks)]


def separate_case(banks, quantized, pertoken, device, iterations):
    rotation = itertools.cycle(banks)

    def call():
        for weight, scale in next(rotation):
            torch_npu.npu_quant_matmul(quantized, weight, scale,
                                       pertoken_scale=pertoken,
                                       output_dtype=torch.bfloat16)

    return timed(call, device, iterations)


def grouped_case(banks, rows, device, iterations):
    """Stack each bank's weights and present the rows as equal groups."""
    stacked = []
    for bank in banks:
        weights = torch.stack([weight for weight, _ in bank])
        scales = torch.stack([scale for _, scale in bank])
        stacked.append((weights, scales))
    groups = len(banks[0])
    inner = banks[0][0][0].shape[0]
    activation = torch.randn(rows * groups, inner, dtype=torch.bfloat16,
                             device=device)
    quantized, pertoken = torch_npu.npu_dynamic_quant(activation)
    group_list = torch.tensor([rows * (index + 1) for index in range(groups)],
                              dtype=torch.int64, device=device)
    rotation = itertools.cycle(stacked)

    def call():
        weights, scales = next(rotation)
        torch_npu.npu_grouped_matmul(
            [quantized], [weights], scale=[scales],
            per_token_scale=[pertoken], group_list=group_list,
            split_item=3, group_type=0, group_list_type=0,
            output_dtype=torch.bfloat16,
        )

    return timed(call, device, iterations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--inner", type=int, default=2048)
    parser.add_argument("--outer", type=int, default=6144)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--groups", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--banks", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--report")
    arguments = parser.parse_args()

    device = torch.device(arguments.device)
    torch.npu.set_device(device)
    activation = torch.randn(arguments.rows, arguments.inner,
                             dtype=torch.bfloat16, device=device)
    quantized, pertoken = torch_npu.npu_dynamic_quant(activation)

    cases = []
    for groups in arguments.groups:
        banks = build_banks(groups, arguments.inner, arguments.outer,
                            arguments.banks, device)
        weight_bytes = arguments.inner * arguments.outer * groups
        separate = separate_case(banks, quantized, pertoken, device,
                                 arguments.iterations)
        grouped = grouped_case(banks, arguments.rows, device,
                               arguments.iterations)
        cases.append({
            "groups": groups, "weight_bytes": weight_bytes,
            "working_set_bytes": weight_bytes * arguments.banks,
            "separate_microseconds": separate,
            "grouped_microseconds": grouped,
            "separate_gigabytes_per_second":
                weight_bytes / separate / 1000.0,
            "grouped_gigabytes_per_second": weight_bytes / grouped / 1000.0,
        })
        del banks

    print(f"[{arguments.inner}, {arguments.outer}] int8  rows={arguments.rows} "
          f"banks={arguments.banks}  L2={L2_BYTES / 2**20:.0f} MiB")
    print(f"{'groups':>7}{'工作集MiB':>11}{'分离µs':>9}{'分组µs':>9}"
          f"{'分离GB/s':>10}{'分组GB/s':>10}{'加速':>7}{'峰值占比':>9}")
    for case in cases:
        print(f"{case['groups']:7d}"
              f"{case['working_set_bytes'] / 2**20:11.0f}"
              f"{case['separate_microseconds']:9.1f}"
              f"{case['grouped_microseconds']:9.1f}"
              f"{case['separate_gigabytes_per_second']:10.1f}"
              f"{case['grouped_gigabytes_per_second']:10.1f}"
              f"{case['separate_microseconds'] / case['grouped_microseconds']:6.2f}x"
              f"{case['grouped_gigabytes_per_second'] / CEILING_GIGABYTES * 100:8.1f}%")
    if arguments.report:
        with open(arguments.report, "w") as handle:
            json.dump({"inner": arguments.inner, "outer": arguments.outer,
                       "rows": arguments.rows, "banks": arguments.banks,
                       "ceiling_gigabytes_per_second": CEILING_GIGABYTES,
                       "cases": cases}, handle, indent=2)
            handle.write("\n")


main()
