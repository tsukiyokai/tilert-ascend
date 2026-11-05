"""Measure the bandwidth a chain of weight-streaming matmuls achieves.

Decode at batch one reads every weight once per round and multiplies it by a
handful of activation rows, so each projection is bound by the global-to-L1
mover rather than by the cube. The round spends 22.80 ms in cube groups
against a 6.00 ms streaming floor, and the question this answers is whether
separate kernels can keep that mover busy or whether only one kernel spanning
the chain can. Run it on an idle die; a co-tenant distorts the rate.
"""

import argparse
import itertools
import json

import torch
import torch_npu

LAYER_SHAPES = {
    "o_proj": (2048, 6144),
    "q_a_proj": (6144, 1536),
    "q_b_proj": (1536, 6144),
    "kv_a_proj": (6144, 576),
    "expert_down": (2048, 6144),
}


def build_banks(shape, count, device):
    """Allocate independent weight banks so a rotation defeats L2 reuse."""
    rows, columns = shape
    return [(torch.randint(-8, 8, (rows, columns), dtype=torch.int8,
                           device=device),
             torch.rand(columns, dtype=torch.float32, device=device) * 0.01)
            for _ in range(count)]


def timed(call, device, iterations):
    for _ in range(20):
        call()
    torch.npu.synchronize(device)
    start, end = torch.npu.Event(True), torch.npu.Event(True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    torch.npu.synchronize(device)
    return start.elapsed_time(end) * 1000.0 / iterations


def chain_case(shape, rows, chain, banks, device, iterations):
    """Time `chain` matmuls that each stream a fresh weight bank."""
    weights = build_banks(shape, banks, device)
    activation = torch.randn(rows, shape[0], dtype=torch.bfloat16,
                             device=device)
    quantized, pertoken = torch_npu.npu_dynamic_quant(activation)
    rotation = itertools.cycle(weights)

    def call():
        for _ in range(chain):
            weight, scale = next(rotation)
            torch_npu.npu_quant_matmul(quantized, weight, scale,
                                       pertoken_scale=pertoken,
                                       output_dtype=torch.bfloat16)

    microseconds = timed(call, device, iterations)
    weight_bytes = shape[0] * shape[1] * chain
    return {
        "shape": list(shape), "rows": rows, "chain": chain, "banks": banks,
        "microseconds": microseconds,
        "per_call_microseconds": microseconds / chain,
        "weight_bytes": weight_bytes,
        "gigabytes_per_second": weight_bytes / microseconds / 1000.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--shape", default="o_proj", choices=LAYER_SHAPES)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--chain", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--banks", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--report")
    arguments = parser.parse_args()

    device = torch.device(arguments.device)
    torch.npu.set_device(device)
    shape = LAYER_SHAPES[arguments.shape]
    bank_bytes = shape[0] * shape[1] * arguments.banks
    cases = [chain_case(shape, arguments.rows, chain, arguments.banks, device,
                        arguments.iterations) for chain in arguments.chain]
    report = {
        "shape_name": arguments.shape, "shape": list(shape),
        "rows": arguments.rows, "bank_working_set_bytes": bank_bytes,
        "l2_bytes": 201326592, "cases": cases,
    }
    print(f"{arguments.shape} {shape} rows={arguments.rows} "
          f"banks={arguments.banks} working set {bank_bytes / 2**20:.1f} MiB")
    print(f"{'chain':>6}{'µs total':>11}{'µs/call':>10}{'GB/s':>9}")
    for case in cases:
        print(f"{case['chain']:6d}{case['microseconds']:11.2f}"
              f"{case['per_call_microseconds']:10.2f}"
              f"{case['gigabytes_per_second']:9.1f}")
    if arguments.report:
        with open(arguments.report, "w") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")


main()
