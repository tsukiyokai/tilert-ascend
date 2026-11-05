"""Can a hand-written kernel stream weights faster than the vendor matmul?

At four decode rows every projection is bound by the global-to-L1 mover, and
the vendor's quantised matmul reaches 23 to 48 per cent of the die's measured
1190.77 GB/s with no dependence on how many calls precede it, so the loss is
inside one kernel. This kernel streams the same weight with an explicit
two-stage pipeline over the reduction axis and reports the rate it achieves,
which decides whether hand-written kernels are the route to the round's
largest block. Run on an idle die.
"""

import argparse
import itertools
import json

import tilelang as tl
import tilelang.language as T
import torch
import torch_npu

CEILING_GIGABYTES = 1190.77
FRACTAL = 16


@tl.jit(out_idx=[-1], pass_configs={
    tl.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
})
def streaming_gemv(rows, inner, outer, block_outer, block_inner, stages,
                   dtype="bfloat16", accumulate="float32"):
    """One cube core owns `block_outer` output columns and streams their weight."""
    outer_blocks = T.ceildiv(outer, block_outer)
    inner_blocks = T.ceildiv(inner, block_inner)

    @T.prim_func
    def main(
        activation: T.Tensor((FRACTAL, inner), dtype),  # type: ignore
        weight: T.Tensor((outer, inner), dtype),  # type: ignore
        result: T.Tensor((FRACTAL, outer), dtype),  # type: ignore
    ):
        with T.Kernel(outer_blocks, is_npu=True) as (core, _):
            weight_l1 = T.alloc_L1((block_outer, block_inner), dtype)
            activation_l1 = T.alloc_L1((FRACTAL, block_inner), dtype)
            accumulator = T.alloc_L0C((FRACTAL, block_outer), accumulate)

            for step in T.Pipelined(inner_blocks, num_stages=stages):
                T.copy(activation[0, step * block_inner], activation_l1)
                T.copy(weight[core * block_outer, step * block_inner],
                       weight_l1)
                T.gemm_v0(activation_l1, weight_l1, accumulator,
                          transpose_B=True, init=(step == 0))

            T.copy(accumulator, result[0, core * block_outer])

    return main


def timed(call, iterations):
    for _ in range(10):
        call()
    torch.npu.synchronize()
    start, end = torch.npu.Event(True), torch.npu.Event(True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def build_banks(outer, inner, banks):
    """Independent weights so a rotation reads from memory, not from L2."""
    return [torch.randn(outer, inner, dtype=torch.bfloat16,
                        device="npu") for _ in range(banks)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--inner", type=int, default=2048)
    parser.add_argument("--outer", type=int, default=6144)
    parser.add_argument("--block-outer", type=int, nargs="+",
                        default=[128, 256, 512])
    parser.add_argument("--block-inner", type=int, nargs="+",
                        default=[256, 512])
    parser.add_argument("--stages", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--banks", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--report")
    arguments = parser.parse_args()

    weights = build_banks(arguments.outer, arguments.inner, arguments.banks)
    activation = torch.randn(FRACTAL, arguments.inner,
                             dtype=torch.bfloat16, device="npu")
    weight_bytes = arguments.outer * arguments.inner * 2

    reference_rotation = itertools.cycle(weights)
    vendor = timed(lambda: torch.matmul(activation,
                                        next(reference_rotation).t()),
                   arguments.iterations)
    print(f"[{arguments.rows}, {arguments.inner}] × "
          f"[{arguments.inner}, {arguments.outer}] bfloat16, "
          f"{weight_bytes / 2**20:.1f} MiB per call, {arguments.banks} banks")
    print(f"vendor torch.matmul {vendor:8.2f} µs  "
          f"{weight_bytes / vendor / 1000.0:7.1f} GB/s  "
          f"{weight_bytes / vendor / 1000.0 / CEILING_GIGABYTES * 100:.1f}% "
          f"of ceiling")
    print(f"{'block_outer':>12}{'block_inner':>12}{'stages':>7}"
          f"{'µs':>9}{'GB/s':>9}{'峰值占比':>9}{'对厂商':>8}")

    cases = []
    first = None
    for block_outer, block_inner, stages in itertools.product(
            arguments.block_outer, arguments.block_inner, arguments.stages):
        kernel = streaming_gemv(arguments.rows, arguments.inner,
                                arguments.outer, block_outer, block_inner,
                                stages)
        rotation = itertools.cycle(weights)
        result = kernel(activation, next(rotation))
        if first is None:
            first = torch.matmul(activation, weights[0].t())
            difference = (result.float() - first.float()).abs().max().item()
            print(f"  numerics: max abs difference {difference:.4g} against "
                  f"torch.matmul")
        microseconds = timed(lambda: kernel(activation, next(rotation)),
                             arguments.iterations)
        rate = weight_bytes / microseconds / 1000.0
        cases.append({"block_outer": block_outer, "block_inner": block_inner,
                      "stages": stages, "microseconds": microseconds,
                      "gigabytes_per_second": rate})
        print(f"{block_outer:12d}{block_inner:12d}{stages:7d}"
              f"{microseconds:9.2f}{rate:9.1f}"
              f"{rate / CEILING_GIGABYTES * 100:8.1f}%"
              f"{vendor / microseconds:7.2f}x")

    if arguments.report:
        with open(arguments.report, "w") as handle:
            json.dump({"rows": arguments.rows, "inner": arguments.inner,
                       "outer": arguments.outer, "banks": arguments.banks,
                       "weight_bytes": weight_bytes,
                       "vendor_microseconds": vendor,
                       "ceiling_gigabytes_per_second": CEILING_GIGABYTES,
                       "cases": cases}, handle, indent=2)
            handle.write("\n")


main()
