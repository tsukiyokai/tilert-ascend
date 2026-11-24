"""Does the cube's fractal weight format lift a batch-one matmul's bandwidth?

A quantised matmul at four rows reaches under half the die's streaming rate.
If the weight arrives in ND the mover must reorder it on the way to L1, so
the same weight pre-cast to FRACTAL_NZ should run closer to the ceiling.
"""

import itertools
import sys

import torch
import torch_npu

FRACTAL_NZ = 29
CEILING_GIGABYTES = 1190.77
BANKS = 24


def timed(call, device, iterations=50):
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


def case(shape, rows, device, cast_to_fractal):
    inner, outer = shape
    weights = []
    for _ in range(BANKS):
        weight = torch.randint(-8, 8, (inner, outer), dtype=torch.int8,
                               device=device)
        if cast_to_fractal:
            weight = torch_npu.npu_format_cast(weight, FRACTAL_NZ)
        weights.append((weight, torch.rand(outer, dtype=torch.float32,
                                           device=device) * 0.01))
    activation = torch.randn(rows, inner, dtype=torch.bfloat16, device=device)
    quantized, pertoken = torch_npu.npu_dynamic_quant(activation)
    rotation = itertools.cycle(weights)

    def call():
        weight, scale = next(rotation)
        torch_npu.npu_quant_matmul(quantized, weight, scale,
                                   pertoken_scale=pertoken,
                                   output_dtype=torch.bfloat16)

    microseconds = timed(call, device)
    rate = inner * outer / microseconds / 1000.0
    return microseconds, rate


def main():
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    shapes = {"o_proj (2048,6144)": (2048, 6144),
              "q_a_proj (6144,1536)": (6144, 1536),
              "q_b_proj (1536,6144)": (1536, 6144)}
    print(f"{'shape':24s}{'rows':>5}{'ND µs':>9}{'ND GB/s':>9}"
          f"{'NZ µs':>9}{'NZ GB/s':>9}{'加速':>7}{'峰值占比':>9}")
    for label, shape in shapes.items():
        for rows in (1, 4):
            nd_us, nd_rate = case(shape, rows, device, False)
            try:
                nz_us, nz_rate = case(shape, rows, device, True)
            except Exception as error:
                print(f"{label:24s}{rows:5d}{nd_us:9.2f}{nd_rate:9.1f}"
                      f"   NZ 拒绝: {str(error).split(chr(10))[0][:60]}")
                continue
            print(f"{label:24s}{rows:5d}{nd_us:9.2f}{nd_rate:9.1f}"
                  f"{nz_us:9.2f}{nz_rate:9.1f}{nd_us / nz_us:7.2f}x"
                  f"{nz_rate / CEILING_GIGABYTES * 100:8.1f}%")


main()
