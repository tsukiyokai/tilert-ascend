"""Measure the quantised matmul at the layout production actually uses.

An earlier sweep fed `npu_quant_matmul` an ND int8 weight and read 526 GB/s,
half the bfloat16 rate for half the bytes, which looked like a two-fold read
amplification worth recovering. It was an artefact: `act_quant_type` defaults
to int8, so `process_weights_after_loading` casts the weight to FRACTAL_NZ,
and `npu_format_cast` silently falls back to ND unless
`torch_npu.npu.config.allow_internal_format` is set. This reports both
layouts side by side so the production rate is on the record, and times each
inside an ACL graph as well as eagerly, because a hand-written kernel must be
compared against the rate production pays rather than against a dispatch
floor. Run on an idle die.
"""

import argparse
import itertools
import json

import torch
import torch_npu

FRACTAL_NZ = 29
CEILING_GIGABYTES = 1190.77
SHAPES = {
    "o_proj": (2048, 6144),
    "q_b_proj": (1536, 6144),
    "q_a_proj": (6144, 1536),
}


def build_banks(inner, outer, banks, fractal):
    """Independent weights, optionally in the cube's fractal layout."""
    weights = []
    for _ in range(banks):
        weight = torch.randint(-8, 8, (inner, outer), dtype=torch.int8,
                               device="npu")
        if fractal:
            weight = torch_npu.npu_format_cast(weight, FRACTAL_NZ)
            if torch_npu.get_npu_format(weight) != FRACTAL_NZ:
                raise RuntimeError("npu_format_cast fell back to ND")
        weights.append((weight, torch.rand(outer, dtype=torch.float32,
                                           device="npu") * 0.01))
    return weights


def eager_microseconds(call, iterations):
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


def graph_microseconds(call, iterations):
    """Replay one captured call, so the host dispatch floor is excluded."""
    for _ in range(20):
        call()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        call()
    torch.npu.synchronize()
    for _ in range(10):
        graph.replay()
    torch.npu.synchronize()
    start, end = torch.npu.Event(True), torch.npu.Event(True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def case(shape, rows, banks, fractal, iterations):
    inner, outer = shape
    weights = build_banks(inner, outer, banks, fractal)
    activation = torch.randn(rows, inner, dtype=torch.bfloat16, device="npu")
    quantized, pertoken = torch_npu.npu_dynamic_quant(activation)
    rotation = itertools.cycle(weights)

    def call():
        weight, scale = next(rotation)
        return torch_npu.npu_quant_matmul(quantized, weight, scale,
                                          pertoken_scale=pertoken,
                                          output_dtype=torch.bfloat16)

    weight_bytes = inner * outer
    eager = eager_microseconds(call, iterations)
    frozen = weights[0]
    graph = graph_microseconds(
        lambda: torch_npu.npu_quant_matmul(quantized, frozen[0], frozen[1],
                                           pertoken_scale=pertoken,
                                           output_dtype=torch.bfloat16),
        iterations,
    )
    return {
        "shape": list(shape), "rows": rows, "fractal": fractal,
        "weight_bytes": weight_bytes,
        "eager_microseconds": eager,
        "graph_microseconds": graph,
        "eager_gigabytes_per_second": weight_bytes / eager / 1000.0,
        "graph_gigabytes_per_second": weight_bytes / graph / 1000.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--banks", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--report")
    arguments = parser.parse_args()

    torch_npu.npu.config.allow_internal_format = True
    cases = [case(shape, rows, arguments.banks, fractal,
                  arguments.iterations)
             for shape in SHAPES.values()
             for rows in arguments.rows
             for fractal in (False, True)]
    names = [name for name in SHAPES
             for _ in arguments.rows for _ in (False, True)]

    print(f"ceiling {CEILING_GIGABYTES} GB/s, {arguments.banks} banks")
    print(f"{'shape':12}{'rows':>5}{'layout':>8}{'eager µs':>10}"
          f"{'graph µs':>10}{'eager GB/s':>12}{'graph GB/s':>12}{'peak':>7}")
    for name, item in zip(names, cases):
        layout = "NZ" if item["fractal"] else "ND"
        share = item["graph_gigabytes_per_second"] / CEILING_GIGABYTES * 100
        print(f"{name:12}{item['rows']:5d}{layout:>8}"
              f"{item['eager_microseconds']:10.2f}"
              f"{item['graph_microseconds']:10.2f}"
              f"{item['eager_gigabytes_per_second']:12.1f}"
              f"{item['graph_gigabytes_per_second']:12.1f}{share:6.1f}%")

    if arguments.report:
        with open(arguments.report, "w") as handle:
            json.dump({"ceiling_gigabytes_per_second": CEILING_GIGABYTES,
                       "banks": arguments.banks,
                       "cases": dict(zip(
                           [f"{n}-rows{c['rows']}-"
                            f"{'nz' if c['fractal'] else 'nd'}"
                            for n, c in zip(names, cases)], cases))},
                      handle, indent=2)
            handle.write("\n")


main()
