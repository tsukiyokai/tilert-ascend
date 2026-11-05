"""Measure MLA tails across distinct weight banks in one captured graph.

This is a synthetic projection sequence, not a decoder layer or TPOT test.
The banks contain equal values at distinct addresses. A large bank count
tests a weight working set beyond L2 and amortizes host graph-replay gaps.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from check import (
    capture, event_time_us, load_inputs, metrics, provenance, sha256_file,
    validate_inputs,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--library", type=Path, default=Path(__file__).parent /
                        "build-composable/mla_tail_torch.so")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--banks", type=int, default=64)
    parser.add_argument("--reuse-bank", action="store_true",
                        help="repeat one weight address with the same graph call count")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--timing-pairs", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.rows <= 32 or not 1 <= args.banks <= 128:
        parser.error("rows must be in [1,32] and banks in [1,128]")
    if min(args.blocks, args.iterations, args.timing_pairs) <= 0:
        parser.error("blocks and timing counts must be positive")
    return args


def main():
    args = arguments()
    import torch
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    torch.ops.load_library(str(args.library.resolve()))
    fixture = load_inputs(torch, args)
    validate_inputs(torch, fixture, args.rows)

    def nd(tensor):
        return torch_npu.npu_format_cast(tensor.to(args.device).contiguous(), 2)

    latent_base = nd(fixture["latent"][:args.rows])
    latent = nd(latent_base.clone())
    value = nd(fixture["value_weight"])
    value_tiles = nd(value.reshape(4, 512, 4, 64).permute(0, 2, 1, 3)
                     .contiguous().reshape(16, 512, 64))
    weight = nd(fixture["output_weight"])
    output_tiles = nd(weight.reshape(1024, 24, 256).permute(1, 0, 2).contiguous())
    scales = nd(fixture["output_scale"].float().reshape(6144))
    bias = fixture.get("bias")
    if bias is not None:
        bias = nd(bias.to(torch.int32 if bias.dtype == torch.int32 else torch.float32))
    distinct_banks = 1 if args.reuse_bank else args.banks
    banks = [(nd(value.clone()), nd(value_tiles.clone()),
              torch_npu.npu_format_cast(weight.clone(), 29), nd(output_tiles.clone()))
             for _ in range(distinct_banks)]
    for column in range(4):
        if len({bank[column].data_ptr() for bank in banks}) != distinct_banks:
            raise RuntimeError("weight banks must own distinct addresses")
    for bank in banks:
        if [torch_npu.get_npu_format(tensor) for tensor in bank] != [2, 2, 29, 2]:
            raise RuntimeError("cloned weight formats differ from the operands")
    if args.reuse_bank:
        banks *= args.banks
    output = nd(torch.empty(args.rows, 6144, dtype=torch.bfloat16))
    capacity = 16 if args.rows <= 16 else 32
    scratch, tiling, blocks, sync_base = torch.ops.tilert.mla_tail_prepare(
        latent, capacity, args.blocks,
    )

    def native():
        for _, packed_value, _, packed_output in banks:
            torch.ops.tilert.mla_tail_out(
                latent, packed_value, packed_output, scales, bias, scratch,
                tiling, output, capacity, blocks, sync_base,
            )
        return output

    def vendor():
        result = None
        for ordinary_value, _, nz_output, _ in banks:
            expanded = torch_npu.npu_transpose_batchmatmul(
                latent, ordinary_value, perm_x1=(1, 0, 2), perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
            ).reshape(args.rows, 1024)
            codes, scale = torch_npu.npu_dynamic_quant(expanded)
            result = torch_npu.npu_quant_matmul(
                codes, nz_output, scales, pertoken_scale=scale, bias=bias,
                output_dtype=torch.bfloat16,
            )
        return result

    stream = torch.npu.Stream(device=args.device)
    stream.wait_stream(torch.npu.current_stream(args.device))
    native_graph, native_output = capture(torch, native, stream)
    vendor_graph, vendor_output = capture(torch, vendor, stream)
    checks = []
    for replay, multiplier in enumerate((1.0, 0.0, -1.0, 0.5)):
        with torch.npu.stream(stream):
            latent.copy_((latent_base.float() * multiplier).bfloat16())
            calls = (native_graph.replay, vendor_graph.replay)
            for call in calls[::1 if replay % 2 == 0 else -1]:
                call()
        stream.synchronize()
        check = metrics(torch, native_output, vendor_output, 0.0, 0.0)
        checks.append({"replay": replay, **check})
        if not check["pass"]:
            break
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "scope": "independent projection sequence with configurable weight reuse",
        "rows": args.rows, "blocks": blocks, "banks": args.banks,
        "distinct_weight_banks": distinct_banks,
        "weight_bytes_per_arm": distinct_banks * (value.nbytes + weight.nbytes),
        "scratch_bytes": scratch.nbytes,
        "input_clone_in_capture": False,
        "provenance": provenance(torch, torch_npu, args),
        "harness_sha256": sha256_file(Path(__file__)),
        "last_output_checks": checks,
        "pass": len(checks) == 4 and all(check["pass"] for check in checks),
    }
    if report["pass"]:
        pairs = []
        for pair in range(args.timing_pairs):
            order = (("native", native_graph), ("vendor", vendor_graph))
            pairs.append({name: event_time_us(torch, graph, args.iterations, stream)
                          for name, graph in order[::1 if pair % 2 == 0 else -1]})
        report["graph_us"] = pairs
        report["iterations"] = args.iterations
        report["median_us_per_projection"] = {
            name: statistics.median(pair[name] for pair in pairs) / args.banks
            for name in ("native", "vendor")
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
