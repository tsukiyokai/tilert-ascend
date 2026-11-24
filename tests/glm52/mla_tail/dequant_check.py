"""Compare the batched MLA epilogue with ordinary native and vendor tails."""

import argparse
import ctypes
import json
from pathlib import Path
import statistics

from check import (
    capture, event_time_us, load_inputs, metrics, provenance, sha256_file,
    validate_inputs,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path(__file__).parent /
                        "build-composable/mla_tail_torch.so")
    parser.add_argument("--candidate-library", type=Path,
                        default=Path(__file__).parent /
                        "build-dequant/mla_tail_dequant.so")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 12, 32])
    parser.add_argument("--blocks", type=int, nargs="+", default=[8])
    parser.add_argument("--bias", default="fixture",
                        choices=["fixture", "none", "int32", "float32"])
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--timing-pairs", type=int, default=6)
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--queued-producer", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(row < 1 or row > 32 for row in args.widths):
        parser.error("widths must lie in [1,32]")
    if any(block < 0 or block > 64 for block in args.blocks):
        parser.error("blocks must lie in [0,64]")
    if min(args.replays, args.iterations, args.timing_pairs) <= 0:
        parser.error("replay and timing counts must be positive")
    return args


def load_launch(path):
    library = ctypes.CDLL(str(path.resolve()))
    launch = library.mla_tail_dequant_launch
    launch.restype = None
    launch.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint64]
        + [ctypes.c_void_p] * 5 + [ctypes.c_int]
        + [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 2
    )
    return library, launch


def run_case(torch, torch_npu, args, fixture, launch, rows, requested_blocks):
    def nd(value):
        return torch_npu.npu_format_cast(value.to(args.device).contiguous(), 2)

    latent_base = nd(fixture["latent"][:rows])
    latent = nd(latent_base.clone())
    value = nd(fixture["value_weight"])
    value_tiles = nd(value.reshape(4, 512, 4, 64).permute(0, 2, 1, 3)
                     .contiguous().reshape(16, 512, 64))
    weight = nd(fixture["output_weight"])
    output_tiles = nd(weight.reshape(1024, 24, 256).permute(1, 0, 2)
                      .contiguous())
    output_nz = torch_npu.npu_format_cast(weight, 29)
    scales = nd(fixture["output_scale"].float().reshape(6144))
    bias = fixture.get("bias")
    if args.bias == "none":
        bias = None
    elif args.bias == "int32":
        bias = (torch.arange(6144, dtype=torch.int32) % 127 - 63) * 1024
    elif args.bias == "float32":
        bias = torch.linspace(-0.25, 0.25, 6144, dtype=torch.float32)
    if bias is not None:
        dtype = torch.int32 if bias.dtype == torch.int32 else torch.float32
        bias = nd(bias.to(dtype))
    bias_kind = 0 if bias is None else (1 if bias.dtype == torch.int32 else 2)
    capacity = 16 if rows <= 16 else 32
    scratch, tiling, blocks, sync_base = torch.ops.tilert.mla_tail_prepare(
        latent, capacity, requested_blocks,
    )
    candidate_scratch = nd(torch.empty_like(scratch))
    output = nd(torch.empty(rows, 6144, dtype=torch.bfloat16))
    candidate_output = nd(torch.empty_like(output))

    def current_latent():
        return latent.clone() if args.queued_producer else latent

    def ordinary():
        return torch.ops.tilert.mla_tail_out(
            current_latent(), value_tiles, output_tiles, scales, bias, scratch,
            tiling, output, capacity, blocks, sync_base,
        )

    def candidate():
        source = current_latent()
        stream_pointer = torch.npu.current_stream(args.device).npu_stream
        launch(blocks, stream_pointer, sync_base, source.data_ptr(),
               value_tiles.data_ptr(), output_tiles.data_ptr(),
               scales.data_ptr(),
               bias.data_ptr() if bias is not None else None,
               bias_kind, candidate_scratch.data_ptr(), tiling.data_ptr(),
               candidate_output.data_ptr(), rows, capacity)
        return candidate_output

    def vendor():
        expanded = torch_npu.npu_transpose_batchmatmul(
            current_latent(), value, perm_x1=(1, 0, 2), perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        ).reshape(rows, 1024)
        codes, scale = torch_npu.npu_dynamic_quant(expanded)
        return torch_npu.npu_quant_matmul(
            codes, output_nz, scales, pertoken_scale=scale, bias=bias,
            output_dtype=torch.bfloat16,
        )

    stream = torch.npu.Stream(device=args.device)
    stream.wait_stream(torch.npu.current_stream(args.device))
    graphs, outputs = {}, {}
    for name, call in (("ordinary", ordinary), ("candidate", candidate),
                       ("vendor", vendor)):
        graphs[name], outputs[name] = capture(torch, call, stream)
    names = tuple(graphs)
    checks = []
    for replay in range(args.replays):
        with torch.npu.stream(stream):
            multiplier = (1.0, 0.0, -1.0, 0.5)[replay % 4]
            changed = (latent_base.float() * multiplier).bfloat16()
            if replay >= 4:
                changed = changed.roll(replay, dims=-1)
            latent.copy_(changed)
            start = replay % len(names)
            for name in names[start:] + names[:start]:
                graphs[name].replay()
        stream.synchronize()
        comparisons = {
            name: metrics(torch, outputs["candidate"], outputs[name], 0.0, 0.0)
            for name in ("ordinary", "vendor")
        }
        checks.append({"replay": replay, "comparisons": comparisons})
        if any(not result["pass"] or result.get("bitwise_equal") is not True
               for result in comparisons.values()):
            return {"rows": rows, "blocks": blocks, "pass": False,
                    "checks": checks, "bias_kind": bias_kind}

    timing = None
    if not args.no_timing:
        pairs = []
        for pair in range(args.timing_pairs):
            start = pair % len(names)
            order = names[start:] + names[:start]
            if pair % 2:
                order = tuple(reversed(order))
            pairs.append({name: event_time_us(torch, graphs[name],
                                              args.iterations, stream)
                          for name in order})
        timing = {"pairs_us_per_kernel": pairs, "iterations": args.iterations,
                  "median_us": {
                      name: statistics.median(pair[name] for pair in pairs)
                      for name in names}}
    return {"rows": rows, "capacity": capacity, "blocks": blocks,
            "pass": True, "checks": checks, "timing": timing,
            "bias_kind": bias_kind, "scratch_bytes_per_arm": scratch.nbytes}


def main():
    args = arguments()
    import torch
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    torch.ops.load_library(str(args.library.resolve()))
    candidate_library, launch = load_launch(args.candidate_library)
    fixture = load_inputs(torch, args)
    validate_inputs(torch, fixture, max(args.widths))
    manifest = args.candidate_library.parent / "build-manifest.json"
    report = {
        "scope": "MLA tail with eight-row output DMA batches",
        "queued_producer": args.queued_producer,
        "queued_producer_scope": "all three arms",
        "provenance": provenance(torch, torch_npu, args),
        "candidate_library_sha256": sha256_file(args.candidate_library),
        "harness_sha256": sha256_file(Path(__file__)),
        "candidate_build_manifest": json.loads(manifest.read_text())
            if manifest.is_file() else None,
        "cases": [],
    }
    try:
        for rows in args.widths:
            for blocks in args.blocks:
                result = run_case(
                    torch, torch_npu, args, fixture, launch, rows, blocks,
                )
                report["cases"].append(result)
                print(json.dumps(result), flush=True)
                if not result["pass"]:
                    raise RuntimeError("batched MLA differential failed")
        report["pass"] = True
    except Exception as error:
        report["pass"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    del candidate_library


if __name__ == "__main__":
    main()
