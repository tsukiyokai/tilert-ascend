"""Measure local stage-counter deltas without changing MLA arithmetic."""

import argparse
import ctypes
import json
from pathlib import Path
import statistics

from check import (
    capture, event_time_us, load_inputs, metrics, provenance, sha256_file,
    validate_inputs,
)


STAGES = ("stage_heads", "value_matmul", "dynamic_quant", "output_matmul",
          "output_dequant")
RECORD_WORDS = 64
TIMESTAMP_OFFSET = 8


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path(__file__).parent /
                        "build-composable/mla_tail_torch.so")
    parser.add_argument("--profile-library", type=Path,
                        default=Path(__file__).parent /
                        "build-profile/mla_tail_profile.so")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 12, 32])
    parser.add_argument("--blocks", type=int, nargs="+", default=[8])
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--timing-pairs", type=int, default=5)
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(row < 1 or row > 32 for row in args.widths):
        parser.error("widths must lie in [1,32]")
    if any(block < 0 or block > 64 for block in args.blocks):
        parser.error("blocks must lie in [0,64]")
    if min(args.samples, args.iterations, args.timing_pairs) <= 0:
        parser.error("sample and timing counts must be positive")
    return args


def decode_trace(records, rows, capacity, blocks):
    if len(records) != 3 * blocks:
        raise ValueError("trace must contain one record per AIC and AIV")
    decoded = []
    for slot, record in enumerate(records):
        kind, core = (0, slot) if slot < blocks else (1, slot - blocks)
        if len(record) != RECORD_WORDS:
            raise ValueError("trace record has the wrong size")
        if record[:6] != [1, kind, core, rows, capacity, blocks]:
            raise ValueError(f"trace record {slot} has invalid metadata")
        ticks = record[TIMESTAMP_OFFSET:TIMESTAMP_OFFSET + 3 * len(STAGES)]
        if any(a > b for a, b in zip(ticks, ticks[1:])):
            raise ValueError(f"trace record {slot} has nonmonotonic ticks")
        window = ticks[-1] - ticks[0]
        if window <= 0:
            raise ValueError(f"trace record {slot} has no elapsed local ticks")
        stages = []
        for stage, name in enumerate(STAGES):
            begin, completed, released = ticks[3 * stage:3 * stage + 3]
            work = completed - begin
            wait = released - completed
            active = (stage in (1, 3)) if kind == 0 else (stage in (0, 2, 4))
            stages.append({
                "stage": name, "active": active, "work_ticks": work,
                "barrier_ticks": wait, "work_fraction_of_local_window":
                    work / window,
            })
        decoded.append({"kind": "AIC" if kind == 0 else "AIV",
                        "core_index": core, "local_window_ticks": window,
                        "stages": stages, "raw_ticks": ticks})
    return decoded


def summarize(samples):
    summaries = []
    for slot, first in enumerate(samples[0]):
        stages = []
        for stage, entry in enumerate(first["stages"]):
            values = [sample[slot]["stages"][stage] for sample in samples]
            stages.append({
                "stage": entry["stage"], "active": entry["active"],
                "median_work_ticks": statistics.median(
                    value["work_ticks"] for value in values),
                "median_barrier_ticks": statistics.median(
                    value["barrier_ticks"] for value in values),
                "median_work_fraction_of_local_window": statistics.median(
                    value["work_fraction_of_local_window"]
                    for value in values),
            })
        summaries.append({"kind": first["kind"],
                          "core_index": first["core_index"], "stages": stages})
    return summaries


def load_launch(path):
    library = ctypes.CDLL(str(path.resolve()))
    launch = library.mla_tail_profile_launch
    launch.restype = None
    launch.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint64]
        + [ctypes.c_void_p] * 5 + [ctypes.c_int]
        + [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 2
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
    output_weight = nd(fixture["output_weight"])
    output_tiles = nd(output_weight.reshape(1024, 24, 256)
                      .permute(1, 0, 2).contiguous())
    scales = nd(fixture["output_scale"].float().reshape(6144))
    bias = fixture.get("bias")
    if bias is not None:
        dtype = torch.int32 if bias.dtype == torch.int32 else torch.float32
        bias = nd(bias.to(dtype))
    bias_kind = 0 if bias is None else (1 if bias.dtype == torch.int32 else 2)
    capacity = 16 if rows <= 16 else 32
    scratch, tiling, blocks, sync_base = torch.ops.tilert.mla_tail_prepare(
        latent, capacity, requested_blocks,
    )
    profile_scratch = nd(torch.empty_like(scratch))
    output = nd(torch.empty(rows, 6144, dtype=torch.bfloat16))
    profile_output = nd(torch.empty_like(output))
    trace = nd(torch.empty(3 * blocks, RECORD_WORDS, dtype=torch.int64))
    if trace.data_ptr() % 512:
        raise ValueError("trace base must be 512-byte aligned")

    def ordinary():
        return torch.ops.tilert.mla_tail_out(
            latent, value_tiles, output_tiles, scales, bias, scratch, tiling,
            output, capacity, blocks, sync_base,
        )

    def instrumented():
        # npu_stream obtains stream(), flushing torch_npu's host task queue.
        # No device synchronization occurs between a producer and this launch.
        stream_pointer = torch.npu.current_stream(args.device).npu_stream
        launch(blocks, stream_pointer, sync_base, latent.data_ptr(),
               value_tiles.data_ptr(), output_tiles.data_ptr(),
               scales.data_ptr(),
               bias.data_ptr() if bias is not None else None, bias_kind,
               profile_scratch.data_ptr(), tiling.data_ptr(),
               profile_output.data_ptr(), trace.data_ptr(), rows, capacity)
        return profile_output

    stream = torch.npu.Stream(device=args.device)
    stream.wait_stream(torch.npu.current_stream(args.device))
    ordinary_graph, ordinary_output = capture(torch, ordinary, stream)
    profile_graph, instrumented_output = capture(torch, instrumented, stream)
    checks = []
    for replay, multiplier in enumerate((1.0, 0.0, -1.0, 0.5)):
        with torch.npu.stream(stream):
            latent.copy_((latent_base.float() * multiplier).bfloat16())
            trace.fill_(-1)
            calls = (ordinary_graph.replay, profile_graph.replay)
            for call in calls[::1 if replay % 2 == 0 else -1]:
                call()
        stream.synchronize()
        check = metrics(torch, instrumented_output, ordinary_output, 0.0, 0.0)
        decoded = decode_trace(trace.cpu().tolist(), rows, capacity, blocks)
        checks.append({"replay": replay, **check})
        if not check["pass"] or check.get("bitwise_equal") is not True:
            return {"rows": rows, "blocks": blocks, "pass": False,
                    "checks": checks, "last_trace": decoded}

    samples = []
    for _ in range(args.samples):
        with torch.npu.stream(stream):
            trace.fill_(-1)
            profile_graph.replay()
        stream.synchronize()
        samples.append(decode_trace(
            trace.cpu().tolist(), rows, capacity, blocks,
        ))

    timing = None
    if not args.no_timing:
        pairs = []
        for pair in range(args.timing_pairs):
            order = (("ordinary", ordinary_graph),
                     ("instrumented", profile_graph))
            pairs.append({
                name: event_time_us(torch, graph, args.iterations, stream)
                for name, graph in order[::1 if pair % 2 == 0 else -1]
            })
        timing = {"pairs_us_per_kernel": pairs, "iterations": args.iterations,
                  "median_us": {
                      name: statistics.median(pair[name] for pair in pairs)
                      for name in ("ordinary", "instrumented")}}
    return {"rows": rows, "capacity": capacity, "blocks": blocks,
            "pass": True, "checks": checks, "timing": timing,
            "raw_samples": samples, "per_core_summary": summarize(samples),
            "trace_bytes": trace.nbytes,
            "scratch_bytes_per_arm": scratch.nbytes}


def main():
    args = arguments()
    import torch
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    torch.ops.load_library(str(args.library.resolve()))
    profile_library, launch = load_launch(args.profile_library)
    fixture = load_inputs(torch, args)
    validate_inputs(torch, fixture, max(args.widths))
    manifest = args.profile_library.parent / "build-manifest.json"
    report = {
        "scope": "MLA tail local SYS_CNT deltas and instrumentation overhead",
        "counter_unit": "raw SYS_CNT ticks; no frequency conversion",
        "cross_core_clock_assumption": None,
        "instrumentation": (
            "PIPE_ALL per stage, final fifth grid barrier, trace stores"
        ),
        "samples": args.samples,
        "provenance": provenance(torch, torch_npu, args),
        "profile_library_sha256": sha256_file(args.profile_library),
        "harness_sha256": sha256_file(Path(__file__)),
        "profile_build_manifest": json.loads(manifest.read_text())
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
                brief = {key: result[key] for key in
                         ("rows", "blocks", "pass", "checks")}
                print(json.dumps(brief), flush=True)
                if not result["pass"]:
                    raise RuntimeError("instrumented MLA changed the output")
        report["pass"] = True
    except Exception as error:
        report["pass"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    del profile_library


if __name__ == "__main__":
    main()
