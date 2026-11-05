"""Inspect routed-stage local SYS_CNT deltas and instrumentation overhead."""

import argparse
import ctypes
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
STAGES = ("restore_dispatch", "metadata_and_split", "gate_up_matmul",
          "activate", "down_matmul", "store_routes", "combine")
RECORD_WORDS, TIMESTAMP_OFFSET, ADDRESS_COUNT = 64, 8, 29


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--library", type=Path,
                        default=HERE / "build-route/routed_int4_torch.so")
    parser.add_argument("--profile-library", type=Path,
                        default=HERE / "build-profile/routed_int4_profile.so")
    parser.add_argument("--native-layout", choices=("nd", "nz"), default="nz")
    parser.add_argument("--native-schedule", choices=("expert", "row"),
                        default="expert")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--fixture", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--rank", type=int, choices=range(16), default=0)
    parser.add_argument("--widths", type=int, nargs="+", default=[12, 32])
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--replays", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--timing-pairs", type=int, default=5)
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(width < 1 or width > 32 for width in args.widths):
        parser.error("widths must lie in [1,32]")
    if min(args.blocks, args.samples, args.warmup,
           args.iterations, args.timing_pairs) < 1 or args.replays < 4:
        parser.error("positive counts and at least four changing-route replays required")
    return args


def decode_trace(records, tokens, blocks, row_balanced=False):
    if len(records) != 3 * blocks:
        raise ValueError("trace requires one record per AIC and AIV")
    decoded = []
    for slot, record in enumerate(records):
        kind, index = (0, slot) if slot < blocks else (1, slot - blocks)
        if len(record) != RECORD_WORDS:
            raise ValueError("trace record has the wrong size")
        if record[:5] != [1, kind, index, tokens, blocks]:
            raise ValueError(f"trace record {slot} has stale or invalid metadata")
        ticks = record[TIMESTAMP_OFFSET:TIMESTAMP_OFFSET + 3 * len(STAGES)]
        if any(before > after for before, after in zip(ticks, ticks[1:])):
            raise ValueError(f"trace record {slot} has nonmonotonic timestamps")
        window = ticks[-1] - ticks[0]
        if window <= 0:
            raise ValueError(f"trace record {slot} has no elapsed local ticks")
        stages = []
        for stage, name in enumerate(STAGES):
            begin, completed, released = ticks[stage * 3:stage * 3 + 3]
            active = (stage in (2, 4)) if kind == 0 else (
                stage in (0, 1, 5, 6)
                or stage == 3 and (row_balanced or index % 2 == 0)
            )
            stages.append({
                "stage": name, "performs_stage_work": active,
                "work_ticks": completed - begin,
                "barrier_ticks": released - completed,
            })
        decoded.append({"kind": "AIC" if kind == 0 else "AIV",
                        "core_index": index, "window_ticks": window,
                        "stages": stages, "raw_ticks": ticks})
    return decoded


def summarize(samples):
    result = []
    for slot, record in enumerate(samples[0]):
        stages = []
        for stage, entry in enumerate(record["stages"]):
            values = [sample[slot]["stages"][stage] for sample in samples]
            stages.append({
                "stage": entry["stage"],
                "performs_stage_work": entry["performs_stage_work"],
                "median_work_ticks": statistics.median(
                    value["work_ticks"] for value in values),
                "median_barrier_ticks": statistics.median(
                    value["barrier_ticks"] for value in values),
            })
        result.append({"kind": record["kind"], "core_index": record["core_index"],
                       "median_window_ticks": statistics.median(
                           sample[slot]["window_ticks"] for sample in samples),
                       "stages": stages})
    return result


def route_geometry(torch, choices, blocks):
    counts = torch.bincount(choices.flatten().long(), minlength=256)
    nonempty = counts[counts != 0].tolist()
    return {
        "active_experts": len(nonempty), "counts": counts.tolist(),
        "rows_per_cube_owner": [sum(nonempty[core::blocks])
                                for core in range(blocks)],
        "experts_per_cube_owner": [len(nonempty[core::blocks])
                                   for core in range(blocks)],
        "minimum_weight_bytes": len(nonempty) * 6144 * 128 * 3 // 2,
    }


def load_launch(path):
    owner = ctypes.CDLL(str(path.resolve()))
    launch = owner.routed_int4_profile_launch
    launch.restype = None
    launch.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint64,
                      ctypes.POINTER(ctypes.c_uint64), ctypes.c_int64,
                      ctypes.c_int64]
    return owner, launch


def run_case(torch, torch_npu, check, args, weights, launch, width):
    def nd(value):
        return torch_npu.npu_format_cast(value.to(args.device).contiguous(), 2)

    hidden = nd(torch.empty(width, check.HIDDEN, dtype=torch.bfloat16))
    choices = nd(torch.empty(width, check.TOP_K, dtype=torch.int32))
    probabilities = nd(torch.empty(width, check.TOP_K, dtype=torch.float32))
    sample = nd(torch.empty(width, check.HIDDEN, dtype=torch.int8))
    layout = int(args.native_layout == "nz")
    operations = (torch.ops.tilert_balanced if args.native_schedule == "row"
                  else torch.ops.tilert)
    scratch, sync = operations.routed_experts_prepare(
        sample, check.EXPERTS, args.blocks, layout, True,
    )
    measured, measured_sync = operations.routed_experts_prepare(
        sample, check.EXPERTS, args.blocks, layout, True,
    )
    trace = nd(torch.empty(3 * args.blocks, RECORD_WORDS, dtype=torch.int64))
    if trace.data_ptr() % 512:
        raise ValueError("trace base must be 512-byte aligned")
    ordered = [weights[name] for name in
               ("gate_up", "gu_bias", "gu_scale", "down", "down_bias", "down_scale")]

    def dispatch():
        return torch_npu.npu_moe_init_routing_v2(
            hidden, choices, active_num=width * check.TOP_K,
            expert_num=check.EXPERTS, expert_tokens_num_type=1,
            expert_tokens_num_flag=True, active_expert_range=[0, check.EXPERTS],
            quant_mode=1,
        )

    def ordinary():
        packed, inverse, counts, scales = dispatch()
        return operations.routed_experts_from_dispatch(
            packed, *ordered, scales.reshape(-1), inverse, counts,
            probabilities, scratch, sync, check.EXPERTS, args.blocks, layout,
        )

    def instrumented():
        packed, inverse, counts, scales = dispatch()
        tensors = ordered + [packed, scales, inverse, counts, probabilities]
        tensors += list(measured) + [trace]
        if len(tensors) != ADDRESS_COUNT:
            raise ValueError("profile launch address contract changed")
        pointers = (ctypes.c_uint64 * ADDRESS_COUNT)(
            *(tensor.data_ptr() for tensor in tensors)
        )
        # Accessing npu_stream flushes the host task queue before direct launch.
        stream_pointer = torch.npu.current_stream(args.device).npu_stream
        launch(args.blocks, stream_pointer, measured_sync, pointers, width, layout)
        return measured[9]

    stream = torch.npu.Stream(device=args.device)
    stream.wait_stream(torch.npu.current_stream(args.device))

    def update(iteration):
        rows, ids, probs = check.route_case(torch, width, iteration, args.seed)
        with torch.npu.stream(stream):
            hidden.copy_(rows)
            choices.copy_(ids)
            probabilities.copy_(probs)
        stream.synchronize()
        return route_geometry(torch, ids, args.blocks)

    update(0)
    ordinary_graph, ordinary_output = check.capture(
        torch, ordinary, stream, args.warmup,
    )
    profile_graph, profile_output = check.capture(
        torch, instrumented, stream, args.warmup,
    )
    report = {"width": width, "blocks": args.blocks, "checks": [], "pass": True}
    for iteration in range(args.replays):
        geometry = update(iteration)
        with torch.npu.stream(stream):
            trace.fill_(-1)
            order = (ordinary_graph.replay, profile_graph.replay)
            for call in order[::1 if iteration % 2 == 0 else -1]:
                call()
        stream.synchronize()
        result = check.compare(torch, profile_output, ordinary_output, 0, 0)
        result.update(iteration=iteration, routing=geometry)
        result["boundaries"] = {
            name: check.compare(torch, measured[index], scratch[index], 0, 0)
            for name, index in (("middle_scale", 3), ("routes", 5))
        }
        result["pass"] = (result["pass"] and result["bitwise_equal"]
                          and all(value["bitwise_equal"]
                                  for value in result["boundaries"].values()))
        report["checks"].append(result)
        decoded = decode_trace(trace.cpu().tolist(), width, args.blocks,
                               args.native_schedule == "row")
        if not result["pass"]:
            report.update({"pass": False, "last_trace": decoded})
            return report

    samples = []
    for _ in range(args.samples):
        with torch.npu.stream(stream):
            trace.fill_(-1)
            profile_graph.replay()
        stream.synchronize()
        samples.append(decode_trace(trace.cpu().tolist(), width, args.blocks,
                                    args.native_schedule == "row"))
    report.update(raw_samples=samples, per_core_summary=summarize(samples),
                  measured_routing=geometry, trace_bytes=trace.nbytes,
                  scratch_bytes_per_arm=sum(value.nbytes for value in scratch))
    if not args.no_timing:
        pairs = []
        for pair in range(args.timing_pairs):
            order = (("ordinary", ordinary_graph), ("instrumented", profile_graph))
            pairs.append({name: check.elapsed(torch, graph, stream, args.iterations)
                          for name, graph in order[::1 if pair % 2 == 0 else -1]})
        report["timing"] = {
            "scope": "BF16 vendor dispatch through final combine, fixed last routes",
            "pairs_us": pairs,
            "median_us": {name: statistics.median(pair[name] for pair in pairs)
                          for name in ("ordinary", "instrumented")},
        }
    return report


def main():
    args = arguments()
    manifest = args.profile_library.parent / "build-manifest.json"
    build_manifest = json.loads(manifest.read_text()) if manifest.is_file() else None
    if (build_manifest is not None and
            build_manifest.get("row_balanced", False)
            != (args.native_schedule == "row")):
        raise ValueError("Profile build and requested row schedule differ")
    sys.path.insert(0, str(HERE.parents[2]))
    import torch
    import torch_npu

    spec = importlib.util.spec_from_file_location("routed_check", HERE / "check.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    torch.ops.load_library(str(args.library.resolve()))
    profile_owner, launch = load_launch(args.profile_library)
    source = check.load_weights(torch, args)
    weights = {name: torch_npu.npu_format_cast(value.to(args.device), 2)
               for name, value in source.items()}
    if args.native_layout == "nz":
        for name in ("gate_up", "down"):
            weights[name] = torch_npu.npu_format_cast(weights[name], 29).view(torch.int32)
    report = {
        "scope": "TP16 routed stage local SYS_CNT deltas and instrumentation overhead",
        "counter_unit": "raw SYS_CNT ticks; no frequency conversion",
        "cross_core_clock_assumption": None,
        "instrumentation": "PIPE_ALL before stage completion, one final grid barrier, "
                           "then one private 512-byte record per core",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "cann": check.cann_versions(torch_npu),
        "device": str(torch.npu.get_device_properties(args.device)),
        "library_sha256": check.digest(args.library),
        "profile_library_sha256": check.digest(args.profile_library),
        "harness_sha256": check.digest(Path(__file__)),
        "profile_build_manifest": build_manifest,
        "cases": [], "pass": False,
    }
    try:
        with torch.inference_mode():
            for width in args.widths:
                case = run_case(torch, torch_npu, check, args, weights, launch, width)
                report["cases"].append(case)
                print(json.dumps({"width": width, "pass": case["pass"]}), flush=True)
                if not case["pass"]:
                    raise RuntimeError("profile instrumentation changed routed output")
        report["pass"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["loaded_libraries_sha256"] = check.loaded_libraries()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    del profile_owner


if __name__ == "__main__":
    main()
