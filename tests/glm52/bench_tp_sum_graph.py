"""Measure graph replay of fixed TP16 HCCL sums, not decoder TPOT."""

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import sys
import time

from bench_tp_sum import CALLS, ELEMENTS, SAMPLES, WARMUP, save, statistics_us

# A layer reduces one row of this width per token, twice, over 78 layers, so a
# 156-call chain at `rows` is the whole round's collective term at that verify
# width. Sweeping rows prices the round model's per-row term against it.
HIDDEN = 6144


def measure(torch, distributed, graph, stream, control):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    raw = {"event_total_us": [], "wall_total_us": []}
    for sample in range(WARMUP + SAMPLES):
        torch.npu.synchronize()
        distributed.barrier(group=control)
        wall_start = time.perf_counter_ns()
        with torch.npu.stream(stream):
            start.record(stream)
            graph.replay()
            end.record(stream)
        torch.npu.synchronize()
        wall_us = (time.perf_counter_ns() - wall_start) / 1000
        event_us = start.elapsed_time(end) * 1000
        if sample >= WARMUP:
            raw["event_total_us"].append(event_us)
            raw["wall_total_us"].append(wall_us)
    return raw


def capture_graph(torch, distributed, graph, stream, control, directory, rank):
    """Capture two diagnostic replays after the latency samples."""
    from profile_native_session import (
        make_profiler, npu_range, profiler_identity, trace_inventory,
    )
    import torch_npu

    directory = directory / f"rank-{rank}"
    directory.mkdir(parents=True, exist_ok=False)
    identity = profiler_identity(torch_npu)
    torch.npu.synchronize()
    distributed.barrier(group=control)
    with make_profiler(directory, rank):
        for _ in range(2):
            with npu_range("tp_sum_graph_replay"):
                with torch.npu.stream(stream):
                    graph.replay()
                torch.npu.synchronize()
    distributed.barrier(group=control)
    return {"calls_per_replay": 156, "replays": 2,
            "dtype": "torch.float32", "elements": 24576,
            "profiler_api": identity, "files": trace_inventory(directory)}


def summarize(reports):
    result = {"status": "pass" if all(r["passed"] for r in reports)
              else "fail", "cases": []}
    for index, case in enumerate(reports[0]["cases"]):
        row = {k: case[k] for k in
               ("dtype", "elements", "rows", "payload_bytes", "calls",
                "scope")}
        for field in ("event_total_us", "wall_total_us"):
            maxima = [max(r["cases"][index][field][sample] for r in reports)
                      for sample in range(SAMPLES)]
            row[field] = {"rank_max_raw": maxima, **statistics_us(maxima)}
            row[field.replace("total", "per_call")] = {
                "meaning": "isolated_graph" if case["calls"] == 1
                else "batch_average",
                **statistics_us([x / case["calls"] for x in maxima]),
            }
        result["cases"].append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", metavar="ROWS",
                        help="Verify widths to price; payload is ROWS x 6144")
    parser.add_argument("--profile-output", type=Path,
                        help="Capture two FP32 96 KiB chain replays separately")
    args = parser.parse_args()
    rank, device = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 16 or not 0 <= device < 16:
        raise ValueError("Launch sixteen local ranks")
    if os.environ.get("HCCL_OP_EXPANSION_MODE") not in ("AIV", "AI_CPU"):
        raise ValueError("Select HCCL AIV or AI_CPU explicitly")
    if any(os.environ.get(k) == "1" for k in
           ("TORCH_HCCL_BLOCKING_WAIT", "HCCL_BLOCKING_WAIT")):
        raise ValueError("Host-blocking HCCL wait cannot be captured")
    toolkit = (Path(os.environ["ASCEND_HOME_PATH"])
               / "aarch64-linux/ascend_toolkit_install.info").read_text()

    import torch
    import torch.distributed as distributed
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(device)
    distributed.init_process_group(backend="hccl")
    control = distributed.new_group(backend="gloo")
    stream = torch.npu.Stream(device=device)
    args.output.mkdir(parents=True, exist_ok=True)
    cases, checks, diagnostic = [], [], None
    payloads = ([row * HIDDEN for row in args.rows] if args.rows
                else list(ELEMENTS))
    for dtype in (torch.bfloat16, torch.float32):
        for elements in payloads:
            with torch.npu.stream(stream):
                buffer = torch.zeros(elements, dtype=dtype,
                                     device=f"npu:{device}")
                distributed.all_reduce(buffer)
            torch.npu.synchronize()
            exact = False
            for calls in CALLS:
                distributed.barrier(group=control)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    for _ in range(calls):
                        distributed.all_reduce(buffer,
                                               op=distributed.ReduceOp.SUM)
                if calls == 1:
                    with torch.npu.stream(stream):
                        buffer.fill_(rank + 1)
                        graph.replay()
                    torch.npu.synchronize()
                    exact = bool((buffer.cpu() == 136).all().item())
                    with torch.npu.stream(stream):
                        buffer.zero_()
                raw = measure(torch, distributed, graph, stream, control)
                zero = bool((buffer.cpu() == 0).all().item())
                cases.append({
                    "dtype": str(dtype), "elements": elements, "calls": calls,
                    "rows": elements / HIDDEN,
                    "payload_bytes": elements * buffer.element_size(),
                    "buffer_ptr": buffer.data_ptr(),
                    "scope": f"graph_replay_{calls}_dependent_inplace_sums",
                    "zero_after_samples": zero, **raw,
                })
                if (args.profile_output and dtype == torch.float32
                        and elements == 24576 and calls == 156):
                    diagnostic = capture_graph(
                        torch, distributed, graph, stream, control,
                        args.profile_output, rank,
                    )
                graph.reset()
            checks.append({"dtype": str(dtype), "elements": elements,
                           "captured_single_sum_exact_136": exact})
    sources = [Path(__file__), Path(__file__).with_name("bench_tp_sum.py"),
               Path(torch_npu.__file__).parent / "npu/graphs.py"]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rank": rank, "device": device, "world_size": 16,
        "host": platform.node(), "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
        "cann": toolkit, "backend": distributed.get_backend(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "command": [sys.executable, *sys.argv],
        "environment": {k: os.environ.get(k) for k in (
            "HCCL_DETERMINISTIC", "HCCL_BUFFSIZE", "HCCL_OP_EXPANSION_MODE",
            "HCCL_NPU_SOCKET_PORT_RANGE", "HCCL_HOST_SOCKET_PORT_RANGE",
            "HCCL_ALGO", "HCCL_IF_IP", "HCCL_SOCKET_IFNAME",
            "TASK_QUEUE_ENABLE", "OMP_NUM_THREADS",
            "TORCH_DEVICE_BACKEND_AUTOLOAD", "ASCEND_HOME_PATH",
            "ASCEND_CUSTOM_OPP_PATH", "ASCEND_RT_VISIBLE_DEVICES",
            "ASCEND_VISIBLE_DEVICES",
        )},
        "warmup": WARMUP, "samples": SAMPLES,
        "timing_scope": {
            "event": "Start/end around one replay, including replay dispatch.",
            "wall": "Replay submission and final device synchronization.",
            "excluded": "Capture, correctness, pre-sync and Gloo barrier.",
            "batch_average": "156-node total/156, not individual call tails.",
            "profile": "Optional diagnostic replay follows all timing samples.",
        },
        "cases": cases, "correctness": checks,
        "passed": all(c["captured_single_sum_exact_136"] for c in checks)
        and all(c["zero_after_samples"] for c in cases),
        "decoder_executed": False, "profiler": False, "graphs": True,
        "bare_hccl_latency_measured": False,
        "diagnostic_profile": diagnostic,
    }
    save(args.output / f"rank-{rank}.json", report)
    reports = [None] * 16
    distributed.all_gather_object(reports, report, group=control)
    summary = summarize(reports)
    if rank == 0:
        save(args.output / "summary.json", summary)
        print(summary["status"], args.output, flush=True)
    distributed.destroy_process_group()
    return int(summary["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
