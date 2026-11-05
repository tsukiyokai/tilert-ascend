"""Measure fixed TP16 HCCL sums; event chains include host dispatch gaps."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

WARMUP = 10
SAMPLES = 30
ELEMENTS = (6144, 24576)
CALLS = (1, 156)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def statistics_us(values):
    quantiles = statistics.quantiles(values, n=100, method="inclusive")
    return {"p50": statistics.median(values), "p90": quantiles[89],
            "p99": quantiles[98], "std": statistics.pstdev(values)}


def measure(torch, distributed, buffer, calls, stream, control):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    raw = {"event_total_us": [], "wall_total_us": []}
    for sample in range(WARMUP + SAMPLES):
        torch.npu.synchronize()
        distributed.barrier(group=control)
        wall_start = time.perf_counter_ns()
        with torch.npu.stream(stream):
            start.record(stream)
            for _ in range(calls):
                distributed.all_reduce(buffer, op=distributed.ReduceOp.SUM)
            end.record(stream)
        torch.npu.synchronize()
        wall_us = (time.perf_counter_ns() - wall_start) / 1000
        event_us = start.elapsed_time(end) * 1000
        if sample >= WARMUP:
            raw["event_total_us"].append(event_us)
            raw["wall_total_us"].append(wall_us)
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    device = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 16 or not 0 <= device < 16:
        raise ValueError("Launch sixteen local ranks")
    if os.environ.get("HCCL_OP_EXPANSION_MODE") not in ("AIV", "AI_CPU"):
        raise ValueError("Select HCCL AIV or AI_CPU explicitly")
    toolkit = (Path(os.environ["ASCEND_HOME_PATH"])
               / "aarch64-linux/ascend_toolkit_install.info")
    cann_version = toolkit.read_text()

    import torch
    import torch.distributed as distributed
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(device)
    distributed.init_process_group(backend="hccl")
    control = distributed.new_group(backend="gloo")
    stream = torch.npu.Stream(device=device)
    args.output.mkdir(parents=True, exist_ok=True)
    cases, correctness = [], []
    for dtype in (torch.bfloat16, torch.float32):
        for elements in ELEMENTS:
            with torch.npu.stream(stream):
                buffer = torch.full(
                    (elements,), rank + 1, dtype=dtype, device=f"npu:{device}"
                )
                distributed.all_reduce(buffer, op=distributed.ReduceOp.SUM)
            torch.npu.synchronize()
            exact = bool((buffer.cpu() == 136).all().item())
            with torch.npu.stream(stream):
                buffer.zero_()
            for calls in CALLS:
                cases.append({
                    "dtype": str(dtype), "elements": elements,
                    "payload_bytes": elements * buffer.element_size(),
                    "calls": calls, "buffer_ptr": buffer.data_ptr(),
                    "scope": ("isolated_call" if calls == 1
                              else "156_dependent_inplace_calls"),
                    **measure(torch, distributed, buffer, calls, stream,
                              control),
                })
            correctness.append({
                "dtype": str(dtype), "elements": elements,
                "rank_plus_one_sum_exact_136": exact,
                "zero_after_all_samples": bool(
                    (buffer.cpu() == 0).all().item()
                ),
            })
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rank": rank, "device": device, "world_size": 16,
        "host": platform.node(), "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
        "cann": cann_version, "backend": distributed.get_backend(),
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "command": [sys.executable, *sys.argv],
        "environment": {name: os.environ.get(name) for name in (
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
            "event": "Device start/end around calls; includes dispatch gaps.",
            "wall": "Host submission and final device synchronization.",
            "excluded": "Pre-sample device synchronization and Gloo barrier.",
            "batch_average": "156-call total divided by 156; not call tails.",
        },
        "correctness": correctness, "cases": cases,
        "decoder_executed": False, "profiler": False, "graphs": False,
    }
    save(args.output / f"rank-{rank}.json", report)
    reports = [None] * 16
    distributed.all_gather_object(reports, report, group=control)
    passed = all(item["rank_plus_one_sum_exact_136"]
                 and item["zero_after_all_samples"]
                 for entry in reports for item in entry["correctness"])
    if rank == 0:
        summary = {"status": "pass" if passed else "fail", "cases": []}
        for index, case in enumerate(cases):
            row = {key: case[key] for key in (
                "dtype", "elements", "payload_bytes", "calls", "scope"
            )}
            for field in ("event_total_us", "wall_total_us"):
                maxima = [max(entry["cases"][index][field][sample]
                              for entry in reports)
                          for sample in range(SAMPLES)]
                row[field] = {"rank_max_raw": maxima,
                              **statistics_us(maxima)}
                averages = [value / case["calls"] for value in maxima]
                row[field.replace("total", "per_call")] = {
                    "meaning": ("isolated_call" if case["calls"] == 1
                                else "batch_average"),
                    **statistics_us(averages),
                }
            summary["cases"].append(row)
        save(args.output / "summary.json", summary)
        print(json.dumps({"status": summary["status"],
                          "output": str(args.output)}), flush=True)
    distributed.destroy_process_group()
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
