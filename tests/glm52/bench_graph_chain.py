"""Measure the in-graph cost of one kernel by replaying a serial chain.

A decode round issues several hundred kernels inside one ACL graph. Capturing
a chain of N dependent elementwise kernels and replaying it gives the per
kernel cost with host dispatch already removed: the slope over N is the
device-side cost of a kernel that moves almost no data, and the intercept is
the replay overhead. One chain length per process so a capture failure is
isolated. One device, one rank, no model.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns

import torch

HIDDEN = 6144


def capture_chain(length, device):
    """Capture `length` dependent one-row adds and return the replay handle."""
    import torch_npu  # noqa: F401

    buffer = torch.zeros(1, HIDDEN, device=device, dtype=torch.bfloat16)
    one = torch.ones(1, HIDDEN, device=device, dtype=torch.bfloat16)
    for _ in range(length):
        torch.add(buffer, one, out=buffer)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
        for _ in range(length):
            torch.add(buffer, one, out=buffer)
    torch.npu.synchronize()
    return graph


def batch_mean_us(graph, replays):
    torch.npu.synchronize()
    start = perf_counter_ns()
    for _ in range(replays):
        graph.replay()
    torch.npu.synchronize()
    return (perf_counter_ns() - start) / replays / 1e3


def measure(args, report):
    import torch_npu

    torch_npu.npu.set_device(args.device)
    device = torch.device("npu", args.device)
    graph = capture_chain(args.chain_length, device)

    for _ in range(args.warmup):
        graph.replay()
    samples = sorted(batch_mean_us(graph, args.replays)
                     for _ in range(args.batches))

    median = statistics.median(samples)
    report.update({
        "status": "pass",
        "median_us": median,
        "min_us": samples[0],
        "p90_us": samples[-1 - len(samples) // 10],
        "us_per_kernel": median / args.chain_length,
    })
    return True


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chain-length", type=int, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batches", type=int, default=20)
    return parser.parse_args()


def main():
    args = arguments()
    report = {
        "status": "error", "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "hostname": platform.node(),
        "chain_length": args.chain_length, "device": args.device,
        "replays": args.replays, "warmup": args.warmup,
        "batches": args.batches, "hidden": HIDDEN,
        "scope": "one die, one rank, shared with another tenant",
        "chain_note": "dependent adds on one row; out= keeps the chain serial",
    }
    try:
        passed = measure(args, report)
    except Exception as error:
        report["error"] = repr(error)[:2000]
        passed = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"chain_length": args.chain_length, "passed": passed,
                      "median_us": report.get("median_us"),
                      "us_per_kernel": report.get("us_per_kernel")}))
    return int(not passed)


if __name__ == "__main__":
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    raise SystemExit(main())
