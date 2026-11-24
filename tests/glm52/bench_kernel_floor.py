"""Measure the per-kernel floor on one die by timing trivial device work.

A decode round issues several hundred kernels. If a kernel that moves almost
no data still costs tens of microseconds back to back, the round cost is set
by the kernel count rather than by any kernel's inner loop, and fusion is the
only lever. Calls are issued back to back with a single synchronize per batch,
so the host cost is amortized and what remains is the device-side floor.
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


def operations(device):
    """Return trivial device operations, smallest work first."""
    import torch_npu

    scalar = torch.ones(1, device=device, dtype=torch.bfloat16)
    row = torch.ones(1, HIDDEN, device=device, dtype=torch.bfloat16)
    gamma = torch.ones(HIDDEN, device=device, dtype=torch.bfloat16)
    table = torch.ones(8192, HIDDEN, device=device, dtype=torch.bfloat16)
    index = torch.zeros(1, device=device, dtype=torch.int64)
    big = torch.ones(1, 4 * HIDDEN, device=device, dtype=torch.bfloat16)
    wide = torch.ones(8, HIDDEN, device=device, dtype=torch.bfloat16)
    widest = torch.ones(32, HIDDEN, device=device, dtype=torch.bfloat16)
    return {
        "add_1_element": lambda: scalar + scalar,
        "add_1_row": lambda: row + row,
        "add_4_rows": lambda: big + big,
        "rms_norm_1_row": lambda: torch_npu.npu_rms_norm(row, gamma)[0],
        "index_select_1_row": lambda: table.index_select(0, index),
        "empty_like_1_row": lambda: torch.empty_like(row),
        # The MoE combine's two spellings. vLLM-Ascend reaches the
        # multi-tensor kernel for a list of one; a capture puts it at 21.282 us
        # against the plain add's 1.820 at the same shape, and flat in rows.
        "foreach_add_1_row": lambda: torch._foreach_add([row], [row]),
        "add_8_rows": lambda: wide + wide,
        "foreach_add_8_rows": lambda: torch._foreach_add([wide], [wide]),
        "add_32_rows": lambda: widest + widest,
        "foreach_add_32_rows": lambda: torch._foreach_add([widest], [widest]),
        # Two kernels: the FP32 bracket the engine wraps each TP reduction in.
        "cast_round_trip_8_rows": lambda: wide.float().bfloat16(),
        # The rest of the residual combine, scale then add, in two spellings.
        # `alpha` keeps the product in the accumulator, so the one-kernel form
        # is more accurate rather than identical.
        "scale_then_add_8_rows": lambda: (wide * 1.5) + wide,
        "fused_axpy_8_rows": lambda: torch.add(wide, wide, alpha=1.5),
    }


def combine_agrees(device):
    """Does the plain add give the multi-tensor kernel's result exactly?"""
    left = torch.randn(8, HIDDEN, device=device, dtype=torch.bfloat16)
    right = torch.randn(8, HIDDEN, device=device, dtype=torch.bfloat16)
    fused = torch._foreach_add([left], [right])[0]
    return bool(torch.equal(fused, left + right))


def batch_mean_us(call, calls):
    torch.npu.synchronize()
    start = perf_counter_ns()
    for _ in range(calls):
        call()
    torch.npu.synchronize()
    return (perf_counter_ns() - start) / calls / 1e3


def measure(args, report):
    import torch_npu

    torch_npu.npu.set_device(args.device)
    device = torch.device("npu", args.device)
    named = operations(device)

    for call in named.values():
        for _ in range(args.warmup):
            call()
    torch.npu.synchronize()

    samples = {name: [] for name in named}
    for _ in range(args.batches):
        for name, call in named.items():
            samples[name].append(batch_mean_us(call, args.calls))

    report["results"] = [
        {"operation": name,
         "median_us": statistics.median(values),
         "min_us": min(values),
         "p90_us": sorted(values)[-1 - len(values) // 10]}
        for name, values in samples.items()
    ]
    report["combine_bit_exact"] = combine_agrees(device)
    report["status"] = "pass"
    return report["combine_bit_exact"]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--calls", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batches", type=int, default=20)
    return parser.parse_args()


def main():
    args = arguments()
    report = {
        "status": "error",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "hostname": platform.node(),
        "device": args.device, "calls": args.calls, "warmup": args.warmup,
        "batches": args.batches, "hidden": HIDDEN,
        "scope": "one die, one rank, shared with another tenant",
        "issue_note": "calls are back to back with one synchronize per batch",
    }
    try:
        passed = measure(args, report)
    except Exception as error:
        report["error"] = repr(error)[:2000]
        passed = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in report.get("results", []):
        print(json.dumps(row))
    print(json.dumps({"passed": passed}))
    return int(not passed)


if __name__ == "__main__":
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    raise SystemExit(main())
