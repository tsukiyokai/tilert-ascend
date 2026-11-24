"""Host wall of the per-forward metadata copies, pageable against pinned.

`CannModel._metadata` runs once per forward -- six chain steps, the target and
the commit in every round -- and each call allocates three CPU tensors and
copies them to the device. A copy from pageable host memory has to stage, and
the runtime blocks the caller to do it, so the cost lands on host wall rather
than on any stream. That is the shape of the 5.53 ms a round that the hooked
phases hold and the compute union does not.

The device must have queued work for the measurement to mean anything: an H2D
into an idle stream stages immediately. One device, no model, no collective.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from time import perf_counter_ns

import torch

HIDDEN = 6144


def queue_work(buffer, depth):
    """Leave `depth` dependent kernels ahead of the copies under test."""
    for _ in range(depth):
        torch.add(buffer, buffer, out=buffer)


def pageable(position, width, slots, positions, seq_lens):
    """What `_metadata` did: fresh host tensors, converting copies."""
    length = torch.tensor([position + width], dtype=torch.int32)
    torch.tensor([position], dtype=torch.int32)
    rows = torch.arange(position, position + width, dtype=torch.long)
    seq_lens.copy_(length)
    if width == 1:
        slots.fill_(position)
        positions.fill_(position)
    else:
        slots.copy_(rows)
        positions.copy_(rows)


def pinned(position, width, slots, positions, seq_lens, host):
    """What it does now: persistent pinned staging, one per dtype."""
    length, computed, rows, rows_int32 = host
    length[0] = position + width
    computed[0] = position
    torch.arange(position, position + width, dtype=torch.long, out=rows)
    seq_lens.copy_(length, non_blocking=True)
    if width == 1:
        slots.fill_(position)
        positions.fill_(position)
    else:
        rows_int32.copy_(rows)
        slots.copy_(rows_int32, non_blocking=True)
        positions.copy_(rows, non_blocking=True)


def measure(arm, width, device, repeats, depth):
    slots = torch.zeros(width, dtype=torch.int32, device=device)
    positions = torch.zeros(width, dtype=torch.long, device=device)
    seq_lens = torch.zeros(1, dtype=torch.int32, device=device)
    work = torch.ones(1, HIDDEN, dtype=torch.bfloat16, device=device)
    host = (torch.zeros(1, dtype=torch.int32).pin_memory(),
            torch.zeros(1, dtype=torch.int32).pin_memory(),
            torch.zeros(width, dtype=torch.long).pin_memory(),
            torch.zeros(width, dtype=torch.int32).pin_memory())

    samples = []
    for step in range(repeats):
        queue_work(work, depth)
        start = perf_counter_ns()
        if arm == "pageable":
            pageable(step, width, slots, positions, seq_lens)
        else:
            pinned(step, width, slots, positions, seq_lens, host)
        samples.append((perf_counter_ns() - start) / 1e3)
        torch.npu.synchronize()
    samples.sort()
    return {
        "arm": arm, "width": width, "queue_depth": depth,
        "repeats": repeats,
        "p10_us": samples[len(samples) // 10],
        "median_us": statistics.median(samples),
        "p90_us": samples[9 * len(samples) // 10],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--queue-depth", type=int, nargs="+",
                        default=[0, 64, 512])
    parser.add_argument("--width", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    rows = []
    for depth in args.queue_depth:
        for width in args.width:
            for arm in ("pageable", "pinned"):
                row = measure(arm, width, args.device, args.repeats, depth)
                rows.append(row)
                print(f"depth {depth:4d}  width {width}  {arm:8s}  "
                      f"p10 {row['p10_us']:8.2f}  "
                      f"median {row['median_us']:8.2f}  "
                      f"p90 {row['p90_us']:8.2f} us")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "host wall of three metadata copies, one device, no model",
        "device": args.device, "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
