"""Decode per-worker A3 SYS_CNT windows for the indexed attention phases."""

import statistics


PHASES = ("prolog", "indexer_front", "indexer", "sfa", "projection_tail")
TICKS_PER_US = 50


def read_trace(torch, scratch, offsets, rows, blocks):
    words = offsets["trace_words"]
    marks = offsets["trace_marks"]
    stamp_offset = offsets["trace_stamp_offset"]
    if (words, marks, stamp_offset) != (16, 6, 8):
        raise ValueError("Unsupported indexed attention trace schema")
    begin = offsets["trace"]
    size = 3 * blocks * words * 8
    records = scratch[begin:begin + size].view(torch.int64)
    records = records.reshape(3 * blocks, words).tolist()
    timestamps = []
    for slot, record in enumerate(records):
        kind = int(slot >= blocks)
        index = slot - blocks if kind else slot
        if record[:5] != [1, kind, index, rows, blocks]:
            raise ValueError(f"Invalid trace identity at worker {slot}")
        stamps = record[stamp_offset:stamp_offset + marks]
        if any(b < a for a, b in zip(stamps, stamps[1:])) or stamps[0] <= 0:
            raise ValueError(f"Non-monotonic SYS_CNT at worker {slot}")
        timestamps.append(stamps)
    return timestamps


def summarize(samples, blocks):
    if not samples or any(len(sample) != 3 * blocks for sample in samples):
        raise ValueError("Trace samples require every AIC and AIV worker")
    workers = []
    for slot in range(3 * blocks):
        phase_us = {}
        for index, name in enumerate(PHASES):
            values = [(sample[slot][index + 1] - sample[slot][index]) /
                      TICKS_PER_US for sample in samples]
            phase_us[name] = statistics.median(values)
        local_windows = [(sample[slot][-1] - sample[slot][0]) / TICKS_PER_US
                         for sample in samples]
        workers.append({
            "kind": "aic" if slot < blocks else "aiv",
            "index": slot if slot < blocks else slot - blocks,
            "median_phase_us": phase_us,
            "median_local_window_us": statistics.median(local_windows),
        })
    return {
        "samples": len(samples), "ticks_per_us": TICKS_PER_US,
        "max_per_worker_median_phase_us": {
            name: max(worker["median_phase_us"][name] for worker in workers)
            for name in PHASES
        },
        "max_per_worker_median_local_window_us": max(
            worker["median_local_window_us"] for worker in workers),
        "workers": workers,
    }
