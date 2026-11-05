"""Interpret local FFN phase windows without summing different workers."""

import statistics


PHASES = (
    "residual_norm", "gate_and_shared_quant", "topk", "body_setup",
    "dispatch_and_shared_stage", "split_and_shared_gate",
    "routed_gate_and_shared_activation", "routed_activation_and_shared_down",
    "routed_down_and_shared_store", "route_store", "combine_and_publish",
    "output_epilogue",
)
TICKS_PER_US = 50


def read_trace(records, rows, blocks):
    if len(records) != 3 * blocks:
        raise ValueError("Trace must contain every resident AIC and AIV")
    samples = []
    for slot, record in enumerate(records):
        kind = int(slot >= blocks)
        index = slot - blocks if kind else slot
        if len(record) != 32 or record[:5] != [1, kind, index, rows, blocks]:
            raise ValueError(f"Invalid trace identity at worker {slot}")
        marks = record[8:21]
        if marks[0] <= 0 or any(b < a for a, b in zip(marks, marks[1:])):
            raise ValueError(f"Non-monotonic SYS_CNT at worker {slot}")
        samples.append(marks)
    return samples


def read_sparse_trace(records, rows, blocks):
    samples = read_trace(records, rows, blocks)
    for slot, (record, marks) in enumerate(zip(records, samples)):
        if marks != [marks[3]] * 4 + [marks[4]] * 9:
            raise ValueError(f"Unexpected sparse timestamp fields at worker {slot}")
        if any(record[5:8] + record[21:]):
            raise ValueError(f"Nonzero trace padding at worker {slot}")
    return samples


def summarize_sparse(samples, blocks):
    if not samples or any(len(sample) != 3 * blocks for sample in samples):
        raise ValueError("Require at least one complete sparse trace")
    workers = []
    for slot in range(3 * blocks):
        windows = [(sample[slot][4] - sample[slot][3]) / TICKS_PER_US
                   for sample in samples]
        workers.append({
            "kind": "aic" if slot < blocks else "aiv",
            "index": slot if slot < blocks else slot - blocks,
            "median_body_setup_us": statistics.median(windows),
        })
    critical = max(workers, key=lambda worker: worker["median_body_setup_us"])
    return {
        "samples": len(samples), "ticks_per_us": TICKS_PER_US,
        "sampled_marks": [3, 4], "clock_reads_per_worker": 2,
        "interpretation": "Only the interval after the top-k EndPhase and before the body is sampled. Repeated timestamp fields retain the trace-store footprint; they do not measure other phases. Launch and final trace stores are outside this window.",
        "max_per_worker_median_body_setup_us": critical["median_body_setup_us"],
        "critical_worker": critical, "workers": workers,
    }


def summarize(samples, blocks):
    if not samples or any(len(sample) != 3 * blocks for sample in samples):
        raise ValueError("Require at least one complete trace")
    workers = []
    for slot in range(3 * blocks):
        windows = [(sample[slot][-1] - sample[slot][0]) / TICKS_PER_US
                   for sample in samples]
        phases = {name: statistics.median(
            (sample[slot][index + 1] - sample[slot][index]) / TICKS_PER_US
            for sample in samples) for index, name in enumerate(PHASES)}
        workers.append({
            "kind": "aic" if slot < blocks else "aiv",
            "index": slot if slot < blocks else slot - blocks,
            "median_local_window_us": statistics.median(windows),
            "median_phase_us": phases,
        })
    critical = max(workers, key=lambda worker: worker["median_local_window_us"])
    return {
        "samples": len(samples), "ticks_per_us": TICKS_PER_US,
        "interpretation": "Per-worker SYS_CNT windows include work, waits and clock reads; maxima from different workers are not additive. Kernel launch and final trace stores are outside these windows.",
        "max_per_worker_median_phase_us": {
            name: max(worker["median_phase_us"][name] for worker in workers)
            for name in PHASES},
        "max_per_worker_median_local_window_us": critical["median_local_window_us"],
        "critical_worker": critical,
        "workers": workers,
    }
