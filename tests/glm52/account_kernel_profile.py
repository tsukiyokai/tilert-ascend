"""Account a steady-state decode round's kernels against the pass budget.

A layer costs `12 us x passes + 72.7 us`, which puts today's layer at 48
passes and the 2.9 ms target at 11. That 48 is inferred from the floor, not
enumerated, so it names no kernel to remove. This reads the profiler's
`kernel_details.csv` and splits the round's device time into the part the
bytes demand and the part the call count demands.

The split is the lever. A kernel whose duration sits at the 12 us floor costs
that whether it moves 4 KB or 400, so folding it into a neighbour is free; a
kernel above the floor is reading its operands and merging it saves only the
floor. Printed per round and per layer, because every budget in plan.md is.

    python3 account_kernel_profile.py --capture <dir> --rounds <n>

`--rounds` comes from the arm's own record: `session.round_calls` of the
profiled request.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


PASS_FLOOR_US = 12.0
LAYERS = 78


def kernel_details(capture):
    """One rank's `kernel_details.csv` below a profiler output directory.

    Sixteen ranks write sixteen traces into one directory, and all sixteen
    replay the same graph, so the inventory is any one of them. The first in
    sorted order is taken and named in the output, because an inventory that
    silently averaged ranks would hide a rank that diverged.
    """
    found = sorted(capture.rglob("kernel_details.csv"))
    if not found:
        raise FileNotFoundError(f"No kernel_details.csv below {capture}")
    return found[0]


def column(fieldnames, *wanted):
    """The first header containing one of the wanted words, case-free."""
    for word in wanted:
        for name in fieldnames:
            if word in name.lower():
                return name
    raise KeyError(f"No column among {fieldnames} matches {wanted}")


def number(cell):
    """A trace cell as a float; the profiler writes "N/A" for a unit a kernel
    never used, and for a kernel whose duration it could not attribute."""
    try:
        return float(cell)
    except (TypeError, ValueError):
        return 0.0


def busy_columns(fieldnames):
    """Columns reporting a unit actually computing, as Level1 supplies them.

    Their share of a kernel's duration separates a stage that is reading or
    multiplying from one that is waiting. In a batch-one decode every stage is
    a thin GEMV, so a low share says the lever is concurrency rather than pass
    count -- which is the open question about the five native stages.
    """
    return [name for name in fieldnames
            if "mac_time" in name.lower() or "vec_time" in name.lower()]


def read_kernels(path):
    """`(name, type, duration_us, busy_us)` per launch, in trace order."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        name = column(reader.fieldnames, "name")
        kind = column(reader.fieldnames, "type")
        duration = column(reader.fieldnames, "duration")
        busy = busy_columns(reader.fieldnames)
        return [(row[name], row[kind], float(row[duration]),
                 sum(number(row[field]) for field in busy))
                for row in reader if number(row[duration])]


def group(kernels, rounds):
    """Per kernel name: launches and microseconds a round, and the mean."""
    totals = defaultdict(lambda: [0, 0.0, 0.0, ""])
    for name, kind, duration, busy in kernels:
        entry = totals[name]
        entry[0] += 1
        entry[1] += duration
        entry[2] += busy
        entry[3] = kind
    return sorted(
        ({"name": name, "kind": kind, "calls_per_round": calls / rounds,
          "us_per_round": total / rounds, "us_per_call": total / calls,
          "busy_share": busy / total if total else 0.0}
         for name, (calls, total, busy, kind) in totals.items()),
        key=lambda entry: -entry["us_per_round"],
    )


def split_at_the_floor(entries):
    """What the call count demands against what the bytes demand.

    A launch at or below the floor spends the floor whichever bytes it moves,
    so the whole of it is answerable to merging. A launch above the floor
    spends the floor plus its own reading, and merging it recovers the floor
    alone.
    """
    floor_bound = sum(entry["us_per_round"] for entry in entries
                      if entry["us_per_call"] <= PASS_FLOOR_US)
    recoverable = sum(entry["calls_per_round"] * PASS_FLOOR_US
                      for entry in entries
                      if entry["us_per_call"] > PASS_FLOOR_US)
    return floor_bound, recoverable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    trace = kernel_details(args.capture)
    kernels = read_kernels(trace)
    entries = group(kernels, args.rounds)
    round_us = sum(entry["us_per_round"] for entry in entries)
    launches = sum(entry["calls_per_round"] for entry in entries)
    floor_bound, recoverable = split_at_the_floor(entries)

    print(f"{trace}")
    print(f"{len(kernels)} launches over {args.rounds} rounds, "
          f"{len(entries)} distinct kernels")
    print(f"round {round_us / 1000:.3f} ms, "
          f"layer {round_us / args.layers:.1f} us, "
          f"passes {round_us / args.layers / PASS_FLOOR_US:.1f}, "
          f"launches {launches:.1f} a round "
          f"({launches / args.layers:.1f} a layer)")
    print(f"at the floor {floor_bound / 1000:.3f} ms a round, "
          f"floor inside the larger kernels {recoverable / 1000:.3f} ms, "
          f"so merging answers for "
          f"{(floor_bound + recoverable) / round_us:.0%}")
    print()
    print(f"{'us/round':>9} {'calls':>7} {'us/call':>8} {'busy':>5}  kernel")
    for entry in entries[:args.top]:
        print(f"{entry['us_per_round']:9.1f} {entry['calls_per_round']:7.1f} "
              f"{entry['us_per_call']:8.2f} {entry['busy_share']:5.0%}  "
              f"{entry['name']} [{entry['kind']}]")


if __name__ == "__main__":
    raise SystemExit(main())
