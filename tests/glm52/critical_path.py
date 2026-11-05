"""Attribute a capture's round to the streams that own its critical path.

`kernel_accounting.py` answers what each kernel costs above its kind's floor.
This answers a different question: of the round's wall time, which stream is
the only one running, and what would its vector work cost if it were limited
by the bytes it moves rather than by the structure around each launch.

Two normalisations decide the answer. A round is one backbone pass, so its
boundary is every 78th attention call, not an even division of the capture --
dividing by a guessed round count read 22% low here. And a stream's share of
the critical path is the time it runs *alone*: a collective that overlaps
compute costs the round nothing, however long its own kernel is.

Run against the directory holding a `*_ascend_pt` tree, after the collector's
`analyse` step has produced `ASCEND_PROFILER_OUTPUT`.
"""

import argparse
import collections
import csv
import glob
import statistics

BACKBONE_LAYERS = 78
MOE_LAYERS = 75
# Per-call floors measured on one die, kernel-floor-20261019. No call can cost
# less than the structure its core kind carries, so the decomposition's call
# counts set a floor on the round independently of what any kernel computes.
FLOOR_US = {"AI_VECTOR_CORE": 0.70, "AI_CORE": 2.30,
            "MIX_AIV": 12.04, "MIX_AIC": 13.58}
BYTES_PER_US = 1.155e6  # a3-hbm-20261017 event median, 1.155 TB/s a die
ELEMENT_BYTES = {"DT_BF16": 2, "FLOAT": 4, "FLOAT16": 2, "INT8": 1,
                 "INT32": 4, "INT64": 8, "UINT8": 1, "BOOL": 1,
                 "DT_INT4": 0.5}
# A grouped matmul declares the whole 256-expert weight tensor whatever it
# reads, so its declared bytes are an upper bound. Scale them by the layer's
# activated union to get the floor: 44.35 of 256 at width 8 on the teacher
# routes. [routes/budget.md]
ROUTED_EXPERTS = 256


def load(root):
    matches = glob.glob(f"{root}/**/ASCEND_PROFILER_OUTPUT/kernel_details.csv",
                        recursive=True)
    if not matches:
        raise SystemExit(f"no kernel_details.csv under {root}")
    rows = []
    for row in csv.DictReader(open(matches[0], newline="")):
        if not row["Start Time(us)"].strip():
            continue
        row["start"] = float(row["Start Time(us)"].strip())
        row["end"] = row["start"] + float(row["Duration(us)"])
        row["shape"] = row["Input Shapes"].strip('"')
        rows.append(row)
    rows.sort(key=lambda row: row["start"])
    return rows


def rounds(rows):
    """Complete backbone passes, as half-open time intervals.

    The first pass is dropped: it carries compilation and first-touch costs
    that no later round pays.
    """
    starts = [row["start"] for row in rows
              if row["Type"] == "SparseFlashAttention"]
    marks = starts[::BACKBONE_LAYERS]
    return list(zip(marks[1:], marks[2:]))


def busy(rows):
    """Wall time at least one of `rows` is running."""
    if not rows:
        return 0.0
    total = 0.0
    spans = sorted((row["start"], row["end"]) for row in rows)
    open_at, close_at = spans[0]
    for start, end in spans[1:]:
        if start > close_at:
            total += close_at - open_at
            open_at, close_at = start, end
        else:
            close_at = max(close_at, end)
    return total + close_at - open_at


def traffic(row):
    """Bytes a kernel limited only by HBM would move: inputs read, outputs
    written. An index-driven gather reads less than its whole source, so this
    over-counts those and under-states nothing else."""
    total = 0
    for shapes, kinds in ((row["Input Shapes"], row["Input Data Types"]),
                          (row["Output Shapes"], row["Output Data Types"])):
        for shape, kind in zip(shapes.strip('"').rstrip(";").split(";"),
                               kinds.split(";")):
            if not shape or kind not in ELEMENT_BYTES:
                continue
            elements = 1
            for extent in shape.split(","):
                elements *= int(extent)
            total += elements * ELEMENT_BYTES[kind]
    return total


def streams(window):
    """Each stream's busy time and the part of it nothing else overlaps."""
    everything = busy(window)
    report = {}
    for name in sorted({row["Stream ID"] for row in window}):
        mine = [row for row in window if row["Stream ID"] == name]
        others = [row for row in window if row["Stream ID"] != name]
        report[name] = (busy(mine), everything - busy(others),
                        collections.Counter(row["Type"] for row in mine))
    return report, everything


def brackets(window, hidden):
    """The FP32 round trip the engine wraps every TP reduction in.

    `cann_model._tp_reduce` widens BF16 to FP32 so the reduction associates in
    FP32, then narrows the result. Both casts are launches of their own, and
    the reduction between them runs on another stream.
    """
    def cast(row, source, target):
        return (row["Type"] == "Cast" and row["shape"] == hidden
                and row["Input Data Types"] == source
                and row["Output Data Types"] == target)

    pairs, opened = [], None
    for index, row in enumerate(window):
        if cast(row, "DT_BF16", "FLOAT"):
            opened = index
        elif cast(row, "FLOAT", "DT_BF16") and opened is not None:
            between = window[opened + 1:index]
            pairs.append((window[opened]["Duration(us)"], row["Duration(us)"],
                          sum(float(each["Duration(us)"])
                              for each in between)))
            opened = None
    return pairs


def vector_cost(window):
    """Vector work against the bytes it moves, and the part that is honest.

    A kernel that runs faster than the bytes attributed to it is one whose
    byte count is wrong -- an index-driven gather touches selected rows, not
    its whole source. Those are reported apart rather than excused by name.
    """
    honest = [0.0, 0.0]
    gathers = 0.0
    for row in window:
        if row["Accelerator Core"] != "AI_VECTOR_CORE":
            continue
        own = float(row["Duration(us)"])
        need = traffic(row) / BYTES_PER_US
        if need > own:
            gathers += own
            continue
        honest[0] += own
        honest[1] += need
    return honest[0], honest[1], gathers


def report(rows, window, hidden):
    inside = [row for row in rows
              if window[0] <= row["start"] < window[1]]
    per_stream, everything = streams(inside)
    span = window[1] - window[0]
    print(f"round {span / 1000:.3f} ms, {len(inside)} kernels, "
          f"{(span - everything) / 1000:.3f} ms with no kernel anywhere")
    print(f"  {'stream':>7}{'kernels':>9}{'busy ms':>10}{'alone ms':>10}"
          f"   heaviest")
    for name, (total, alone, kinds) in sorted(
            per_stream.items(), key=lambda item: -item[1][1]):
        heaviest = ", ".join(kind for kind, _ in kinds.most_common(3))
        print(f"  {name:>7}{sum(kinds.values()):>9}{total / 1000:>10.3f}"
              f"{alone / 1000:>10.3f}   {heaviest}")
    pairs = brackets(inside, hidden)
    if pairs:
        cast = sum(float(up) + float(down) for up, down, _ in pairs)
        reduction = sum(middle for _, _, middle in pairs)
        print(f"  {len(pairs)} FP32 reduction brackets: cast "
              f"{cast / 1000:.3f} ms, reduction between {reduction / 1000:.3f}"
              f" ms, cast share {cast / (cast + reduction):.1%}")
    spent, bound, gathers = vector_cost(inside)
    print(f"  AI_VECTOR_CORE {spent / 1000:.3f} ms against "
          f"{bound / 1000:.3f} ms of bytes, {spent / bound:.0f}x; "
          f"{gathers / 1000:.3f} ms more in kernels this over-counts")
    floor = sum(FLOOR_US.get(row["Accelerator Core"], 0.0) for row in inside)
    mix = sum(FLOOR_US[row["Accelerator Core"]] for row in inside
              if row["Accelerator Core"].startswith("MIX"))
    print(f"  launch floor {floor / 1000:.3f} ms over {len(inside)} calls, "
          f"{mix / floor:.0%} of it on the MIX cores")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("--hidden", default="8,6144",
                        help="hidden-state shape the TP reduction carries")
    args = parser.parse_args()

    rows = load(args.capture)
    windows = rounds(rows)
    print(f"{len(rows)} kernels, {len(windows)} complete rounds after the "
          f"first")
    for window in windows:
        report(rows, window, args.hidden)
    spans = [window[1] - window[0] for window in windows]
    print(f"median round {statistics.median(spans) / 1000:.3f} ms")


if __name__ == "__main__":
    main()
