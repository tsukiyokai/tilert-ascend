"""Split a capture's device time into structure, floors and execution.

Three questions this answers on any profile, which thirteen one-off queries
answered once each today:

  what does each kind of kernel cost at minimum -- its floor, since no call
  can be cheaper than the structure it carries;

  what does each operator spend above its kind's floor, which is its own
  execution rather than the structure or the dispatch;

  and at what shapes, which is what a floor in bytes needs.

Run against the directory holding a `*_ascend_pt` tree. The capture must be
analysed first: the collector runs with `analyse: False`, so
`torch_npu.profiler.profiler.analyse(profiler_path=...)` has to produce
`ASCEND_PROFILER_OUTPUT` before this reads it.
"""

import argparse
import collections
import csv
import glob
import statistics

KINDS = ("AI_VECTOR_CORE", "AI_CORE", "MIX_AIV", "MIX_AIC")


def load(root):
    matches = glob.glob(f"{root}/**/ASCEND_PROFILER_OUTPUT/kernel_details.csv",
                        recursive=True)
    if not matches:
        raise SystemExit(f"no analysed capture under {root}")
    with open(sorted(matches)[0], newline="") as handle:
        return sorted(matches)[0], list(csv.DictReader(handle))


def floors(rows):
    """The cheapest call of each kind, which bounds that kind's fixed cost."""
    by_kind = collections.defaultdict(list)
    for row in rows:
        if row["Accelerator Core"] in KINDS:
            by_kind[row["Accelerator Core"]].append(float(row["Duration(us)"]))
    return {kind: min(times) for kind, times in by_kind.items()}


def report_floors(rows, verifies):
    found = floors(rows)
    counts = collections.Counter(row["Accelerator Core"] for row in rows)
    print(f"{'kind':>16} {'calls':>7} {'a verify':>9} {'floor':>7} {'ms a verify':>12}")
    for kind in KINDS:
        if kind not in found:
            continue
        per_verify = counts[kind] / verifies
        print(f"{kind:>16} {counts[kind]:7} {per_verify:9.0f} "
              f"{found[kind]:7.2f} {per_verify * found[kind] / 1000:12.3f}")

    single = min((found[k] for k in ("AI_VECTOR_CORE", "AI_CORE") if k in found),
                 default=None)
    mixed = min((found[k] for k in ("MIX_AIV", "MIX_AIC") if k in found),
                default=None)
    if single is not None and mixed is not None:
        mix_calls = sum(counts[k] for k in ("MIX_AIV", "MIX_AIC")) / verifies
        print(f"\nthe cross-core structure costs at least {mixed - single:.2f} us, "
              f"and over {mix_calls:.0f} MIX kernels a verify "
              f"{mix_calls * (mixed - single) / 1000:.2f} ms")


def report_above_floor(rows, verifies, limit):
    """What each operator spends beyond the floor of its own kind."""
    found = floors(rows)
    by_op = collections.defaultdict(list)
    for row in rows:
        kind = row["Accelerator Core"]
        if kind in found:
            by_op[(row["Type"], kind)].append(float(row["Duration(us)"]))

    ranked = []
    for (name, kind), times in by_op.items():
        times.sort()
        above = max(times[len(times) // 10] - found[kind], 0.0)
        ranked.append((above * len(times) / verifies / 1000, above,
                       times[len(times) // 10], len(times) / verifies, name, kind))
    ranked.sort(reverse=True)

    print(f"\n{'ms a verify':>11} {'above':>7} {'p10':>7} {'calls':>6}  operator")
    for cost, above, p10, calls, name, kind in ranked[:limit]:
        print(f"{cost:11.3f} {above:7.2f} {p10:7.2f} {calls:6.0f}  "
              f"{name[:32]} [{kind}]")
    print(f"{sum(item[0] for item in ranked):11.3f} {'':7} {'':7} {'':6}  total")


def report_shapes(rows, names):
    for wanted in names:
        items = [row for row in rows if row["Type"] == wanted]
        if not items:
            print(f"\n### {wanted}: absent")
            continue
        times = sorted(float(row["Duration(us)"]) for row in items)
        print(f"\n### {wanted}: {len(items)} calls, p10 {times[len(times)//10]:.2f}, "
              f"median {statistics.median(times):.2f} us")
        shapes = collections.Counter(row["Input Shapes"].strip('"') for row in items)
        for shape, count in shapes.most_common(2):
            print(f"  {count:5}  blocks {items[0]['Block Num']}/{items[0]['Mix Block Num']}"
                  f"  {shape[:150]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("--verifies", type=int, default=4,
                        help="forwards of the measured graph in the capture")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--shapes", nargs="*", default=(),
                        help="operator types whose shapes to print")
    args = parser.parse_args()

    path, rows = load(args.capture)
    print(f"{path}\n{len(rows)} kernels, {args.verifies} forwards assumed")
    report_floors(rows, args.verifies)
    report_above_floor(rows, args.verifies, args.limit)
    report_shapes(rows, args.shapes)


if __name__ == "__main__":
    main()
