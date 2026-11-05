"""How many kernels sit inside one SuperKernel group, and what do they cost?

If a group is one dispatch wrapping N complete kernels, the hardware timeline
should show those N inside the group's span. That decides whether the
per-operator prologue survives grouping -- without running anything.
"""
import collections
import glob
import json

path = glob.glob("/workspace/experiments/profile-rope-20261018/raw/raw-rank-0/"
                 "*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json")[0]
with open(path) as handle:
    events = json.load(handle)

pids = {}
for event in events:
    if event.get("ph") == "M" and event.get("name") == "process_name":
        pids[event["pid"]] = event["args"]["name"]

spans = []
for event in events:
    if event.get("ph") != "X" or pids.get(event["pid"]) != "Ascend Hardware":
        continue
    begin = float(event["ts"])
    spans.append((begin, begin + float(event.get("dur", 0)), event.get("name", ""),
                  event.get("tid")))
spans.sort()

groups = [s for s in spans if s[2].startswith("sk_")]
others = [s for s in spans if not s[2].startswith("sk_")]
print(f"hardware spans {len(spans)}  groups {len(groups)}  non-group {len(others)}")

# Does anything land strictly inside a group's span on any track?
nested = collections.Counter()
inside_total = 0.0
group_total = sum(end - begin for begin, end, _, _ in groups)
for begin, end, _, _ in groups:
    for obegin, oend, oname, _ in others:
        if obegin >= begin and oend <= end:
            nested[oname.split("_")[0][:28]] += 1
            inside_total += oend - obegin
print(f"group wall total {group_total/1000:.3f} ms")
print(f"spans nested inside a group: {sum(nested.values())}, their wall {inside_total/1000:.3f} ms")
for name, count in nested.most_common(8):
    print(f"   {count:5}  {name}")

# A group's own duration against the gap to the next group on the same track.
by_track = collections.defaultdict(list)
for begin, end, name, tid in groups:
    by_track[tid].append((begin, end, name))
gaps = []
for track, items in by_track.items():
    items.sort()
    for earlier, later in zip(items, items[1:]):
        gaps.append(later[0] - earlier[1])
if gaps:
    gaps.sort()
    print(f"\ninter-group gap on the same track: median {gaps[len(gaps)//2]:.3f} us, "
          f"p10 {gaps[len(gaps)//10]:.3f}, p90 {gaps[9*len(gaps)//10]:.3f}")
durations = sorted(end - begin for begin, end, _, _ in groups)
print(f"group duration: median {durations[len(durations)//2]:.3f} us, "
      f"p10 {durations[len(durations)//10]:.3f}, p90 {durations[9*len(durations)//10]:.3f}")
