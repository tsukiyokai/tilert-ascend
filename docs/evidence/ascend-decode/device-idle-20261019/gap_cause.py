"""What ends just before the device's large idle gaps?

The steady-state idle is entirely in gaps of 100 us and more, 172 of them. The
span that ends at a gap's start names what the device was last told to do, and
the span that starts at its end names what it waited for.
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
clock = []
for event in events:
    if event.get("ph") != "X" or pids.get(event["pid"]) != "Ascend Hardware":
        continue
    name = event.get("name", "")
    if name.startswith("MEM") or name.startswith("EVENT WAIT"):
        continue
    begin = float(event["ts"])
    spans.append((begin, begin + float(event.get("dur", 0)), name))
    if "all_reduce_float" in name:
        clock.append(begin)
spans.sort()
clock.sort()
lo, hi = clock[len(clock) // 4], clock[3 * len(clock) // 4]
inside = [s for s in spans if s[0] >= lo and s[1] <= hi]


def label(name):
    if name.startswith("sk_"):
        import re
        body = re.sub(r"^sk_\d+_full_model_start_", "", name)
        return "sk:" + body.split("_end_")[0][:40]
    return name.split("_")[0][:40]


before = collections.Counter()
after = collections.Counter()
weight = collections.Counter()
edge, edge_name = inside[0][1], inside[0][2]
for begin, end, name in inside[1:]:
    if begin > edge + 100:
        before[label(edge_name)] += 1
        weight[label(edge_name)] += begin - edge
        after[label(name)] += 1
    if end > edge:
        edge, edge_name = end, name

print("what ends before a gap, by idle it precedes:")
for name, total in weight.most_common(10):
    print(f"  {total/1000:8.3f} ms  {before[name]:4} gaps  {total/before[name]:8.1f} us each  {name}")
print("\nwhat starts after a gap:")
for name, count in after.most_common(6):
    print(f"  {count:4}  {name}")
