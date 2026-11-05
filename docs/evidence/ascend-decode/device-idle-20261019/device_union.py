"""How much of the wall is the device actually computing?

Summing per-operator times double counts concurrent tracks, which is how the
earlier 35 ms "overhead" figure was built. The union of busy spans against the
profile's own wall clock does not have that defect.
"""
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


def union(spans):
    spans = sorted(spans)
    total = 0.0
    edge = None
    for begin, end in spans:
        if edge is None or begin > edge:
            total += end - begin
            edge = end
        elif end > edge:
            total += end - edge
            edge = end
    return total


work = []
groups = []
for event in events:
    if event.get("ph") != "X" or pids.get(event["pid"]) != "Ascend Hardware":
        continue
    name = event.get("name", "")
    if name.startswith("MEM") or name.startswith("EVENT WAIT"):
        continue
    begin = float(event["ts"])
    span = (begin, begin + float(event.get("dur", 0)))
    work.append(span)
    if name.startswith("sk_"):
        groups.append(span)

first = min(begin for begin, _ in work)
last = max(end for _, end in work)
wall = last - first
busy = union(work)
group_union = union(groups)

print(f"profile wall            {wall/1000:9.3f} ms")
print(f"device busy (union)     {busy/1000:9.3f} ms   {busy/wall*100:5.1f} % of wall")
print(f"device idle             {(wall-busy)/1000:9.3f} ms   {(wall-busy)/wall*100:5.1f} %")
print(f"  of the busy union, SuperKernel groups {group_union/1000:.3f} ms "
      f"({group_union/busy*100:.1f} %)")
