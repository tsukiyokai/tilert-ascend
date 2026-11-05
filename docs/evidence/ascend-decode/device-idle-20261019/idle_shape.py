"""What shape is the device's idle: many small gaps or a few large ones?

Many small gaps mean the host cannot feed the device. A few large ones mean
something blocks. The answer picks the lever, and they need opposite work.
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

spans = []
for event in events:
    if event.get("ph") != "X" or pids.get(event["pid"]) != "Ascend Hardware":
        continue
    name = event.get("name", "")
    if name.startswith("MEM") or name.startswith("EVENT WAIT"):
        continue
    begin = float(event["ts"])
    spans.append((begin, begin + float(event.get("dur", 0))))
spans.sort()

gaps = []
edge = spans[0][1]
for begin, end in spans[1:]:
    if begin > edge:
        gaps.append((begin - edge, edge))
    edge = max(edge, end)

gaps.sort(reverse=True)
total = sum(size for size, _ in gaps)
print(f"gaps {len(gaps)}  total idle {total/1000:.3f} ms")
print(f"\nthe ten largest:")
for size, at in gaps[:10]:
    print(f"   {size:10.1f} us  at {at/1000:9.3f} ms")

for cut in (1000, 100, 10):
    big = [size for size, _ in gaps if size >= cut]
    print(f"\ngaps >= {cut:5} us: {len(big):5}  summing {sum(big)/1000:8.3f} ms "
          f"({sum(big)/total*100:4.1f} % of idle)")
small = [size for size, _ in gaps if size < 10]
print(f"gaps <    10 us: {len(small):5}  summing {sum(small)/1000:8.3f} ms "
      f"({sum(small)/total*100:4.1f} % of idle)")
