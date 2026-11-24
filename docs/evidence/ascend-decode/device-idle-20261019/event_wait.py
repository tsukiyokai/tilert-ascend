"""Is EVENT WAIT lost time, or bookkeeping while another stream works?

404 ms over 599 spans is the largest item on the device timeline. A wait on
one stream costs nothing if another stream is computing through it, so the
question is what overlaps it.
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

waits = []
work = []
for event in events:
    if event.get("ph") != "X" or pids.get(event["pid"]) != "Ascend Hardware":
        continue
    name = event.get("name", "")
    if name.startswith("MEM"):
        continue
    begin = float(event["ts"])
    span = (begin, begin + float(event.get("dur", 0)), name, event.get("tid"))
    (waits if name.startswith("EVENT WAIT") else work).append(span)

waits.sort()
work.sort()
print(f"waits {len(waits)}  work spans {len(work)}")
tracks = collections.Counter(tid for _, _, _, tid in waits)
print(f"wait tracks: {dict(tracks.most_common(5))}")

# For each wait, how much of its span is covered by work on any other track?
covered_total = 0.0
wait_total = 0.0
fully_idle = 0
for wbegin, wend, _, wtid in waits:
    wait_total += wend - wbegin
    pieces = [(max(b, wbegin), min(e, wend)) for b, e, _, tid in work
              if tid != wtid and b < wend and e > wbegin]
    pieces.sort()
    covered = 0.0
    edge = wbegin
    for begin, end in pieces:
        if end <= edge:
            continue
        covered += end - max(begin, edge)
        edge = end
    covered_total += covered
    if covered == 0.0:
        fully_idle += 1

print(f"\nwait wall           {wait_total/1000:9.3f} ms")
print(f"overlapped by work  {covered_total/1000:9.3f} ms  ({covered_total/wait_total*100:.1f} %)")
print(f"unoverlapped        {(wait_total-covered_total)/1000:9.3f} ms")
print(f"waits with nothing running at all: {fully_idle} of {len(waits)}")
