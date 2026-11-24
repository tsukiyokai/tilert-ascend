"""Per-round wall time from a run's own record, no profiler involved.

TPOT mixes two quantities -- how long a round takes and how many rounds a
thousand tokens need -- and the four-patch episode moved both at once in
opposite directions. The record already carries one `advance_mtp_n` entry per
round with its own elapsed time, so the round can be read directly.
"""
import json
import pathlib
import statistics
import sys


def rounds_of(request):
    return [entry["elapsed_ns"] / 1e6 for entry in request.get("operations", [])
            if entry.get("operation") == "advance_mtp_n"]


for directory in sys.argv[1:]:
    path = pathlib.Path(directory) / "rank-0.json"
    if not path.exists():
        print(f"{pathlib.Path(directory).name}: no record")
        continue
    record = json.loads(path.read_text())
    print(f"=== {pathlib.Path(directory).name}")
    for request in record.get("requests", []):
        if request.get("warmup"):
            continue
        times = rounds_of(request)
        session = request.get("session") or {}
        if not times:
            print("    no rounds recorded")
            continue
        times.sort()
        print(f"    rounds {session.get('completed_rounds')}  "
              f"accepted {session.get('accepted')}  "
              f"tpot {request.get('tpot_ms'):.4f}  "
              f"round median {statistics.median(times):.3f} ms  "
              f"p10 {times[len(times)//10]:.3f}  p90 {times[9*len(times)//10]:.3f}  "
              f"mean {statistics.fmean(times):.3f}")
