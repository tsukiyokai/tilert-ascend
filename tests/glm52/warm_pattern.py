"""Is the first measured request still warming?

Each arm's three measured requests are reported in order. If the first is
systematically the slowest, the arm's mean is contaminated and the minimum or
the last is the better floor estimate.
"""
import glob
import json
import pathlib
import statistics


def p10s(directory):
    path = pathlib.Path(directory) / "rank-0.json"
    if not path.exists():
        return []
    record = json.loads(path.read_text())
    out = []
    for request in record.get("requests", []):
        if request.get("warmup"):
            continue
        rounds = sorted(entry["elapsed_ns"] / 1e6
                        for entry in request.get("operations", [])
                        if entry.get("operation") == "advance_mtp_n")
        if rounds:
            out.append(rounds[len(rounds) // 10])
    return out


arms = sorted(glob.glob("/workspace/experiments/*-20261019/*/rank-0.json"))
positions = {0: [], 1: [], 2: []}
print(f"{'arm':44} {'r1':>8} {'r2':>8} {'r3':>8}  {'spread':>7}")
for path in arms:
    directory = str(pathlib.Path(path).parent)
    values = p10s(directory)
    if len(values) != 3:
        continue
    name = "/".join(directory.split("/")[-2:])
    print(f"{name:44} " + " ".join(f"{v:8.3f}" for v in values)
          + f"  {max(values)-min(values):7.3f}")
    for index, value in enumerate(values):
        positions[index].append(value)

if positions[0]:
    print(f"\n{'':44} " + " ".join(
        f"{statistics.fmean(positions[i]):8.3f}" for i in (0, 1, 2))
        + "   <- mean by position")
    print(f"first request is slowest in "
          f"{sum(1 for i in range(len(positions[0])) if positions[0][i] == max(positions[0][i], positions[1][i], positions[2][i]))}"
          f" of {len(positions[0])} arms")
