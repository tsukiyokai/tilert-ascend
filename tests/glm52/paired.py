"""Compare two arms pass by pass, so run-to-run drift differences out.

Two identical configurations in successive chains differed by 0.201 ms in a
round's tenth percentile. Pairing an A with the B that ran beside it removes
that; comparing pooled means does not.
"""
import json
import pathlib
import statistics
import sys


def p10(directory):
    """An arm's floor: the lowest tenth percentile of its measured requests.

    The minimum rather than the mean. Across 28 arms the spread within an arm
    is under 0.6 ms in twenty of them and between 1.9 and 2.9 in a few, and
    the position of the slow request is not systematic -- the means by
    position are 54.766, 54.607, 54.625 and the first request is slowest in 15
    of 28. So the spread is occasional interference rather than warming, and a
    floor is best estimated by the fastest observation. On the minimum, two
    identical configurations differ by 0.163 ms, the metadata patch comes out
    at 1.04x that and the three kernel patches at 9.3x; on the mean the first
    read as 1.39x and had to be withdrawn.
    """
    path = pathlib.Path(directory) / "rank-0.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    values = []
    for request in record.get("requests", []):
        if request.get("warmup") or request.get("profiled"):
            continue
        rounds = sorted(entry["elapsed_ns"] / 1e6
                        for entry in request.get("operations", [])
                        if entry.get("operation") == "advance_mtp_n")
        if rounds:
            values.append(rounds[len(rounds) // 10])
    return min(values) if values else None


root = pathlib.Path(sys.argv[1])
name_a, name_b = sys.argv[2], sys.argv[3]
deltas = []
for pass_number in (1, 2, 3):
    left = p10(root / f"{name_a}-{pass_number}")
    right = p10(root / f"{name_b}-{pass_number}")
    if left is None or right is None:
        print(f"pass {pass_number}: missing")
        continue
    deltas.append(right - left)
    print(f"pass {pass_number}: {name_a} {left:7.3f}  {name_b} {right:7.3f}  "
          f"delta {right - left:+7.3f} ms")

if len(deltas) == 3:
    print(f"\npaired mean delta {statistics.fmean(deltas):+.3f} ms a round "
          f"({statistics.fmean(deltas) * 227 / 1023:+.4f} ms a token)")
    print(f"signs {''.join('+' if d > 0 else '-' for d in deltas)}  "
          f"spread {max(deltas) - min(deltas):.3f}")
