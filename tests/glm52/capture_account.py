"""Account a steady-state capture, with the scope checks today's work needed.

Every device capture this project had was taken at `max_tokens: 6` and
`mtp_width: 4` with warmup and prefill inside the window, and a day's
device-time account was derived from one before that was noticed. So this reads
the run's own record first and refuses to divide by a round count it cannot
justify.

Run against an arm's output directory: the record is `rank-0.json` and the
capture is `capture/`.
"""
import argparse
import collections
import csv
import glob
import json
import pathlib
import sys

LAYERS = 78


def rounds_of(record):
    """How many decode rounds the captured request ran.

    From the record, not from kernel counts. A count inferred by dividing
    attention calls by the layer count assumes every call is decode, which is
    false whenever warmup shares the window -- the mistake that voided the
    earlier account.
    """
    for request in record.get("requests", []):
        if request.get("warmup"):
            continue
        advances = [entry for entry in request.get("operations", [])
                    if entry.get("operation") == "advance_mtp_n"]
        if advances:
            return len(advances), request
    return 0, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=pathlib.Path)
    arguments = parser.parse_args()

    record = json.loads((arguments.directory / "rank-0.json").read_text())
    scope = record.get("capture_scope")
    if scope != "last request, decode only":
        print(f"capture_scope is {scope!r}; this account needs a capture taken "
              f"with --profile-last")
        return 1
    rounds, request = rounds_of(record)
    if not rounds:
        print("the record holds no non-warmup request with rounds")
        return 1
    width = record.get("width") or record.get("mtp_width")
    print(f"width {width}, {rounds} decode rounds captured, "
          f"accepted {request.get('accepted_mean')}")

    found = glob.glob(
        f"{arguments.directory}/capture/**/ASCEND_PROFILER_OUTPUT/"
        f"kernel_details.csv", recursive=True)
    if not found:
        print("no kernel_details.csv under capture/")
        return 1
    rows = list(csv.DictReader(open(found[0])))
    name = next(k for k in rows[0] if "Name" in k)
    kind = next(k for k in rows[0] if "Type" in k)
    duration = next(k for k in rows[0] if "Duration" in k)

    # Deduplicate by name: a kernel appearing once across many rounds is setup.
    appearances = collections.Counter(row[name] for row in rows)
    groups = collections.defaultdict(lambda: [0, 0.0])
    for row in rows:
        if appearances[row[name]] <= 1 or not row[duration]:
            continue
        groups[row[kind]][0] += 1
        groups[row[kind]][1] += float(row[duration])
    total = sum(value[1] for value in groups.values())
    print(f"{len(rows)} records, {total / 1000 / rounds:.2f} ms of device time "
          f"a round")
    for key, (calls, spent) in sorted(groups.items(),
                                      key=lambda pair: -pair[1][1])[:10]:
        print(f"  {key:<28} {calls / rounds:>7.1f}/round "
              f"{spent / calls:>8.1f} us  {spent / 1000 / rounds:>6.2f} ms  "
              f"{spent / total * 100:>4.1f}%  "
              f"{calls / rounds / LAYERS:>5.2f}/layer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
