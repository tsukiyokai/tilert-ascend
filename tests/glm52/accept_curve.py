"""Acceptance length at every width no wider than the one that was captured.

TPOT is the round divided by the acceptance, and widening buys acceptance at a
fixed price in chain steps. Choosing a width therefore needs the whole curve,
not the single point a run reports.

The curve needs no extra run. The draft chain is autoregressive, so a narrower
width drafts a prefix of what the captured width drafted, and the target
accepts a prefix of that prefix. A round that matched k drafts at the captured
width would have matched min(k, w - 1) at width w:

    acceptance(w) = 1 + mean(min(matched_drafts, w - 1))

The leading 1 is the token the target produces at the anchor position, which
arrives whether or not any draft survives.
"""
import collections
import json
import pathlib
import sys


def spans_of(record):
    for request in record.get("requests", []):
        if request.get("warmup"):
            continue
        yield from request["session"]["accepted_spans"]


def acceptance(matched, width):
    return 1 + sum(min(k, width - 1) for k in matched) / len(matched)


for directory in sys.argv[1:]:
    path = pathlib.Path(directory) / "rank-0.json"
    record = json.loads(path.read_text())
    matched = [span["matched_drafts"] for span in spans_of(record)]
    captured = record["width"]

    print(f"=== {pathlib.Path(directory).name}  width {captured}  "
          f"{len(matched)} rounds")
    histogram = collections.Counter(matched)
    for drafts in range(captured):
        count = histogram[drafts]
        print(f"    matched {drafts}  {count:5d}  {count / len(matched):6.2%}")

    print(f"    {'width':>5} {'acceptance':>11} {'marginal':>9}")
    previous = 1.0
    for width in range(2, captured + 1):
        value = acceptance(matched, width)
        print(f"    {width:5d} {value:11.4f} {value - previous:9.4f}")
        previous = value
