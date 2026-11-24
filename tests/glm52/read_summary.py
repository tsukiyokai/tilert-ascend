"""What has actually been failing in every run's own verdict file.

The rank verdict, including each rank's validation errors, is written to
summary.json rather than into the rank records -- which is why reading the
records found no errors while every arm exited nonzero.
"""
import json
import pathlib
import sys

for arm in sys.argv[1:]:
    path = pathlib.Path(arm) / "summary.json"
    if not path.exists():
        print(f"{pathlib.Path(arm).name}: no summary.json")
        continue
    summary = json.loads(path.read_text())
    status = summary.get("status")
    agreement = summary.get("all_rank_state_agreement")
    print(f"{pathlib.Path(arm).name}: status={status} agreement={agreement}")
    grouped = {}
    for entry in summary.get("ranks", []):
        for error in entry.get("errors", []):
            grouped.setdefault(error, []).append(entry["rank"])
    for error, ranks in grouped.items():
        print(f"    {len(ranks)} ranks: {error[:160]}")
    if not grouped:
        print("    no rank errors")
