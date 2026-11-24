"""Read the levers run's arms and set each against its registered prediction.

The predictions were written before the arms ran, in
docs/evidence/ascend-decode/predictions-20261021. This prints them beside what
the arms measured, so a lever that came in at half its expected size is visible
rather than explained.
"""
import argparse
import json
import pathlib
import sys

# Written before the arms ran. A lever's expectation is in ms saved a round.
EXPECTED = {
    "baseline": (None, "the reference, AIV on by default"),
    "narrow": (17.41, "the bank's device-side price, two experts of sixteen"),
    "aicpu": (0.0, "AIV's control: within noise, since the collectives are "
                   "inside the graph"),
    "hoist": (0.13, "a floor -- the metadata rebuild's host half only"),
    "hoist-rope": (7.28, "the rope refresh captured, which needs hoisting"),
    "all": (24.8, "the three that act on different terms"),
    "steady": (None, "the capture, not a lever"),
}


def read(directory):
    """One arm's round time, acceptance and phase breakdown."""
    record = json.loads((directory / "rank-0.json").read_text())
    requests = [item for item in record.get("requests", [])
                if not item.get("warmup") and not item.get("profiled")]
    if not requests:
        return None
    rounds = []
    tokens = 0
    for request in requests:
        tokens += request.get("output_tokens") or 0
        for entry in request.get("operations", []):
            if entry["operation"].startswith("advance"):
                rounds.append(entry["elapsed_ns"] / 1e6)
    if not rounds:
        return None
    rounds.sort()
    return {
        "tpot": sum(item["tpot_ms"] for item in requests) / len(requests),
        "round": rounds[len(rounds) // 2],
        "rounds": len(rounds),
        "accepted": tokens / len(rounds),
        "phases": record.get("round_phases"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path,
                        default=pathlib.Path(
                            "/workspace/experiments/levers-20261021"))
    arguments = parser.parse_args()

    held = {}
    for name in EXPECTED:
        directory = arguments.root / name
        if not (directory / "rank-0.json").exists():
            continue
        result = read(directory)
        if result:
            held[name] = result

    if not held:
        print("  no arm has a record yet")
        return 0

    base = held.get("baseline")
    print(f"  {'arm':<12}{'tpot':>8}{'round':>9}{'accepted':>10}"
          f"{'saved':>8}{'expected':>10}")
    for name in EXPECTED:
        if name not in held:
            continue
        item = held[name]
        expected, _ = EXPECTED[name]
        saved = (base["round"] - item["round"]) if base else None
        print(f"  {name:<12}{item['tpot']:>8.3f}{item['round']:>9.2f}"
              f"{item['accepted']:>10.3f}"
              f"{(f'{saved:+.2f}' if saved is not None else ''):>8}"
              f"{(f'{expected:.2f}' if expected is not None else ''):>10}")
    print()
    for name in EXPECTED:
        if name in held and held[name]["phases"]:
            phases = held[name]["phases"]
            steps = {key: value for key, value in phases.items()
                     if key != "rounds"}
            total = sum(steps.values())
            print(f"  {name}'s round, by step, ms a round "
                  f"({phases.get('rounds')} rounds):")
            for step, value in sorted(steps.items(), key=lambda i: -i[1]):
                print(f"    {step:<16}{value:>8.3f}{value / total:>8.0%}")
            print(f"    {'the marks':<16}{total:>8.3f}")
            print(f"    {'the round':<16}{held[name]['round']:>8.3f}")
            print(f"    {'unmarked':<16}"
                  f"{held[name]['round'] - total:>8.3f}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
