"""No-retention three-arm CPU control; no NPU execution.

Unlike the sibling retention driver, the measurement window creates no round
dicts, interval tuples, or retained proposal list chain. Oracle proposals and
accepted commit chunks are frozen tuples prepared before trials. Each actual
proposal is checked against that oracle and copied into numeric buffers; raw
rounds and the actual-output signature are rebuilt after the window.

This also moves commit-input slicing and acceptance replay outside timing.
Its difference from the sibling driver is not an index-only speedup. Fresh
starts control GC state; carry starts retain cross-trial state. The observer
changes execution: gc_on_plain is its timing control. No work runs on import.
"""

import argparse
from array import array
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
from itertools import permutations
import json
import math
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns
import traceback


ROOT = Path("/Users/shanshan/repo/tile-ai/TileRT-Ascend")
DIRECTORY = Path(__file__).resolve().parent
PRODUCTION = ROOT / "tilert/models/glm_5_2_ascend/lookup_draft.py"
COORDINATE = DIRECTORY / "tilert-lookup-arena-prototype-20261028.py"
TUPLE = DIRECTORY / "tilert-lookup-arena-tuple-prototype-20261028.py"
NGRAM = ROOT / "tests/glm52/ngram_acceptance.py"
ARMS = ("production", "tuple_arena", "coordinate_arena")
ORDERS = tuple(permutations(ARMS))
MODES = ("gc_on_plain", "gc_on_observed", "gc_off_observed")
PHASES = ("driver", "setup", "propose", "commit", "teardown")
CASE_NAMES = ("exact_copy", "copy_5pct_edits", "ambiguous_bigram",
              "random_no_match")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GCPhases:
    """Preallocate event storage; callbacks allocate no result containers."""

    def __init__(self, maximum=4096):
        # kind, generation, phase, position, entered_ns, left_ns,
        # collected, uncollectable. Python timer integers remain untracked.
        self.buffer = array("q", [0]) * (maximum * 8)
        self.maximum = maximum
        self.count = 0
        self.overflow = 0
        self.phase = 0
        self.position = -1

    def callback(self, action, info):
        entered = perf_counter_ns()
        if self.count == self.maximum:
            self.overflow += 1
            return
        offset = self.count * 8
        values = self.buffer
        values[offset] = 0 if action == "start" else 1
        values[offset + 1] = info["generation"]
        values[offset + 2] = self.phase
        values[offset + 3] = self.position
        values[offset + 4] = entered
        values[offset + 6] = info.get("collected", -1)
        values[offset + 7] = info.get("uncollectable", -1)
        self.count += 1
        values[offset + 5] = perf_counter_ns()

    def finish(self, origin):
        if self.overflow:
            raise RuntimeError("GC callback buffer overflow")
        events, collections, pending = [], [], None
        for index in range(self.count):
            row = self.buffer[index * 8:index * 8 + 8].tolist()
            kind, generation, phase, position, entered, left, count, lost = row
            events.append({
                "event": "start" if kind == 0 else "stop",
                "generation": generation, "phase": PHASES[phase],
                "position": position, "entered_ns": entered - origin,
                "left_ns": left - origin, "collected": count,
                "uncollectable": lost,
            })
            if kind == 0:
                if pending is not None:
                    raise RuntimeError("Nested GC start callbacks")
                pending = row
            else:
                if pending is None or pending[1] != generation:
                    raise RuntimeError("Unmatched GC stop callback")
                collections.append({
                    "generation": generation, "phase": PHASES[pending[2]],
                    "stop_phase": PHASES[phase], "position": pending[3],
                    "start_ns": pending[4] - origin,
                    "stop_ns": entered - origin,
                    "observed_interval_ns": entered - pending[4],
                    "between_our_callbacks_ns": entered - pending[5],
                    "our_callback_ns": (pending[5] - pending[4]) + left - entered,
                    "collected": count, "uncollectable": lost,
                })
                pending = None
        if pending is not None:
            raise RuntimeError("Missing GC stop callback")
        return events, collections


def matched_prefix(proposal, actual):
    matched = 0
    for left, right in zip(proposal, actual):
        if left != right:
            break
        matched += 1
    return matched


def signature(rounds):
    values = [(row["position"], row["proposal"], row["accepted"])
              for row in rounds]
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def validate_case(case):
    prompt, tokens = case["prompt_token_ids"], case["output_token_ids"]
    if len(prompt) != 8192 or len(tokens) != 1024:
        raise ValueError("Each workload requires 8192 prompt and 1024 output IDs")
    position = 1
    for row in case["oracle"]["rounds"]:
        if row["position"] != position or len(row["proposal"]) > 31:
            raise ValueError("Frozen oracle positions or width differ")
        padded = row["proposal"] + [tokens[position - 1]] * (
            31 - len(row["proposal"]))
        accepted = min(matched_prefix(padded, tokens[position:]) + 1,
                       len(tokens) - position)
        if row["accepted"] != accepted:
            raise ValueError("Frozen oracle acceptance differs")
        position += accepted
    if position != len(tokens) or signature(case["oracle"]["rounds"]) != (
        case["oracle"]["signature_sha256"]
    ):
        raise ValueError("Frozen oracle signature or final position differs")
    rows = case["oracle"]["rounds"]
    case["positions"] = tuple(row["position"] for row in rows)
    case["accepted"] = tuple(row["accepted"] for row in rows)
    case["proposals"] = tuple(tuple(row["proposal"]) for row in rows)
    case["commits"] = tuple(tuple(tokens[row["position"]:
                                         row["position"] + row["accepted"]])
                            for row in rows)
    case["opening"] = tuple(tokens[:1])
    case["oracle_signature_sha256"] = case["oracle"]["signature_sha256"]
    del case["oracle"]


def replay(factory, case, mode, gc_start):
    prompt, tokens = case["prompt_token_ids"], case["output_token_ids"]
    observer = GCPhases()
    callback = observer.callback
    count = len(case["positions"])
    timing_buffer = array("q", [0]) * (count * 4)
    proposal_buffer = array("q", [0]) * (count * 31)
    proposal_lengths = array("q", [0]) * count
    initial_enabled = gc.isenabled()
    gc.enable()
    pre_count = gc.get_count()
    collect_start = perf_counter_ns()
    pre_collected = gc.collect() if gc_start == "fresh" else None
    pre_collect_ns = perf_counter_ns() - collect_start
    initial_count, initial_stats = gc.get_count(), gc.get_stats()
    observed = mode != "gc_on_plain"
    if observed:
        gc.callbacks.append(callback)
    if mode == "gc_off_observed":
        gc.disable()
    origin = perf_counter_ns()
    drafter = None
    try:
        observer.phase = 1
        start = perf_counter_ns()
        drafter = factory(prompt)
        drafter.commit(case["opening"])
        stop = perf_counter_ns()
        observer.phase = 0
        setup_ns = stop - start
        setup_start, setup_stop = start - origin, stop - origin
        for row_index in range(count):
            position = case["positions"][row_index]
            observer.position = position
            observer.phase = 2
            start = perf_counter_ns()
            proposal = drafter.propose(31)
            stop = perf_counter_ns()
            observer.phase = 0
            propose_ns = stop - start
            timing_buffer[row_index * 4] = start - origin
            timing_buffer[row_index * 4 + 1] = propose_ns
            expected = case["proposals"][row_index]
            length = len(proposal)
            if length != len(expected):
                raise AssertionError("Actual proposal length differs from oracle")
            proposal_lengths[row_index] = length
            for token_index in range(length):
                token = proposal[token_index]
                if token != expected[token_index]:
                    raise AssertionError("Actual proposal differs from oracle")
                proposal_buffer[row_index * 31 + token_index] = token
            proposal = None
            observer.phase = 3
            start = perf_counter_ns()
            drafter.commit(case["commits"][row_index])
            stop = perf_counter_ns()
            observer.phase = 0
            commit_ns = stop - start
            timing_buffer[row_index * 4 + 2] = start - origin
            timing_buffer[row_index * 4 + 3] = commit_ns
        expected_context = prompt + tokens
        if drafter.context != expected_context:
            raise AssertionError("Final committed context differs")
        final_report = drafter.report()
        observer.phase = 4
        observer.position = -1
        start = perf_counter_ns()
        drafter = None
        stop = perf_counter_ns()
        observer.phase = 0
        teardown_ns = stop - start
        teardown_start, teardown_stop = start - origin, stop - origin
        final_count, final_stats = gc.get_count(), gc.get_stats()
        wall_ns = perf_counter_ns() - origin
    finally:
        observer.phase = 0
        if observed:
            gc.callbacks.remove(callback)
        drafter = None
        if initial_enabled:
            gc.enable()
        else:
            gc.disable()
    events, collections = observer.finish(origin)
    if mode == "gc_off_observed" and collections:
        raise AssertionError("Automatic collections occurred with GC disabled")
    # Every result container below is created after the sampler was detached
    # and after the index teardown. The actual proposals came from the buffer.
    rounds = []
    intervals = [("setup", -1, setup_start, setup_stop)]
    for index in range(count):
        propose_start, propose_ns, commit_start, commit_ns = (
            timing_buffer[index * 4:index * 4 + 4])
        position = case["positions"][index]
        rounds.append({
            "position": position,
            "proposal": proposal_buffer[index * 31:
                                         index * 31 + proposal_lengths[index]].tolist(),
            "accepted": case["accepted"][index],
            "propose_ns": propose_ns, "commit_ns": commit_ns,
        })
        intervals.append(("propose", position, propose_start,
                          propose_start + propose_ns))
        intervals.append(("commit", position, commit_start,
                          commit_start + commit_ns))
    intervals.append(("teardown", -1, teardown_start, teardown_stop))
    actual_signature = signature(rounds)
    if actual_signature != case["oracle_signature_sha256"]:
        raise AssertionError("Buffered signature differs from independent scan")
    overlap = {phase: 0 for phase in PHASES}
    for item in collections:
        for phase, _, begin, end in intervals:
            overlap[phase] += max(0, min(end, item["stop_ns"]) - max(
                begin, item["start_ns"]))
    overlap["driver"] = sum(item["observed_interval_ns"] for item in collections) - (
        sum(overlap.values()))
    steady_ns = sum(row["propose_ns"] + row["commit_ns"] for row in rounds)
    return {
        "setup_ns": setup_ns, "steady_ns": steady_ns,
        "total_ns": setup_ns + steady_ns, "teardown_ns": teardown_ns,
        "observed_wall_ns": wall_ns, "signature_sha256": actual_signature,
        "final_report": final_report, "rounds": rounds,
        "phase_intervals": intervals, "gc_events": events,
        "gc_collections": collections, "gc_overlap_ns": overlap,
        "gc_count_before_precollect": pre_count,
        "pre_collect_ns": pre_collect_ns, "pre_collected": pre_collected,
        "gc_count_before": initial_count, "gc_count_after": final_count,
        "gc_stats_before": initial_stats, "gc_stats_after": final_stats,
    }


def stats(values):
    values = sorted(values)
    return {"count": len(values), "p50": statistics.median(values),
            "p95": values[math.ceil(len(values) * .95) - 1],
            "p99": values[math.ceil(len(values) * .99) - 1],
            "max": values[-1]}


def summarize(raw_path, modes):
    groups, pairs = {}, {}
    with raw_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["warmup"]:
                continue
            key = row["workload"], row["mode"], row["arm"]
            group = groups.setdefault(key, {
                "setup_ms": [], "steady_ms": [], "total_ms": [],
                "teardown_ms": [], "propose_us": [], "commit_us": [],
                "gc_overlap_ms": [], "collection_count": [],
                "observed_collection_count": [],
            })
            for name in ("setup", "steady", "total", "teardown"):
                group[name + "_ms"].append(row[name + "_ns"] / 1e6)
            group["propose_us"].extend(r["propose_ns"] / 1e3 for r in row["rounds"])
            group["commit_us"].extend(r["commit_ns"] / 1e3 for r in row["rounds"])
            group["gc_overlap_ms"].append(sum(row["gc_overlap_ns"][phase]
                                            for phase in ("setup", "propose",
                                                          "commit")) / 1e6)
            group["collection_count"].append(sum(
                after["collections"] - before["collections"]
                for before, after in zip(row["gc_stats_before"],
                                         row["gc_stats_after"])))
            group["observed_collection_count"].append(len(row["gc_collections"]))
            pairs.setdefault((row["workload"], row["mode"], row["sample"]), {})[
                row["arm"]] = {field + "_ns": row[field + "_ns"]
                               for field in ("setup", "steady", "total")}
    output = []
    for workload in CASE_NAMES:
        for mode in modes:
            item = {"workload": workload, "mode": mode, "arms": {},
                    "paired_production_over_arm": {}}
            for arm in ARMS:
                item["arms"][arm] = {name: stats(values) for name, values in
                                      groups[workload, mode, arm].items()}
                if arm != "production":
                    item["paired_production_over_arm"][arm] = {
                        field: stats([pair["production"][field + "_ns"] /
                                      pair[arm][field + "_ns"]
                                      for (name, label, _), pair in pairs.items()
                                      if name == workload and label == mode])
                        for field in ("setup", "steady", "total")
                    }
            output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path(
        "/tmp/tilert-lookup-rfind-probe-20261028.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--gc-start", choices=("fresh", "carry"), default="fresh")
    parser.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0 or len(set(args.modes)) != len(args.modes):
        parser.error("Positive samples, nonnegative warmup, unique modes required")
    if gc.callbacks:
        raise RuntimeError("Use a fresh Python process without existing GC callbacks")
    paths = (Path(__file__).resolve(), PRODUCTION, COORDINATE, TUPLE, NGRAM,
             ROOT / "tests/glm52/bench_lookup.py", args.reference.resolve())
    hashes = {str(path): digest(path) for path in paths}
    reference = json.loads(args.reference.read_text())
    if reference["status"] != "pass" or reference["source_sha256"].get(
        str(NGRAM)
    ) != hashes[str(NGRAM)]:
        raise ValueError("Frozen oracle report or current reverse-scan source differs")
    # Retaining the reference's old timing arms would add a large unrelated
    # generation-2 heap to every observation. Keep inputs and oracle only.
    cases = [{name: case[name] for name in (
        "name", "prompt_token_ids", "output_token_ids", "oracle",
    )} for case in reference["results"]]
    del reference
    if [case["name"] for case in cases] != list(CASE_NAMES):
        raise ValueError("Expected the four frozen lookup workloads")
    for case in cases:
        validate_case(case)
    factories = {
        "production": load("production_radix", PRODUCTION).LookupDrafter,
        "tuple_arena": load("tuple_arena", TUPLE).TupleArenaLookupDrafter,
        "coordinate_arena": load("coordinate_arena", COORDINATE).ArenaLookupDrafter,
    }
    # Discard the source report's round dictionaries and untrack immutable
    # oracle tuples before either panel. Carry then proceeds without resets.
    initial_plan_collected = gc.collect()
    raw_path = args.output.with_suffix(".samples.jsonl")
    if args.output.exists() or raw_path.exists():
        raise FileExistsError("Choose new output paths for immutable sample artifacts")
    thresholds = gc.get_threshold()
    report = {
        "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "source_sha256": hashes,
        "scope": "Synthetic CPU setup+propose+commit; no inference or NPU timing",
        "hostname": platform.node(), "python": sys.version,
        "implementation": platform.python_implementation(),
        "machine": platform.machine(), "gc_thresholds": thresholds,
        "gc_start": args.gc_start, "modes": args.modes,
        "samples": args.samples, "warmup": args.warmup,
        "arm_permutations": ORDERS, "raw_samples": str(raw_path),
        "environment_control": "No affinity or clock locking",
        "p99_scope": "With 30 request samples, nearest-rank p99 is their maximum",
        "teardown_in_total": False, "callback_changes_execution": True,
        "driver_retains_round_objects_during_measurement": False,
        "commit_inputs_prepared_as_tuples": True,
        "acceptance_replayed_before_trials": True,
        "actual_proposals_checked_and_buffered": True,
        "initial_plan_collected": initial_plan_collected,
        "workloads": [{"name": case["name"], "prompt_token_ids":
                       case["prompt_token_ids"], "output_token_ids":
                       case["output_token_ids"], "oracle_signature_sha256":
                       case["oracle_signature_sha256"]} for case in cases],
        "npu_executed": False, "results": [],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    ordinal = 0
    try:
        with raw_path.open("x") as stream:
            for case in cases:
                for sample in range(-args.warmup, args.samples):
                    mode_order = args.modes[sample % len(args.modes):] + (
                        args.modes[:sample % len(args.modes)])
                    for mode in mode_order:
                        order = ORDERS[(sample + args.modes.index(mode)) % len(ORDERS)]
                        reports = []
                        for arm_order, arm in enumerate(order):
                            row = replay(factories[arm], case, mode, args.gc_start)
                            reports.append(row["final_report"])
                            row.update(workload=case["name"], arm=arm, mode=mode,
                                       sample=sample, warmup=sample < 0,
                                       arm_order=arm_order, arm_permutation=order,
                                       mode_order=mode_order, ordinal=ordinal)
                            ordinal += 1
                            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                            stream.flush()
                            del row
                        if any(item != reports[0] for item in reports):
                            raise AssertionError("Public reports differ between arms")
                print(json.dumps({"workload": case["name"],
                                  "phase": "samples_saved"}), flush=True)
        if gc.get_threshold() != thresholds:
            raise AssertionError("GC thresholds changed")
        report["results"] = summarize(raw_path, args.modes)
        report["status"] = "pass"
        report["all_signatures_match_frozen_reverse_scan"] = True
    except Exception as error:
        report.update(status="fail", error=repr(error), traceback=traceback.format_exc())
    report["source_stable"] = all(digest(Path(path)) == sha
                                  for path, sha in hashes.items())
    if not report["source_stable"]:
        report["status"] = "fail"
    if raw_path.exists():
        report["raw_samples_sha256"] = digest(raw_path)
    report["raw_sample_count"] = ordinal
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "raw_samples": str(raw_path), "source_stable":
                      report["source_stable"]}), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
