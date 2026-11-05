"""Compare production lookup with the radix source at Git 019c97b on CPU.

The frozen reference supplies four 8192+1024 trajectories and independent
reverse-scan signatures. Each actual proposal is checked and saved in numeric
buffers. Round dictionaries and interval tuples are built after index teardown.

Setup includes construction and the opening token. Steady time sums propose
and commit calls with prebuilt tuple inputs; acceptance, validation, teardown
and JSON output are excluded. Fresh starts collect before each trial; carry
starts retain cross-trial GC state. Callback modes diagnose phase overlap and
observer perturbation. No target model, NPU or serving latency is measured.
"""

import argparse
from array import array
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import importlib.util
import inspect
from itertools import permutations
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter_ns
import traceback
from types import ModuleType


LOOKUP = "tilert/models/glm_5_2_ascend/lookup_draft.py"
NGRAM = "tests/glm52/ngram_acceptance.py"
BASELINE_REF = "019c97b"
BASELINE_SHA256 = (
    "4d7c8502aeba3ade42742dbc447230ab8c3765085a048073c80a427101073891"
)
WIDTH, LONGEST, SHORTEST = 32, 32, 2
PROMPT_LENGTH, OUTPUT_LENGTH = 8192, 1024
ARMS = ("baseline", "candidate")
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
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git(root, *arguments):
    return subprocess.check_output(["git", *arguments], cwd=root)


def git_commit(root, revision):
    return git(root, "rev-parse", revision + "^{commit}").decode().strip()


def read_reference(path):
    raw = path.read_bytes()
    data = gzip.decompress(raw) if path.suffix == ".gz" else raw
    return json.loads(data), hashlib.sha256(data).hexdigest()


def load_arms(root, hashes):
    commit = git_commit(root, BASELINE_REF)
    source = git(root, "show", commit + ":" + LOOKUP)
    baseline_hash = hashlib.sha256(source).hexdigest()
    if baseline_hash != BASELINE_SHA256:
        raise ValueError("Git 019c97b lookup source differs from the baseline")
    baseline = ModuleType("git_baseline_lookup")
    exec(compile(source, commit + ":" + LOOKUP, "exec"), baseline.__dict__)
    candidate_path = (root / LOOKUP).resolve()
    candidate = load("production_lookup_candidate", candidate_path)
    loaded_path = Path(inspect.getsourcefile(
        candidate.LookupDrafter)).resolve()
    if (loaded_path != candidate_path
            or digest(loaded_path) != hashes[str(candidate_path)]):
        raise ValueError("Loaded candidate differs from the production source")
    head = git_commit(root, "HEAD")
    head_source = git(root, "show", head + ":" + LOOKUP)
    head_hash = hashlib.sha256(head_source).hexdigest()
    blob = git(root, "rev-parse", commit + ":" + LOOKUP).decode().strip()
    provenance = {
        "baseline_ref": BASELINE_REF, "baseline_commit": commit,
        "baseline_blob": blob,
        "baseline_source_sha256": baseline_hash,
        "candidate_source_path": str(candidate_path),
        "candidate_loaded_path": str(loaded_path),
        "candidate_source_sha256": hashes[str(candidate_path)],
        "current_head": head, "head_lookup_source_sha256": head_hash,
        "candidate_matches_head": head_hash == hashes[str(candidate_path)],
    }
    return {"baseline": baseline.LookupDrafter,
            "candidate": candidate.LookupDrafter}, provenance


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
                    "our_callback_ns": (
                        pending[5] - pending[4] + left - entered
                    ),
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
    if len(prompt) != PROMPT_LENGTH or len(tokens) != OUTPUT_LENGTH:
        raise ValueError(
            "Each workload requires 8192 prompt and 1024 output IDs")
    position = 1
    for row in case["oracle"]["rounds"]:
        if row["position"] != position or len(row["proposal"]) >= WIDTH:
            raise ValueError("Frozen oracle positions or width differ")
        padded = row["proposal"] + [tokens[position - 1]] * (
            WIDTH - 1 - len(row["proposal"]))
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
    proposal_buffer = array("q", [0]) * (count * (WIDTH - 1))
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
        drafter = factory(prompt, longest=LONGEST, shortest=SHORTEST)
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
            proposal = drafter.propose(WIDTH - 1)
            stop = perf_counter_ns()
            observer.phase = 0
            propose_ns = stop - start
            timing_buffer[row_index * 4] = start - origin
            timing_buffer[row_index * 4 + 1] = propose_ns
            expected = case["proposals"][row_index]
            length = len(proposal)
            if length != len(expected):
                raise AssertionError(
                    "Actual proposal length differs from oracle")
            proposal_lengths[row_index] = length
            for token_index in range(length):
                token = proposal[token_index]
                if token != expected[token_index]:
                    raise AssertionError("Actual proposal differs from oracle")
                proposal_buffer[row_index * (WIDTH - 1) + token_index] = token
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
        if (final_report["longest_tail"] != LONGEST
                or final_report["shortest_tail"] != SHORTEST):
            raise AssertionError("Effective lookup parameters differ")
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
        first = index * (WIDTH - 1)
        last = first + proposal_lengths[index]
        rounds.append({
            "position": position,
            "proposal": proposal_buffer[first:last].tolist(),
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
        raise AssertionError(
            "Buffered signature differs from independent scan")
    overlap = {phase: 0 for phase in PHASES}
    for item in collections:
        for phase, _, begin, end in intervals:
            overlap[phase] += max(0, min(end, item["stop_ns"]) - max(
                begin, item["start_ns"]))
    observed_total = sum(item["observed_interval_ns"] for item in collections)
    overlap["driver"] = observed_total - sum(overlap.values())
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
            "p90": values[math.ceil(len(values) * .90) - 1],
            "p95": values[math.ceil(len(values) * .95) - 1],
            "p99": values[math.ceil(len(values) * .99) - 1],
            "max": values[-1], "std": statistics.pstdev(values)}


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
            group["propose_us"].extend(
                r["propose_ns"] / 1e3 for r in row["rounds"])
            group["commit_us"].extend(
                r["commit_ns"] / 1e3 for r in row["rounds"])
            group["gc_overlap_ms"].append(sum(row["gc_overlap_ns"][phase]
                                              for phase in ("setup", "propose",
                                                            "commit")) / 1e6)
            group["collection_count"].append(sum(
                after["collections"] - before["collections"]
                for before, after in zip(row["gc_stats_before"],
                                         row["gc_stats_after"])))
            group["observed_collection_count"].append(
                len(row["gc_collections"]))
            pair_key = row["workload"], row["mode"], row["sample"]
            pairs.setdefault(pair_key, {})[row["arm"]] = {
                field + "_ns": row[field + "_ns"]
                for field in ("setup", "steady", "total")
            }
    output = []
    for workload in CASE_NAMES:
        for mode in modes:
            item = {"workload": workload, "mode": mode, "arms": {},
                    "paired_baseline_over_candidate": {},
                    "paired_wins": {}}
            for arm in ARMS:
                item["arms"][arm] = {name: stats(values) for name, values in
                                     groups[workload, mode, arm].items()}
            matching = [pair for (name, label, _), pair in pairs.items()
                        if name == workload and label == mode]
            for field in ("setup", "steady", "total"):
                differences = [pair["candidate"][field + "_ns"] -
                               pair["baseline"][field + "_ns"]
                               for pair in matching]
                item["paired_wins"][field] = {
                    "wins": sum(value < 0 for value in differences),
                    "ties": sum(value == 0 for value in differences),
                    "losses": sum(value > 0 for value in differences),
                }
                item["paired_baseline_over_candidate"][field] = stats([
                    pair["baseline"][field + "_ns"] /
                    pair["candidate"][field + "_ns"] for pair in matching
                ])
            item["candidate_regressions"] = [
                {"metric": metric, "stat": statistic,
                 "baseline": item["arms"]["baseline"][metric][statistic],
                 "candidate": item["arms"]["candidate"][metric][statistic]}
                for metric in ("setup_ms", "steady_ms", "total_ms",
                               "teardown_ms", "propose_us", "commit_us")
                for statistic in ("p50", "p90", "p95", "p99", "max", "std")
                if item["arms"]["candidate"][metric][statistic] >
                item["arms"]["baseline"][metric][statistic]
            ]
            output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--gc-start", choices=("fresh", "carry"), default="fresh")
    parser.add_argument("--modes", choices=MODES,
                        nargs="+", default=list(MODES))
    parser.add_argument("--check-only", action="store_true",
                        help="Check sources and frozen inputs without timing")
    args = parser.parse_args()
    if (args.samples < 1 or args.warmup < 0
            or len(set(args.modes)) != len(args.modes)):
        parser.error(
            "Positive samples, nonnegative warmup, unique modes required")
    if gc.callbacks:
        raise RuntimeError(
            "Use a fresh Python process without existing GC callbacks")
    root = args.root.resolve()
    candidate_path = (root / LOOKUP).resolve()
    ngram_path = (root / NGRAM).resolve()
    paths = (Path(__file__).resolve(), candidate_path, ngram_path,
             args.reference.resolve())
    hashes = {str(path): digest(path) for path in paths}
    reference, decoded_hash = read_reference(args.reference)
    scorer_hashes = [sha for name, sha in reference["source_sha256"].items()
                     if name.endswith(NGRAM)]
    if (reference["status"] != "pass"
            or scorer_hashes != [hashes[str(ngram_path)]]):
        raise ValueError(
            "Frozen oracle report or current reverse-scan source differs")
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
    factories, provenance = load_arms(root, hashes)
    # Discard the source report's round dictionaries and untrack immutable
    # oracle tuples before either panel. Carry then proceeds without resets.
    initial_plan_collected = gc.collect()
    raw_path = args.output.with_suffix(".samples.jsonl")
    if args.output.exists() or raw_path.exists():
        raise FileExistsError(
            "Choose new output paths for immutable sample artifacts")
    thresholds = gc.get_threshold()
    parameters = {
        "width": WIDTH, "longest": LONGEST, "shortest": SHORTEST,
        "prompt_length": PROMPT_LENGTH, "output_length": OUTPUT_LENGTH,
        "samples": args.samples, "warmup": args.warmup,
        "gc_start": args.gc_start, "modes": args.modes,
        "gc_thresholds": thresholds,
    }
    parameter_hash = hashlib.sha256(json.dumps(
        parameters, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    report = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "source_sha256": hashes,
        "root": str(root), "implementation_sources": provenance,
        "parameters": parameters, "parameters_sha256": parameter_hash,
        "reference_decoded_sha256": decoded_hash,
        "scope": "Synthetic CPU setup+propose+commit; "
                 "no inference or NPU timing",
        "hostname": platform.node(), "python": sys.version,
        "implementation": platform.python_implementation(),
        "machine": platform.machine(), "gc_thresholds": thresholds,
        "gc_start": args.gc_start, "modes": args.modes,
        "samples": args.samples, "warmup": args.warmup,
        "arm_permutations": ORDERS, "raw_samples": str(raw_path),
        "environment_control": "No affinity or clock locking",
        "p99_scope": "With 30 request samples, nearest-rank p99 "
                     "is their maximum",
        "teardown_in_total": False, "callback_changes_execution": True,
        "driver_retains_round_objects_during_measurement": False,
        "commit_inputs_prepared_as_tuples": True,
        "acceptance_replayed_before_trials": True,
        "actual_proposals_checked_and_buffered": True,
        "gc_stats_scope": "Setup, calls, validation, teardown, counter reads",
        "performance_measured": not args.check_only,
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
        if args.check_only:
            report["status"] = "checked"
        else:
            ordinal = measure(args, cases, factories, raw_path)
            report["results"] = summarize(raw_path, args.modes)
            report["status"] = "pass"
            report["all_signatures_match_frozen_reverse_scan"] = True
        if gc.get_threshold() != thresholds:
            raise AssertionError("GC thresholds changed")
    except Exception as error:
        report.update(status="fail", error=repr(error),
                      traceback=traceback.format_exc())
    report["source_stable"] = all(digest(Path(path)) == sha
                                  for path, sha in hashes.items())
    report["current_head_after"] = git_commit(root, "HEAD")
    report["head_stable"] = (
        report["current_head_after"] == provenance["current_head"]
    )
    if not report["source_stable"]:
        report["status"] = "fail"
    if raw_path.exists():
        report["raw_samples_sha256"] = digest(raw_path)
        with raw_path.open() as stream:
            ordinal = sum(1 for _ in stream)
    report["raw_sample_count"] = ordinal
    report["performance_measured"] = ordinal > 0
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "raw_samples": str(raw_path), "source_stable":
                      report["source_stable"]}), flush=True)
    return 0 if report["status"] in ("checked", "pass") else 1


def measure(args, cases, factories, raw_path):
    ordinal = 0
    with raw_path.open("x") as stream:
        for case in cases:
            for sample in range(-args.warmup, args.samples):
                mode_order = args.modes[sample % len(args.modes):] + (
                    args.modes[:sample % len(args.modes)])
                for mode in mode_order:
                    order = ORDERS[(sample + args.modes.index(mode)) %
                                   len(ORDERS)]
                    reports = []
                    for arm_order, arm in enumerate(order):
                        row = replay(factories[arm], case, mode, args.gc_start)
                        reports.append(row["final_report"])
                        row.update(workload=case["name"], arm=arm, mode=mode,
                                   sample=sample, warmup=sample < 0,
                                   arm_order=arm_order, arm_permutation=order,
                                   mode_order=mode_order, ordinal=ordinal)
                        ordinal += 1
                        stream.write(json.dumps(
                            row, separators=(",", ":")) + "\n")
                        stream.flush()
                        del row
                    if any(item != reports[0] for item in reports):
                        raise AssertionError(
                            "Public reports differ between arms")
            print(json.dumps({"workload": case["name"],
                              "phase": "samples_saved"}), flush=True)
    return ordinal


if __name__ == "__main__":
    raise SystemExit(main())
