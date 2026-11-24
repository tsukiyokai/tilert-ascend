"""Audit completed TP timing reports and recompute paired maxima from both ranks."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


MANIFEST_SHA = "5f18735036c413ae657d8df9727bdab4969d8638c2f5cc5be8aa96b93c7387ac"
SOURCES = {
    "wrapper": "8e1c269e331d005f1361acc5cec24402bff7ae39b75e952ea890da7812804955",
    "capture_context": "a6c51c8bf24d676816f0d4450d24520abd6029cd3a1031a2b8d9140eed40407f",
    "torch_graph_manager": "7d76706e096c0249dd63a3000a625b956df3841f62a163efb8dbc1bf76b26c6a",
}
TAIL = {"consumer.output", "consumer.retained", "layer6.reduction.attention",
        "layer6.reduction.output", "layer7.reduction.attention", "layer7.reduction.output",
        "layer6.persistent_topk", "layer6.body.active_list", "layer7.body.active_list"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def close(actual, expected):
    require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9),
            f"Recomputed statistic differs: {actual} != {expected}")


def distribution(values):
    data = sorted(values)
    require(bool(data), "An event series is empty")

    def quantile(q):
        x = (len(data) - 1) * q
        low, high = math.floor(x), math.ceil(x)
        return data[low] * (high - x) + data[high] * (x - low) if low != high else data[low]

    return {"samples": len(data), "p50": quantile(.5), "p90": quantile(.9),
            "p99": quantile(.99), "std": statistics.pstdev(data),
            "min": data[0], "max": data[-1]}


def compare_statistics(recorded, expected):
    require(recorded.keys() == expected.keys(), "Statistic fields changed")
    for key, value in expected.items():
        close(recorded[key], value)


def bitwise(checks, expected_names, counts, category):
    require(set(checks) == set(expected_names), f"{category}: boundary names differ")
    for name, value in checks.items():
        require(value["pass"] is True and value["bitwise_equal"] is True
                and value["bitwise_different_bytes"] == 0,
                f"{category}: {name} failed raw-byte equality")
        require(value.get("different_elements", 0) == 0 and value.get("max_abs", 0) == 0,
                f"{category}: {name} has a numerical difference")
    counts[category] += len(checks)


def witnesses(records):
    require([row["rank"] for row in records] == [0, 1], "Rank witness order differs")
    equal = ("input_hidden", "input_residual", "layer6.retained", "layer7.retained",
             "layer6.ar1", "layer6.ar2", "layer7.ar1", "layer7.ar2")
    distinct = ("layer6.attention_partial", "layer6.ffn_partial",
                "layer7.attention_partial", "layer7.ffn_partial")
    for name in equal:
        require(records[0][name] == records[1][name], f"Replicated/reduced witness differs: {name}")
    for name in distinct:
        require(records[0][name] != records[1][name], f"Rank partials are identical: {name}")


def literals(records, counts):
    require([(value["layer"], value["phase"]) for value in records]
            == [(n, p) for n in (6, 7) for p in ("ar1", "ar2")], "Missing literal reduction cut")
    for item in records:
        bitwise(item["checks"], {"ordered_fp32_sum", "converted", "reduced", "narrowed"},
                counts, "literal_fp32")
        require(item["partials"][0]["sha256"] != item["partials"][1]["sha256"],
                "Literal reduction has indistinguishable rank partials")
        for value in (*item["partials"], item["reduced_fp32"], item["output"]):
            require(value["finite"] and value["shape"] == [12, 6144], "Invalid reduction witness")
        require(set(item["operators"]) == {"literal_up", "literal_down", "prepared_up", "prepared_down"}
                and all(item["operators"].values()), "Missing actual conversion operator record")


def audit_arena(report, rank, arena, boundary_names, counts):
    require(report["pass"] and report["rank"] == rank and report["arena"] == arena,
            "An arena did not complete")
    require((report["rows"], report["blocks"], report["graph_pair_calls"],
             report["graph_replays_per_event"], report["samples_per_arm"]) == (12, 24, 16, 30, 30),
            "The declared timing geometry differs")
    require(report["environment"]["HCCL_OP_EXPANSION_MODE"] == "AIV"
            and report["environment"]["HCCL_DETERMINISTIC"] == "true", "Collective environment changed")
    require(report["collective_temporary_bytes"] == 4 * 12 * 6144 * 4
            and len(set(report["collective_temporary_addresses"].values())) == 4,
            "FP32 temporary geometry/owners differ")
    owners = report["production_owners"]
    require(set(owners) == set(boundary_names), "Missing production owner")
    require(owners["consumer.output"] == owners["layer7.reduction.output"]
            and owners["consumer.retained"] == owners["layer7.bridge.retained"]
            and owners["layer6.persistent_topk"] == owners["layer7.persistent_topk"],
            "Observation aliases no longer name the production owners")
    require(not (set(report["collective_temporary_addresses"].values())
                 & {value[0] for value in owners.values()}), "A FP32 temporary aliases a production pointer")
    for view in report["rstd_views"]:
        require(view["shape"] == [12, 1] and view["address"] == view["original_address"],
                "Native norm replaced its rstd owner")
    require(report["window_epoch_reset"] is False and report["same_production_addresses"] is True
            and report["observation_copies_inside_graph"] == 0
            and report["production_topk_copies_per_pair"] == 1, "Timing production contract differs")
    captures = {(item["pair_calls"], item["arm"]): item for item in report["captures"]}
    require(len(captures) == len(report["captures"]) == 4, "Expected four distinct graph captures")
    for key in ((n, a) for n in (1, 16) for a in ("control", "candidate")):
        item = captures[key]
        require(item["phase"] == "captured_not_executed" and item["runnable_calls"] == 1
                and item["capture_counter_delta"] == 1 and item["unchanged_capture_boundaries"] == 113
                and item["graph_type"] == "UpdatableGraph", "Official capture evidence differs")
        require(all(item[name] == report["window_stream"] for name in
                    ("capture_stream", "window_stream", "caller_stream")), "Capture/window stream differs")
        require(item["output_owner_ptrs"] == [owners["consumer.output"][0], owners["consumer.retained"][0]],
                "The graph output owner differs")
        for name, sha in SOURCES.items():
            require(item[name]["sha256"] == sha, f"The actual {name} source changed")
        counts["capture_unchanged"] += 113
        counts["captures"] += 1
    bitwise(report["native_control_check"], boundary_names, counts, "production")
    literals(report["literal_conversion_checks"], counts)
    require([item["iteration"] for item in report["checks"]] == list(range(8)), "Changing inputs are incomplete")
    for item in report["checks"]:
        literals(item["literal_conversions"], counts)
        witnesses(item["rank_witnesses"])
        require([(row["pair_calls"], row["arm"]) for row in item["checks"]]
                == [(n, a) for n in (1, 16) for a in ("control", "candidate")], "A numerical graph arm is missing")
        for row in item["checks"]:
            bitwise(row["checks"], boundary_names, counts, "production")
    for name in ("input_hidden", "input_residual", "layer6.ar2", "layer7.ar2",
                 "layer6.retained", "layer7.retained"):
        require(len({item["rank_witnesses"][rank][name] for item in report["checks"]}) == 8,
                f"Eight changing {name} witnesses are not distinct")
    literals(report["timed_literal_conversions"], counts)
    witnesses(report["timed_rank_witnesses"])
    require([(item["arm"], item["replay"]) for item in report["idempotence_checks"]]
            == [(a, n) for a in ("control", "candidate") for n in range(3)], "Idempotence gate is incomplete")
    for item in report["idempotence_checks"]:
        require(item["cache_reset_between_replays"] is False and item["pair_calls"] == 16,
                "Idempotence was tested with a reset or different call count")
        bitwise(item["checks"], boundary_names, counts, "production")
    require([item["arm"] for item in report["postchecks"]] == ["control", "candidate"], "Missing full postcheck")
    for item in report["postchecks"]:
        bitwise(item["checks"], boundary_names, counts, "production")
    for item in report["graph_execution_counts"]:
        calls, arm = item["pair_calls"], item["arm"]
        require(calls in (1, 16) and arm in ("candidate", "control")
                and item["captures"] == 1 and item["python_replay_calls"] == 0
                and item["replays"] == (1002 if calls == 16 else 998), "Unexpected graph invocation count")
        require(item["expected_candidate_epoch_increments_from_calls"]
                == (4 * calls * item["replays"] if arm == "candidate" else 0), "Derived epoch count is inconsistent")
    return captures


def paired_series(left, right, field, calls, arena, counts):
    require(len(left[field]) == len(right[field]) == 30, "A paired series has fewer than 30 samples")
    result = []
    for index, (a, b) in enumerate(zip(left[field], right[field])):
        order = ["candidate", "control"] if index % 2 == 0 else ["control", "candidate"]
        require(a["sample"] == b["sample"] == index and a["order"] == b["order"] == order
                and a["pair_calls"] == b["pair_calls"] == calls, "Rank pairing or AB/BA order differs")
        row = {"sample": index, "per_pair_us": {}, "raw_rank_intervals_us": {}}
        for arm in order:
            first, second = a["arms"][arm], b["arms"][arm]
            require(first["rank_intervals"] == second["rank_intervals"], "The ranks retained different event intervals")
            raw = first["rank_intervals"]
            key = [arena, calls, index, arm]
            require([entry["rank"] for entry in raw] == [0, 1]
                    and all(entry["key"] == key for entry in raw), "An interval belongs to another rank/sample/arm")
            durations = [entry["duration_us"] for entry in raw]
            require(all(math.isfinite(x) and x > 0 for x in durations), "Invalid raw event interval")
            maximum = max(durations)
            per_pair = maximum / (calls * 30)
            for recorded in (first, second):
                close(recorded["maximum_event_us"], maximum)
                close(recorded["per_pair_us"], per_pair)
                bitwise(recorded["tail_checks"], TAIL, counts, "event_tail")
            row["per_pair_us"][arm] = per_pair
            row["raw_rank_intervals_us"][arm] = durations
        saving = row["per_pair_us"]["control"] - row["per_pair_us"]["candidate"]
        close(a["paired_saving_us"], saving)
        close(b["paired_saving_us"], saving)
        row["paired_saving_us"] = saving
        result.append(row)
    summary = {arm: distribution([row["per_pair_us"][arm] for row in result])
               for arm in ("control", "candidate")}
    summary["paired_saving_us"] = distribution([row["paired_saving_us"] for row in result])
    summary["candidate_wins"] = sum(row["paired_saving_us"] > 0 for row in result)
    stored = "summary_us" if calls == 16 else "one_pair_summary_us"
    for report in (left, right):
        for arm in ("control", "candidate"):
            compare_statistics(report[stored][arm], summary[arm])
        if calls == 16:
            compare_statistics(report["paired_saving_us"], summary["paired_saving_us"])
            require(report["candidate_wins"] == summary["candidate_wins"], "Recorded paired win count differs")
    return {"pair_calls": calls, "unit": "us per two-layer pair, maximum of local rank intervals",
            "summary": summary, "samples": result}


def audit(directory, staging, boundary_path):
    require(digest(staging / "manifest.json") == MANIFEST_SHA, "The frozen timing manifest changed")
    frozen = dict(line.split("  ", 1)[::-1] for line in (staging / "frozen-inputs.sha256").read_text().splitlines())
    for name, sha in frozen.items():
        require(digest(staging / name) == sha, f"A frozen input changed: {name}")
    names = read(boundary_path)["boundaries"]
    require(len(set(names)) == len(names) == 113, "The independent boundary-name oracle changed")
    controller = read(directory / "summary.json")
    require(controller["success"] and controller["exitcodes"] == [0] * 16, "The controller did not finish all 16 ranks")
    require(controller["script_sha256"] == frozen["tp_timing_device.py"], "The device driver source differs")
    ranks = [read(directory / f"rank-{rank:02d}.json") for rank in range(16)]
    for rank, report in enumerate(ranks):
        require(report["rank"] == rank and report["success"] and "error" not in report
                and report["npu_initialized_before"] is False
                and report["npu_initialized_after"] == (rank < 2)
                and all(stage["success"] for stage in report["stages"]), "A rank failed or initialized an unauthorized NPU")
        env = report["collective_environment_before_torch_import"]
        require(env["HCCL_OP_EXPANSION_MODE"] == "AIV" and env["HCCL_DETERMINISTIC"] == "true",
                "A rank imported Torch before the required environment")
    counts = Counter()
    all_reports = {}
    source_files = [directory / "summary.json"]
    for rank in (0, 1):
        folder = directory / f"device-rank{rank}"
        outer = read(folder / "tp-chain-timing.json")
        require(outer["pass"] and outer["rank"] == rank and outer["window"]["bytes"] == 1573184
                and outer["tp16_collectives_executed"] == 0, "The TP timing chain did not complete")
        require(not (set(outer["private_arena_allocations"][0]) & set(outer["private_arena_allocations"][1])),
                "The retained arenas share private allocations")
        alias = read(folder / "nz-storage-correction.json")
        require(alias["completed"] and len(alias["banks"]) == 4 and alias["additional_hbm_bytes"] == 0,
                "The actual weight alias gate is incomplete")
        for bank in alias["banks"]:
            require(bank["raw_bytes_equal"] and bank["raw_sha256_before"] == bank["raw_sha256_after"]
                    and bank["before"]["data_ptr"] == bank["after"]["data_ptr"]
                    and bank["new_device_allocation_bytes"] == 0, "A true model weight bank changed")
        require([item["arena"] for item in outer["arenas"]] == [0, 1], "Both retained arenas did not finish")
        for arena in (0, 1):
            path = folder / f"tp-pair-timing-arena{arena}.json"
            require(digest(path) == outer["arenas"][arena]["report_sha256"], "The chain references different timing evidence")
            report = read(path)
            audit_arena(report, rank, arena, names, counts)
            require(report["script_sha256"] == frozen["tp_timing_benchmark.py"]
                    and report["window_address"] == outer["window"]["address"]
                    and report["window_stream"] == outer["window"]["stream"], "An arena changed source or window")
            all_reports[rank, arena] = report
            source_files.append(path)
        source_files.extend((folder / "tp-chain-timing.json", folder / "nz-storage-correction.json"))
    arenas = []
    for arena in (0, 1):
        left, right = all_reports[0, arena], all_reports[1, arena]
        for a, b in zip(left["checks"], right["checks"]):
            require(a["rank_witnesses"] == b["rank_witnesses"], "Rank witness gathers differ")
        arenas.append({"arena": arena,
                       "primary_16_pairs": paired_series(left, right, "samples", 16, arena, counts),
                       "one_pair_sensitivity": paired_series(left, right, "one_pair_sensitivity", 1, arena, counts)})
    require(dict(counts) == {"capture_unchanged": 1808, "captures": 16,
                            "production": 18532, "literal_fp32": 640, "event_tail": 4320},
            f"The complete evidence counts differ: {counts}")
    source_files.extend(directory / f"rank-{rank:02d}.json" for rank in range(16))
    return {"pass": True, "scope": "Independent report-field audit and raw event-interval recomputation; numerical checks are recorded device metrics",
            "manifest_sha256": MANIFEST_SHA, "report_sha256": {str(path): digest(path) for path in source_files},
            "counts": dict(counts), "arenas": arenas,
            "epoch_scope": "Expected increments from graph invocations, not a measured device counter",
            "auditor_sha256": digest(Path(__file__)), "boundary_oracle_sha256": digest(boundary_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--staging", type=Path, default=Path("/tmp/tilert-tp-timing-staging-v3"))
    parser.add_argument("--boundaries", type=Path, default=Path(__file__).with_name("audit-boundaries.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory, args.staging, args.boundaries)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pass": result["pass"], "counts": result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
