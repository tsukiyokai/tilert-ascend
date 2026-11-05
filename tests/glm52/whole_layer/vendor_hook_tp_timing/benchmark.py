"""Time two serial TP-layer graphs with same-owner native FP32 HCCL parents."""

import hashlib
import json
from pathlib import Path
import statistics

from local_chain import publish_topk, signature, snapshot, tensor_witness
from local_chain_views import poison, strict_checks
from tp_graph_runtime import CapturedChain
from tp_timing_native import Reductions, prepare, separate
from tp_timing_environment import check_environment


CALLS = 16
REPLAYS = 30
SAMPLES = 30


def distribution(values):
    ordered = sorted(values)

    def percentile(fraction):
        position = fraction * (len(ordered) - 1)
        left = int(position)
        right = min(left + 1, len(ordered) - 1)
        return ordered[left] + (ordered[right] - ordered[left]) * (position - left)

    return {"samples": len(values), "p50": percentile(.5), "p90": percentile(.9),
            "p99": percentile(.99), "std": statistics.pstdev(values),
            "min": min(values), "max": max(values)}


def paired_maximum(records, key):
    if (len(records) != 2 or [value["rank"] for value in records] != [0, 1]
            or any(value["key"] != key for value in records)):
        raise RuntimeError("The two ranks timed different arms or samples")
    durations = [value["duration_us"] for value in records]
    if any(not 0 < value < float("inf") for value in durations):
        raise RuntimeError("A rank event interval is not finite and positive")
    return max(durations)


def production_values(cases, caches, topk, rows):
    return {**{f"layer{number}.{name}": value
               for number, case in zip((6, 7), cases)
               for name, value in case.values().items()},
            **caches,
            "layer6.persistent_topk": topk[:rows],
            "layer7.persistent_topk": topk[:rows],
            "consumer.output": cases[1].output,
            "consumer.retained": cases[1].local.bridge[1]}


def run_timing(torch, config, layers, inputs, cases, caches, bases, topk,
               window, stream, args, arena):
    from attention_check import metric
    from tp_graph import check_rank_witnesses

    if inputs.rows != 12 or args.forward_widths != [12]:
        raise ValueError("The first paired TP timing gate is M12 only")
    environment = check_environment()
    reductions = Reductions(torch, cases, args.oracle_hccl, args.oracle_cpu)
    rstd = prepare(torch, cases)
    values = production_values(cases, caches, topk, inputs.rows)
    if len(values) != 113:
        raise RuntimeError("The timing graph lost a defined production boundary")
    outputs = {"output": cases[1].output, "retained": cases[1].local.bridge[1]}
    owners = [case.local.owners() for case in cases]
    reduction_owners = [signature(value) for case in cases
                        for value in (case.attention_sum, case.output)]
    cache_owners = {name: signature(value) for name, value in caches.items()}
    topk_owner = signature(topk)
    graphs = {}
    input_bytes = {}
    destination = args.output / f"tp-pair-timing-arena{arena}.json"
    report = {
        "rank": args.device_rank, "arena": arena, "rows": inputs.rows, "blocks": 24,
        "scope": "Two controlled TP16-shaped shards: same-owner native FP32 HCCL2 graph versus C2 SHMEM2 graph",
        "unit": "us per indexed/shared two-layer pair, maximum of two rank intervals",
        "environment": environment,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "control": "Preallocated BF16-to-FP32 copy, HCCL2 FP32 SUM, FP32-to-BF16 copy",
        "literal_control": "Event-external value.float(), HCCL2 SUM, value.bfloat16()",
        "norm_owner_copies": 0, "observation_copies_inside_graph": 0,
        "production_topk_copies_per_pair": 1,
        "same_production_addresses": True, "weight_banks_per_arena": 1,
        "weight_sets": 2, "constructor_cpu_peers": 14,
        "collective_temporary_bytes": sum(value.nbytes for value in reductions.buffers.values()),
        "collective_temporary_addresses": {f"{number}.{phase}": address
            for (number, phase), address in reductions.pointers.items()},
        "rstd_views": [{"shape": list(value.shape), "address": value.data_ptr(),
                        "original_address": case.local.norm[5].data_ptr()}
                       for case, value in zip(cases, rstd)],
        "window_address": window.address, "window_stream": stream.npu_stream,
        "window_epoch_reset": False, "graph_pair_calls": CALLS,
        "graph_replays_per_event": REPLAYS, "samples_per_arm": SAMPLES,
        "warm_replays_before_event": 3, "timing_skew_injected": False,
        "captures": [], "checks": [], "idempotence_checks": [],
        "samples": [], "one_pair_sensitivity": [], "pass": False,
        "production_owners": {name: signature(value) for name, value in values.items()},
        "weight_owners": [{name: [signature(value) for value in sequence]
            for name, sequence in (("main", case.local.main),
                                   ("front", case.local.front),
                                   ("tail", case.local.tail),
                                   ("routed", case.local.ffn.weights),
                                   ("shared", case.local.ffn.shared))} for case in cases],
    }

    def persist():
        destination.write_text(json.dumps(report, indent=2) + "\n")

    def gather(value):
        records = [None, None]
        torch.distributed.all_gather_object(records, value, group=args.oracle_cpu)
        return records

    def reset(iteration, arm):
        for name, cache in caches.items():
            cache.copy_(bases[name].roll(iteration, 0))
        topk.fill_(-1)
        for case in cases:
            poison(case.values(), 1 if arm == "control" else 2)
            case.local.norm[5].fill_(float("nan"))
            case.local.body[20].fill_(1)
            case.local.body[20][0].zero_()
        for temporary in reductions.buffers.values():
            temporary.fill_(float("nan"))

    def pair(arm, literal_records=None):
        for number, case, stats in zip((6, 7), cases, rstd):
            if arm == "candidate":
                case.combined()
            else:
                def reduce(partial, result, phase):
                    if literal_records is not None:
                        reductions.literal(number, phase, partial, result, literal_records)
                    else:
                        reductions.sum(number, phase, partial, result)

                if arm == "original":
                    case.separate(reduce)
                else:
                    separate(case, stats, reduce)
            if number == 6:
                publish_topk(case.local.selected, topk, inputs.rows)

    def collect():
        torch.npu.synchronize()
        return {name: snapshot(torch, value) for name, value in values.items()}

    def compare(actual, expected, label):
        checks, passed = strict_checks(torch, actual, expected, metric)
        if not all(gather(passed)):
            torch.save({"actual": actual, "expected": expected},
                       args.output / f"tp-timing-{arena}-{label}.pt")
            report["failure"] = {"label": label, "checks": checks}
            persist()
            raise RuntimeError("A paired graph differs from the original FP32 oracle")
        return checks

    def guards():
        inputs.check_owners()
        reductions.check_owners()
        if (owners != [case.local.owners() for case in cases]
                or reduction_owners != [signature(value) for case in cases
                    for value in (case.attention_sum, case.output)]
                or cache_owners != {name: signature(value) for name, value in caches.items()}
                or signature(topk) != topk_owner):
            raise RuntimeError("A timed graph changed a live producer or consumer owner")
        if not bool((topk[inputs.rows:] == -1).all()):
            raise RuntimeError("A timed graph wrote inactive persistent top-k rows")
        for name, value in inputs.fields().items():
            if not torch.equal(input_bytes[name].view(torch.uint8),
                               snapshot(torch, value).view(torch.uint8)):
                raise RuntimeError("A graph overwrote live input or metadata bytes")

    def update(iteration):
        inputs.update(iteration)
        input_bytes.clear()
        input_bytes.update({name: snapshot(torch, value)
                            for name, value in inputs.fields().items()})

    def rank_witness(actual):
        witness = {"rank": args.device_rank,
                   "input_hidden": tensor_witness(torch, input_bytes["hidden"])["sha256"],
                   "input_residual": tensor_witness(torch, input_bytes["residual"])["sha256"]}
        for number in (6, 7):
            for name, source in (("retained", "bridge.retained"),
                                 ("attention_partial", "attention.partial"),
                                 ("ffn_partial", "bridge.combined"),
                                 ("ar1", "reduction.attention"),
                                 ("ar2", "reduction.output")):
                witness[f"layer{number}.{name}"] = tensor_witness(
                    torch, actual[f"layer{number}.{source}"])["sha256"]
        peers = gather(witness)
        check_rank_witnesses(peers)
        return peers

    persist()
    try:
        update(0)
        reset(0, "control")
        literal_records = []
        pair("original", literal_records)
        expected = collect()
        reset(0, "control")
        pair("control")
        report["native_control_check"] = compare(collect(), expected, "native-control")
        report["literal_conversion_checks"] = literal_records
        for calls in (1, CALLS):
            for arm in ("control", "candidate"):
                reset(0, arm)

                def execute(combined, observations, arm=arm, calls=calls):
                    for _ in range(calls):
                        pair(arm)

                graph = CapturedChain(torch, config, layers, inputs, cases, window,
                                      stream, execute, outputs, values)
                graphs[calls, arm] = graph
                captured = graph.capture()
                report["captures"].append({"arm": arm, "pair_calls": calls, **captured})
                persist()

        for iteration in range(8):
            update(iteration)
            reset(iteration, "control")
            literal_records = []
            pair("original", literal_records)
            expected = collect()
            guards()
            record = {"iteration": iteration, "literal_conversions": literal_records,
                      "inputs": inputs.witness(), "rank_witnesses": rank_witness(expected),
                      "checks": []}
            for calls in (1, CALLS):
                for arm in ("control", "candidate"):
                    reset(iteration, arm)
                    graphs[calls, arm].replay()
                    actual = collect()
                    guards()
                    record["checks"].append({"arm": arm, "pair_calls": calls,
                        "checks": compare(actual, expected, f"r{iteration}-n{calls}-{arm}")})
            report["checks"].append(record)
            persist()

        timed_iteration = 16
        update(timed_iteration)
        reset(timed_iteration, "control")
        literal_records = []
        pair("original", literal_records)
        expected = collect()
        report["timed_inputs"] = inputs.witness()
        report["timed_literal_conversions"] = literal_records
        report["timed_rank_witnesses"] = rank_witness(expected)
        for arm in ("control", "candidate"):
            reset(timed_iteration, arm)
            for replay in range(3):
                graphs[CALLS, arm].replay()
                actual = collect()
                guards()
                report["idempotence_checks"].append({"arm": arm, "replay": replay,
                    "pair_calls": CALLS, "cache_reset_between_replays": False,
                    "checks": compare(actual, expected, f"idempotence-{arm}-{replay}")})
        persist()

        tail_names = ("consumer.output", "consumer.retained", "layer6.reduction.attention",
                      "layer6.reduction.output", "layer7.reduction.attention",
                      "layer7.reduction.output", "layer6.persistent_topk",
                      "layer6.body.active_list", "layer7.body.active_list")
        expected_tail = {name: expected[name] for name in tail_names}
        for calls, collection in ((CALLS, "samples"), (1, "one_pair_sensitivity")):
            for sample in range(SAMPLES):
                order = (("candidate", "control") if sample % 2 == 0
                         else ("control", "candidate"))
                record = {"sample": sample, "order": order, "pair_calls": calls, "arms": {}}
                for arm in order:
                    reset(timed_iteration, arm)
                    for _ in range(3):
                        graphs[calls, arm].replay()
                    torch.npu.synchronize()
                    key = [arena, calls, sample, arm]
                    if any(value != key for value in gather(key)):
                        raise RuntimeError("The ranks disagree on event order")
                    torch.distributed.barrier(group=args.oracle_cpu)
                    start = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    start.record(stream)
                    for _ in range(REPLAYS):
                        graphs[calls, arm].replay()
                    end.record(stream)
                    end.synchronize()
                    duration = start.elapsed_time(end) * 1000
                    durations = gather({"rank": args.device_rank, "key": key,
                                        "duration_us": duration})
                    maximum = paired_maximum(durations, key)
                    actual = {name: snapshot(torch, values[name]) for name in tail_names}
                    guards()
                    record["arms"][arm] = {
                        "rank_intervals": durations, "maximum_event_us": maximum,
                        "per_pair_us": maximum / (calls * REPLAYS),
                        "tail_checks": compare(actual, expected_tail,
                                               f"n{calls}-sample{sample}-{arm}")}
                record["paired_saving_us"] = (record["arms"]["control"]["per_pair_us"]
                                             - record["arms"]["candidate"]["per_pair_us"])
                report[collection].append(record)
                persist()
        report["postchecks"] = []
        for arm in ("control", "candidate"):
            reset(timed_iteration, arm)
            graphs[CALLS, arm].replay()
            actual = collect()
            guards()
            report["postchecks"].append({"arm": arm, "checks": compare(
                actual, expected, f"post-{arm}")})
        report["summary_us"] = {arm: distribution([
            value["arms"][arm]["per_pair_us"] for value in report["samples"]])
            for arm in ("control", "candidate")}
        report["paired_saving_us"] = distribution([
            value["paired_saving_us"] for value in report["samples"]])
        report["candidate_wins"] = sum(
            value["paired_saving_us"] > 0 for value in report["samples"])
        report["one_pair_summary_us"] = {arm: distribution([
            value["arms"][arm]["per_pair_us"] for value in report["one_pair_sensitivity"]])
            for arm in ("control", "candidate")}
        report["graph_execution_counts"] = [
            {"arm": arm, "pair_calls": calls, "captures": graph.calls,
             "replays": graph.replays, "python_replay_calls": graph.calls - 1,
             "expected_candidate_epoch_increments_from_calls": (
                 4 * calls * graph.replays if arm == "candidate" else 0)}
            for (calls, arm), graph in graphs.items()]
        report["epoch_count_scope"] = "Derived from explicit graph-call counts; no device epoch counter read"
        report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report["pass"] = True
        persist()
        return report
    finally:
        for graph in graphs.values():
            graph.close()
