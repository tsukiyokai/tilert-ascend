"""Pair two local-layer graphs with shared weights and identical tensor owners."""

import hashlib
import json
from pathlib import Path
import statistics

from tiling_packets import TilingPair, load_control
from local_chain_views import poison, strict_checks
from vendor_hook_forward import snapshot
from vendor_hook_graph import tensor_witness


CONTROL_SHA = "609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f"
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


def pair_values(cases, caches, topk):
    rows = cases[0].rows
    return {**{f"layer{number}.{name}": value
               for number, case in zip((6, 7), cases)
               for name, value in case.values.items()},
            **caches,
            "layer6.persistent_topk": topk[:rows],
            "layer7.persistent_topk": topk[:rows],
            "consumer.output": cases[1].bridge[10],
            "consumer.retained": cases[1].bridge[1]}


def run_pair_timing(torch, cases, inputs, topk, caches, bases, arena, output,
                    comparison="down-tiling"):
    from attention_check import metric
    from local_chain import publish_topk, signature

    if comparison != "down-tiling":
        raise ValueError("Both arms must be 609 with a private Down tiling")
    from native_components import prepare, separate as native_entry

    prepare(torch, cases)
    control_native = load_control(torch)
    packets = [TilingPair(case, control_native) for case in cases]
    owners = [case.owners() for case in cases]
    topk_signature = signature(topk)
    values = pair_values(cases, caches, topk)
    if len(values) != 109:
        raise RuntimeError("The timing fixture lost a defined numerical boundary")
    report = {
        "arena": arena, "rows": inputs.rows, "blocks": 24,
        "scope": "Two consecutive local layers with identity TP reductions",
        "comparison": "609 plus persistent top-k copy in both arms; N512 versus N1024 Down tiling",
        "comparison_mode": comparison,
        "unit": "us per indexed/shared two-layer pair",
        "weight_banks": 1, "weight_sets": 2,
        "weight_reuse": "The same two actual synthetic Decoder weight sets repeat in each graph",
        "shared_between_arenas": "Weights, original persistent top-k owner and caches",
        "scratch_owners": owners, "same_host_record_bytes": False,
        "host_record_contract": "Only body.scratch[7] differs per layer",
        "prepared_records": [packet.check() for packet in packets],
        "graph_pair_calls": CALLS, "graph_replays_per_event": REPLAYS,
        "samples_per_arm": SAMPLES, "warm_replays_before_event": 3,
        "profiler_enabled": False, "copies_and_poison_outside_events": True,
        "observation_copies_inside_graph": 0,
        "norm_owner_copies_per_layer": {"control": 0, "candidate": 0},
        "topk_copy_calls_per_pair": {"control": 1, "candidate": 1},
        "checks": [], "samples": [], "pass": False,
    }
    destination = output / f"layer-pair-timing-arena{arena}.json"

    def persist():
        destination.write_text(json.dumps(report, indent=2) + "\n")

    def reset(iteration, arm):
        for name, cache in caches.items():
            cache.copy_(bases[name].roll(iteration, 0))
        topk.fill_(-1)
        for case in cases:
            poison(case.values, 1 if arm == "control" else 2)
            case.norm[5].fill_(float("nan"))
            case.body[20].fill_(1)
            case.body[20][0].zero_()
        cases[0].selected.fill_(1 if arm == "control" else 2)

    def pair(arm):
        for index, packet in enumerate(packets):
            getattr(packet, arm).launch()
            if index == 0:
                publish_topk(cases[index].selected, topk, inputs.rows)

    def native_pair():
        for index, case in enumerate(cases):
            native_entry(case)
            if index == 0:
                publish_topk(case.selected, topk, inputs.rows)

    def sequence(arm):
        for _ in range(CALLS):
            pair(arm)

    def collect():
        torch.npu.synchronize()
        return {name: snapshot(torch, value) for name, value in values.items()}

    def compare(actual, expected, label):
        checks, passed = strict_checks(torch, actual, expected, metric)
        if not passed:
            torch.save({"actual": actual, "expected": expected},
                       output / f"layer-pair-{arena}-{label}.pt")
            raise RuntimeError(f"The paired graph differs at {label}")
        return checks

    def guards(arm):
        if (owners != [case.owners() for case in cases]
                or signature(topk) != topk_signature):
            raise RuntimeError("A timed graph replaced a retained owner")
        if not bool((topk[inputs.rows:] == -1).all()):
            raise RuntimeError("A timed graph overwrote inactive top-k rows")
        inputs.check_owners()
        return [packet.check() for packet in packets]

    graphs = {}
    persist()
    try:
        if native_entry is not None:
            report["native_component_checks"] = []
            report["control_owner_checks"] = []
            for iteration in range(8):
                inputs.update(iteration)
                reset(iteration, "control")
                for index, case in enumerate(cases):
                    case.separate()
                    if index == 0:
                        publish_topk(case.selected, topk, inputs.rows)
                expected = collect()
                reset(iteration, "candidate")
                native_pair()
                actual = collect()
                report["native_component_checks"].append({
                    "iteration": iteration,
                    "checks": compare(actual, expected, f"native-input{iteration}"),
                    "inputs": inputs.witness(), "down_tiling": guards("control")})
                for packet in packets:
                    expected_ffn = {name: actual[f"layer{packet.case.number}.{name}"]
                                    for name in packet.values}
                    observed = packet.validate_control()
                    report["control_owner_checks"].append({
                        "iteration": iteration, "layer": packet.case.number,
                        "checks": compare(observed, expected_ffn,
                                          f"control-owner-{iteration}-{packet.case.number}"),
                        "records": packet.check()})
                persist()
        inputs.update(8)
        for arm in ("control", "candidate"):
            reset(8, arm)
            for _ in range(3):
                sequence(arm)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                sequence(arm)
            graphs[arm] = graph
            report.setdefault("capture_records", {})[arm] = guards(arm)
            persist()

        for iteration in range(8, 16):
            inputs.update(iteration)
            snapshots = {}
            tilings = {}
            for arm in ("control", "candidate"):
                reset(iteration, arm)
                graphs[arm].replay()
                snapshots[arm] = collect()
                tilings[arm] = guards(arm)
            checks = compare(snapshots["candidate"], snapshots["control"],
                             f"input{iteration}")
            report["checks"].append({
                "iteration": iteration, "checks": checks, "down_tiling": tilings,
                "inputs": inputs.witness(),
                "output": tensor_witness(torch, snapshots["candidate"]["consumer.output"]),
                "retained": tensor_witness(torch, snapshots["candidate"]["consumer.retained"])})
            persist()

        tail = {"output": cases[1].bridge[10], "retained": cases[1].bridge[1],
                "layer6.active_list": cases[0].body[20],
                "layer7.active_list": cases[1].body[20],
                "persistent_topk": topk[:inputs.rows]}
        inputs.update(16)
        reset(16, "control")
        graphs["control"].replay()
        torch.npu.synchronize()
        expected = {name: snapshot(torch, value) for name, value in tail.items()}
        report["timed_inputs"] = inputs.witness()
        report["route_counts"] = [case.body[12].cpu().tolist() for case in cases]
        weight_owners = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
                         for case in cases
                         for value in (case.norm[2], *case.main, *case.front, *case.tail,
                                       case.ffn.gamma, case.ffn.gate, case.ffn.bias,
                                       *case.ffn.weights, *case.ffn.shared)}
        report["resident_weight_bytes"] = sum(weight_owners.values())
        report["active_routed_matrix_bytes"] = [
            sum(count > 0 for count in counts) * sum(
                case.ffn.weights[index].untyped_storage().nbytes() // 256
                for index in (0, 3))
            for case, counts in zip(cases, report["route_counts"])]
        for sample in range(SAMPLES):
            order = ("candidate", "control") if sample % 2 == 0 else ("control", "candidate")
            record = {"sample": sample, "order": order,
                      "event_total_us": {}, "per_pair_us": {}, "tail_checks": {}}
            for arm in order:
                reset(16, arm)
                for _ in range(3):
                    graphs[arm].replay()
                torch.npu.synchronize()
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                for _ in range(REPLAYS):
                    graphs[arm].replay()
                end.record()
                end.synchronize()
                total = start.elapsed_time(end) * 1000
                if total <= 0:
                    raise RuntimeError("The device event interval is not positive")
                record["event_total_us"][arm] = total
                record["per_pair_us"][arm] = total / (REPLAYS * CALLS)
                actual = {name: snapshot(torch, value) for name, value in tail.items()}
                record["tail_checks"][arm] = compare(actual, expected, f"sample{sample}-{arm}")
                record.setdefault("down_tiling", {})[arm] = guards(arm)
            record["paired_saving_us"] = (record["per_pair_us"]["control"]
                                         - record["per_pair_us"]["candidate"])
            report["samples"].append(record)
            persist()
        report["post_timing_checks"] = []
        reset(16, "control")
        native_pair()
        expected_full = collect()
        for arm in ("control", "candidate"):
            reset(16, arm)
            graphs[arm].replay()
            observed = collect()
            report["post_timing_checks"].append({
                "arm": arm, "checks": compare(observed, expected_full, f"post-{arm}"),
                "records": guards(arm)})
        report["summary"] = {arm: distribution([
            sample["per_pair_us"][arm] for sample in report["samples"]])
            for arm in ("control", "candidate")}
        report["paired_saving_us"] = distribution([
            sample["paired_saving_us"] for sample in report["samples"]])
        report["candidate_wins"] = sum(
            sample["paired_saving_us"] > 0 for sample in report["samples"])
        report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report["pass"] = True
        persist()
        return report
    finally:
        torch.npu.synchronize()
        for graph in graphs.values():
            graph.reset()
