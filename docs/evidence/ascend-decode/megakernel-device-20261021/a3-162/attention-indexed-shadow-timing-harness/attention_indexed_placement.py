"""Compare two indexed-attention kernels on each of two identical-layout arenas."""

import statistics

from attention_check import capture, metric, time_graph


def compare_placements(torch, args, stream, invoke, arenas, observe, operators):
    expected = observe(arenas["A"])
    with torch.npu.stream(stream):
        initial_caches = [value.clone() for value in arenas["A"]["caches"]]
    stream.synchronize()

    def reset(arena):
        with torch.npu.stream(stream):
            for destination, source in zip(arena["caches"], initial_caches):
                destination.copy_(source)
        stream.synchronize()

    graphs = {}
    for placement, arena in arenas.items():
        for name, ops in operators.items():
            reset(arena)
            graphs[placement, name], _ = capture(
                torch, lambda ops=ops, arena=arena: invoke(ops, arena), stream)

    def verify(arena):
        values = observe(arena)
        metrics = {name: metric(torch, value, expected[name])
                   for name, value in values.items()}
        passed = all(value["pass"] and value["bitwise_equal"]
                     for name, value in metrics.items()
                     if name != "norm_stats")
        return {"pass": passed, "boundaries": metrics}

    checks = {}
    for (placement, name), graph in graphs.items():
        reset(arenas[placement])
        with torch.npu.stream(stream):
            graph.replay()
        stream.synchronize()
        checks[placement + "/" + name] = verify(arenas[placement])
    report = {
        "scope": (
            "Four recaptured graphs O(A), P(A), O(B), P(B). Within each arena, "
            "ordinary and candidate share scratch, tiling, output and all cache "
            "addresses. All graphs share input, weights and metadata. Replays "
            "are serialized. Each arm restores the same cache values and warms "
            "three graph replays outside its event; timed slot writes are idempotent."),
        "arena_addresses": {
            name: {"scratch": arena["scratch"].data_ptr(),
                   "tiling": arena["tiling"].data_ptr(),
                   "output": arena["output"].data_ptr(),
                   "caches": [value.data_ptr() for value in arena["caches"]]}
            for name, arena in arenas.items()
        },
        "pre_timing": checks, "pairs_us_per_call": {name: [] for name in arenas},
        "post_timing": {}, "pass": all(value["pass"] for value in checks.values()),
    }
    if not report["pass"]:
        return report
    placements = tuple(arenas)
    names = tuple(operators)
    for pair in range(args.timing_pairs):
        step = 1 if pair % 2 == 0 else -1
        for placement in placements[::step]:
            measurements = {"order": list(names[::step])}
            for name in names[::step]:
                reset(arenas[placement])
                graph = graphs[placement, name]
                with torch.npu.stream(stream):
                    for _ in range(3):
                        graph.replay()
                stream.synchronize()
                measurements[name] = time_graph(
                    torch, graph, stream, args.iterations) / args.calls
            report["pairs_us_per_call"][placement].append(measurements)
    for (placement, name), graph in graphs.items():
        reset(arenas[placement])
        with torch.npu.stream(stream):
            graph.replay()
        stream.synchronize()
        result = verify(arenas[placement])
        report["post_timing"][placement + "/" + name] = result
        report["pass"] = report["pass"] and result["pass"]
    report["summary"] = {}
    for placement, pairs in report["pairs_us_per_call"].items():
        report["summary"][placement] = {
            "medians_us": {name: statistics.median(pair[name] for pair in pairs)
                           for name in names},
            "median_ordinary_minus_candidate_us": statistics.median(
                pair["ordinary"] - pair["candidate"] for pair in pairs),
            "candidate_wins": sum(pair["candidate"] < pair["ordinary"]
                                  for pair in pairs),
        }
    return report
