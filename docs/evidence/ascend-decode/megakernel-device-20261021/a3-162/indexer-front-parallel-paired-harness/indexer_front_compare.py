"""Pair two native indexer front schedules on identical inputs and weights."""

import argparse
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import attention_check
import indexer_front_check


HERE = Path(__file__).resolve().parent
ARMS = ("ordinary", "candidate")
OUTPUTS = ("query", "key", "weights", "query_scale", "key_scale")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinary-library", type=Path, default=HERE /
                        "indexer_front_build-scale-order/whole_layer_indexer_front.so")
    parser.add_argument("--candidate-library", type=Path, default=HERE /
                        "indexer_front_build-parallel-v2/whole_layer_indexer_front.so")
    parser.add_argument("--ordinary-namespace", default="tilert")
    parser.add_argument("--candidate-namespace", default="tilert_front_parallel")
    parser.add_argument("--widths", type=int, nargs="+", default=[12, 32])
    parser.add_argument("--blocks", type=int, nargs="+", default=[24])
    parser.add_argument("--modes", nargs="+", choices=["int8", "bf16"],
                        default=["int8", "bf16"])
    parser.add_argument("--query-scale-dtypes", nargs="+",
                        choices=["float32", "bfloat16"],
                        default=["float32", "bfloat16"])
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--replays", type=int, default=32)
    parser.add_argument("--timing-pairs", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timed-replay", type=int, default=0)
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(value < 1 or value > 32 for value in args.widths):
        parser.error("widths must lie in [1,32]")
    if any(value < 1 or value > 64 for value in args.blocks):
        parser.error("blocks must lie in [1,64]")
    if min(args.replays, args.iterations, args.warmup) < 1:
        parser.error("replays, iterations and warmup must be positive")
    if args.timing_pairs < 30:
        parser.error("timing requires at least 30 independent samples per arm")
    if args.timed_replay < 0 or args.timed_replay % 4 == 1:
        parser.error("timed-replay must select nonzero inputs")
    return args


def summary(values):
    percentiles = statistics.quantiles(values, n=100, method="inclusive")
    return {"samples": len(values), "p50_us": statistics.median(values),
            "p90_us": percentiles[89], "p99_us": percentiles[98],
            "std_us": statistics.pstdev(values), "min_us": min(values),
            "max_us": max(values)}


def boundaries(torch, prepared, rows, quantized):
    scratch, _, outputs, _, _ = prepared
    cpu = scratch.cpu()
    offsets, _ = indexer_front_check.layout(rows)
    capacity = 16 if rows <= 16 else 32
    result = {name: value for name, value in zip(OUTPUTS, outputs)
              if value.numel()}
    shapes = {
        "query_projection": (torch.bfloat16, (rows, 32, 128)),
        "key_projection": (torch.bfloat16, (rows, 160)),
        "key_normalized": (torch.bfloat16, (rows, 128)),
        "query_rope": (torch.bfloat16, (rows, 32, 128)),
        "key_rope": (torch.bfloat16, (rows, 128)),
        "norm_stats": (torch.float32, (rows, 2)),
    }
    for name, (dtype, shape) in shapes.items():
        start, size = offsets[name]
        result[name] = cpu[start:start + size].view(dtype).reshape(shape)
    for name, tiles, columns, dtype in (
            ("query_acc", 16, 256, torch.int32),
            ("key_acc", 5, 32, torch.float32)):
        start, size = offsets[name]
        # Matmul writes [N tile, padded M, tile N]; padded rows are not read.
        result[name] = cpu[start:start + size].view(dtype).reshape(
            tiles, capacity, columns)[:, :rows].permute(1, 0, 2).reshape(
                rows, tiles * columns)
    if quantized:
        for name, columns in (("query_hadamard", 4096),
                              ("key_hadamard", 128)):
            start, size = offsets[name]
            result[name] = cpu[start:start + size].view(torch.float32)[
                :rows * columns].reshape(rows, columns)
    return result


def compare(torch, observed):
    return {name: attention_check.metric(torch, candidate,
                                        observed["ordinary"][name])
            for name, candidate in observed["candidate"].items()}


def passed(metrics):
    return all(item["pass"] and item["bitwise_equal"]
               for item in metrics.values())


def run_case(torch, torch_npu, args, fixture, rows, mode, blocks, scale_dtype):
    def nd(value):
        return torch_npu.npu_format_cast(
            value.to(args.device).contiguous(), 2)

    hidden_base = nd(fixture["hidden"][:rows])
    query_base = nd(fixture["query_norm"][:rows])
    scale_base = nd(fixture["query_norm_scale"][:rows])
    hidden, query, token_scale = (nd(value.clone()) for value in
                                  (hidden_base, query_base, scale_base))
    query_weight = nd(fixture["query_weight"].reshape(
        2048, 16, 256).permute(1, 0, 2))
    key_weight = nd(fixture["key_weight"].reshape(
        6144, 5, 32).permute(1, 0, 2))
    channel = nd(fixture["query_weight_scale"].to(
        getattr(torch, scale_dtype)))
    gamma, beta, hadamard = (nd(fixture[name]) for name in
                            ("gamma", "beta", "hadamard"))
    cosine = nd(torch.ones(rows, 64, dtype=torch.bfloat16))
    sine = nd(torch.zeros(rows, 64, dtype=torch.bfloat16))
    quantized = mode == "int8"
    ops = {name: getattr(torch.ops, getattr(args, name + "_namespace"))
           for name in ARMS}
    prepared = {name: ops[name].indexer_front_prepare(
        hidden, quantized, blocks) for name in ARMS}
    expected_bytes = indexer_front_check.layout(rows)[1]
    for arm in prepared.values():
        if arm[0].nbytes != expected_bytes or arm[3] != blocks:
            raise ValueError("prepare returned a different scratch/launch geometry")
    for index in (0, 1):
        if prepared["ordinary"][index].data_ptr() == (
                prepared["candidate"][index].data_ptr()):
            raise ValueError("scratch and tiling must be owned independently")

    def call(name, queued):
        values = (hidden, query, token_scale)
        if queued:
            values = tuple(value.clone() for value in values)
        scratch, tiling, outputs, actual_blocks, sync = prepared[name]
        return ops[name].indexer_front_out(
            *values, query_weight, channel, key_weight, gamma, beta, cosine,
            sine, hadamard, scratch, tiling, outputs, quantized,
            actual_blocks, sync)

    stream = torch.npu.Stream(device=args.device)
    stream.wait_stream(torch.npu.current_stream(args.device))

    def update(replay):
        with torch.npu.stream(stream):
            factor = (1.0, 0.0, -1.0, .5)[replay % 4]
            hidden.copy_((hidden_base.float() * factor).bfloat16())
            query.copy_((query_base.float() * factor).to(torch.int8))
            token_scale.copy_(scale_base * (1 + replay / 8))
            positions = torch.arange(rows, device=args.device) + 8192 + replay
            frequency = torch.arange(32, device=args.device).float() + 1
            angle = positions.float()[:, None] * frequency[None, :] / 1024
            cosine.copy_(angle.cos().repeat(1, 2).bfloat16())
            sine.copy_(angle.sin().repeat(1, 2).bfloat16())
        stream.synchronize()

    def observe():
        return {name: boundaries(torch, prepared[name], rows, quantized)
                for name in ARMS}

    report = {"rows": rows, "mode": mode, "blocks": blocks,
              "query_scale_dtype": scale_dtype, "pass": False,
              "scratch_bytes_per_arm": expected_bytes, "checks": []}
    update(0)
    numeric_graphs = {name: attention_check.capture(
        torch, lambda name=name: call(name, True), stream)[0] for name in ARMS}
    for replay in range(args.replays):
        update(replay)
        with torch.npu.stream(stream):
            for name in ARMS[::1 if replay % 2 == 0 else -1]:
                numeric_graphs[name].replay()
        stream.synchronize()
        observed = observe()
        metrics = compare(torch, observed)
        report["checks"].append({"replay": replay, "pass": passed(metrics),
                                  "boundaries": metrics})
        if not passed(metrics):
            path = args.output.with_name(
                f"{args.output.stem}-{scale_dtype}-{mode}-m{rows}-r{replay}.pt")
            torch.save({name: {key: value.detach().cpu().clone()
                              for key, value in values.items()}
                        for name, values in observed.items()}, path)
            report["failure_dump"] = str(path)
            return report
    if not args.no_timing:
        update(args.timed_replay)
        graphs = {name: attention_check.capture(
            torch, lambda name=name: call(name, False), stream)[0]
                  for name in ARMS}
        with torch.npu.stream(stream):
            for iteration in range(args.warmup):
                for name in ARMS[::1 if iteration % 2 == 0 else -1]:
                    graphs[name].replay()
        stream.synchronize()
        pairs = []
        for pair in range(args.timing_pairs):
            order = ARMS[::1 if pair % 2 == 0 else -1]
            measurements = {name: attention_check.time_graph(
                torch, graphs[name], stream, args.iterations) for name in order}
            pairs.append({"order": list(order), **measurements})
        final = compare(torch, observe())
        report["timing"] = {
            "pairs_us": pairs,
            "summary": {name: summary([pair[name] for pair in pairs])
                        for name in ARMS},
            "candidate_minus_ordinary_us": summary([
                pair["candidate"] - pair["ordinary"] for pair in pairs]),
            "candidate_faster_pairs": sum(pair["candidate"] <
                                           pair["ordinary"] for pair in pairs),
            "final_boundaries": final, "pass": passed(final),
            "timed_replay": args.timed_replay,
            "input_nonzero": {"hidden": int(hidden.count_nonzero()),
                              "query": int(query.count_nonzero())},
        }
        if not passed(final):
            return report
    report["pass"] = True
    return report


def main():
    args = arguments()
    import torch
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    identity = {}
    for name in ARMS:
        library = getattr(args, name + "_library").resolve()
        torch.ops.load_library(str(library))
        metadata = SimpleNamespace(library=library, device=args.device,
                                   fixture=args.fixture)
        identity[name] = attention_check.provenance(torch, torch_npu, metadata)
        path = library.parent / "build-manifest.json"
        if path.exists():
            identity[name]["build_manifest"] = json.loads(path.read_text())
        identity[name]["namespace"] = getattr(args, name + "_namespace")
    report = {
        "scope": "One native indexer front; shared input/weights, separate arenas",
        "numeric_scope": "captured producer clones followed by one front launch",
        "timing_scope": "one captured front launch; excludes producer clones",
        "identity": identity,
        "helpers_sha256": {path.name: attention_check.sha256(path) for path in
                           (Path(__file__), Path(attention_check.__file__),
                            Path(indexer_front_check.__file__))},
        "replays": args.replays, "timing_pairs": args.timing_pairs,
        "iterations_per_event_sample": args.iterations,
        "warmup_replays": args.warmup,
        "capture_warmup_calls_per_graph": 3,
        "cases": [], "pass": False,
    }
    try:
        data = indexer_front_check.fixture(torch, args)
        with torch.inference_mode():
            for dtype in args.query_scale_dtypes:
                for mode in args.modes:
                    for rows in args.widths:
                        for blocks in args.blocks:
                            result = run_case(torch, torch_npu, args, data,
                                              rows, mode, blocks, dtype)
                            report["cases"].append(result)
                            print(json.dumps({key: result[key] for key in
                                  ("rows", "mode", "blocks",
                                   "query_scale_dtype", "pass")}), flush=True)
                            if not result["pass"]:
                                raise RuntimeError("front schedule comparison failed")
        report["pass"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
