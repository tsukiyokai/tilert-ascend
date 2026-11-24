"""Compare the native BF16 routing front with retained CANN dispatch."""

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import statistics
import sys

import check


HERE = Path(__file__).resolve().parent
HIDDEN, TOP_K, EXPERTS = check.HIDDEN, check.TOP_K, check.EXPERTS


def metadata_reference(torch, choices, probabilities):
    original = choices.flatten().argsort(stable=True)
    count = torch.bincount(choices.flatten().long(), minlength=EXPERTS).int()
    start = count.cumsum(0).int() - count
    source = torch.zeros(EXPERTS, dtype=torch.int32)
    slot = torch.zeros_like(source)
    weight = torch.zeros(EXPERTS, dtype=torch.float32)
    routes = original.numel()
    source[:routes] = original // TOP_K
    slot[:routes] = original % TOP_K
    weight[:routes] = probabilities.bfloat16().float().flatten()[original]
    return count, start, source, slot, weight


def timing_summary(values):
    def percentile(fraction):
        ordered = sorted(values)
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {"samples": len(values), "p50": percentile(0.5),
            "p90": percentile(0.9), "p99": percentile(0.99),
            "std": statistics.pstdev(values)}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--library", type=Path,
                        default=HERE / "build-input-front/routed_int4_torch.so")
    parser.add_argument("--balanced-library", type=Path,
                        default=HERE / "build-balanced-zero/routed_int4_torch.so")
    parser.add_argument("--vendor-library", type=Path,
                        default=HERE / "build-reference/nz_reference.so")
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 12, 32])
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--timing-samples", type=int, default=30)
    parser.add_argument("--timed-route", type=int, default=3)
    parser.add_argument("--banks", type=int, default=1)
    parser.add_argument("--calls", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261021)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--fixture", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replays < 8 or any(width < 1 or width > 32 for width in args.widths):
        parser.error("M must be in [1,32] and at least eight replays are required")
    if min(args.banks, args.calls, args.warmup, args.iterations, args.blocks) < 1:
        parser.error("bank, call, warmup, iteration and block counts must be positive")
    if args.calls % args.banks:
        parser.error("calls must be divisible by banks")
    if not args.no_timing and (args.timing_samples < 30 or args.timing_samples % 6):
        parser.error("timing-samples must cover at least five complete six-order rounds")
    return args


def run_width(torch, torch_npu, args, weights, banks, encoded, width):
    hidden = torch.empty((width, HIDDEN), dtype=torch.bfloat16, device=args.device)
    choices = torch.empty((width, TOP_K), dtype=torch.int32, device=args.device)
    probabilities = torch.empty((width, TOP_K), dtype=torch.float32, device=args.device)
    sample = torch.empty_like(hidden, dtype=torch.int8)
    namespaces = {"native_front": torch.ops.tilert_front,
                  "balanced_dispatch": torch.ops.tilert_balanced}
    arenas = {name: namespace.routed_experts_prepare(
        sample, EXPERTS, args.blocks, weight_layout=1, from_dispatch=True,
    ) for name, namespace in namespaces.items()}
    stream = torch.npu.Stream(device=args.device)
    sequence = banks * (args.calls // args.banks)
    details = {}

    def dispatch():
        return torch_npu.npu_moe_init_routing_v2(
            hidden, choices, active_num=width * TOP_K, expert_num=EXPERTS,
            expert_tokens_num_type=1, expert_tokens_num_flag=True,
            active_expert_range=[0, EXPERTS], quant_mode=1,
        )

    def native_call(name, queued=False):
        scratch, sync = arenas[name]
        namespace = namespaces[name]
        for bank in sequence:
            arguments = (bank["gate_up"], weights["gu_bias"], weights["gu_scale"],
                         bank["down"], weights["down_bias"], weights["down_scale"])
            if name == "native_front":
                x, ids, probs = (hidden.clone(), choices.clone(), probabilities.clone()) \
                    if queued else (hidden, choices, probabilities)
                output = namespace.routed_experts_from_bf16(
                    x, *arguments, ids, probs, scratch, sync, EXPERTS,
                    args.blocks, weight_layout=1,
                )
            else:
                rows, inverse, counts, scales = dispatch()
                output = namespace.routed_experts_from_dispatch(
                    rows, *arguments, scales.reshape(-1), inverse, counts,
                    probabilities, scratch, sync, EXPERTS, args.blocks,
                    weight_layout=1,
                )
        return output

    def vendor_call(keep=True):
        for bank in sequence:
            rows, inverse, counts, scales = dispatch()
            counts64 = counts.long()
            codes, middle_scale = torch.ops.tilert_reference.nz_gmm1(
                rows, bank["gate_up"], encoded["gu_scale"], scales,
                counts64, weights["gu_bias"],
            )
            saved = codes.clone() if keep else codes
            routed = torch.ops.tilert_reference.nz_gmm2(
                codes, bank["down"], encoded["down_scale"], middle_scale,
                counts64, weights["down_bias"],
            )
            output = torch_npu.npu_moe_token_unpermute(
                routed, inverse, probs=probabilities.bfloat16(),
            )
            if keep:
                details.update(rows=rows, scales=scales, inverse=inverse,
                               codes=saved, middle_scale=middle_scale, routed=routed)
        return output

    def update(iteration):
        values, ids, probs = check.route_case(torch, width, iteration, args.seed)
        with torch.npu.stream(stream):
            hidden.copy_(values)
            choices.copy_(ids)
            probabilities.copy_(probs)
        stream.synchronize()
        return metadata_reference(torch, ids, probs)

    torch.npu.synchronize()
    update(0)
    calls = {name: (lambda name=name: native_call(name)) for name in namespaces}
    calls["vendor"] = vendor_call
    captured = {name: check.capture(torch, call, stream, args.warmup)
                for name, call in calls.items()}
    report = {"width": width, "checks": [], "pass": True,
              "scratch_bytes": {name: sum(t.nbytes for t in arena[0])
                                for name, arena in arenas.items()}}
    for iteration in range(args.replays):
        metadata = update(iteration)
        with torch.npu.stream(stream):
            for graph, _ in captured.values():
                graph.replay()
        stream.synchronize()
        scratch = arenas["native_front"][0]
        baseline = arenas["balanced_dispatch"][0]
        inverse = details["inverse"].long()
        required = {}

        def compare(name, actual, expected):
            required[name] = check.compare(torch, actual, expected, 0, 0)

        for name in namespaces:
            compare(name + "_final", captured[name][1], captured["vendor"][1])
        compare("input_codes", scratch[10].repeat_interleave(TOP_K, 0),
                details["rows"].index_select(0, inverse))
        compare("input_scale", scratch[11].repeat_interleave(TOP_K, 0),
                details["scales"].index_select(0, inverse).reshape(-1))
        for name, actual, expected in zip(
                ("counts", "starts", "source", "slot", "probability"), scratch[12:17], metadata):
            compare(name, actual, expected)
        codes, scales = check.middle_stages(torch, scratch, scratch[12:17], width)
        compare("middle_codes", codes, details["codes"].index_select(0, inverse))
        compare("middle_scale", scales, details["middle_scale"].index_select(0, inverse))
        compare("routes", scratch[5].reshape(-1, HIDDEN),
                details["routed"].index_select(0, inverse))
        for name, index in (("stacked", 0), ("gate_int32", 1),
                            ("middle_packed", 2), ("down_int32", 4)):
            compare(name, check.ordered_partials(torch, scratch[index], scratch[12:17], width, 2),
                    check.ordered_partials(torch, baseline[index], baseline[12:17], width, 2))
        with torch.npu.stream(stream):
            queued = native_call("native_front", queued=True)
        stream.synchronize()
        compare("queued_producer", queued, captured["vendor"][1])
        passed = all(value["pass"] and value["bitwise_equal"] is True
                     for value in required.values())
        report["checks"].append({"iteration": iteration, "boundaries": required,
                                 "pass": passed, "active_experts": int((metadata[0] > 0).sum())})
        if not passed:
            failure = args.output.with_name(args.output.stem + f"-m{width}-r{iteration}.pt")
            torch.save({"hidden": hidden.cpu(), "choices": choices.cpu(),
                        "probabilities": probabilities.cpu(),
                        "native_scratch": [t.cpu() for t in scratch],
                        "balanced_scratch": [t.cpu() for t in baseline],
                        "vendor": {key: value.cpu() for key, value in details.items()},
                        "vendor_output": captured["vendor"][1].cpu()}, failure)
            report.update({"pass": False, "failure": str(failure)})
            return report
    if args.no_timing:
        return report
    metadata = update(args.timed_route)
    captured["vendor"] = check.capture(
        torch, lambda: vendor_call(keep=False), stream, args.warmup,
    )
    orders = list(itertools.permutations(captured))
    samples = {name: [] for name in captured}
    for sample_index in range(args.timing_samples):
        for name in orders[sample_index % len(orders)]:
            samples[name].append(check.elapsed(
                torch, captured[name][0], stream, args.iterations,
            ) / args.calls)
    report["timing"] = {
        "scope": "BF16 input through BF16 combine; native replaces InitRouting; "
                 "retained native and CANN include InitRouting; no queued clones or input copies",
        "timed_route_case": args.timed_route, "counts": metadata[0].tolist(),
        "orders": [orders[index % 6] for index in range(args.timing_samples)],
        "samples_us_per_call": samples,
        "summary_us_per_call": {name: timing_summary(values) for name, values in samples.items()},
    }
    report["timing"]["last_outputs"] = {
        name: check.compare(torch, captured[name][1], captured["vendor"][1], 0, 0)
        for name in namespaces
    }
    report["pass"] = all(
        value["pass"] and value["bitwise_equal"] is True
        for value in report["timing"]["last_outputs"].values()
    )
    return report


def main():
    args = arguments()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(check.ROOT))
    import torch
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    for path in (args.library, args.balanced_library, args.vendor_library):
        torch.ops.load_library(str(path.resolve()))
    source = check.load_weights(torch, args)
    weights = {name: torch_npu.npu_format_cast(value.to(args.device), 2)
               for name, value in source.items()}
    banks = [{name: torch_npu.npu_format_cast(weights[name].clone(), 29).view(torch.int32)
              for name in ("gate_up", "down")} for _ in range(args.banks)]
    for name in ("gate_up", "down"):
        assert len({bank[name].data_ptr() for bank in banks}) == args.banks
        assert all(torch_npu.get_npu_format(bank[name]) == 29 for bank in banks)
    encoded = {name: (source[name].view(torch.int32).long() & 0xFFFFFFFF).to(args.device)
               for name in ("gu_scale", "down_scale")}
    encoded["down_scale"] = encoded["down_scale"].unsqueeze(1)
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "cann": check.cann_versions(torch_npu),
        "device": str(torch.npu.get_device_properties(args.device)),
        "library_sha256": {str(path): check.digest(path)
                           for path in (args.library, args.balanced_library, args.vendor_library)},
        "source_sha256": {path.name: check.digest(path) for path in
                          (Path(__file__), HERE / "check.py")},
        "build_manifest": json.loads((args.library.parent / "build-manifest.json").read_text()),
        "banks": [{name: {"address": bank[name].data_ptr(),
                          "format": torch_npu.get_npu_format(bank[name]),
                          "bytes": bank[name].untyped_storage().nbytes()}
                   for name in bank} for bank in banks],
        "cases": [], "pass": False,
    }
    if args.checkpoint is not None:
        report["checkpoint_index_sha256"] = check.digest(
            args.checkpoint / "quant_model_weights.safetensors.index.json",
        )
    if args.fixture is not None:
        report["fixture_sha256"] = check.digest(args.fixture)
    try:
        with torch.inference_mode():
            for width in args.widths:
                result = run_width(torch, torch_npu, args, weights, banks, encoded, width)
                report["cases"].append(result)
                if not result["pass"]:
                    break
        report["pass"] = all(case["pass"] for case in report["cases"])
    finally:
        report["loaded_libraries_sha256"] = check.loaded_libraries()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": report["pass"], "report": str(args.output)}))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
