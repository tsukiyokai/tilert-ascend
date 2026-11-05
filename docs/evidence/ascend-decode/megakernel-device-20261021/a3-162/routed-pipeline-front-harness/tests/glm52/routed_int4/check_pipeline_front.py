"""Compare routed input-front composition with retained shared interleaving.

Each arm quantizes a separate shared BF16 input through DynamicQuant. Native
and reference routed banks share NZ allocations; shared projections retain
their native ND tiles and equal-valued reference NZ allocations.
"""

import argparse
import ctypes
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys

import check
import check_input_front as front
import check_pipeline as pipeline


HERE = Path(__file__).resolve().parent
HIDDEN, TOP_K, EXPERTS = check.HIDDEN, check.TOP_K, check.EXPERTS


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--library", type=Path,
                        default=HERE / "build-pipeline-front/routed_int4_torch.so")
    parser.add_argument("--baseline-library", type=Path,
                        default=HERE / "build-pipeline-zero/routed_int4_torch.so")
    parser.add_argument("--vendor-library", type=Path,
                        default=HERE / "build-reference/nz_reference.so")
    parser.add_argument("--shared-library", type=Path,
                        default=HERE.parent / "shared_w8/build-grouped/shared_w8.so")
    parser.add_argument("--swiglu-library", type=Path,
                        default=HERE.parent / "shared_w8/build-swiglu-reference-v4/shared_swiglu_reference.so")
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 12, 32])
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--banks", type=int, default=1)
    parser.add_argument("--calls", type=int, default=1)
    parser.add_argument("--timed-route", type=int, default=3)
    parser.add_argument("--timing-samples", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--shared-scale-dtype", choices=("float32", "bfloat16"),
                        default="float32")
    parser.add_argument("--engine-adapter", action="store_true")
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
    args.fixture = args.checkpoint = None
    return args


def input_case(torch, width, iteration, seed):
    hidden, choices, probabilities = check.route_case(torch, width, iteration, seed)
    shared, _, _ = check.route_case(torch, width, iteration + 3, seed + 100000)
    return hidden, shared, choices, probabilities


def shared_staged_input(torch, payload, width, capacity):
    return payload[:6 * capacity * 1024].view(torch.int8).reshape(
        6, capacity, 1024,
    )[:, :width].permute(1, 0, 2).reshape(width, HIDDEN)


def shared_down_products(torch, payload, offset, width, capacity):
    return payload[offset:offset + capacity * HIDDEN * 4].view(torch.int32).reshape(
        24, capacity, 256,
    )[:, :width].permute(1, 0, 2).reshape(width, HIDDEN)


def snapshot(value):
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [snapshot(item) for item in value]
    return value.detach().cpu().clone()


def run_width(torch, torch_npu, args, weights, banks, shared_lib, swiglu, width):
    def nd(value):
        return torch_npu.npu_format_cast(value.to(args.device).contiguous(), 2)

    hidden = nd(torch.empty(width, HIDDEN, dtype=torch.bfloat16))
    shared_hidden = nd(torch.empty_like(hidden, device="cpu"))
    choices = nd(torch.empty(width, TOP_K,
                            dtype=torch.int64 if args.engine_adapter else torch.int32))
    probabilities = nd(torch.empty(width, TOP_K))
    sample = nd(torch.empty(width, HIDDEN, dtype=torch.int8))
    capacity = 16 if width <= 16 else 32
    layout = [shared_lib.shared_w8_workspace_field(capacity, i) for i in range(7)]
    namespaces = {"interleaved_front": torch.ops.tilert_pipeline_front,
                  "interleaved_dispatch": torch.ops.tilert_pipeline}
    arenas = {}
    adapter_pool = None
    if args.engine_adapter:
        from tilert.models.glm_5_2_ascend import routed_shared as adapter

        adapter_pool = adapter.WorkspacePool(args.blocks, input_front=True)
        adapter_pool.reserve_capture(sample)
        arenas["interleaved_front"] = next(iter(adapter_pool.capture_reserves.values()))
    for name, namespace in namespaces.items():
        if name not in arenas:
            arenas[name] = namespace.routed_shared_prepare(
                sample, EXPERTS, args.blocks, weight_layout=1,
            )
    reference_codes = nd(torch.empty(width, 128, dtype=torch.int8))
    reference_scales = nd(torch.empty(width))
    stream = torch.npu.Stream(device=args.device)
    stream.wait_stream(torch.npu.current_stream(args.device))
    sequence = banks * (args.calls // args.banks)

    def dispatch(x, ids):
        return torch_npu.npu_moe_init_routing_v2(
            x, ids.to(torch.int32), active_num=width * TOP_K, expert_num=EXPERTS,
            expert_tokens_num_type=1, expert_tokens_num_flag=True,
            active_expert_range=[0, EXPERTS], quant_mode=1,
        )

    def native_call(name, selected, queued=False, details=None):
        scratch, sync = arenas[name]
        namespace = namespaces[name]
        for bank in selected:
            x, sx, ids, probs = (hidden, shared_hidden, choices, probabilities)
            if queued:
                x, sx, ids, probs = [value.clone() for value in (x, sx, ids, probs)]
            shared = bank["shared"]
            projections = [bank["gate_up"], weights["gu_bias"], weights["gu_scale"],
                           bank["down"], weights["down_bias"], weights["down_scale"]]
            if args.engine_adapter and name == "interleaved_front":
                outputs = adapter.run_body(
                    x, probs, ids, sx, projections,
                    [shared["gate_tiles"], shared["gate_scale"],
                     shared["down_tiles"], shared["down_scale"]], adapter_pool,
                )[:2]
                if details is not None:
                    codes, scale = torch_npu.npu_dynamic_quant(sx)
                    details.update(shared_codes=codes, shared_scale=scale)
                continue
            shared_codes, shared_scale = torch_npu.npu_dynamic_quant(sx)
            shared_args = [shared_codes, shared_scale.reshape(-1),
                           shared["gate_tiles"], shared["gate_scale"],
                           shared["down_tiles"], shared["down_scale"]]
            if name == "interleaved_front":
                outputs = namespace.routed_shared_from_bf16(
                    x, *projections, ids, probs, shared_args, scratch, sync,
                    EXPERTS, args.blocks, weight_layout=1,
                )
            else:
                rows, inverse, counts, scale = dispatch(x, ids)
                outputs = namespace.routed_shared_from_dispatch(
                    rows, *projections, scale.reshape(-1), inverse, counts, probs,
                    shared_args, scratch, sync, EXPERTS, args.blocks, weight_layout=1,
                )
            if details is not None:
                details.update(shared_codes=shared_codes, shared_scale=shared_scale)
        return outputs

    def vendor_call(selected, details=None):
        for bank in selected:
            shared_codes, shared_scale = torch_npu.npu_dynamic_quant(shared_hidden)
            rows, inverse, counts, scale = dispatch(hidden, choices)
            counts64 = counts.long()
            codes, middle_scale = torch.ops.tilert_reference.nz_gmm1(
                rows, bank["gate_up"], weights["vendor_gu_scale"], scale,
                counts64, weights["gu_bias"],
            )
            saved = codes.clone() if details is not None else codes
            routes = torch.ops.tilert_reference.nz_gmm2(
                codes, bank["down"], weights["vendor_down_scale"], middle_scale,
                counts64, weights["down_bias"],
            )
            routed = torch_npu.npu_moe_token_unpermute(
                routes, inverse, probs=probabilities.bfloat16(),
            )
            shared = bank["shared"]
            product = torch_npu.npu_quant_matmul(
                shared_codes, shared["gate_nz"], shared["vendor_gate_scale"],
                output_dtype=torch.int32,
            )
            addresses = [product, shared["gate_scale"], shared_scale,
                         reference_codes, reference_scales]
            swiglu.shared_swiglu_reference_launch(
                min(width, 48), torch.npu.current_stream(args.device).npu_stream,
                *(value.data_ptr() for value in addresses), width,
            )
            shared_output = torch_npu.npu_quant_matmul(
                reference_codes, shared["down_nz"], shared["vendor_down_scale"],
                pertoken_scale=reference_scales, output_dtype=torch.bfloat16,
            )
            if details is not None:
                details.update(rows=rows, inverse=inverse, scales=scale, codes=saved,
                               middle_scale=middle_scale, routes=routes,
                               shared_input=shared_codes, shared_input_scale=shared_scale,
                               shared=dict(gate=product, codes=reference_codes,
                                           scale=reference_scales, output=shared_output))
        return routed, shared_output

    def update(iteration):
        x, sx, ids, probs = input_case(torch, width, iteration, args.seed)
        with torch.npu.stream(stream):
            hidden.copy_(x)
            shared_hidden.copy_(sx)
            choices.copy_(ids)
            probabilities.copy_(probs)
        stream.synchronize()
        return front.metadata_reference(torch, ids, probs)

    def capture_arms(selected, details=None, producers=None):
        calls = {name: (lambda name=name: native_call(
            name, selected, details=None if producers is None else producers[name],
        )) for name in namespaces}
        calls["vendor"] = lambda: vendor_call(selected, details)
        return {name: check.capture(
                    torch, call, stream,
                    0 if args.engine_adapter and name == "interleaved_front" else args.warmup,
                )
                for name, call in calls.items()}

    update(0)
    captures = []
    for bank_index, bank in enumerate(banks):
        if args.engine_adapter:
            for _ in range(args.warmup):
                native_call("interleaved_front", [bank])
            torch.npu.current_stream(args.device).synchronize()
        details = {}
        producers = {name: {} for name in namespaces}
        captures.append((bank_index, bank, capture_arms([bank], details, producers),
                         details, producers))
    if args.engine_adapter:
        with torch.npu.stream(stream):
            captured_scratch, _ = adapter_pool.get(sample)
        if (adapter_pool.capture_reserves
                or captured_scratch[0].data_ptr() != arenas["interleaved_front"][0][0].data_ptr()):
            raise RuntimeError("Capture did not claim the reserved adapter workspace")
    report = {"width": width, "checks": [], "pass": True,
              "distinct_shared_input": True,
              "topk_ids_dtype": str(choices.dtype),
              "scratch_bytes": {name: sum(t.nbytes for t in arena[0])
                                for name, arena in arenas.items()}}
    for iteration in range(args.replays):
        metadata = update(iteration)
        for bank_index, bank, graphs, details, producers in captures:
            with torch.npu.stream(stream):
                for graph, _ in graphs.values():
                    graph.replay()
            stream.synchronize()
            candidate, control = [arenas[name][0] for name in namespaces]
            inverse = details["inverse"].long()
            boundaries = {}

            def compare(name, actual, expected):
                boundaries[name] = check.compare(torch, actual, expected, 0, 0)

            for name in namespaces:
                scratch = arenas[name][0]
                compare(name + "/routed_output", graphs[name][1][0], graphs["vendor"][1][0])
                compare(name + "/input_codes", scratch[10].repeat_interleave(TOP_K, 0),
                        details["rows"].index_select(0, inverse))
                compare(name + "/input_scale", scratch[11].repeat_interleave(TOP_K, 0),
                        details["scales"].index_select(0, inverse).reshape(-1))
                codes, scales = check.middle_stages(torch, scratch, scratch[12:17], width)
                compare(name + "/middle_codes", codes, details["codes"].index_select(0, inverse))
                compare(name + "/middle_scale", scales, details["middle_scale"].index_select(0, inverse))
                compare(name + "/routes", scratch[5].reshape(-1, HIDDEN),
                        details["routes"].index_select(0, inverse))
                compare(name + "/shared_staged_input", shared_staged_input(
                    torch, scratch[17], width, capacity,
                ), details["shared_input"])
                compare(name + "/shared_input_scale", producers[name]["shared_scale"],
                        details["shared_input_scale"])
                for stage, value in pipeline.shared_boundaries(
                        torch, scratch[17], layout, width, capacity, graphs[name][1][1]).items():
                    compare(name + "/shared_" + stage, value, details["shared"][stage])
            for name, actual, expected in zip(
                    ("counts", "starts", "source", "slot", "probability"),
                    candidate[12:17], metadata):
                compare(name, actual, expected)
            for name, index in (("stacked", 0), ("gate_int32", 1),
                                ("middle_packed", 2), ("down_int32", 4)):
                compare(name, check.ordered_partials(
                    torch, candidate[index], candidate[12:17], width, 2,
                ), check.ordered_partials(
                    torch, control[index], control[12:17], width, 2,
                ))
            offset, elements = layout[1], 6 * 256 * capacity
            compare("shared_gate_partials",
                    candidate[17][offset:offset + elements * 4].view(torch.int32),
                    control[17][offset:offset + elements * 4].view(torch.int32))
            compare("shared_down_int32", shared_down_products(
                torch, candidate[17], layout[4], width, capacity,
            ), shared_down_products(torch, control[17], layout[4], width, capacity))
            with torch.npu.stream(stream):
                queued = native_call("interleaved_front", [bank], queued=True)
            stream.synchronize()
            for index, stage in enumerate(("routed", "shared")):
                compare("queued/" + stage, queued[index], graphs["vendor"][1][index])
            passed = all(value["pass"] and value["bitwise_equal"] is True
                         for value in boundaries.values())
            report["checks"].append({
                "iteration": iteration, "bank": bank_index, "boundaries": boundaries,
                "active_experts": int((metadata[0] > 0).sum()), "pass": passed,
            })
            if not passed:
                failure = args.output.with_name(
                    args.output.stem + f"-m{width}-r{iteration}-b{bank_index}.pt",
                )
                torch.save(snapshot({"hidden": hidden, "shared_hidden": shared_hidden,
                    "choices": choices, "probabilities": probabilities,
                    "candidate_scratch": candidate, "control_scratch": control,
                    "vendor": details, "vendor_outputs": graphs["vendor"][1]}), failure)
                report.update({"pass": False, "failure": str(failure)})
                return report
    if args.no_timing:
        return report
    metadata = update(args.timed_route)
    graphs = capture_arms(sequence)
    orders = list(itertools.permutations(graphs))
    samples = {name: [] for name in graphs}
    for sample_index in range(args.timing_samples):
        for name in orders[sample_index % len(orders)]:
            samples[name].append(check.elapsed(
                torch, graphs[name][0], stream, args.iterations,
            ) / args.calls)
    outputs = {f"{name}/{stage}": check.compare(
        torch, graphs[name][1][index], graphs["vendor"][1][index], 0, 0,
    ) for name in namespaces for index, stage in enumerate(("routed", "shared"))}
    report["timing"] = {
        "scope": "independent routed/shared BF16 inputs through two BF16 outputs; "
                 "shared DynamicQuant included in every arm; no TP reduce or final add",
        "timed_route_case": args.timed_route, "counts": metadata[0].tolist(),
        "shared_route_case_offset": 3,
        "input_nonzero": {"routed": int(hidden.cpu().count_nonzero()),
                          "shared": int(shared_hidden.cpu().count_nonzero())},
        "orders": [orders[index % 6] for index in range(args.timing_samples)],
        "samples_us_per_call": samples, "last_outputs": outputs,
        "summary_us_per_call": {name: front.timing_summary(values)
                                for name, values in samples.items()},
    }
    report["pass"] = all(value["pass"] and value["bitwise_equal"] is True
                         for value in outputs.values())
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
    for path in (args.library, args.baseline_library, args.vendor_library):
        torch.ops.load_library(str(path.resolve()))
    shared_lib = pipeline.shared_library(args.shared_library)
    swiglu = ctypes.CDLL(str(args.swiglu_library.resolve()))
    swiglu.shared_swiglu_reference_launch.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 5 + [ctypes.c_int64])
    swiglu.shared_swiglu_reference_launch.restype = None

    def nd(value):
        return torch_npu.npu_format_cast(value.to(args.device).contiguous(), 2)

    source = check.load_weights(torch, args)
    weights = {name: nd(value) for name, value in source.items()}
    for name in ("gu_scale", "down_scale"):
        weights["vendor_" + name] = nd(source[name].view(torch.int32).long() & 0xFFFFFFFF)
    weights["vendor_down_scale"] = weights["vendor_down_scale"].unsqueeze(1)
    shared = pipeline.shared_weights(torch, nd, torch.Generator().manual_seed(args.seed + 17),
                                     scale_dtype=args.shared_scale_dtype)
    banks = []
    for _ in range(args.banks):
        bank = {name: torch_npu.npu_format_cast(weights[name].clone(), 29).view(torch.int32)
                for name in ("gate_up", "down")}
        bank["shared"] = {name: shared[name].clone() for name in (
            "gate_tiles", "down_tiles", "gate_scale", "down_scale",
            "vendor_gate_scale", "vendor_down_scale",
        )}
        for name in ("gate", "down"):
            bank["shared"][name + "_nz"] = torch_npu.npu_format_cast(shared[name].clone(), 29)
        banks.append(bank)
    for name in ("gate_up", "down"):
        assert len({bank[name].data_ptr() for bank in banks}) == args.banks
        assert all(torch_npu.get_npu_format(bank[name]) == 29 for bank in banks)
    for name in ("gate_tiles", "down_tiles", "gate_nz", "down_nz"):
        assert len({bank["shared"][name].data_ptr() for bank in banks}) == args.banks
    libraries = (args.library, args.baseline_library, args.vendor_library,
                 args.shared_library, args.swiglu_library)
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "cann": check.cann_versions(torch_npu),
        "device": str(torch.npu.get_device_properties(args.device)),
        "library_sha256": {str(path): check.digest(path) for path in libraries},
        "source_sha256": {path.name: check.digest(path) for path in
                          (Path(__file__), HERE / "check.py", HERE / "check_input_front.py",
                           HERE / "check_pipeline.py")},
        "build_manifest": json.loads((args.library.parent / "build-manifest.json").read_text()),
        "banks": [{name: {"address": value.data_ptr(),
                          "format": torch_npu.get_npu_format(value),
                          "bytes": value.untyped_storage().nbytes()}
                   for name, value in {**{k: bank[k] for k in ("gate_up", "down")},
                                       **bank["shared"]}.items()} for bank in banks],
        "cases": [], "pass": False,
    }
    if args.engine_adapter:
        from tilert.models.glm_5_2_ascend import routed_expert, routed_shared

        report["adapter_scope"] = "run_body and capture workspace ownership; no vLLM runner or TP finalize"
        report["source_sha256"].update({
            Path(module.__file__).name: check.digest(Path(module.__file__))
            for module in (routed_expert, routed_shared)
        })
    try:
        with torch.inference_mode():
            for width in args.widths:
                result = run_width(torch, torch_npu, args, weights, banks,
                                   shared_lib, swiglu, width)
                report["cases"].append(result)
                if not result["pass"]:
                    break
        report["pass"] = all(case["pass"] for case in report["cases"])
    finally:
        report["loaded_libraries_sha256"] = check.loaded_libraries()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": report["pass"], "report": str(args.output)}))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
