"""Compare routed chains with fixed routes and configurable NZ weight reuse.

The independent calls share inputs and do not form dependent decoder layers.
Native and stock CANN arms borrow the same immutable NZ weight allocations.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from check import (
    EXPERTS, HIDDEN, INTERMEDIATE, TOP_K, cann_versions, capture, compare,
    digest, elapsed, load_weights, loaded_libraries, route_case,
)


def arguments():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--native-partial", choices=("int32", "fp16"), default="int32")
    parser.add_argument("--native-schedule", choices=("expert", "row"), default="expert")
    parser.add_argument("--native-cube", choices=("norm", "gu-mdl"), default="norm")
    comparisons = parser.add_mutually_exclusive_group()
    comparisons.add_argument("--compare-native", action="store_true",
                             help="compare retained INT32, current INT32 control and candidate")
    comparisons.add_argument("--compare-balanced", action="store_true",
                             help="compare GU MDL, retained row schedule and CANN NZ")
    parser.add_argument("--balanced-library", type=Path,
                        default=here / "build-balanced-zero/routed_int4_torch.so")
    parser.add_argument("--original-library", type=Path,
                        default=here / "build-route/routed_int4_torch.so")
    parser.add_argument("--control-library", type=Path,
                        default=here / "build-control/routed_int4_torch.so")
    parser.add_argument("--vendor-library", type=Path,
                        default=here / "build-reference/nz_reference.so")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fixture", type=Path)
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--rank", type=int, choices=range(16), default=0)
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--banks", type=int, default=4)
    parser.add_argument("--calls", type=int, default=16)
    parser.add_argument("--reuse-bank", action="store_true")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--timing-pairs", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.library is None:
        directory = ("build-gu-mdl" if args.native_cube == "gu-mdl"
                     else "build-balanced" if args.native_schedule == "row"
                     else "build-fixpipe" if args.native_partial == "fp16"
                     else "build-route")
        args.library = here / directory / "routed_int4_torch.so"
    if args.native_schedule == "row" and args.native_partial != "int32":
        parser.error("row scheduling requires INT32 partials")
    if args.native_cube == "gu-mdl" and (
            args.native_schedule != "row" or args.native_partial != "int32"
            or args.compare_native):
        parser.error("gu-mdl requires the INT32 row schedule; use --compare-balanced")
    if args.compare_balanced and args.native_cube != "gu-mdl":
        parser.error("--compare-balanced requires --native-cube gu-mdl")
    if (args.compare_native and args.native_partial != "fp16"
            and args.native_schedule != "row"):
        parser.error("--compare-native requires FP16 partials or row scheduling")
    if (not 1 <= args.rows <= 32 or not 1 <= args.blocks <= 24
            or not 1 <= args.banks <= 8 or not 1 <= args.calls <= 64
            or args.calls % args.banks):
        parser.error("invalid rows, blocks, banks or calls; calls must be a multiple of banks")
    if min(args.iterations, args.timing_pairs) < 1:
        parser.error("timing counts must be positive")
    return args


def main():
    args = arguments()
    import torch
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    torch.ops.load_library(str(args.library.resolve()))
    torch.ops.load_library(str(args.vendor_library.resolve()))
    if args.compare_native:
        torch.ops.load_library(str(args.original_library.resolve()))
        torch.ops.load_library(str(args.control_library.resolve()))
    if args.compare_balanced:
        torch.ops.load_library(str(args.balanced_library.resolve()))

    def nd(value):
        return torch_npu.npu_format_cast(value.to(args.device).contiguous(), 2)

    source = load_weights(torch, args)
    weights = {name: nd(value) for name, value in source.items()}
    unique_banks = 1 if args.reuse_bank else args.banks
    banks = [{name: torch_npu.npu_format_cast(weights[name], 29)
              .view(torch.int32).contiguous() for name in ("gate_up", "down")}
             for _ in range(unique_banks)]
    for name in ("gate_up", "down"):
        if len({bank[name].data_ptr() for bank in banks}) != unique_banks:
            raise RuntimeError("NZ banks do not own distinct addresses")
        if any(torch_npu.get_npu_format(bank[name]) != 29 for bank in banks):
            raise RuntimeError("NZ bank lost its physical format")
    vendor_scales = {}
    for name in ("gu_scale", "down_scale"):
        bits = source[name].view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        vendor_scales[name] = nd(bits)
    vendor_scales["down_scale"] = vendor_scales["down_scale"].unsqueeze(1)
    sequence = banks * (args.calls // unique_banks)
    hidden = nd(torch.empty(args.rows, HIDDEN, dtype=torch.bfloat16))
    choices = nd(torch.empty(args.rows, TOP_K, dtype=torch.int32))
    probabilities = nd(torch.empty(args.rows, TOP_K, dtype=torch.float32))
    sample = nd(torch.empty(args.rows, HIDDEN, dtype=torch.int8))

    def dispatch():
        return torch_npu.npu_moe_init_routing_v2(
            hidden, choices, active_num=args.rows * TOP_K, expert_num=EXPERTS,
            expert_tokens_num_type=1, expert_tokens_num_flag=True,
            active_expert_range=[0, EXPERTS], quant_mode=1,
        )

    def native_arm(namespace, partial):
        scratch, sync = namespace.routed_experts_prepare(
            sample, EXPERTS, args.blocks, weight_layout=1, from_dispatch=True,
        )
        gu_scale = (vendor_scales["gu_scale"] if partial == "fp16"
                    else weights["gu_scale"])
        down_scale = (vendor_scales["down_scale"].squeeze(1) if partial == "fp16"
                      else weights["down_scale"])

        def call():
            result = None
            for bank in sequence:
                packed, inverse, counts, scale = dispatch()
                result = namespace.routed_experts_from_dispatch(
                    packed, bank["gate_up"], weights["gu_bias"], gu_scale,
                    bank["down"], weights["down_bias"], down_scale,
                    scale.reshape(-1), inverse, counts, probabilities, scratch,
                    sync, EXPERTS, args.blocks, weight_layout=1,
                )
            return result

        return call, scratch

    namespace = (torch.ops.tilert_mdl if args.native_cube == "gu-mdl"
                 else torch.ops.tilert_balanced if args.native_schedule == "row"
                 else torch.ops.tilert_fixpipe if args.native_partial == "fp16"
                 else torch.ops.tilert)
    selected, scratch = native_arm(namespace, args.native_partial)
    calls = {"native": selected}
    scratch_owners = {"native": scratch}
    if args.compare_native:
        candidate = ("balanced_int32" if args.native_schedule == "row"
                     else "fixpipe_fp16")
        calls = {candidate: selected}
        scratch_owners = {candidate: scratch}
        for name, operations in (("original_int32", torch.ops.tilert),
                                 ("control_int32", torch.ops.tilert_control)):
            calls[name], scratch_owners[name] = native_arm(operations, "int32")
    if args.compare_balanced:
        calls = {"gu_mdl": selected}
        scratch_owners = {"gu_mdl": scratch}
        calls["balanced_norm"], scratch_owners["balanced_norm"] = native_arm(
            torch.ops.tilert_balanced, "int32",
        )

    def vendor():
        result = None
        for bank in sequence:
            packed, inverse, counts, scale = dispatch()
            counts = counts.to(torch.int64)
            codes, token_scale = torch.ops.tilert_reference.nz_gmm1(
                packed, bank["gate_up"], vendor_scales["gu_scale"], scale,
                counts, weights["gu_bias"],
            )
            down = torch.ops.tilert_reference.nz_gmm2(
                codes, bank["down"], vendor_scales["down_scale"], token_scale,
                counts, weights["down_bias"],
            )
            result = torch_npu.npu_moe_token_unpermute(
                down, inverse, probs=probabilities.to(torch.bfloat16),
            )
        return result

    stream = torch.npu.Stream(device=args.device)
    stream.wait_stream(torch.npu.current_stream(args.device))

    def update(replay):
        rows, ids, probs = route_case(torch, args.rows, replay, args.seed)
        with torch.npu.stream(stream):
            hidden.copy_(rows)
            choices.copy_(ids)
            probabilities.copy_(probs)
        stream.synchronize()
        return ids.unique().numel()

    update(0)
    captured = {name: capture(torch, call, stream, 3)
                for name, call in calls.items()}
    vendor_graph, vendor_output = capture(torch, vendor, stream, 3)
    checks = []
    for replay in range(4):
        active = update(replay)
        with torch.npu.stream(stream):
            order = tuple(graph.replay for graph, _ in captured.values())
            order += (vendor_graph.replay,)
            for call in order[::1 if replay % 2 == 0 else -1]:
                call()
        stream.synchronize()
        arm_checks = {name: compare(torch, output, vendor_output, 0.0, 0.0)
                      for name, (_, output) in captured.items()}
        passed = all(result["pass"] and result["bitwise_equal"]
                     for result in arm_checks.values())
        check = next(iter(arm_checks.values()))
        checks.append({"replay": replay, "active_experts": active, **check,
                       "arms": arm_checks, "pass": passed})
        if not passed:
            break
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "scope": "independent routed chains with configurable weight reuse",
        "rows": args.rows, "blocks": args.blocks, "calls": args.calls,
        "seed": args.seed, "warmup": 3,
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "unique_weight_banks": unique_banks,
        "shared_weight_allocations_between_arms": True,
        "weight_bytes": sum(bank[name].untyped_storage().nbytes()
                            for bank in banks for name in ("gate_up", "down")),
        "minimum_selected_weight_bytes": (unique_banks * active * HIDDEN *
                                          INTERMEDIATE * 3 // 2),
        "scratch_bytes": sum(value.nbytes for value in scratch),
        "scratch_bytes_per_arm": {name: sum(value.nbytes for value in owner)
                                  for name, owner in scratch_owners.items()},
        "input_copies_in_capture": False,
        "dispatch_in_each_call": True,
        "last_output_checks": checks,
        "pass": len(checks) == 4 and all(check["pass"] and check["bitwise_equal"]
                                        for check in checks),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "cann": cann_versions(torch_npu),
        "device": str(torch.npu.get_device_properties(args.device)),
        "library_sha256": digest(args.library),
        "comparison_library_sha256": {
            name: digest(path) for name, path in
            (("original_int32", args.original_library),
             ("control_int32", args.control_library))
        } if args.compare_native else None,
        "balanced_library_sha256": (digest(args.balanced_library)
                                    if args.compare_balanced else None),
        "reference_library_sha256": digest(args.vendor_library),
        "harness_sha256": digest(Path(__file__)),
        "helpers_sha256": digest(Path(__file__).with_name("check.py")),
        "reference": "stock CANN canonical NZ descriptor bridge",
    }
    if report["pass"]:
        pairs = []
        timing_graphs = {name: graph for name, (graph, _) in captured.items()}
        if not args.compare_native:
            timing_graphs["vendor"] = vendor_graph
        for pair in range(args.timing_pairs):
            order = tuple(timing_graphs.items())
            # Rotate all arms through each position; reverse alternate cycles.
            offset = pair % len(order)
            order = order[offset:] + order[:offset]
            if len(order) > 2 and (pair // len(order)) % 2:
                order = order[::-1]
            pairs.append({name: elapsed(torch, graph, stream, args.iterations)
                          for name, graph in order})
        report["iterations"] = args.iterations
        report["pairs_us_per_graph"] = pairs
        report["median_us_per_call"] = {
            name: statistics.median(pair[name] for pair in pairs) / args.calls
            for name in timing_graphs
        }
    report["loaded_libraries_sha256"] = loaded_libraries()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
