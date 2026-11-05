"""Compare interleaved routed/shared execution with separate native and CANN chains.

Each arm includes the same BF16 producers: vendor dispatch for routed experts
and standalone DynamicQuant for the shared expert. Weight banks contain equal
values at distinct addresses; repeated calls do not form dependent layers.
"""

import argparse
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from check import (
    EXPERTS, HIDDEN, TOP_K, cann_versions, capture, compare, digest, elapsed,
    load_weights, loaded_libraries, middle_stages, route_case,
)


HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "shared_w8"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path,
                        default=HERE / "build-pipeline/routed_int4_torch.so")
    parser.add_argument("--routed-library", type=Path,
                        default=HERE / "build-balanced/routed_int4_torch.so")
    parser.add_argument("--shared-library", type=Path,
                        default=SHARED / "build-grouped/shared_w8.so")
    parser.add_argument("--vendor-library", type=Path,
                        default=HERE / "build-reference/nz_reference.so")
    parser.add_argument("--swiglu-library", type=Path,
                        default=SHARED / "build-swiglu-reference-v4/shared_swiglu_reference.so")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 12, 32])
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--banks", type=int, default=1)
    parser.add_argument("--calls", type=int, default=1)
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--timing-pairs", type=int, default=6)
    parser.add_argument("--queued-producer", action="store_true")
    parser.add_argument("--engine-adapter", action="store_true",
                        help="Call the TileRT body and capture workspace pool")
    parser.add_argument("--shared-scale-dtype",
                        choices=("float32", "bfloat16", "float16"),
                        default="float32")
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (any(row not in range(1, 33) for row in args.widths)
            or args.blocks not in range(1, 25)
            or args.banks not in range(1, 9)
            or args.calls not in range(1, 65) or args.calls % args.banks):
        parser.error("invalid width, blocks, banks or calls")
    if args.replays < 4 or min(args.iterations, args.timing_pairs) < 1:
        parser.error("use at least four changing inputs and positive timing counts")
    # Reuse the routed synthetic-fixture generator without checkpoint loading.
    args.fixture = args.checkpoint = None
    return args


def shared_library(path):
    lib = ctypes.CDLL(str(path.resolve()))
    lib.shared_w8_prepare.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                     ctypes.POINTER(ctypes.c_uint64)]
    lib.shared_w8_prepare.restype = ctypes.c_int
    lib.shared_w8_workspace_field.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.shared_w8_workspace_field.restype = ctypes.c_uint64
    lib.shared_w8_launch.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint64]
        + [ctypes.c_void_p] * 9 + [ctypes.c_int64] * 2)
    lib.shared_w8_launch.restype = None
    if not lib.shared_w8_grouped_quant():
        raise ValueError("shared reference arm requires the grouped quant contract")
    return lib


def shared_weights(torch, nd, generator, engine_adapter=False,
                   scale_dtype="float32"):
    gate = torch.randint(-128, 128, (HIDDEN, 256), dtype=torch.int8,
                         generator=generator)
    down = torch.randint(-128, 128, (128, HIDDEN), dtype=torch.int8,
                         generator=generator)
    if engine_adapter:
        from tilert.models.glm_5_2_ascend.routed_shared import pack_shared_weights

        gate_tiles, down_tiles = pack_shared_weights(gate, down)
    else:
        gate_tiles = gate.reshape(6, 1024, 4, 64).permute(0, 2, 1, 3)
        gate_tiles = gate_tiles.contiguous().reshape(24, 1024, 64)
        down_tiles = down.reshape(128, 24, 256).permute(1, 0, 2).contiguous()
    dtype = getattr(torch, scale_dtype)
    gate_scale = nd((torch.rand(256, generator=generator) * 0.001 + 0.0005).to(dtype))
    down_scale = nd((torch.rand(HIDDEN, generator=generator) * 0.001 + 0.0005).to(dtype))
    return {
        "gate": nd(gate), "down": nd(down),
        "gate_tiles": nd(gate_tiles), "down_tiles": nd(down_tiles),
        "gate_scale": gate_scale.float(), "down_scale": down_scale.float(),
        "vendor_gate_scale": gate_scale, "vendor_down_scale": down_scale,
    }


def shared_boundaries(torch, payload, layout, rows, capacity, output):
    partial = payload[layout[1]:layout[1] + 6 * 256 * capacity * 4]
    partial = partial.cpu().view(torch.int32).reshape(6, 4, capacity, 64)
    gate = partial.sum(0).permute(1, 0, 2).reshape(capacity, 256).int()[:rows]
    codes = payload[layout[2]:layout[2] + capacity * 128]
    codes = codes.view(torch.int8).reshape(capacity, 128)[:rows]
    scale = payload[layout[3]:layout[3] + capacity * 8 * 4]
    scale = scale.view(torch.float32).reshape(capacity, 8)[:rows, 0]
    return dict(gate=gate, codes=codes, scale=scale, output=output)


def run_case(torch, torch_npu, args, weights, banks, lib, swiglu, rows):
    def nd(value):
        return torch_npu.npu_format_cast(value.to(args.device).contiguous(), 2)

    capacity = 16 if rows <= 16 else 32
    layout = [lib.shared_w8_workspace_field(capacity, field) for field in range(7)]
    tiling_host = ctypes.create_string_buffer(1024)
    shared_sync = ctypes.c_uint64()
    status = lib.shared_w8_prepare(capacity, args.blocks, tiling_host,
                                   ctypes.byref(shared_sync))
    if status:
        raise RuntimeError(f"shared prepare failed: {status}")
    shared_tiling = nd(torch.frombuffer(bytearray(tiling_host.raw), dtype=torch.uint8))
    shared_scratch = nd(torch.empty(layout[-1], dtype=torch.uint8))
    shared_output = nd(torch.empty(rows, HIDDEN, dtype=torch.bfloat16))
    reference_codes = nd(torch.empty(rows, 128, dtype=torch.int8))
    reference_scales = nd(torch.empty(rows, dtype=torch.float32))
    live = nd(torch.empty(rows, HIDDEN, dtype=torch.bfloat16))
    choices = nd(torch.empty(rows, TOP_K, dtype=torch.int32))
    probabilities = nd(torch.empty(rows, TOP_K))
    sample = nd(torch.empty(rows, HIDDEN, dtype=torch.int8))
    if args.engine_adapter:
        from tilert.models.glm_5_2_ascend import routed_shared as adapter

        adapter_pool = adapter.WorkspacePool(args.blocks, input_front=False)
        adapter_pool.reserve_capture(sample)
        fused_scratch, fused_sync = next(iter(adapter_pool.capture_reserves.values()))
    else:
        fused_scratch, fused_sync = torch.ops.tilert_pipeline.routed_shared_prepare(
            sample, EXPERTS, args.blocks, weight_layout=1)
    routed_scratch, routed_sync = torch.ops.tilert_balanced.routed_experts_prepare(
        sample, EXPERTS, args.blocks, weight_layout=1, from_dispatch=True)
    details = {}

    def producer():
        hidden = live.clone() if args.queued_producer else live
        shared_input, shared_scale = torch_npu.npu_dynamic_quant(hidden)
        dispatched = torch_npu.npu_moe_init_routing_v2(
            hidden, choices, active_num=rows * TOP_K, expert_num=EXPERTS,
            expert_tokens_num_type=1, expert_tokens_num_flag=True,
            active_expert_range=[0, EXPERTS], quant_mode=1)
        return shared_input, shared_scale.reshape(-1), dispatched

    def call(name, sequence, retain=False):
        outputs = None
        for bank in sequence:
            if name == "interleaved" and args.engine_adapter:
                hidden = live.clone() if args.queued_producer else live
                shared = bank["shared"]
                routed_weights = [bank["gate_up"], weights["gu_bias"],
                                  weights["gu_scale"], bank["down"],
                                  weights["down_bias"], weights["down_scale"]]
                shared_values = [shared["gate_tiles"], shared["gate_scale"],
                                 shared["down_tiles"], shared["down_scale"]]
                outputs = adapter.run_body(
                    hidden, probabilities, choices, hidden, routed_weights,
                    shared_values, adapter_pool,
                )[:2]
                continue
            shared_input, shared_scale, dispatched = producer()
            packed, inverse, counts, token_scale = dispatched
            routed_args = [packed, bank["gate_up"], weights["gu_bias"],
                           weights["gu_scale"], bank["down"], weights["down_bias"],
                           weights["down_scale"], token_scale.reshape(-1),
                           inverse, counts, probabilities]
            shared = bank["shared"]
            shared_args = [shared_input, shared_scale, shared["gate_tiles"],
                           shared["gate_scale"], shared["down_tiles"],
                           shared["down_scale"]]
            if name == "interleaved":
                outputs = torch.ops.tilert_pipeline.routed_shared_from_dispatch(
                    *routed_args, shared_args, fused_scratch, fused_sync,
                    EXPERTS, args.blocks, weight_layout=1)
            elif name == "separate_native":
                routed_output = torch.ops.tilert_balanced.routed_experts_from_dispatch(
                    *routed_args, routed_scratch, routed_sync,
                    EXPERTS, args.blocks, weight_layout=1)
                addresses = [*shared_args, shared_scratch, shared_tiling, shared_output]
                lib.shared_w8_launch(args.blocks,
                    torch.npu.current_stream(args.device).npu_stream, shared_sync.value,
                    *(value.data_ptr() for value in addresses), rows, capacity)
                outputs = routed_output, shared_output
            else:
                vendor_counts = counts.to(torch.int64)
                codes, scale = torch.ops.tilert_reference.nz_gmm1(
                    packed, bank["gate_up"], weights["vendor_gu_scale"], token_scale,
                    vendor_counts, weights["gu_bias"])
                saved_codes = codes.clone() if retain else codes
                down = torch.ops.tilert_reference.nz_gmm2(
                    codes, bank["down"], weights["vendor_down_scale"], scale,
                    vendor_counts, weights["down_bias"])
                routed_output = torch_npu.npu_moe_token_unpermute(
                    down, inverse, probs=probabilities.to(torch.bfloat16))
                product = torch_npu.npu_quant_matmul(
                    shared_input, shared["gate_nz"], shared["vendor_gate_scale"],
                    output_dtype=torch.int32)
                addresses = [product, shared["gate_scale"], shared_scale,
                             reference_codes, reference_scales]
                swiglu.shared_swiglu_reference_launch(
                    min(rows, 48), torch.npu.current_stream(args.device).npu_stream,
                    *(value.data_ptr() for value in addresses), rows)
                result = torch_npu.npu_quant_matmul(
                    reference_codes, shared["down_nz"], shared["vendor_down_scale"],
                    pertoken_scale=reference_scales, output_dtype=torch.bfloat16)
                outputs = routed_output, result
                if retain:
                    details.update(input=packed, input_scale=token_scale,
                                   inverse=inverse, codes=saved_codes, scale=scale,
                                   down=down, routed_output=routed_output,
                                   shared=dict(gate=product,
                                   codes=reference_codes, scale=reference_scales,
                                   output=result))
        return outputs

    stream = torch.npu.Stream(device=args.device)
    stream.wait_stream(torch.npu.current_stream(args.device))

    def update(replay):
        hidden, ids, probs = route_case(torch, rows, replay, args.seed)
        with torch.npu.stream(stream):
            live.copy_(hidden)
            choices.copy_(ids)
            probabilities.copy_(probs)
        stream.synchronize()
        return ids.unique().numel()

    names = ("interleaved", "separate_native", "vendor")
    update(0)
    if args.engine_adapter:
        # Eager warmup uses the default stream; capture claims the separately
        # reserved buffers on its own stream without preparing during capture.
        for _ in range(2):
            call("interleaved", banks[:1])
        torch.npu.current_stream(args.device).synchronize()
    graphs = {name: capture(torch, lambda name=name: call(name, banks[:1], True),
                           stream, 0 if args.engine_adapter
                           and name == "interleaved" else 2) for name in names}
    if args.engine_adapter:
        with torch.npu.stream(stream):
            captured, _ = adapter_pool.get(sample)
        if (adapter_pool.capture_reserves
                or captured[0].data_ptr() != fused_scratch[0].data_ptr()):
            raise RuntimeError("Capture did not claim the reserved adapter workspace")
    checks = []
    result = {"rows": rows, "blocks": args.blocks, "capacity": capacity,
              "checks": checks, "pass": False,
              "scratch_bytes": {"interleaved": sum(x.nbytes for x in fused_scratch),
                                "separate_native": sum(x.nbytes for x in routed_scratch)
                                + shared_scratch.nbytes + shared_tiling.nbytes
                                + shared_output.nbytes}}
    for replay in range(args.replays):
        active = update(replay)
        with torch.npu.stream(stream):
            for name in names[::1 if replay % 2 == 0 else -1]:
                graphs[name][0].replay()
        stream.synchronize()
        boundaries = {}

        def check(name, actual, expected):
            boundaries[name] = compare(torch, actual, expected, 0, 0)

        inverse = details["inverse"].long()
        for name, scratch in (("interleaved", fused_scratch),
                              ("separate_native", routed_scratch)):
            check(name + "/routed_output", graphs[name][1][0], graphs["vendor"][1][0])
            check(name + "/restored_input", scratch[10].repeat_interleave(TOP_K, 0),
                  details["input"].index_select(0, inverse))
            check(name + "/restored_scale", scratch[11].repeat_interleave(TOP_K, 0),
                  details["input_scale"].index_select(0, inverse).reshape(-1))
            codes, scale = middle_stages(torch, scratch, scratch[12:17], rows)
            check(name + "/routed_middle", codes,
                  details["codes"].index_select(0, inverse))
            check(name + "/routed_scale", scale, details["scale"].index_select(0, inverse))
            check(name + "/routed_before_combine", scratch[5].reshape(-1, HIDDEN),
                  details["down"].index_select(0, inverse))
        for index in (0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13):
            check(f"native_scratch/{index}", fused_scratch[index], routed_scratch[index])
        for index in (14, 15, 16):
            check(f"native_scratch/{index}", fused_scratch[index][:rows * TOP_K],
                  routed_scratch[index][:rows * TOP_K])
        for name, payload, output in (
                ("interleaved", fused_scratch[17], graphs["interleaved"][1][1]),
                ("separate_native", shared_scratch, shared_output)):
            for stage, value in shared_boundaries(
                    torch, payload, layout, rows, capacity, output).items():
                check(name + "/shared_" + stage, value, details["shared"][stage])
        passed = all(item["pass"] and item["bitwise_equal"]
                     for item in boundaries.values())
        checks.append({"replay": replay, "active_experts": active,
                       "zero_probabilities": int((probabilities.cpu().bfloat16() == 0).sum()),
                       "pass": passed, "boundaries": boundaries})
        print(json.dumps({"rows": rows, "replay": replay, "pass": passed,
                          "failures": [key for key, item in boundaries.items()
                                       if not item["pass"] or not item["bitwise_equal"]]}),
              flush=True)
        if not passed:
            dump = args.output.with_name(f"{args.output.stem}-m{rows}-r{replay}.pt")
            dump.parent.mkdir(parents=True, exist_ok=True)

            def snapshot(value):
                if isinstance(value, dict):
                    return {key: snapshot(item) for key, item in value.items()}
                return value.detach().cpu().clone()

            payload = {
                "interleaved": {str(index): value for index, value in
                                enumerate(fused_scratch) if index not in (6, 7, 8, 18)},
                "separate_native": {str(index): value for index, value in
                                    enumerate(routed_scratch) if index not in (6, 7, 8)},
                "separate_shared": shared_scratch[:layout[5]],
                "reference": details, "input": live, "choices": choices,
                "probabilities": probabilities,
            }
            torch.save(snapshot(payload), dump)
            return result
    result["pass"] = True
    if args.no_timing:
        return result
    sequence = banks * (args.calls // len(banks))
    timing_graphs = {name: capture(torch, lambda name=name: call(name, sequence),
                                  stream, 2) for name in names}
    result["timing_output_checks"] = []
    for replay in range(4):
        update(replay)
        with torch.npu.stream(stream):
            for name in names[::1 if replay % 2 == 0 else -1]:
                timing_graphs[name][0].replay()
        stream.synchronize()
        outputs = {f"{name}/{stage}": compare(
            torch, timing_graphs[name][1][index], timing_graphs["vendor"][1][index], 0, 0)
            for name in names[:-1] for index, stage in enumerate(("routed", "shared"))}
        result["timing_output_checks"].append({"replay": replay, "outputs": outputs})
        if not all(value["pass"] and value["bitwise_equal"] for value in outputs.values()):
            result["pass"] = False
            return result
    pairs = []
    for pair in range(args.timing_pairs):
        offset = pair % len(names)
        order = names[offset:] + names[:offset]
        if (pair // len(names)) % 2:
            order = order[::-1]
        pairs.append({name: elapsed(torch, timing_graphs[name][0], stream,
                                     args.iterations) / args.calls for name in order})
    result["pairs_us_per_call"] = pairs
    result["median_us_per_call"] = {
        name: statistics.median(pair[name] for pair in pairs) for name in names}
    return result


def main():
    args = arguments()
    import torch
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    for path in (args.library, args.routed_library, args.vendor_library):
        torch.ops.load_library(str(path.resolve()))
    lib = shared_library(args.shared_library)
    swiglu = ctypes.CDLL(str(args.swiglu_library.resolve()))
    swiglu.shared_swiglu_reference_launch.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 5 + [ctypes.c_int64])
    swiglu.shared_swiglu_reference_launch.restype = None

    def nd(value):
        return torch_npu.npu_format_cast(value.to(args.device).contiguous(), 2)

    source = load_weights(torch, args)
    weights = {name: nd(value) for name, value in source.items()}
    for name in ("gu_scale", "down_scale"):
        bits = source[name].view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        weights["vendor_" + name] = nd(bits)
    weights["vendor_down_scale"] = weights["vendor_down_scale"].unsqueeze(1)
    shared = shared_weights(torch, nd, torch.Generator().manual_seed(args.seed + 17),
                            args.engine_adapter, args.shared_scale_dtype)
    banks = []
    for _ in range(args.banks):
        bank = {name: torch_npu.npu_format_cast(weights[name], 29)
                .view(torch.int32).contiguous() for name in ("gate_up", "down")}
        bank["shared"] = {name: shared[name].clone() for name in
                          ("gate_tiles", "down_tiles", "gate_scale", "down_scale",
                           "vendor_gate_scale", "vendor_down_scale")}
        for name in ("gate", "down"):
            bank["shared"][name + "_nz"] = torch_npu.npu_format_cast(shared[name], 29)
        banks.append(bank)
    for name in ("gate_up", "down"):
        if len({bank[name].data_ptr() for bank in banks}) != args.banks:
            raise RuntimeError("routed NZ weight banks must own distinct addresses")
        if any(torch_npu.get_npu_format(bank[name]) != 29 for bank in banks):
            raise RuntimeError("routed weight bank lost its canonical NZ format")
    for name in ("gate_tiles", "down_tiles", "gate_nz", "down_nz"):
        if len({bank["shared"][name].data_ptr() for bank in banks}) != args.banks:
            raise RuntimeError("shared weight banks must own distinct addresses")
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "BF16 input through independent routed and shared BF16 outputs; no TP reduce",
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "cann": cann_versions(torch_npu),
        "device": str(torch.npu.get_device_properties(args.device)),
        "libraries_sha256": {name: digest(getattr(args, name)) for name in
                             ("library", "routed_library", "shared_library",
                              "vendor_library", "swiglu_library")},
        "source_sha256": {path.name: digest(path) for path in
                          (Path(__file__), HERE / "check.py")},
        "build_identity": {
            f"{name}/{filename}": {"sha256": digest(path), "text": path.read_text()}
            for name in ("library", "routed_library", "shared_library", "swiglu_library")
            for filename in ("identity.sha256", "build-manifest.json")
            if (path := getattr(args, name).parent / filename).is_file()
        },
        "shared_reference": "CANN QMM + unmodified vLLM grouped SwiGLU source + CANN QMM",
        "shared_weight_layouts": {"interleaved": "ND tiles", "separate_native": "same ND allocations",
                                  "vendor": "separate NZ allocations, equal logical values"},
        "routed_reference": "stock CANN canonical NZ bridge with shared weight allocations",
        "weight_storage_bytes": {
            "routed_shared_between_arms": sum(bank[name].untyped_storage().nbytes()
                                               for bank in banks for name in ("gate_up", "down")),
            "shared_native": sum(bank["shared"][name].untyped_storage().nbytes()
                                  for bank in banks for name in ("gate_tiles", "down_tiles")),
            "shared_vendor": sum(bank["shared"][name].untyped_storage().nbytes()
                                  for bank in banks for name in ("gate_nz", "down_nz")),
        },
        "cases": [], "pass": False,
    }
    if args.engine_adapter:
        from tilert.models.glm_5_2_ascend import routed_expert, routed_shared

        report["adapter_scope"] = (
            "TileRT run_body, shared tile packing and stream/capture workspace "
            "ownership; excludes the vLLM runner, top-k and TP finalize")
        report["source_sha256"].update({
            Path(module.__file__).name: digest(Path(module.__file__))
            for module in (routed_expert, routed_shared)
        })
    try:
        with torch.inference_mode():
            for rows in args.widths:
                case = run_case(torch, torch_npu, args, weights, banks, lib, swiglu, rows)
                report["cases"].append(case)
                if not case["pass"]:
                    raise RuntimeError("interleaved routed/shared differential failed")
        report["pass"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["loaded_libraries_sha256"] = loaded_libraries()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
