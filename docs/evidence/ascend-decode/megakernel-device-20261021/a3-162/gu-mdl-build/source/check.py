"""Compare routed INT4 with the CANN fused W4A8 chain on an assigned A3 die.

Weights may come from a ModelSlim checkpoint, a CPU fixture, or seeded random
INT4 codes. Route values and activations change while graph storage is fixed.
No device work happens when this module is imported.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HIDDEN, INTERMEDIATE, TOP_K, EXPERTS = 6144, 128, 8, 256


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--native-layout", choices=("nd", "nz"), default="nd",
                        help="native B format; NZ defaults to the build-nz library")
    parser.add_argument("--native-partial", choices=("int32", "fp16"), default="int32",
                        help="FP16 selects the separate Fixpipe candidate and namespace")
    parser.add_argument("--native-schedule", choices=("expert", "row"), default="expert",
                        help="row selects the separate AIV scheduling candidate")
    parser.add_argument("--native-cube", choices=("norm", "gu-mdl"), default="norm",
                        help="gu-mdl selects grouped K loads for gate/up only")
    parser.add_argument("--native-api", choices=("isolated", "tilert"), default="isolated",
                        help="tilert selects the engine ABI for the row schedule")
    parser.add_argument("--native-routing", choices=("metadata", "dispatch"),
                        default="metadata", help="reuse vendor dispatch inside the kernel")
    parser.add_argument("--vendor-library", type=Path)
    parser.add_argument("--vendor-backend", choices=("vllm-ascend", "torch-npu", "cann-nz"),
                        default="vllm-ascend")
    parser.add_argument("--vendor-layout", choices=("nd", "nz"),
                        help="defaults to NZ for vLLM-Ascend and ND for torch_npu")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--fixture", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--rank", type=int, default=0, choices=range(16))
    parser.add_argument("--widths", type=int, nargs="+",
                        default=[1, 2, 8, 12, 32])
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--timing-pairs", type=int, default=5)
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--eager", action="store_true",
                        help="diagnose arithmetic without graph capture; requires --no-timing")
    parser.add_argument("--atol", type=float, default=0)
    parser.add_argument("--rtol", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.library is None:
        directory = ("build-gu-mdl" if args.native_cube == "gu-mdl"
                     else "build-balanced-promoted" if args.native_schedule == "row"
                     and args.native_api == "tilert"
                     else "build-balanced" if args.native_schedule == "row"
                     else "build-fixpipe" if args.native_partial == "fp16"
                     else "build-route" if args.native_routing == "dispatch"
                     else "build-nz" if args.native_layout == "nz" else "build")
        args.library = HERE / directory / "routed_int4_torch.so"
    if args.native_schedule == "row" and args.native_partial != "int32":
        parser.error("row scheduling requires INT32 partials")
    if args.native_api == "tilert" and args.native_partial != "int32":
        parser.error("the engine ABI requires INT32 partials")
    if args.native_cube == "gu-mdl" and (
            args.native_schedule != "row" or args.native_partial != "int32"
            or args.native_api != "isolated"):
        parser.error("gu-mdl requires the isolated row schedule with INT32 partials")
    if args.vendor_layout is None:
        args.vendor_layout = "nd" if args.vendor_backend == "torch-npu" else "nz"
    if args.vendor_backend in ("vllm-ascend", "cann-nz") and args.vendor_layout != "nz":
        parser.error("the selected reference requires NZ weights")
    if args.vendor_backend == "cann-nz" and args.vendor_library is None:
        args.vendor_library = HERE / "build-reference/nz_reference.so"
    if args.eager and not args.no_timing:
        parser.error("--eager requires --no-timing")
    if any(width not in range(1, 33) for width in args.widths):
        parser.error("widths must be in [1,32]")
    if min(args.blocks, args.warmup, args.iterations, args.timing_pairs) < 1:
        parser.error("blocks, warmup, iterations and timing-pairs must be positive")
    if args.replays < 4:
        parser.error("at least four replays are needed to cover route changes")
    if min(args.atol, args.rtol) < 0:
        parser.error("tolerances must be nonnegative")
    return args


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def loaded_libraries():
    names = ("vllm_ascend", "libopapi.so", "libcust_opapi.so",
             "libtorch_npu.so", "libruntime.so", "libascendcl.so")
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and any(name in fields[5] for name in names):
            path = Path(fields[5])
            if path.is_file():
                paths.add(path)
    return {str(path): digest(path) for path in sorted(paths)}


def cann_versions(torch_npu):
    query = getattr(torch_npu.utils, "get_cann_version", None)
    if query is None:
        return {"status": "version API unavailable"}
    versions = {}
    for component in ("CANN", "RUNTIME", "COMPILER", "OPP", "OPP_KERNEL", "DRIVER"):
        try:
            versions[component] = query(module=component)
        except (RuntimeError, TypeError, UnicodeError) as error:
            versions[component] = str(error)
    return versions


def random_bank(torch, generator, inputs, outputs):
    packed = torch.randint(-128, 128, (EXPERTS, inputs, outputs // 2),
                           dtype=torch.int8, generator=generator)
    scale = (torch.rand(EXPERTS, outputs, generator=generator) + 1) / 2048
    correction = torch.empty_like(scale)
    for expert in range(EXPERTS):
        values = packed[expert].to(torch.int16)
        low = ((values & 15) ^ 8) - 8
        high = values >> 4
        sums = torch.stack((low.sum(0), high.sum(0)), dim=-1).flatten()
        correction[expert] = sums.float() * 8 * scale[expert]
    return packed, correction, scale


def load_weights(torch, args):
    if args.fixture is not None:
        payload = torch.load(args.fixture, map_location="cpu", weights_only=True)
    elif args.checkpoint is not None:
        from tilert.models.glm_5_2_ascend.weights import ModelSlimCheckpoint

        checkpoint = ModelSlimCheckpoint(args.checkpoint)
        payload = {
            "gate_up": torch.empty(EXPERTS, HIDDEN, INTERMEDIATE, dtype=torch.int8),
            "gu_bias": torch.empty(EXPERTS, 2 * INTERMEDIATE),
            "gu_scale": torch.empty(EXPERTS, 2 * INTERMEDIATE),
            "down": torch.empty(EXPERTS, INTERMEDIATE, HIDDEN // 2, dtype=torch.int8),
            "down_bias": torch.empty(EXPERTS, HIDDEN),
            "down_scale": torch.empty(EXPERTS, HIDDEN),
        }
        for expert in range(EXPERTS):
            weight = checkpoint.expert(args.layer, expert, args.rank)
            payload["gate_up"][expert] = torch.cat(
                (weight.gate.values, weight.up.values),
            ).T.contiguous()
            payload["gu_bias"][expert] = torch.cat(
                (weight.gate.scale_bias, weight.up.scale_bias),
            )
            payload["gu_scale"][expert] = torch.cat(
                (weight.gate.scale, weight.up.scale),
            )
            payload["down"][expert] = weight.down.values.T.contiguous()
            payload["down_bias"][expert] = weight.down.scale_bias
            payload["down_scale"][expert] = weight.down.scale
    else:
        generator = torch.Generator().manual_seed(args.seed)
        gate, gu_bias, gu_scale = random_bank(
            torch, generator, HIDDEN, 2 * INTERMEDIATE,
        )
        down, down_bias, down_scale = random_bank(
            torch, generator, INTERMEDIATE, HIDDEN,
        )
        payload = dict(gate_up=gate, gu_bias=gu_bias, gu_scale=gu_scale,
                       down=down, down_bias=down_bias, down_scale=down_scale)
    shapes = {"gate_up": (EXPERTS, HIDDEN, INTERMEDIATE),
              "gu_bias": (EXPERTS, 2 * INTERMEDIATE),
              "gu_scale": (EXPERTS, 2 * INTERMEDIATE),
              "down": (EXPERTS, INTERMEDIATE, HIDDEN // 2),
              "down_bias": (EXPERTS, HIDDEN),
              "down_scale": (EXPERTS, HIDDEN)}
    for name, shape in shapes.items():
        value = payload[name]
        dtype = torch.int8 if name in ("gate_up", "down") else torch.float32
        if value.shape != shape or value.dtype != dtype or not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous {dtype} {shape}")
    return {name: payload[name] for name in shapes}


def route_case(torch, width, iteration, seed):
    generator = torch.Generator().manual_seed(seed + iteration)
    rows = torch.randn(width, HIDDEN, generator=generator)
    rows *= torch.rand(width, 1, generator=generator) + 0.5
    rows = rows.to(torch.bfloat16)
    if iteration % 4 in (0, 1):
        # The same hot count moves to new expert IDs at a fixed graph shape.
        offset = (iteration * 37) % EXPERTS
        choices = ((torch.arange(TOP_K) + offset) % EXPERTS).repeat(width, 1)
    else:
        choices = torch.stack([
            torch.randperm(EXPERTS, generator=generator)[:TOP_K]
            for _ in range(width)
        ])
    if iteration % 4 == 3:
        choices = choices.flip(1)
    probabilities = torch.rand(width, TOP_K, generator=generator)
    probabilities /= probabilities.sum(-1, keepdim=True)
    case = iteration % 8
    if case == 4:
        rows.zero_()
    elif case == 5:
        rows[:, ::2], rows[:, 1::2] = -64, 64
    elif case == 6:
        probabilities[:, 0] = 0
        probabilities /= probabilities.sum(-1, keepdim=True)
    elif case == 7:
        probabilities[:, 1::2] = 0
        probabilities /= probabilities.sum(-1, keepdim=True)
    return rows, choices.to(torch.int32), probabilities


def compare(torch, actual, expected, atol, rtol):
    actual_cpu = actual.detach().cpu().contiguous()
    expected_cpu = expected.detach().cpu().contiguous()
    bitwise_equal = different_bytes = None
    if actual_cpu.dtype == expected_cpu.dtype and actual_cpu.shape == expected_cpu.shape:
        actual_bytes = actual_cpu.reshape(-1).clone(
            memory_format=torch.contiguous_format).view(torch.uint8)
        expected_bytes = expected_cpu.reshape(-1).clone(
            memory_format=torch.contiguous_format).view(torch.uint8)
        different_bytes = int((actual_bytes != expected_bytes).sum())
        bitwise_equal = different_bytes == 0
    actual, expected = actual_cpu.float(), expected_cpu.float()
    difference = (actual - expected).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    result = {
        "pass": finite and bool(torch.all(difference <= atol + rtol * expected.abs())),
        "exact": torch.equal(actual, expected),
        "bitwise_equal": bitwise_equal,
        "bitwise_different_bytes": different_bytes,
        "unequal": int((actual != expected).sum()),
        "elements": actual.numel(), "max_abs": float(difference.max()),
        "nrmse": float(difference.norm() / expected.norm().clamp_min(1e-30)),
    }
    if not result["pass"]:
        for name, values in (("actual", actual), ("expected", expected)):
            result[name] = {
                "finite": int(torch.isfinite(values).sum()),
                "nan": int(torch.isnan(values).sum()),
                "zero": int((values == 0).sum()),
                "prefix": values.reshape(-1)[:16].tolist(),
            }
    return result


def accepts_comparison(result, require_bitwise):
    return result["pass"] and (
        not require_bitwise or result["bitwise_equal"] is True)


def capture(torch, call, stream, warmup):
    with torch.npu.stream(stream):
        for _ in range(warmup):
            call()
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = call()
    return graph, output


def elapsed(torch, graph, stream, iterations):
    with torch.npu.stream(stream):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def middle_stages(torch, scratch, metadata, width):
    count, start, source, slot, _ = [tensor.cpu() for tensor in metadata]
    packed = scratch[2].cpu().to(torch.int16)
    low = ((packed & 15) ^ 8) - 8
    high = packed >> 4
    unpacked = torch.stack((low, high), dim=-1).reshape(
        2 * width * TOP_K, INTERMEDIATE,
    )
    scales = scratch[3].cpu()
    codes = torch.empty(width * TOP_K, INTERMEDIATE, dtype=torch.int8)
    ordered_scale = torch.empty(width * TOP_K)
    for expert in range(EXPERTS):
        size, begin = int(count[expert]), int(start[expert])
        if size:
            codes[begin:begin + size] = (
                16 * unpacked[2 * begin:2 * begin + size]
                + unpacked[2 * begin + size:2 * (begin + size)] + 8
            ).to(torch.int8)
            ordered_scale[begin:begin + size] = scales[begin:begin + size]
    original = source[:width * TOP_K].long() * TOP_K + slot[:width * TOP_K]
    inverse = torch.empty_like(original)
    inverse[original] = torch.arange(width * TOP_K)
    return codes[inverse], ordered_scale[inverse]


def inspect_first_expert(torch, scratch, metadata, weights, native_details):
    """Locate the first divergent boundary with exact CPU integer products."""
    from tilert.models.glm_5_2_ascend.reference_weights import w4_msd_product

    counts, starts, sources = [tensor.cpu() for tensor in metadata[:3]]
    expert = int(torch.nonzero(counts)[0])
    count, begin = int(counts[expert]), int(starts[expert])
    rows = native_details["rows"].cpu()[sources[begin:begin + count].long()]
    scale = native_details["scale"].cpu()[sources[begin:begin + count].long()]

    def unpack(packed):
        packed = packed.cpu().to(torch.int16)
        low = ((packed & 15) ^ 8) - 8
        high = packed >> 4
        return torch.stack((low, high), dim=-1).flatten(-2).to(torch.int32)

    stacked = unpack(scratch[0][2 * begin:2 * (begin + count)])
    integer_rows = rows.int()
    expected_stacked = torch.cat((
        torch.div(integer_rows, 16, rounding_mode="floor"),
        (integer_rows & 15) - 8,
    ))
    weight = unpack(weights["gate_up"][expert])
    product = scratch[1][2 * begin:2 * (begin + count)].cpu()
    expected_product = expected_stacked @ weight
    product_name = "gate_integer_product"
    if product.dtype == torch.float16:
        scale_bits = weights["gu_scale"][expert].cpu().contiguous().view(torch.int32)
        truncated_scale = (scale_bits & -8192).view(torch.float32)
        expected_product = (expected_product.float() * truncated_scale).half()
        product_name = "gate_scaled_fp16_partial"
    projection = w4_msd_product(
        integer_rows, weight.T, weights["gu_scale"][expert].cpu(),
        weights["gu_bias"][expert].cpu(), scale[:, None],
        output_dtype=torch.float32,
    )
    activated = torch.nn.functional.silu(projection[:, :INTERMEDIATE])
    activated *= projection[:, INTERMEDIATE:]
    expected_scale = activated.abs().amax(-1) / 127
    actual_scale = scratch[3][begin:begin + count].cpu()
    return {
        "expert": expert,
        "count": count,
        "stacked_input": compare(torch, stacked, expected_stacked, 0, 0),
        product_name: compare(torch, product, expected_product, 0, 0),
        "gmm1_scale_cpu_formula": compare(torch, actual_scale, expected_scale, 0, 0),
        "cpu_projection_prefix": projection[0, :16].tolist(),
    }


def run_width(torch, torch_npu, args, weights, native_weights, vendor, width):
    from tilert.models.glm_5_2_ascend.routed_expert import route_metadata

    device = args.device
    native = (torch.ops.tilert_mdl if args.native_cube == "gu-mdl"
              else torch.ops.tilert if args.native_api == "tilert"
              else torch.ops.tilert_balanced if args.native_schedule == "row"
              else torch.ops.tilert_fixpipe if args.native_partial == "fp16"
              else torch.ops.tilert)
    live_hidden = torch.empty(width, HIDDEN, dtype=torch.bfloat16, device=device)
    live_choices = torch.empty(width, TOP_K, dtype=torch.int32, device=device)
    probabilities = torch.empty(width, TOP_K, dtype=torch.float32, device=device)
    metadata = [torch.empty(256, dtype=dtype, device=device)
                for dtype in (torch.int32, torch.int32, torch.int32,
                              torch.int32, torch.float32)]
    sample_rows = torch.empty(width, HIDDEN, dtype=torch.int8, device=device)
    native_options = {"weight_layout": 1} if args.native_layout == "nz" else {}
    from_dispatch = args.native_routing == "dispatch"
    prepare_options = dict(native_options)
    if from_dispatch:
        prepare_options["from_dispatch"] = True
    scratch, sync_base = native.routed_experts_prepare(
        sample_rows, EXPERTS, args.blocks, **prepare_options,
    )
    if from_dispatch:
        original_scratch, original_sync = native.routed_experts_prepare(
            sample_rows, EXPERTS, args.blocks, **native_options,
        )
    stream = torch.npu.Stream(device=device)
    ordered_weights = [native_weights[name] for name in (
        "gate_up", "gu_bias", "gu_scale", "down", "down_bias", "down_scale",
    )]
    native_details = {}
    vendor_details = {}

    def update(iteration):
        rows, choices, probs = route_case(torch, width, iteration, args.seed)
        fresh = route_metadata(choices, probs)
        with torch.npu.stream(stream):
            live_hidden.copy_(rows)
            live_choices.copy_(choices)
            probabilities.copy_(probs)
            for destination, value in zip(metadata, fresh):
                destination.copy_(value)
        stream.synchronize()

    def dispatch():
        return torch_npu.npu_moe_init_routing_v2(
            live_hidden, live_choices, active_num=width * TOP_K,
            expert_num=EXPERTS, expert_tokens_num_type=1,
            expert_tokens_num_flag=True, active_expert_range=[0, EXPERTS],
            quant_mode=1,
        )

    def native_call():
        packed, inverse, count, packed_scale = dispatch()
        if from_dispatch:
            native_details.update(rows=scratch[10], scale=scratch[11],
                                  inverse=inverse)
            return native.routed_experts_from_dispatch(
                packed, *ordered_weights, packed_scale.reshape(-1),
                inverse, count, probabilities, scratch, sync_base,
                EXPERTS, args.blocks, **native_options,
            )
        # expanded_row_idx[token * TOP_K + slot] is its sorted row.
        first_slot = inverse.reshape(width, TOP_K)[:, 0].long()
        rows = packed.index_select(0, first_slot)
        scale = packed_scale.index_select(0, first_slot).reshape(width)
        native_details.update(rows=rows, scale=scale)
        return native.routed_experts(
            rows, *ordered_weights, scale, *metadata,
            scratch, sync_base, EXPERTS, args.blocks, **native_options,
        )

    def original_full_chain():
        packed, inverse, _, packed_scale = dispatch()
        first_slot = inverse.reshape(width, TOP_K)[:, 0].long()
        rows = packed.index_select(0, first_slot)
        scale = packed_scale.index_select(0, first_slot).reshape(width)
        routes = route_metadata(live_choices, probabilities)
        return native.routed_experts(
            rows, *ordered_weights, scale, *routes, original_scratch,
            original_sync, EXPERTS, args.blocks, **native_options,
        )

    def vendor_call(keep_intermediates=True):
        packed, inverse, count, packed_scale = dispatch()
        vendor_count = count.to(torch.int64)
        if args.vendor_backend == "cann-nz":
            codes, scale = torch.ops.tilert_reference.nz_gmm1(
                packed, vendor["gate_up"], vendor["gu_scale"], packed_scale,
                vendor_count, weights["gu_bias"],
            )
        else:
            fused = (torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2
                     if args.vendor_backend == "vllm-ascend"
                     else torch_npu.npu_grouped_matmul_swiglu_quant_v2)
            keywords = ({"swiglu_limit": 0.0}
                        if args.vendor_backend == "vllm-ascend" else {})
            codes, scale = fused(
                x=packed, weight=[vendor["gate_up"]],
                weight_scale=[vendor["gu_scale"]], x_scale=packed_scale,
                group_list=vendor_count, weight_assist_matrix=[weights["gu_bias"]],
                dequant_mode=0, group_list_type=1, **keywords,
            )
        # The MSD down projection reuses the input-code storage for its
        # nibble expansion. Preserve the GMM1 boundary only for comparison.
        saved_codes = codes.clone() if keep_intermediates else codes
        if args.vendor_backend == "cann-nz":
            routed = torch.ops.tilert_reference.nz_gmm2(
                codes, vendor["down"], vendor["down_scale"], scale,
                vendor_count, weights["down_bias"],
            )
        else:
            routed = torch_npu.npu_grouped_matmul(
                x=[codes], weight=[vendor["down"]], scale=[vendor["down_scale"]],
                bias=[weights["down_bias"]], per_token_scale=[scale],
                group_list=vendor_count, split_item=2, group_type=0,
                group_list_type=1, output_dtype=torch.bfloat16,
            )[0]
        vendor_details.update(codes=saved_codes, scale=scale, routed=routed,
                              input_codes=packed, input_scale=packed_scale,
                              inverse=inverse)
        return torch_npu.npu_moe_token_unpermute(
            routed, inverse, probs=probabilities.to(torch.bfloat16),
        )

    torch.npu.synchronize()
    update(0)
    if not args.eager:
        native_graph, native_output = capture(torch, native_call, stream, args.warmup)
        vendor_graph, vendor_output = capture(torch, vendor_call, stream, args.warmup)
        if from_dispatch:
            original_graph, original_output = capture(
                torch, original_full_chain, stream, args.warmup,
            )
    report = {"width": width, "scratch_bytes": sum(t.nbytes for t in scratch),
              "checks": [], "pass": True}
    for iteration in range(args.replays):
        update(iteration)
        with torch.npu.stream(stream):
            if args.eager:
                native_output = native_call()
                vendor_output = vendor_call()
                if from_dispatch:
                    original_output = original_full_chain()
            else:
                native_graph.replay()
                vendor_graph.replay()
                if from_dispatch:
                    original_graph.replay()
        stream.synchronize()
        check = compare(torch, native_output, vendor_output, args.atol, args.rtol)
        check["iteration"] = iteration
        check["maximum_expert_rows"] = int(metadata[0].max().cpu())
        inverse = vendor_details["inverse"].long()
        restored_codes = native_details["rows"].repeat_interleave(TOP_K, dim=0)
        restored_scale = native_details["scale"].repeat_interleave(TOP_K, dim=0)
        check["input_routing_codes"] = compare(
            torch, restored_codes,
            vendor_details["input_codes"].index_select(0, inverse), 0, 0,
        )
        check["input_routing_scale"] = compare(
            torch, restored_scale,
            vendor_details["input_scale"].index_select(0, inverse).reshape(-1), 0, 0,
        )
        with torch.npu.stream(stream):
            standalone_codes, standalone_scale = torch_npu.npu_dynamic_quant(live_hidden)
        stream.synchronize()
        check["standalone_quant_diagnostic"] = {
            "codes": compare(torch, standalone_codes, native_details["rows"], 0, 0),
            "scale": compare(torch, standalone_scale.reshape(-1),
                             native_details["scale"], 0, 0),
        }
        required = [check, check["input_routing_codes"], check["input_routing_scale"]]
        if from_dispatch:
            check["original_full_chain"] = compare(
                torch, native_output, original_output, args.atol, args.rtol,
            )
            # An expert's internal row order can differ from argsort's order.
            # Inverse identifies the original slot independently of stability.
            expected_original = (native_details["inverse"].cpu().argsort()
                                 .to(torch.int32))
            actual_original = scratch[14][:width * TOP_K] * TOP_K
            actual_original += scratch[15][:width * TOP_K]
            expected_weight = probabilities.cpu().bfloat16().float().flatten()[
                expected_original.long()
            ]
            check["dispatch_metadata"] = {
                "counts": compare(torch, scratch[12], metadata[0], 0, 0),
                "starts": compare(torch, scratch[13], metadata[1], 0, 0),
                "original_slots": compare(torch, actual_original,
                                          expected_original, 0, 0),
                "weights": compare(torch, scratch[16][:width * TOP_K],
                                   expected_weight, 0, 0),
            }
            required.append(check["original_full_chain"])
            required.extend(check["dispatch_metadata"].values())
        # Standalone quantization has a distinct contract and stays diagnostic.
        require_bitwise = args.atol == 0 and args.rtol == 0
        check["acceptance_requires_bitwise"] = require_bitwise
        check["pass"] = all(accepts_comparison(result, require_bitwise)
                            for result in required)
        report["checks"].append(check)
        if not check["pass"]:
            actual_metadata = scratch[12:17] if from_dispatch else metadata
            codes, scale = middle_stages(torch, scratch, actual_metadata, width)
            routed = vendor_details["routed"].index_select(0, inverse.long())
            report["stages"] = {
                "gmm1_codes": compare(torch, codes,
                                      vendor_details["codes"].index_select(0, inverse), 0, 0),
                "gmm1_scale": compare(torch, scale,
                                      vendor_details["scale"].index_select(0, inverse), 0, 0),
                "down_before_combine": compare(
                    torch, scratch[5].reshape(-1, HIDDEN), routed, 0, 0,
                ),
            }
            report["first_expert"] = inspect_first_expert(
                torch, scratch, actual_metadata, weights, native_details,
            )
            report["pass"] = False
            return report
    if args.no_timing:
        return report
    vendor_graph, _ = capture(
        torch, lambda: vendor_call(keep_intermediates=False), stream, args.warmup,
    )
    timings = {"native_us": [], "vendor_us": []}
    if from_dispatch:
        timings["original_full_chain_us"] = []
    for pair in range(args.timing_pairs):
        order = [("native_us", native_graph), ("vendor_us", vendor_graph)]
        if from_dispatch:
            order.append(("original_full_chain_us", original_graph))
        if pair % 2:
            order.reverse()
        for name, graph in order:
            timings[name].append(elapsed(torch, graph, stream, args.iterations))
    report["timing"] = timings
    report["median_us"] = {key: statistics.median(value)
                           for key, value in timings.items()}
    if from_dispatch:
        report["full_replacement_timing"] = {
            "scope": "BF16 vendor dispatch through final BF16 combine; original "
                     "chain includes captured argsort/counts/prefix/gather and "
                     "both index_select calls; reuse chain builds metadata and "
                     "restores rows inside the mixed kernel",
            "samples_us": timings,
            "median_us": report["median_us"],
        }
    return report


def run(args):
    sys.path.insert(0, str(ROOT))
    import torch
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    torch.ops.load_library(str(args.library.resolve()))
    native = (torch.ops.tilert_mdl if args.native_cube == "gu-mdl"
              else torch.ops.tilert if args.native_api == "tilert"
              else torch.ops.tilert_balanced if args.native_schedule == "row"
              else torch.ops.tilert_fixpipe if args.native_partial == "fp16"
              else torch.ops.tilert)
    if args.vendor_backend == "cann-nz":
        torch.ops.load_library(str(args.vendor_library.resolve()))
        vendor_gmm1 = torch.ops.tilert_reference.nz_gmm1
    elif args.vendor_backend == "vllm-ascend":
        if args.vendor_library is not None:
            torch.ops.load_library(str(args.vendor_library.resolve()))
        else:
            from vllm_ascend.utils import enable_custom_op

            if not enable_custom_op():
                raise RuntimeError(
                    "The deployed vllm-ascend custom operators are unavailable"
                )
        vendor_gmm1 = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2
    else:
        vendor_gmm1 = torch.ops.npu.npu_grouped_matmul_swiglu_quant_v2
    source = load_weights(torch, args)
    weights = {name: value.to(args.device) for name, value in source.items()}
    native_weights = dict(weights)
    vendor = {}
    for name in ("gate_up", "down"):
        # The production loader converts packed INT8 to NZ before exposing
        # four physical bytes as one INT32 carrier for the GMM API.
        nd = weights[name]
        nz = None
        if "nz" in (args.native_layout, args.vendor_layout):
            nz = torch_npu.npu_format_cast(nd, 29).view(torch.int32).contiguous()
        if args.native_layout == "nz":
            native_weights[name] = nz
        vendor[name] = (nz if args.vendor_layout == "nz"
                        else nd.view(torch.int32).contiguous())
    for name in ("gu_scale", "down_scale"):
        bits = source[name].view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        vendor[name] = bits.to(args.device)
        if args.native_partial == "fp16":
            native_weights[name] = vendor[name]
        if name == "down_scale":
            vendor[name] = vendor[name].unsqueeze(1)
    report = {
        "scope": "one TP16 W4A8 routed stage, before shared expert and TP reduction",
        "status": "running", "arguments": {key: str(value) if isinstance(value, Path)
                                             else value for key, value in vars(args).items()},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "torch_npu": getattr(torch_npu, "__version__", None),
        "cann": cann_versions(torch_npu),
        "device": str(torch.npu.get_device_properties(args.device)),
        "library_sha256": digest(args.library),
        "vendor_library_sha256": digest(args.vendor_library) if args.vendor_library else None,
        "operator_schemas": {
            "native": str(native.routed_experts.default._schema),
            "vendor_gmm1": str(vendor_gmm1.default._schema),
        },
        "weight_storage": {
            path: {
                name: {"dtype": str(bank[name].dtype),
                       "logical_shape": list(bank[name].shape),
                       "npu_format": torch_npu.get_npu_format(bank[name]),
                       "physical_bytes": bank[name].untyped_storage().nbytes()}
                for name in ("gate_up", "down")
            }
            for path, bank in (("native", native_weights), ("vendor", vendor))
        },
        "source_sha256": {str(path.relative_to(ROOT)): digest(path)
                          for path in sorted(HERE.glob("*"))
                          if path.is_file() and path.suffix in (".py", ".cpp", ".h", ".sh")},
        "cases": [],
        "timing_scope": "graph device events from BF16 init-routing quantization "
                        "through combine, including the native path's restoration "
                        "of source rows; route metadata construction, input copies "
                        "and weight preparation excluded from both measurements; "
                        "not full dispatch or model TPOT",
    }
    if args.native_routing == "dispatch":
        report["operator_schemas"]["native_from_dispatch"] = str(
            native.routed_experts_from_dispatch.default._schema,
        )
        report["timing_scope"] = (
            "complete routed replacement from BF16 vendor init-routing through "
            "BF16 combine, including all route metadata and row restoration; "
            "paired against the original native chain and vendor GMM chain; "
            "input copies, router/top-k, weights, shared experts and TP reduction "
            "are outside all measurements; not model TPOT"
        )
    if args.vendor_backend == "cann-nz":
        report["reference_contract"] = {
            "implementation": "stock CANN SDK WeightNz APIs with canonical "
                              "descriptors; not the deployed vLLM custom library",
            "gmm1_api": "aclnnGroupedMatmulSwigluQuantWeightNzV2",
            "gmm2_api": "aclnnGroupedMatmulWeightNz",
            "weight_descriptor_dtype": "ACL_INT32",
            "weight_descriptor_storage": {
                "gate_up": [EXPERTS, 2 * INTERMEDIATE // 64, HIDDEN // 16, 16, 8],
                "down": [EXPERTS, HIDDEN // 64, INTERMEDIATE // 16, 16, 8],
            },
            "scale_descriptor_dtype": "ACL_UINT64",
            "copies_weight_bytes": False,
            "mutates_torch_storage_descriptor": False,
        }
    if args.checkpoint is not None:
        report["checkpoint_index_sha256"] = digest(
            args.checkpoint / "quant_model_weights.safetensors.index.json",
        )
    if args.fixture is not None:
        report["fixture_sha256"] = digest(args.fixture)
    try:
        with torch.inference_mode():
            for width in args.widths:
                case = run_width(torch, torch_npu, args, weights,
                                 native_weights, vendor, width)
                report["cases"].append(case)
                if not case["pass"]:
                    report["status"] = "numerical_failure"
                    return report
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["loaded_libraries_sha256"] = loaded_libraries()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    args = arguments()
    report = run(args)
    print(json.dumps({"status": report["status"], "report": str(args.output)}))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
