"""Compare the single-launch MLA tail with the original CANN chain on A3.

This file performs device work only when invoked explicitly. It accepts a
CPU tensor fixture or creates deterministic synthetic BF16/INT8 inputs.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library", type=Path,
        default=Path(__file__).resolve().parent / "build/mla_tail_torch.so",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output-scale-dtype", default="float32",
                        choices=("float32", "bfloat16"),
                        help="Original scale dtype passed to the stock QMM")
    parser.add_argument("--random-output-scale", action="store_true",
                        help="Use seed-830 channel scales with varied exponents")
    parser.add_argument("--widths", type=int, nargs="+",
                        default=[1, 2, 4, 6, 8, 12, 16, 24, 32])
    parser.add_argument("--blocks", type=int, nargs="+", default=[0])
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--timing-pairs", type=int, default=5)
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--queued-producer", action="store_true",
                        help="clone latent inside capture immediately before native")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(width not in range(1, 33) for width in args.widths):
        parser.error("widths must be between 1 and 32")
    if any(blocks < 0 for blocks in args.blocks):
        parser.error("blocks must be nonnegative")
    if min(args.replays, args.iterations, args.timing_pairs) <= 0:
        parser.error("replays, iterations and timing-pairs must be positive")
    if any(not math.isfinite(value) or value < 0
           for value in (args.atol, args.rtol)):
        parser.error("tolerances must be finite and nonnegative")
    return args


def load_inputs(torch, args):
    if args.fixture is not None:
        fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
        required = {"latent", "value_weight", "output_weight", "output_scale"}
        missing = required - fixture.keys()
        if missing:
            raise ValueError(f"fixture is missing {sorted(missing)}")
        return fixture
    generator = torch.Generator().manual_seed(829)
    return {
        "latent": torch.randn(32, 4, 512, generator=generator).to(torch.bfloat16),
        "value_weight": (
            torch.randn(4, 512, 256, generator=generator) / 32
        ).to(torch.bfloat16),
        "output_weight": torch.randint(-127, 128, (1024, 6144),
                                        generator=generator, dtype=torch.int8),
        "output_scale": torch.full((6144,), 1 / 1024, dtype=torch.float32),
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(torch, torch_npu, args):
    here = Path(__file__).resolve().parent
    source_names = (
        "mla_tail.h", "mla_tail_native.h", "mla_tail_kernel.cpp", "mla_tail_tiling.cpp",
        "mla_tail_binding.cpp", "build_so.sh", "check.py",
    )
    versions = {}
    get_version = getattr(torch_npu.utils, "get_cann_version", None)
    for module in ("CANN", "RUNTIME", "DRIVER"):
        if get_version is None:
            versions[module] = "version API unavailable"
            continue
        try:
            versions[module] = get_version(module=module)
        except (RuntimeError, TypeError, UnicodeError) as error:
            versions[module] = f"unavailable: {error}"
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "device_name": torch.npu.get_device_name(args.device),
        "device_properties": str(torch.npu.get_device_properties(args.device)),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "versions": versions,
        "library_sha256": sha256_file(args.library.resolve()),
        "source_sha256": {name: sha256_file(here / name)
                          for name in source_names},
        "fixture_sha256": sha256_file(args.fixture) if args.fixture else None,
    }


def validate_inputs(torch, fixture, maximum_width):
    expected = {
        "value_weight": ((4, 512, 256), torch.bfloat16),
        "output_weight": ((1024, 6144), torch.int8),
        "output_scale": ((6144,), None),
    }
    latent = fixture["latent"]
    if (not torch.is_tensor(latent) or latent.dim() != 3
            or tuple(latent.shape[1:]) != (4, 512)
            or latent.shape[0] < maximum_width
            or latent.dtype != torch.bfloat16):
        raise ValueError("latent must be BF16 [M,4,512] with enough rows")
    for name, (shape, dtype) in expected.items():
        tensor = fixture[name]
        if (not torch.is_tensor(tensor) or tuple(tensor.shape) != shape
                or (dtype is not None and tensor.dtype != dtype)):
            raise ValueError(f"{name} has the wrong shape or dtype")
        if name != "output_weight" and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains nonfinite data")
    if not bool(torch.isfinite(latent).all()):
        raise ValueError("latent contains nonfinite data")
    bias = fixture.get("bias")
    if bias is not None and (
        not torch.is_tensor(bias) or tuple(bias.shape) != (6144,)
        or not bool(torch.isfinite(bias).all())
    ):
        raise ValueError("bias must contain 6144 finite values")


def metrics(torch, actual, expected, atol, rtol):
    actual = actual.detach().cpu()
    expected = expected.detach().cpu()
    different_bytes = None
    if actual.dtype == expected.dtype and actual.shape == expected.shape:
        actual_bytes = actual.reshape(-1).clone(
            memory_format=torch.contiguous_format).view(torch.uint8)
        expected_bytes = expected.reshape(-1).clone(
            memory_format=torch.contiguous_format).view(torch.uint8)
        different_bytes = int((actual_bytes != expected_bytes).sum())
    actual, expected = actual.float(), expected.float()
    difference = (actual - expected).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    accepted = finite and bool(torch.all(difference <= atol + rtol * expected.abs()))
    return {
        "pass": accepted,
        "exact": bool(torch.equal(actual, expected)),
        "bitwise_equal": different_bytes == 0 if different_bytes is not None else None,
        "bitwise_different_bytes": different_bytes,
        "max_abs": float(difference.max()),
        "rms": float((difference.square().mean()).sqrt()),
        "different_elements": int((actual != expected).sum()),
        "elements": actual.numel(),
    }


def capture(torch, call, stream):
    with torch.npu.stream(stream):
        for _ in range(3):
            result = call()
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        result = call()
    with torch.npu.stream(stream):
        graph.replay()
    stream.synchronize()
    return graph, result


def event_time_us(torch, graph, iterations, stream):
    with torch.npu.stream(stream):
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        begin.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / iterations


def inspect_intermediates(torch, torch_npu, scratch, capacity, latent,
                          value_weight):
    from contract import workspace_layout

    regions = {region.name: region for region in workspace_layout(capacity)}
    # Read only user scratch, excluding the Matmul system workspace.
    raw = scratch[:regions["system"].offset].cpu()

    def region(name, dtype):
        part = regions[name]
        return raw[part.offset:part.offset + part.size].view(dtype)

    rows = latent.shape[0]
    values = region("values", torch.float32).reshape(16, capacity, 64)
    values = values.permute(1, 0, 2).reshape(capacity, 1024)
    values = values[:rows].to(torch.bfloat16)
    codes = region("quantized", torch.int8).reshape(capacity, 1024)[:rows]
    scales = region("token_scales", torch.float32).reshape(capacity, 8)[:rows, 0]
    expected_values = torch_npu.npu_transpose_batchmatmul(
        latent, value_weight, perm_x1=(1, 0, 2), perm_x2=(0, 1, 2),
        perm_y=(1, 0, 2),
    ).reshape(rows, 1024)
    expected_codes, expected_scales = torch_npu.npu_dynamic_quant(expected_values)
    expected_codes_cpu = expected_codes.cpu()
    return {
        "value_up": metrics(torch, values, expected_values, 0, 0),
        "a8_codes": metrics(torch, codes, expected_codes, 0, 0),
        "a8_scales": metrics(torch, scales, expected_scales, 0, 0),
        "a8_code_samples": {
            "actual_zero_count": int((codes == 0).sum()),
            "expected_zero_count": int((expected_codes_cpu == 0).sum()),
            "actual_row_prefixes": codes[:, :16].tolist(),
            "expected_row_prefixes": expected_codes_cpu[:, :16].tolist(),
        },
    }


def run_case(torch, torch_npu, args, source, weights, width, requested_blocks):
    (value_weight, value_tiles, output_weight, output_tiles, scale,
     vendor_scale, bias) = weights
    capacity = 16 if width <= 16 else 32
    latent = source[:width].clone()
    output = torch.empty((width, 6144), dtype=torch.bfloat16, device=args.device)
    scratch, tiling, blocks, sync_base = torch.ops.tilert.mla_tail_prepare(
        latent, capacity, requested_blocks,
    )

    def native():
        native_latent = latent.clone() if args.queued_producer else latent
        return torch.ops.tilert.mla_tail_out(
            native_latent, value_tiles, output_tiles, scale, bias, scratch, tiling,
            output, capacity, blocks, sync_base,
        )

    def vendor():
        vendor_latent = latent.clone() if args.queued_producer else latent
        expanded = torch_npu.npu_transpose_batchmatmul(
            vendor_latent, value_weight, perm_x1=(1, 0, 2), perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        ).reshape(width, 1024)
        codes, token_scale = torch_npu.npu_dynamic_quant(expanded)
        return torch_npu.npu_quant_matmul(
            codes, output_weight, vendor_scale, pertoken_scale=token_scale,
            bias=bias, output_dtype=torch.bfloat16,
        )

    stream = torch.npu.Stream(device=args.device)
    stream.wait_stream(torch.npu.current_stream(args.device))
    native_graph, native_output = capture(torch, native, stream)
    vendor_graph, vendor_output = capture(torch, vendor, stream)
    checks = []
    for replay in range(args.replays):
        # Include zero and sign changes so stale output or capture-time data
        # cannot pass by replaying a single constant tensor.
        multiplier = (1.0, 0.0, -1.0, 0.5)[replay % 4]
        with torch.npu.stream(stream):
            changed = (source[:width].float() * multiplier).to(torch.bfloat16)
            if replay >= 4:
                changed = changed.roll(replay, dims=-1)
            latent.copy_(changed)
            if replay % 2:
                vendor_graph.replay()
                native_graph.replay()
            else:
                native_graph.replay()
                vendor_graph.replay()
        stream.synchronize()
        result = metrics(torch, native_output, vendor_output, args.atol, args.rtol)
        result["replay"] = replay
        require_bitwise = args.atol == 0 and args.rtol == 0
        result["acceptance_requires_bitwise"] = require_bitwise
        result["pass"] = result["pass"] and (
            not require_bitwise or result["bitwise_equal"] is True)
        checks.append(result)
        if not result["pass"]:
            if args.output is not None:
                failure = args.output.with_name(
                    f"{args.output.stem}-m{width}-r{replay}.pt")
                failure.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    name: value.detach().cpu().clone(
                        memory_format=torch.contiguous_format)
                    for name, value in {
                        "actual": native_output, "reference": vendor_output,
                        "latent": latent, "vendor_scale": vendor_scale,
                        "native_scale": scale, "scratch": scratch,
                    }.items()
                }, failure)
                result["failure_tensors"] = str(failure)
            return {
                "width": width, "capacity": capacity, "blocks": blocks,
                "pass": False, "replay_checks": checks,
                "intermediates": inspect_intermediates(
                    torch, torch_npu, scratch, capacity, latent, value_weight,
                ),
            }
    if args.no_timing:
        return {
            "width": width, "capacity": capacity, "blocks": blocks,
            "pass": True, "replay_checks": checks,
        }
    native_times = []
    vendor_times = []
    for pair in range(args.timing_pairs):
        order = ((native_graph, native_times), (vendor_graph, vendor_times))
        if pair % 2:
            order = tuple(reversed(order))
        for graph, times in order:
            times.append(event_time_us(torch, graph, args.iterations, stream))
    return {
        "width": width, "capacity": capacity, "blocks": blocks, "pass": True,
        "replay_checks": checks,
        "native_graph_us": native_times,
        "vendor_graph_us": vendor_times,
        "native_graph_median_us": statistics.median(native_times),
        "vendor_graph_median_us": statistics.median(vendor_times),
    }


def main():
    args = parse_arguments()
    import torch
    import torch_npu

    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    torch.ops.load_library(str(args.library.resolve()))
    fixture = load_inputs(torch, args)
    validate_inputs(torch, fixture, max(args.widths))
    source = fixture["latent"].to(device=args.device, dtype=torch.bfloat16)
    value_weight = fixture["value_weight"].to(
        device=args.device, dtype=torch.bfloat16,
    ).contiguous()
    logical_output = fixture["output_weight"].to(
        device=args.device, dtype=torch.int8,
    ).contiguous()
    value_tiles = value_weight.reshape(4, 512, 4, 64)
    value_tiles = value_tiles.permute(0, 2, 1, 3).contiguous().reshape(16, 512, 64)
    value_tiles = torch_npu.npu_format_cast(value_tiles, 2)
    output_tiles = logical_output.reshape(1024, 24, 256).permute(1, 0, 2).contiguous()
    output_tiles = torch_npu.npu_format_cast(output_tiles, 2)
    output_weight = torch_npu.npu_format_cast(logical_output, 29)
    scale = fixture["output_scale"]
    if args.random_output_scale:
        generator = torch.Generator().manual_seed(830)
        mantissa = torch.rand(6144, generator=generator) + 0.5
        exponent = torch.randint(-14, -4, (6144,), generator=generator)
        scale = torch.ldexp(mantissa, exponent)
    vendor_scale = scale.to(getattr(torch, args.output_scale_dtype))
    vendor_scale = vendor_scale.to(args.device).reshape(6144).contiguous()
    scale = vendor_scale.float()
    bias = fixture.get("bias")
    if bias is not None:
        bias_type = torch.int32 if bias.dtype == torch.int32 else torch.float32
        bias = bias.to(device=args.device, dtype=bias_type).contiguous()
    weights = (value_weight, value_tiles, output_weight, output_tiles, scale,
               vendor_scale, bias)
    report = {
        "scope": "single-rank MLA projection tail; excludes SFA and TP reduction",
        "device": args.device,
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "library": str(args.library.resolve()),
        "fixture": str(args.fixture) if args.fixture else "synthetic-seed-829",
        "atol": args.atol, "rtol": args.rtol,
        "queued_producer": args.queued_producer,
        "queued_producer_scope": "both native and vendor",
        "output_scale_dtype": args.output_scale_dtype,
        "output_scale_distribution": (
            "seed-830 varied exponents" if args.random_output_scale else "fixture"),
        "provenance": provenance(torch, torch_npu, args),
        "cases": [],
    }
    print(json.dumps({key: value for key, value in report.items()
                      if key != "cases"}), flush=True)
    for width in args.widths:
        for blocks in args.blocks:
            result = run_case(torch, torch_npu, args, source, weights, width, blocks)
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
            if not result["pass"]:
                break
        if not report["cases"][-1]["pass"]:
            break
    report["pass"] = all(case["pass"] for case in report["cases"])
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
