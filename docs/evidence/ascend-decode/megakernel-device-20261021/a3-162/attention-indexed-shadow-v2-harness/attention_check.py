"""Compare shared-indexer native Prolog/SFA with the same-input CANN chain."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path(__file__).parent /
                        "attention_build/whole_layer_attention.so")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--widths", type=int, nargs="+", default=[1])
    parser.add_argument("--blocks", type=int, nargs="+", default=[0])
    parser.add_argument("--sequence", type=int, default=8192)
    parser.add_argument("--replays", type=int, default=4)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--timing-pairs", type=int, default=3)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    return parser


def arguments(parser=None):
    parser = argument_parser() if parser is None else parser
    args = parser.parse_args()
    if any(width < 1 or width > 32 for width in args.widths):
        parser.error("widths must lie in [1,32]")
    if any(blocks < 0 or blocks > 64 for blocks in args.blocks):
        parser.error("blocks must lie in [0,64]; zero selects the default")
    if min(args.replays, args.iterations, args.timing_pairs) <= 0:
        parser.error("replays, iterations and timing-pairs must be positive")
    if args.sequence < 0 or args.sequence > 131072:
        parser.error("sequence must lie in [0,131072]")
    if any(not math.isfinite(value) or value < 0
           for value in (args.atol, args.rtol)):
        parser.error("tolerances must be finite and nonnegative")
    return args


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(torch, torch_npu, args):
    versions = {}
    for name in ("CANN", "RUNTIME", "DRIVER"):
        try:
            versions[name] = torch_npu.utils.get_cann_version(name)
        except Exception as error:
            versions[name] = f"unavailable: {error}"
    build_report = args.library.parent / "provenance.json"
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(), "platform": platform.platform(),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "versions": versions,
        "device_properties": str(torch.npu.get_device_properties(args.device)),
        "library": str(args.library.resolve()),
        "library_sha256": sha256(args.library),
        "build_manifest": json.loads(build_report.read_text())
        if build_report.exists() else None,
        "harness_sha256": sha256(Path(__file__)),
        "fixture": str(args.fixture) if args.fixture else "synthetic-seed-20261021",
        "fixture_sha256": sha256(args.fixture) if args.fixture else None,
    }


def fixture(torch, args):
    if args.fixture:
        data = torch.load(args.fixture, map_location="cpu", weights_only=True)
    else:
        generator = torch.Generator().manual_seed(20261021)

        def randn(shape, scale=1.0):
            return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)

        def weight(shape):
            return torch.randint(-8, 8, shape, generator=generator, dtype=torch.int8)

        data = {
            "hidden": randn((32, 6144)),
            "weight_dq": weight((6144, 2048)),
            "weight_uq_qr": weight((2048, 1024)),
            "weight_uk": randn((4, 192, 512), 1 / 16),
            "weight_dkv_kr": weight((6144, 576)),
            "scale_dq": torch.full((1, 2048), 1 / 128),
            "scale_uq_qr": torch.full((1, 1024), 1 / 128),
            "scale_dkv_kr": torch.full((1, 576), 1 / 128),
            "gamma_cq": randn((2048,), 0.125) + 1,
            "gamma_ckv": randn((512,), 0.125) + 1,
        }
    shapes = {
        "weight_dq": ((6144, 2048), torch.int8),
        "weight_uq_qr": ((2048, 1024), torch.int8),
        "weight_uk": ((4, 192, 512), torch.bfloat16),
        "weight_dkv_kr": ((6144, 576), torch.int8),
        "scale_dq": ((1, 2048), torch.float32),
        "scale_uq_qr": ((1, 1024), torch.float32),
        "scale_dkv_kr": ((1, 576), torch.float32),
        "gamma_cq": ((2048,), torch.bfloat16),
        "gamma_ckv": ((512,), torch.bfloat16),
    }
    for name, (shape, dtype) in shapes.items():
        value = data.get(name)
        if (not torch.is_tensor(value) or value.shape != shape
                or value.dtype != dtype or not torch.isfinite(value).all()):
            raise ValueError(f"{name} requires finite {dtype} with shape {shape}")
    hidden = data.get("hidden")
    if (not torch.is_tensor(hidden) or hidden.dim() != 2
            or hidden.shape[1] != 6144 or hidden.shape[0] < max(args.widths)
            or hidden.dtype != torch.bfloat16 or not torch.isfinite(hidden).all()):
        raise ValueError("hidden requires finite BF16 [M,6144] with enough rows")
    return data


def metric(torch, actual, expected, atol=0.0, rtol=0.0):
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
    delta = (actual - expected).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return {
        "pass": finite and bool((delta <= atol + rtol * expected.abs()).all()),
        "exact": bool(torch.equal(actual, expected)),
        "bitwise_equal": different_bytes == 0 if different_bytes is not None else None,
        "bitwise_different_bytes": different_bytes,
        "max_abs": float(delta.max()),
        "rms": float(delta.square().mean().sqrt()),
        "different_elements": int((actual != expected).sum()),
        "elements": actual.numel(),
    }


def capture(torch, call, stream):
    with torch.npu.stream(stream):
        for _ in range(3):
            call()
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        result = call()
    return graph, result


def time_graph(torch, graph, stream, iterations):
    with torch.npu.stream(stream):
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        begin.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / iterations


def run_case(torch, torch_npu, args, weights, rows, requested_blocks):
    def nd(value):
        return torch_npu.npu_format_cast(value.to(args.device).contiguous(), 2)

    def empty(shape, dtype):
        return nd(torch.empty(shape, dtype=dtype))

    table_pages = (args.sequence + rows + args.replays + 127) // 128
    pages = table_pages + 3
    generator = torch.Generator().manual_seed(12345 + rows)
    cache_base = nd(torch.randn(pages, 128, 1, 512, generator=generator).to(torch.bfloat16))
    rope_base = nd(torch.randn(pages, 128, 1, 64, generator=generator).to(torch.bfloat16))
    cache_native, cache_vendor = nd(cache_base.clone()), nd(cache_base.clone())
    rope_native, rope_vendor = nd(rope_base.clone()), nd(rope_base.clone())
    x = empty((rows, 6144), torch.int8)
    x_scale = empty((rows, 1), torch.float32)
    sin, cos = empty((rows, 64), torch.bfloat16), empty((rows, 64), torch.bfloat16)
    slots = empty((rows,), torch.int64)
    topk = empty((rows, 1, 2048), torch.int32)
    query_lengths = nd(torch.tensor([rows], dtype=torch.int32))
    kv_lengths = empty((1,), torch.int32)
    page_table = empty((1, table_pages), torch.int32)
    output = empty((rows, 4, 512), torch.bfloat16)
    scratch, blocks, sync_base = torch.ops.tilert.shared_attention_prepare(
        x, pages, table_pages, requested_blocks,
    )
    scale, epsilon = 1 / 16, 1e-5
    ordered = [weights[name] for name in (
        "weight_dq", "weight_uq_qr", "weight_uk", "weight_dkv_kr",
        "scale_dq", "scale_uq_qr", "scale_dkv_kr", "gamma_cq", "gamma_ckv",
    )]

    def native():
        return torch.ops.tilert.shared_attention_out(
            x, x_scale, *ordered, sin, cos, slots, cache_native, rope_native,
            topk, query_lengths, kv_lengths, page_table, scratch, output,
            scale, epsilon, epsilon, blocks, sync_base,
        )

    def vendor():
        query, query_rope, _, _, _ = torch_npu.npu_mla_prolog_v3(
            token_x=x, weight_dq=weights["weight_dq"],
            weight_uq_qr=weights["weight_uq_qr"], weight_uk=weights["weight_uk"],
            weight_dkv_kr=weights["weight_dkv_kr"],
            rmsnorm_gamma_cq=weights["gamma_cq"],
            rmsnorm_gamma_ckv=weights["gamma_ckv"], rope_sin=sin, rope_cos=cos,
            cache_index=slots, kv_cache=cache_vendor, kr_cache=rope_vendor,
            dequant_scale_x=x_scale, dequant_scale_w_dq=weights["scale_dq"],
            dequant_scale_w_uq_qr=weights["scale_uq_qr"],
            dequant_scale_w_dkv_kr=weights["scale_dkv_kr"],
            rmsnorm_epsilon_cq=epsilon, rmsnorm_epsilon_ckv=epsilon,
            cache_mode="PA_BSND", query_norm_flag=False,
            weight_quant_mode=2, kv_cache_quant_mode=0, query_quant_mode=0,
            qc_qr_scale=1.0, kc_scale=1.0,
        )
        attention = torch_npu.npu_sparse_flash_attention(
            query, cache_vendor, cache_vendor, topk, scale,
            block_table=page_table, actual_seq_lengths_query=query_lengths,
            actual_seq_lengths_kv=kv_lengths, query_rope=query_rope,
            key_rope=rope_vendor, sparse_block_size=1,
            layout_query="TND", layout_kv="PA_BSND", sparse_mode=3,
            attention_mode=2, return_softmax_lse=False,
        )[0]
        return query, query_rope, attention

    def update(replay):
        prefix = args.sequence + replay
        positions = torch.arange(prefix, prefix + rows, dtype=torch.int64)
        table = torch.arange(pages, dtype=torch.int32).roll(replay + 1)[:table_pages]
        physical_slots = table[positions // 128].long() * 128 + positions % 128
        frequencies = torch.pow(10000.0, -torch.arange(32).float() / 32)
        angles = positions.float().unsqueeze(1) * frequencies
        sin.copy_(nd(angles.sin().repeat(1, 2).to(torch.bfloat16)))
        cos.copy_(nd(angles.cos().repeat(1, 2).to(torch.bfloat16)))
        selected = torch.arange(max(0, prefix + rows - 2048), prefix + rows,
                                dtype=torch.int32)
        if replay % 2:
            selected = selected.flip(0)
        indices = torch.full((rows, 1, 2048), -1, dtype=torch.int32)
        indices[:, 0, :selected.numel()] = selected
        topk.copy_(nd(indices))
        slots.copy_(nd(physical_slots))
        page_table.copy_(nd(table.reshape(1, -1)))
        kv_lengths.copy_(nd(torch.tensor([prefix + rows], dtype=torch.int32)))
        multiplier = (1.0, 0.0, -1.0, 0.5)[replay % 4]
        hidden = (weights["hidden"][:rows].float() * multiplier).to(torch.bfloat16)
        if replay >= 4:
            hidden = hidden.roll(replay, dims=1)
        codes, scales = torch_npu.npu_dynamic_quant(nd(hidden))
        x.copy_(codes)
        x_scale.copy_(scales.reshape(rows, 1))
        cache_native.copy_(cache_base)
        cache_vendor.copy_(cache_base)
        rope_native.copy_(rope_base)
        rope_vendor.copy_(rope_base)

    stream = torch.npu.Stream(device=args.device)
    stream.wait_stream(torch.npu.current_stream(args.device))
    with torch.npu.stream(stream):
        update(0)
    stream.synchronize()
    if args.capture:
        native_graph, native_output = capture(torch, native, stream)
        vendor_graph, vendor_outputs = capture(torch, vendor, stream)
    checks = []
    for replay in range(args.replays):
        with torch.npu.stream(stream):
            update(replay)
            if args.capture:
                first, second = (native_graph, vendor_graph) if replay % 2 == 0 else (
                    vendor_graph, native_graph
                )
                first.replay()
                second.replay()
            elif replay % 2 == 0:
                native_output = native()
                vendor_outputs = vendor()
            else:
                vendor_outputs = vendor()
                native_output = native()
        stream.synchronize()
        query_bytes = rows * 4 * 512 * 2
        rope_offset = (query_bytes + 511) // 512 * 512
        rope_bytes = rows * 4 * 64 * 2
        intermediates = scratch[:rope_offset + rope_bytes].cpu()
        query = intermediates[:query_bytes].view(torch.bfloat16).reshape(rows, 4, 512)
        query_rope = intermediates[rope_offset:].view(torch.bfloat16).reshape(rows, 4, 64)
        result = {
            "replay": replay,
            "query": metric(torch, query, vendor_outputs[0]),
            "query_rope": metric(torch, query_rope, vendor_outputs[1]),
            "kv_cache": metric(torch, cache_native, cache_vendor),
            "rope_cache": metric(torch, rope_native, rope_vendor),
            "attention": metric(torch, native_output, vendor_outputs[2], args.atol, args.rtol),
        }
        result["pass"] = all(value["pass"] for value in result.values() if isinstance(value, dict))
        checks.append(result)
        if not result["pass"]:
            break
    report = {
        "width": rows, "blocks": blocks, "sequence": args.sequence,
        "pages": pages, "table_pages": table_pages,
        "scratch_bytes": scratch.numel(), "capture": args.capture,
        "pass": all(result["pass"] for result in checks), "checks": checks,
    }
    if report["pass"] and args.capture and not args.no_timing:
        native_times, vendor_times = [], []
        for pair in range(args.timing_pairs):
            order = [(native_graph, native_times), (vendor_graph, vendor_times)]
            for graph, target in order[::1 if pair % 2 == 0 else -1]:
                target.append(time_graph(torch, graph, stream, args.iterations))
        report.update(native_graph_us=native_times, vendor_graph_us=vendor_times,
                      median_ratio=statistics.median(native_times) / statistics.median(vendor_times))
    return report


def main():
    args = arguments()
    import torch
    import torch_npu

    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    torch.ops.load_library(str(args.library.resolve()))
    inputs = fixture(torch, args)
    weights = {}
    for name, value in inputs.items():
        if name == "hidden":
            weights[name] = value
            continue
        physical_format = 29 if name in {
            "weight_dq", "weight_uq_qr", "weight_dkv_kr"
        } else 2
        weights[name] = torch_npu.npu_format_cast(value.to(args.device), physical_format)
    report = {
        "scope": "shared-indexer TP16 Prolog V3/cache/SFA; excludes input norm/A8, indexer, tail and TP reduction",
        "atol": args.atol, "rtol": args.rtol,
        "provenance": provenance(torch, torch_npu, args), "cases": [],
    }
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}), flush=True)
    for blocks in args.blocks:
        for width in args.widths:
            case = run_case(torch, torch_npu, args, weights, width, blocks)
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
            if args.output:
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            if not case["pass"]:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
