"""Compare one native attention launch with separate native attention/tail."""

import json
from pathlib import Path
import statistics

from attention_check import (
    argument_parser, arguments, capture, fixture, metric, provenance,
    sha256, time_graph,
)


def full_arguments():
    here = Path(__file__).resolve().parent
    parser = argument_parser()
    parser.description = __doc__
    parser.set_defaults(library=here / "attention_full_build/whole_layer_full_attention.so")
    parser.add_argument("--attention-library", type=Path,
                        default=here / "attention_build/whole_layer_attention.so")
    parser.add_argument("--tail-library", type=Path,
                        default=here.parent / "mla_tail/build-composable/mla_tail_torch.so")
    return arguments(parser)


def tail_fixture(torch, args, inputs):
    required = ("value_weight", "output_weight", "output_scale")
    if args.fixture and any(name not in inputs for name in required):
        raise ValueError("full fixture also requires value_weight, output_weight, output_scale")
    if not args.fixture:
        generator = torch.Generator().manual_seed(523)
        inputs.update(
            value_weight=(torch.randn(4, 512, 256, generator=generator) / 32).to(torch.bfloat16),
            output_weight=torch.randint(-127, 128, (1024, 6144),
                                        generator=generator, dtype=torch.int8),
            output_scale=torch.full((6144,), 1 / 1024, dtype=torch.float32),
        )
    shapes = {
        "value_weight": ((4, 512, 256), torch.bfloat16),
        "output_weight": ((1024, 6144), torch.int8),
        "output_scale": ((6144,), torch.float32),
    }
    for name, (shape, dtype) in shapes.items():
        value = inputs[name]
        if (not torch.is_tensor(value) or value.shape != shape or value.dtype != dtype
                or not torch.isfinite(value).all()):
            raise ValueError(f"{name} requires finite {dtype} with shape {shape}")
    bias = inputs.get("bias")
    if bias is not None and (
        not torch.is_tensor(bias) or bias.shape != (6144,)
        or bias.dtype not in (torch.float32, torch.int32)
        or not torch.isfinite(bias).all()
    ):
        raise ValueError("bias requires finite INT32 or FP32 [6144]")


def run_case(torch, torch_npu, args, inputs, weights, rows, requested_blocks):
    def nd(value):
        return torch_npu.npu_format_cast(value.to(args.device).contiguous(), 2)

    def empty(shape, dtype):
        return nd(torch.empty(shape, dtype=dtype))

    table_pages = (args.sequence + rows + args.replays + 127) // 128
    pages = table_pages + 3
    generator = torch.Generator().manual_seed(12345 + rows)
    cache_base = nd(torch.randn(pages, 128, 1, 512, generator=generator).to(torch.bfloat16))
    rope_base = nd(torch.randn(pages, 128, 1, 64, generator=generator).to(torch.bfloat16))
    full_cache, separate_cache = nd(cache_base.clone()), nd(cache_base.clone())
    full_rope, separate_rope = nd(rope_base.clone()), nd(rope_base.clone())
    x, x_scale = empty((rows, 6144), torch.int8), empty((rows, 1), torch.float32)
    sin, cos = empty((rows, 64), torch.bfloat16), empty((rows, 64), torch.bfloat16)
    slots = empty((rows,), torch.int64)
    topk = empty((rows, 1, 2048), torch.int32)
    query_lengths = nd(torch.tensor([rows], dtype=torch.int32))
    kv_lengths = empty((1,), torch.int32)
    page_table = empty((1, table_pages), torch.int32)
    full_output = empty((rows, 6144), torch.bfloat16)
    separate_output = empty((rows, 6144), torch.bfloat16)
    latent = empty((rows, 4, 512), torch.bfloat16)
    full_scratch, full_tiling, blocks, full_sync = torch.ops.tilert.shared_full_attention_prepare(
        x, pages, table_pages, requested_blocks,
    )
    attention_scratch, attention_blocks, attention_sync = torch.ops.tilert.shared_attention_prepare(
        x, pages, table_pages, blocks,
    )
    capacity = 16 if rows <= 16 else 32
    tail_scratch, tail_tiling, tail_blocks, tail_sync = torch.ops.tilert.mla_tail_prepare(
        latent, capacity, blocks,
    )
    if attention_blocks != blocks or tail_blocks != blocks:
        raise RuntimeError("full and separate launches need the same core count")
    core_weights = [weights[name] for name in (
        "weight_dq", "weight_uq_qr", "weight_uk", "weight_dkv_kr",
        "scale_dq", "scale_uq_qr", "scale_dkv_kr", "gamma_cq", "gamma_ckv",
    )]
    value_tiles, output_tiles = weights["value_tiles"], weights["output_tiles"]
    output_scale, bias = weights["output_scale"], weights.get("bias")
    scale, epsilon = 1 / 16, 1e-5

    def full():
        return torch.ops.tilert.shared_full_attention_out(
            x, x_scale, *core_weights, sin, cos, slots, full_cache, full_rope,
            topk, query_lengths, kv_lengths, page_table,
            value_tiles, output_tiles, output_scale, bias,
            full_scratch, full_tiling, full_output, scale, epsilon, epsilon,
            blocks, full_sync,
        )

    def separate():
        torch.ops.tilert.shared_attention_out(
            x, x_scale, *core_weights, sin, cos, slots, separate_cache, separate_rope,
            topk, query_lengths, kv_lengths, page_table,
            attention_scratch, latent, scale, epsilon, epsilon,
            blocks, attention_sync,
        )
        return torch.ops.tilert.mla_tail_out(
            latent, value_tiles, output_tiles, output_scale, bias,
            tail_scratch, tail_tiling, separate_output, capacity, blocks, tail_sync,
        )

    def update(replay):
        prefix = args.sequence + replay
        positions = torch.arange(prefix, prefix + rows, dtype=torch.int64)
        table = torch.arange(pages, dtype=torch.int32).roll(replay + 1)[:table_pages]
        slots.copy_(nd(table[positions // 128].long() * 128 + positions % 128))
        page_table.copy_(nd(table.reshape(1, -1)))
        frequencies = torch.pow(10000.0, -torch.arange(32).float() / 32)
        angles = positions.float().unsqueeze(1) * frequencies
        sin.copy_(nd(angles.sin().repeat(1, 2).to(torch.bfloat16)))
        cos.copy_(nd(angles.cos().repeat(1, 2).to(torch.bfloat16)))
        selected = torch.arange(max(0, prefix + rows - 2048), prefix + rows,
                                dtype=torch.int32)
        indices = torch.full((rows, 1, 2048), -1, dtype=torch.int32)
        indices[:, 0, :selected.numel()] = selected if replay % 2 == 0 else selected.flip(0)
        topk.copy_(nd(indices))
        kv_lengths.copy_(nd(torch.tensor([prefix + rows], dtype=torch.int32)))
        multiplier = (1.0, 0.0, -1.0, 0.5)[replay % 4]
        hidden = (inputs["hidden"][:rows].float() * multiplier).to(torch.bfloat16)
        if replay >= 4:
            hidden = hidden.roll(replay, dims=1)
        codes, scales = torch_npu.npu_dynamic_quant(nd(hidden))
        x.copy_(codes)
        x_scale.copy_(scales.reshape(rows, 1))
        full_cache.copy_(cache_base)
        separate_cache.copy_(cache_base)
        full_rope.copy_(rope_base)
        separate_rope.copy_(rope_base)

    stream = torch.npu.Stream(device=args.device)
    stream.wait_stream(torch.npu.current_stream(args.device))
    with torch.npu.stream(stream):
        update(0)
    stream.synchronize()
    if args.capture:
        full_graph, _ = capture(torch, full, stream)
        separate_graph, _ = capture(torch, separate, stream)
    checks = []
    for replay in range(args.replays):
        with torch.npu.stream(stream):
            update(replay)
            if args.capture:
                calls = (full_graph.replay, separate_graph.replay)
            else:
                calls = (full, separate)
            for call in calls[::1 if replay % 2 == 0 else -1]:
                call()
        stream.synchronize()
        full_latent = full_scratch[:rows * 4 * 512 * 2].cpu()
        full_latent = full_latent.view(torch.bfloat16).reshape(rows, 4, 512)
        result = {
            "replay": replay,
            "latent": metric(torch, full_latent, latent),
            "output": metric(torch, full_output, separate_output, args.atol, args.rtol),
            "kv_cache": metric(torch, full_cache, separate_cache),
            "rope_cache": metric(torch, full_rope, separate_rope),
        }
        result["pass"] = all(value["pass"] for value in result.values() if isinstance(value, dict))
        checks.append(result)
        if not result["pass"]:
            break
    full_bytes = full_scratch.numel() + full_tiling.numel()
    separate_bytes = (attention_scratch.numel() + tail_scratch.numel() +
                      latent.numel() * latent.element_size() + tail_tiling.numel())
    report = {
        "width": rows, "blocks": blocks, "capture": args.capture,
        "sequence": args.sequence, "checks": checks,
        "pass": all(result["pass"] for result in checks),
        "requested_workspace_bytes": {"full": full_bytes, "separate": separate_bytes},
    }
    if report["pass"] and args.capture and not args.no_timing:
        full_times, separate_times = [], []
        for pair in range(args.timing_pairs):
            order = [(full_graph, full_times), (separate_graph, separate_times)]
            for graph, target in order[::1 if pair % 2 == 0 else -1]:
                target.append(time_graph(torch, graph, stream, args.iterations))
        report.update(full_graph_us=full_times, separate_graph_us=separate_times,
                      median_ratio=statistics.median(full_times) / statistics.median(separate_times))
    return report


def main():
    args = full_arguments()
    import torch
    import torch_npu

    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    for library in (args.library, args.attention_library, args.tail_library):
        torch.ops.load_library(str(library.resolve()))
    inputs = fixture(torch, args)
    tail_fixture(torch, args, inputs)
    weights = {}
    for name in ("weight_dq", "weight_uq_qr", "weight_dkv_kr", "weight_uk",
                 "scale_dq", "scale_uq_qr", "scale_dkv_kr", "gamma_cq", "gamma_ckv",
                 "output_scale", "bias"):
        if name not in inputs:
            continue
        form = 29 if name in ("weight_dq", "weight_uq_qr", "weight_dkv_kr") else 2
        weights[name] = torch_npu.npu_format_cast(inputs[name].to(args.device), form)
    values = inputs["value_weight"].reshape(4, 512, 4, 64).permute(0, 2, 1, 3)
    values = values.contiguous().reshape(16, 512, 64)
    outputs = inputs["output_weight"].reshape(1024, 24, 256).permute(1, 0, 2).contiguous()
    weights["value_tiles"] = torch_npu.npu_format_cast(values.to(args.device), 2)
    weights["output_tiles"] = torch_npu.npu_format_cast(outputs.to(args.device), 2)
    report = {
        "scope": "one native shared-indexer attention launch versus separate native attention/tail; before TP reduction",
        "atol": args.atol, "rtol": args.rtol,
        "provenance": provenance(torch, torch_npu, args),
        "comparison_libraries": {str(path): sha256(path) for path in
                                 (args.attention_library, args.tail_library)},
        "harness_sha256": sha256(Path(__file__)), "cases": [],
    }
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}), flush=True)
    for blocks in args.blocks:
        for width in args.widths:
            result = run_case(torch, torch_npu, args, inputs, weights, width, blocks)
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            if not result["pass"]:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
