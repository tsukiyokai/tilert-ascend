"""Compare fused norm/attention with the two separate native stages."""

import json
from pathlib import Path
import statistics

from attention_check import (
    argument_parser, arguments, capture, fixture, metric, provenance,
    sha256, time_graph,
)
from attention_full_check import tail_fixture


def norm_arguments():
    here = Path(__file__).resolve().parent
    parser = argument_parser()
    parser.description = __doc__
    parser.set_defaults(
        library=here / "attention_norm_build/whole_layer_norm_attention.so",
    )
    parser.add_argument("--attention-library", type=Path,
                        default=here / "attention_full_build/"
                        "whole_layer_full_attention.so")
    parser.add_argument("--norm-library", type=Path,
                        default=here / "norm_build/whole_layer_norm.so")
    return arguments(parser)


def norm_fixture(torch, args, inputs):
    if not args.fixture:
        generator = torch.Generator().manual_seed(629)
        inputs["residual"] = torch.randn(
            inputs["hidden"].shape, generator=generator,
        ).bfloat16()
        inputs["norm_gamma"] = (
            1 + torch.randn(6144, generator=generator) / 8
        ).bfloat16()
    shapes = {"residual": inputs["hidden"].shape, "norm_gamma": (6144,)}
    if inputs.get("norm_beta") is not None:
        shapes["norm_beta"] = (6144,)
    for name, shape in shapes.items():
        value = inputs.get(name)
        if (not torch.is_tensor(value) or value.shape != shape
                or value.dtype != torch.bfloat16
                or not torch.isfinite(value).all()):
            raise ValueError(f"{name} requires finite BF16 with shape {shape}")


def prepare_weights(torch_npu, args, inputs):
    weights = {}
    for name in (
        "weight_dq", "weight_uq_qr", "weight_dkv_kr", "weight_uk",
        "scale_dq", "scale_uq_qr", "scale_dkv_kr", "gamma_cq", "gamma_ckv",
        "output_scale", "bias", "norm_gamma", "norm_beta",
    ):
        if inputs.get(name) is None:
            continue
        form = 29 if name in (
            "weight_dq", "weight_uq_qr", "weight_dkv_kr",
        ) else 2
        weights[name] = torch_npu.npu_format_cast(
            inputs[name].to(args.device), form,
        )
    values = inputs["value_weight"].reshape(4, 512, 4, 64)
    values = values.permute(0, 2, 1, 3).contiguous().reshape(16, 512, 64)
    outputs = inputs["output_weight"].reshape(1024, 24, 256)
    outputs = outputs.permute(1, 0, 2).contiguous()
    weights["value_tiles"] = torch_npu.npu_format_cast(
        values.to(args.device), 2,
    )
    weights["output_tiles"] = torch_npu.npu_format_cast(
        outputs.to(args.device), 2,
    )
    return weights


def run_case(torch, torch_npu, args, inputs, weights, rows, requested_blocks):
    def nd(value):
        return torch_npu.npu_format_cast(value.to(args.device).contiguous(), 2)

    def empty(shape, dtype):
        return nd(torch.empty(shape, dtype=dtype))

    table_pages = (args.sequence + rows + args.replays + 127) // 128
    pages = table_pages + 3
    generator = torch.Generator().manual_seed(12345 + rows)
    cache_base = nd(torch.randn(
        pages, 128, 1, 512, generator=generator,
    ).bfloat16())
    rope_base = nd(torch.randn(
        pages, 128, 1, 64, generator=generator,
    ).bfloat16())
    fused_cache = nd(cache_base.clone())
    separate_cache = nd(cache_base.clone())
    fused_rope, separate_rope = nd(rope_base.clone()), nd(rope_base.clone())
    hidden = empty((rows, 6144), torch.bfloat16)
    residual = empty((rows, 6144), torch.bfloat16)
    retained = empty((rows, 6144), torch.bfloat16)
    separate_retained = empty((rows, 6144), torch.bfloat16)
    normalized = empty((rows, 6144), torch.bfloat16)
    rstd = empty((rows, 1), torch.float32)
    codes = empty((rows, 6144), torch.int8)
    scales = empty((rows, 1), torch.float32)
    sin = empty((rows, 64), torch.bfloat16)
    cos = empty((rows, 64), torch.bfloat16)
    slots = empty((rows,), torch.int64)
    topk = empty((rows, 1, 2048), torch.int32)
    query_lengths = nd(torch.tensor([rows], dtype=torch.int32))
    kv_lengths = empty((1,), torch.int32)
    page_table = empty((1, table_pages), torch.int32)
    fused_output = empty((rows, 6144), torch.bfloat16)
    separate_output = empty((rows, 6144), torch.bfloat16)
    scratch, tiling, blocks, sync = (
        torch.ops.tilert.shared_norm_attention_prepare(
            hidden, pages, table_pages, requested_blocks,
        )
    )
    separate_scratch, separate_tiling, separate_blocks, separate_sync = (
        torch.ops.tilert.shared_full_attention_prepare(
            codes, pages, table_pages, blocks,
        )
    )
    norm_blocks, norm_sync = torch.ops.tilert.norm_prepare(hidden, blocks)
    if separate_blocks != blocks or norm_blocks != blocks:
        raise RuntimeError("all comparison stages need the same block count")
    core = [weights[name] for name in (
        "weight_dq", "weight_uq_qr", "weight_uk", "weight_dkv_kr",
        "scale_dq", "scale_uq_qr", "scale_dkv_kr", "gamma_cq", "gamma_ckv",
    )]
    tail = [weights["value_tiles"], weights["output_tiles"],
            weights["output_scale"], weights.get("bias")]
    gamma, beta = weights["norm_gamma"], weights.get("norm_beta")
    epsilon, scale = 1e-5, 1 / 16

    def fused():
        produced = hidden.clone()
        return torch.ops.tilert.shared_norm_attention_out(
            produced, residual, gamma, beta, retained, *core,
            sin, cos, slots, fused_cache, fused_rope, topk,
            query_lengths, kv_lengths, page_table, *tail,
            scratch, tiling, fused_output, epsilon, scale, epsilon, epsilon,
            blocks, sync,
        )

    def separate():
        produced = hidden.clone()
        torch.ops.tilert.norm_quant_out(
            produced, residual, gamma, beta, normalized, separate_retained,
            rstd, codes, scales, epsilon, norm_blocks, norm_sync,
        )
        return torch.ops.tilert.shared_full_attention_out(
            codes, scales, *core, sin, cos, slots, separate_cache,
            separate_rope, topk, query_lengths, kv_lengths, page_table, *tail,
            separate_scratch, separate_tiling, separate_output, scale,
            epsilon, epsilon, blocks, separate_sync,
        )

    def update(replay):
        prefix = args.sequence + replay
        positions = torch.arange(prefix, prefix + rows, dtype=torch.int64)
        table = torch.arange(pages, dtype=torch.int32).roll(replay + 1)
        table = table[:table_pages]
        slots.copy_(nd(table[positions // 128].long() * 128 + positions % 128))
        page_table.copy_(nd(table.reshape(1, -1)))
        frequencies = torch.pow(10000.0, -torch.arange(32).float() / 32)
        angles = positions.float().unsqueeze(1) * frequencies
        sin.copy_(nd(angles.sin().repeat(1, 2).bfloat16()))
        cos.copy_(nd(angles.cos().repeat(1, 2).bfloat16()))
        selected = torch.arange(max(0, prefix + rows - 2048), prefix + rows,
                                dtype=torch.int32)
        if replay % 2:
            selected = selected.flip(0)
        indices = torch.full((rows, 1, 2048), -1, dtype=torch.int32)
        indices[:, 0, :selected.numel()] = selected
        topk.copy_(nd(indices))
        kv_lengths.copy_(nd(torch.tensor([prefix + rows], dtype=torch.int32)))
        live_hidden = inputs["hidden"][:rows].roll(replay, dims=1)
        live_residual = inputs["residual"][:rows].roll(-replay, dims=1)
        if replay % 4 == 1:
            live_residual = -live_hidden
        elif replay % 4 == 2:
            live_hidden = -live_hidden
        elif replay % 4 == 3:
            live_hidden = (live_hidden.float() / 16).bfloat16()
        hidden.copy_(nd(live_hidden))
        residual.copy_(nd(live_residual))
        fused_cache.copy_(cache_base)
        separate_cache.copy_(cache_base)
        fused_rope.copy_(rope_base)
        separate_rope.copy_(rope_base)

    stream = torch.npu.Stream(device=args.device)
    stream.wait_stream(torch.npu.current_stream(args.device))
    with torch.npu.stream(stream):
        update(0)
    stream.synchronize()
    if args.capture:
        fused_graph, _ = capture(torch, fused, stream)
        separate_graph, _ = capture(torch, separate, stream)
    checks = []
    scale_offset = ((rows * 6144 + 511) // 512) * 512
    for replay in range(args.replays):
        with torch.npu.stream(stream):
            update(replay)
            calls = ((fused_graph.replay, separate_graph.replay)
                     if args.capture else (fused, separate))
            for call in calls[::1 if replay % 2 == 0 else -1]:
                call()
        stream.synchronize()
        observed_codes = scratch[:rows * 6144].cpu().view(torch.int8)
        observed_scales = scratch[scale_offset:scale_offset + rows * 4]
        observed_scales = observed_scales.cpu().view(torch.float32)
        result = {
            "replay": replay,
            "codes": metric(torch, observed_codes.reshape(rows, 6144), codes),
            "scales": metric(torch, observed_scales.reshape(rows, 1), scales),
            "retained": metric(torch, retained, separate_retained),
            "output": metric(torch, fused_output, separate_output,
                             args.atol, args.rtol),
            "kv_cache": metric(torch, fused_cache, separate_cache),
            "rope_cache": metric(torch, fused_rope, separate_rope),
        }
        result["pass"] = all(value["pass"] for value in result.values()
                             if isinstance(value, dict))
        checks.append(result)
        if not result["pass"]:
            break
    temporary = sum(tensor.numel() * tensor.element_size() for tensor in (
        normalized, rstd, codes, scales,
    ))
    report = {
        "rows": rows, "blocks": blocks, "sequence": args.sequence,
        "checks": checks, "pass": all(result["pass"] for result in checks),
        "requested_workspace_bytes": {
            "fused": scratch.numel() + tiling.numel(),
            "separate": separate_scratch.numel() + separate_tiling.numel() +
            temporary,
        },
    }
    if report["pass"] and args.capture and not args.no_timing:
        pairs = []
        for pair in range(args.timing_pairs):
            order = [("fused", fused_graph), ("separate", separate_graph)]
            times = {}
            for name, graph in order[::1 if pair % 2 == 0 else -1]:
                times[name] = time_graph(torch, graph, stream, args.iterations)
            pairs.append(times)
        report["timing_us_per_replay"] = {
            "pairs": pairs,
            "median_fused": statistics.median(x["fused"] for x in pairs),
            "median_separate": statistics.median(x["separate"] for x in pairs),
        }
    return report


def main():
    args = norm_arguments()
    import torch
    import torch_npu

    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    for library in (args.library, args.attention_library, args.norm_library):
        torch.ops.load_library(str(library.resolve()))
    inputs = fixture(torch, args)
    tail_fixture(torch, args, inputs)
    norm_fixture(torch, args, inputs)
    weights = prepare_weights(torch_npu, args, inputs)
    report = {
        "scope": "one native norm/shared-attention launch versus separate "
        "native norm/full-attention; before TP reduction",
        "atol": args.atol, "rtol": args.rtol,
        "provenance": provenance(torch, torch_npu, args),
        "comparison_libraries": {str(path): sha256(path) for path in
                                 (args.attention_library, args.norm_library)},
        "harness_sha256": sha256(Path(__file__)), "cases": [],
    }
    try:
        for blocks in args.blocks:
            for rows in args.widths:
                result = run_case(
                    torch, torch_npu, args, inputs, weights, rows, blocks,
                )
                report["cases"].append(result)
                print(json.dumps(result), flush=True)
                if not result["pass"]:
                    raise RuntimeError("norm-attention differential failed")
        report["pass"] = True
    except Exception as error:
        report["pass"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
