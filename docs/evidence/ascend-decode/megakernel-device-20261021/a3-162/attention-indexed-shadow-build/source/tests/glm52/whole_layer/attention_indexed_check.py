"""Compare one indexed-attention launch with its separately validated stages."""

import argparse
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

from attention_check import capture, fixture, metric, provenance
from attention_check import sha256, time_graph
from attention_export_check import AttentionInputs, load_weights, WEIGHTS
from attention_full_check import tail_fixture
from indexer_front_check import fixture as front_fixture
from indexer_front_check import layout as front_layout


HERE = Path(__file__).resolve().parent
FRONT_WEIGHTS = (
    "query_weight", "query_weight_scale", "key_weight", "gamma", "beta",
    "hadamard",
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=(
        HERE / "attention_indexed_build-scale-order" /
        "whole_layer_indexed_attention.so"))
    parser.add_argument("--front-library", type=Path, default=(
        HERE / "indexer_front_build-scale-order" /
        "whole_layer_indexer_front.so"))
    parser.add_argument("--profile-library", type=Path)
    parser.add_argument("--candidate-library", type=Path)
    parser.add_argument("--shadow-library", type=Path)
    parser.add_argument("--negative-main-slot", action="store_true")
    parser.add_argument("--compare-placements", action="store_true")
    parser.add_argument("--profile-samples", type=int, default=30)
    parser.add_argument("--indexer-library", type=Path, default=HERE /
                        "indexer_build-default/whole_layer_indexer.so")
    parser.add_argument("--tail-library", type=Path, default=HERE.parent /
                        "mla_tail/build-dequant-promoted/mla_tail_torch.so")
    parser.add_argument("--shared-library", type=Path, default=HERE /
                        "attention_full_build-dequant/whole_layer_full_attention.so")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--sequence", type=int, default=8192)
    parser.add_argument("--widths", type=int, nargs="+", default=[1])
    parser.add_argument("--blocks", type=int, nargs="+", default=[0])
    parser.add_argument("--modes", nargs="+", choices=["int8", "bf16"],
                        default=["int8", "bf16"])
    parser.add_argument("--replays", type=int, default=4)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--queued-producer", action="store_true")
    parser.add_argument("--engine-adapter", action="store_true",
                        help="Use the TileRT body, live slot casts and capture pool")
    parser.add_argument("--query-scale-dtype", choices=["float32", "bfloat16"],
                        default="float32")
    parser.add_argument("--output-scale-dtype", choices=["float32", "bfloat16"],
                        default="float32")
    parser.add_argument("--banks", type=int, default=1)
    parser.add_argument("--calls", type=int, default=1)
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--timing-pairs", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(row < 1 or row > 32 for row in args.widths):
        parser.error("widths must lie in [1,32]")
    if any(block < 0 or block > 64 for block in args.blocks):
        parser.error("blocks must lie in [0,64]")
    if min(args.replays, args.iterations, args.timing_pairs,
           args.profile_samples) < 1:
        parser.error("replay and timing counts must be positive")
    if not 1 <= args.sequence <= 131072:
        parser.error("sequence must lie in [1,131072]")
    if args.engine_adapter and not args.no_timing:
        parser.error("engine adapter comparisons require --no-timing")
    if sum(bool(path) for path in (args.profile_library, args.candidate_library,
                                  args.shadow_library)) > 1:
        parser.error("choose one profile, candidate or shadow library")
    args.comparison_library = (args.profile_library or args.candidate_library
                               or args.shadow_library)
    args.comparison_label = ("profile" if args.profile_library else
                             "shadow" if args.shadow_library else "candidate")
    args.comparison_namespace = {
        "profile": "tilert_indexed_profile", "candidate": "tilert_indexed_parallel",
        "shadow": "tilert_indexed_shadow",
    }[args.comparison_label]
    if args.comparison_library and (args.engine_adapter or not args.capture):
        parser.error("native comparisons require capture without the engine adapter")
    if args.compare_placements and (
            not args.candidate_library or args.queued_producer or args.no_timing):
        parser.error("placement comparisons require a timed candidate without queued clones")
    if args.negative_main_slot and not args.shadow_library:
        parser.error("negative-main-slot is a packed-shadow numerical probe")
    if (not 1 <= args.banks <= 8 or not 1 <= args.calls <= 64
            or args.calls % args.banks):
        parser.error("banks must be 1..8 and divide calls in 1..64")
    if args.engine_adapter and (args.banks != 1 or args.calls != 1):
        parser.error("engine adapter comparisons require one bank and one call")
    return args


def inputs_fixture(torch, args):
    data = fixture(torch, args)
    tail_fixture(torch, args, data)
    if not args.fixture:
        front = front_fixture(torch, SimpleNamespace(
            fixture=None, widths=args.widths))
        data.update({name: front[name] for name in FRONT_WEIGHTS})
    shapes = (
        ((2048, 4096), torch.int8), ((4096,), torch.float32),
        ((6144, 160), torch.bfloat16), ((128,), torch.float32),
        ((128,), torch.float32), ((128, 128), torch.bfloat16),
    )
    for name, (shape, dtype) in zip(FRONT_WEIGHTS, shapes):
        value = data.get(name)
        if (not torch.is_tensor(value) or value.shape != shape
                or value.dtype != dtype or not bool(torch.isfinite(value).all())):
            raise ValueError(f"{name} requires finite {dtype} {shape}")
    data["query_weight_scale"] = data["query_weight_scale"].to(
        getattr(torch, args.query_scale_dtype))
    data["output_scale"] = data["output_scale"].to(
        getattr(torch, args.output_scale_dtype))
    return data


def packed_weights(torch_npu, args, data):
    def nd(value):
        return torch_npu.npu_format_cast(value.to(args.device).contiguous(), 2)

    main = load_weights(torch_npu, args, {
        name: data[name] for name in (*WEIGHTS, "hidden")})
    if args.engine_adapter:
        from tilert.models.glm_5_2_ascend.attention_native import (
            pack_indexer_weights,
        )

        query_tiles, key_tiles = pack_indexer_weights(
            data["query_weight"], data["key_weight"].T,
        )
    else:
        query_tiles = data["query_weight"].reshape(2048, 16, 256).permute(1, 0, 2)
        key_tiles = data["key_weight"].reshape(6144, 5, 32).permute(1, 0, 2)
    front = [nd(query_tiles), nd(data["query_weight_scale"]), nd(key_tiles),
             *(nd(data[name]) for name in ("gamma", "beta", "hadamard"))]
    values = data["value_weight"].reshape(4, 512, 4, 64).permute(0, 2, 1, 3)
    tail = [nd(values.contiguous().reshape(16, 512, 64)),
            nd(data["output_weight"].reshape(1024, 24, 256).permute(1, 0, 2)),
            nd(data["output_scale"].float())]
    bias = nd(data["bias"]) if data.get("bias") is not None else None
    if args.engine_adapter and bias is not None:
        raise ValueError("The TileRT attention adapter requires an unbiased output")
    return main, front, tail, bias


def decode_front(torch, cpu, offsets, rows, quantized, prefix=""):
    shapes = {
        "query_projection": (rows, 32, 128), "key_projection": (rows, 160),
        "key_normalized": (rows, 128), "query_rope": (rows, 32, 128),
        "key_rope": (rows, 128), "norm_stats": (rows, 2),
    }
    if quantized:
        shapes.update(query_hadamard=(rows, 32, 128),
                      key_hadamard=(rows, 128))
    result = {}
    for name, shape in shapes.items():
        dtype = torch.float32 if name == "norm_stats" or (
            "hadamard" in name) else torch.bfloat16
        count = 1
        for dimension in shape:
            count *= dimension
        start = offsets[prefix + name]
        size = count * (4 if dtype == torch.float32 else 2)
        value = cpu[start:start + size].view(dtype).reshape(shape)
        result[name] = value.bfloat16() if "hadamard" in name else value
    return result


def tensor_from_bytes(torch, cpu, offset, shape, dtype):
    count = 1
    for dimension in shape:
        count *= dimension
    size = count * torch.empty((), dtype=dtype).element_size()
    return cpu[offset:offset + size].view(dtype).reshape(shape)


def run_case(torch, torch_npu, args, banks, rows, mode, requested):
    weights, front_weights, tail_weights, bias = banks[0]
    sequence = banks * (args.calls // args.banks)
    common = AttentionInputs(torch, torch_npu, args, weights, rows)
    nd, empty = common.nd, common.empty
    quantized = mode == "int8"
    index_table_pages = common.table_pages + 1
    index_pages = index_table_pages + 5
    generator = torch.Generator().manual_seed(526 + rows)
    if quantized:
        cache_base = nd(torch.randint(-127, 128, (index_pages, 128, 1, 128),
                                     generator=generator, dtype=torch.int8))
        scale_base = nd((torch.rand(index_pages, 128, generator=generator)
                         / 128 + 1 / 64).half())
    else:
        cache_base = nd(torch.randn(index_pages, 128, 1, 128,
                                   generator=generator).bfloat16())
        scale_base = empty((0,), torch.float16)
    index_caches = [nd(cache_base.clone()) for _ in range(2)]
    index_scales = [nd(scale_base.clone()) for _ in range(2)]
    index_sine = empty((rows, 64), torch.bfloat16)
    index_cosine = empty((rows, 64), torch.bfloat16)
    index_slots = empty((rows,), torch.int64)
    index_qlen = nd(torch.tensor([rows], dtype=torch.int32))
    index_klen = empty((1,), torch.int32)
    index_table = empty((1, index_table_pages), torch.int32)
    main_metadata = [common.sine, common.cosine, common.slots,
                     common.query_lengths, common.key_lengths, common.page_table]
    index_metadata = [index_sine, index_cosine, index_slots,
                      index_qlen, index_klen, index_table]
    scratch, tiling, offsets, blocks, sync = (
        torch.ops.tilert.indexed_attention_prepare(
            common.codes, common.pages, common.table_pages,
            index_pages, index_table_pages, quantized, requested))
    output = empty((rows, 6144), torch.bfloat16)
    front_scratch, front_tiling, front_outputs, front_blocks, front_sync = (
        torch.ops.tilert.indexer_front_prepare(common.hidden, quantized, blocks))
    index_scratch, index_blocks, index_sync = (
        torch.ops.tilert.indexer_prepare(
            front_outputs[0], index_pages, index_table_pages, blocks))
    indices = empty((rows, 1, 2048), torch.int32)
    reference_output = empty((rows, 6144), torch.bfloat16)
    latent_example = empty((rows, 4, 512), torch.bfloat16)
    capacity = offsets["capacity"]
    tail_scratch, tail_tiling, tail_blocks, tail_sync = (
        torch.ops.tilert.mla_tail_prepare(latent_example, capacity, blocks))
    if {front_blocks, index_blocks, tail_blocks} != {blocks}:
        raise ValueError("composition and reference require identical core counts")
    if scratch.nbytes != offsets["bytes"]:
        raise ValueError("native scratch allocation and description differ")
    if args.comparison_library:
        comparison_ops = getattr(torch.ops, args.comparison_namespace)
        prepared_comparison = comparison_ops.indexed_attention_prepare(
            common.codes, common.pages, common.table_pages,
            index_pages, index_table_pages, quantized, blocks)
        (comparison_scratch, comparison_tiling, comparison_offsets,
         pblocks, psync) = prepared_comparison
        if (pblocks, psync) != (blocks, sync):
            raise ValueError("comparison and ordinary launches require the same context")
        for name, offset in offsets.items():
            if name != "bytes" and comparison_offsets[name] != offset:
                raise ValueError(f"comparison changed the ordinary {name} offset")
        comparison_caches = [empty(value.shape, value.dtype) for value in (
            common.caches[0], common.ropes[0], index_caches[0], index_scales[0])]
        if args.shadow_library:
            comparison_caches.append(empty((common.pages, 128, 576), torch.bfloat16))
        comparison_output = empty((rows, 6144), torch.bfloat16)
    front_offsets, expected_front_bytes = front_layout(rows)
    if front_scratch.nbytes != expected_front_bytes:
        raise ValueError("front diagnostic layout differs from its allocation")
    adapter_state = {}
    if args.engine_adapter:
        from tilert.models.glm_5_2_ascend import attention_native as adapter

        pool = adapter.WorkspacePool(blocks=requested)
        adapter_main = list(main_metadata)
        adapter_index = list(index_metadata)
        adapter_main[2] = empty((rows,), torch.int32)
        adapter_index[2] = empty((rows,), torch.int32)
        adapter_caches = [common.caches[0], common.ropes[0],
                          index_caches[0], index_scales[0]]
        geometry = adapter.geometry_for(
            mode, adapter_caches, adapter_main, adapter_index,
        )
        persistent_topk = empty((rows, 1, 2048), torch.int32)
        shared_output = empty((rows, 6144), torch.bfloat16)
        pool.reserve(common.hidden, geometry,
                     [adapter_main[2], adapter_index[2]])
        shared_geometry = adapter.geometry_for(
            "shared", adapter_caches[:2], adapter_main,
        )
        pool.reserve(common.hidden, shared_geometry, [adapter_main[2]])

    def full_one(bank):
        weights, front_weights, tail_weights, bias = bank
        hidden = common.hidden.clone() if args.queued_producer else common.hidden
        if args.engine_adapter:
            partial, selected = adapter.run_body(
                hidden, [weights[name] for name in WEIGHTS], front_weights,
                tail_weights, adapter_caches, adapter_main, adapter_index,
                pool, mode=mode, scale=common.scale,
                epsilon_cq=common.epsilon, epsilon_ckv=common.epsilon,
            )
            output.copy_(partial)
            persistent_topk.copy_(selected)
            adapter_state["prepared"] = pool.get(hidden, geometry)
            shared_partial, _ = adapter.run_body(
                hidden, [weights[name] for name in WEIGHTS], [], tail_weights,
                adapter_caches[:2], adapter_main, None, pool, mode="shared",
                scale=common.scale, epsilon_cq=common.epsilon,
                epsilon_ckv=common.epsilon, topk=persistent_topk,
            )
            shared_output.copy_(shared_partial)
            return output
        codes, scales = common.input_pair()
        return torch.ops.tilert.indexed_attention_out(
            hidden, codes, scales, [weights[name] for name in WEIGHTS],
            front_weights, tail_weights, bias, main_metadata, index_metadata,
            [common.caches[0], common.ropes[0], index_caches[0], index_scales[0]],
            scratch, tiling, output, common.scale, common.epsilon,
            common.epsilon, blocks, sync,
        )

    def full():
        for bank in sequence:
            result = full_one(bank)
        return result

    def native_sequence(ops, arena):
        for weights, front, tail, bias in sequence:
            hidden = (common.hidden.clone() if args.queued_producer
                      else common.hidden)
            codes, scales = common.input_pair()
            ops.indexed_attention_out(
                hidden, codes, scales, [weights[name] for name in WEIGHTS],
                front, tail, bias, main_metadata, index_metadata,
                arena["caches"], arena["scratch"], arena["tiling"],
                arena["output"],
                common.scale, common.epsilon, common.epsilon, blocks, sync,
            )
        return arena["output"]

    def compared():
        return native_sequence(comparison_ops, {
            "caches": comparison_caches, "scratch": comparison_scratch,
            "tiling": comparison_tiling, "output": comparison_output,
        })

    def separate_one(bank):
        weights, front_weights, tail_weights, bias = bank
        common.weights = weights
        hidden = common.hidden.clone() if args.queued_producer else common.hidden
        query, rope, qnorm, qscale = common.vendor_prolog()
        qnorm, qscale = nd(qnorm), nd(qscale)
        torch.ops.tilert.indexer_front_out(
            hidden, qnorm, qscale, *front_weights[:5], index_cosine, index_sine,
            front_weights[5], front_scratch, front_tiling, front_outputs,
            quantized, blocks, front_sync,
        )
        q, key, weight, qs, ks = front_outputs
        torch.ops.tilert.indexer_out(
            q, key, weight, qs if quantized else None,
            ks if quantized else None, index_slots, index_caches[1],
            index_scales[1] if quantized else None, index_qlen, index_klen,
            index_table, index_scratch, indices, blocks, index_sync,
        )
        latent = nd(common.vendor_sfa(query, rope, indices))
        torch.ops.tilert.mla_tail_out(
            latent, *tail_weights, bias, tail_scratch, tail_tiling,
            reference_output, capacity, blocks, tail_sync,
        )
        return {"attention_query": query, "attention_rope": rope,
                "query_norm": qnorm, "query_norm_scale": qscale,
                "latent": latent, "output": reference_output,
                "query": q, "key": key, "weights": weight, "topk": indices,
                **({"query_scale": qs, "key_scale": ks} if quantized else {})}

    def separate():
        for bank in sequence:
            result = separate_one(bank)
        return result

    def update(replay):
        common.update(replay)
        if args.negative_main_slot:
            common.slots[-1] = -1
        prefix = args.sequence + replay
        positions = torch.arange(prefix, prefix + rows, dtype=torch.int64)
        table = torch.arange(index_pages, dtype=torch.int32).roll(replay + 3)
        table = table[:index_table_pages]
        slots = table[positions // 128].long() * 128 + positions % 128
        index_slots.copy_(nd(slots))
        if args.engine_adapter:
            adapter_main[2].copy_(common.slots)
            adapter_index[2].copy_(index_slots)
        index_table.copy_(nd(table.reshape(1, -1)))
        index_klen.copy_(nd(torch.tensor([prefix + rows], dtype=torch.int32)))
        index_sine.copy_(common.sine)
        index_cosine.copy_(common.cosine)
        for cache, scales in zip(index_caches, index_scales):
            cache.copy_(cache_base)
            scales.copy_(scale_base)
        if args.comparison_library:
            for destination, source in zip(comparison_caches, (
                    common.caches[0], common.ropes[0], index_caches[0],
                    index_scales[0])):
                destination.copy_(source)
            if args.shadow_library:
                comparison_caches[4].copy_(torch.cat(
                    (comparison_caches[0].squeeze(2),
                     comparison_caches[1].squeeze(2)), dim=-1))

    def shadow_bytes():
        expected = torch.cat((comparison_caches[0].cpu().squeeze(2),
                              comparison_caches[1].cpu().squeeze(2)), dim=-1)
        actual = comparison_caches[4].cpu()
        differences = int((actual.view(torch.uint8) !=
                           expected.view(torch.uint8)).sum())
        return {"pass": differences == 0, "bitwise_equal": differences == 0,
                "bitwise_different_bytes": differences, "bytes": actual.nbytes}

    stream = torch.npu.Stream(device=args.device)
    stream.wait_stream(torch.npu.current_stream(args.device))
    with torch.npu.stream(stream):
        update(0)
    stream.synchronize()
    initial_shadow = shadow_bytes() if args.shadow_library else None
    if initial_shadow is not None and not initial_shadow["pass"]:
        raise RuntimeError("initial shadow differs from the complete original caches")
    if args.capture:
        if args.engine_adapter:
            full()
            torch.npu.current_stream(args.device).synchronize()
            eager_scratch = adapter_state["prepared"][0].data_ptr()
            full_graph = torch.npu.NPUGraph()
            with torch.npu.graph(full_graph, stream=stream):
                full()
            if (adapter_state["prepared"][0].data_ptr() == eager_scratch
                    or pool.capture_reserves or pool.slot_reserves):
                raise RuntimeError("Capture did not claim independent reserved storage")
        else:
            full_graph, _ = capture(torch, full, stream)
        separate_graph, reference = capture(torch, separate, stream)
        if args.comparison_library:
            comparison_graph, _ = capture(torch, compared, stream)
    checks = []
    for replay in range(args.replays):
        with torch.npu.stream(stream):
            update(replay)
            if args.capture:
                calls = (full_graph.replay, separate_graph.replay)
                for call in calls[::1 if replay % 2 == 0 else -1]:
                    call()
                if args.comparison_library:
                    comparison_graph.replay()
            else:
                full()
                reference = separate()
        stream.synchronize()
        if args.engine_adapter:
            scratch, tiling, blocks, sync, _, _ = adapter_state["prepared"]
        cpu = scratch.cpu()
        observed = {"output": output}
        shapes = {
            "attention_query": ((rows, 4, 512), torch.bfloat16),
            "attention_rope": ((rows, 4, 64), torch.bfloat16),
            "query_norm": ((rows, 2048), torch.int8),
            "query_norm_scale": ((rows,), torch.float32),
            "latent": ((rows, 4, 512), torch.bfloat16),
            "query": ((rows, 32, 128), torch.int8 if quantized else torch.bfloat16),
            "key": ((rows, 128), torch.int8 if quantized else torch.bfloat16),
            "weights": ((rows, 32), torch.float16 if quantized else torch.bfloat16),
            "topk": ((rows, 1, 2048), torch.int32),
        }
        if quantized:
            shapes.update(query_scale=((rows, 32), torch.float16),
                          key_scale=((rows,), torch.float16))
        for name, (shape, dtype) in shapes.items():
            observed[name] = tensor_from_bytes(torch, cpu, offsets[name], shape,
                                               dtype)
        observed.update(decode_front(torch, cpu, offsets, rows, quantized,
                                     "front_"))
        reference.update(decode_front(
            torch, front_scratch.cpu(),
            {name: region[0] for name, region in front_offsets.items()},
            rows, quantized))
        cache_pairs = {
            "kv_cache": common.caches, "rope_cache": common.ropes,
            "indexer_cache": index_caches,
        }
        if quantized:
            cache_pairs["indexer_scale_cache"] = index_scales
        for name, pair in cache_pairs.items():
            observed[name], reference[name] = pair
        if args.engine_adapter:
            observed["persistent_topk"] = persistent_topk
            reference["persistent_topk"] = indices
            observed["shared_consumer_output"] = shared_output
            reference["shared_consumer_output"] = reference_output
        metrics = {name: metric(torch, observed[name], value)
                   for name, value in reference.items()}
        if args.comparison_library:
            comparison_cpu = comparison_scratch.cpu()
            comparison_values = {"output": comparison_output}
            for name, (shape, dtype) in shapes.items():
                comparison_values[name] = tensor_from_bytes(
                    torch, comparison_cpu, comparison_offsets[name], shape, dtype)
            comparison_values.update(decode_front(
                torch, comparison_cpu, comparison_offsets, rows, quantized, "front_"))
            for name, value in zip(("kv_cache", "rope_cache", "indexer_cache",
                                    "indexer_scale_cache"), comparison_caches):
                if name in observed:
                    comparison_values[name] = value
            metrics.update({
                args.comparison_label + "/" + name: metric(
                    torch, value, observed[name])
                for name, value in comparison_values.items()
            })
            if args.shadow_library:
                metrics["shadow/cache_bytes"] = shadow_bytes()
            if args.profile_library:
                from attention_indexed_profile import read_trace

                read_trace(torch, comparison_cpu, comparison_offsets, rows, blocks)
        passed = all(item["pass"] and item["bitwise_equal"]
                     for name, item in metrics.items()
                     if name.rsplit("/", 1)[-1] != "norm_stats")
        result = {"replay": replay, "pass": passed, "boundaries": metrics}
        checks.append(result)
        if not passed:
            print(json.dumps({"mode": mode, "rows": rows, **result}), flush=True)
            path = args.output.with_name(
                f"{args.output.stem}-{mode}-m{rows}-r{replay}.pt")

            def cpu_copy(value):
                return value.detach().cpu().clone(
                    memory_format=torch.contiguous_format)

            saved_inputs = common.retained_inputs()
            saved_inputs.update({f"indexer_metadata_{i}": cpu_copy(value)
                                 for i, value in enumerate(index_metadata)})
            saved_inputs.update({f"front_weight_{i}": cpu_copy(value)
                                 for i, value in enumerate(front_weights)})
            saved_inputs.update({f"tail_weight_{i}": cpu_copy(value)
                                 for i, value in enumerate(tail_weights)})
            if bias is not None:
                saved_inputs["bias"] = cpu_copy(bias)
            try:
                torch.save({"inputs": saved_inputs, "offsets": dict(offsets),
                            "scratch": cpu_copy(cpu),
                            "front_scratch": cpu_copy(front_scratch),
                            "observed": {name: cpu_copy(value) for name, value
                                         in observed.items()},
                            "reference": {name: cpu_copy(value) for name, value
                                          in reference.items()},
                            "comparison": ({name: cpu_copy(value) for name, value
                                            in comparison_values.items()}
                                           if args.comparison_library else None)},
                           path)
                result["failure_tensors"] = str(path)
            except Exception as error:
                result["failure_dump_error"] = f"{type(error).__name__}: {error}"
            break
    separate_bytes = sum(value.nbytes for value in (
        front_scratch, front_tiling, index_scratch, tail_scratch, tail_tiling,
        indices, latent_example, *front_outputs))
    report = {
        "mode": mode, "rows": rows, "blocks": blocks,
        "sequence": args.sequence, "capture": args.capture,
        "queued_producer": args.queued_producer, "checks": checks,
        "engine_adapter": args.engine_adapter,
        "weight_bank_graph": {
            "banks": args.banks, "calls": args.calls,
            "scope": (
                "Independent calls with equal-valued weights at distinct "
                "addresses; shared input, metadata, scratch and caches. "
                "Retained boundaries belong to the final call. No hidden-state "
                "dependency between calls and no TP reduction."),
        },
        "pass": all(check["pass"] for check in checks),
        "scratch_layout": dict(offsets),
        "requested_workspace_bytes": {
            "indexed_native_scratch_and_tiling": scratch.nbytes + tiling.nbytes,
            "separate_explicit_native_buffers": separate_bytes,
        },
        "separate_workspace_scope": (
            "Explicit front/selector/tail buffers only; excludes stock CANN "
            "Prolog/SFA internal workspace and graph allocations."),
        "page_maps": {"main_pages": common.pages,
                      "main_table_pages": common.table_pages,
                      "indexer_pages": index_pages,
                      "indexer_table_pages": index_table_pages},
    }
    if args.shadow_library:
        report["shadow_cache"] = {
            "initial_bytes": initial_shadow,
            "shape": list(comparison_caches[4].shape),
            "storage_bytes": comparison_caches[4].nbytes,
            "negative_last_main_slot": args.negative_main_slot,
            "scope": (
                "Full shadow initialization follows each fixture cache restore "
                "outside capture/timing. Each native call updates current INT64 "
                "slots during the first front projection; no added phase barrier."),
        }
    if args.engine_adapter:
        report["adapter_scope"] = (
            "Production run_body, indexer packing, independent INT32 slot "
            "sources and capture pool; persistent top-k is copied by this "
            "probe and consumed by a second shared-indexer call with the "
            "same hidden/weights/cache. Excludes vendor forward hook and TP "
            "reduction; the two calls are not two dependent model layers.")
        report["adapter_state"] = {
            "workspace_entries": len(pool.entries),
            "slot_entries": len(pool.slot_entries),
            "capture_reserves_remaining": len(pool.capture_reserves),
            "slot_reserves_remaining": len(pool.slot_reserves),
            "distinct_main_index_slot_sources": (
                adapter_main[2].data_ptr() != adapter_index[2].data_ptr()),
            "persistent_topk_owns_storage": (
                persistent_topk.data_ptr() != adapter_state["prepared"][5].data_ptr()),
        }
    if report["pass"] and args.capture and not args.no_timing:
        pairs = []
        for pair in range(args.timing_pairs):
            result = {}
            order = (("indexed", full_graph), ("separate", separate_graph))
            for name, graph in order[::1 if pair % 2 == 0 else -1]:
                result[name] = time_graph(
                    torch, graph, stream, args.iterations) / args.calls
            pairs.append(result)
        report["graph_us"] = {"unit": "microseconds per call",
                              "pairs": pairs, "medians": {
            name: statistics.median(pair[name] for pair in pairs)
            for name in ("indexed", "separate")}}
        if args.comparison_library:
            comparison_pairs = []
            for pair in range(args.timing_pairs):
                result = {}
                order = (("ordinary", full_graph),
                         (args.comparison_label, comparison_graph))
                for name, graph in order[::1 if pair % 2 == 0 else -1]:
                    result[name] = time_graph(
                        torch, graph, stream, args.iterations) / args.calls
                comparison_pairs.append(result)
            report["native_comparison_us_per_call"] = comparison_pairs
            if args.profile_library:
                from attention_indexed_profile import read_trace, summarize

                records = []
                for _ in range(args.profile_samples):
                    with torch.npu.stream(stream):
                        comparison_graph.replay()
                    stream.synchronize()
                    records.append(read_trace(
                        torch, comparison_scratch.cpu(), comparison_offsets,
                        rows, blocks))
                report["phase_profile"] = {
                    "event_pairs_us_per_call": comparison_pairs,
                    "summary": summarize(records, blocks),
                    "timestamps": records,
                    "scope": (
                        "Each sample retains the final call's timestamps after "
                        "replaying the full weight-bank graph. Stage intervals "
                        "include existing barriers. Same-worker differences only; "
                        "no cross-core clock origin assumption. Trace stores "
                        "follow the last timestamp and appear in event overhead."),
                }
            final_metric = metric(torch, comparison_output, output)
            report["native_comparison_final_output"] = final_metric
            report["pass"] = (report["pass"] and final_metric["pass"]
                              and final_metric["bitwise_equal"])
            if args.shadow_library:
                report["shadow_cache"]["post_timing_bytes"] = shadow_bytes()
                report["pass"] = (report["pass"] and
                                  report["shadow_cache"]["post_timing_bytes"]["pass"])
        if args.compare_placements and report["pass"]:
            from attention_indexed_placement import compare_placements

            if (offsets != comparison_offsets or not torch.equal(
                    tiling.cpu(), comparison_tiling.cpu())):
                raise ValueError("placement comparison requires identical layouts and tiling")
            arenas = {
                "A": {"caches": [common.caches[0], common.ropes[0],
                                  index_caches[0], index_scales[0]],
                      "scratch": scratch, "tiling": tiling, "output": output},
                "B": {"caches": comparison_caches, "scratch": comparison_scratch,
                      "tiling": comparison_tiling, "output": comparison_output},
            }

            def observe(arena):
                cpu = arena["scratch"].cpu()
                values = {name: tensor_from_bytes(torch, cpu, offsets[name],
                                                 shape, dtype)
                          for name, (shape, dtype) in shapes.items()}
                values.update(decode_front(torch, cpu, offsets, rows,
                                           quantized, "front_"))
                values["output"] = arena["output"].cpu().clone()
                for name, value in zip(("kv_cache", "rope_cache", "indexer_cache",
                                        "indexer_scale_cache"), arena["caches"]):
                    if name != "indexer_scale_cache" or quantized:
                        values[name] = value.cpu().clone()
                return values

            result = compare_placements(
                torch, args, stream, native_sequence, arenas, observe,
                {"ordinary": torch.ops.tilert, "candidate": comparison_ops})
            report["placement_comparison"] = result
            report["pass"] = report["pass"] and result["pass"]
    return report


def main():
    args = arguments()
    import torch
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    libraries = (args.library, args.front_library, args.indexer_library,
                 args.tail_library)
    if args.engine_adapter:
        libraries += (args.shared_library,)
    if args.comparison_library:
        libraries += (args.comparison_library,)
    for library in libraries:
        torch.ops.load_library(str(library.resolve()))
    data = inputs_fixture(torch, args)
    banks = [packed_weights(torch_npu, args, data) for _ in range(args.banks)]
    bank_identities = []
    addresses = {}
    for main_weights, front_weights, tail_weights, bias in banks:
        tensors = {name: main_weights[name] for name in WEIGHTS}
        tensors.update({f"front_{name}": value for name, value
                        in zip(FRONT_WEIGHTS, front_weights)})
        tensors.update({f"tail_{index}": value for index, value
                        in enumerate(tail_weights)})
        if bias is not None:
            tensors["bias"] = bias
        identity = {}
        for name, value in tensors.items():
            address = value.data_ptr()
            previous = addresses.setdefault(name, set())
            if address in previous:
                raise RuntimeError(f"Weight bank {name} reuses a prior allocation")
            previous.add(address)
            identity[name] = {
                "address": address, "shape": list(value.shape),
                "dtype": str(value.dtype),
                "format": torch_npu.get_npu_format(value),
                "storage_bytes": value.untyped_storage().nbytes(),
            }
        bank_identities.append(identity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": ("One native Prolog/export/indexer-front/cache/selector/SFA/"
                  "projection launch versus stock Prolog/SFA and validated "
                  "separate native front/selector/projection; before TP reduction"),
        "provenance": provenance(torch, torch_npu, args),
        "comparison_libraries": {str(path): sha256(path) for path in libraries[1:]},
        "harness_sources": {str(path): sha256(path) for path in (
            Path(__file__), HERE / "attention_export_check.py",
            HERE / "attention_check.py", HERE / "attention_full_check.py",
            HERE / "indexer_front_check.py")},
        "production_boundary_gate": "zero tolerance and raw-byte equality",
        "original_scale_dtypes": {
            "query_projection": args.query_scale_dtype,
            "output_projection": args.output_scale_dtype,
        },
        "weight_banks": bank_identities,
        "diagnostic_only": ["norm_stats"], "cases": [], "pass": False,
    }
    if args.comparison_library:
        report["diagnostic_only"].append(args.comparison_label + "/norm_stats")
    if args.profile_library:
        path = HERE / "attention_indexed_profile.py"
        report["harness_sources"][str(path)] = sha256(path)
    if args.compare_placements:
        path = HERE / "attention_indexed_placement.py"
        report["harness_sources"][str(path)] = sha256(path)
    if args.engine_adapter:
        from tilert.models.glm_5_2_ascend import attention_native, mla_tail

        report["adapter_sources"] = {
            module.__file__: sha256(Path(module.__file__))
            for module in (attention_native, mla_tail)
        }
    try:
        with torch.inference_mode():
            for mode in args.modes:
                for rows in args.widths:
                    for blocks in args.blocks:
                        case = run_case(torch, torch_npu, args, banks, rows,
                                        mode, blocks)
                        report["cases"].append(case)
                        print(json.dumps({"mode": mode, "rows": rows,
                                          "blocks": case["blocks"],
                                          "pass": case["pass"]}), flush=True)
                        if not case["pass"]:
                            raise RuntimeError("indexed attention differential failed")
        report["pass"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
