"""Compare the real outer attention call before and after native installation."""

from contextlib import contextmanager
import ctypes
import hashlib
import inspect
import json
from pathlib import Path
import traceback


CACHE_NAMES = ("main_cache", "rope_cache", "index_cache", "index_scale")


def source_identity(value):
    path = Path(inspect.getfile(value)).resolve()
    return {"path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def snapshot(torch, value):
    return value.detach().cpu().clone(memory_format=torch.contiguous_format)


def page_plan(torch, rows, iteration, sequence, pages, index_pages):
    """Map one causal query suffix into independently allocated cache pages."""
    length = sequence - iteration % 11
    if not rows <= length or length < 2048:
        raise ValueError("The query suffix must fit a sequence of at least 2048")
    positions = torch.arange(length - rows, length, dtype=torch.int64)
    count = (length + 127) // 128
    if count > min(pages, index_pages):
        raise ValueError("The allocated caches cannot cover the sequence")
    main_table = torch.arange(pages, dtype=torch.int32).roll(iteration + 1)[:count]
    index_table = torch.arange(index_pages, dtype=torch.int32).flip(0)
    index_table = index_table.roll(iteration + 3)[:count]

    def slots(table):
        return (table[positions // 128].long() * 128 + positions % 128).int()

    return {
        "positions": positions, "lengths": torch.tensor([length], dtype=torch.int32),
        "query_lengths": torch.tensor([rows], dtype=torch.int32),
        "query_start": torch.tensor([0, rows], dtype=torch.int32),
        "main_table": main_table.reshape(1, -1),
        "index_table": index_table.reshape(1, -1),
        "main_slots": slots(main_table), "index_slots": slots(index_table),
    }


def make_metadata(torch, torch_npu, layer, plan, device):
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.attention.indexer import AscendSFAIndexerMetadata
    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata
    from vllm_ascend.ops import rotary_embedding

    def nd(value):
        return torch_npu.npu_format_cast(value.to(device).contiguous(), 2)

    rows = plan["positions"].numel()
    # This fixture constructs parameters on CPU before moving the layer. The
    # official module's published RoPE table can therefore still be on CPU.
    rope_positions = plan["positions"].to(rotary_embedding._cos_cache.device)
    cos, sin = rotary_embedding.get_cos_and_sin_mla(rope_positions, use_cache=False)
    cosine, sine = nd(cos), nd(sin)
    positions = nd(plan["positions"])
    lengths, query = nd(plan["lengths"]), nd(plan["query_lengths"])
    state = (AscendAttentionState.DecodeOnly if rows == 1
             else AscendAttentionState.SpecDecoding)
    main = AscendSFAMetadata(
        num_actual_tokens=rows, num_input_tokens=rows,
        slot_mapping=nd(plan["main_slots"]), seq_lens=lengths,
        seq_lens_cpu=plan["lengths"].clone(), cum_query_lens=query,
        block_table=nd(plan["main_table"]), sin=sine, cos=cosine,
        positions=positions, query_start_loc=nd(plan["query_start"]),
        max_query_len=rows, max_seq_len=int(plan["lengths"].item()),
        attn_state=state, num_decodes=1, num_decode_tokens=rows, block_size=128,
    )
    index = AscendSFAIndexerMetadata(
        num_actual_tokens=rows, slot_mapping=nd(plan["index_slots"]),
        seq_lens=lengths.clone(), cum_query_lens=query.clone(),
        block_table=nd(plan["index_table"]), sin=sine.clone(), cos=cosine.clone(),
        actual_seq_lengths_query=query.clone(), actual_seq_lengths_key=lengths.clone(),
        num_decode_tokens=rows, block_size=128,
    )
    attention = layer.mla_attn.mla_attn
    return positions, {attention.layer_name: main,
                       attention.impl.indexer.k_cache.prefix: index}


@contextmanager
def count_native_calls(adapter):
    original = adapter.run_body
    calls = []

    def counted(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append({"rows": args[0].shape[0], "mode": kwargs["mode"]})
        return result

    adapter.run_body = counted
    try:
        yield calls
    finally:
        adapter.run_body = original


def initial_cache_values(torch, caches):
    generator = torch.Generator().manual_seed(41022)
    images = []
    for index, cache in enumerate(caches):
        if cache.dtype == torch.int8:
            value = torch.randint(-127, 128, cache.shape, dtype=torch.int8,
                                  generator=generator)
        elif index == 3:
            value = (torch.rand(cache.shape, generator=generator) / 128 + 1 / 64)
            value = value.to(cache.dtype)
        else:
            value = torch.randn(cache.shape, generator=generator).to(cache.dtype)
        images.append(value)
    return images


def load_sfa_adapter(torch, path):
    torch.ops.load_library(str(path))
    library = ctypes.CDLL(str(path))
    resolve = library.vendor_hook_sfa_api_library
    resolve.argtypes = [ctypes.c_char_p]
    resolve.restype = ctypes.c_char_p
    functions = {}
    for name in ("aclnnSparseFlashAttentionGetWorkspaceSize", "aclnnSparseFlashAttention"):
        raw = resolve(name.encode())
        if raw is None:
            raise RuntimeError(f"The official SFA adapter cannot resolve {name}")
        source = Path(raw.decode()).resolve()
        functions[name] = {"path": str(source),
                           "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "schema": str(torch.ops._C_ascend.npu_sparse_flash_attention.default._schema),
            "resolved_api_libraries": functions}


def run_outer_forward(torch, layer, config, adapter, original_forward, args):
    """Run eager local-shard calls through the real vLLM custom-op boundary."""
    import torch_npu
    from vllm.forward_context import set_forward_context
    from vllm_ascend.ops.mla import mla_forward

    from attention_check import metric

    attention = layer.mla_attn.mla_attn
    impl = attention.impl
    if impl.o_proj.reduce_results or impl.skip_topk or impl.indexer is None:
        raise ValueError("The first outer-forward gate requires an unreduced full-indexer layer")
    installed_forward = impl.forward
    caches = (*attention.kv_cache, *impl.indexer.k_cache.kv_cache)
    device = caches[0].device
    images = initial_cache_values(torch, caches)
    bases = [torch_npu.npu_format_cast(value.to(device), 2) for value in images]
    topk = impl.topk_indices_buffer
    report = {
        "scope": "real outer layer eager forward, one TP16 local shard before TP reduction",
        "script": source_identity(run_outer_forward),
        "outer_forward": source_identity(type(layer)),
        "mla_custom_op": source_identity(mla_forward),
        "forward_context": source_identity(set_forward_context),
        "adapter": source_identity(adapter.run_body),
        "mode": args.mode, "widths": args.forward_widths,
        "iterations": args.forward_replays, "sequence": args.forward_sequence,
        "cache_shapes": [list(cache.shape) for cache in caches],
        "cache_dtypes": [str(cache.dtype) for cache in caches],
        "cache_seed": 41022, "hidden_seed_base": 41023,
        "collective": False, "graph_capture": False, "cases": [], "pass": False,
        "sfa_adapter": load_sfa_adapter(torch, args.sfa_library),
    }
    output_path = args.output / "outer-forward.json"

    def persist():
        output_path.write_text(json.dumps(report, indent=2) + "\n")

    def reset():
        for cache, base in zip(caches, bases):
            cache.copy_(base)
        topk.fill_(-1)

    def call(forward, hidden, plan):
        impl.forward = forward
        reset()
        positions, metadata = make_metadata(torch, torch_npu, layer, plan, device)
        slots = {name: item.slot_mapping for name, item in metadata.items()}
        with set_forward_context(metadata, config, num_tokens=hidden.shape[0],
                                 slot_mapping=slots):
            result = layer(positions, hidden.clone(), None)
        torch.npu.synchronize()
        state = {"output": snapshot(torch, result), "topk": snapshot(torch, topk)}
        state.update({name: snapshot(torch, value)
                      for name, value in zip(CACHE_NAMES, caches)})
        return state

    persist()
    try:
        with torch.inference_mode(), count_native_calls(adapter) as hits:
            for rows in args.forward_widths:
                for iteration in range(args.forward_replays):
                    plan = page_plan(torch, rows, iteration, args.forward_sequence,
                                     caches[0].shape[0], caches[2].shape[0])
                    generator = torch.Generator().manual_seed(41023 + rows * 100 + iteration)
                    hidden_cpu = torch.randn(rows, 6144, generator=generator).bfloat16()
                    hidden = torch_npu.npu_format_cast(hidden_cpu.to(device), 2)
                    case = {"rows": rows, "iteration": iteration, "pass": False,
                            "phase": "original_outer_forward",
                            "main_slots": plan["main_slots"].tolist(),
                            "index_slots": plan["index_slots"].tolist()}
                    report["cases"].append(case)
                    persist()
                    before = len(hits)
                    baseline = call(original_forward, hidden, plan)
                    if len(hits) != before:
                        raise RuntimeError("The original outer call unexpectedly entered the native body")
                    case["phase"] = "installed_outer_forward"
                    persist()
                    native = call(installed_forward, hidden, plan)
                    if len(hits) != before + 1:
                        raise RuntimeError("The installed outer call fell back instead of entering the native body")
                    checks = {name: metric(torch, native[name], expected)
                              for name, expected in baseline.items()}
                    passed = all(check["pass"] and check["bitwise_equal"]
                                 for check in checks.values())
                    case.update({"pass": passed, "phase": "comparison",
                                 "native_calls": hits[before:], "checks": checks})
                    persist()
                    print(json.dumps({"outer_forward_case": case}), flush=True)
                    if not passed:
                        dump = args.output / f"outer-forward-{args.mode}-m{rows}-r{iteration}.pt"
                        torch.save({"hidden": hidden_cpu, "metadata": plan,
                                    "initial_caches": images,
                                    "baseline": baseline, "native": native}, dump)
                        case["failure_dump"] = str(dump)
                        persist()
                        raise RuntimeError("The real outer attention call differs; see outer-forward.json")
        report["pass"] = True
        persist()
        return report
    except Exception:
        report["error"] = traceback.format_exc()
        persist()
        raise
    finally:
        impl.forward = installed_forward

