"""Run frozen TP16 inputs through one resident CANN layer at a time.

Each decoder invocation uses M1 or the complete M32 teacher chunk. Hidden and
residual travel separately; shared index selections are restored per query.
An optional trace replays two overlapping M8 target calls with frozen commits.
This records numerical and residency evidence, without measuring latency.
"""

import argparse
from contextlib import contextmanager, nullcontext
import gc
import hashlib
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch
import weakref

from prefix_io import prompt_identity
from run_cann_model import check_prefix, digest, load_capture
from run_cann_streamed import (
    QUOTA_BYTES, bound_rope_tables, checked, gather, load_scope,
    make_skeleton, selections,
)
from run_native_decode import compare_tensor


CAPACITY = 9344


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "capture-case", "teacher-json", "prompt-json",
                 "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--layers", type=int, choices=range(1, 79), default=78,
                        help="Execute the first N main-model layers")
    parser.add_argument("--steps", type=int, choices=range(1, 33), default=32)
    parser.add_argument("--query-width", type=int, choices=(1, 8, 32),
                        default=1)
    parser.add_argument("--round-trace", type=Path,
                        help="Replay the first two measured M8 target calls")
    parser.add_argument("--baseline", type=Path,
                        help="Full CANN layer traces for every requested row")
    parser.add_argument("--attention-kernel",
                        choices=("sfa", "gather_fia", "shared_fia"),
                        default="sfa")
    parser.add_argument("--fp32-head", action="store_true")
    args = parser.parse_args()
    if args.round_trace:
        if (args.query_width != 8 or args.steps != 32
                or args.attention_kernel != "sfa" or not args.fp32_head
                or args.baseline is not None):
            parser.error("Round replay requires M8, SFA, FP32 head, the full "
                         "teacher reference and no token-major baseline")
    elif args.query_width == 8:
        parser.error("M8 requires --round-trace")
    if args.query_width == 32 and args.steps != 32:
        parser.error("M32 requires all 32 teacher inputs")
    if args.query_width == 32 and args.attention_kernel == "gather_fia":
        parser.error("gather_fia is not validated at M32")
    return args


def read_teacher(args, inputs):
    path = args.teacher_json
    report = json.loads(path.read_text())
    prompt = json.loads(args.prompt_json.read_text())
    identity = prompt_identity(prompt)
    if (digest(args.prompt_json) != report["prompt_sha256"]
            or identity["prompt_sha256"] != inputs["prompt_sha256"]
            or identity["prompt_length"] != report["prompt_tokens"]
            or report["prompt_tokens"] != inputs["initial_position"]
            or report["first_token_id"] != inputs["first_token_id"]):
        raise ValueError("Teacher prompt/first token differs from capture")
    tokens = report["teacher_tokens"]
    expected = report["expected_next_tokens"]
    if (len(tokens) != 32 or len(expected) != 32
            or expected[:-1] != tokens[1:]
            or tokens[0] != inputs["first_token_id"]
            or any(type(token) is not int or not 0 <= token < 154880
                   for token in tokens + expected)):
        raise ValueError("Teacher file requires 32 linked successor labels")
    return tokens[:args.steps], expected[:args.steps], {
        "path": str(path.resolve()), "sha256": digest(path),
        "baseline_request_id": report["baseline_request_id"],
        "prompt_path": str(args.prompt_json.resolve()),
        "prompt_file_sha256": report["prompt_sha256"],
        "prompt_identity": identity, "teacher_tokens": tokens,
        "expected_next_tokens": expected, "scope": report["scope"],
    }


def read_round_trace(path, inputs, checkpoint, teacher, expected, rank):
    raw = path.read_bytes()
    report = json.loads(raw)
    identity = ("prompt_sha256", "prompt_length", "last_prompt_token",
                "initial_position", "first_token_id", "request_id",
                "generation", "capture_manifest_sha256")
    if (report["rank"] != 0 or report["world_size"] != 16
            or report["width"] != 8 or report["capacity"] != CAPACITY
            or report["fp32_head"] is not True
            or report["checkpoint_sha256"] != checkpoint
            or any(report["inputs"][k] != inputs[k] for k in identity)
            or (rank == 0 and report["inputs"]["prefix_sha256"]
                != inputs["prefix_sha256"])):
        raise ValueError("Round trace target/capture identity differs")
    index = report["warmup"]
    request = report["rank_requests"][index]
    if (report["requests"][index]["warmup"] is not False
            or request["injected"][0] != 8192
            or request["first_token"] != inputs["first_token_id"]
            or len(request["rounds"]) < 2):
        raise ValueError("Round trace requires two measured target calls")
    tokens, schedule, scored_rows, labels = [], [], [], []
    committed, position = [request["first_token"]], 8192
    for call in request["rounds"][:2]:
        candidates, predictions = call["candidates"], call["predictions"]
        if (call["width"] != 8 or len(candidates) != len(predictions)
                or len(candidates) != 8 or any(
                    type(x) is not int or not 0 <= x < 154880
                    for x in candidates + predictions)):
            raise ValueError("Round trace requires eight vocabulary IDs")
        matched = 0
        for proposed, predicted in zip(candidates[1:], predictions):
            if proposed != predicted:
                break
            matched += 1
        accepted = predictions[:min(matched + 1,
                                    report["max_tokens"] - len(committed))]
        if (not accepted or candidates[0] != committed[-1]
                or call["before"] != position
                or call["written"] != position + 8
                or call["retained"] != position + len(accepted)
                or call["accepted_tokens"] != accepted
                or call["matched_drafts"] != matched
                or call["natural_accepted"] != matched + 1):
            raise ValueError("Round trace commit/position contract differs")
        offset = position - 8192
        if committed == teacher[:len(committed)]:
            for row, token in enumerate(candidates):
                if (offset + row >= len(teacher)
                        or token != teacher[offset + row]):
                    break
                scored_rows.append(len(tokens) + row)
                labels.append(expected[offset + row])
        schedule.append({"position": position, "retained": call["retained"],
                         "tokens": list(candidates),
                         "historical_predictions": list(predictions),
                         "historical_accepted_tokens": list(accepted)})
        tokens.extend(candidates)
        committed.extend(accepted)
        position = call["retained"]
    if not scored_rows:
        raise ValueError("Round trace has no trustworthy teacher successors")
    return tokens, schedule, scored_rows, labels, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(), "request_index": index,
        "source_commit": report["source_commit"], "schedule": schedule,
        "historical_execution": {name: report[name] for name in (
            "target_graph", "compile_target", "super_kernel",
            "draft_graph", "conditioned_graph")},
        "scope": "Frozen target calls; historical predictions are diagnostic",
    }


@contextmanager
def replay_attention(layer, metadata, number):
    impl = layer.self_attn.mla_attn.mla_attn.impl
    original, calls = impl._execute_sparse_flash_attention_process, []

    def values(query, key, table):
        return {"query": query.cpu().tolist(), "key": key.cpu().tolist(),
                "block_table": table.cpu().clone()}

    def capture(ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
                actual_seq_lengths_query, actual_seq_lengths_key,
                block_table=None):
        table = (attn_metadata.block_table if block_table is None
                 else block_table)
        record = {"sfa": values(actual_seq_lengths_query,
                                actual_seq_lengths_key, table)}
        indexer = metadata.get(
            f"model.layers.{number}.self_attn.indexer.k_cache",
        )
        if indexer is not None:
            record["indexer"] = values(indexer.actual_seq_lengths_query,
                                       indexer.actual_seq_lengths_key,
                                       indexer.block_table)
        calls.append(record)
        return original(ql_nope, q_pe, kv_cache, topk_indices, attn_metadata,
                        actual_seq_lengths_query, actual_seq_lengths_key,
                        block_table=block_table)

    method = "_execute_sparse_flash_attention_process"
    with patch.object(impl, method, capture):
        yield calls


def snapshot_round(engine, plan, width, protected, attention):
    import torch

    start, retained = plan["position"], plan["retained"]
    written, cache = start + width, engine.cache[0]
    snapshot = {name: value[start:written].cpu().clone()
                for name, value in cache.items()}
    def same(left, right):
        return torch.equal(left.contiguous().view(torch.uint8),
                           right.contiguous().view(torch.uint8))

    if len(attention) != 1:
        raise ValueError("Expected one actual SFA call per replay chunk")
    checks = []
    pages = (written + 127) // 128
    expected_table = engine.block_table.cpu()[:, :pages]
    for name in ("sfa", "indexer") if "index_keys" in cache else ("sfa",):
        actual = attention[0].get(name)
        checks.append({"name": name + ".lengths", "passed":
                       actual is not None and actual["query"] == [width]
                       and actual["key"] == [written]})
        table = actual["block_table"] if actual is not None else None
        checks.append({"name": name + ".block_table", "passed":
                       table is not None and table.ndim == 2
                       and table.shape[0] == 1 and table.shape[1] >= pages
                       and torch.equal(table[:, :pages], expected_table)})
    for name, expected in (protected or {}).items():
        actual = cache[name][8192:start].cpu()
        checks.append({"name": "accepted_prefix." + name,
                       "passed": same(actual, expected)})
    metadata = {"input_ids": engine.input_ids.cpu().tolist(),
                "positions": engine.positions.cpu().tolist(),
                "slots": engine.slots.cpu().tolist(),
                "seq_lens": engine.seq_lens.cpu().tolist()}
    checks.append({"name": "call_metadata", "passed":
                   metadata["input_ids"] == plan["tokens"]
                   and metadata["positions"] == metadata["slots"]
                   == list(range(start, written))
                   and metadata["seq_lens"] == [written]})
    engine.position = written
    engine.truncate(retained)
    for name, value in snapshot.items():
        checks.append({"name": "new_cache_finite." + name,
                       "passed": bool(torch.isfinite(value).all())})
        checks.append({"name": "rejected_tail_preserved." + name,
                       "passed": same(cache[name][retained:written].cpu(),
                                      value[retained - start:])})
    protected = {name: value[8192:retained].cpu().clone()
                 for name, value in cache.items()}
    return {"position": start, "written": written, "retained": retained,
            **metadata, "attention": attention[0],
            "cache_rows": snapshot}, protected, checks


def read_baseline(args, rank, checkpoint, inputs, teacher):
    import torch

    if args.baseline is None:
        return [], {}
    path = args.baseline / f"rank-{rank}.json"
    report = json.loads(path.read_text())
    if (report["rank"] != rank
            or report["checkpoint_sha256"] != checkpoint
            or report["inputs"] != inputs
            or "float32" not in (report.get("tp_reduction"),
                                 report.get("model_tp_reduction"))):
        raise ValueError("Baseline prefix, checkpoint or TP policy differs")
    traces, hashes = [], {str(path): digest(path)}
    for offset, token in enumerate(teacher):
        position = inputs["initial_position"] + offset
        name = f"step-{position:05d}.pt"
        path = args.baseline / f"rank-{rank}" / name
        hashes[str(path)] = digest(path)
        if hashes[str(path)] != report["trace_sha256"][name]:
            raise ValueError("Baseline trace hash differs")
        trace = torch.load(path, map_location="cpu", weights_only=True,
                           mmap=True)
        if trace["position"] != position or trace["token"] != token:
            raise ValueError("Baseline teacher position differs")
        if len(trace["hidden_states"]) != 79:
            raise ValueError("Baseline needs all main hidden boundaries")
        traces.append(trace)
    return traces, hashes


def bind_cache(engine, number, prefix):
    import torch
    from vllm_ascend.attention.indexer import AscendSFAIndexerMetadataBuilder
    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
    from vllm_ascend.core.kv_cache_interface import (
        AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec,
    )
    from vllm_ascend.patch.worker.patch_bind_kv_cache import (
        bind_kv_cache_to_layers,
    )
    from tilert.models.glm_5_2_ascend.cann_attention import install_gather_fia
    from tilert.pd_vllm.ascend_layout import (
        FULL_INDEX_LAYERS, INT8_INDEX_LAYERS,
    )

    caches, rows, specifications = {}, {}, {}
    engine.builders = []
    context = engine.config.compilation_config.static_forward_context
    for name, spec in engine.runner.get_kv_cache_spec().items():
        if (not name.startswith(f"model.layers.{number}.")
                or spec.block_size != 128 or spec.num_kv_heads != 1):
            raise ValueError(f"Unexpected resident cache: {name}: {spec}")
        if isinstance(spec, AscendSFAIndexerCacheSpec):
            quantized = number in INT8_INDEX_LAYERS
            expected_dtype = torch.int8 if quantized else torch.bfloat16
            if (number not in FULL_INDEX_LAYERS or spec.head_size != 128
                    or spec.dtype != expected_dtype
                    or spec.scale_dim != int(quantized)
                    or quantized and spec.scale_dtype != torch.float16):
                raise ValueError(f"Layer {number}: index cache recipe differs")
            fields = [("index_keys", 128, spec.dtype)]
            if quantized:
                fields.append(("index_scales", 1, spec.scale_dtype))
            builder = AscendSFAIndexerMetadataBuilder
        elif isinstance(spec, AscendMLAAttentionSpec):
            if (spec.head_size != 576 or spec.dtype != torch.bfloat16
                    or spec.cache_sparse_sfa_c8):
                raise ValueError("Expected separate BF16 latent/RoPE pages")
            fields = [("latent", 512, spec.dtype), ("rope", 64, spec.dtype)]
            builder = AscendSFAMetadataBuilder
            if engine.attention_kernel == "gather_fia":
                install_gather_fia(context[name].impl)
            elif engine.attention_kernel == "shared_fia":
                engine._shared_attention.bind(context[name].impl)
        else:
            raise ValueError(f"Unsupported cache: {spec}")
        values = []
        for field, width, dtype in fields:
            page = torch.zeros((CAPACITY // 128, 128, 1, width), dtype=dtype,
                               device=engine.device)
            values.append(page)
            rows[field] = (page.view(-1) if field == "index_scales"
                           else page.view(-1, width))
        caches[name] = tuple(values)
        specifications[name] = repr(spec)
        engine.builders.append((
            [name], builder(spec, [name], engine.config, engine.device),
        ))
    expected = {"latent", "rope"}
    if number in FULL_INDEX_LAYERS:
        expected.add("index_keys")
    if number in INT8_INDEX_LAYERS:
        expected.add("index_scales")
    prefix = {name: value for name, value in prefix.items()
              if value is not None}
    if rows.keys() != expected or prefix.keys() != expected:
        raise ValueError(f"Layer {number}: missing or extra cache fields")
    for name, value in prefix.items():
        if (value.shape != rows[name][:8192].shape
                or value.dtype != rows[name].dtype):
            raise ValueError(f"Layer {number}: invalid prefix {name}")
        rows[name][:8192].copy_(value)
    engine.cache = [rows]
    bind_kv_cache_to_layers(caches, context)
    return specifications


def run_layer(engine, factory, checkpoint, number, prefix, state, teacher,
              width=1, schedule=None):
    import torch
    from vllm.model_executor.model_loader.utils import (
        process_weights_after_loading,
    )
    from vllm.model_executor.models.utils import PPMissingLayer
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
    from vllm_ascend.quantization.methods.kv_cache.kv_c8 import (
        AscendSFAQuantAttentionMethod,
    )
    from tilert.pd_vllm.ascend_layout import INT8_INDEX_LAYERS

    hidden, residual, shared = state
    compilation = engine.config.compilation_config
    old_keys = set(compilation.static_forward_context)
    old_moe = list(compilation.static_all_moe_layers)
    old_eplb = list(VllmEplbAdaptor._registered_moe_layers)
    checks, inputs, reductions = [], [], []
    call_snapshots, protected = [], None
    positions = (list(range(8192, 8192 + len(teacher))) if schedule is None
                 else [plan["position"] + row for plan in schedule
                       for row in range(width)])
    try:
        def prepare():
            with torch.device(engine.device):
                layer = factory(prefix=f"model.layers.{number}").eval()
            engine.model.model.layers[number] = layer
            rope = bound_rope_tables(layer)
            if number in INT8_INDEX_LAYERS:
                # C8 uses indexer's Hadamard buffers, not these placeholders.
                unused = layer.self_attn.mla_attn.mla_attn.indexer
                del unused.q_rot, unused.k_rot
            torch.npu.empty_cache()
            coverage = load_scope(engine, checkpoint,
                                  (f"model.layers.{number}.",))
            process_weights_after_loading(layer, engine.config.model_config,
                                          engine.device)
            coverage.update(rope_tables=rope,
                            cache_specs=bind_cache(engine, number, prefix))
            if number in INT8_INDEX_LAYERS:
                coverage["removed_unused_parameters"] = ["q_rot", "k_rot"]
            return layer, coverage

        layer, coverage = checked(f"layer{number}.prepare", prepare)
        references = {"layer": weakref.ref(layer), **{
            name: weakref.ref(value)
            for name, value in layer.named_parameters()
        }}
        skip_topk = layer.self_attn.mla_attn.skip_topk
        if skip_topk and shared is None:
            raise ValueError("Shared index layer has no preceding selections")
        check_prefix(engine, [prefix], checks, "before")
        raw_hidden = hidden.cpu().clone()
        raw_residual = residual.cpu().clone() if residual is not None else None
        next_hidden, next_residual, merged = [], [], []
        original = engine._tp_sum_original

        def reduce(value):
            reductions.append((str(value.dtype), list(value.shape)))
            return original(value)

        def capture_input(module, args, output):
            value = args[0] if len(args) == 1 else output[1]
            inputs.append(value.detach().clone())

        with (layer.input_layernorm.register_forward_hook(capture_input),
              selections(layer) as selected,
              patch.object(engine, "_tp_sum_original", reduce)):
            for offset in range(0, len(teacher), width):
                rows = slice(offset, offset + width)
                plan = (None if schedule is None
                        else schedule[offset // width])
                if plan is not None:
                    engine.replay_progress.update(
                        layer=number, round=offset // width,
                        position=plan["position"],
                        written=plan["position"] + width,
                        retained=plan["retained"], phase="forward",
                    )
                engine.position = (8192 + offset if plan is None
                                   else plan["position"])
                metadata = engine._metadata(width)
                engine.input_ids.copy_(torch.tensor(
                    teacher[rows], dtype=engine.input_ids.dtype,
                    device=engine.device,
                ))
                if skip_topk:
                    layer.self_attn.topk_indices_buffer[:width].copy_(
                        shared[rows],
                    )
                capture = (nullcontext([]) if plan is None
                           else replay_attention(layer, metadata, number))
                with set_ascend_forward_context(
                    metadata, engine.config, num_tokens=width,
                    num_actual_tokens=width, model_instance=engine.model,
                ), capture as attention:
                    h, r = layer(
                        engine.positions, hidden[rows],
                        None if residual is None else residual[rows],
                    )
                next_hidden.append(h.detach().clone())
                next_residual.append(r.detach().clone())
                merged.append(torch.ops._C_ascend.npu_add_rms_norm_bias(
                    h, r, engine.norm_units, None, 1e-5,
                )[2].detach().clone())
                if plan is not None:
                    engine.replay_progress["phase"] = "cache_snapshot"
                    snapshot, protected, local = snapshot_round(
                        engine, plan, width, protected, attention,
                    )
                    call_snapshots.append(snapshot)
                    checks.extend({**item, "name":
                                   f"round{offset // width}." + item["name"]}
                                  for item in local)
                    if offset == 0:
                        check_prefix(engine, [prefix], checks, "round0.after")
        if schedule is None:
            engine.position = 8192 + len(teacher)
        torch.npu.synchronize(engine.device)
        calls = len(teacher) // width
        if len(selected) != calls or len(inputs) != calls:
            raise ValueError("Expected one norm and attention per chunk")
        indices = torch.cat([value.reshape(width, -1) for value in selected])
        choices = indices.cpu()
        checks.append({"name": "fp32_tp_reductions", "passed": reductions
                       == [("torch.float32", [width, 6144])] * (2 * calls)})
        for offset, row in enumerate(choices):
            checks.append({"name": f"token{offset}.causal_indices", "passed":
                           row.numel() == 2048 and row.unique().numel() == 2048
                           and bool(torch.all((row >= 0)
                                              & (row <= positions[offset])))})
        if skip_topk:
            checks.append({"name": "shared_indices_restored_per_token",
                           "passed": torch.equal(choices, shared.cpu())})
        check_prefix(engine, [prefix], checks, "after")
        sources = engine.source_metadata()
        for value in (run_layer, make_skeleton, load_scope, type(checkpoint),
                      type(layer), type(layer.self_attn.rotary_emb),
                      type(layer.input_layernorm), VllmEplbAdaptor,
                      AscendSFAQuantAttentionMethod,
                      process_weights_after_loading):
            path = Path(inspect.getfile(inspect.unwrap(value))).resolve()
            sources[str(path)] = digest(path)
        output_h, output_r = torch.cat(next_hidden), torch.cat(next_residual)
        checks.append({"name": ("teacher_row_shapes" if schedule is None
                                else "replay_row_shapes"), "passed":
                       output_h.shape == output_r.shape
                       == (len(teacher), 6144)
                       and indices.shape == (len(teacher), 2048)})
        token_field = ("teacher_tokens" if schedule is None
                       else "candidate_tokens")
        data = {
            "layer": number,
            token_field: teacher,
            "positions": positions,
            "input_hidden": raw_hidden, "input_residual": raw_residual,
            "input_merged": torch.cat(inputs).cpu(),
            "hidden": output_h.cpu(), "residual": output_r.cpu(),
            "merged": torch.cat(merged).cpu(), "indices": choices,
            "skip_topk": skip_topk,
        }
        if schedule is None:
            data["cache_rows"] = {
                name: value[8192:engine.position].cpu().clone()
                for name, value in engine.cache[0].items()
            }
        else:
            data["round_calls"] = call_snapshots
        return ((output_h, output_r, shared if skip_topk else indices), data,
                checks, coverage, sources, references)
    finally:
        torch.npu.synchronize(engine.device)
        engine.model.model.layers[number] = PPMissingLayer()
        engine.builders, engine.cache = [], []
        for name in set(compilation.static_forward_context) - old_keys:
            del compilation.static_forward_context[name]
        compilation.static_all_moe_layers[:] = old_moe
        VllmEplbAdaptor._registered_moe_layers[:] = old_eplb


def compare_baseline(data, traces, exact):
    checks = []
    number = data["layer"]
    names = {"latent": "latent", "rope": "rope", "index_keys": "index",
             "index_scales": "index_scale"}
    for offset, expected in enumerate(traces):
        pairs = [("input_merged", data["input_merged"][offset:offset + 1],
                  expected["hidden_states"][number])]
        if number < 77:
            pairs.append(("merged", data["merged"][offset:offset + 1],
                          expected["hidden_states"][number + 1]))
        for name, actual in data["cache_rows"].items():
            row = (actual[offset:offset + 1] if name == "index_scales"
                   else actual[offset])
            pairs.append((name, row,
                          expected["cache_rows"][number][names[name]]))
        pairs.append(("indices", data["indices"][offset],
                      expected["indices"][number].reshape(-1)))
        for name, actual, reference in pairs:
            label = f"token{offset}.{name}"
            if not exact and name == "indices":
                left, right = set(actual.tolist()), set(reference.tolist())
                checks.append({"name": label, "actual_count": len(left),
                               "reference_count": len(right),
                               "overlap": len(left & right)})
            else:
                compare_tensor(label, actual, reference, checks, exact=True)
                if not exact:
                    checks[-1].pop("passed")
                    checks[-1].pop("exact")
    return checks


def final_logits(engine, hidden, residual, width):
    import torch

    logits, normalized = [], []
    for offset in range(0, len(hidden), width):
        rows = slice(offset, offset + width)
        value, _ = engine.model.model.norm(hidden[rows], residual[rows])
        normalized.append(value.cpu())
        logits.append(engine.model.compute_logits(value).cpu())
    return torch.cat(normalized), torch.cat(logits)


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def execute(args):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    import torch
    import torch_npu  # noqa: F401
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import (
        process_weights_after_loading,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm_ascend.worker.worker import NPUWorker
    from tilert.models.glm_5_2_ascend.cann_model import (
        CannModel, create_config,
    )
    from tilert.models.glm_5_2_ascend.weights import ModelSlimCheckpoint
    from tilert.pd_vllm.ascend_cache import checkpoint_fingerprint

    torch.set_num_threads(1)
    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != 16 or not 0 <= rank < 16:
        raise ValueError("Launch sixteen TP ranks")
    directory = args.output / f"rank-{rank}"
    directory.mkdir(parents=True, exist_ok=False)
    fingerprint = checkpoint_fingerprint(args.checkpoint)
    prefixes, _, inputs = load_capture(args.capture_case, rank, fingerprint)
    if inputs["initial_position"] != 8192 or len(prefixes) != 78:
        raise ValueError("Expected the captured 8192-row, 78-layer prefix")
    teacher, expected, teacher_source = read_teacher(args, inputs)
    schedule, round_trace = None, None
    scored_rows = list(range(len(teacher)))
    if args.round_trace:
        replay = read_round_trace(
            args.round_trace, inputs, fingerprint, teacher, expected, rank,
        )
        teacher, schedule, scored_rows, expected, round_trace = replay
    token_field = "candidate_tokens" if round_trace else "teacher_tokens"
    baseline, baseline_hashes = read_baseline(
        args, rank, fingerprint, inputs, teacher,
    )
    engine = CannModel.__new__(CannModel)
    engine.rank, engine.device = rank, torch.device(f"npu:{rank}")
    engine.capacity, engine.position = CAPACITY, 8192
    engine.checkpoint = str(args.checkpoint.resolve())
    engine.attention_kernel = args.attention_kernel
    engine.fp32_head = args.fp32_head
    engine.graph_width, engine.hoist_metadata = 32, False
    engine.replay_progress = args.replay_call = {} if round_trace else None
    engine.onesided_endpoint, engine._onesided = None, None
    engine._tp_sum_group, engine.graph = None, None
    engine._routed_workspace_pool = None
    engine._routed_shared_workspace_pool = None
    engine._mla_tail_workspace_pool = None
    engine._native_attention_workspace_pool = None
    engine._restore_combine, engine._restore_routed = None, None
    engine._restore_routed_shared = []
    engine._restore_mla_tail = []
    engine._restore_native_attention = []
    torch.npu.set_device(engine.device)
    total = torch.npu.get_device_properties(engine.device).total_memory
    torch.npu.set_per_process_memory_fraction(
        QUOTA_BYTES / total, engine.device,
    )
    engine.config = create_config(args.checkpoint, CAPACITY, 0.02,
                                  graph_width=32)
    checks, layers, sources, artifacts = [], [], {}, {}
    with (set_current_vllm_config(engine.config),
          set_default_torch_dtype(torch.bfloat16), torch.inference_mode()):
        engine.worker = NPUWorker(
            engine.config, local_rank=rank, rank=rank,
            distributed_init_method="env://", is_driver_worker=rank == 0,
        )
        engine.worker.init_device()
        try:
            if engine.worker.device != engine.device:
                raise ValueError("Worker remapped the physical device")
            identities = gather((teacher, expected, schedule, scored_rows,
                                 round_trace, fingerprint,
                                 inputs["prompt_sha256"],
                                 inputs["request_id"], inputs["generation"]))
            if any(item != identities[0] for item in identities):
                raise ValueError("Rank teacher/capture identities differ")
            engine.runner = engine.worker.model_runner
            engine._install_tp_sum()
            factory = checked("skeleton", lambda: make_skeleton(engine))
            checkpoint = ModelSlimCheckpoint(args.checkpoint)
            coverage = checked("base_weights", lambda: load_scope(
                engine, checkpoint,
                ("model.embed_tokens.", "model.norm.", "lm_head."),
            ))
            checked("base_postprocess", lambda: process_weights_after_loading(
                engine.model, engine.config.model_config, engine.device,
            ))
            if engine.fp32_head:
                from tilert.models.glm_5_2_ascend.cann_head import (
                    install_fp32_head,
                )

                checked("fp32_head", lambda: install_fp32_head(
                    engine.model.lm_head, engine.model.logits_processor,
                ))
            engine.layout = {"capacity_tokens": CAPACITY}
            engine._allocate_inputs()
            if engine.attention_kernel == "shared_fia":
                from tilert.models.glm_5_2_ascend.cann_attention import (
                    SharedKVAttention,
                )

                engine._shared_attention = SharedKVAttention(
                    32, CAPACITY, engine.device,
                )
            engine.norm_units = torch.ones(6144, dtype=torch.bfloat16,
                                           device=engine.device)
            embeddings = []
            for offset in range(0, len(teacher), args.query_width):
                tokens = torch.tensor(
                    teacher[offset:offset + args.query_width],
                    dtype=torch.int32, device=engine.device,
                )
                embeddings.append(engine.model.embed_input_ids(
                    tokens,
                ).detach().clone())
            state = (torch.cat(embeddings), None, None)
            del embeddings
            for number in range(args.layers):
                args.layer = number
                torch.npu.reset_peak_memory_stats(engine.device)
                result = run_layer(engine, factory, checkpoint, number,
                                   prefixes[number], state, teacher,
                                   args.query_width, schedule)
                state, data, local, loaded, source, refs = result
                del result
                gc.collect()
                alive = [name for name, ref in refs.items()
                         if ref() is not None]
                local.append({"name": "owners_released", "passed": not alive,
                              "alive": alive})
                for field in ("input_merged", "hidden", "residual", "merged"):
                    local.append({"name": field + ".finite", "passed": bool(
                        torch.isfinite(data[field]).all(),
                    )})
                exact = (args.query_width == 1
                         and args.attention_kernel == "sfa")
                diagnostics = compare_baseline(data, baseline, exact)
                if exact:
                    local.extend(diagnostics)
                checks.extend(
                    {**item, "name": f"layer{number}." + item["name"]}
                    for item in local
                )
                data.update(rank=rank, checkpoint_sha256=fingerprint,
                            inputs=inputs, teacher_source=teacher_source)
                path = directory / f"layer-{number:02d}.pt"
                def save():
                    with path.open("xb") as stream:
                        torch.save(data, stream)
                checked(f"layer{number}.save", save)
                artifacts[str(path)] = digest(path)
                sources.update(source)
                torch.npu.empty_cache()
                layers.append({
                    "layer": number, "coverage": loaded, "checks": local,
                    "baseline_diagnostics": diagnostics,
                    "peak_allocated": torch.npu.max_memory_allocated(
                        engine.device),
                    "allocated_after_release": torch.npu.memory_allocated(
                        engine.device),
                })
                checked(f"layer{number}.release",
                        lambda: require_released(alive))
            predictions, scored_predictions = [], []
            if args.layers == 78:
                hidden, residual, _ = state
                normalized, logits = final_logits(
                    engine, hidden, residual, args.query_width,
                )
                predictions = logits.argmax(dim=-1).tolist()
                scored_predictions = [predictions[row] for row in scored_rows]
                path = directory / "final.pt"
                final = {"normalized": normalized, "logits": logits,
                         token_field: teacher, "predicted_tokens": predictions}
                if round_trace:
                    top2 = logits.topk(2, dim=-1)
                    final.update(
                        scored_rows=scored_rows, expected_tokens=expected,
                        top2_tokens=top2.indices, top2_scores=top2.values,
                        expected_scores=logits[scored_rows, expected],
                    )
                with path.open("xb") as stream:
                    torch.save(final, stream)
                artifacts[str(path)] = digest(path)
                label_check = ("trusted_successor_labels" if round_trace
                               else "greedy_matches_frozen_trajectory")
                checks.append({"name": label_check,
                               "passed": scored_predictions == expected})
                dtype = torch.float32 if args.fp32_head else torch.bfloat16
                checks.append({"name": "logits_shape_and_dtype", "passed":
                               logits.shape == (len(teacher), 154880)
                               and logits.dtype == dtype})
                checks.append({"name": "final_normalized_shape_and_dtype",
                               "passed": normalized.shape
                               == (len(teacher), 6144)
                               and normalized.dtype == torch.bfloat16})
                checks.append({"name": "final_outputs_finite", "passed":
                               bool(torch.isfinite(normalized).all()
                                    and torch.isfinite(logits).all())})
            all_predictions = gather(predictions)
            checks.append({"name": "rank_predictions_agree", "passed": all(
                item == predictions for item in all_predictions
            )})
            passed = all(item["passed"] for item in checks)
            scope = ("Layer-major frozen target-round replay; eager, no MTP"
                     if round_trace else "Layer-major frozen teacher replay")
            report = {
                "status": "pass" if passed else "fail",
                "rank": rank,
                "scope": scope,
                "precision_verified": False, "performance_measured": False,
                "layers": args.layers, "steps": len(teacher),
                "capacity": CAPACITY, "query_width": args.query_width,
                "position": engine.position, token_field: teacher,
                "expected_tokens": expected, "predicted_tokens": predictions,
                "greedy_correct": sum(a == b for a, b
                                      in zip(scored_predictions, expected)),
                "greedy_total": len(scored_predictions),
                "full_model_executed": args.layers == 78,
                "attention_kernel": args.attention_kernel,
                "fp32_head": args.fp32_head,
                "model_tp_reduction": engine.tp_reduction,
                "checkpoint_sha256": fingerprint, "inputs": inputs,
                "teacher_source": teacher_source,
                "base_load_coverage": coverage,
                "source_sha256": sources, "artifact_sha256": artifacts,
                "baseline_sha256": baseline_hashes, "layers_report": layers,
                "torch_quota_bytes": QUOTA_BYTES, "checks": checks,
                "command": sys.argv,
                "packages": {name: version(name) for name in (
                    "torch", "torch-npu", "vllm", "vllm-ascend")},
                "environment": {name: os.environ.get(name) for name in (
                    "ASCEND_CUSTOM_OPP_PATH", "HCCL_OP_EXPANSION_MODE",
                    "HCCL_DETERMINISTIC", "HCCL_BUFFSIZE",
                    "ASCEND_LAUNCH_BLOCKING", "TASK_QUEUE_ENABLE")},
            }
            if round_trace:
                report.update(round_trace=round_trace, scored_rows=scored_rows,
                              scored_predictions=scored_predictions,
                              target_graph=False, conditioned_mtp=False)
            path = args.output / f"rank-{rank}.json"
            checked("report", lambda: write_json(path, report))
            ranks = gather({"rank": rank, "status": report["status"],
                            "report_sha256": digest(path)})
            passed = all(item["status"] == "pass" for item in ranks)
            if rank == 0:
                write_json(args.output / "summary.json", {
                    "status": "pass" if passed else "fail", "ranks": ranks,
                    "precision_verified": False, "performance_measured": False,
                })
            return 0 if passed else 1
        finally:
            engine.shutdown()


def require_released(alive):
    if alive:
        raise RuntimeError(f"Layer owners remain resident: {alive}")


def main():
    args = arguments()
    args.layer = None
    try:
        return execute(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / f"rank-{os.environ.get('RANK')}-error.json"
        if not path.exists():
            write_json(path, {"layer": args.layer, "status": "error",
                              "replay_call": getattr(args, "replay_call",
                                                     None),
                              "round_trace": (str(args.round_trace)
                                              if args.round_trace else None),
                              "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
