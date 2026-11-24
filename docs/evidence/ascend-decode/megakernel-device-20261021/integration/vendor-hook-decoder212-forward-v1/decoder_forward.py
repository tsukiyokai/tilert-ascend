"""Compare real decoder calls with explicit local TP boundary inputs."""

from contextlib import contextmanager
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import traceback
from types import MethodType


def tensor_witness(torch, value):
    raw = value.detach().cpu().contiguous()
    return {"shape": list(raw.shape), "dtype": str(raw.dtype),
            "sha256": hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest(),
            "nonzero": int(torch.count_nonzero(raw)),
            "finite": bool(torch.isfinite(raw).all())}


class TPCuts:
    """Expose two local boundaries without emulating a multi-rank reduction."""

    def __init__(self, torch, post_attention):
        if (post_attention.dtype != torch.bfloat16 or post_attention.ndim != 2
                or post_attention.shape[1] != 6144 or not 1 <= post_attention.shape[0] <= 32):
            raise ValueError("The external post-AR1 input requires BF16 [M,6144], M=1..32")
        self.torch = torch
        self.post_attention = post_attention
        self.phase = None
        self.values = []
        self.trace = []

    def reduce(self, value):
        expected = ("attention", "ffn")
        ordinal = len(self.values)
        if ordinal >= 2 or self.phase != expected[ordinal]:
            raise RuntimeError(f"Unexpected TP reduction: {self.phase}, ordinal {ordinal}")
        if (value.dtype != self.torch.bfloat16
                or value.shape != self.post_attention.shape
                or value.device != self.post_attention.device):
            raise RuntimeError("The TP cut requires BF16 [M,6144] on the fixture device")
        self.values.append(value.clone())
        self.trace.append({"ordinal": ordinal + 1, "phase": self.phase,
                           "source_data_ptr": value.data_ptr(),
                           "action": "external_post_ar1" if ordinal == 0 else "local_pre_ar2",
                           "stack": [{"file": frame.filename, "line": frame.lineno,
                                      "function": frame.name}
                                     for frame in traceback.extract_stack(limit=10)[:-1]]})
        return self.post_attention.clone() if ordinal == 0 else value

    def finish(self):
        if len(self.values) != 2 or self.phase is not None:
            raise RuntimeError("A complete decoder must reach exactly two declared TP cuts")


@contextmanager
def decoder_cuts(torch, group, layer, post_attention):
    cuts = TPCuts(torch, post_attention)
    original = group.all_reduce
    handles = []

    def enter(phase):
        def hook(unused_module, unused_inputs):
            if cuts.phase is not None:
                raise RuntimeError("The decoder module phases unexpectedly overlap")
            cuts.phase = phase
        return hook

    def leave(phase):
        def hook(unused_module, unused_inputs, output):
            if cuts.phase != phase:
                raise RuntimeError("The decoder module phases are unbalanced")
            if phase == "attention":
                cuts.attention_output = output.clone()
            cuts.phase = None
        return hook

    group.all_reduce = MethodType(lambda unused_group, value: cuts.reduce(value), group)
    for phase, module in (("attention", layer.self_attn), ("ffn", layer.mlp)):
        handles.append(module.register_forward_pre_hook(enter(phase)))
        handles.append(module.register_forward_hook(leave(phase)))
    try:
        yield cuts
        cuts.finish()
    finally:
        group.all_reduce = original
        for handle in handles:
            handle.remove()


def metadata_for_layers(torch, torch_npu, layers, plan, device):
    """Use actual metadata classes; the shared layer owns only its main cache."""
    from dataclasses import replace
    from vendor_hook_forward import make_metadata

    positions, metadata = make_metadata(torch, torch_npu, layers[0].self_attn, plan, device)
    first = layers[0].self_attn.mla_attn.mla_attn
    second = layers[1].self_attn.mla_attn.mla_attn
    source = metadata[first.layer_name]
    metadata[second.layer_name] = replace(
        source, slot_mapping=source.slot_mapping.clone(),
        block_table=source.block_table.clone(), sin=source.sin.clone(), cos=source.cos.clone())
    if second.impl.indexer is not None or not second.impl.skip_topk:
        raise RuntimeError("The actual shared-indexer successor changed selection")
    return positions, metadata


def bind_caches(torch, torch_npu, layers, config, mode):
    from vllm.v1.worker.utils import bind_kv_cache

    def make(shape, dtype):
        return torch_npu.npu_format_cast(torch.zeros(shape, dtype=dtype, device="npu"), 2)

    allocated = {}
    names = {}
    for number, layer in zip((6, 7), layers):
        attention = layer.self_attn.mla_attn.mla_attn
        allocated[attention.layer_name] = tuple(
            make((66, 128, 1, width), torch.bfloat16) for width in (512, 64))
        names[f"layer{number}.main_cache"] = allocated[attention.layer_name][0]
        names[f"layer{number}.rope_cache"] = allocated[attention.layer_name][1]
        if attention.impl.indexer is not None:
            key = make((66, 128, 1, 128), torch.int8 if mode == "int8" else torch.bfloat16)
            values = (key,)
            names[f"layer{number}.index_cache"] = key
            if mode == "int8":
                scale = make((66, 128, 1, 1), torch.float16)
                values += (scale,)
                names[f"layer{number}.index_scale"] = scale
            allocated[attention.impl.indexer.k_cache.prefix] = values
    owners = []
    bind_kv_cache(allocated, config.compilation_config.static_forward_context, owners)
    for layer in layers:
        attention = layer.self_attn.mla_attn.mla_attn
        if any(actual.data_ptr() != expected.data_ptr()
               for actual, expected in zip(attention.kv_cache, allocated[attention.layer_name])):
            raise RuntimeError("The official cache binder changed a supplied cache owner")
    return names, owners


def run_decoder_forward(torch, layers, config, state, args):
    import torch_npu
    import triton
    from attention_check import metric
    from vendor_hook_forward import (
        count_native_calls, load_sfa_adapter, page_plan, snapshot, source_identity,
    )
    from vllm.forward_context import set_forward_context
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    init_device_properties_triton()
    properties = triton.runtime.driver.active.utils.get_device_properties(torch.npu.current_device())
    if len(layers) != 2:
        raise ValueError("The fixture requires actual decoder layers 6 and 7")
    topk = layers[0].self_attn.mla_attn.topk_indices_buffer
    if (topk.device.type != "npu" or topk.dtype != torch.int32
            or tuple(topk.shape) != (32, 2048)):
        raise RuntimeError("The actual constructor must already own a persistent NPU top-k buffer")

    def topk_owners():
        entries = []
        for number, layer in zip((6, 7), layers):
            wrapper = layer.self_attn.mla_attn
            impl = wrapper.mla_attn.impl
            if wrapper.topk_indices_buffer is not topk or impl.topk_indices_buffer is not topk:
                raise RuntimeError("Both actual attention layers must retain the same NPU top-k owner")
            entries.append({"layer": number, "device": str(topk.device),
                            "data_ptr": topk.data_ptr(),
                            "storage_ptr": topk.untyped_storage().data_ptr(),
                            "shape": list(topk.shape)})
        return entries

    before_cache_owners = topk_owners()
    caches, cache_owners = bind_caches(torch, torch_npu, layers, config, args.mode)
    after_cache_owners = topk_owners()
    images, bases = {}, {}
    generator = torch.Generator().manual_seed(42022)
    for name, cache in caches.items():
        if cache.dtype == torch.int8:
            value = torch.randint(-127, 128, cache.shape, generator=generator, dtype=torch.int8)
        elif name.endswith("index_scale"):
            value = (torch.rand(cache.shape, generator=generator) / 128 + 1 / 64).to(cache.dtype)
        else:
            value = torch.randn(cache.shape, generator=generator).to(cache.dtype)
        images[name] = value
        bases[name] = torch_npu.npu_format_cast(value.to("npu"), 2)

    os.environ["TILERT_INDEXED_ATTENTION_LIBRARY"] = str(args.indexed_library)
    os.environ["TILERT_SHARED_ATTENTION_LIBRARY"] = str(args.shared_library)
    sys.path.insert(0, str(args.adapter.resolve().parents[3]))
    adapter = importlib.import_module("tilert.models.glm_5_2_ascend.attention_native")
    if Path(adapter.__file__).resolve() != args.adapter.resolve():
        raise RuntimeError("The imported native attention adapter differs from the frozen source")
    sfa = load_sfa_adapter(torch, args.sfa_library)
    adapter.load()
    pool = adapter.WorkspacePool(blocks=24)
    implementations = [layer.self_attn.mla_attn.mla_attn.impl for layer in layers]
    originals = [impl.forward for impl in implementations]
    restores = []
    for layer in layers:
        restores.append(adapter.install(layer.self_attn.mla_attn.mla_attn, pool))
    installed = [impl.forward for impl in implementations]
    report = {
        "scope": "Original DecoderLayer.forward for layers 6 then 7; native attention versus original attention; both arms use the official FFN",
        "script": source_identity(run_decoder_forward),
        "decoder_forward": source_identity(type(layers[0]).forward),
        "forward_context": source_identity(inspect.unwrap(set_forward_context)),
        "adapter": source_identity(adapter.run_body), "sfa_adapter": sfa,
        "native_libraries": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                             for path in (args.indexed_library, args.shared_library)],
        "triton_initializer": source_identity(init_device_properties_triton),
        "device_properties": properties, "mode": args.mode,
        "widths": args.forward_widths, "iterations": args.forward_replays,
        "tp_geometry": 16, "tp_collectives_executed": 0,
        "ar1_contract": "Independent BF16 post-attention reduction input supplied identically to both arms",
        "ar2_contract": "Return the rank-local FFN partial; no numerical multi-rank sum is emulated",
        "successor_input_contract": "Layer7 consumes layer6 local pre-AR2 output and its separately returned residual",
        "topk_owners_before_cache_binding": before_cache_owners,
        "topk_owners_after_cache_binding": after_cache_owners,
        "cache_shapes": {name: list(value.shape) for name, value in caches.items()},
        "cases": [], "pass": False,
    }
    output_path = args.output / "decoder-forward.json"

    def persist():
        output_path.write_text(json.dumps(report, indent=2) + "\n")

    def reset():
        for name, cache in caches.items():
            cache.copy_(bases[name])
        topk.fill_(-1)

    def call(forwards, hidden, residual, plan, post_ar1, case, arm):
        for impl, forward in zip(implementations, forwards):
            impl.forward = forward
        reset()
        positions, metadata = metadata_for_layers(torch, torch_npu, layers, plan, hidden.device)
        slots = {name: item.slot_mapping for name, item in metadata.items()}
        values = {}
        details = {}
        hidden = hidden.clone()
        residual = None if residual is None else residual.clone()
        with set_forward_context(metadata, config, num_tokens=hidden.shape[0], slot_mapping=slots):
            for number, layer, boundary in zip((6, 7), layers, post_ar1):
                norm_outputs = {}

                def remember_norm(module, unused_inputs, outputs):
                    label = "input_norm" if module is layer.input_layernorm else "post_norm"
                    outputs = outputs if isinstance(outputs, tuple) else (outputs,)
                    norm_outputs[label] = (outputs, tuple(value.clone() for value in outputs))

                hooks = [norm.register_forward_hook(remember_norm)
                         for norm in (layer.input_layernorm, layer.post_attention_layernorm)]
                try:
                    with decoder_cuts(torch, state.get_tp_group(), layer, boundary) as cuts:
                        hidden, residual = layer(positions, hidden, residual)
                    torch.npu.synchronize()
                    post_original, _ = norm_outputs["post_norm"]
                    if len(post_original) != 2 or residual is not post_original[1]:
                        raise RuntimeError("DecoderLayer did not retain its actual post-attention residual separately")
                    for ordinal, value in enumerate(cuts.values, 1):
                        values[f"layer{number}.pre_ar{ordinal}"] = snapshot(torch, value)
                    attention_output = snapshot(torch, cuts.attention_output)
                    if not torch.equal(attention_output.view(torch.uint8),
                                       snapshot(torch, boundary).view(torch.uint8)):
                        raise RuntimeError("The attention module did not consume the external post-AR1 boundary")
                    values[f"layer{number}.post_ar1"] = attention_output
                    for label, (_, outputs) in norm_outputs.items():
                        for index, value in enumerate(outputs):
                            values[f"layer{number}.{label}.{index}"] = snapshot(torch, value)
                    values[f"layer{number}.output"] = snapshot(torch, hidden)
                    values[f"layer{number}.residual"] = snapshot(torch, residual)
                    values[f"layer{number}.topk"] = snapshot(torch, topk)
                    details[f"layer{number}"] = {
                        "tp_cuts": cuts.trace, "residual_is_post_norm_owner": True,
                        "output": tensor_witness(torch, hidden),
                        "retained_residual": tensor_witness(torch, residual),
                        "post_ar1": tensor_witness(torch, boundary)}
                    if not torch.equal(values[f"layer{number}.output"].view(torch.uint8),
                                       values[f"layer{number}.pre_ar2"].view(torch.uint8)):
                        raise RuntimeError("The decoder changed the local FFN partial after the second cut")
                finally:
                    for handle in hooks:
                        handle.remove()
        if not torch.equal(values["layer6.topk"].view(torch.uint8),
                           values["layer7.topk"].view(torch.uint8)):
            raise RuntimeError("The shared-indexer successor changed the full-indexer's top-k owner")
        topk_owners()
        values.update({name: snapshot(torch, cache) for name, cache in caches.items()})
        case.setdefault("arms", {})[arm] = details
        persist()
        return values

    persist()
    try:
        with torch.inference_mode(), count_native_calls(adapter) as hits:
            for rows in args.forward_widths:
                for iteration in range(args.forward_replays):
                    generator = torch.Generator().manual_seed(42023 + rows * 100 + iteration)
                    hidden_cpu = torch.randn(rows, 6144, generator=generator).bfloat16()
                    residual_cpu = (torch.randn(rows, 6144, generator=generator).bfloat16()
                                    if iteration % 2 else None)
                    boundary_cpu = [torch.randn(rows, 6144, generator=generator).bfloat16()
                                    for unused in layers]

                    def nd(value):
                        return torch_npu.npu_format_cast(value.to("npu"), 2)

                    hidden = nd(hidden_cpu)
                    residual = None if residual_cpu is None else nd(residual_cpu)
                    boundary = [nd(value) for value in boundary_cpu]
                    plan = page_plan(torch, rows, iteration, args.forward_sequence, 66, 66)
                    case = {"rows": rows, "iteration": iteration, "pass": False,
                            "initial_residual": "provided" if residual is not None else "None",
                            "input": tensor_witness(torch, hidden),
                            "lengths": plan["lengths"].tolist(),
                            "phase": "official_decoder"}
                    report["cases"].append(case)
                    persist()
                    before = len(hits)
                    original = call(originals, hidden, residual, plan, boundary, case, "original")
                    if len(hits) != before:
                        raise RuntimeError("The original decoder unexpectedly invoked native attention")
                    case["phase"] = "native_attention_decoder"
                    persist()
                    native = call(installed, hidden, residual, plan, boundary, case, "native_attention")
                    if len(hits) != before + 2 or [x["mode"] for x in hits[before:]] != [args.mode, "shared"]:
                        raise RuntimeError("Both real decoder layers must invoke their intended native attention body")
                    checks = {name: metric(torch, native[name], expected)
                              for name, expected in original.items()}
                    passed = all(item["pass"] and item["bitwise_equal"] for item in checks.values())
                    case.update({"pass": passed, "phase": "comparison", "checks": checks,
                                 "native_calls": hits[before:]})
                    persist()
                    print(json.dumps({"decoder_forward_case": case}), flush=True)
                    if not passed:
                        dump = args.output / f"decoder-forward-{args.mode}-m{rows}-r{iteration}.pt"
                        torch.save({"hidden": hidden_cpu, "residual": residual_cpu,
                                    "post_ar1": boundary_cpu, "metadata": plan,
                                    "original": original, "native": native}, dump)
                        case["failure_dump"] = str(dump)
                        persist()
                        raise RuntimeError("The real local decoder comparison differs")
        report["pass"] = True
        persist()
        return report
    except Exception:
        report["error"] = traceback.format_exc()
        persist()
        raise
    finally:
        for impl, forward in zip(implementations, installed):
            impl.forward = forward
        for restore in reversed(restores):
            restore()
