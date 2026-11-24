"""Own layer68 snapshots without changing arithmetic or moving tensors to CPU.

The runner supplies frozen ``run_cann_model`` and ``routes`` on sys.path.
Use one context per forward. Repeated calls within that forward are retained.
"""

import hashlib
import inspect
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from unittest.mock import patch

LAYER = 68


def source_identity(function):
    path = Path(inspect.getfile(inspect.unwrap(function))).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {str(path): digest}


@contextmanager
def observe(engine):
    """Yield tensors, metadata, original MoE stages and actual route samples.

    Tensor keys include call numbers, e.g. ``layer68.input_norm.call0.out.1``
    and ``layer68.tp.attention.call0.in``. Metadata records execution order.
    All copies stay on the source tensor's device; no head is evaluated.
    """
    import torch
    from routes import install_routes
    from run_cann_model import capture_mlp

    layer = engine.model.model.layers[LAYER]
    next_norm = engine.model.model.layers[LAYER + 1].input_layernorm
    impl = layer.self_attn.mla_attn.mla_attn.impl
    tensors, route_samples = {}, []
    metadata = {
        "layer": LAYER,
        "scope": "Original forward observations, no precision/performance certification",
        "calls": [],
        "tensor_layouts": {},
        "scalar_values": {},
        "source_sha256": {
            **source_identity(observe),
            **source_identity(capture_mlp),
            **source_identity(install_routes),
        },
        "tp_calls": [],
        "sfa_calls": [],
        "h69": "layer69.input_norm.call0.out.1 is merged residual, not normalized output",
    }
    result = {
        "tensors": tensors,
        "metadata": metadata,
        "moe_stages": None,
        "routes": route_samples,
    }
    counts, pending, phases = {}, {}, []

    def save(tag, value):
        if isinstance(value, torch.Tensor):
            if tag in tensors:
                raise ValueError("Repeated snapshot key: " + tag)
            tensors[tag] = value.detach().clone()
            metadata["tensor_layouts"][tag] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                save(f"{tag}.{index}", item)
        elif isinstance(value, dict):
            for name, item in value.items():
                save(f"{tag}.{name}", item)
        elif value is None or type(value) in (int, float, bool, str):
            metadata["scalar_values"][tag] = value

    def call_tag(name):
        index = counts.get(name, 0)
        counts[name] = index + 1
        return f"{name}.call{index}"

    def module(stack, name, target, phase=None):
        pending[name] = []

        def before(target, args, kwargs):
            tag = call_tag(name)
            pending[name].append(tag)
            save(tag + ".in", args)
            save(tag + ".kwargs", kwargs)
            metadata["calls"].append({"event": "enter", "tag": tag})
            if phase is not None:
                phases.append((phase, tag))

        def after(target, args, kwargs, output):
            tag = pending[name].pop()
            save(tag + ".out", output)
            metadata["calls"].append({"event": "exit", "tag": tag})
            if phase is not None:
                active = phases.pop()
                if active != (phase, tag):
                    raise ValueError("Unexpected nested attention/MLP phase")

        handle = target.register_forward_pre_hook(before, with_kwargs=True)
        stack.callback(handle.remove)
        handle = target.register_forward_hook(after, with_kwargs=True, always_call=True)
        stack.callback(handle.remove)

    original_sfa = impl._execute_sparse_flash_attention_process
    signature = inspect.signature(original_sfa)
    metadata["source_sha256"].update(source_identity(original_sfa))
    original_reduce = engine._tp_sum_original
    metadata["source_sha256"].update(source_identity(original_reduce))

    @wraps(original_sfa)
    def sfa(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        tag = call_tag("layer68.sfa")
        for name in (
            "ql_nope",
            "q_pe",
            "kv_cache",
            "topk_indices",
            "actual_seq_lengths_query",
            "actual_seq_lengths_key",
        ):
            save(tag + "." + name, values[name])
        table = values.get("block_table")
        if table is None:
            table = values["attn_metadata"].block_table
        save(tag + ".block_table", table)
        save(tag + ".W_UV", impl.W_UV)
        metadata["sfa_calls"].append(
            {
                "tag": tag,
                "scale": impl.scale,
                "explicit_block_table": values.get("block_table") is not None,
            }
        )
        output = original_sfa(*args, **kwargs)
        save(tag + ".out", output)
        return output

    @wraps(original_reduce)
    def reduce(value, *args, **kwargs):
        if not phases:
            return original_reduce(value, *args, **kwargs)
        phase, owner = phases[-1]
        tag = call_tag("layer68.tp." + phase)
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ValueError("Expected actual FP32 TP input inside layer68")
        save(tag + ".in", value)
        metadata["tp_calls"].append({"tag": tag, "phase": phase, "owner": owner})
        output = original_reduce(value, *args, **kwargs)
        if not isinstance(output, torch.Tensor) or output.dtype != torch.float32:
            raise ValueError("Expected actual FP32 TP output inside layer68")
        save(tag + ".out", output)
        return output

    with ExitStack() as stack:
        module(stack, "layer68.forward", layer)
        module(stack, "layer68.input_norm", layer.input_layernorm)
        module(stack, "layer68.post_norm", layer.post_attention_layernorm)
        module(stack, "layer68.attention", layer.self_attn, "attention")
        module(stack, "layer68.mlp", layer.mlp, "mlp")
        module(stack, "layer69.input_norm", next_norm)
        stack.enter_context(
            patch.object(impl, "_execute_sparse_flash_attention_process", sfa)
        )
        stack.enter_context(patch.object(engine, "_tp_sum_original", reduce))
        stack.callback(install_routes(layer, route_samples))
        result["moe_stages"] = stack.enter_context(capture_mlp(engine, LAYER))
        try:
            yield result
        finally:
            metadata["call_counts"] = dict(counts)
