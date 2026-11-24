"""Capture the real indexer Q/K boundaries and retained native producer regions."""

from contextlib import contextmanager, ExitStack
import hashlib
import importlib
import json
import traceback

from vendor_hook_diagnostic import compare, cpu_copy, replace_attribute
from vendor_hook_diagnostic import tensor_identity
from vendor_hook_forward import source_identity


def decode_front(torch, raw, offsets, rows, quantized):
    """Decode live producer regions; the vendor work arenas are not inspected."""
    capacity = offsets["capacity"]

    def region(offset, shape, dtype):
        elements = 1
        for extent in shape:
            elements *= extent
        size = elements * torch.empty((), dtype=dtype).element_size()
        return raw.narrow(0, offset, size).view(dtype).reshape(shape).clone()

    shapes = {
        "query_projection": ((rows, 32, 128), torch.bfloat16),
        "key_projection": ((rows, 160), torch.bfloat16),
        "key_normalized": ((rows, 128), torch.bfloat16),
        "query_rope": ((rows, 32, 128), torch.bfloat16),
        "key_rope": ((rows, 128), torch.bfloat16),
        "norm_stats": ((rows, 2), torch.float32),
    }
    values = {name: region(offsets["front_" + name], shape, dtype)
              for name, (shape, dtype) in shapes.items()}
    wire_dtype = torch.int8 if quantized else torch.bfloat16
    for name, shape, dtype in (
            ("query_norm", (rows, 2048), torch.int8),
            ("query_norm_scale", (rows,), torch.float32),
            ("query", (rows, 32, 128), wire_dtype),
            ("key", (rows, 128), wire_dtype),
            ("weights", (rows, 32), torch.float16 if quantized else torch.bfloat16)):
        values[name] = region(offsets[name], shape, dtype)
    if quantized:
        for name, shape in (("query", (rows, 32, 128)),
                            ("key", (capacity, 128))):
            value = region(offsets["front_" + name + "_hadamard"],
                           shape, torch.float32)[:rows]
            values[name + "_hadamard_fp32"] = value.clone()
            values[name + "_hadamard"] = value.bfloat16()
        values["query_scale"] = region(offsets["query_scale"],
                                        (rows, 32), torch.float16)
        values["key_scale"] = region(offsets["key_scale"],
                                      (rows,), torch.float16)
    front = offsets["front_begin"]
    key_acc = front + capacity * 4096 * 4
    if offsets["front_query_projection"] != key_acc + capacity * 160 * 4:
        raise RuntimeError("The executed front accumulator layout changed")
    q_acc = region(front, (16, capacity, 256), torch.int32)
    k_acc = region(key_acc, (5, capacity, 32), torch.float32)
    values["query_accumulators"] = q_acc.permute(1, 0, 2).reshape(
        capacity, 4096)[:rows].clone()
    values["key_accumulators"] = k_acc.permute(1, 0, 2).reshape(
        capacity, 160)[:rows].clone()
    return values


def rope_reference(value, cosine, sine):
    """CPU FP32 products/sum and BF16 rounding, for the interleaved RoPE64 path."""
    rows = value.shape[0]
    source = value.reshape(rows, -1, 128).float()
    cosine = cosine.reshape(rows, 1, 64)[..., :32].float()
    sine = sine.reshape(rows, 1, 64)[..., :32].float()
    result = source.clone()
    left, right = source[..., :64:2], source[..., 1:64:2]
    result[..., :64:2] = left * cosine - right * sine
    result[..., 1:64:2] = right * cosine + left * sine
    return result.bfloat16().reshape(value.shape)


class IndexerTaps:
    def __init__(self, torch, torch_npu, impl, adapter):
        self.torch = torch
        self.torch_npu = torch_npu
        self.impl = impl
        self.backend = adapter._indexer_backend(impl.indexer)
        self.adapter = adapter
        self.module = importlib.import_module(type(self.backend).__module__)
        self.vendor = {}
        self.native = {}
        self.geometry = None
        self.quantizing = None
        self.active = False
        self.calls = {"vendor": 0, "native": 0, "rope_snapshots": 0}
        self.descriptors = {}
        self.sources = {
            name: source_identity(value) for name, value in (
                ("adapter_run_body", adapter.run_body),
                ("indexer_forward", self.backend.forward),
                ("indexer_forward_k", self.backend.forward_k),
                ("indexer_quantize", self.backend._quantize_li_tensor),
                ("rope", self.module.rope_forward_triton_siso),
                ("key_projection_class", type(self.backend.wk_weights_proj)),
                ("key_norm_class", type(self.backend.k_norm)),
                ("key_norm_dispatch", self.backend.k_norm._forward_method),
            )
        }

    def descriptor(self, value):
        return {"shape": list(value.shape), "stride": list(value.stride()),
                "dtype": str(value.dtype), "data_ptr": value.data_ptr(),
                "storage_ptr": value.untyped_storage().data_ptr(),
                "storage_offset": value.storage_offset(),
                "format": self.torch_npu.get_npu_format(value)}

    @contextmanager
    def installed(self):
        backend = self.backend
        forward = backend.forward
        key_projection = backend.wk_weights_proj.forward
        layer_norm = self.torch.nn.functional.layer_norm
        rotate = self.module.rope_forward_triton_siso
        quantize = backend._quantize_li_tensor
        dynamic_quant = self.torch_npu.npu_dynamic_quant
        native_body = self.adapter.run_body

        def tapped_forward(hidden, query_norm, key_hidden, metadata, *args, **kwargs):
            if not isinstance(query_norm, tuple) or len(query_norm) != 2:
                raise RuntimeError("The diagnostic requires the actual W8 query-norm tuple")
            self.vendor = {"hidden": hidden, "key_hidden": key_hidden,
                           "query_norm": query_norm[0],
                           "query_norm_scale": query_norm[1],
                           "cosine": metadata.cos, "sine": metadata.sin,
                           "index_slots": metadata.slot_mapping,
                           "key_weight": backend.wk_weights_proj.weight,
                           "query_weight": backend.wq_b.weight,
                           "query_weight_scale": backend.wq_b.weight_scale,
                           "gamma": backend.k_norm.weight,
                           "beta": backend.k_norm.bias}
            self.descriptors = {"same_query_key_hidden": hidden is key_hidden,
                                "normalization_epsilon": backend.k_norm.eps,
                                "rope_is_neox": backend.is_rope_neox_style,
                                "layer_norm_calls": 0, "key_projection_calls": 0}
            self.active = True
            try:
                result = forward(hidden, query_norm, key_hidden, metadata,
                                 *args, **kwargs)
            finally:
                self.active = False
            self.calls["vendor"] += 1
            return result

        def tapped_key_projection(*args, **kwargs):
            result = key_projection(*args, **kwargs)
            if self.active:
                self.vendor["key_projection"] = result[0]
                self.descriptors["key_projection_calls"] += 1
            return result

        def tapped_layer_norm(value, normalized_shape, weight=None, bias=None, eps=1e-5):
            result = layer_norm(value, normalized_shape, weight, bias, eps)
            if self.active and tuple(normalized_shape) == (128,):
                self.vendor["key_norm_fp32_input"] = value
                self.vendor["key_norm_fp32_output"] = result
                self.descriptors["layer_norm_calls"] += 1
                self.descriptors["layer_norm"] = {
                    "input": self.descriptor(value),
                    "output": self.descriptor(result), "epsilon": eps,
                    "weight": self.descriptor(weight),
                    "bias": self.descriptor(bias)}
            return result

        def tapped_rotate(value, *args, **kwargs):
            name = None
            if self.active:
                if value.ndim != 3 or value.shape[-1] != 128 or value.shape[1] not in (1, 32):
                    raise RuntimeError("Unexpected real indexer RoPE shape")
                name = "key" if value.shape[1] == 1 else "query"
                # The official function overwrites qk. A retained reference
                # alone would no longer represent the projection/LN boundary.
                saved = value.clone()
                boundary = "key_normalized" if name == "key" else "query_projection"
                self.vendor[boundary] = saved
                self.calls["rope_snapshots"] += 1
                self.descriptors[name + "_rope"] = {
                    "original_input": self.descriptor(value),
                    "pre_rope_snapshot": self.descriptor(saved)}
            result = rotate(value, *args, **kwargs)
            if name is not None:
                self.vendor[name + "_rope"] = result
                self.vendor[name] = result
                self.descriptors[name + "_rope"]["output"] = self.descriptor(result)
            return result

        def tapped_quantize(value):
            previous = self.quantizing
            if self.active:
                self.quantizing = "key" if value.shape[1] == 1 else "query"
            try:
                result = quantize(value)
                if self.quantizing is not None:
                    self.vendor[self.quantizing] = result[0]
                    self.vendor[self.quantizing + "_scale"] = result[1]
                return result
            finally:
                self.quantizing = previous

        def tapped_dynamic_quant(value, *args, **kwargs):
            result = dynamic_quant(value, *args, **kwargs)
            if self.quantizing is not None:
                self.vendor[self.quantizing + "_hadamard"] = value
                self.vendor[self.quantizing + "_scale_fp32"] = result[1]
            return result

        def tapped_native(*args, **kwargs):
            result = native_body(*args, **kwargs)
            hidden, main, front, tail, caches, metadata, index, pool = args
            geometry = self.adapter.geometry_for(kwargs["mode"], caches, metadata, index)
            entry = pool.get(hidden, geometry)
            self.native = {"hidden": hidden, "scratch": entry[0],
                           "front": front, "blocks": entry[2],
                           "cosine": index[1], "sine": index[0],
                           "index_slots": index[2]}
            self.geometry = geometry
            self.calls["native"] += 1
            return result

        with ExitStack() as stack:
            for owner, name, value in (
                    (backend, "forward", tapped_forward),
                    (backend.wk_weights_proj, "forward", tapped_key_projection),
                    (self.torch.nn.functional, "layer_norm", tapped_layer_norm),
                    (self.module, "rope_forward_triton_siso", tapped_rotate),
                    (backend, "_quantize_li_tensor", tapped_quantize),
                    (self.torch_npu, "npu_dynamic_quant", tapped_dynamic_quant),
                    (self.adapter, "run_body", tapped_native)):
                stack.enter_context(replace_attribute(owner, name, value))
            yield

    def read_native(self):
        torch = self.torch
        hidden = self.native["hidden"]
        geometry = self.geometry
        codes = torch.empty_like(hidden, dtype=torch.int8)
        temporary, tiling, offsets, blocks, sync = torch.ops.tilert.indexed_attention_prepare(
            codes, geometry.pages, geometry.table_pages, geometry.index_pages,
            geometry.index_table_pages, geometry.mode == "int8", self.native["blocks"])
        if temporary.nbytes != self.native["scratch"].nbytes:
            raise RuntimeError("The executed workspace differs from its declared layout")
        del temporary, tiling, sync, codes
        raw = cpu_copy(torch, self.native["scratch"])
        values = decode_front(torch, raw, offsets, hidden.shape[0], geometry.mode == "int8")
        front = [cpu_copy(torch, value) for value in self.native["front"]]
        values.update({
            "query_weight": front[0].permute(1, 0, 2).reshape(2048, 4096).clone(),
            "query_weight_scale": front[1],
            "key_weight": front[2].permute(1, 0, 2).reshape(6144, 160).t().contiguous(),
            "gamma": front[3], "beta": front[4], "hadamard": front[5],
        })
        for name in ("hidden", "cosine", "sine", "index_slots"):
            values[name] = cpu_copy(torch, self.native[name])
        return values, dict(offsets), blocks


def normalize_shapes(values, rows):
    shapes = {
        "query_norm": (rows, 2048), "query_norm_scale": (rows,),
        "query_projection": (rows, 32, 128), "key_projection": (rows, 160),
        "key_normalized": (rows, 128), "query_rope": (rows, 32, 128),
        "key_rope": (rows, 128), "query": (rows, 32, 128), "key": (rows, 128),
        "query_scale": (rows, 32), "key_scale": (rows,),
        "query_hadamard": (rows, 32, 128), "key_hadamard": (rows, 128),
        "query_scale_fp32": (rows, 32), "key_scale_fp32": (rows,),
        "cosine": (rows, 64), "sine": (rows, 64), "index_slots": (rows,),
    }
    for name, shape in shapes.items():
        if name in values:
            values[name] = values[name].reshape(shape)


def reproduction_checks(torch, parent, current):
    checks = {"hidden": compare(torch, current["hidden"], parent["hidden"])}
    for name, expected in parent["metadata"].items():
        checks["metadata/" + name] = compare(torch, current["metadata"][name], expected)
    for arm in ("baseline", "native"):
        for name, expected in parent[arm].items():
            checks[arm + "/" + name] = compare(torch, current[arm][name], expected)
    return {"pass": all(item["bitwise_equal"] for item in checks.values()),
            "checks": checks}


def save_index_diagnostic(torch, taps, args, outer_error):
    torch.npu.synchronize()
    vendor = {name: cpu_copy(torch, value) for name, value in taps.vendor.items()}
    original_shapes = {name: list(value.shape) for name, value in vendor.items()}
    native, offsets, blocks = taps.read_native()
    rows = native["hidden"].shape[0]
    normalize_shapes(vendor, rows)
    normalize_shapes(native, rows)
    quantized = args.mode == "int8"
    vendor["weights"] = vendor["key_projection"][:, 128:].to(
        torch.float16 if quantized else torch.bfloat16).contiguous()
    order = ["hidden", "query_norm", "query_norm_scale", "key_projection",
             "key_normalized", "key_rope", "query_projection", "query_rope"]
    if quantized:
        order.extend(("key_hadamard", "key", "key_scale", "query_hadamard",
                      "query", "query_scale"))
    else:
        order.extend(("key", "query"))
    order.append("weights")
    checks = {name: compare(torch, native[name], vendor[name]) for name in order}
    for name in ("key_weight", "query_weight", "query_weight_scale", "gamma",
                 "beta", "cosine", "sine", "index_slots"):
        checks[name] = compare(torch, native[name], vendor[name])
    oracles = {}
    for arm, values in (("vendor", vendor), ("native", native)):
        for name, source in (("key", "key_normalized"),
                             ("query", "query_projection")):
            output = rope_reference(values[source], values["cosine"], values["sine"])
            label = arm + "_" + name + "_rope_cpu_fp32"
            oracles[label] = output
    oracle_checks = {
        name: compare(torch, value,
                      (vendor if name.startswith("vendor") else native)[
                          "key_rope" if "_key_" in name else "query_rope"])
        for name, value in oracles.items()}
    wk_dot = vendor["key_hidden"].double() @ vendor["key_weight"].double().t()
    oracles["key_projection_cpu_fp64_dot"] = wk_dot
    for arm, values in (("vendor", vendor), ("native", native)):
        oracle_checks[arm + "_key_projection_cpu_fp64"] = compare(
            torch, wk_dot.bfloat16(), values["key_projection"])
    q_dot = vendor["query_norm"].long() @ vendor["query_weight"].long()
    if int(q_dot.min()) < -(1 << 31) or int(q_dot.max()) >= 1 << 31:
        raise RuntimeError("The CPU WQ integer oracle exceeds INT32")
    oracles["query_accumulators_cpu_i32"] = q_dot.int()
    oracle_checks["native_query_accumulators_cpu_i32"] = compare(
        torch, native["query_accumulators"], q_dot.int())
    outer = json.loads((args.output / "outer-forward.json").read_text())
    case = outer["cases"][-1]
    reproduction = None
    if args.index_diagnostic_parent is not None:
        if "failure_dump" in case:
            parent = torch.load(args.index_diagnostic_parent, map_location="cpu", weights_only=False)
            current = torch.load(case["failure_dump"], map_location="cpu", weights_only=False)
            reproduction = reproduction_checks(torch, parent, current)
        else:
            reproduction = {"pass": False, "reason": "No final failure dump was produced"}
        reproduction["parent_path"] = str(args.index_diagnostic_parent)
        reproduction["parent_sha256"] = hashlib.sha256(
            args.index_diagnostic_parent.read_bytes()).hexdigest()
    dump = args.output / "outer-index-diagnostic.pt"
    torch.save({"vendor": vendor, "native": native, "oracles": oracles,
                "offsets": offsets, "case": case}, dump)
    report = {
        "scope": "Actual outer indexer calls; existing arithmetic and official operators are unchanged",
        "script": source_identity(save_index_diagnostic), "sources_before_taps": taps.sources,
        "perturbation": "Two device clones per vendor call preserve Q/K values before in-place RoPE; other taps extend tensor lifetimes. No timing claim.",
        "sampling": "Latest complete vendor/native call; CPU reads follow outer synchronization",
        "case": {"rows": case["rows"], "iteration": case["iteration"],
                 "original_final_gate_pass": case["pass"]},
        "outer_error": outer_error, "calls": taps.calls, "blocks": blocks,
        "first_unequal_boundary_in_recorded_order": next(
            (name for name in order if not checks[name]["bitwise_equal"]), None),
        "production_boundary_order": order, "checks": checks,
        "rope_cpu_oracle_checks": oracle_checks,
        "cpu_oracle_scope": "RoPE uses separate FP32 products/sum; WK uses a CPU FP64 dot as an auxiliary reference, not a claim about device accumulation; WQ uses an independent INT64 dot checked for INT32 range",
        "norm_stats_scope": "Native diagnostic mean/rstd only; official F.layer_norm does not return these statistics",
        "vendor_original_shapes": original_shapes,
        "shape_normalization": "CPU views only; one key head, 32 query heads, one scale per head/token",
        "vendor_descriptors": taps.descriptors, "original_failure_reproduction": reproduction,
        "vendor_tensors": {name: tensor_identity(torch, value) for name, value in vendor.items()},
        "native_tensors": {name: tensor_identity(torch, value) for name, value in native.items()},
        "dump": {"path": str(dump), "sha256": hashlib.sha256(dump.read_bytes()).hexdigest()},
    }
    destination = args.output / "outer-index-diagnostic.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"outer_index_diagnostic": report}), flush=True)
    return report


def run_index_diagnostic(torch, layer, config, adapter, original_forward, args):
    import torch_npu
    from vendor_hook_forward import run_outer_forward

    taps = IndexerTaps(torch, torch_npu, layer.mla_attn.mla_attn.impl, adapter)
    try:
        with taps.installed():
            result = run_outer_forward(torch, layer, config, adapter, original_forward, args)
    except Exception:
        outer_error = traceback.format_exc()
        try:
            save_index_diagnostic(torch, taps, args, outer_error)
        except Exception:
            error = {"outer_error": outer_error, "diagnostic_error": traceback.format_exc()}
            (args.output / "outer-index-diagnostic-error.json").write_text(
                json.dumps(error, indent=2) + "\n")
            print(json.dumps(error), flush=True)
        raise
    report = save_index_diagnostic(torch, taps, args, None)
    if args.index_diagnostic_parent is not None and not report["original_failure_reproduction"]["pass"]:
        raise RuntimeError("The retained-stage diagnostic did not reproduce its parent failure")
    return result
