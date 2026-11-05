"""Retain actual outer-attention boundaries without changing their arithmetic."""

from contextlib import contextmanager, ExitStack
import hashlib
import json
import traceback


@contextmanager
def replace_attribute(owner, name, replacement):
    present = name in vars(owner)
    previous = vars(owner).get(name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        if present:
            setattr(owner, name, previous)
        else:
            delattr(owner, name)


def cpu_copy(torch, tensor):
    return tensor.detach().cpu().clone(memory_format=torch.contiguous_format)


def raw_digest(torch, tensor):
    raw = tensor.contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def tensor_identity(torch, tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "raw_sha256": raw_digest(torch, tensor),
            "nonzero_elements": int(torch.count_nonzero(tensor)),
            "finite": bool(torch.isfinite(tensor).all())}


def compare(torch, actual, expected):
    from attention_check import metric

    result = metric(torch, actual, expected)
    left = actual.contiguous().reshape(-1).view(torch.uint8)
    right = expected.contiguous().reshape(-1).view(torch.uint8)
    different = (left != right).reshape(-1, actual.element_size()).any(1)
    indices = different.nonzero().flatten()[:16]
    details = []
    for index in indices.tolist():
        position = torch.unravel_index(torch.tensor(index), actual.shape)
        position = tuple(int(value) for value in position)
        details.append({"coordinate": list(position),
                        "native": actual[position].item(),
                        "vendor": expected[position].item()})
    result["first_differences"] = details
    return result


def decode_native(torch, scratch, offsets, rows):
    """Read only retained producer regions, excluding reused vendor arenas."""
    capacity = offsets["capacity"]

    def region(name, shape, dtype):
        elements = 1
        for extent in shape:
            elements *= extent
        size = elements * torch.empty((), dtype=dtype).element_size()
        value = scratch.narrow(0, offsets[name], size)
        value = value.detach().cpu().view(dtype).reshape(shape)
        return value.clone(memory_format=torch.contiguous_format)

    result = {
        "attention_query": region("attention_query", (rows, 4, 512),
                                  torch.bfloat16),
        "attention_rope": region("attention_rope", (rows, 4, 64),
                                 torch.bfloat16),
        "query_norm": region("query_norm", (rows, 2048), torch.int8),
        "query_norm_scale": region("query_norm_scale", (rows,), torch.float32),
        "latent": region("latent", (rows, 4, 512), torch.bfloat16),
    }
    values = region("tail_values", (16, capacity, 64), torch.float32)
    values = values.permute(1, 0, 2).reshape(capacity, 1024)[:rows]
    result["value_up_fp32"] = values.clone()
    result["value_up"] = values.bfloat16()
    result["o_codes"] = region("tail_quantized", (capacity, 1024),
                               torch.int8)[:rows].clone()
    scales = region("tail_scales", (capacity, 8), torch.float32)
    result["o_token_scale"] = scales[:rows, 0].clone()
    accumulators = region("tail_accumulators", (24, capacity, 256), torch.int32)
    result["o_accumulators"] = accumulators.permute(1, 0, 2).reshape(
        capacity, 6144)[:rows].clone()
    return result


class ProductionTaps:
    """Hold existing tensor references until both original calls have finished."""

    def __init__(self, torch, torch_npu, impl, adapter):
        from vendor_hook_forward import source_identity

        self.torch = torch
        self.torch_npu = torch_npu
        self.impl = impl
        self.adapter = adapter
        self.vendor = {}
        self.native = {}
        self.vendor_calls = 0
        self.native_calls = 0
        self.vendor_o_descriptor = {}
        self.native_geometry = None
        self.original_sources = {
            name: source_identity(function) for name, function in (
                ("adapter_run_body", adapter.run_body),
                ("vendor_prolog", impl._sfa_preprocess_prolog_v3),
                ("vendor_sfa", impl._execute_sparse_flash_attention_process),
                ("vendor_value_up", impl._v_up_proj),
            )
        }

    @contextmanager
    def installed(self):
        prolog = self.impl._sfa_preprocess_prolog_v3
        sfa = self.impl._execute_sparse_flash_attention_process
        value_up = self.impl._v_up_proj
        quant_matmul = self.torch_npu.npu_quant_matmul
        native_body = self.adapter.run_body

        def describe_argument(value):
            if isinstance(value, self.torch.Tensor):
                return {"shape": list(value.shape), "dtype": str(value.dtype),
                        "format": self.torch_npu.get_npu_format(value)}
            if value is None or isinstance(value, (bool, int, float, str)):
                return value
            return str(value)

        def tapped_prolog(*args, **kwargs):
            result = prolog(*args, **kwargs)
            self.vendor_calls += 1
            hidden, query, rope, query_norm = result
            if not isinstance(query_norm, tuple) or len(query_norm) != 2:
                raise RuntimeError("Diagnostic requires W8 Prolog query-norm export")
            self.vendor = {
                "hidden": hidden, "attention_query": query,
                "attention_rope": rope, "query_norm": query_norm[0],
                "query_norm_scale": query_norm[1],
            }
            return result

        def tapped_sfa(*args, **kwargs):
            result = sfa(*args, **kwargs)
            self.vendor["latent"] = result
            return result

        def tapped_value_up(*args, **kwargs):
            result = value_up(*args, **kwargs)
            self.vendor["value_up"] = result
            self.vendor["value_weight"] = self.impl.W_UV
            return result

        def tapped_quant_matmul(*args, **kwargs):
            result = quant_matmul(*args, **kwargs)
            if (len(args) >= 3 and args[0].ndim == 2
                    and tuple(args[1].shape) == (1024, 6144)):
                self.vendor.update({
                    "o_codes": args[0], "o_weight": args[1],
                    "o_weight_scale": args[2],
                    "o_token_scale": kwargs["pertoken_scale"],
                    "o_quant_matmul_output": result,
                })
                bias = kwargs.get("bias")
                if bias is not None:
                    self.vendor["o_bias"] = bias
                offset = getattr(self.impl.o_proj, "weight_offset", None)
                if offset is not None:
                    self.vendor["o_weight_offset"] = offset
                self.vendor_o_descriptor = {
                    "positional_arguments": [describe_argument(value)
                                             for value in args],
                    "keyword_arguments": {name: describe_argument(value)
                                          for name, value in kwargs.items()},
                    "weight_shape": list(args[1].shape),
                    "weight_dtype": str(args[1].dtype),
                    "weight_format": self.torch_npu.get_npu_format(args[1]),
                    "scale_shape": list(args[2].shape),
                    "scale_dtype": str(args[2].dtype),
                    "scale_format": self.torch_npu.get_npu_format(args[2]),
                    "bias": None if bias is None else {
                        "shape": list(bias.shape), "dtype": str(bias.dtype)},
                    "offset_passed_to_quant_matmul": False,
                    "output_dtype": str(kwargs.get("output_dtype")),
                }
            return result

        def tapped_native(*args, **kwargs):
            result = native_body(*args, **kwargs)
            hidden, main, front, tail, caches, metadata, index, pool = args
            geometry = self.adapter.geometry_for(kwargs["mode"], caches,
                                                 metadata, index)
            entry = pool.get(hidden, geometry)
            self.native_calls += 1
            self.native = {"hidden": hidden, "scratch": entry[0],
                           "output": result[0], "output_weight_tiles": tail[1],
                           "value_weight_tiles": tail[0],
                           "o_weight_scale": tail[2], "blocks": entry[2]}
            self.native_geometry = geometry
            return result

        with ExitStack() as stack:
            for owner, name, function in (
                    (self.impl, "_sfa_preprocess_prolog_v3", tapped_prolog),
                    (self.impl, "_execute_sparse_flash_attention_process", tapped_sfa),
                    (self.impl, "_v_up_proj", tapped_value_up),
                    (self.torch_npu, "npu_quant_matmul", tapped_quant_matmul),
                    (self.adapter, "run_body", tapped_native)):
                stack.enter_context(replace_attribute(owner, name, function))
            yield

    def read_native(self):
        torch = self.torch
        hidden = self.native["hidden"]
        geometry = self.native_geometry
        codes = torch.empty_like(hidden, dtype=torch.int8)
        temporary, tiling, offsets, blocks, sync = (
            torch.ops.tilert.indexed_attention_prepare(
                codes, geometry.pages, geometry.table_pages,
                geometry.index_pages, geometry.index_table_pages,
                geometry.mode == "int8", self.native["blocks"]))
        if temporary.nbytes != self.native["scratch"].nbytes:
            raise RuntimeError("The returned layout differs from the executed pool entry")
        del temporary, tiling, codes, sync
        rows = hidden.shape[0]
        values = decode_native(torch, self.native["scratch"], offsets, rows)
        values["output"] = cpu_copy(torch, self.native["output"])
        values["hidden"] = cpu_copy(torch, hidden)
        value_weight = cpu_copy(torch, self.native["value_weight_tiles"])
        values["value_weight"] = value_weight.reshape(4, 4, 512, 64).permute(
            0, 2, 1, 3).reshape(4, 512, 256).clone()
        weight = cpu_copy(torch, self.native["output_weight_tiles"])
        values["o_weight"] = weight.permute(1, 0, 2).reshape(1024, 6144).clone()
        values["o_weight_scale"] = cpu_copy(torch, self.native["o_weight_scale"])
        return values, dict(offsets), blocks


def output_oracles(torch, vendor, native):
    """Compute integer dots independently on CPU; retain both FP32 scale orders."""
    result = {}
    bounds = {}
    for name, values in (("vendor", vendor), ("native", native)):
        accumulators = values["o_codes"].long() @ values["o_weight"].long()
        low, high = int(accumulators.min()), int(accumulators.max())
        if low < -(1 << 31) or high >= 1 << 31:
            raise RuntimeError("The independent O accumulator overflows INT32")
        bounds[name] = {"minimum": low, "maximum": high}
        result[f"{name}_i32_oracle"] = accumulators.int()
        bias = vendor.get("o_bias")
        if bias is not None and bias.dtype == torch.int32:
            accumulators = accumulators + bias.long()
        floating = accumulators.float()
        channel = values["o_weight_scale"].float().reshape(1, -1)
        token = values["o_token_scale"].float().reshape(-1, 1)
        channel_then_token = (floating * channel) * token
        combined_scales = floating * (channel * token)
        if bias is not None and bias.dtype.is_floating_point:
            channel_then_token = channel_then_token + bias.float()
            combined_scales = combined_scales + bias.float()
        result[f"{name}_channel_then_token"] = channel_then_token.bfloat16()
        result[f"{name}_combined_scales"] = combined_scales.bfloat16()
    return result, bounds


def normalize_vendor_shapes(vendor, rows):
    shapes = {
        "attention_query": (rows, 4, 512),
        "attention_rope": (rows, 4, 64),
        "query_norm": (rows, 2048),
        "query_norm_scale": (rows,),
        "latent": (rows, 4, 512),
        "value_up": (rows, 1024),
        "o_codes": (rows, 1024),
        "o_token_scale": (rows,),
        "o_weight_scale": (6144,),
        "o_quant_matmul_output": (rows, 6144),
    }
    for name, shape in shapes.items():
        vendor[name] = vendor[name].reshape(shape)


def save_diagnostic(torch, taps, args, outer_error):
    from vendor_hook_forward import source_identity

    torch.npu.synchronize()
    vendor = {name: cpu_copy(torch, value) for name, value in taps.vendor.items()}
    vendor_shapes = {name: list(value.shape) for name, value in vendor.items()}
    native, offsets, blocks = taps.read_native()
    normalize_vendor_shapes(vendor, native["output"].shape[0])
    oracles, accumulator_bounds = output_oracles(torch, vendor, native)
    names = ("attention_query", "attention_rope", "query_norm", "query_norm_scale",
             "latent", "value_up", "o_codes", "o_token_scale")
    checks = {name: compare(torch, native[name], vendor[name]) for name in names}
    checks["o_weight"] = compare(torch, native["o_weight"], vendor["o_weight"])
    checks["value_weight"] = compare(torch, native["value_weight"], vendor["value_weight"])
    checks["o_weight_scale_f32"] = compare(
        torch, native["o_weight_scale"], vendor["o_weight_scale"].float())
    checks["o_accumulators_native_oracle"] = compare(
        torch, native["o_accumulators"], oracles["native_i32_oracle"])
    checks["o_accumulators_vendor_oracle"] = compare(
        torch, native["o_accumulators"], oracles["vendor_i32_oracle"])
    checks["output"] = compare(torch, native["output"], vendor["o_quant_matmul_output"])
    reconstruction = {}
    for name, value in oracles.items():
        if not name.endswith("i32_oracle"):
            target = native["output"] if name.startswith("native") else vendor["o_quant_matmul_output"]
            reconstruction[name] = compare(torch, value, target)
    outer = json.loads((args.output / "outer-forward.json").read_text())
    case = outer["cases"][-1]
    dump = args.output / "outer-forward-diagnostic.pt"
    torch.save({"vendor": vendor, "native": native, "oracles": oracles,
                "offsets": offsets, "case": case}, dump)
    production_order = (*names, "o_accumulators_vendor_oracle", "output")
    quantization = taps.impl.o_proj.quant_method
    quantization_classes = []
    while quantization is not None:
        quantization_classes.append(source_identity(type(quantization)))
        quantization = getattr(quantization, "quant_method", None)
    report = {
        "scope": "Actual eager outer calls with retained stage references; no arithmetic substitution",
        "script": source_identity(save_diagnostic),
        "sources_before_taps": taps.original_sources,
        "reference_lifetime_change": "Taps extend tensor lifetimes and may change allocator addresses",
        "sampling": "Latest complete vendor/native call; all CPU reads occur after outer-forward synchronization",
        "case": {"rows": case["rows"], "iteration": case["iteration"],
                 "original_final_gate_pass": case["pass"]},
        "calls": {"vendor_prolog": taps.vendor_calls, "native_body": taps.native_calls},
        "blocks": blocks, "outer_error": outer_error,
        "first_unequal_production_boundary": next(
            (name for name in production_order if not checks[name]["bitwise_equal"]), None),
        "checks": checks, "cpu_scale_order_reconstructions": reconstruction,
        "integer_oracle": {"expression": "INT64 CPU matrix product of actual INT8 codes and logical weights, checked before INT32 conversion",
                           "accumulator_bounds": accumulator_bounds},
        "vendor_o_descriptor": taps.vendor_o_descriptor,
        "vendor_original_shapes": vendor_shapes,
        "shape_normalization": "CPU views only: logical [M,H,D] tensors and one scale per token/channel",
        "o_projection_class": source_identity(type(taps.impl.o_proj)),
        "o_quantization_classes": quantization_classes,
        "quant_matmul_schema": str(torch.ops.npu.npu_quant_matmul.default._schema),
        "vendor_tensors": {name: tensor_identity(torch, value)
                           for name, value in vendor.items()},
        "native_tensors": {name: tensor_identity(torch, value)
                           for name, value in native.items()},
        "dump": {"path": str(dump), "sha256": hashlib.sha256(dump.read_bytes()).hexdigest()},
    }
    destination = args.output / "outer-forward-diagnostic.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"outer_forward_diagnostic": report}), flush=True)


def run_outer_diagnostic(torch, layer, config, adapter, original_forward, args):
    import torch_npu
    from vendor_hook_forward import run_outer_forward

    taps = ProductionTaps(torch, torch_npu, layer.mla_attn.mla_attn.impl, adapter)
    try:
        with taps.installed():
            result = run_outer_forward(torch, layer, config, adapter,
                                       original_forward, args)
    except Exception:
        outer_error = traceback.format_exc()
        try:
            save_diagnostic(torch, taps, args, outer_error)
        except Exception:
            error = {"outer_error": outer_error,
                     "diagnostic_error": traceback.format_exc()}
            (args.output / "outer-forward-diagnostic-error.json").write_text(
                json.dumps(error, indent=2) + "\n")
            print(json.dumps(error), flush=True)
        raise
    save_diagnostic(torch, taps, args, None)
    return result
