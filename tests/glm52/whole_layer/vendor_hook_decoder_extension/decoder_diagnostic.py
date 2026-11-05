"""Locate the original decoder failure without changing operator arguments."""

from contextlib import contextmanager
import functools
import hashlib
import inspect
import json
from pathlib import Path
import time
import traceback
import weakref


def source_identity(value):
    source = Path(inspect.getsourcefile(value)).resolve()
    return {"file": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


def tensor_descriptor(torch, torch_npu, value, owners):
    storage = value.untyped_storage()
    result = {
        "python_id": id(value), "tensor_impl": value._cdata,
        "shape": list(value.shape), "stride": list(value.stride()),
        "dtype": str(value.dtype), "device": str(value.device),
        "storage_offset": value.storage_offset(), "data_ptr": value.data_ptr(),
        "storage_ptr": storage.data_ptr(), "storage_bytes": storage.nbytes(),
        "logical_bytes": value.numel() * value.element_size(),
        "is_contiguous": value.is_contiguous(),
        "parameter_owner_names": owners.get(id(value), []),
        "base_python_id": None if value._base is None else id(value._base),
    }
    if value.device.type == "npu":
        result["format"] = int(torch_npu.get_npu_format(value))
        result["physical_storage_elements"] = int(torch_npu.get_storage_size(value))
        result["serialization_metadata"] = torch._C._get_tensor_metadata(value)
    return result


def describe_tree(torch, torch_npu, value, owners):
    if isinstance(value, torch.Tensor):
        return tensor_descriptor(torch, torch_npu, value, owners)
    if isinstance(value, (list, tuple)):
        return [describe_tree(torch, torch_npu, item, owners) for item in value]
    if isinstance(value, dict):
        return {key: describe_tree(torch, torch_npu, item, owners)
                for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"python_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def count_contract(values, group_list_type, rows):
    if group_list_type == 1:
        counts = values
        semantics = "Per-expert row counts"
    elif group_list_type == 0:
        counts = [end - start for start, end in zip([0] + values[:-1], values)]
        semantics = "Cumulative row ends; differenced for the reported bounds"
    else:
        return {"values": values, "group_list_type": group_list_type,
                "semantics": "Unsupported by this diagnostic", "valid": False}
    ends, total = [], 0
    for count in counts:
        total += count
        ends.append(total)
    return {
        "values": values, "group_list_type": group_list_type,
        "semantics": semantics, "counts": counts, "cumulative_ends": ends,
        "rows": rows, "sum_counts": total,
        "active_experts": [index for index, count in enumerate(counts) if count],
        "nonnegative": all(count >= 0 for count in counts),
        "all_row_ends_in_bounds": all(0 <= end <= rows for end in ends),
        "valid": all(count >= 0 for count in counts) and total == rows,
    }


class DiagnosticTrace:
    """Persist submission boundaries before any operation that can fail."""

    def __init__(self, torch, torch_npu, layers, output, synchronize, current_stream):
        self.torch = torch
        self.torch_npu = torch_npu
        self.output = output
        self.synchronize = synchronize
        self.current_stream = current_stream
        self.events = []
        self.calls = 0
        self.observed = []
        self.owners = {}
        self.phase = []
        for number, layer in zip((6, 7), layers):
            for name, value in layer.named_parameters():
                self.owners.setdefault(id(value), []).append(f"layer{number}.{name}")
        self.report = {
            "scope": "Original decoder/operator bodies with GMM1 pre/post synchronization and CPU snapshots",
            "intervention": "Synchronization drains queued producers and completes GMM1 before the original dispose_tensor",
            "tensor_lifetime": "Weak references only; no extra NPU views or clones retain input storage",
            "physical_descriptor_scope": "Format and physical element count are direct TorchNPU queries; serialization_metadata may describe ND reconstruction",
            "events": self.events, "completed": False,
        }

    def describe(self, value):
        return describe_tree(self.torch, self.torch_npu, value, self.owners)

    def emit(self, event, **details):
        item = {"ordinal": len(self.events), "event": event,
                "monotonic_ns": time.monotonic_ns(), "phase": list(self.phase),
                **details}
        self.events.append(item)
        with (self.output / "decoder-diagnostic.jsonl").open("a") as handle:
            handle.write(json.dumps(item) + "\n")
        self.persist()
        return item

    def persist(self):
        path = self.output / "decoder-diagnostic.json"
        path.write_text(json.dumps(self.report, indent=2) + "\n")

    def fence(self, name, call):
        self.emit(name + "_begin", call=call)
        try:
            self.synchronize()
        except Exception:
            self.emit(name + "_failed", call=call, error=traceback.format_exc())
            raise
        self.emit(name + "_complete", call=call)

    def snapshot_small_inputs(self, arguments, call):
        values = {}
        for name in ("x", "x_scale", "group_list", "weight_scale",
                     "weight_assist_matrix", "bias", "smooth_scale"):
            value = arguments.get(name)
            tensors = value if isinstance(value, (tuple, list)) else [value]
            for index, tensor in enumerate(tensors):
                if isinstance(tensor, self.torch.Tensor):
                    key = f"{name}.{index}" if isinstance(value, (tuple, list)) else name
                    values[key] = tensor.detach().cpu().clone(
                        memory_format=self.torch.contiguous_format)
        path = self.output / f"gmm1-call-{call:03d}-inputs.pt"
        self.torch.save(values, path)
        return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "keys": list(values), "weight_payload_saved": False}

    def gmm1(self, original, *args, **kwargs):
        schema = original.default._schema
        names = [argument.name for argument in schema.arguments]
        arguments = dict(zip(names, args))
        arguments.update(kwargs)
        call = self.calls
        self.calls += 1
        self.emit("gmm1_enter", call=call, arguments=self.describe(arguments),
                  schema=str(schema), stream=int(self.current_stream()))
        self.fence("gmm1_producers", call)
        group = arguments["group_list"].detach().cpu().tolist()
        contract = count_contract(group, arguments.get("group_list_type", 0),
                                  arguments["x"].shape[0])
        self.observed.append((call, weakref.ref(arguments["x"])))
        self.emit("gmm1_inputs_ready", call=call, group_list=contract,
                  snapshot=self.snapshot_small_inputs(arguments, call),
                  arguments=self.describe(arguments))
        self.emit("gmm1_submit", call=call)
        try:
            result = original(*args, **kwargs)
        except Exception:
            self.emit("gmm1_submit_failed", call=call, error=traceback.format_exc())
            raise
        self.emit("gmm1_submitted", call=call, outputs=self.describe(result))
        self.fence("gmm1_completion", call)
        self.emit("gmm1_complete", call=call,
                  input_after_completion=self.describe(arguments["x"]),
                  outputs=self.describe(result))
        return result

    def dispose(self, original, value):
        matches = [call for call, reference in self.observed if reference() is value]
        self.emit("dispose_enter", gmm1_calls=matches, tensor=self.describe(value))
        result = original(value)
        self.emit("dispose_return", gmm1_calls=matches, tensor=self.describe(value))
        return result

    def enter_module(self, name):
        def hook(unused_module, inputs):
            self.phase.append(name)
            self.emit("module_enter", name=name, inputs=self.describe(inputs),
                      device_completion_checked=False)
        return hook

    def leave_module(self, name):
        def hook(unused_module, unused_inputs, output):
            self.emit("module_return", name=name, outputs=self.describe(output),
                      device_completion_checked=False)
            if not self.phase or self.phase.pop() != name:
                raise RuntimeError("Diagnostic module boundaries are unbalanced")
        return hook


@contextmanager
def observe_original_calls(trace, layers, namespace, w4a8):
    original_gmm1 = namespace.grouped_matmul_swiglu_quant_v2
    original_dispose = w4a8.dispose_tensor
    handles = []
    trace.report["gmm1_schema"] = str(original_gmm1.default._schema)
    trace.report["official_dispose"] = source_identity(original_dispose)
    trace.report["official_w4a8"] = source_identity(w4a8.AscendW4A8DynamicFusedMoEMethod)

    @functools.wraps(original_gmm1)
    def gmm1(*args, **kwargs):
        return trace.gmm1(original_gmm1, *args, **kwargs)

    @functools.wraps(original_dispose)
    def dispose(value):
        return trace.dispose(original_dispose, value)

    namespace.grouped_matmul_swiglu_quant_v2 = gmm1
    w4a8.dispose_tensor = dispose
    try:
        for number, layer in zip((6, 7), layers):
            for name, module in (("decoder", layer),
                                 ("input_norm", layer.input_layernorm),
                                 ("attention", layer.self_attn),
                                 ("post_attention_norm", layer.post_attention_layernorm),
                                 ("ffn", layer.mlp)):
                label = f"layer{number}.{name}"
                handles.append(module.register_forward_pre_hook(trace.enter_module(label)))
                handles.append(module.register_forward_hook(trace.leave_module(label)))
        yield
    finally:
        namespace.grouped_matmul_swiglu_quant_v2 = original_gmm1
        w4a8.dispose_tensor = original_dispose
        for handle in handles:
            handle.remove()


def run_decoder_forward(torch, layers, config, state, args):
    import torch_npu
    from decoder_forward import run_decoder_forward as original_forward
    from vllm_ascend.quantization.methods.w4a8 import w4a8

    trace = DiagnosticTrace(
        torch, torch_npu, layers, args.output, torch.npu.synchronize,
        lambda: torch.npu.current_stream().npu_stream)
    trace.report.update(diagnostic_source=source_identity(run_decoder_forward),
                        original_forward=source_identity(original_forward),
                        torch=torch.__version__, torch_npu=torch_npu.__version__)
    try:
        with observe_original_calls(trace, layers, torch.ops._C_ascend, w4a8):
            result = original_forward(torch, layers, config, state, args)
        if trace.calls == 0:
            raise RuntimeError("The real decoder did not reach the observed original GMM1")
        trace.report["completed"] = True
        return result
    except Exception:
        trace.report["error"] = traceback.format_exc()
        raise
    finally:
        trace.report["gmm1_call_count"] = trace.calls
        trace.report["observed_inputs_alive_at_exit"] = [
            {"call": call, "tensor_object_alive": reference() is not None}
            for call, reference in trace.observed]
        trace.persist()
