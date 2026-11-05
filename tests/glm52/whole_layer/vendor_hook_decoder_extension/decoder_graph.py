"""Replay two actual decoder layers through the official ACL graph wrapper."""

import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import traceback

from decoder_forward import bind_caches, decoder_cuts, metadata_for_layers
from vendor_hook_forward import (
    count_native_calls, load_sfa_adapter, page_plan, snapshot, source_identity,
)
from vendor_hook_graph import (
    METADATA_FIELDS, copy_fields, signature, tensor_witness, validate_page_plan,
)


class DecoderInputs:
    """Own all graph producers, including both external post-AR1 tensors."""

    def __init__(self, torch, torch_npu, layers, rows, with_residual, caches):
        self.torch = torch
        self.torch_npu = torch_npu
        self.layers = layers
        self.rows = rows
        self.device = caches["layer6.main_cache"].device
        self.with_residual = with_residual
        self.positions, self.metadata = metadata_for_layers(
            torch, torch_npu, layers, self.plan(0), self.device)

        def empty():
            return torch_npu.npu_format_cast(torch.empty(
                rows, 6144, dtype=torch.bfloat16, device=self.device), 2)

        self.hidden = empty()
        self.residual = empty() if with_residual else None
        self.post_ar1 = [empty(), empty()]
        self.main = [self.metadata[layer.self_attn.mla_attn.mla_attn.layer_name]
                     for layer in layers]
        prefix = layers[0].self_attn.mla_attn.mla_attn.impl.indexer.k_cache.prefix
        self.index = self.metadata[prefix]
        self.slots_i64 = []
        for item in self.main:
            item.max_seq_len = 8192
            slots = torch_npu.npu_format_cast(torch.empty(
                rows, dtype=torch.int64, device=self.device), 2)
            item.kv_slots_i64 = (item.slot_mapping, slots)
            self.slots_i64.append(slots)
        sources = [item.slot_mapping for item in (*self.main, self.index)]
        if len({value.data_ptr() for value in sources}) != 3:
            raise RuntimeError("Each main/indexer slot producer needs independent storage")
        self.signatures = None

    def plan(self, iteration):
        plan = page_plan(self.torch, self.rows, iteration, 8192, 66, 66)
        validate_page_plan(self.torch, plan, self.rows)
        return plan

    def metadata_fields(self, positions=None, metadata=None):
        positions = self.positions if positions is None else positions
        metadata = self.metadata if metadata is None else metadata
        fields = {"positions": positions}
        for name, item in metadata.items():
            for field in METADATA_FIELDS:
                value = getattr(item, field, None)
                if isinstance(value, self.torch.Tensor):
                    fields[f"{name}:{field}"] = value
        return fields

    def fields(self):
        values = {**self.metadata_fields(), "hidden": self.hidden}
        if self.residual is not None:
            values["residual"] = self.residual
        for index, value in enumerate(self.post_ar1):
            values[f"post_ar1:{index}"] = value
        for index, value in enumerate(self.slots_i64):
            values[f"slots_i64:{index}"] = value
        return values

    def update(self, iteration):
        torch = self.torch
        plan = self.plan(iteration)
        positions, metadata = metadata_for_layers(
            torch, self.torch_npu, self.layers, plan, self.device)
        copy_fields(self.metadata_fields(), self.metadata_fields(positions, metadata))
        generator = torch.Generator().manual_seed(42023 + self.rows * 100 + iteration)
        self.hidden.copy_(torch.randn(self.rows, 6144, generator=generator).bfloat16())
        if self.residual is not None:
            self.residual.copy_(torch.randn(self.rows, 6144, generator=generator).bfloat16())
        for boundary in self.post_ar1:
            boundary.copy_(torch.randn(self.rows, 6144, generator=generator).bfloat16())
        for item, destination in zip(self.main, self.slots_i64):
            destination.copy_(item.slot_mapping)
            item.max_seq_len = 8192
        self.check_addresses()
        return plan

    def lock_addresses(self):
        self.signatures = {name: signature(value) for name, value in self.fields().items()}

    def check_addresses(self):
        if self.signatures is None:
            return
        if self.signatures != {name: signature(value) for name, value in self.fields().items()}:
            raise RuntimeError("A captured decoder producer changed address or geometry")
        for item, slots in zip(self.main, self.slots_i64):
            if item.kv_slots_i64[0] is not item.slot_mapping or item.kv_slots_i64[1] is not slots:
                raise RuntimeError("The official int64 slot cast lost its persistent owners")

    def witness(self):
        torch = self.torch
        for item, slots in zip(self.main, self.slots_i64):
            if not torch.equal(slots.cpu(), item.slot_mapping.cpu().long()):
                raise RuntimeError("The runner's main int64 slots are stale")
        return {
            "initial_residual": "provided" if self.with_residual else "None",
            "actual_main_lengths": [item.seq_lens.cpu().tolist() for item in self.main],
            "host_main_lengths": [item.seq_lens_cpu.tolist() for item in self.main],
            "actual_index_length": self.index.actual_seq_lengths_key.cpu().tolist(),
            "host_max_seq_len_bounds": [item.max_seq_len for item in self.main],
            "tensors": {name: tensor_witness(torch, value) for name, value in self.fields().items()},
        }


class DecoderObservation:
    """Retain device-side boundary clones without reading tensors during capture."""

    def __init__(self, torch, layers, group, topk, post_ar1):
        self.torch = torch
        self.layers = layers
        self.group = group
        self.topk = topk
        self.post_ar1 = post_ar1
        self.calls = 0
        self.values = {}
        self.traces = {}
        self.output = None
        self.capture_stream = None

    def __call__(self, positions, hidden, residual):
        self.calls += 1
        self.values = {}
        self.traces = {}
        hidden = hidden.clone()
        residual = None if residual is None else residual.clone()
        for number, layer, boundary in zip((6, 7), self.layers, self.post_ar1):
            norms = {}

            def remember_norm(module, unused_inputs, outputs):
                label = "input_norm" if module is layer.input_layernorm else "post_norm"
                if label in norms:
                    raise RuntimeError("A decoder norm ran more than once")
                values = outputs if isinstance(outputs, tuple) else (outputs,)
                norms[label] = (values, tuple(value.clone() for value in values))

            handles = [norm.register_forward_hook(remember_norm)
                       for norm in (layer.input_layernorm, layer.post_attention_layernorm)]
            try:
                with decoder_cuts(self.torch, self.group, layer, boundary) as cuts:
                    hidden, residual = layer(positions, hidden, residual)
                post_original, _ = norms["post_norm"]
                if len(post_original) != 2 or residual is not post_original[1]:
                    raise RuntimeError("The decoder lost its post-attention residual owner")
                for ordinal, value in enumerate(cuts.values, 1):
                    self.values[f"layer{number}.pre_ar{ordinal}"] = value
                self.values[f"layer{number}.post_ar1"] = cuts.attention_output
                for label, (_, values) in norms.items():
                    for index, value in enumerate(values):
                        self.values[f"layer{number}.{label}.{index}"] = value
                self.values[f"layer{number}.output"] = hidden.clone()
                self.values[f"layer{number}.residual"] = residual.clone()
                self.values[f"layer{number}.topk"] = self.topk.clone()
                self.traces[f"layer{number}"] = {
                    "tp_cuts": cuts.trace, "residual_is_post_norm_owner": True,
                }
            finally:
                for handle in handles:
                    handle.remove()
        self.output = hidden, residual
        return self.output

    def poison(self):
        """Require every observed boundary to be freshly written by replay."""
        for value in self.values.values():
            if value.is_floating_point():
                value.fill_(float("nan"))
            else:
                value.fill_(-2147483648)


def validate_observation(torch, values, boundaries, result):
    """Check TP cuts and ownership semantics after device completion."""
    def equal(left, right):
        return (left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left.contiguous().view(torch.uint8),
                                right.contiguous().view(torch.uint8)))

    for number, boundary in zip((6, 7), boundaries):
        if not equal(values[f"layer{number}.post_ar1"], boundary):
            raise RuntimeError("The attention cut did not consume its live external input")
        if not equal(values[f"layer{number}.output"], values[f"layer{number}.pre_ar2"]):
            raise RuntimeError("The decoder changed its local partial after the FFN cut")
        if not equal(values[f"layer{number}.residual"], values[f"layer{number}.post_norm.1"]):
            raise RuntimeError("The retained residual differs from post-attention norm")
    if not equal(values["layer6.topk"], values["layer7.topk"]):
        raise RuntimeError("The shared-indexer successor changed its producer's top-k")
    for label, value in zip(("output", "residual"), result):
        if not equal(values[f"layer7.{label}"], value):
            raise RuntimeError("The graph returned a different final output owner value")
    if any(not bool(torch.isfinite(value).all()) for value in values.values()
           if value.is_floating_point()):
        raise RuntimeError("A decoder boundary was not freshly written with finite values")


def run_decoder_graph(torch, layers, config, state, args):
    import torch_npu
    from attention_check import metric
    from vllm.compilation import monitor
    from vllm.compilation.counter import compilation_counter
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor, get_forward_context
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.attention.sfa_v1 import AscendSFABackend
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper, get_graph_params, set_graph_params
    from vllm_ascend.compilation.updatable_graph import UpdatableGraph
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    from vllm_ascend.utils import use_updatable_graph
    from vllm_ascend.worker.model_runner_v1 import graph_capture

    if args.forward_sequence != 8192 or len(set(args.forward_widths)) != len(args.forward_widths):
        raise ValueError("Use unique widths with a fixed 8192 sequence upper bound")
    if len(layers) != 2 or use_updatable_graph(AscendSFABackend):
        raise RuntimeError("This gate requires two layers and the installed direct SFA replay branch")
    init_device_properties_triton()
    topk = layers[0].self_attn.mla_attn.topk_indices_buffer
    if topk.device.type != "npu" or topk.dtype != torch.int32 or tuple(topk.shape) != (32, 2048):
        raise RuntimeError("The actual layers require a persistent NPU top-k owner")
    for layer in layers:
        wrapper = layer.self_attn.mla_attn
        if wrapper.topk_indices_buffer is not topk or wrapper.mla_attn.impl.topk_indices_buffer is not topk:
            raise RuntimeError("Both layers must retain the same top-k storage")
    caches, cache_owners = bind_caches(torch, torch_npu, layers, config, args.mode)
    bases = {}
    generator = torch.Generator().manual_seed(42022)
    for name, cache in caches.items():
        if cache.dtype == torch.int8:
            value = torch.randint(-127, 128, cache.shape, generator=generator, dtype=torch.int8)
        elif name.endswith("index_scale"):
            value = (torch.rand(cache.shape, generator=generator) / 128 + 1 / 64).to(cache.dtype)
        else:
            value = torch.randn(cache.shape, generator=generator).to(cache.dtype)
        bases[name] = torch_npu.npu_format_cast(value.to("npu"), 2)
    device = topk.device
    os.environ["TILERT_INDEXED_ATTENTION_LIBRARY"] = str(args.indexed_library)
    os.environ["TILERT_SHARED_ATTENTION_LIBRARY"] = str(args.shared_library)
    sys.path.insert(0, str(args.adapter.resolve().parents[3]))
    adapter = importlib.import_module("tilert.models.glm_5_2_ascend.attention_native")
    if Path(adapter.__file__).resolve() != args.adapter.resolve():
        raise RuntimeError("The native attention adapter differs from the frozen source")
    sfa = load_sfa_adapter(torch, args.sfa_library)
    adapter.load()
    pool = adapter.WorkspacePool(blocks=24)
    implementations = [layer.self_attn.mla_attn.mla_attn.impl for layer in layers]
    originals = [impl.forward for impl in implementations]
    restores = [adapter.install(layer.self_attn.mla_attn.mla_attn, pool) for layer in layers]
    installed = [impl.forward for impl in implementations]
    report = {
        "scope": "Official ACLGraphWrapper around original DecoderLayer 6 then 7; original FFN in both arms; native attention in one arm",
        "script": source_identity(run_decoder_graph),
        "decoder_forward": source_identity(type(layers[0]).forward),
        "forward_context": source_identity(inspect.unwrap(set_ascend_forward_context)),
        "graph_wrapper": source_identity(ACLGraphWrapper),
        "graph_capture_context": source_identity(inspect.unwrap(graph_capture)),
        "adapter": source_identity(adapter.run_body), "sfa_adapter": sfa,
        "native_libraries": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                             for path in (args.indexed_library, args.shared_library)],
        "mode": args.mode, "widths": args.forward_widths,
        "iterations": args.forward_replays, "residual_modes": args.graph_residual_modes,
        "tp_geometry": 16, "tp_collectives_executed": 0,
        "ar1_contract": "Independent live BF16 inputs at both attention reductions",
        "ar2_contract": "Original rank-local FFN partial; no multi-rank sum",
        "successor_contract": "Layer7 consumes layer6 pre-AR2 output and its separately returned residual",
        "host_sequence_bound": 8192, "actual_length_rule": "8192 - iteration % 11",
        "page_table_columns": 64, "updatable_graph_branch": False,
        "initial_cache_rule": "base.roll(iteration, page dimension), restored before every arm",
        "original_ffn_disposal": "Unmodified; no GMM hooks or diagnostic synchronization",
        "capture_observation": "Strong device clones; CPU snapshots only after explicit replay",
        "configuration_enforce_eager": config.model_config.enforce_eager,
        "captures": [], "cases": [], "pass": False,
    }
    destination = args.output / "decoder-graph.json"
    arms = []

    def persist():
        destination.write_text(json.dumps(report, indent=2) + "\n")

    def weight_owners():
        result = {}
        for number, layer in zip((6, 7), layers):
            routed = layer.mlp.experts.routed_experts
            for name in ("w13_weight", "w2_weight"):
                descriptor = dict(torch.ops.tilert_nz_storage.describe(getattr(routed, name)))
                descriptor.pop("tensor_impl_use_count")
                result[f"layer{number}.{name}"] = descriptor
        return result

    owners = weight_owners()
    cache_signatures = {name: signature(value) for name, value in caches.items()}
    topk_signature = signature(topk)
    report.update(weight_owners=owners, topk_signature=topk_signature,
                  cache_signatures=cache_signatures)

    def check_owners():
        if owners != weight_owners() or signature(topk) != topk_signature:
            raise RuntimeError("A captured weight or top-k storage owner changed")
        if cache_signatures != {name: signature(value) for name, value in caches.items()}:
            raise RuntimeError("An actual bound cache changed its storage owner")
        for layer in layers:
            if layer.self_attn.mla_attn.mla_attn.impl.topk_indices_buffer is not topk:
                raise RuntimeError("A shared top-k consumer changed its owner")

    def reset(iteration):
        for name, cache in caches.items():
            cache.copy_(bases[name].roll(iteration % bases[name].shape[0], dims=0))
        topk.fill_(-1)

    def context(inputs, mode):
        return set_ascend_forward_context(
            inputs.metadata, config, num_tokens=inputs.rows, num_actual_tokens=inputs.rows,
            aclgraph_runtime_mode=mode,
            batch_descriptor=BatchDescriptor(inputs.rows, 1, uniform=True),
            model_instance=layers[0])

    def choose(forwards):
        for impl, forward in zip(implementations, forwards):
            impl.forward = forward

    def collect(observation, inputs, result):
        torch.npu.synchronize()
        values = {name: snapshot(torch, value) for name, value in observation.values.items()}
        boundaries = [snapshot(torch, value) for value in inputs.post_ar1]
        outputs = tuple(snapshot(torch, value) for value in result)
        validate_observation(torch, values, boundaries, outputs)
        values.update({name: snapshot(torch, value) for name, value in caches.items()})
        inputs.check_addresses()
        check_owners()
        return values

    def destroy_graphs():
        torch.npu.synchronize()
        for arm in arms:
            for entry in arm["wrapper"].concrete_aclgraph_entries.values():
                if entry.aclgraph is not None:
                    entry.aclgraph.reset()
            arm["wrapper"].concrete_aclgraph_entries.clear()
            arm["observation"].values.clear()
            arm["observation"].output = None
        arms.clear()
        # No surviving graph may refer to a workspace released here.
        for name in ("entries", "capture_reserves", "slot_entries", "slot_reserves"):
            getattr(pool, name).clear()
        pool.captured_slot_sources.clear()

    persist()
    try:
        parameters = get_graph_params()
        if parameters is None:
            set_graph_params(args.forward_widths)
        elif any(rows not in parameters.workspaces for rows in args.forward_widths):
            raise RuntimeError("Existing graph parameters omit a requested width")
        with torch.inference_mode(), count_native_calls(adapter) as hits:
            for rows in args.forward_widths:
                for residual_mode in args.graph_residual_modes:
                    inputs = DecoderInputs(torch, torch_npu, layers, rows,
                                           residual_mode == "provided", caches)
                    inputs.update(0)
                    for forwards in (originals, installed):
                        choose(forwards)
                        for _ in range(2):
                            reset(0)
                            warmup = DecoderObservation(torch, layers, state.get_tp_group(),
                                                        topk, inputs.post_ar1)
                            with context(inputs, CUDAGraphMode.NONE):
                                warmup(inputs.positions, inputs.hidden, inputs.residual)
                            torch.npu.synchronize()
                    inputs.lock_addresses()
                    descriptor = BatchDescriptor(rows, 1, uniform=True)
                    for name, forwards in (("vendor", originals), ("native", installed)):
                        choose(forwards)
                        observation = DecoderObservation(
                            torch, layers, state.get_tp_group(), topk, inputs.post_ar1)

                        def runnable(positions, hidden, residual, observation=observation):
                            if not torch.npu.is_current_stream_capturing():
                                raise RuntimeError("The graph runnable fell back to Python execution")
                            observation.capture_stream = torch.npu.current_stream(device)
                            return observation(positions, hidden, residual)

                        wrapper = ACLGraphWrapper(runnable, config, runtime_mode=CUDAGraphMode.FULL)
                        wrapper.set_attn_backend(AscendSFABackend)
                        arm = {"name": name, "wrapper": wrapper, "observation": observation,
                               "forwards": forwards, "replays": 0}
                        arms.append(arm)
                        reset(0)
                        before_hits = len(hits)
                        counter = compilation_counter.num_cudagraph_captured
                        caller = torch.npu.current_stream(device)
                        record = {"rows": rows, "initial_residual": residual_mode,
                                  "arm": name, "phase": "capture"}
                        report["captures"].append(record)
                        persist()
                        with context(inputs, CUDAGraphMode.FULL):
                            reserves, slots = {}, {}
                            if name == "native":
                                pool.reserve_capture(inputs.hidden)
                                reserves = dict(pool.capture_reserves)
                                slots = dict(pool.slot_reserves)
                                if len(reserves) != 2 or len(slots) != 3:
                                    raise RuntimeError("Expected two geometries and three independent slot sources")
                            enabled = monitor.cudagraph_capturing_enabled
                            monitor.set_cudagraph_capturing_enabled(True)
                            try:
                                with graph_capture(device) as capture:
                                    output = wrapper(inputs.positions, inputs.hidden, inputs.residual)
                                caller.wait_stream(observation.capture_stream)
                            finally:
                                monitor.set_cudagraph_capturing_enabled(enabled)
                            get_forward_context().capturing = False
                        entries = wrapper.concrete_aclgraph_entries
                        entry = entries.get(descriptor)
                        if len(entries) != 1 or entry is None or not isinstance(entry.aclgraph, UpdatableGraph):
                            raise RuntimeError("The official wrapper did not capture one real graph")
                        if compilation_counter.num_cudagraph_captured != counter + 1 or observation.calls != 1:
                            raise RuntimeError("The wrapper did not enter the real decoder exactly once")
                        expected_modes = [] if name == "vendor" else [args.mode, "shared"]
                        if [value["mode"] for value in hits[before_hits:]] != expected_modes:
                            raise RuntimeError("Capture entered the wrong attention implementation")
                        output_ptrs = [value.data_ptr() for value in observation.output]
                        if ([value.data_ptr() for value in output] != output_ptrs
                                or [value.data_ptr() for value in entry.output] != output_ptrs):
                            raise RuntimeError("The graph lost its strong final output owners")
                        stream = observation.capture_stream.npu_stream
                        if stream == caller.npu_stream:
                            raise RuntimeError("Capture unexpectedly reused the warmup stream")
                        claimed = None
                        if name == "native":
                            actual = [value for key, value in pool.entries.items()
                                      if key[1] == stream and key[2] == rows]
                            claimed = (len(actual) == 2 and all(
                                any(value is candidate for candidate in actual)
                                for value in reserves.values()))
                            if (not claimed or any(key in pool.capture_reserves for key in reserves)
                                    or any(key in pool.slot_reserves for key in slots)):
                                raise RuntimeError("Capture did not claim both workspace and slot reserves")
                            if any(pool.slot_entries.get((stream, *key)) is not value
                                   for key, value in slots.items()):
                                raise RuntimeError("Capture substituted a reserved slot destination")
                        record.update(
                            phase="captured_not_executed", runnable_calls=observation.calls,
                            native_python_modes=expected_modes, capture_counter_delta=1,
                            capture_stream=stream, caller_stream=caller.npu_stream,
                            outer_capture_stream=capture.stream.npu_stream,
                            workspace_reserve_claimed=claimed, output_owner_ptrs=output_ptrs,
                            observed_boundary_count=len(observation.values),
                            graph_type=type(entry.aclgraph).__name__, graph_tasks=len(entry.aclgraph.tasks),
                            input_signatures=inputs.signatures, tp_cuts=observation.traces)
                        persist()
                    for iteration in range(args.forward_replays):
                        plan = inputs.update(iteration)
                        case = {"rows": rows, "initial_residual": residual_mode,
                                "iteration": iteration, "phase": "eager_oracle",
                                "inputs": inputs.witness(), "arms": {}, "pass": False}
                        report["cases"].append(case)
                        persist()
                        choose(originals)
                        reset(iteration)
                        oracle = DecoderObservation(torch, layers, state.get_tp_group(),
                                                    topk, inputs.post_ar1)
                        before_hits = len(hits)
                        with context(inputs, CUDAGraphMode.NONE):
                            result = oracle(inputs.positions, inputs.hidden, inputs.residual)
                        expected = collect(oracle, inputs, result)
                        if len(hits) != before_hits:
                            raise RuntimeError("The eager oracle entered native attention")
                        case["eager_output"] = tensor_witness(torch, expected["layer7.output"])
                        case["eager_attention"] = {
                            f"layer{number}": tensor_witness(
                                torch, expected[f"layer{number}.pre_ar1"])
                            for number in (6, 7)
                        }
                        for arm in arms:
                            case["phase"] = arm["name"] + "_graph_replay"
                            persist()
                            choose(arm["forwards"])
                            reset(iteration)
                            observation = arm["observation"]
                            observation.poison()
                            before_hits, calls = len(hits), observation.calls
                            counter = compilation_counter.num_cudagraph_captured
                            with context(inputs, CUDAGraphMode.FULL):
                                result = arm["wrapper"](inputs.positions, inputs.hidden, inputs.residual)
                            if (len(hits) != before_hits or observation.calls != calls
                                    or compilation_counter.num_cudagraph_captured != counter):
                                raise RuntimeError("Replay ran Python or captured another graph")
                            if [value.data_ptr() for value in result] != [
                                    value.data_ptr() for value in observation.output]:
                                raise RuntimeError("Replay replaced a strongly owned output")
                            actual = collect(observation, inputs, result)
                            arm["replays"] += 1
                            checks = {name: metric(torch, actual[name], value)
                                      for name, value in expected.items()}
                            passed = all(value["pass"] and value["bitwise_equal"] for value in checks.values())
                            case["arms"][arm["name"]] = {
                                "pass": passed, "checks": checks, "replay_count": arm["replays"],
                                "python_calls_during_replay": 0, "native_python_calls_during_replay": 0,
                                "output": tensor_witness(torch, actual["layer7.output"]),
                            }
                            persist()
                            if not passed:
                                path = args.output / (f"decoder-graph-{args.mode}-m{rows}-"
                                                      f"{residual_mode}-r{iteration}-{arm['name']}.pt")
                                torch.save({"plan": plan, "expected": expected, "actual": actual}, path)
                                case["failure_dump"] = str(path)
                                raise RuntimeError("The complete decoder graph differs from its eager oracle")
                        case["phase"] = "comparison"
                        case["pass"] = True
                        persist()
                        print(json.dumps({"decoder_graph_case": case}), flush=True)
                    selected = [value for value in report["cases"]
                                if value["rows"] == rows and value["initial_residual"] == residual_mode]
                    if args.forward_replays > 1:
                        hashes = {value["eager_output"]["sha256"] for value in selected}
                        lengths = {tuple(value["inputs"]["actual_main_lengths"][0]) for value in selected}
                        if len(hashes) < 2 or len(lengths) < 2:
                            raise RuntimeError("Changing graph inputs lack observable output/length witnesses")
                        for number in (6, 7):
                            hashes = {value["eager_attention"][f"layer{number}"]["sha256"]
                                      for value in selected}
                            if len(hashes) < 2:
                                raise RuntimeError("The attention boundary did not change with live inputs")
                    if any(value["eager_output"]["nonzero_elements"] == 0 for value in selected):
                        raise RuntimeError("The decoder output witness is entirely zero")
                    if any(witness["nonzero_elements"] == 0 for value in selected
                           for witness in value["eager_attention"].values()):
                        raise RuntimeError("An attention boundary witness is entirely zero")
                    destroy_graphs()
        report["pass"] = True
        persist()
        return report
    except Exception:
        report["error"] = traceback.format_exc()
        persist()
        raise
    finally:
        destroy_graphs()
        for restore in reversed(restores):
            restore()
