"""Exercise the official ACL wrapper around the real outer attention layer."""

import hashlib
import inspect
import json
import traceback

from vendor_hook_forward import (
    CACHE_NAMES,
    count_native_calls,
    initial_cache_values,
    load_sfa_adapter,
    make_metadata,
    page_plan,
    snapshot,
    source_identity,
)


METADATA_FIELDS = (
    "slot_mapping", "seq_lens", "seq_lens_cpu", "cum_query_lens",
    "block_table", "sin", "cos", "positions", "query_start_loc",
    "actual_seq_lengths_query", "actual_seq_lengths_key",
)


def signature(value):
    return (value.data_ptr(), tuple(value.shape), tuple(value.stride()),
            str(value.dtype), str(value.device), value.storage_offset())


def copy_fields(destinations, sources):
    """Reject geometry changes before updating any persistent producer."""
    if destinations.keys() != sources.keys():
        raise ValueError("The persistent metadata field set changed")
    for name, destination in destinations.items():
        source = sources[name]
        if destination.shape != source.shape or destination.dtype != source.dtype:
            raise ValueError(f"The persistent shape or dtype changed: {name}")
    before = {name: signature(value) for name, value in destinations.items()}
    for name, destination in destinations.items():
        destination.copy_(sources[name])
    if before != {name: signature(value) for name, value in destinations.items()}:
        raise RuntimeError("Updating metadata replaced a persistent buffer")


def validate_page_plan(torch, plan, rows, sequence=8192):
    length = int(plan["lengths"].item())
    if sequence != 8192 or not rows <= length <= sequence or length <= 8064:
        raise ValueError("This graph gate retains 64 page-table columns")
    expected_positions = torch.arange(length - rows, length, dtype=torch.int64)
    if not torch.equal(plan["positions"], expected_positions):
        raise ValueError("Positions must describe the actual causal suffix")
    if plan["query_lengths"].tolist() != [rows] or plan["query_start"].tolist() != [0, rows]:
        raise ValueError("Query geometry differs from the fixed width")
    for prefix in ("main", "index"):
        table = plan[prefix + "_table"]
        if tuple(table.shape) != (1, 64):
            raise ValueError("The page-table shape changed")
        expected = table[0, expected_positions // 128].long() * 128
        expected += expected_positions % 128
        if not torch.equal(plan[prefix + "_slots"].long(), expected):
            raise ValueError(f"The {prefix} slots do not match its page table")


def tensor_witness(torch, value):
    value = snapshot(torch, value)
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "nonzero_elements": int(torch.count_nonzero(value)),
            "finite": bool(torch.isfinite(value).all())}


class GraphInputs:
    """Own independent main/indexer metadata and the runner's int64 slot cast."""

    def __init__(self, torch, torch_npu, layer, rows, sequence, caches):
        self.torch = torch
        self.torch_npu = torch_npu
        self.layer = layer
        self.rows = rows
        self.sequence = sequence
        self.caches = caches
        self.device = caches[0].device
        plan = self.plan(0)
        self.positions, self.metadata = make_metadata(
            torch, torch_npu, layer, plan, self.device)
        attention = layer.mla_attn.mla_attn
        self.main = self.metadata[attention.layer_name]
        self.index = self.metadata[attention.impl.indexer.k_cache.prefix]
        self.main.max_seq_len = sequence
        self.hidden = torch_npu.npu_format_cast(torch.empty(
            rows, 6144, dtype=torch.bfloat16, device=self.device), 2)
        self.slots_i64 = torch_npu.npu_format_cast(torch.empty(
            rows, dtype=torch.int64, device=self.device), 2)
        self.main.kv_slots_i64 = (self.main.slot_mapping, self.slots_i64)
        if self.main.slot_mapping.data_ptr() == self.index.slot_mapping.data_ptr():
            raise RuntimeError("Main and indexer slot sources must be independent")
        self.signatures = None

    def plan(self, iteration):
        plan = page_plan(self.torch, self.rows, iteration, self.sequence,
                         self.caches[0].shape[0], self.caches[2].shape[0])
        validate_page_plan(self.torch, plan, self.rows, self.sequence)
        return plan

    def fields(self, positions=None, metadata=None):
        metadata = self.metadata if metadata is None else metadata
        positions = self.positions if positions is None else positions
        result = {"positions": positions}
        for name, item in metadata.items():
            for field in METADATA_FIELDS:
                value = getattr(item, field, None)
                if isinstance(value, self.torch.Tensor):
                    result[f"{name}:{field}"] = value
        return result

    def update(self, iteration):
        torch = self.torch
        plan = self.plan(iteration)
        positions, metadata = make_metadata(
            torch, self.torch_npu, self.layer, plan, self.device)
        copy_fields(self.fields(), self.fields(positions, metadata))
        generator = torch.Generator().manual_seed(41023 + self.rows * 100 + iteration)
        hidden_cpu = torch.randn(self.rows, 6144, generator=generator).bfloat16()
        self.hidden.copy_(hidden_cpu)
        # The official helper caches casts by source object identity. The real
        # caller must refresh this persistent cast outside every graph replay.
        self.slots_i64.copy_(self.main.slot_mapping)
        self.main.max_seq_len = self.sequence
        self.check_addresses()
        return plan

    def lock_addresses(self):
        fields = {**self.fields(), "hidden": self.hidden,
                  "main_slots_i64": self.slots_i64}
        self.signatures = {name: signature(value) for name, value in fields.items()}

    def check_addresses(self):
        if self.signatures is None:
            return
        fields = {**self.fields(), "hidden": self.hidden,
                  "main_slots_i64": self.slots_i64}
        current = {name: signature(value) for name, value in fields.items()}
        if current != self.signatures:
            raise RuntimeError("A captured producer changed its address or geometry")
        if self.main.kv_slots_i64[0] is not self.main.slot_mapping:
            raise RuntimeError("The cached int64 cast lost its live source owner")
        if self.main.kv_slots_i64[1] is not self.slots_i64:
            raise RuntimeError("The cached int64 destination was replaced")

    def witness(self):
        torch = self.torch
        if not torch.equal(self.slots_i64.cpu(), self.main.slot_mapping.cpu().long()):
            raise RuntimeError("The main int64 slot producer is stale")
        return {"host_max_seq_len_bound": self.main.max_seq_len,
                "actual_main_length": self.main.seq_lens.cpu().tolist(),
                "actual_index_length": self.index.actual_seq_lengths_key.cpu().tolist(),
                "host_actual_main_length": self.main.seq_lens_cpu.tolist(),
                "main_slots": self.main.slot_mapping.cpu().tolist(),
                "index_slots": self.index.slot_mapping.cpu().tolist(),
                "main_slots_i64": self.slots_i64.cpu().tolist(),
                "tensors": {name: tensor_witness(torch, value)
                            for name, value in {**self.fields(), "hidden": self.hidden}.items()}}


def run_outer_graph(torch, layer, config, adapter, original_forward, args, pool):
    """Compare official graph replays with the original eager local shard."""
    import torch_npu
    from vllm.compilation import monitor
    from vllm.compilation.counter import compilation_counter
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor, get_forward_context
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.attention.sfa_v1 import AscendSFABackend
    from vllm_ascend.compilation.acl_graph import (
        ACLGraphWrapper, get_graph_params, set_graph_params,
    )
    from vllm_ascend.compilation.updatable_graph import UpdatableGraph
    from vllm_ascend.ops.mla import mla_forward
    from vllm_ascend.utils import use_updatable_graph
    from vllm_ascend.worker.model_runner_v1 import graph_capture

    from attention_check import metric

    attention = layer.mla_attn.mla_attn
    impl = attention.impl
    installed_forward = impl.forward
    if impl.o_proj.reduce_results or impl.skip_topk or impl.indexer is None:
        raise ValueError("The graph gate requires an unreduced full-indexer layer")
    if args.forward_sequence != 8192 or len(set(args.forward_widths)) != len(args.forward_widths):
        raise ValueError("The graph gate uses unique widths and a fixed 8192 upper bound")
    caches = (*attention.kv_cache, *impl.indexer.k_cache.kv_cache)
    device = caches[0].device
    images = initial_cache_values(torch, caches)
    bases = [torch_npu.npu_format_cast(value.to(device), 2) for value in images]
    topk = impl.topk_indices_buffer
    report = {
        "scope": ("Official ACLGraphWrapper around the real outer attention "
                  "local shard; no TP collective"),
        "script": source_identity(run_outer_graph), "mode": args.mode,
        "widths": args.forward_widths, "iterations": args.forward_replays,
        "host_sequence_bound": 8192, "actual_length_rule": "8192 - iteration % 11",
        "page_table_columns": 64,
        "initial_cache_update": "base.roll(iteration, page dimension), restored for each arm",
        "outer_layer": source_identity(type(layer)),
        "mla_custom_op": source_identity(mla_forward),
        "forward_context": source_identity(inspect.unwrap(set_ascend_forward_context)),
        "graph_wrapper": source_identity(ACLGraphWrapper),
        "graph_capture_context": source_identity(inspect.unwrap(graph_capture)),
        "graph_class": source_identity(UpdatableGraph),
        "adapter": source_identity(adapter.run_body),
        "sfa_adapter": load_sfa_adapter(torch, args.sfa_library),
        "updatable_graph_branch": use_updatable_graph(AscendSFABackend),
        "configuration_enforce_eager": config.model_config.enforce_eager,
        "graph_runtime_mode": "FULL explicitly selected for this local-shard wrapper",
        "captures": [], "cases": [], "pass": False,
    }
    if report["updatable_graph_branch"]:
        raise RuntimeError("This source-scoped SFA gate expects direct replay, not task update")
    output_path = args.output / "outer-graph.json"
    arms = []

    def persist():
        output_path.write_text(json.dumps(report, indent=2) + "\n")

    def reset(iteration):
        for cache, base in zip(caches, bases):
            cache.copy_(base.roll(iteration % base.shape[0], dims=0))
        topk.fill_(-1)

    def context(inputs, mode):
        return set_ascend_forward_context(
            inputs.metadata, config, num_tokens=inputs.rows,
            num_actual_tokens=inputs.rows, aclgraph_runtime_mode=mode,
            batch_descriptor=BatchDescriptor(inputs.rows, 1, uniform=True),
            model_instance=layer)

    def state(result):
        torch.npu.synchronize()
        values = {"output": snapshot(torch, result), "topk": snapshot(torch, topk)}
        values.update({name: snapshot(torch, value)
                       for name, value in zip(CACHE_NAMES, caches)})
        return values

    def destroy_graphs():
        torch.npu.synchronize()
        for arm in arms:
            for entry in arm["wrapper"].concrete_aclgraph_entries.values():
                if entry.aclgraph is not None:
                    entry.aclgraph.reset()
            arm["wrapper"].concrete_aclgraph_entries.clear()
            arm["owner"].clear()
        arms.clear()

    persist()
    try:
        parameters = get_graph_params()
        if parameters is None:
            set_graph_params(args.forward_widths)
        elif any(rows not in parameters.workspaces for rows in args.forward_widths):
            raise RuntimeError("Existing graph parameters omit a requested width")
        with torch.inference_mode(), count_native_calls(adapter) as hits:
            for rows in args.forward_widths:
                inputs = GraphInputs(torch, torch_npu, layer, rows, 8192, caches)
                inputs.update(0)
                for forward in (original_forward, installed_forward):
                    impl.forward = forward
                    for _ in range(2):
                        reset(0)
                        with context(inputs, CUDAGraphMode.NONE):
                            layer(inputs.positions, inputs.hidden, None)
                    torch.npu.synchronize()
                inputs.lock_addresses()
                descriptor = BatchDescriptor(rows, 1, uniform=True)
                for name, forward in (("vendor", original_forward), ("native", installed_forward)):
                    impl.forward = forward
                    owner = {"python_calls": 0}

                    def runnable(positions, hidden, owner=owner):
                        # The wrapper's default output is a weak tensor alias.
                        # Retain the actual outer result before that conversion.
                        owner["output"] = layer(positions, hidden, None)
                        owner["python_calls"] += 1
                        return owner["output"]

                    wrapper = ACLGraphWrapper(runnable, config,
                                              runtime_mode=CUDAGraphMode.FULL)
                    wrapper.set_attn_backend(AscendSFABackend)
                    arm = {"name": name, "wrapper": wrapper, "owner": owner,
                           "forward": forward, "replays": 0}
                    arms.append(arm)
                    reset(0)
                    before_hits = len(hits)
                    counter = compilation_counter.num_cudagraph_captured
                    caller_stream = torch.npu.current_stream(device)
                    capture_record = {"rows": rows, "arm": name, "phase": "capture"}
                    report["captures"].append(capture_record)
                    persist()
                    with context(inputs, CUDAGraphMode.FULL):
                        reserves = {}
                        slot_reserves = {}
                        if name == "native":
                            pool.reserve_capture(inputs.hidden)
                            reserves = dict(pool.capture_reserves)
                            slot_reserves = dict(pool.slot_reserves)
                            if len(reserves) != 1 or len(slot_reserves) != 2:
                                raise RuntimeError(
                                    "Reserve one geometry and two independent slot producers")
                        enabled = monitor.cudagraph_capturing_enabled
                        monitor.set_cudagraph_capturing_enabled(True)
                        try:
                            with graph_capture(device) as capture:
                                arm["stream"] = capture.stream
                                capture_output = wrapper(inputs.positions, inputs.hidden)
                            caller_stream.wait_stream(capture.stream)
                        finally:
                            monitor.set_cudagraph_capturing_enabled(enabled)
                        get_forward_context().capturing = False
                    entries = wrapper.concrete_aclgraph_entries
                    entry = entries.get(descriptor)
                    if (len(entries) != 1 or entry is None
                            or not isinstance(entry.aclgraph, UpdatableGraph)):
                        raise RuntimeError(
                            "The official wrapper did not create exactly one actual graph")
                    if compilation_counter.num_cudagraph_captured != counter + 1:
                        raise RuntimeError("Capture count did not increase exactly once")
                    if owner["python_calls"] != 1 or len(hits) - before_hits != (name == "native"):
                        raise RuntimeError(
                            "Capture did not enter the intended real outer path exactly once")
                    if (capture_output.data_ptr() != owner["output"].data_ptr()
                            or entry.output.data_ptr() != owner["output"].data_ptr()):
                        raise RuntimeError("The captured output lost its strong storage owner")
                    stream_id = capture.stream.npu_stream
                    if stream_id == caller_stream.npu_stream:
                        raise RuntimeError("The official capture stream must differ from warmup")
                    reserve_claimed = None
                    if name == "native":
                        claimed = [value for key, value in pool.entries.items()
                                   if key[1] == stream_id and key[2] == rows]
                        reserve_claimed = (len(claimed) == 1 and
                                           any(claimed[0] is value for value in reserves.values())
                                           and not any(key in pool.capture_reserves
                                                       for key in reserves))
                        for slot_key, destination in slot_reserves.items():
                            if pool.slot_entries.get((stream_id, *slot_key)) is not destination:
                                raise RuntimeError(
                                    "Capture did not claim its preallocated live slot cast")
                        if (not reserve_claimed
                                or any(key in pool.slot_reserves for key in slot_reserves)):
                            raise RuntimeError(
                                "Capture did not consume its workspace and slot reserves")
                    capture_record.update(
                        phase="captured_not_executed", native_python_calls=len(hits) - before_hits,
                        runnable_calls=owner["python_calls"], capture_counter_delta=1,
                        capture_stream=stream_id, caller_stream=caller_stream.npu_stream,
                        workspace_reserve_claimed=reserve_claimed,
                        output_owner_ptr=owner["output"].data_ptr(),
                        weak_output_ptr=entry.output.data_ptr(),
                        graph_type=type(entry.aclgraph).__name__,
                        graph_tasks=len(entry.aclgraph.tasks), input_signatures=inputs.signatures)
                    inputs.check_addresses()
                    persist()
                for iteration in range(args.forward_replays):
                    plan = inputs.update(iteration)
                    case = {"rows": rows, "iteration": iteration, "phase": "eager_oracle",
                            "inputs": inputs.witness(), "pass": False, "arms": {}}
                    report["cases"].append(case)
                    persist()
                    impl.forward = original_forward
                    reset(iteration)
                    before_hits = len(hits)
                    with context(inputs, CUDAGraphMode.NONE):
                        expected = state(layer(inputs.positions, inputs.hidden, None))
                    if len(hits) != before_hits:
                        raise RuntimeError("The eager oracle entered the native body")
                    case["eager_output"] = tensor_witness(torch, expected["output"])
                    order = arms if iteration % 2 == 0 else list(reversed(arms))
                    for arm in order:
                        case["phase"] = arm["name"] + "_graph_replay"
                        persist()
                        impl.forward = arm["forward"]
                        reset(iteration)
                        inputs.check_addresses()
                        before_hits = len(hits)
                        python_calls = arm["owner"]["python_calls"]
                        counter = compilation_counter.num_cudagraph_captured
                        with context(inputs, CUDAGraphMode.FULL):
                            result = arm["wrapper"](inputs.positions, inputs.hidden)
                        if (len(hits) != before_hits
                                or arm["owner"]["python_calls"] != python_calls
                                or compilation_counter.num_cudagraph_captured != counter):
                            raise RuntimeError("Replay ran Python or captured a new graph")
                        if result.data_ptr() != arm["owner"]["output"].data_ptr():
                            raise RuntimeError("Replay changed the owned output storage")
                        actual = state(result)
                        arm["replays"] += 1
                        checks = {key: metric(torch, actual[key], value)
                                  for key, value in expected.items()}
                        passed = all(item["pass"] and item["bitwise_equal"]
                                     for item in checks.values())
                        case["arms"][arm["name"]] = {
                            "checks": checks, "pass": passed, "python_calls_during_replay": 0,
                            "native_python_calls_during_replay": 0,
                            "replay_count": arm["replays"],
                            "output": tensor_witness(torch, actual["output"])}
                        persist()
                        if not passed:
                            filename = (f"outer-graph-{args.mode}-m{rows}"
                                        f"-r{iteration}-{arm['name']}.pt")
                            path = args.output / filename
                            torch.save({"plan": plan, "hidden": snapshot(torch, inputs.hidden),
                                        "expected": expected, "actual": actual}, path)
                            case["failure_dump"] = str(path)
                            raise RuntimeError(
                                "Actual outer graph replay differs from the eager oracle")
                    case["pass"] = all(arm["pass"] for arm in case["arms"].values())
                    case["phase"] = "comparison"
                    persist()
                    print(json.dumps({"outer_graph_case": case}), flush=True)
                width_cases = [case for case in report["cases"] if case["rows"] == rows]
                if args.forward_replays > 1:
                    hashes = {case["eager_output"]["sha256"] for case in width_cases}
                    lengths = {tuple(case["inputs"]["actual_main_length"]) for case in width_cases}
                    if len(hashes) < 2 or len(lengths) < 2:
                        raise RuntimeError(
                            "Changing inputs and actual lengths lack an observable output witness")
                if not all(case["eager_output"]["nonzero_elements"] > 0 for case in width_cases):
                    raise RuntimeError("The output witness is entirely zero")
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
        impl.forward = installed_forward
