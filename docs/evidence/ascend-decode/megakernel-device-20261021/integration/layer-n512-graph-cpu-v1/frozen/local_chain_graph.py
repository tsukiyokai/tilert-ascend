"""Capture the two local layer launches through the official ACL wrapper."""

import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import traceback

from local_chain import (
    BLOCKS, CHAIN_SHA, INDEXED_SHA, SHARED_SHA, Inputs, LayerCase,
    LayerFFNAdapter, Library, bind_caches, digest, load_ffn, nd, publish_topk,
    signature, snapshot, source_identity, successor_inputs, tensor_witness,
)
from local_chain_views import poison, strict_checks
from down_tiling import check_down_tiling


class Observation:
    """Retain every graph input, output, packet and immediate copy target."""

    def __init__(self, torch, cases, inputs, topk, caches):
        self.torch, self.cases, self.inputs = torch, cases, inputs
        self.topk, self.caches = topk, caches
        self.calls = 0
        self.topk_copy_calls = 0
        self.capture_stream = None
        self.launch_modes = []
        self.topk_copies = {
            f"layer{number}.persistent_topk": torch.empty_like(topk[:inputs.rows])
            for number in (6, 7)}
        self.consumers = {"consumer.output": torch.empty_like(cases[1].bridge[10]),
                          "consumer.retained": torch.empty_like(cases[1].bridge[1])}
        self.output = tuple(self.consumers.values())

    def execute(self, combined):
        for number, case in zip((6, 7), self.cases):
            if combined:
                case.combined()
                self.launch_modes.append(case.mode)
            else:
                case.separate()
            if number == 6 and not combined:
                publish_topk(case.selected, self.topk, self.inputs.rows)
                self.topk_copy_calls += 1
            self.topk_copies[f"layer{number}.persistent_topk"].copy_(
                self.topk[:self.inputs.rows])
        for target, source in zip(self.output,
                                  (self.cases[1].bridge[10], self.cases[1].bridge[1])):
            target.copy_(source)
        return self.output

    def __call__(self, positions, hidden, residual):
        if (positions is not self.inputs.positions or hidden is not self.inputs.hidden
                or residual is not self.inputs.residual):
            raise RuntimeError("Capture substituted a static input owner")
        if not self.torch.npu.is_current_stream_capturing():
            raise RuntimeError("The ACL wrapper fell back to Python execution")
        self.calls += 1
        self.capture_stream = self.torch.npu.current_stream(hidden.device)
        return self.execute(True)

    def values(self):
        return {**{f"layer{number}.{name}": value
                   for number, case in zip((6, 7), self.cases)
                   for name, value in case.values.items()},
                **self.caches, **self.topk_copies, **self.consumers}

    def poison(self, integer):
        for case in self.cases:
            poison(case.values, integer)
            case.norm[5].fill_(float("nan"))
            case.body[20].fill_(1)
            case.body[20][0].zero_()
        if integer == 2:
            self.cases[0].selected.fill_(2)
        poison(self.topk_copies, integer)
        poison(self.consumers, integer)


def collect(torch, observation):
    torch.npu.synchronize()
    return {name: snapshot(torch, value)
            for name, value in observation.values().items()}


def run_local_graph(torch, layers, config, state, args):
    import torch_npu
    from attention_check import metric
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
    from vllm_ascend.utils import use_updatable_graph
    from vllm_ascend.worker.model_runner_v1 import graph_capture

    if (args.forward_sequence != 8192 or len(layers) != 2
            or len(set(args.forward_widths)) != len(args.forward_widths)
            or args.graph_residual_modes != ["provided"]
            or use_updatable_graph(AscendSFABackend)):
        raise ValueError("Use two layers, unique fixed widths, provided residual and direct SFA graph replay")
    for path, expected in ((args.indexed_library, INDEXED_SHA),
                           (args.shared_library, SHARED_SHA)):
        if digest(path) != expected:
            raise RuntimeError("Use the accepted eager parent attention libraries")
    os.environ["TILERT_INDEXED_ATTENTION_LIBRARY"] = str(args.indexed_library)
    os.environ["TILERT_SHARED_ATTENTION_LIBRARY"] = str(args.shared_library)
    sys.path.insert(0, str(args.adapter.resolve().parents[3]))
    adapter = importlib.import_module("tilert.models.glm_5_2_ascend.attention_native")
    if Path(adapter.__file__).resolve() != args.adapter.resolve():
        raise RuntimeError("The frozen attention packing source changed")
    adapter.load()
    native_ffn = load_ffn(args.ffn_library)
    chain_library = Library(args.chain_library, CHAIN_SHA)
    topk = layers[0].self_attn.mla_attn.topk_indices_buffer
    if (topk.device.type != "npu" or topk.dtype != torch.int32
            or tuple(topk.shape) != (32, 2048)
            or any(layer.self_attn.mla_attn.topk_indices_buffer is not topk
                   or layer.self_attn.mla_attn.mla_attn.impl.topk_indices_buffer is not topk
                   for layer in layers)):
        raise RuntimeError("The actual layers lost their shared persistent top-k owner")
    caches, cache_owners = bind_caches(torch, torch_npu, layers, config, args.mode)
    generator = torch.Generator().manual_seed(42022)
    bases = {}
    for name, cache in caches.items():
        if cache.dtype == torch.int8:
            base = torch.randint(-127, 128, cache.shape, generator=generator, dtype=torch.int8)
        elif name.endswith("index_scale"):
            base = (torch.rand(cache.shape, generator=generator) / 128 + 1 / 64).to(cache.dtype)
        else:
            base = torch.randn(cache.shape, generator=generator).to(cache.dtype)
        bases[name] = nd(torch_npu, base.to(topk.device))
    ffns = [LayerFFNAdapter(torch, torch_npu, layer, config, native_ffn)
            for layer in layers]
    cache_signatures = {name: signature(value) for name, value in caches.items()}
    topk_signature = signature(topk)
    report = {
        "scope": "Official ACLGraphWrapper: two N512 local layer launches with direct persistent top-k and immediate consumers; eager separate parents as oracle",
        "script": source_identity(run_local_graph),
        "component_fixture": source_identity(LayerCase),
        "actual_decoder_class": source_identity(type(layers[0])),
        "graph_wrapper": source_identity(ACLGraphWrapper),
        "graph_capture": source_identity(inspect.unwrap(graph_capture)),
        "forward_context": source_identity(inspect.unwrap(set_ascend_forward_context)),
        "libraries": {str(path): digest(path) for path in (
            args.chain_library, args.indexed_library, args.shared_library, args.ffn_library)},
        "mode": args.mode, "widths": args.forward_widths,
        "iterations": args.forward_replays, "blocks": BLOCKS,
        "tp_geometry": 16, "tp_collectives_executed": 0,
        "tp_contract": "Two identities; attention partial directly feeds FFN; no external post-AR1 tensor",
        "original_decoder_forward_called": False,
        "workspace_contract": "All parent-prepared tensors and C ABI packets are allocated before capture and strongly retained",
        "length_contract": "Actual lengths in live GPU tensors; 8192 is only the fixed maximum; page tables retain 64 columns",
        "poison_contract": "Floating NaN; non-address integers 1/2; bounded indices zero; active table one with E=0; cache reset separately",
        "input_rstd_contract": "Candidate diagnostic; excluded from strict oracle",
        "cache_signatures": cache_signatures, "topk_signature": topk_signature,
        "timing": False, "captures": [], "cases": [], "pass": False,
        "down_tiling_checks": [],
    }
    destination = args.output / "local-chain-graph.json"
    current_wrapper = None
    current_observation = None

    def persist():
        destination.write_text(json.dumps(report, indent=2) + "\n")

    def reset(iteration, observation, integer):
        for name, cache in caches.items():
            cache.copy_(bases[name].roll(iteration, 0))
        topk.fill_(-1)
        observation.poison(integer)

    def context(inputs, mode):
        return set_ascend_forward_context(
            inputs.metadata, config, num_tokens=inputs.rows,
            num_actual_tokens=inputs.rows, aclgraph_runtime_mode=mode,
            batch_descriptor=BatchDescriptor(inputs.rows, 1, uniform=True),
            model_instance=layers[0])

    def destroy_graph():
        nonlocal current_wrapper, current_observation
        if current_wrapper is not None:
            torch.npu.synchronize()
            for entry in current_wrapper.concrete_aclgraph_entries.values():
                if entry.aclgraph is not None:
                    entry.aclgraph.reset()
            current_wrapper.concrete_aclgraph_entries.clear()
        current_wrapper = None
        current_observation = None

    persist()
    try:
        parameters = get_graph_params()
        if parameters is None:
            set_graph_params(args.forward_widths)
        elif any(rows not in parameters.workspaces for rows in args.forward_widths):
            raise RuntimeError("The official graph parameter registry omits a width")
        with torch.inference_mode():
            for rows in args.forward_widths:
                inputs = Inputs(torch, torch_npu, layers, rows, topk.device)
                first = LayerCase(torch, torch_npu, layers[0], 6, inputs,
                                  inputs.hidden, inputs.residual, topk, adapter,
                                  ffns[0], chain_library)
                next_hidden, next_residual = successor_inputs(first.bridge)
                second = LayerCase(torch, torch_npu, layers[1], 7, inputs,
                                   next_hidden, next_residual, topk, adapter,
                                   ffns[1], chain_library)
                if first.mode != args.mode or second.mode != "shared":
                    raise RuntimeError("The original full/shared indexer selection changed")
                observation = Observation(torch, (first, second), inputs, topk, caches)
                current_observation = observation
                inputs.update(0)
                for _ in range(2):
                    reset(0, observation, 1)
                    observation.execute(False)
                    torch.npu.synchronize()
                if not all(case.parent_validated for case in observation.cases):
                    raise RuntimeError("The parent host validators did not run")
                inputs.lock_owners()
                owners = [case.owners() for case in observation.cases]
                output_pointers = [value.data_ptr() for value in observation.output]
                wrapper = ACLGraphWrapper(observation, config,
                                          runtime_mode=CUDAGraphMode.FULL)
                wrapper.set_attn_backend(AscendSFABackend)
                current_wrapper = wrapper
                caller = torch.npu.current_stream(topk.device)
                counter = compilation_counter.num_cudagraph_captured
                reset(0, observation, 2)
                copies_before_capture = observation.topk_copy_calls
                with context(inputs, CUDAGraphMode.FULL):
                    enabled = monitor.cudagraph_capturing_enabled
                    monitor.set_cudagraph_capturing_enabled(True)
                    try:
                        with graph_capture(topk.device) as capture:
                            output = wrapper(inputs.positions, inputs.hidden, inputs.residual)
                        caller.wait_stream(observation.capture_stream)
                    finally:
                        monitor.set_cudagraph_capturing_enabled(enabled)
                    get_forward_context().capturing = False
                entry = wrapper.concrete_aclgraph_entries.get(BatchDescriptor(rows, 1, uniform=True))
                if (len(wrapper.concrete_aclgraph_entries) != 1 or entry is None
                        or not isinstance(entry.aclgraph, UpdatableGraph)
                        or compilation_counter.num_cudagraph_captured != counter + 1
                        or observation.calls != 1
                        or observation.launch_modes != [args.mode, "shared"]):
                    raise RuntimeError("The official wrapper did not capture the two local launches once")
                if observation.topk_copy_calls != copies_before_capture:
                    raise RuntimeError("The candidate graph copied its top-k output")
                if ([value.data_ptr() for value in output] != output_pointers
                        or [value.data_ptr() for value in entry.output] != output_pointers):
                    raise RuntimeError("Capture lost the preallocated strong output owners")
                stream = observation.capture_stream.npu_stream
                if stream == caller.npu_stream:
                    raise RuntimeError("Capture unexpectedly reused the eager stream")
                report["captures"].append({
                    "rows": rows, "phase": "captured_not_executed",
                    "runnable_calls": observation.calls,
                    "launch_modes": observation.launch_modes.copy(),
                    "capture_counter_delta": 1, "capture_stream": stream,
                    "outer_capture_stream": capture.stream.npu_stream,
                    "caller_stream": caller.npu_stream,
                    "output_owner_ptrs": output_pointers,
                    "owners": owners, "input_signatures": inputs.signatures,
                    "observed_boundaries": len(observation.values()),
                    "graph_type": type(entry.aclgraph).__name__,
                    "graph_tasks": len(entry.aclgraph.tasks),
                    "workspace_preallocated": True,
                    "candidate_topk_copy_calls": 0,
                    "down_tiling": [case.down_tiling_before
                                    for case in observation.cases],
                })
                persist()
                for iteration in range(args.forward_replays):
                    plan = inputs.update(iteration)
                    original_inputs = {name: snapshot(torch, value)
                                       for name, value in inputs.fields().items()}
                    reset(iteration, observation, 1)
                    with context(inputs, CUDAGraphMode.NONE):
                        observation.execute(False)
                    expected = collect(torch, observation)
                    report["down_tiling_checks"].append({
                        "rows": rows, "iteration": iteration, "arm": "separate",
                        "layers": [check_down_tiling(torch, case.body[7])
                                   for case in observation.cases]})
                    reset(iteration, observation, 2)
                    calls = observation.calls
                    modes = observation.launch_modes.copy()
                    counter = compilation_counter.num_cudagraph_captured
                    with context(inputs, CUDAGraphMode.FULL):
                        result = wrapper(inputs.positions, inputs.hidden, inputs.residual)
                    if (observation.calls != calls or observation.launch_modes != modes
                            or compilation_counter.num_cudagraph_captured != counter):
                        raise RuntimeError("Replay re-entered Python or captured another graph")
                    if [value.data_ptr() for value in result] != output_pointers:
                        raise RuntimeError("Replay replaced a preallocated output owner")
                    actual = collect(torch, observation)
                    internal_unchanged = bool((first.selected == 2).all())
                    inactive_unchanged = bool((topk[rows:] == -1).all())
                    if not internal_unchanged or not inactive_unchanged:
                        raise RuntimeError("Replay wrote outside the active persistent top-k owner")
                    report["down_tiling_checks"].append({
                        "rows": rows, "iteration": iteration, "arm": "combined",
                        "layers": [check_down_tiling(torch, case.body[7])
                                   for case in observation.cases]})
                    inputs.check_owners()
                    if (owners != [case.owners() for case in observation.cases]
                            or signature(topk) != topk_signature
                            or cache_signatures != {name: signature(value)
                                                    for name, value in caches.items()}):
                        raise RuntimeError("Replay changed a retained input or workspace owner")
                    for name, value in original_inputs.items():
                        now = snapshot(torch, inputs.fields()[name])
                        if not torch.equal(value.view(torch.uint8), now.view(torch.uint8)):
                            raise RuntimeError("Replay overwrote a live metadata or input producer")
                    checks, passed = strict_checks(torch, actual, expected, metric)
                    record = {
                        "rows": rows, "iteration": iteration, "pass": passed,
                        "checks": checks, "explicit_replay": iteration + 1,
                        "python_calls_during_replay": 0,
                        "candidate_topk_copy_calls": 0,
                        "unused_internal_topk_unchanged": internal_unchanged,
                        "inactive_persistent_rows_unchanged": inactive_unchanged,
                        "inputs": inputs.witness(),
                        "output": tensor_witness(torch, actual["consumer.output"]),
                        "retained": tensor_witness(torch, actual["consumer.retained"]),
                        "attention": {f"layer{number}": tensor_witness(
                            torch, actual[f"layer{number}.attention.partial"])
                            for number in (6, 7)},
                    }
                    report["cases"].append(record)
                    persist()
                    if not passed:
                        path = args.output / f"local-graph-{args.mode}-m{rows}-r{iteration}.pt"
                        torch.save({"plan": plan, "inputs": original_inputs,
                                    "expected": expected, "actual": actual}, path)
                        raise RuntimeError("The local-chain graph differs from its eager parents")
                observed = [item for item in report["cases"] if item["rows"] == rows]
                if any(item[name]["nonzero_elements"] == 0
                       for item in observed for name in ("output", "retained")):
                    raise RuntimeError("The graph produced an entirely zero output or residual witness")
                if args.forward_replays > 1:
                    if (len({item["output"]["sha256"] for item in observed}) < 2
                            or len({item["retained"]["sha256"] for item in observed}) < 2
                            or len({tuple(item["inputs"]["actual_main_lengths"][0])
                                    for item in observed}) < 2):
                        raise RuntimeError("The graph lacks changing output, residual or actual-length witnesses")
                    for number in (6, 7):
                        if len({item["attention"][f"layer{number}"]["sha256"]
                                for item in observed}) < 2:
                            raise RuntimeError("An attention partial did not consume the changing inputs")
                destroy_graph()
                del observation, wrapper, first, second, inputs
        report["pass"] = True
        persist()
        return report
    except Exception:
        report["error"] = traceback.format_exc()
        persist()
        raise
    finally:
        destroy_graph()
