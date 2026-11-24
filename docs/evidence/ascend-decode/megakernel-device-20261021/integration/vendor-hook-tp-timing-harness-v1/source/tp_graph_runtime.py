"""Own the official capture stream, graph and TP communication lifetime."""

import inspect


def reserve_capture_stream(torch):
    """Construct the official manager without starting capture or allocating SHMEM."""
    from vllm_ascend.compilation.updatable_graph import UpdatableGraph

    graph = UpdatableGraph()
    manager = torch.npu.graph(graph)
    return manager.capture_stream, (manager, graph)


def prepare_graph_registry(widths):
    from vllm_ascend.attention.sfa_v1 import AscendSFABackend
    from vllm_ascend.compilation.acl_graph import get_graph_params, set_graph_params
    from vllm_ascend.utils import use_updatable_graph

    if use_updatable_graph(AscendSFABackend):
        raise RuntimeError("The fixture requires direct SFA graph replay")
    parameters = get_graph_params()
    if parameters is None:
        set_graph_params(widths)
    elif any(rows not in parameters.workspaces for rows in widths):
        raise RuntimeError("The actual graph registry omits a requested width")


class CapturedChain:
    """Retain one fixed-width two-layer graph and every live input/output owner."""

    def __init__(self, torch, config, layers, inputs, cases, window, stream,
                 execute, consumers, values):
        from vllm.config import CUDAGraphMode
        from vllm_ascend.attention.sfa_v1 import AscendSFABackend
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        self.torch, self.config, self.layers = torch, config, layers
        self.inputs, self.cases, self.window, self.stream = inputs, cases, window, stream
        self.execute, self.values = execute, values
        self.output = tuple(consumers.values())
        self.output_pointers = tuple(value.data_ptr() for value in self.output)
        self.calls = 0
        self.replays = 0
        self.capture_stream = None
        self.closed = False
        self.wrapper = ACLGraphWrapper(self, config, runtime_mode=CUDAGraphMode.FULL)
        self.wrapper.set_attn_backend(AscendSFABackend)

    def context(self):
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context

        return set_ascend_forward_context(
            self.inputs.metadata, self.config, num_tokens=self.inputs.rows,
            num_actual_tokens=self.inputs.rows, aclgraph_runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=BatchDescriptor(self.inputs.rows, 1, uniform=True),
            model_instance=self.layers[0])

    def check_stream(self):
        actual = self.torch.npu.current_stream(self.inputs.hidden.device)
        if actual.npu_stream != self.stream.npu_stream:
            raise RuntimeError("The graph entered a stream different from its SHMEM window")
        if self.window.handle == 0:
            raise RuntimeError("The graph lost its live SHMEM window")
        return actual

    def __call__(self, positions, hidden, residual):
        if (positions is not self.inputs.positions or hidden is not self.inputs.hidden
                or residual is not self.inputs.residual):
            raise RuntimeError("The graph substituted a static input owner")
        if not self.torch.npu.is_current_stream_capturing():
            raise RuntimeError("The official wrapper fell back to Python")
        self.capture_stream = self.check_stream()
        self.calls += 1
        self.execute(True, [])
        return self.output

    def snapshot(self):
        from local_chain import snapshot

        self.torch.npu.synchronize()
        return {name: snapshot(self.torch, value) for name, value in self.values.items()}

    def capture(self):
        from local_chain import source_identity
        from vllm.compilation import monitor
        from vllm.compilation.counter import compilation_counter
        from vllm.forward_context import BatchDescriptor, get_forward_context
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
        from vllm_ascend.compilation.updatable_graph import UpdatableGraph
        from vllm_ascend.worker.model_runner_v1 import graph_capture

        if self.calls or self.closed or not all(case.local.parent_validated for case in self.cases):
            raise RuntimeError("Capture requires validated parents and a fresh graph owner")
        caller = self.check_stream()
        before = self.snapshot()
        counter = compilation_counter.num_cudagraph_captured
        with self.context():
            enabled = monitor.cudagraph_capturing_enabled
            monitor.set_cudagraph_capturing_enabled(True)
            try:
                with graph_capture(self.inputs.hidden.device) as outer:
                    result = self.wrapper(self.inputs.positions, self.inputs.hidden,
                                          self.inputs.residual)
                caller.wait_stream(self.capture_stream)
            finally:
                monitor.set_cudagraph_capturing_enabled(enabled)
            get_forward_context().capturing = False
        entries = self.wrapper.concrete_aclgraph_entries
        entry = entries.get(BatchDescriptor(self.inputs.rows, 1, uniform=True))
        if (len(entries) != 1 or entry is None or not isinstance(entry.aclgraph, UpdatableGraph)
                or self.calls != 1
                or compilation_counter.num_cudagraph_captured != counter + 1):
            raise RuntimeError("The official wrapper did not capture exactly once")
        for outputs in (result, entry.output):
            if tuple(value.data_ptr() for value in outputs) != self.output_pointers:
                raise RuntimeError("Capture lost the preallocated output owners")
        after = self.snapshot()
        if any(not self.torch.equal(before[name].view(self.torch.uint8),
                                    after[name].view(self.torch.uint8)) for name in before):
            raise RuntimeError("Capture executed work before the first explicit replay")
        return {"rows": self.inputs.rows, "phase": "captured_not_executed",
                "runnable_calls": self.calls, "capture_counter_delta": 1,
                "capture_stream": self.capture_stream.npu_stream,
                "window_stream": self.stream.npu_stream,
                "outer_capture_stream": outer.stream.npu_stream,
                "caller_stream": caller.npu_stream,
                "output_owner_ptrs": self.output_pointers,
                "unchanged_capture_boundaries": len(before),
                "graph_type": type(entry.aclgraph).__name__,
                "graph_tasks": len(entry.aclgraph.tasks),
                "wrapper": source_identity(ACLGraphWrapper),
                "capture_context": source_identity(inspect.unwrap(graph_capture)),
                "torch_graph_manager": source_identity(self.torch.npu.graph),
                "workspace_preallocated": True, "window_allocated_before_capture": True}

    def replay(self):
        from vllm.compilation.counter import compilation_counter

        if self.calls != 1 or self.closed:
            raise RuntimeError("Replay requires one live captured graph")
        self.check_stream()
        before = compilation_counter.num_cudagraph_captured
        with self.context():
            result = self.wrapper(self.inputs.positions, self.inputs.hidden, self.inputs.residual)
        if self.calls != 1 or compilation_counter.num_cudagraph_captured != before:
            raise RuntimeError("Replay entered Python or captured another graph")
        if tuple(value.data_ptr() for value in result) != self.output_pointers:
            raise RuntimeError("Replay replaced a preallocated output owner")
        self.replays += 1

    def close(self):
        if self.closed:
            return
        self.torch.npu.synchronize()
        for entry in self.wrapper.concrete_aclgraph_entries.values():
            if entry.aclgraph is not None:
                entry.aclgraph.reset()
        self.wrapper.concrete_aclgraph_entries.clear()
        self.closed = True
