"""Fixed-width ACL graph for the CANN model and fixed cache storage."""


# The widths a captured graph is replayed at. Narrower rounds decode eagerly,
# so an arm that wants the narrowest comparison takes four, not two.
CAPTURED_WIDTHS = (4, 6, 8, 12, 16, 24, 32)


class CannGraph:
    """Capture model forward; prepare metadata and sample outside the graph.

    All TP ranks must finish uncaptured warmup before the first call.
    The caller updates inputs and metadata before each forward, retaining
    their storage through close(). Returned hidden storage is reused on replay.
    """

    _METADATA_FIELDS = (
        "slot_mapping", "seq_lens", "cum_query_lens", "block_table",
        "cos", "sin",
    )

    def __init__(self, engine, width=1):
        import torch
        from vllm.config import (
            CompilationMode, CUDAGraphMode, set_current_vllm_config,
        )
        from vllm.forward_context import BatchDescriptor
        from vllm_ascend.attention.sfa_v1 import AscendSFABackend
        from vllm_ascend.compilation.acl_graph import (
            ACLGraphWrapper, get_graph_params, set_graph_params,
        )

        self.device = engine.device
        self._prepare_fused_workspaces = engine._prepare_fused_workspaces
        self.config = engine.config
        self.model = engine.model
        self.width = width
        self.tokens, self.positions, self.slots = (
            engine._input_buffers[width][:3]
        )
        self.mode = CUDAGraphMode.FULL
        self.descriptor = BatchDescriptor(width, 1, uniform=True)
        self.capture_count = 0
        self.replay_count = 0
        self.capture_stream = None
        self._captured_metadata = None
        self._addresses = None
        self._closed = False
        if not self.config.model_config.enforce_eager:
            backbone = self.model.model
            compiled = getattr(backbone, "compiled", False) or callable(
                getattr(backbone, "aot_compiled_fn", None),
            )
            # One message for four unrelated conditions cost an arm: a width
            # the capture does not carry reported itself as a cold model.
            if width not in CAPTURED_WIDTHS:
                raise ValueError(
                    f"Capture carries widths {CAPTURED_WIDTHS}, not {width}"
                )
            if (self.config.compilation_config.mode
                    != CompilationMode.VLLM_COMPILE
                    or backbone.do_not_compile or not compiled):
                raise ValueError("Warm the compiled target before capture")
        if self.config.speculative_config is not None:
            raise ValueError("CannGraph captures only the target model")
        with torch.npu.device(self.device), (
            set_current_vllm_config(self.config)
        ):
            params = get_graph_params()
            if params is None:
                set_graph_params([width])
            elif width not in params.workspaces:
                raise ValueError("Existing graph parameters omit this width")
            self.slots_i64 = torch.empty(
                width, dtype=torch.int64, device=self.device,
            )
            self.wrapper = ACLGraphWrapper(
                self.model, self.config, runtime_mode=self.mode,
            )
            self.wrapper.set_attn_backend(AscendSFABackend)

    @staticmethod
    def _signature(tensor):
        return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
                str(tensor.dtype), str(tensor.device))

    def _metadata_addresses(self, metadata):
        return {
            name: {field: self._signature(getattr(value, field))
                   for field in self._METADATA_FIELDS}
            for name, value in metadata.items()
        }

    def _bind_capture_metadata(self, metadata):
        from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata

        if not metadata:
            raise ValueError("Capture requires SFA/indexer metadata")
        unique = {id(value): value for value in metadata.values()}
        sfa_count = 0
        for value in unique.values():
            if value.num_actual_tokens != self.width:
                raise ValueError("Metadata differs from the captured width")
            if value.slot_mapping.data_ptr() != self.slots.data_ptr():
                raise ValueError("Metadata must use the persistent slots")
            if isinstance(value, AscendSFAMetadata):
                # _int64_kv_slots otherwise reuses a warmup-only cast result.
                value.kv_slots_i64 = (value.slot_mapping, self.slots_i64)
                sfa_count += 1
        if not sfa_count:
            raise ValueError("No SFA metadata was supplied")
        self._addresses = self._metadata_addresses(metadata)
        self._captured_metadata = metadata

    def validate_metadata(self, metadata):
        """Check replay addresses during acceptance, outside the hot path."""
        if self._addresses is None:
            raise RuntimeError("Capture before validating replay addresses")
        if self._metadata_addresses(metadata) != self._addresses:
            raise ValueError("SFA/indexer replay buffer addresses changed")

    def _capture(self):
        import torch
        from vllm.compilation import monitor
        from vllm.forward_context import get_forward_context
        from vllm_ascend.worker.model_runner_v1 import graph_capture

        caller_stream = torch.npu.current_stream(self.device)
        self._prepare_fused_workspaces(self.width)
        enabled = monitor.cudagraph_capturing_enabled
        monitor.set_cudagraph_capturing_enabled(True)
        try:
            with graph_capture(self.device) as capture:
                self.capture_stream = capture.stream
                self.wrapper(self.tokens, self.positions)
            caller_stream.wait_stream(self.capture_stream)
        finally:
            monitor.set_cudagraph_capturing_enabled(enabled)
        entry = self.wrapper.concrete_aclgraph_entries[self.descriptor]
        if entry.aclgraph is None:
            raise RuntimeError("The wrapper did not create an ACL graph")
        self._entry = entry
        # Ascend capture records tasks; its output is not a completed forward.
        get_forward_context().capturing = False
        hidden = self.wrapper(self.tokens, self.positions)
        self.capture_count += 1
        self.replay_count += 1
        return hidden

    def forward(self, metadata):
        """Consume the fixed query width using freshly prepared metadata."""
        import torch
        from vllm.config import set_current_vllm_config
        from vllm_ascend.ascend_forward_context import (
            set_ascend_forward_context,
        )

        if self._closed:
            raise RuntimeError("CannGraph is closed")
        if self.capture_count:
            if not getattr(self, "_rope_handles_slots", False):
                self.slots_i64.copy_(self.slots, non_blocking=True)
            self._entry.aclgraph.replay()
            self.replay_count += 1
            return self._entry.output
        with torch.npu.device(self.device), torch.inference_mode(), (
            set_current_vllm_config(self.config)
        ):
            self.slots_i64.copy_(self.slots)
            if not self.capture_count:
                self._bind_capture_metadata(metadata)
            with set_ascend_forward_context(
                metadata, self.config, num_tokens=self.width,
                num_actual_tokens=self.width,
                aclgraph_runtime_mode=self.mode,
                batch_descriptor=self.descriptor, model_instance=self.model,
            ):
                if not self.capture_count:
                    return self._capture()
                hidden = self.wrapper(self.tokens, self.positions)
                self.replay_count += 1
                return hidden

    def close(self):
        """Drain device work and release the graph before worker shutdown."""
        import torch

        if self._closed:
            return
        with torch.npu.device(self.device):
            torch.npu.synchronize(self.device)
            for entry in self.wrapper.concrete_aclgraph_entries.values():
                if entry.aclgraph is not None:
                    entry.aclgraph.reset()
            self.wrapper.concrete_aclgraph_entries.clear()
        self._captured_metadata = None
        self.capture_stream = None
        self._closed = True
