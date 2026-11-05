"""Fixed-width graph for GLM-5.2's MTP draft layer and its own head."""


WIDTHS = (1, 4, 6, 8, 12, 16)


class CannMtpGraph:
    """Replay the draft and its own head; returned storage is reused.

    Warm the eager path at this width on every TP rank before enabling
    the graph. The caller owns ``set_skip_topk``: an M1 chain replays
    against shared indices, the M4 target-conditioned pass selects its
    own. Seed forwards retain eager execution.
    """

    def __init__(self, draft, width=1):
        import torch

        if width not in WIDTHS:
            raise ValueError(f"MTP graphs capture widths {WIDTHS}")
        self.draft = draft
        self.width = width
        self.tokens, self.positions, self.slots = (
            draft._input_buffers[width][:3]
        )
        self.parent = torch.empty(
            width, 6144, dtype=torch.bfloat16, device=draft.device,
        )
        self.slots_i64 = torch.empty(
            width, dtype=torch.int64, device=draft.device,
        )
        self.graph = torch.npu.NPUGraph()
        self.pool = torch.npu.graph_pool_handle()
        self.capture_count = 0
        self.replay_count = 0
        self.output = None
        self.metadata = None
        self.addresses = None
        self.closed = False

    @staticmethod
    def _addresses(metadata):
        from .cann_graph import CannGraph

        return {
            name: {field: CannGraph._signature(getattr(value, field))
                   for field in CannGraph._METADATA_FIELDS}
            for name, value in metadata.items()
        }

    def validate_metadata(self, metadata):
        """Check persistent replay buffers outside the measured path."""
        if self._addresses(metadata) != self.addresses:
            raise ValueError("MTP graph metadata addresses changed")

    def _execute(self):
        from .cann_model import CannModel

        # Captured here rather than issued per round: the gather reads the
        # resident positions and writes the rope slices this graph captured,
        # so a replay refreshes them with no host launch at all.
        CannModel.refresh_hoisted_rope(self.draft, self.width, self.positions)
        model = self.draft.model
        pre_norm, recycle = model(
            input_ids=self.tokens, positions=self.positions,
            hidden_states=self.parent, spec_step_idx=0,
        )
        logits = model.compute_logits(pre_norm, spec_step_idx=0)
        return (pre_norm, recycle, logits, logits.argmax(dim=-1),
                self.draft.indices_buffer[:self.width].clone())

    def forward(self, metadata, parent):
        import torch
        from vllm.config import set_current_vllm_config
        from vllm_ascend.ascend_forward_context import (
            set_ascend_forward_context,
        )
        from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata
        from vllm_ascend.worker.model_runner_v1 import graph_capture

        if self.closed:
            raise RuntimeError("MTP graph is closed")
        # A chained parent can be this graph's previous recycle output.
        self.parent.copy_(parent)
        self.slots_i64.copy_(self.slots)
        if not self.capture_count:
            for value in metadata.values():
                if isinstance(value, AscendSFAMetadata):
                    value.kv_slots_i64 = (value.slot_mapping, self.slots_i64)
            self.metadata = metadata
            self.addresses = self._addresses(metadata)
        with (
            set_current_vllm_config(self.draft.config),
            set_ascend_forward_context(
                metadata, self.draft.config, num_tokens=self.width,
                num_actual_tokens=self.width,
                model_instance=self.draft.model,
                is_draft_model=True,
            ),
        ):
            if not self.capture_count:
                caller = torch.npu.current_stream(self.draft.device)
                with graph_capture(self.draft.device) as capture:
                    with torch.npu.graph(
                        self.graph, pool=self.pool, stream=capture.stream,
                    ):
                        self.output = self._execute()
                caller.wait_stream(capture.stream)
                self.capture_count += 1
            self.graph.replay()
            self.replay_count += 1
        return self.output

    def close(self):
        import torch

        if not self.closed:
            with torch.npu.device(self.draft.device):
                torch.npu.synchronize(self.draft.device)
                self.graph.reset()
            self.metadata = self.output = None
            self.closed = True
