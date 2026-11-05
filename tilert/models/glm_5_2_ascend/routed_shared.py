"""Interleave TP16 routed INT4 and shared W8A8 bodies on one AIC/AIV launch.

The routed callback hook preserves vendor input preparation, top-k, finalize
and context lifetimes. The owning MoE runner retains routed scaling, addition
and TP reduction. Each stream owns its scratch, shared across sequential layers.
"""

import inspect
import os
from pathlib import Path
from types import MethodType

from .routed_expert import (
    EXPERTS_PER_RANK,
    HIDDEN,
    MAX_TOKENS,
    _dispatch_rows,
    _WorkspacePool,
)


LIBRARY_DIR = Path(os.environ.get(
    "TILERT_ROUTED_SHARED_LIB_DIR",
    Path(__file__).resolve().parents[3]
    / "tests/glm52/routed_int4/build-pipeline-front",
))
ACTIVE_LIBRARY_DIR = Path(os.environ.get(
    "TILERT_ROUTED_SHARED_ACTIVE_LIB_DIR",
    Path(__file__).resolve().parents[3]
    / "tests/glm52/routed_int4/build-pipeline-front-active",
))


def candidate_built():
    """Whether the active-list variant is there to be compared against."""
    return (ACTIVE_LIBRARY_DIR / "routed_int4_torch.so").exists()


def library_paths():
    """The accepted library, and the candidate once it has been built.

    The candidate is a comparison, not a dependency: an arm must neither fail
    to start without it nor name a namespace it never loaded.
    """
    return ([LIBRARY_DIR / "routed_int4_torch.so"]
            + ([ACTIVE_LIBRARY_DIR / "routed_int4_torch.so"]
               if candidate_built() else []))


def pack_shared_weights(gate_up, down):
    """Pack independent K/N tiles without changing signed INT8 codes."""
    import torch

    if (gate_up.shape != (HIDDEN, 256) or down.shape != (128, HIDDEN)
            or gate_up.dtype != torch.int8 or down.dtype != torch.int8):
        raise ValueError("Shared weights require INT8 [6144,256] and [128,6144]")
    gate = gate_up.reshape(6, 1024, 4, 64).permute(0, 2, 1, 3)
    output = down.reshape(128, 24, 256).permute(1, 0, 2)
    return gate.contiguous().reshape(24, 1024, 64), output.contiguous()


class WorkspacePool(_WorkspacePool):
    """Reserve routed and shared buffers together before graph capture."""

    def __init__(self, blocks=24, input_front=True):
        super().__init__(EXPERTS_PER_RANK, blocks, weight_layout=1,
                         from_dispatch=True)
        self.input_front = input_front

    def namespace_name(self, width):
        if not self.input_front:
            return "tilert_pipeline"
        if self.blocks == 24 and width == 12 and candidate_built():
            return "tilert_pipeline_front_active"
        return "tilert_pipeline_front"

    def _prepare(self, rows):
        import torch

        namespace = getattr(torch.ops, self.namespace_name(rows.shape[0]))
        return namespace.routed_shared_prepare(
            rows, self.experts, self.blocks, self.weight_layout,
        )


def load():
    import torch

    for path in library_paths():
        torch.ops.load_library(str(path))


def _validate_runner(runner):
    """Require the rank-local TP contract used by the device comparison."""
    config = runner.moe_config
    shared = runner.ascend_shared_experts
    routed = runner.routed_experts
    if (config.tp_size != 16 or config.ep_size != 1 or config.dp_size != 1
            or config.pcp_size != 1
            or config.is_sequence_parallel
            or runner.routed_input_transform is not None
            or runner.routed_output_transform is not None):
        raise ValueError(
            "Routed/shared requires TP16, EP1, DP1, PCP1 and no SP transforms"
        )
    if (shared is None or shared.weights_replicated
            or shared.situ_activation is not None or shared.swiglu_limit
            or shared.swiglu_alpha != 1 or shared.swiglu_beta != 0
            or getattr(shared.layer, "expert_gate", None) is not None
            or not routed.return_with_event):
        raise ValueError(
            "Routed/shared requires one ungated, unclipped TP shared expert"
        )
    if "fused_experts" not in inspect.signature(routed.forward_impl).parameters:
        raise RuntimeError(
            "Apply patches/vllm-ascend/routed-shared-hook.patch before "
            "enabling the routed/shared body"
        )


def _shared_weights(shared):
    import torch
    import torch_npu
    from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import (
        AscendW8A8DynamicLinearMethod,
    )

    gate, down = shared.layer.gate_up_proj, shared.layer.down_proj
    for projection in (gate, down):
        scheme = projection.quant_method.quant_method
        if (not isinstance(scheme, AscendW8A8DynamicLinearMethod)
                or scheme.act_quant_type != torch.int8
                or projection.tp_size != 16
                or getattr(projection, "bias", None) is not None):
            raise ValueError("Shared projections require unbiased TP16 W8A8_DYNAMIC")
        if projection.weight_scale.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Shared projection scales require FP32 or BF16")
    packed_gate, packed_down = pack_shared_weights(
        torch_npu.npu_format_cast(gate.weight.detach(), 2),
        torch_npu.npu_format_cast(down.weight.detach(), 2),
    )
    gate_scale = gate.weight_scale_fp32.detach().contiguous()
    down_scale = down.weight_scale.detach().float().contiguous()
    if (gate_scale.shape != (256,) or gate_scale.dtype != torch.float32
            or down_scale.shape != (HIDDEN,)):
        raise ValueError("Shared projection scales require 256 and 6144 channels")
    return (torch_npu.npu_format_cast(packed_gate, 2), gate_scale,
            torch_npu.npu_format_cast(packed_down, 2), down_scale)


def run_body(x, topk_weights, topk_ids, shared_input, routed_weights,
             shared_weights, pool):
    """Return workspace-owned outputs and counts for this stream's next consumer."""
    import torch
    import torch_npu

    if (x.dtype != torch.bfloat16 or shared_input.dtype != torch.bfloat16
            or x.shape != shared_input.shape or x.shape[1] != HIDDEN):
        raise ValueError("Routed/shared inputs require equal BF16 [M,6144] shapes")
    shared_codes, shared_scale = torch_npu.npu_dynamic_quant(shared_input)
    shared_arguments = [shared_codes, shared_scale.reshape(-1), *shared_weights]
    if pool.input_front:
        scratch, sync = pool.get(shared_codes)
        namespace = getattr(torch.ops, pool.namespace_name(x.shape[0]))
        routed, shared = namespace.routed_shared_from_bf16(
            x, *routed_weights, topk_ids.to(torch.int32), topk_weights.float(),
            shared_arguments, scratch, sync, pool.experts, pool.blocks,
            pool.weight_layout,
        )
        return routed, shared, scratch[12]
    rows, inverse, counts, token_scale = _dispatch_rows(x, topk_ids)
    scratch, sync = pool.get(rows[:x.shape[0]])
    routed, shared = torch.ops.tilert_pipeline.routed_shared_from_dispatch(
        rows, *routed_weights, token_scale.reshape(-1), inverse, counts,
        topk_weights.float(),
        shared_arguments,
        scratch, sync, pool.experts, pool.blocks, pool.weight_layout,
    )
    return routed, shared, counts


def install(runner, routed_weights, pool):
    """Replace an owning runner's local body, retaining its outer finalize."""
    import torch
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm_ascend.ops.fused_moe.moe_comm_method import (
        AllGatherCommImpl,
        FusedExpertsResult,
    )
    from vllm_ascend.ops.fused_moe.moe_mlp import maybe_record_event

    _validate_runner(runner)
    shared_weights = _shared_weights(runner.ascend_shared_experts)
    original = runner._forward_impl

    def forward(self, hidden_states, router_logits, shared_experts_input,
                input_ids=None):
        shared_input = (hidden_states if shared_experts_input is None
                        else shared_experts_input)
        if (not 1 <= hidden_states.shape[0] <= MAX_TOKENS
                or hidden_states.dtype != torch.bfloat16
                or shared_input.shape != hidden_states.shape
                or shared_input.dtype != torch.bfloat16
                or self.ascend_shared_experts.lora_context is not None
                or getattr(self.routed_experts, "_ascend_moe_lora_context", None)
                is not None):
            return original(hidden_states, router_logits, shared_experts_input,
                            input_ids)
        shared_output = None

        def body(layer, x, topk_weights, topk_ids, shared_experts,
                 shared_experts_input):
            nonlocal shared_output
            if (not isinstance(_EXTRA_CTX.moe_comm_method, AllGatherCommImpl)
                    or layer.ascend_pertoken_scale is not None):
                raise ValueError("Routed/shared requires unquantized AllGather input")
            start = maybe_record_event()
            routed, shared_output, counts = run_body(
                x, topk_weights, topk_ids, shared_input, routed_weights,
                shared_weights, pool,
            )
            end = maybe_record_event()
            return FusedExpertsResult(
                routed_out=routed, before_dispatch_evt=start,
                before_gmm2_evt=end, before_combine_evt=end,
                shared_activation_overlap_start_evt=end,
                group_list_type=1, expert_tokens=counts,
            )

        with self._sequence_parallel_context():
            if self.is_internal_router:
                router_logits = self._compute_router_logits(
                    shared_input, router_logits,
                )
            routed_output, _ = self.routed_experts.forward_impl(
                hidden_states=hidden_states, router_logits=router_logits,
                input_ids=input_ids, fused_experts=body,
            )
        if shared_output is None:
            raise RuntimeError("The routed forward did not call its fused body")
        return shared_output, routed_output

    runner._forward_impl = MethodType(forward, runner)

    def restore():
        runner._forward_impl = original

    return restore
