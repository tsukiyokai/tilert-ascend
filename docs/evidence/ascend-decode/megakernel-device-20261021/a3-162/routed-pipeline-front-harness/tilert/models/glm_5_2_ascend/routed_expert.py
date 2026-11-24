"""TP16 W4A8 routed experts with one AscendC matrix/activation launch.

Install before ModelSlim weights are converted to NZ. Native weights are
copied ND INT4 B[K,N] by default, or borrowed final vendor NZ tensors.
Both pack adjacent output columns low nibble first. Route metadata retains
the original top-8 slot and its BF16 probability boundary. Vendor dispatch
quantization precedes the fused projections and combine; shared experts
and TP finalization remain with the caller.

Workspaces belong to a device, stream, token count and weight format. Layers on that
stream reuse them in execution order. Returned storage is valid until the
next call with the same workspace; the model consumes it before entering
the next layer. Reserve each captured width before recording; its first
consumer binds the reserved storage to the actual capture stream.
"""

import os
from pathlib import Path


HIDDEN = 6144
GATE_UP = 256
EXPERTS_PER_RANK = 256
TOP_K = 8
MAX_TOKENS = 32
MAX_ROUTES = MAX_TOKENS * TOP_K
BLOCKS = 16
LIBRARY_DIR = Path(os.environ.get(
    "TILERT_ROUTED_LIB_DIR",
    Path(__file__).resolve().parents[3]
    / "tests/glm52/routed_int4/build",
))


def _repack(packed, columns):
    """ModelSlim INT8 [E,N/2,K] to non-transposed packed B[E,K,N/2]."""
    import torch

    if (packed.dtype != torch.int8 or packed.ndim != 3
            or packed.shape[1] * 2 != columns):
        raise ValueError("Expected ModelSlim INT8 weights [E,N/2,K]")
    return packed.transpose(1, 2).contiguous()


def unpack_codes(packed):
    """Decode checkpoint INT4 [E,N/2,K] to signed INT8 [E,N,K]."""
    import torch

    if packed.ndim != 3 or packed.dtype != torch.int8:
        raise ValueError("Expected ModelSlim INT8 weights [E,N/2,K]")
    experts, halved, inner = packed.shape
    low = (packed & 15).to(torch.int8)
    high = ((packed >> 4) & 15).to(torch.int8)
    low = torch.where(low < 8, low, low - 16)
    high = torch.where(high < 8, high, high - 16)
    return torch.stack((low, high), dim=2).reshape(
        experts, 2 * halved, inner,
    )


def route_metadata(topk_ids, topk_weights):
    """Build fixed-capacity expert groups on device without host reads.

    Inputs are [M,8], with distinct expert IDs per token in [0,256).
    Active rows occupy the first M*8 entries, ordered by expert, token and
    slot. Padding is outside every group interval. Probabilities round to
    BF16, as in the vendor BF16 unpermute path.
    """
    import torch

    if (topk_ids.ndim != 2 or topk_ids.shape[1] != TOP_K
            or not 1 <= topk_ids.shape[0] <= MAX_TOKENS
            or topk_weights.shape != topk_ids.shape
            or topk_weights.device != topk_ids.device
            or topk_ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Routes require integer IDs and weights [M,8]")
    ids = topk_ids.reshape(-1).to(torch.int64)
    route = torch.arange(ids.numel(), device=ids.device)
    order = torch.argsort(ids * MAX_ROUTES + route)
    counts = torch.zeros(EXPERTS_PER_RANK, device=ids.device,
                         dtype=torch.int32)
    counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
    starts = counts.cumsum(0, dtype=torch.int32) - counts
    source_row = torch.zeros(MAX_ROUTES, device=ids.device,
                             dtype=torch.int32)
    source_slot = torch.zeros_like(source_row)
    weights = torch.zeros(MAX_ROUTES, device=ids.device,
                          dtype=torch.float32)
    source_row[:ids.numel()] = torch.div(
        order, TOP_K, rounding_mode="floor",
    ).to(torch.int32)
    source_slot[:ids.numel()] = (order % TOP_K).to(torch.int32)
    probabilities = topk_weights.to(torch.bfloat16).float().flatten()
    weights[:ids.numel()] = probabilities[order]
    return counts, starts, source_row, source_slot, weights


def _restore_token_rows(sorted_rows, sorted_scales, inverse, tokens):
    """Select original top-8 slot zero from token-major inverse indices."""
    positions = inverse.reshape(tokens, TOP_K)[:, 0].long()
    return (sorted_rows.index_select(0, positions),
            sorted_scales.reshape(-1).index_select(0, positions))


def _dispatch_rows(x, topk_ids):
    """Return all four outputs of the vendor W4A8 dispatch quantizer."""
    import torch_npu

    return torch_npu.npu_moe_init_routing_v2(
        x, topk_ids, active_num=x.shape[0] * TOP_K,
        expert_num=EXPERTS_PER_RANK, expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[0, EXPERTS_PER_RANK], quant_mode=1,
    )


def _quant_rows(x, topk_ids, token_scale):
    """Preserve the dispatch quantizer used by the vendor W4A8 chain."""
    import torch

    if x.dtype == torch.int8 and token_scale is not None:
        return x, token_scale
    if x.dtype != torch.bfloat16 or token_scale is not None:
        raise ValueError("Routed input requires BF16 or scaled INT8 rows")
    rows, inverse, _, scales = _dispatch_rows(x, topk_ids)
    return _restore_token_rows(rows, scales, inverse, x.shape[0])


class _WorkspacePool:
    """Own native allocations across layers, keeping streams independent."""

    def __init__(self, experts, blocks, weight_layout=0, from_dispatch=False):
        self.experts = experts
        self.blocks = blocks
        self.weight_layout = weight_layout
        self.from_dispatch = from_dispatch
        self.layer_weights = {}
        self.entries = {}
        self.capture_reserves = {}

    def _prepare(self, rows):
        import torch

        # ND's original calling convention also accepts the validated old .so.
        options = {"weight_layout": self.weight_layout} if self.weight_layout else {}
        if self.from_dispatch:
            options["from_dispatch"] = True
        return torch.ops.tilert.routed_experts_prepare(
            rows, self.experts, self.blocks, **options,
        )

    def reserve_capture(self, rows):
        """Allocate before capture; the first consumer binds its stream."""
        key = (rows.device, rows.shape[0], self.from_dispatch, self.weight_layout)
        self.capture_reserves[key] = self._prepare(rows)

    def get(self, rows):
        import torch

        stream = torch.npu.current_stream(rows.device)
        key = (rows.device, stream.npu_stream, rows.shape[0],
               self.from_dispatch, self.weight_layout)
        if key not in self.entries:
            if torch.npu.is_current_stream_capturing():
                reserve_key = (rows.device, rows.shape[0],
                               self.from_dispatch, self.weight_layout)
                if reserve_key not in self.capture_reserves:
                    raise RuntimeError(
                        "Reserve routed workspace before NPU graph capture"
                    )
                self.entries[key] = self.capture_reserves.pop(reserve_key)
            else:
                self.entries[key] = self._prepare(rows)
        return self.entries[key]


def _load():
    import torch

    torch.ops.load_library(str(LIBRARY_DIR / "routed_int4_torch.so"))


def _check_checkpoint_weights(layer, experts):
    import torch

    expected = {
        "w13_weight": (experts, GATE_UP // 2, HIDDEN),
        "w2_weight": (experts, HIDDEN // 2, GATE_UP // 2),
        "w13_weight_scale": (experts, GATE_UP, 1),
        "w2_weight_scale": (experts, HIDDEN, 1),
        "w13_scale_bias": (experts, GATE_UP, 1),
        "w2_scale_bias": (experts, HIDDEN, 1),
    }
    for name, shape in expected.items():
        tensor = getattr(layer, name)
        dtype = torch.int8 if name.endswith("_weight") else torch.float32
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(f"{name} requires {dtype} {shape} at TP16")


def _weights(layer, experts):
    """Copy rank-local ModelSlim tensors before vendor conversion."""
    _check_checkpoint_weights(layer, experts)
    return (
        _repack(layer.w13_weight.detach(), GATE_UP),
        layer.w13_scale_bias.detach().squeeze(-1).clone(),
        layer.w13_weight_scale.detach().squeeze(-1).clone(),
        _repack(layer.w2_weight.detach(), HIDDEN),
        layer.w2_scale_bias.detach().squeeze(-1).clone(),
        layer.w2_weight_scale.detach().squeeze(-1).clone(),
    )


def _borrow_nz_weights(layer, experts, scales):
    """Retain the vendor's final INT32 carriers and FP32 corrections."""
    import torch
    import torch_npu

    for name, shape in (
        ("w13_weight", (experts, HIDDEN, GATE_UP // 8)),
        ("w2_weight", (experts, GATE_UP // 2, HIDDEN // 8)),
    ):
        value = getattr(layer, name)
        if (value.dtype != torch.int32 or tuple(value.shape) != shape
                or not value.is_contiguous() or value.storage_offset() != 0
                or torch_npu.get_npu_format(value) != 29):
            raise ValueError(
                f"{name} requires a complete physical NZ INT32 bank {shape}"
            )
    for name, columns in (("w13_scale_bias", GATE_UP),
                          ("w2_scale_bias", HIDDEN)):
        value = getattr(layer, name)
        if (value.dtype != torch.float32
                or tuple(value.shape) != (experts, columns)
                or not value.is_contiguous()):
            raise ValueError(f"{name} requires contiguous FP32 [{experts},{columns}]")
    return (
        layer.w13_weight, layer.w13_scale_bias, scales[0],
        layer.w2_weight, layer.w2_scale_bias, scales[1],
    )


def _run(method, layer, x, topk_weights, topk_ids, weights, pool):
    """Replace the entire routed body; finalize remains with the caller."""
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm_ascend.ops.fused_moe.moe_comm_method import (
        AllGatherCommImpl,
        FusedExpertsResult,
    )
    from vllm_ascend.ops.fused_moe.moe_mlp import maybe_record_event
    import torch

    if not isinstance(_EXTRA_CTX.moe_comm_method, AllGatherCommImpl):
        raise ValueError("The routed kernel requires the TP all-gather path")
    start = maybe_record_event()
    options = {"weight_layout": pool.weight_layout} if pool.weight_layout else {}
    if pool.from_dispatch:
        rows, inverse, expert_tokens, token_scale = _dispatch_rows(x, topk_ids)
        # Only the shape is consumed by the workspace pool. This view does
        # not restore or copy rows; the kernel owns that data movement.
        scratch, sync_base = pool.get(rows[:x.shape[0]])
        output = torch.ops.tilert.routed_experts_from_dispatch(
            rows, *weights, token_scale.reshape(-1), inverse, expert_tokens,
            topk_weights.float(), scratch, sync_base, pool.experts,
            pool.blocks, **options,
        )
    else:
        rows, token_scale = _quant_rows(
            x, topk_ids, layer.ascend_pertoken_scale,
        )
        routes = route_metadata(topk_ids, topk_weights)
        scratch, sync_base = pool.get(rows)
        output = torch.ops.tilert.routed_experts(
            rows, *weights, token_scale.reshape(-1), *routes,
            scratch, sync_base, pool.experts, pool.blocks, **options,
        )
        expert_tokens = routes[0].to(torch.int64)
    end = maybe_record_event()
    return FusedExpertsResult(
        routed_out=output,
        before_dispatch_evt=start,
        before_gmm2_evt=end,
        before_combine_evt=end,
        shared_activation_overlap_start_evt=end,
        group_list_type=1,
        expert_tokens=expert_tokens,
    )


def _install_layer(layer, method, weights, pool):
    """Dispatch on layer identity because a quant method can be shared."""
    original = method.apply

    def apply(layer, x, topk_weights, topk_ids, shared_experts,
              shared_experts_input, _target=layer):
        import torch

        scaled_input = (layer is _target and getattr(pool, "from_dispatch", False)
                        and (x.dtype != torch.bfloat16
                             or layer.ascend_pertoken_scale is not None))
        if layer is not _target or x.shape[0] > MAX_TOKENS or scaled_input:
            return original(
                layer, x, topk_weights, topk_ids,
                shared_experts, shared_experts_input,
            )
        return _run(method, layer, x, topk_weights, topk_ids, weights, pool)

    method.apply = apply

    def restore():
        method.apply = original

    return restore


def install_before_conversion(quant_class, experts=EXPERTS_PER_RANK,
                              blocks=BLOCKS, weight_layout=None,
                              reuse_dispatch=None):
    """Install a routed body with copied ND or borrowed vendor NZ weights.

    Supports TP16, EP=1, 256 experts, unclipped SwiGLU, and M=1..32.
    Larger prefill batches keep the vendor path. Restore removes both the
    loading hook and the installed per-object methods. ND is the default;
    select NZ with weight_layout="nz" or TILERT_ROUTED_WEIGHT_LAYOUT=nz.
    NZ retains final vendor tensors and copies only the original FP32 scales.
    """
    if experts != EXPERTS_PER_RANK:
        raise ValueError("TP16 shards all 256 experts on each rank")
    if weight_layout is None:
        weight_layout = os.environ.get("TILERT_ROUTED_WEIGHT_LAYOUT", "nd")
    if weight_layout not in ("nd", "nz"):
        raise ValueError("weight_layout must be 'nd' or 'nz'")
    if reuse_dispatch is None:
        reuse_dispatch = os.environ.get("TILERT_ROUTED_REUSE_DISPATCH", "0") == "1"
    _load()
    pool = _WorkspacePool(experts, blocks, int(weight_layout == "nz"),
                          reuse_dispatch)
    original = quant_class.process_weights_after_loading
    installed, undo = [], []

    def taking(self, layer):
        if (getattr(layer, "ascend_expert_map", None) is not None
                or getattr(layer, "apply_router_weight_on_input", False)
                or getattr(layer, "global_redundant_expert_num", 0)
                or getattr(layer, "dynamic_eplb", False)):
            raise ValueError("Routed kernel requires fixed TP16 expert IDs")
        activation = getattr(layer.activation, "name", layer.activation)
        if str(activation).lower() != "silu":
            raise ValueError("Routed kernel requires SwiGLU activation")
        if getattr(layer.moe_config, "swiglu_limit", 0):
            raise ValueError("Routed kernel requires unclipped SwiGLU")
        if weight_layout == "nz":
            _check_checkpoint_weights(layer, experts)
            scales = (
                layer.w13_weight_scale.detach().squeeze(-1).clone(),
                layer.w2_weight_scale.detach().squeeze(-1).clone(),
            )
        else:
            weights = _weights(layer, experts)
        result = original(self, layer)
        if weight_layout == "nz":
            weights = _borrow_nz_weights(layer, experts, scales)
        pool.layer_weights[id(layer)] = weights
        undo.append(_install_layer(layer, self, weights, pool))
        installed.append(layer)
        return result

    quant_class.process_weights_after_loading = taking

    def restore():
        quant_class.process_weights_after_loading = original
        for action in reversed(undo):
            action()
        pool.layer_weights.clear()

    return restore, installed, pool
