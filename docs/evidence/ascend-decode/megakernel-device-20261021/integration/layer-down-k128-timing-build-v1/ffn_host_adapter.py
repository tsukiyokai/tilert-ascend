"""Bind one complete native FFN to the two original Decoder module calls."""

from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
from types import MethodType


LIBRARY_SHA = "0842cd0f3a946c39aefb3e926e01eb870a61d6d1ad8689aebcf52a1b3daa2b65"
NAMESPACE = "tilert_ffn_down_k128"
SCALE_SOURCE_SHA = "efdcc610c05a44952c2548e0d8b0cd48ca7704c3e2cf8247f8f08f5e765b830b"
DECODER_SOURCE_SHA = "96f4f755f62caa85869fb884209e7f8869882ddb6efbd40c930778b27bd3e39d"
RUNNER_SOURCE_SHA = "d4dfab36e66bd0a2ab9d78b9a867335adfc8cf0c418c93b34f5ccef63df00e27"
LOOKUP_SOURCE_SHA = "20b69fe514bebd6723a1378f89a564c8fec4144c30244101f629c3f32c188183"
HIDDEN = 6144
BLOCKS = 24
LAYOUT_NZ = 1


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(library):
    import torch

    if digest(library) != LIBRARY_SHA:
        raise RuntimeError("Use the accepted K128 complete FFN library")
    torch.ops.load_library(str(library))
    return getattr(torch.ops, NAMESPACE)


def stream_state(reference):
    import torch

    return (reference.device.index,
            torch.npu.current_stream(reference.device).npu_stream,
            reference.shape[0], torch.npu.is_current_stream_capturing())


@dataclass
class Scratch:
    bridge: list
    body: list
    sync: int

    def bytes(self):
        return sum(value.numel() * value.element_size()
                   for value in (*self.bridge, *self.body))


class LayerWorkspacePool:
    """Retain separate exact-width buffers for one layer and each stream."""

    def __init__(self, owner, prepare):
        self.owner = owner
        self.prepare = prepare
        self.entries = {}
        self.reserves = {}
        self.claims = []

    def _prepare(self, reference):
        bridge, body, sync = self.prepare(reference, BLOCKS, LAYOUT_NZ)
        return Scratch(bridge, body, sync)

    def reserve(self, reference):
        device, _, rows, capturing = stream_state(reference)
        if capturing:
            raise RuntimeError("Reserve complete-FFN buffers before capture")
        key = device, rows
        if key not in self.reserves:
            self.reserves[key] = self._prepare(reference)
        return self.reserves[key]

    def get(self, reference):
        device, stream, rows, capturing = stream_state(reference)
        key = device, stream, rows
        if key not in self.entries:
            if capturing:
                reserve = device, rows
                if reserve not in self.reserves:
                    raise RuntimeError("Capture requires this layer's reserved FFN buffers")
                self.entries[key] = self.reserves.pop(reserve)
                self.claims.append(key)
            else:
                self.entries[key] = self._prepare(reference)
        return self.entries[key], key


class PendingFFN:
    """Require the next MLP call to consume this post-norm result once."""

    def __init__(self):
        self.value = None

    def require_empty(self):
        if self.value is not None:
            raise RuntimeError("The preceding post-norm result was not consumed")

    def begin(self, scratch, key):
        self.require_empty()
        self.value = scratch, key
        return scratch.bridge[0], scratch.bridge[1]

    def consume(self, normalized, key):
        if self.value is None:
            raise RuntimeError("The MLP has no preceding complete-FFN result")
        scratch, expected = self.value
        if normalized is not scratch.bridge[0] or key != expected:
            raise RuntimeError("The MLP changed its normalized owner, stream or width")
        self.value = None
        return scratch


def lookup_runner(runner):
    """Preserve the original custom-op lookup and its legacy layer cursor."""
    from vllm.model_executor.layers.fused_moe.runner import moe_runner

    resolved = moe_runner.get_layer_from_name(
        moe_runner._resolve_layer_name(runner._encode_layer_name()))
    if resolved is not runner:
        raise RuntimeError("The original MoE lookup selected another runner")
    return resolved


def validate_layer(layer, config):
    import torch
    from vllm.model_executor.layers.fused_moe.runner import moe_runner
    from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
        ZeroExpertRouter,
    )
    from vllm_ascend.ops.fused_moe.shared_experts import SharedExpertMLPPath

    if digest(inspect.getfile(type(layer))) != DECODER_SOURCE_SHA:
        raise RuntimeError("The original Decoder class source changed")
    runner = layer.mlp.experts
    if digest(inspect.getfile(type(runner))) != RUNNER_SOURCE_SHA:
        raise RuntimeError("The original Ascend runner source changed")
    if digest(inspect.getfile(moe_runner)) != LOOKUP_SOURCE_SHA:
        raise RuntimeError("The original MoE lookup source changed")
    moe = runner.moe_config
    if (moe.tp_size != 16 or moe.ep_size != 1 or moe.dp_size != 1
            or moe.pcp_size != 1 or moe.is_sequence_parallel
            or layer.use_sequence_parallel_moe
            or config.parallel_config.enable_eplb
            or runner.routed_input_transform is not None
            or runner.routed_output_transform is not None
            or runner._should_reduce_routed_before_combine()
            or isinstance(runner.router, ZeroExpertRouter)):
        raise ValueError("Require the untransformed TP16/EP1/DP1/PCP1 Decoder MoE")
    if (moe.hidden_dim != HIDDEN or moe.hidden_dim_unpadded != HIDDEN
            or moe.intermediate_size_per_partition != 128
            or moe.intermediate_size_per_partition_unpadded != 128
            or moe.num_experts != 256 or moe.num_local_experts != 256
            or moe.experts_per_token != 8 or moe.in_dtype != torch.bfloat16
            or moe.has_bias or moe.is_lora_enabled or moe.has_hash_routing
            or moe.skip_final_all_reduce or runner.routed_scaling_factor != 2.5
            or runner.enable_dbo or runner.shared_expert_gate is not None):
        raise ValueError("Require the exact BF16 H6144/I128/E256/K8 MoE shard")
    expected = {"hidden_size": HIDDEN, "n_routed_experts": 256,
                "num_experts_per_tok": 8, "n_group": 1, "topk_group": 1,
                "routed_scaling_factor": 2.5, "scoring_func": "sigmoid",
                "norm_topk_prob": True}
    if any(getattr(config.model_config.hf_config, key) != value
           for key, value in expected.items()):
        raise ValueError("The native router geometry or arithmetic changed")
    norm = layer.post_attention_layernorm
    if (norm.bias is not None or norm.weight.dtype != torch.bfloat16
            or tuple(norm.weight.shape) != (HIDDEN,)):
        raise ValueError("Require unbiased BF16 post-attention RMSNorm")
    shared = runner.ascend_shared_experts
    if (shared is None or shared.weights_replicated
            or shared.lora_context is not None
            or shared.situ_activation is not None or shared.swiglu_limit
            or shared.swiglu_alpha != 1 or shared.swiglu_beta != 0
            or getattr(shared.layer, "expert_gate", None) is not None
            or shared._select_mlp_path() is not SharedExpertMLPPath.A8_INT_FUSED
            or getattr(runner.routed_experts, "_ascend_moe_lora_context", None)
            is not None):
        raise ValueError("Require the original ungated A8_INT_FUSED shared expert")
    return runner


class LayerFFNAdapter:
    """Keep the original norm/MLP call sites and runtime-aware final reduction."""

    def __init__(self, torch, torch_npu, layer, config, native):
        from decoder_ffn import shared_weights, unpack_scale

        self.torch = torch
        self.torch_npu = torch_npu
        self.layer = layer
        self.native = native
        self.runner = validate_layer(layer, config)
        self.norm = layer.post_attention_layernorm
        self.original_norm = self.norm.forward
        self.original_mlp = layer.mlp.forward
        self.pending = PendingFFN()
        self.pool = LayerWorkspacePool(layer, native.prepare)
        self.hits = []
        self.enabled = False
        routed = self.runner.routed_experts
        self.device = routed.w13_weight.device
        scheme = routed.quant_method.quant_method
        if digest(inspect.getfile(type(scheme))) != SCALE_SOURCE_SHA:
            raise RuntimeError("The original packed scale encoding changed")
        scales = [self.nd(unpack_scale(torch, getattr(routed, name), columns))
                  for name, columns in (("w13_weight_scale", 256),
                                        ("w2_weight_scale", HIDDEN))]
        self.weights = [routed.w13_weight, routed.w13_scale_bias, scales[0],
                        routed.w2_weight, routed.w2_scale_bias, scales[1]]
        self.shared = shared_weights(torch, torch_npu,
                                     self.runner.ascend_shared_experts)
        self.gamma = self.norm.weight.detach()
        self.gate = self.nd(torch_npu.npu_format_cast(
            layer.mlp.gate.weight.detach(), 2).t())
        self.bias = layer.mlp.gate.e_score_correction_bias.detach()
        self.epsilon = self.norm.variance_epsilon
        self.packed_weight_pointers = [self.weights[i].data_ptr() for i in (0, 3)]

    def nd(self, value):
        return self.torch_npu.npu_format_cast(value.to(self.device).contiguous(), 2)

    def check_runtime(self):
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        from vllm_ascend.ops.fused_moe.moe_comm_method import MoECommType

        if (_EXTRA_CTX.moe_comm_type != MoECommType.ALLGATHER
                or self.runner._fused_output_is_reduced):
            raise RuntimeError("The complete FFN requires an unreduced AllGather local body")

    def post_norm(self, unused_module, hidden, residual=None):
        self.pending.require_empty()
        self.check_runtime()
        if (residual is None or hidden.dtype != self.torch.bfloat16
                or residual.dtype != self.torch.bfloat16
                or hidden.device != self.device or residual.device != self.device
                or hidden.shape != residual.shape or hidden.ndim != 2
                or hidden.shape[1] != HIDDEN or not 1 <= hidden.shape[0] <= 32):
            raise ValueError("Require post-AR1 BF16 [M,6144] and its retained residual")
        hidden, residual = self.nd(hidden), self.nd(residual)
        scratch, key = self.pool.get(hidden)
        self.native.forward_out(
            hidden, residual, self.gamma, self.gate, self.bias,
            self.weights, self.shared, scratch.bridge, scratch.body,
            self.epsilon, BLOCKS, LAYOUT_NZ, scratch.sync)
        self.hits.append({"phase": "complete_ffn", "stream_key": key})
        return self.pending.begin(scratch, key)

    def mlp(self, unused_module, normalized, already_sequence_parallel=False):
        from vllm.forward_context import get_forward_context

        if already_sequence_parallel:
            raise ValueError("Sequence-parallel MLP calls are outside this fixture")
        device, stream, rows, _ = stream_state(normalized)
        scratch = self.pending.consume(normalized, (device, stream, rows))
        self.check_runtime()
        context = get_forward_context()
        cursor_before = context.moe_layer_index
        runner = lookup_runner(self.runner)
        cursor_after = context.moe_layer_index
        result = runner._maybe_reduce_final_output(
            scratch.bridge[10], None, False)
        result = runner._maybe_add_zero_expert_output(result)
        self.hits.append({"phase": "original_final_reduce",
                          "stream_key": (device, stream, rows),
                          "layer_name": runner.layer_name,
                          "lookup_same_runner": runner is self.runner,
                          "moe_layer_index_before": cursor_before,
                          "moe_layer_index_after": cursor_after})
        return result

    def enable(self):
        self.pending.require_empty()
        self.norm.forward = MethodType(self.post_norm, self.norm)
        self.layer.mlp.forward = MethodType(self.mlp, self.layer.mlp)
        self.enabled = True

    def disable(self):
        self.pending.require_empty()
        self.norm.forward = self.original_norm
        self.layer.mlp.forward = self.original_mlp
        self.enabled = False

    def restore(self):
        self.norm.forward = self.original_norm
        self.layer.mlp.forward = self.original_mlp
        self.pending.value = None
        self.enabled = False
