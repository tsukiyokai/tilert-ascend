"""Incremental BF16 GLM-5.2 execution with a fixed-capacity KV cache."""

from dataclasses import dataclass
from typing import Mapping

import torch
from torch.nn import functional as F

from .config import DecodeConfig
from .ops import rms_norm, rms_norm_reference


@dataclass
class LayerCache:
    latent: torch.Tensor
    rope: torch.Tensor
    projected: torch.Tensor
    index_keys: torch.Tensor | None
    length: int = 0


@dataclass(frozen=True)
class StepResult:
    logits: torch.Tensor | None
    hidden_states: tuple[torch.Tensor, ...] = ()
    indices: tuple[torch.Tensor, ...] = ()
    routes: tuple[torch.Tensor | None, ...] = ()
    last_hidden: torch.Tensor | None = None
    token_ids: torch.Tensor | None = None


class DecodeEngine:
    """Single-device, batch-one decoder consuming one token per step.

    Weights use the unquantized HF state-dict layout. NPU normalization uses
    TileLang Ascend; projections, indexing and attention use Torch/NPU ops.
    This execution path establishes cache and layer semantics before W4A8C8
    loading, tensor parallelism, and fused tile scheduling.
    """

    def __init__(
        self,
        config: DecodeConfig,
        state_dict: Mapping[str, torch.Tensor],
        device: str | torch.device = "npu:0",
        capacity: int = 9216,
    ):
        self.config = config
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "npu"):
            raise ValueError("DecodeEngine supports cpu and npu devices")
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.position = 0
        self.norm_calls = 0
        self.weights = {
            name: tensor.detach().to(
                device=self.device,
                dtype=torch.float32 if name.endswith("e_score_correction_bias") else torch.bfloat16,
            ).contiguous()
            for name, tensor in state_dict.items()
        }
        self.norm_weights = {
            name: weight.float()
            for name, weight in self.weights.items()
            if name.endswith("layernorm.weight") or name == "model.norm.weight"
        }
        self.cache = [self._allocate_cache(layer) for layer in range(config.num_hidden_layers)]
        # Host construction fixes identical RoPE tables on CPU and NPU.
        frequencies = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.qk_rope_head_dim, 2).float() / config.qk_rope_head_dim)
        )
        angles = torch.arange(capacity).float().unsqueeze(1) * frequencies
        self.cos = angles.cos().to(device=self.device, dtype=torch.bfloat16)
        self.sin = angles.sin().to(device=self.device, dtype=torch.bfloat16)

    def _allocate_cache(self, layer: int) -> LayerCache:
        config = self.config
        options = {"device": self.device, "dtype": torch.bfloat16}
        return LayerCache(
            latent=torch.empty((self.capacity, config.kv_lora_rank), **options),
            rope=torch.empty((self.capacity, config.qk_rope_head_dim), **options),
            projected=torch.empty(
                (self.capacity, config.num_attention_heads,
                 config.qk_nope_head_dim + config.v_head_dim), **options
            ),
            index_keys=(
                torch.empty((self.capacity, config.index_head_dim), **options)
                if config.has_indexer(layer)
                else None
            ),
        )

    def reset(self) -> None:
        """Start a request; cache entries beyond length are never read."""
        self.position = 0
        for cache in self.cache:
            cache.length = 0

    def _linear(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return F.linear(x, self.weights[name + ".weight"])

    def _norm(self, x: torch.Tensor, name: str, eps: float) -> torch.Tensor:
        self.norm_calls += 1
        operation = rms_norm if self.device.type == "npu" else rms_norm_reference
        return operation(x.contiguous(), self.norm_weights[name + ".weight"], eps)

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        even, odd = x[..., 0::2], x[..., 1::2]
        cos, sin = self.cos[self.position], self.sin[self.position]
        return torch.cat((even * cos - odd * sin, odd * cos + even * sin), dim=-1)

    def _select_indices(
        self, x: torch.Tensor, q_latent: torch.Tensor, layer: int
    ) -> torch.Tensor:
        config = self.config
        prefix = f"model.layers.{layer}.self_attn.indexer"
        query = self._linear(q_latent, prefix + ".wq_b").view(
            config.index_n_heads, config.index_head_dim
        )
        key = self._linear(x, prefix + ".wk")
        key = F.layer_norm(
            key,
            (config.index_head_dim,),
            self.weights[prefix + ".k_norm.weight"],
            self.weights[prefix + ".k_norm.bias"],
            eps=1e-6,
        )
        rope_dim = config.qk_rope_head_dim
        query = torch.cat((self._rope(query[:, :rope_dim]), query[:, rope_dim:]), dim=-1)
        key = torch.cat((self._rope(key[:, :rope_dim]), key[:, rope_dim:]), dim=-1)
        keys = self.cache[layer].index_keys
        keys[self.position : self.position + 1].copy_(key)
        scores = F.relu(query.float() @ keys[: self.position + 1].float().T)
        scores = scores * (config.index_head_dim**-0.5)
        head_weights = self._linear(x, prefix + ".weights_proj").float()
        head_weights = head_weights * (config.index_n_heads**-0.5)
        scores = (scores * head_weights.T).sum(dim=0)
        # Earlier context positions win exact score ties on every device.
        return scores.argsort(descending=True, stable=True)[: config.index_topk]

    def expanded_cache(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the valid cache in the HF [1, heads, length, dim] layout."""
        config = self.config
        cache = self.cache[layer]
        projected = cache.projected[: cache.length]
        key_nope, value = projected.split(
            (config.qk_nope_head_dim, config.v_head_dim), dim=-1
        )
        key_rope = cache.rope[: cache.length, None, :].expand(
            -1, config.num_attention_heads, -1
        )
        key = torch.cat((key_nope, key_rope), dim=-1)
        return key.permute(1, 0, 2).unsqueeze(0), value.permute(1, 0, 2).unsqueeze(0)

    def _attention(
        self, x: torch.Tensor, layer: int, previous_indices: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        prefix = f"model.layers.{layer}.self_attn"
        q_latent = self._norm(
            self._linear(x, prefix + ".q_a_proj"), prefix + ".q_a_layernorm", 1e-6
        )
        query = self._linear(q_latent, prefix + ".q_b_proj").view(config.num_attention_heads, -1)
        query_nope, query_rope = query.split(
            (config.qk_nope_head_dim, config.qk_rope_head_dim), dim=-1
        )
        compressed = self._linear(x, prefix + ".kv_a_proj_with_mqa")
        latent, key_rope = compressed.split(
            (config.kv_lora_rank, config.qk_rope_head_dim), dim=-1
        )
        latent = self._norm(latent, prefix + ".kv_a_layernorm", 1e-6)
        cache = self.cache[layer]
        cache.latent[self.position : self.position + 1].copy_(latent)
        cache.rope[self.position : self.position + 1].copy_(self._rope(key_rope))
        # Preserve the per-token BF16 rounding of kvB across cache lengths.
        # A growing GEMM can round differently from the original one-row projection.
        projected = self._linear(latent, prefix + ".kv_b_proj")
        cache.projected[self.position].copy_(projected.view(config.num_attention_heads, -1))
        cache.length = self.position + 1
        indices = (
            self._select_indices(x, q_latent, layer)
            if config.has_indexer(layer)
            else previous_indices
        )
        query = torch.cat((query_nope, self._rope(query_rope)), dim=-1)[None, :, None, :]
        key, value = self.expanded_cache(layer)
        scale = (config.qk_nope_head_dim + config.qk_rope_head_dim) ** -0.5
        scores = (query @ key.transpose(-1, -2)) * scale
        mask = torch.full(
            (cache.length,), torch.finfo(scores.dtype).min,
            dtype=scores.dtype, device=self.device,
        )
        mask[indices] = 0
        probabilities = torch.softmax(scores + mask, dim=-1, dtype=torch.float32).to(query.dtype)
        attended = (probabilities @ value).reshape(1, -1)
        return self._linear(attended, prefix + ".o_proj"), indices

    def _dense(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        gate = F.silu(self._linear(x, prefix + ".gate_proj"))
        return self._linear(gate * self._linear(x, prefix + ".up_proj"), prefix + ".down_proj")

    def _moe(self, x: torch.Tensor, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        prefix = f"model.layers.{layer}.mlp"
        scores = F.linear(x.float(), self.weights[prefix + ".gate.weight"].float()).sigmoid()
        choices = scores + self.weights[prefix + ".gate.e_score_correction_bias"].float()
        indices = choices.argsort(dim=-1, descending=True, stable=True)
        indices = indices[:, : config.num_experts_per_tok]
        routing = scores.gather(1, indices)
        routing = routing / (routing.sum(dim=-1, keepdim=True) + 1e-20)
        routing = routing * config.routed_scaling_factor
        ordered, permutation = indices[0].sort()
        routing = routing[0, permutation]
        gate_up = self.weights[prefix + ".experts.gate_up_proj"].index_select(0, ordered)
        down = self.weights[prefix + ".experts.down_proj"].index_select(0, ordered)
        routed = torch.zeros_like(x)
        for slot in range(config.num_experts_per_tok):
            gate, up = F.linear(x, gate_up[slot]).chunk(2, dim=-1)
            expert = F.linear(F.silu(gate) * up, down[slot])
            routed = routed + (expert * routing[slot : slot + 1].view(1, 1)).to(x.dtype)
        return routed + self._dense(x, prefix + ".shared_experts"), indices[0]

    @torch.inference_mode()
    def step(self, token: int, trace: bool = False) -> StepResult:
        """Consume token at position, append every layer's KV, predict its successor."""
        if not 0 <= token < self.config.vocab_size:
            raise ValueError("token is outside the vocabulary")
        if self.position >= self.capacity:
            raise ValueError("decode cache capacity exceeded")
        x = self.weights["model.embed_tokens.weight"][token : token + 1]
        hidden_states = [x] if trace else []
        index_trace, routes = [], []
        indices = None
        for layer in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            normalized = self._norm(x, prefix + ".input_layernorm", self.config.rms_norm_eps)
            attention, indices = self._attention(normalized, layer, indices)
            x = x + attention
            normalized = self._norm(
                x, prefix + ".post_attention_layernorm", self.config.rms_norm_eps
            )
            if layer < self.config.first_k_dense_replace:
                output, route = self._dense(normalized, prefix + ".mlp"), None
            else:
                output, route = self._moe(normalized, layer)
            x = x + output
            if trace:
                hidden_states.append(x)
                index_trace.append(indices)
                routes.append(route)
        x = self._norm(x, "model.norm", self.config.rms_norm_eps)
        if trace:
            hidden_states[-1] = x
        logits = self._linear(x, "lm_head")
        self.position += 1
        return StepResult(logits, tuple(hidden_states), tuple(index_trace), tuple(routes))
