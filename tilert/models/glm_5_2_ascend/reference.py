"""CPU HF decoder with independent ModelSlim and TP16 arithmetic adapters."""

import json
import math
from pathlib import Path
from types import FunctionType, MethodType

import torch
from torch import nn
import torch.nn.functional as functional
from torch.overrides import TorchFunctionMode
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaRotaryEmbedding, eager_attention_forward,
)

from .reference_weights import ReferenceWeights, quantize_a8, tp_sum
from .ops.indexer import (
    CANN_ARCH22_SOURCE_DIRECTORY,
    cann_arch22_index_scores,
    cann_arch22_input_domain,
    cann_arch22_scoring_contract,
)


TP_SIZE = 16
LAYERS = 78
ROTARY_CONTRACT = {
    "intermediates": "FP32 products and sum; one BF16 output cast",
    "angles": "HF FP32 generation followed by BF16 cast",
    "main_basis": "real half, imaginary half",
    "index_basis": "alternating real and imaginary",
    "cann_source_sha256": (
        "96028ea9c3f2b5426b60c00e03951fafbc"
        "d4903ceff5880b4ad31482d37e0a8a"
    ),
    "hf_adapter": "Instance forward globals; HF bytecode unchanged",
}


class StableTopk(TorchFunctionMode):
    """Resolve equal router scores by ascending expert ID."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func not in (torch.topk, torch.Tensor.topk):
            return func(*args, **kwargs)
        scores = args[0] if args else kwargs["input"]
        count = args[1] if len(args) > 1 else kwargs["k"]
        dimension = kwargs.get("dim", args[2] if len(args) > 2 else -1)
        largest = kwargs.get("largest", args[3] if len(args) > 3 else True)
        indices = scores.argsort(
            dim=dimension, descending=largest, stable=True
        ).narrow(dimension, 0, count)
        return torch.return_types.topk((
            scores.gather(dimension, indices), indices,
        ))


class ReferenceLinear(nn.Module):
    def __init__(self, weights, prefix, partition="replicated"):
        super().__init__()
        self.weights = weights
        self.prefix = prefix
        self.partition = partition

    def forward(self, x):
        return self.weights.linear(self.prefix, x, partition=self.partition)


class ReferenceEmbedding(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, tokens):
        if tokens.shape != (1, 1):
            raise ValueError("The reference consumes one token at a time")
        token = int(tokens.item())
        row = self.weights.tensor(
            "model.embed_tokens.weight", rows=(token, token + 1),
            dtype=torch.bfloat16,
        )
        return row.unsqueeze(0)


def fused_swiglu(gate, up):
    """Round the FP32 activation/product once at the BF16 FFN boundary."""
    return (functional.silu(gate.float()) * up.float()).to(torch.bfloat16)


def ffn_partials(weights, prefix, hidden):
    gate = weights.linear(prefix + ".gate_proj", hidden, partition="output")
    up = weights.linear(prefix + ".up_proj", hidden, partition="output")
    activated = fused_swiglu(gate, up)
    return weights.input_partials(prefix + ".down_proj", activated)


class ReferenceMlp(nn.Module):
    def __init__(self, weights, prefix):
        super().__init__()
        self.weights = weights
        self.prefix = prefix

    def forward(self, hidden):
        return tp_sum(ffn_partials(self.weights, self.prefix, hidden))


class ReferenceMoe(nn.Module):
    """Retain HF routing and combine experts before each rank's TP sum."""

    def __init__(self, weights, prefix, moe):
        super().__init__()
        self.weights = weights
        self.prefix = prefix
        self.gate = moe.gate
        self.route = getattr(type(moe), "route_tokens_to_experts", None)
        if self.route is not None:
            for name in ("n_routed_experts", "n_group", "topk_group",
                         "norm_topk_prob", "top_k"):
                setattr(self, name, getattr(moe, name))
        self.output_scale = 1.0
        self.fused_quantization = False
        self.last_route = None

    @property
    def routed_scaling_factor(self):
        return self.gate.routed_scaling_factor

    def forward(self, hidden):
        with StableTopk():
            router_output = self.gate(hidden)
            if self.route is None:
                _, probabilities, choices = router_output
            else:
                choices, probabilities = self.route(self, router_output)
        self.last_route = choices[0].detach().clone()
        probabilities = probabilities[0].to(torch.bfloat16)
        flat = hidden.reshape(1, -1)
        expert_partials = []
        for expert in choices[0].tolist():
            prefix = f"{self.prefix}.experts.{expert}"
            values = (self.weights.fused_expert_partials(prefix, flat)
                      if self.fused_quantization
                      else ffn_partials(self.weights, prefix, flat))
            expert_partials.append(values)
        shared = ffn_partials(
            self.weights, self.prefix + ".shared_experts", flat
        )
        rank_outputs = []
        for rank in range(TP_SIZE):
            products = [
                values[rank].float() * probability.float()
                for values, probability in zip(expert_partials, probabilities)
            ]
            combined = torch.stack(products).sum(dim=0).to(torch.bfloat16)
            if self.output_scale != 1.0:
                combined = combined * self.output_scale
            rank_outputs.append(combined + shared[rank])
        return tp_sum(rank_outputs).reshape_as(hidden)


def rotary_components(x, cosine, sine, head_axis):
    """Apply CANN's FP32 rotary intermediates to BF16 inputs and angles."""
    half = x.shape[-1] // 2
    cosine = cosine[..., :half].float().unsqueeze(head_axis)
    sine = sine[..., :half].float().unsqueeze(head_axis)
    even, odd = x[..., 0::2].float(), x[..., 1::2].float()
    return even * cosine + (-odd) * sine, odd * cosine + even * sine


def interleaved_rope(x, cosine, sine):
    """Round once to BF16 and retain alternating indexer coordinates."""
    first, second = rotary_components(x, cosine, sine, head_axis=2)
    return torch.stack((first, second), dim=-1).flatten(-2).to(x.dtype)


def attention_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Retain HF's half-basis layout with CANN's rotary rounding boundary."""
    query = rotary_components(q, cos, sin, unsqueeze_dim)
    key = rotary_components(k, cos, sin, unsqueeze_dim)
    return (torch.cat(query, dim=-1).to(q.dtype),
            torch.cat(key, dim=-1).to(k.dtype))


def install_attention_rope(attention):
    """Change only this instance's HF forward rotary function lookup."""
    original = attention.forward.__func__
    namespace = dict(original.__globals__)
    namespace["apply_rotary_pos_emb_interleave"] = attention_rope
    adapted = FunctionType(original.__code__, namespace, original.__name__,
                           original.__defaults__, original.__closure__)
    adapted.__kwdefaults__ = original.__kwdefaults__
    attention.forward = MethodType(adapted, attention)


def prolog_attention_forward(
    self, hidden_states, position_embeddings, attention_mask,
    past_key_values=None, position_ids=None, prev_topk_indices=None, **kwargs,
):
    """Use CPU Prolog arithmetic before HF cache, attention and projection."""
    from .reference_prolog import preprocess

    batch, length, hidden = hidden_states.shape
    cosine, sine = position_embeddings
    prepared = preprocess(
        self._reference_weights, self._reference_prefix,
        hidden_states.reshape(-1, hidden),
        cosine.reshape(-1, cosine.shape[-1]),
        sine.reshape(-1, sine.shape[-1]),
    )
    self._prolog = prepared
    query = torch.cat((prepared["q_nope"], prepared["q_rope"]), dim=-1)
    query = query.view(batch, length, self.num_heads, -1).transpose(1, 2)
    expanded = self.kv_b_proj(prepared["latent"].view(batch, length, -1))
    expanded = expanded.view(batch, length, self.num_heads, -1).transpose(1, 2)
    key, value = expanded.split((self.qk_nope_head_dim, self.v_head_dim), -1)
    rope = prepared["k_rope"].view(batch, 1, length, self.qk_rope_head_dim)
    key = torch.cat((key, rope.expand(*key.shape[:-1], -1)), dim=-1)
    if past_key_values is not None:
        key, value = past_key_values.update(key, value, self.layer_idx)
    if self.indexer is not None:
        mask = attention_mask[:, 0] if attention_mask is not None else None
        indices = self.indexer(
            hidden_states, prepared["q_norm"].view(batch, length, -1),
            position_embeddings, mask, position_ids,
            past_key_values=past_key_values,
        )
    else:
        if prev_topk_indices is None:
            raise ValueError("Shared DSA requires the preceding index choices")
        indices = prev_topk_indices
    excluded = indices.new_ones((batch, length, key.shape[2]),
                                dtype=torch.bool)
    excluded.scatter_(-1, indices.long(), False)
    excluded = excluded.unsqueeze(1)
    if attention_mask is None:
        positions = torch.arange(key.shape[2], device=hidden_states.device)
        excluded = excluded | (positions[None, None, None]
                               > position_ids[:, None, :, None])
        attention_mask = hidden_states.new_zeros((batch, 1, length,
                                                  key.shape[2]))
    attention_mask = attention_mask.masked_fill(
        excluded, torch.finfo(hidden_states.dtype).min,
    )
    output, weights = eager_attention_forward(
        self, query, key, value, attention_mask, dropout=0.0,
        scaling=self.scaling, **kwargs,
    )
    output = output.reshape(batch, length, -1).contiguous()
    return self.o_proj(output), weights, indices


def prolog_input_norm(self, hidden):
    """Keep FP32 normalized values through gamma before the BF16 output."""
    values = hidden.float()
    variance = values.square().mean(-1, keepdim=True)
    normalized = values * torch.rsqrt(variance + self.variance_epsilon)
    return (normalized * self.weight.float()).to(hidden.dtype)


class PrologIndexerQuery(nn.Module):
    def __init__(self, weights, prefix):
        super().__init__()
        self.weights = weights
        self.prefix = prefix

    def forward(self, normalized):
        from .reference_prolog import index_query

        return index_query(self.weights, self.prefix,
                           normalized.reshape(-1, normalized.shape[-1]))


def hadamard_matrix():
    signs = [
        [1 if (row & column).bit_count() % 2 == 0 else -1
         for column in range(128)]
        for row in range(128)
    ]
    return torch.tensor(signs, dtype=torch.bfloat16) / math.sqrt(128)


def prefix_length(prefixes):
    """Validate the replicated backbone cache before changing HF state."""
    if len(prefixes) != LAYERS:
        raise ValueError("The reference needs 78 backbone prefixes")
    length = prefixes[0]["latent"].shape[0]
    for layer, entry in enumerate(prefixes):
        expected = {
            "latent": ((length, 512), torch.bfloat16),
            "rope": ((length, 64), torch.bfloat16),
        }
        full = layer < 3 or (layer - 2) % 4 == 0
        if full:
            quantized = 6 <= layer <= 66
            dtype = torch.int8 if quantized else torch.bfloat16
            expected["index_keys"] = (length, 128), dtype
            if quantized:
                expected["index_scales"] = (length,), torch.float16
        if entry.keys() != expected.keys():
            raise ValueError(f"Wrong prefix regions at layer {layer}")
        for name, (shape, dtype) in expected.items():
            value = entry[name]
            if (tuple(value.shape) != shape or value.dtype != dtype
                    or value.device.type != "cpu"):
                raise ValueError(f"Invalid prefix {layer}.{name}")
    return length


class ReferenceIndexer(nn.Module):
    """Independent CPU scoring over BF16 or C8 keys in HF DynamicCache."""

    def __init__(self, weights, prefix, layer, quantized):
        super().__init__()
        self.weights = weights
        self.prefix = prefix
        self.layer_idx = layer
        self.quantized = quantized
        self.rotation = hadamard_matrix() if quantized else None
        self.scales = None
        self.last_key = None
        self.last_scale = None
        self.last_scores = None
        self.query_projection = ReferenceLinear(weights, prefix + ".wq_b")

    def reset(self):
        self.scales = None
        self.last_key = None
        self.last_scale = None
        self.last_scores = None

    def encode(self, x):
        if not self.quantized:
            return x, None
        encoded, scale = quantize_a8(x @ self.rotation)
        return encoded, scale.to(torch.float16)

    def inject(self, cache, keys, scales):
        expected = torch.int8 if self.quantized else torch.bfloat16
        if keys.dtype != expected or keys.ndim != 2 or keys.shape[1] != 128:
            raise ValueError("Invalid encoded index prefix")
        cache.update_indexer(keys.unsqueeze(0), self.layer_idx)
        if self.quantized:
            if (scales is None or scales.dtype != torch.float16
                    or scales.numel() != keys.shape[0]):
                raise ValueError("C8 prefix needs one FP16 scale per key")
            self.scales = scales.reshape(-1).clone()

    def forward(
        self, hidden_states, q_resid, position_embeddings, attention_mask,
        position_ids, past_key_values=None,
    ):
        if hidden_states.shape[:2] != (1, 1) or past_key_values is None:
            raise ValueError("The reference indexer requires cached M=1")
        query = self.query_projection(q_resid)
        query = query.reshape(1, 1, 32, 128)
        key = self.weights.linear(
            self.prefix + ".wk", hidden_states, partition="replicated"
        )
        gamma = self.weights.tensor(self.prefix + ".k_norm.weight")
        beta = self.weights.tensor(self.prefix + ".k_norm.bias")
        key = functional.layer_norm(
            key.float(), (128,), gamma.float(), beta.float(), eps=1e-6
        ).to(torch.bfloat16).unsqueeze(2)
        cosine, sine = position_embeddings
        query = torch.cat((
            interleaved_rope(query[..., :64], cosine, sine), query[..., 64:]
        ), dim=-1).reshape(32, 128)
        key = torch.cat((
            interleaved_rope(key[..., :64], cosine, sine), key[..., 64:]
        ), dim=-1).reshape(1, 128)
        encoded_key, key_scale = self.encode(key)
        keys = past_key_values.update_indexer(
            encoded_key.unsqueeze(0), self.layer_idx
        )[0]
        self.last_key = encoded_key[0].clone()
        self.last_scale = None if key_scale is None else key_scale.reshape(1)
        query, query_scale = self.encode(query)
        weights = self.weights.linear(
            self.prefix + ".weights_proj", hidden_states,
            partition="replicated",
        ).reshape(32)
        if self.quantized:
            scale = key_scale.reshape(-1)
            self.scales = (scale if self.scales is None else
                           torch.cat((self.scales, scale)))
            rounded_weights = weights.to(torch.float16)
            domain = cann_arch22_input_domain(
                query_scale.reshape(32).tolist(), self.scales.tolist(),
                rounded_weights.tolist(),
            )
            if not domain["accepted"]:
                raise ValueError(
                    "Unverified CANN FP16 scoring boundary: "
                    + json.dumps(domain, sort_keys=True)
                )
            scores = cann_arch22_index_scores(
                query, keys, query_scale.reshape(32), self.scales,
                rounded_weights,
            )
        else:
            scores = query.float() @ keys.float().T
            scores = (scores.relu() * weights.float()[:, None]).sum(dim=0)
        if attention_mask is not None:
            scores = scores + attention_mask.reshape(-1).float()
        else:
            scores = scores.masked_fill(
                torch.arange(keys.shape[0]) > position_ids.item(), -torch.inf
            )
        self.last_scores = scores.detach().clone()
        indices = scores.argsort(descending=True, stable=True)[:2048]
        return indices.to(torch.int32).reshape(1, 1, -1)


class HfReference:
    """Run the installed HF model without CANN or production weight helpers."""

    def __init__(self, checkpoint, mla_preprocess="native",
                 layer_count=LAYERS, *, fp32_head=False,
                 c8_source_directory=None):
        if mla_preprocess not in ("native", "prolog_v3"):
            raise ValueError("MLA preprocess must be native or prolog_v3")
        if type(layer_count) is not int or not 1 <= layer_count <= LAYERS:
            raise ValueError("The reference layer count must be in [1, 78]")
        self.mla_preprocess = mla_preprocess
        self.weights = ReferenceWeights(checkpoint, fp32_head=fp32_head)
        self.weights.contract["rotary"] = dict(ROTARY_CONTRACT)
        if c8_source_directory is None:
            c8_source_directory = CANN_ARCH22_SOURCE_DIRECTORY
        self.weights.contract["c8_index_scoring"] = (
            cann_arch22_scoring_contract(Path(c8_source_directory).resolve())
        )
        raw = json.loads((self.weights.directory / "config.json").read_text())
        config = GlmMoeDsaConfig.from_dict(raw)
        self.weights.contract["lm_head"] = {
            "input_dtype": "bfloat16", "weight_dtype": "bfloat16",
            "accumulator_dtype": "float32",
            "projection_output_dtype": "float32" if fp32_head else "bfloat16",
            "tensor_parallel_size": TP_SIZE,
            "vocabulary_range": [0, config.vocab_size],
        }
        config._attn_implementation = "eager"
        config._experts_implementation = "eager"
        if (config.num_hidden_layers != LAYERS or config.n_group != 1
                or config.topk_group != 1 or config.n_shared_experts != 1
                or config.rms_norm_eps != 1e-5):
            raise ValueError("The reference requires the GLM-5.2 TP16 layout")
        config.num_hidden_layers = layer_count
        config.layer_types = config.layer_types[:layer_count]
        with torch.device("meta"):
            self.model = GlmMoeDsaForCausalLM(config).to(torch.bfloat16)
        self.config = config
        self._install_adapters()
        if mla_preprocess == "prolog_v3":
            self._install_prolog()
        with torch.device("cpu"):
            self.model.model.rotary_emb = GlmMoeDsaRotaryEmbedding(config=config)
        self.model.eval().requires_grad_(False)
        remaining = [name for name, value in (
            *self.model.named_parameters(), *self.model.named_buffers()
        ) if value.is_meta]
        if remaining:
            raise ValueError(f"Unreplaced HF meta tensors: {remaining}")
        self._attach_trace_hooks()
        self.reset()

    def _install_prolog(self):
        from .reference_prolog import CONTRACT

        self.weights.contract["mla_preprocess"] = {
            **CONTRACT,
            "input_norm": "Layer0 FP32 norm and gamma; one BF16 output",
        }
        self.weights.contract["expert_combine"] = (
            "Unscaled BF16 probabilities; FP32 weighted sum; BF16 output; "
            "BF16 routed scale; BF16 shared add; FP32 TP sum; BF16 output"
        )
        self.weights.contract["fused_expert_quantization"] = (
            "FP32 GMM1 and SwiGLU; scale=amax/127; reciprocal scale; "
            "FP16 then INT8 nearest-even; BF16 GMM2"
        )
        norm = self.model.model.layers[0].input_layernorm
        norm.forward = MethodType(prolog_input_norm, norm)
        for number, layer in enumerate(self.model.model.layers):
            if isinstance(layer.mlp, ReferenceMoe):
                layer.mlp.output_scale = layer.mlp.gate.routed_scaling_factor
                layer.mlp.gate.routed_scaling_factor = 1.0
                layer.mlp.fused_quantization = True
            attention = layer.self_attn
            prefix = f"model.layers.{number}.self_attn"
            attention._reference_weights = self.weights
            attention._reference_prefix = prefix
            attention.forward = MethodType(prolog_attention_forward, attention)
            if attention.indexer is not None:
                attention.indexer.query_projection = PrologIndexerQuery(
                    self.weights, prefix + ".indexer",
                )

    def _norm(self, module, prefix):
        module.weight = nn.Parameter(self.weights.tensor(
            prefix + ".weight", dtype=torch.bfloat16
        ), requires_grad=False)
        module.variance_epsilon = 1e-5

    def _install_adapters(self):
        self.model.model.embed_tokens = ReferenceEmbedding(self.weights)
        self.model.lm_head = ReferenceLinear(
            self.weights, "lm_head", partition="output"
        )
        self._norm(self.model.model.norm, "model.norm")
        for number, layer in enumerate(self.model.model.layers):
            install_attention_rope(layer.self_attn)
            prefix = f"model.layers.{number}"
            attention = prefix + ".self_attn"
            self._norm(layer.input_layernorm, prefix + ".input_layernorm")
            self._norm(layer.post_attention_layernorm,
                       prefix + ".post_attention_layernorm")
            self._norm(layer.self_attn.q_a_layernorm,
                       attention + ".q_a_layernorm")
            self._norm(layer.self_attn.kv_a_layernorm,
                       attention + ".kv_a_layernorm")
            for name, partition in (
                ("q_a_proj", "replicated"), ("q_b_proj", "output"),
                ("kv_a_proj_with_mqa", "replicated"),
                ("kv_b_proj", "output"), ("o_proj", "input"),
            ):
                setattr(layer.self_attn, name, ReferenceLinear(
                    self.weights, attention + "." + name, partition
                ))
            if layer.self_attn.indexer is not None:
                index = attention + ".indexer"
                recipe = self.weights.quantization.get(index + ".quant_type")
                layer.self_attn.indexer = ReferenceIndexer(
                    self.weights, index, number, recipe == "INT8_DYNAMIC"
                )
            mlp = prefix + ".mlp"
            if hasattr(layer.mlp, "gate"):
                gate = layer.mlp.gate
                gate.weight = nn.Parameter(self.weights.tensor(
                    mlp + ".gate.weight", dtype=torch.float32
                ), requires_grad=False)
                gate.e_score_correction_bias = self.weights.tensor(
                    mlp + ".gate.e_score_correction_bias", dtype=torch.float32
                )
                layer.mlp = ReferenceMoe(self.weights, mlp, layer.mlp)
            else:
                layer.mlp = ReferenceMlp(self.weights, mlp)

    def _attach_trace_hooks(self):
        def record_last_merged(module, inputs):
            if inputs[0].ndim == 3:
                self.last_merged_hidden = inputs[0][:, -1].detach().clone()

        self.model.model.norm.register_forward_pre_hook(record_last_merged)
        for number, layer in enumerate(self.model.model.layers):
            def record_attention(module, inputs, output, index=number):
                self.indices[index] = output[2][0, -1].detach().clone()
                self.attention_weights[index] = (
                    output[1][0, :, -1, :].detach().clone()
                )
                if module.indexer is not None:
                    self.index_scores[index] = module.indexer.last_scores
                if self.mla_preprocess == "prolog_v3":
                    self.latents[index] = module._prolog["latent"][0].clone()

            def record_latent(module, inputs, output, index=number):
                self.latents[index] = output[0, -1].detach().clone()

            def record_route(module, inputs, output, index=number):
                self.routes[index] = module.last_route.clone()

            layer.self_attn.register_forward_hook(record_attention)
            layer.self_attn.kv_a_layernorm.register_forward_hook(record_latent)
            if hasattr(layer.mlp, "gate"):
                layer.mlp.register_forward_hook(record_route)

    def reset(self):
        self.cache = DynamicCache(config=self.config)
        for layer in self.model.model.layers:
            if layer.self_attn.indexer is not None:
                layer.self_attn.indexer.reset()

    @property
    def position(self):
        return self.cache.get_seq_length()

    @torch.backends.mkldnn.flags(
        enabled=False, deterministic=None, allow_tf32=None,
        fp32_precision=None,
    )
    def inject(self, prefixes):
        prefix_length(prefixes)
        if self.position:
            raise ValueError("Inject prefixes into an empty reference")
        for number, (layer, entry) in enumerate(zip(
            self.model.model.layers, prefixes
        )):
            latent, rope = entry["latent"], entry["rope"]
            expanded = layer.self_attn.kv_b_proj(latent)
            expanded = expanded.reshape(-1, 64, 448)
            key, value = expanded.split((192, 256), dim=-1)
            rope = rope[:, None, :].expand(-1, 64, 64)
            key = torch.cat((key, rope), dim=-1).transpose(0, 1).unsqueeze(0)
            value = value.transpose(0, 1).unsqueeze(0)
            self.cache.update(key.contiguous(), value.contiguous(), number)
            if layer.self_attn.indexer is not None:
                layer.self_attn.indexer.inject(
                    self.cache, entry["index_keys"], entry.get("index_scales")
                )

    @torch.inference_mode()
    @torch.backends.mkldnn.flags(
        enabled=False, deterministic=None, allow_tf32=None,
        fp32_precision=None,
    )
    def step(self, token):
        layer_count = len(self.model.model.layers)
        self.indices, self.routes, self.latents = (
            [None] * layer_count for _ in range(3)
        )
        self.index_scores = [None] * layer_count
        self.attention_weights = [None] * layer_count
        self.last_merged_hidden = None
        position = self.position
        output = self.model(
            input_ids=torch.tensor([[token]]), past_key_values=self.cache,
            use_cache=True, output_hidden_states=True, return_dict=True,
        )
        self.cache = output.past_key_values
        hidden = tuple(value[:, -1].clone() for value in output.hidden_states)
        if self.last_merged_hidden is None:
            raise ValueError("HF final norm did not expose its merged input")
        logits = output.logits[:, -1].clone()
        layer_logits = (logits,)
        projected_hidden = hidden[1:-1]
        if layer_count < LAYERS:
            projected_hidden = (*projected_hidden, self.last_merged_hidden)
        if projected_hidden:
            intermediate = torch.cat(projected_hidden)
            normalized = (
                prolog_input_norm(self.model.model.norm, intermediate)
                if self.mla_preprocess == "prolog_v3"
                else self.model.model.norm(intermediate)
            )
            projected = self.model.lm_head(normalized)
            layer_logits = tuple(projected.split(1, dim=0))
            if layer_count == LAYERS:
                layer_logits = (*layer_logits, logits)
        rows = []
        for number, layer in enumerate(self.model.model.layers):
            indexer = layer.self_attn.indexer
            rows.append({
                "latent": self.latents[number],
                "rope": self.cache.layers[number].keys[0, 0, -1, 192:].clone(),
                "index": None if indexer is None else indexer.last_key.clone(),
                "index_scale": (
                    None if indexer is None or not indexer.quantized
                    else indexer.last_scale.clone()
                ),
            })
        return {
            "token": token, "position": position,
            "committed_position": self.position, "logits": logits,
            "hidden_states": hidden, "layer_logits": layer_logits,
            "last_merged_hidden": self.last_merged_hidden,
            "index_scores": tuple(self.index_scores),
            "attention_weights": tuple(self.attention_weights),
            "indices": tuple(self.indices), "routes": tuple(self.routes),
            "cache_rows": tuple(rows),
        }
