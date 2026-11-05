"""Batch-one GLM-5.2 decode using existing CANN operators and TP16 HCCL."""

from dataclasses import dataclass, field, fields
import json

import torch
import torch.distributed as distributed
import torch.nn.functional as functional

from .cann_model import device_tokens
from .config import DecodeConfig
from .engine import StepResult
from .moe_plan import build_plan
from .ops.expert import NpuExpert
from .ops.indexer import NativeIndexer
from .ops.mla import NativeMla
from .ops.quantized import NpuQuantizedLinear
from .weights import ModelSlimCheckpoint, QuantizedWeight


TP_SIZE = 16


@dataclass
class NativeLayer:
    projections: dict
    norms: dict
    floating: dict
    mla: NativeMla
    indexer: NativeIndexer | None
    ffn: tuple | None = None
    shared: tuple | None = None
    experts: dict = field(default_factory=dict)
    expert_bank: object | None = None


class NativeDecodeEngine:
    """Consume 1, 2 or 4 causal rows on sixteen ranks using CANN operators.

    Attention and FFN outputs are reduced before the residual update. Expert
    partitions load on first use and remain resident. ``prepare_experts``
    loads every routed expert before profiling. This path uses no custom
    TileLang kernel. ``verify`` writes tentative rows; ``truncate`` commits
    an accepted prefix. Draft generation and acceptance live outside it.
    """

    def __init__(
        self, checkpoint, device, capacity=9216, group=None,
        validate_indices=True, use_expert_bank=False,
    ):
        import torch_npu  # noqa: F401

        if type(capacity) is not int or capacity < 1:
            raise ValueError("Decode cache capacity must be positive")
        self.validate_indices = bool(validate_indices)
        self.use_expert_bank = bool(use_expert_bank)
        self.device = torch.device(device)
        self.group = group
        if distributed.get_world_size(group) != TP_SIZE:
            raise ValueError("NativeDecodeEngine requires TP16")
        self.rank = distributed.get_rank(group)
        torch.npu.set_device(self.device)
        self.checkpoint = ModelSlimCheckpoint(checkpoint)
        config_path = self.checkpoint.directory / "config.json"
        raw = json.loads(config_path.read_text())
        names = {item.name for item in fields(DecodeConfig)}
        values = {name: value for name, value in raw.items() if name in names}
        values["rope_theta"] = raw["rope_parameters"]["rope_theta"]
        self.config = DecodeConfig(**values)
        if self.config.num_hidden_layers != 78:
            raise ValueError("NativeDecodeEngine requires 78 GLM-5.2 layers")
        self.capacity = capacity
        self.position = 0
        self.vocab_width = self.config.vocab_size // TP_SIZE
        self.vocab_start = self.rank * self.vocab_width
        vocabulary = slice(
            self.vocab_start, self.vocab_start + self.vocab_width
        )
        self.embedding = self._tensor(
            "model.embed_tokens.weight", (vocabulary, slice(None))
        )
        self.head = self._tensor("lm_head.weight", (vocabulary, slice(None)))
        self.final_norm = self._tensor("model.norm.weight")
        self.layers = [self._load_layer(layer) for layer in range(78)]
        self.norm_units = {
            width: torch.ones(width, dtype=torch.bfloat16, device=self.device)
            for width in {
                self.config.hidden_size, self.config.q_lora_rank,
                self.config.kv_lora_rank,
            }
        }
        dimension = self.config.qk_rope_head_dim
        frequency = 1.0 / (
            self.config.rope_theta
            ** (torch.arange(0, dimension, 2).float() / dimension)
        )
        angles = torch.arange(capacity).float()[:, None] * frequency
        cos, sin = angles.cos(), angles.sin()
        self.cos = torch.cat((cos, cos), dim=-1).to(
            device=self.device, dtype=torch.bfloat16
        )
        self.sin = torch.cat((sin, sin), dim=-1).to(
            device=self.device, dtype=torch.bfloat16
        )

    def _tensor(self, name, selection=None, dtype=torch.bfloat16):
        return self.checkpoint.tensor(name, selection).to(
            device=self.device, dtype=dtype
        ).contiguous()

    def _projection(self, name, output_range=None, input_range=None):
        source = self.checkpoint.linear(name, output_range, input_range)
        return NpuQuantizedLinear.from_checkpoint(source, str(self.device))

    def _ffn_weights(self, prefix, intermediate):
        width = intermediate // TP_SIZE
        channels = (self.rank * width, (self.rank + 1) * width)
        gate = self.checkpoint.linear(
            prefix + ".gate_proj", output_range=channels
        )
        up = self.checkpoint.linear(prefix + ".up_proj", output_range=channels)
        merged = QuantizedWeight(
            values=torch.cat((gate.values, up.values)),
            scale=torch.cat((gate.scale, up.scale)), bits=8,
        )
        gate_up = NpuQuantizedLinear.from_checkpoint(merged, str(self.device))
        down = self._projection(prefix + ".down_proj", input_range=channels)
        return gate_up, down

    def _load_layer(self, layer):
        config = self.config
        prefix = f"model.layers.{layer}"
        attention = prefix + ".self_attn"
        heads = config.num_attention_heads // TP_SIZE
        head_start, head_end = self.rank * heads, (self.rank + 1) * heads
        query_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        kv_dim = config.qk_nope_head_dim + config.v_head_dim
        projections = {
            "q_a": self._projection(attention + ".q_a_proj"),
            "kv_a": self._projection(attention + ".kv_a_proj_with_mqa"),
            "q_b": self._projection(
                attention + ".q_b_proj",
                output_range=(head_start * query_dim, head_end * query_dim),
            ),
            "o": self._projection(
                attention + ".o_proj",
                input_range=(head_start * config.v_head_dim,
                             head_end * config.v_head_dim),
            ),
        }
        norms = {
            "input": self._tensor(prefix + ".input_layernorm.weight"),
            "post": self._tensor(prefix + ".post_attention_layernorm.weight"),
            "q": self._tensor(attention + ".q_a_layernorm.weight"),
            "kv": self._tensor(attention + ".kv_a_layernorm.weight"),
        }
        kv_b = self._tensor(
            attention + ".kv_b_proj.weight",
            (slice(head_start * kv_dim, head_end * kv_dim), slice(None)),
        )
        mla = NativeMla(
            kv_b, capacity=self.capacity, device=str(self.device),
            validate_indices_on_host=self.validate_indices,
        )
        indexer, floating = None, {}
        if config.has_indexer(layer):
            index = attention + ".indexer"
            recipe = self.checkpoint.quantization.get(index + ".quant_type")
            dtype = torch.int8 if recipe == "INT8_DYNAMIC" else torch.bfloat16
            indexer = NativeIndexer(self.capacity, dtype, str(self.device))
            projections["index_q"] = self._projection(index + ".wq_b")
            floating = {
                "index_k": self._tensor(index + ".wk.weight"),
                "index_weights": self._tensor(index + ".weights_proj.weight"),
                "index_gamma": self._tensor(
                    index + ".k_norm.weight", dtype=torch.float32
                ),
                "index_beta": self._tensor(
                    index + ".k_norm.bias", dtype=torch.float32
                ),
            }
        result = NativeLayer(projections, norms, floating, mla, indexer)
        if layer < config.first_k_dense_replace:
            result.ffn = self._ffn_weights(
                prefix + ".mlp", config.intermediate_size
            )
        else:
            result.shared = self._ffn_weights(
                prefix + ".mlp.shared_experts", config.moe_intermediate_size
            )
            floating["router"] = self._tensor(
                prefix + ".mlp.gate.weight", dtype=torch.float32
            )
            floating["router_bias"] = self._tensor(
                prefix + ".mlp.gate.e_score_correction_bias",
                dtype=torch.float32,
            )
        return result

    def _expert(self, layer_number, expert, layer=None):
        layer = self.layers[layer_number] if layer is None else layer
        cache = layer.experts
        if expert not in cache:
            source = self.checkpoint.expert(layer_number, expert, self.rank)
            cache[expert] = NpuExpert.from_checkpoint(source, str(self.device))
        return cache[expert]

    def prepare_experts(self):
        """Make all weight reads device-resident before a profiling run."""
        for layer in range(self.config.first_k_dense_replace, 78):
            self.prepare_layer_experts(layer, self.layers[layer])
        torch.npu.synchronize(self.device)

    def prepare_layer_experts(self, layer_number, layer):
        if self.use_expert_bank:
            if layer.expert_bank is None:
                from .ops.expert_bank import NpuExpertBank

                layer.experts.clear()
                layer.expert_bank = NpuExpertBank.from_checkpoint(
                    self.checkpoint, layer_number, self.rank, str(self.device)
                )
        else:
            for expert in range(self.config.n_routed_experts):
                self._expert(layer_number, expert, layer)

    def reset(self):
        """Clear valid lengths; retain weights and cache allocations."""
        self.position = 0
        for layer in self.layers:
            layer.mla.reset()
            if layer.indexer is not None:
                layer.indexer.reset()

    def truncate(self, length):
        """Commit a prefix of already written main-model rows."""
        if not 0 <= length <= self.position:
            raise ValueError("Cannot commit rows that have not been written")
        for layer in self.layers:
            layer.mla.truncate(length)
            if layer.indexer is not None:
                layer.indexer.truncate(length)
        self.position = length

    def inject(self, prefixes):
        """Install post-norm, post-RoPE backbone KV and encoded index keys.

        Each of the 78 entries has ``latent`` and ``rope`` rows. Full index
        layers also provide ``index_keys`` and, for C8, ``index_scales``.
        MLA RoPE rows use npu_interleave_rope's even/odd-half output basis.
        Index keys retain the indexer's interleaved basis before Hadamard.
        The first generated token has not yet been appended to these rows.
        """
        if len(prefixes) != len(self.layers):
            raise ValueError("A prefix is required for every backbone layer")
        lengths = {item["latent"].shape[0] for item in prefixes}
        if len(lengths) != 1:
            raise ValueError("All backbone prefixes must have the same length")
        for layer, prefix in zip(self.layers, prefixes):
            layer.mla.inject(prefix["latent"], prefix["rope"])
            if layer.indexer is not None:
                layer.indexer.inject(
                    prefix["index_keys"], prefix.get("index_scales")
                )
                if layer.indexer.length != layer.mla.length:
                    raise ValueError("MLA and indexer prefix lengths differ")
        self.position = lengths.pop()

    def _sum(self, x):
        # Preserve FP32 accumulation before the BF16 residual boundary.
        total = x.float()
        distributed.all_reduce(total, group=self.group)
        return total.to(x.dtype)

    def project_logits(self, hidden, normalized=False):
        """Apply the final norm/head to a layer output for parity checks."""
        if not normalized:
            hidden = self._norm(hidden, self.final_norm)
        local = functional.linear(hidden, self.head)
        pieces = [torch.empty_like(local) for _ in range(TP_SIZE)]
        distributed.all_gather(pieces, local, group=self.group)
        return torch.cat(pieces, dim=-1)

    def project_tokens(self, hidden, normalized=False):
        """Select exact greedy IDs with two scalars per row and rank.

        Local argmax chooses the first vocabulary ID. All ranks compare
        candidates in ascending shard order, retaining that global tie rule.
        Full logits remain available through ``project_logits`` for parity.
        """
        if not normalized:
            hidden = self._norm(hidden, self.final_norm)
        local = functional.linear(hidden, self.head)
        scores, tokens = local.max(dim=-1)
        candidates = torch.stack((scores.float(),
                                  (tokens + self.vocab_start).float()), -1)
        pieces = [torch.empty_like(candidates) for _ in range(TP_SIZE)]
        distributed.all_gather(pieces, candidates, group=self.group)
        gathered = torch.stack(pieces, dim=0)
        winner = gathered[:, :, 0].argmax(dim=0, keepdim=True)
        return gathered[:, :, 1].gather(0, winner).squeeze(0).long()

    def _norm(self, x, weight):
        import torch_npu

        # HF rounds the normalized row to BF16 before multiplying gamma.
        output, _ = torch_npu.npu_rms_norm(
            x.contiguous(), self.norm_units[weight.numel()],
            epsilon=self.config.rms_norm_eps,
        )
        return output * weight

    def _rope(self, x, position=None):
        import torch_npu

        dimension = self.config.qk_rope_head_dim
        position = self.position if position is None else position
        rows = x.shape[0]
        cos = self.cos[position:position + rows].view(rows, 1, 1, dimension)
        sin = self.sin[position:position + rows].view(rows, 1, 1, dimension)
        result = torch_npu.npu_interleave_rope(
            x.contiguous().view(rows, -1, 1, dimension), cos, sin
        )
        return result.view_as(x)

    def _index_rope(self, x, position=None):
        """Keep indexer coordinates interleaved before Hadamard rotation."""
        rotated = self._rope(x, position)
        half = self.config.qk_rope_head_dim // 2
        return rotated.reshape(*x.shape[:-1], 2, half).transpose(
            -1, -2
        ).reshape_as(x)

    def _indices(
        self, x, q_latent, layer, position=None, external_indices=None,
    ):
        config = self.config
        floating = layer.floating
        key = functional.linear(x, floating["index_k"])
        key = functional.layer_norm(
            key.float(), (config.index_head_dim,), floating["index_gamma"],
            floating["index_beta"], eps=1e-6,
        ).to(x.dtype)
        dimension = config.qk_rope_head_dim
        key = torch.cat((self._index_rope(key[..., :dimension], position),
                         key[..., dimension:]), dim=-1)
        layer.indexer.append_many(key)
        if external_indices is not None:
            return external_indices
        query = layer.projections["index_q"](q_latent).view(
            x.shape[0], config.index_n_heads, config.index_head_dim
        )
        query = torch.cat((self._index_rope(query[..., :dimension], position),
                           query[..., dimension:]), dim=-1)
        weights = functional.linear(x, floating["index_weights"])
        return layer.indexer.select_many(query, weights)

    def _attention(
        self, x, layer, previous_indices, position=None, external_indices=None,
    ):
        config = self.config
        q_latent = self._norm(layer.projections["q_a"](x), layer.norms["q"])
        query = layer.projections["q_b"](q_latent).view(
            x.shape[0], config.num_attention_heads // TP_SIZE, -1
        )
        q_nope, q_rope = query.split(
            (config.qk_nope_head_dim, config.qk_rope_head_dim), dim=-1
        )
        compressed = layer.projections["kv_a"](x)
        latent, rope = compressed.split(
            (config.kv_lora_rank, config.qk_rope_head_dim), dim=-1
        )
        latent = self._norm(latent, layer.norms["kv"])
        layer.mla.append_many(latent, self._rope(rope, position))
        indices = (
            self._indices(x, q_latent, layer, position, external_indices)
            if layer.indexer is not None else previous_indices
        )
        attended = layer.mla.attend_many(
            q_nope, self._rope(q_rope, position), indices,
            allow_partial=external_indices is not None,
        )
        return self._sum(layer.projections["o"](attended)), indices

    @staticmethod
    def _ffn(x, operations):
        import torch_npu

        gate_up, down = operations
        return down(torch_npu.npu_swiglu(gate_up(x)))

    def _moe(self, x, layer_number, layer=None):
        import torch_npu

        layer = self.layers[layer_number] if layer is None else layer
        scores = functional.linear(
            x.float(), layer.floating["router"]
        ).sigmoid()
        corrected = scores + layer.floating["router_bias"]
        choices = corrected.argsort(dim=-1, descending=True, stable=True)
        choices = choices[:, :self.config.num_experts_per_tok]
        routing = scores.gather(1, choices)
        routing = routing / (routing.sum(dim=-1, keepdim=True) + 1e-20)
        routing = routing * self.config.routed_scaling_factor
        if self.use_expert_bank:
            self.prepare_layer_experts(layer_number, layer)
            output = layer.expert_bank(x, choices, routing.to(x.dtype))
        else:
            plan = build_plan(choices.cpu().tolist())
            tokens = torch.tensor(plan.token_ids, device=self.device)
            packed = x.index_select(0, tokens)
            outputs = []
            for group, expert in enumerate(plan.expert_ids):
                left, right = plan.group_offsets[group:group + 2]
                weights = self._expert(layer_number, expert, layer)
                outputs.append(self._ffn(
                    packed[left:right], (weights.gate_up, weights.down)
                ))
            rows = torch.tensor(plan.inverse, dtype=torch.int32,
                                device=self.device)
            output = torch_npu.npu_moe_token_unpermute(
                torch.cat(outputs), rows, probs=routing.to(x.dtype)
            )
        return output + self._ffn(x, layer.shared), choices

    @torch.inference_mode()
    def step(self, token, trace=False):
        """Consume a token, append KV on all ranks, predict its successor."""
        result = self.verify((token,), trace=trace)
        count = min(self.position, self.config.index_topk)
        selections = tuple(value[0, :count] for value in result.indices)
        routes = tuple(value[0] if value is not None else None
                       for value in result.routes)
        return StepResult(result.logits, result.hidden_states, selections,
                          routes, result.last_hidden, result.token_ids)

    def host_tokens(self, *groups):
        """Read several token groups to the host in one transfer."""
        tensors = [device_tokens(group, self.device) for group in groups]
        read = (tensors[0] if len(tensors) == 1 else torch.cat(tensors)).tolist()
        result, start = [], 0
        for tensor in tensors:
            width = tensor.shape[0]
            result.append(tuple(int(value) for value in
                                read[start:start + width]))
            start += width
        return tuple(result)

    def embed(self, tokens):
        """Look up replicated token rows from the sixteen vocabulary shards."""
        values = device_tokens(tokens, self.device)
        local = values - self.vocab_start
        owns = (local >= 0) & (local < self.vocab_width)
        valid_local = local.clamp(0, self.vocab_width - 1)
        x = self.embedding.index_select(0, valid_local)
        return self._sum(x.masked_fill(~owns[:, None], 0))

    @torch.inference_mode()
    def verify(self, tokens, trace=False, full_logits=True):
        """Consume M=1/2/4 rows; each row attends only to its causal prefix.

        ``last_hidden`` is the final RMSNorm output used to condition MTP.
        ``full_logits=False`` uses distributed top1 without vocabulary gather.
        """
        tokens = device_tokens(tokens, self.device)
        if tokens.shape[0] not in (1, 2, 4):
            raise ValueError("Verification width must be 1, 2 or 4")
        if self.position + tokens.shape[0] > self.capacity:
            raise ValueError("Decode cache capacity exceeded")
        x = self.embed(tokens)
        hidden = [x] if trace else []
        selections, routes, indices = [], [], None
        for number, layer in enumerate(self.layers):
            normalized = self._norm(x, layer.norms["input"])
            attention, indices = self._attention(normalized, layer, indices)
            x = x + attention
            normalized = self._norm(x, layer.norms["post"])
            if layer.ffn is not None:
                output, route = self._ffn(normalized, layer.ffn), None
            else:
                output, route = self._moe(normalized, number)
            x = x + self._sum(output)
            if trace:
                hidden.append(x)
                selections.append(indices)
                routes.append(route)
        x = self._norm(x, self.final_norm)
        last_hidden = x
        logits, token_ids = None, None
        if full_logits:
            logits = self.project_logits(x, normalized=True)
        else:
            token_ids = self.project_tokens(x, normalized=True)
        if trace:
            hidden[-1] = x
        self.position += len(tokens)
        return StepResult(
            logits, tuple(hidden), tuple(selections), tuple(routes),
            last_hidden, token_ids,
        )
