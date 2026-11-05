"""CPU MTP oracle using HF's decoder and independent TP16 weight arithmetic.

Importing this module requires only the standard library. Construction loads
Torch and Transformers; inference uses expanded HF KV and no CANN operators.
"""

import json
from pathlib import Path
import struct


LAYER = 78
HIDDEN = 6144
VOCAB = 154880
TOP_K = 2048
PREFIX = f"model.layers.{LAYER}"
IMPLEMENTATION = "transformers.GlmMoeDsaDecoderLayer.MTP"


def checkpoint_contract(directory):
    """Validate the MTP configuration and tensor headers without Torch."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    description = directory / "quant_model_description.json"
    quant = json.loads(description.read_text())
    index = directory / "quant_model_weights.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"]
    expected = {
        "num_hidden_layers": 78, "num_nextn_predict_layers": 1,
        "hidden_size": HIDDEN, "vocab_size": VOCAB,
        "num_attention_heads": 64, "q_lora_rank": 2048,
        "kv_lora_rank": 512, "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64, "v_head_dim": 256,
        "n_routed_experts": 256, "num_experts_per_tok": 8,
        "n_shared_experts": 1, "n_group": 1, "topk_group": 1,
        "rms_norm_eps": 1e-5, "index_topk": TOP_K,
        "index_n_heads": 32, "index_head_dim": 128,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"MTP reference requires {name}={value}")
    if not quant.get("is_rot_used") or "compress_ratios" in config:
        raise ValueError("MTP requires rotation and uncompressed embeddings")
    recipe = quant.get(PREFIX + ".self_attn.indexer.quant_type")
    if recipe not in (None, "FLOAT"):
        raise ValueError("The MTP index cache must use BF16")
    shapes = {
        "rot.weight": [HIDDEN, HIDDEN],
        "model.embed_tokens.weight": [VOCAB, HIDDEN],
        PREFIX + ".enorm.weight": [HIDDEN],
        PREFIX + ".hnorm.weight": [HIDDEN],
        PREFIX + ".eh_proj.weight": [HIDDEN, 2 * HIDDEN],
        PREFIX + ".shared_head.norm.weight": [HIDDEN],
        PREFIX + ".shared_head.head.weight": [VOCAB, HIDDEN],
    }
    headers, tensors = {}, {}
    for name, shape in shapes.items():
        filename = weight_map[name]
        if filename not in headers:
            with (directory / filename).open("rb") as source:
                size = struct.unpack("<Q", source.read(8))[0]
                headers[filename] = json.loads(source.read(size))
        entry = headers[filename][name]
        if entry["shape"] != shape or entry["dtype"] != "BF16":
            raise ValueError(f"Unexpected MTP tensor metadata: {name}")
        tensors[name] = {"file": filename, "shape": shape, "dtype": "BF16"}
    return {
        "layer": LAYER, "hf_layer_index": 0, "tp_size": 16,
        "position_convention": "parent", "epsilon": 1e-5,
        "embedding": "model.embed_tokens; PP1 without compression",
        "head": "layer-78 shared_head.head, independently read",
        "index_cache_dtype": "BF16", "tensors": tensors,
        "arithmetic": "HF expanded BF16 KV; independent CPU TP16 adapters",
    }


def _shared_indexer(weights):
    from .reference import ReferenceIndexer

    class SharedIndexer(ReferenceIndexer):
        def __init__(self):
            super().__init__(weights, PREFIX + ".self_attn.indexer", 0, False)
            self.external_indices = None

        def forward(self, *args, **kwargs):
            selected = super().forward(*args, **kwargs)
            if self.external_indices is None:
                return selected
            return self.external_indices.reshape(1, 1, -1)

    return SharedIndexer()


class MtpReference:
    """Execute M=1/2/4 as causal HF rows at parent RoPE/cache positions.

    The single HF layer at index zero reads checkpoint layer 78. Prefixes
    contain replicated compressed latent/RoPE/index rows; only HF attention
    expands them into per-head K/V. Trace fields match the native MTP runner.
    """

    def __init__(self, checkpoint, capacity=9216):
        import torch
        from transformers import GlmMoeDsaConfig
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
            GlmMoeDsaDecoderLayer,
            GlmMoeDsaRMSNorm,
            GlmMoeDsaRotaryEmbedding,
        )

        from .reference import ReferenceEmbedding, ReferenceLinear
        from .reference_weights import ReferenceWeights

        if type(capacity) is not int or capacity < 1:
            raise ValueError("Reference capacity must be positive")
        self.capacity = capacity
        self.contract = checkpoint_contract(checkpoint)
        self.weights = ReferenceWeights(checkpoint)
        raw = dict(self.weights.config)
        raw.update(
            num_hidden_layers=1, first_k_dense_replace=0,
            mlp_layer_types=["sparse"], indexer_types=["full"],
            layer_types=["deepseek_sparse_attention"],
        )
        self.config = GlmMoeDsaConfig.from_dict(raw)
        self.config._attn_implementation = "eager"
        self.config._experts_implementation = "eager"
        with torch.device("meta"):
            self.decoder = GlmMoeDsaDecoderLayer(self.config, 0).to(
                torch.bfloat16
            )
        self._install_adapters()
        self.embedding = ReferenceEmbedding(self.weights)
        self.rot = ReferenceLinear(self.weights, "rot")
        self.eh_proj = ReferenceLinear(self.weights, PREFIX + ".eh_proj")
        self.head = ReferenceLinear(
            self.weights, PREFIX + ".shared_head.head", "output"
        )
        for name, prefix in (
            ("enorm", PREFIX + ".enorm"),
            ("hnorm", PREFIX + ".hnorm"),
            ("final_norm", PREFIX + ".shared_head.norm"),
        ):
            norm = GlmMoeDsaRMSNorm(HIDDEN, eps=1e-5)
            self._install_norm(norm, prefix)
            setattr(self, name, norm)
        self.rotary = GlmMoeDsaRotaryEmbedding(self.config, device="cpu")
        self.decoder.eval().requires_grad_(False)
        remaining = [name for name, value in (
            *self.decoder.named_parameters(), *self.decoder.named_buffers()
        ) if value.is_meta]
        if remaining:
            raise ValueError(f"Unreplaced HF meta tensors: {remaining}")
        self._attach_hooks()
        self.reset()

    def _install_norm(self, module, prefix):
        import torch
        from torch import nn

        module.weight = nn.Parameter(self.weights.tensor(
            prefix + ".weight", dtype=torch.bfloat16
        ), requires_grad=False)
        module.variance_epsilon = 1e-5

    def _install_adapters(self):
        import torch
        from torch import nn

        from .reference import (
            ROTARY_CONTRACT,
            ReferenceLinear,
            ReferenceMoe,
            install_attention_rope,
        )

        layer = self.decoder
        install_attention_rope(layer.self_attn)
        self.weights.contract["rotary"] = dict(ROTARY_CONTRACT)
        attention = PREFIX + ".self_attn"
        for module, prefix in (
            (layer.input_layernorm, PREFIX + ".input_layernorm"),
            (layer.post_attention_layernorm,
             PREFIX + ".post_attention_layernorm"),
            (layer.self_attn.q_a_layernorm, attention + ".q_a_layernorm"),
            (layer.self_attn.kv_a_layernorm, attention + ".kv_a_layernorm"),
        ):
            self._install_norm(module, prefix)
        for name, partition in (
            ("q_a_proj", "replicated"), ("q_b_proj", "output"),
            ("kv_a_proj_with_mqa", "replicated"),
            ("kv_b_proj", "output"), ("o_proj", "input"),
        ):
            setattr(layer.self_attn, name, ReferenceLinear(
                self.weights, attention + "." + name, partition
            ))
        layer.self_attn.indexer = _shared_indexer(self.weights)
        gate = layer.mlp.gate
        gate.weight = nn.Parameter(self.weights.tensor(
            PREFIX + ".mlp.gate.weight", dtype=torch.float32
        ), requires_grad=False)
        gate.e_score_correction_bias = self.weights.tensor(
            PREFIX + ".mlp.gate.e_score_correction_bias", dtype=torch.float32
        )
        layer.mlp = ReferenceMoe(self.weights, PREFIX + ".mlp", gate)

    def _attach_hooks(self):
        def attention(module, inputs, output):
            self._stages["attention"] = output[0].reshape(1, HIDDEN).clone()

        def post_attention(module, inputs):
            value = inputs[0].reshape(1, HIDDEN).clone()
            self._stages["post_attention"] = value

        def ffn(module, inputs, output):
            self._stages["ffn"] = output.reshape(1, HIDDEN).clone()

        def latent(module, inputs, output):
            self._latent = output.reshape(1, 512).clone()

        def routes(module, inputs, output):
            self._routes = output[2].reshape(1, 8).clone()

        self.decoder.self_attn.register_forward_hook(attention)
        self.decoder.post_attention_layernorm.register_forward_pre_hook(
            post_attention
        )
        self.decoder.mlp.register_forward_hook(ffn)
        self.decoder.self_attn.kv_a_layernorm.register_forward_hook(latent)
        self.decoder.mlp.gate.register_forward_hook(routes)

    @property
    def position(self):
        return self.cache.get_seq_length()

    def reset(self):
        from transformers.cache_utils import DynamicCache

        self.cache = DynamicCache(config=self.config)
        self.decoder.self_attn.indexer.reset()
        self.decoder.self_attn.indexer.external_indices = None

    def check_lengths(self, expected):
        if len(self.cache.layers) != 1 or self.position != expected:
            raise ValueError("HF MLA length differs from parent position")
        keys = self.cache.layers[0].indexer_keys
        length = 0 if keys is None else keys.shape[1]
        if length != expected:
            raise ValueError("HF index cache length differs from MLA cache")

    def inject(self, prefix):
        import torch

        if self.position:
            raise ValueError("Inject a prefix into an empty MTP reference")
        dimensions = {"latent": 512, "rope": 64, "index_keys": 128}
        if set(prefix) != set(dimensions):
            raise ValueError("MTP prefix requires latent, rope and index_keys")
        length = prefix["latent"].shape[0]
        if length > self.capacity:
            raise ValueError("MTP prefix exceeds capacity")
        for name, width in dimensions.items():
            value = prefix[name]
            if (tuple(value.shape) != (length, width)
                    or value.dtype != torch.bfloat16
                    or value.device.type != "cpu"):
                raise ValueError(f"Invalid MTP prefix {name}")
        if length:
            expanded = self.decoder.self_attn.kv_b_proj(prefix["latent"])
            key, value = expanded.reshape(length, 64, 448).split(
                (192, 256), dim=-1
            )
            rope = prefix["rope"][:, None, :].expand(-1, 64, 64)
            key = torch.cat((key, rope), dim=-1).transpose(0, 1).unsqueeze(0)
            value = value.transpose(0, 1).unsqueeze(0)
            self.cache.update(key.contiguous(), value.contiguous(), 0)
            self.decoder.self_attn.indexer.inject(
                self.cache, prefix["index_keys"], None
            )
        self.check_lengths(length)

    def _validate(self, tokens, parent, external_indices):
        import torch

        if (not isinstance(tokens, (tuple, list))
                or len(tokens) not in (1, 2, 4)):
            raise ValueError("MTP tokens must contain M=1/2/4 entries")
        if any(type(token) is not int or not 0 <= token < VOCAB
               for token in tokens):
            raise ValueError("MTP token is outside the vocabulary")
        rows = len(tokens)
        if (tuple(parent.shape) != (rows, HIDDEN)
                or parent.dtype != torch.bfloat16
                or parent.device.type != "cpu"):
            raise ValueError("MTP parents must be CPU BF16 [M,6144]")
        if self.position + rows > self.capacity:
            raise ValueError("MTP rows exceed capacity")
        if external_indices is None:
            return [None] * rows
        if (tuple(external_indices.shape) != (rows, TOP_K)
                or external_indices.dtype not in (torch.int32, torch.int64)
                or external_indices.device.type != "cpu"):
            raise ValueError("Shared indices must be CPU INT32/INT64 [M,2048]")
        selections = []
        for row, indices in enumerate(external_indices.tolist()):
            count = indices.index(-1) if -1 in indices else TOP_K
            valid, padding = indices[:count], indices[count:]
            bound = self.position + row + 1
            if (not 1 <= count <= min(bound, TOP_K)
                    or len(set(valid)) != count
                    or any(index < 0 or index >= bound for index in valid)
                    or any(index != -1 for index in padding)):
                raise ValueError("Shared indices violate the causal prefix")
            selections.append(external_indices[row, :count].to(torch.int32))
        return selections

    def _step(self, token, parent, external_indices):
        import torch

        position = self.position
        embedding = self.embedding(torch.tensor([[token]], dtype=torch.int64))
        if position == 0:
            embedding = torch.zeros_like(embedding)
        rotated = self.rot(parent.reshape(1, 1, HIDDEN))
        enorm, hnorm = self.enorm(embedding), self.hnorm(rotated)
        hidden = self.eh_proj(torch.cat((enorm, hnorm), dim=-1))
        self._stages = {
            name: value.reshape(1, HIDDEN).clone() for name, value in (
                ("masked_embedding", embedding), ("rotated_parent", rotated),
                ("enorm", enorm), ("hnorm", hnorm), ("eh_proj", hidden),
            )
        }
        positions = torch.tensor([[position]], dtype=torch.int64)
        indexer = self.decoder.self_attn.indexer
        indexer.external_indices = external_indices
        try:
            pre_norm, selected = self.decoder(
                hidden, position_ids=positions, past_key_values=self.cache,
                use_cache=True, attention_mask=None,
                position_embeddings=self.rotary(hidden, positions),
            )
        finally:
            indexer.external_indices = None
        recycle = self.final_norm(pre_norm)
        logits = self.head(recycle).reshape(1, VOCAB)
        indices = torch.full((1, TOP_K), -1, dtype=torch.int32)
        indices[:, :selected.shape[-1]] = selected.reshape(1, -1)
        self.check_lengths(position + 1)
        return {
            "logits": logits, "pre_norm_hidden": pre_norm.reshape(1, HIDDEN),
            "recycle_hidden": recycle.reshape(1, HIDDEN),
            "indices": indices, "routes": self._routes,
            "stages": self._stages,
            "cache": {
                "latent": self._latent,
                "rope": self.cache.layers[0].keys[0, 0, -1, 192:].reshape(
                    1, 64
                ).clone(),
                "index_keys": indexer.last_key.reshape(1, 128).clone(),
            },
        }

    def forward(self, tokens, normalized_parent_hidden, external_indices=None):
        """Return the native snapshot schema, computed as CPU HF M=1 rows."""
        import torch

        selections = self._validate(
            tokens, normalized_parent_hidden, external_indices
        )
        start = self.position
        self.check_lengths(start)
        with torch.inference_mode():
            rows = [self._step(token, normalized_parent_hidden[row:row + 1],
                               selections[row])
                    for row, token in enumerate(tokens)]
            result = {
                name: torch.cat([row[name] for row in rows], dim=0).clone()
                for name in ("logits", "pre_norm_hidden", "recycle_hidden",
                             "indices", "routes")
            }
            for group in ("stages", "cache"):
                result[group] = {
                    name: torch.cat([row[group][name] for row in rows]).clone()
                    for name in rows[0][group]
                }
        result.update(position=start, committed_position=self.position)
        return result
