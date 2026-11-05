"""GLM-5.2 MTP layer 78 through the existing TP16/CANN decode operators."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import TYPE_CHECKING

from .cann_model import device_tokens

if TYPE_CHECKING:
    import torch


LAYER = 78
HIDDEN = 6144
VOCAB = 154880
TP_SIZE = 16
PREFIX = f"model.layers.{LAYER}"


def checkpoint_contract(directory):
    """Read only JSON and safetensors headers; no tensor payload or Torch."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    description = directory / "quant_model_description.json"
    quant = json.loads(description.read_text())
    index = directory / "quant_model_weights.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"]
    expected_config = {
        "num_hidden_layers": LAYER, "num_nextn_predict_layers": 1,
        "hidden_size": HIDDEN, "vocab_size": VOCAB, "rms_norm_eps": 1e-5,
    }
    for name, expected in expected_config.items():
        if config.get(name) != expected:
            raise ValueError(f"GLM-5.2 MTP requires {name}={expected}")
    if not quant.get("is_rot_used"):
        raise ValueError("This MTP path requires the checkpoint rot.weight")
    if "compress_ratios" in config:
        raise ValueError("Compressed MTP has a different embedding contract")
    expected_shapes = {
        "rot.weight": [HIDDEN, HIDDEN],
        PREFIX + ".enorm.weight": [HIDDEN],
        PREFIX + ".hnorm.weight": [HIDDEN],
        PREFIX + ".eh_proj.weight": [HIDDEN, 2 * HIDDEN],
        PREFIX + ".shared_head.norm.weight": [HIDDEN],
        PREFIX + ".shared_head.head.weight": [VOCAB, HIDDEN],
        "model.embed_tokens.weight": [VOCAB, HIDDEN],
    }
    headers, tensors = {}, {}
    for name, shape in expected_shapes.items():
        filename = weight_map[name]
        if filename not in headers:
            with (directory / filename).open("rb") as source:
                size = struct.unpack("<Q", source.read(8))[0]
                headers[filename] = json.loads(source.read(size))
        entry = headers[filename][name]
        if entry["shape"] != shape or entry["dtype"] != "BF16":
            raise ValueError(f"Unexpected MTP tensor metadata: {name}")
        tensors[name] = {"file": filename, "shape": shape, "dtype": "BF16"}
    index_recipe = quant.get(PREFIX + ".self_attn.indexer.quant_type")
    if index_recipe not in (None, "FLOAT"):
        raise ValueError("GLM-5.2 MTP index cache must be BF16")
    return {
        "architecture": config["architectures"], "layer": LAYER,
        "tp_size": TP_SIZE, "epsilon": config["rms_norm_eps"],
        "index_share_for_mtp_iteration": config.get(
            "index_share_for_mtp_iteration", False
        ),
        "index_cache_dtype": "BF16", "tensors": tensors,
        "embedding": "target embedding, PP1 and no compression",
        "head": "checkpoint MTP head; equality with target is not assumed",
        "position": "MTP row p consumes token[p+1] and parent hidden[p]",
        "parent_hidden": "normalized target or normalized MTP recycle",
        "rotation": "every forward: parent @ rot.weight.T before hnorm",
        "concat": ["enorm(embedding)", "hnorm(rotated_parent)"],
        "logits": "own_head(shared_head.norm(pre_norm_hidden))",
        "recycle_hidden": "shared_head.norm(pre_norm_hidden)",
    }


@dataclass
class MtpResult:
    logits: torch.Tensor | None
    token_ids: torch.Tensor | None
    pre_norm_hidden: torch.Tensor
    recycle_hidden: torch.Tensor
    indices: torch.Tensor
    routes: torch.Tensor
    start_position: int
    stages: dict[str, torch.Tensor]


class NativeMtp:
    """One independent MTP cache, sharing only backbone helpers/embedding.

    forward() consumes rows at the current parent positions. Every parent
    is normalized: the target final norm for target-conditioned rows, or
    the previous MTP final norm for a chained draft. external_indices is
    the first draft step's selected row when index sharing is enabled.
    It changes selection only; the new index key is still written.
    """

    def __init__(self, backbone):
        self.backbone = backbone
        self.device = backbone.device
        self.capacity = backbone.capacity
        self.position = 0
        self.contract = checkpoint_contract(backbone.checkpoint.directory)
        self.layer = backbone._load_layer(LAYER)
        self.rot = backbone._tensor("rot.weight")
        self.enorm = backbone._tensor(PREFIX + ".enorm.weight")
        self.hnorm = backbone._tensor(PREFIX + ".hnorm.weight")
        self.eh_proj = backbone._tensor(PREFIX + ".eh_proj.weight")
        self.final_norm = backbone._tensor(PREFIX + ".shared_head.norm.weight")
        vocabulary = slice(backbone.vocab_start,
                           backbone.vocab_start + backbone.vocab_width)
        self.head = backbone._tensor(
            PREFIX + ".shared_head.head.weight", (vocabulary, slice(None))
        )

    def reset(self):
        self.layer.mla.reset()
        self.layer.indexer.reset()
        self.position = 0

    def prepare_experts(self):
        """Make layer-78 experts resident before a timed draft/verify run."""
        import torch

        self.backbone.prepare_layer_experts(LAYER, self.layer)
        torch.npu.synchronize(self.device)

    def truncate(self, length):
        if isinstance(length, bool) or not isinstance(length, int):
            raise ValueError("MTP length must be an integer")
        if not 0 <= length <= self.position:
            raise ValueError("Cannot retain unwritten MTP rows")
        self.layer.mla.truncate(length)
        self.layer.indexer.truncate(length)
        self.position = length

    def inject(self, prefix):
        """Install layer-78 latent/RoPE/BF16-index rows at parent positions."""
        if prefix.get("index_scales") is not None:
            raise ValueError("MTP BF16 index cache has no scales")
        length = prefix["latent"].shape[0]
        if prefix["rope"].shape[0] != length:
            raise ValueError("MTP latent and RoPE prefix lengths differ")
        if prefix["index_keys"].shape[0] != length:
            raise ValueError("MTP MLA and index prefix lengths differ")
        self.layer.mla.inject(prefix["latent"], prefix["rope"])
        self.layer.indexer.inject(prefix["index_keys"])
        self.position = length

    def _validate(self, tokens, parent, external_indices):
        import torch

        tokens = device_tokens(tokens, self.device)
        if tokens.shape[0] not in (1, 2, 4):
            raise ValueError("MTP width must be 1, 2 or 4")
        if (parent.shape != (tokens.shape[0], HIDDEN)
                or parent.dtype != torch.bfloat16
                or parent.device != self.device):
            raise ValueError("MTP parent must be device BF16 [M,6144]")
        if self.position + len(tokens) > self.capacity:
            raise ValueError("MTP cache capacity exceeded")
        if self.layer.mla.length != self.position:
            raise ValueError("MTP MLA position differs from valid length")
        if self.layer.indexer.length != self.position:
            raise ValueError("MTP index position differs from valid length")
        if external_indices is not None:
            if (external_indices.shape != (len(tokens), 2048)
                    or external_indices.dtype not in (torch.int32, torch.int64)
                    or external_indices.device != self.device):
                raise ValueError("Shared MTP indices must be device [M,2048]")
        return tokens

    def _project(self, normalized, full_logits):
        import torch
        import torch.distributed as distributed
        import torch.nn.functional as functional

        local = functional.linear(normalized, self.head)
        if full_logits:
            pieces = [torch.empty_like(local) for _ in range(TP_SIZE)]
            distributed.all_gather(pieces, local, group=self.backbone.group)
            return torch.cat(pieces, dim=-1), None
        scores, tokens = local.max(dim=-1)
        global_tokens = (tokens + self.backbone.vocab_start).float()
        candidates = torch.stack((scores.float(), global_tokens), dim=-1)
        pieces = [torch.empty_like(candidates) for _ in range(TP_SIZE)]
        distributed.all_gather(pieces, candidates, group=self.backbone.group)
        gathered = torch.stack(pieces)
        winner = gathered[:, :, 0].argmax(dim=0, keepdim=True)
        token_ids = gathered[:, :, 1].gather(0, winner).squeeze(0).long()
        return None, token_ids

    def step(self, token, normalized_parent_hidden, **kwargs):
        return self.forward((token,), normalized_parent_hidden, **kwargs)

    def forward(self, tokens, normalized_parent_hidden, *, trace=False,
                full_logits=True, external_indices=None):
        import torch

        tokens = self._validate(
            tokens, normalized_parent_hidden, external_indices
        )
        with torch.inference_mode():
            return self._forward(tokens, normalized_parent_hidden, trace,
                                 full_logits, external_indices)

    def _forward(self, tokens, parent, trace, full_logits, external_indices):
        import torch
        import torch.nn.functional as functional

        base = self.backbone
        position = self.position
        embedding = base.embed(tokens)
        if position == 0:
            embedding[0].zero_()
        rotated = functional.linear(parent, self.rot)
        embedding_norm = base._norm(embedding, self.enorm)
        parent_norm = base._norm(rotated, self.hnorm)
        x = functional.linear(
            torch.cat((embedding_norm, parent_norm), dim=-1), self.eh_proj
        )
        stages = {
            "masked_embedding": embedding, "rotated_parent": rotated,
            "enorm": embedding_norm, "hnorm": parent_norm, "eh_proj": x,
        } if trace else {}
        attention, indices = base._attention(
            base._norm(x, self.layer.norms["input"]), self.layer, None,
            position=position, external_indices=external_indices,
        )
        residual = x + attention
        ffn, routes = base._moe(
            base._norm(residual, self.layer.norms["post"]), LAYER,
            layer=self.layer,
        )
        reduced_ffn = base._sum(ffn)
        pre_norm = residual + reduced_ffn
        recycle = base._norm(pre_norm, self.final_norm)
        logits, token_ids = self._project(recycle, full_logits)
        if trace:
            stages.update(attention=attention, post_attention=residual,
                          ffn=reduced_ffn)
        self.position += len(tokens)
        return MtpResult(logits, token_ids, pre_norm, recycle, indices,
                         routes, position, stages)
