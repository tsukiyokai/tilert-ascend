"""GLM-5.2 layer-78 MTP through the pinned vLLM-Ascend model."""

import copy
from dataclasses import dataclass
import json
from pathlib import Path

from .cann_model import CannModel, device_tokens


LAYER = 78
HIDDEN = 6144
TOPK = 2048
MAIN_CACHE = "model.layers.78.self_attn.attn"
INDEX_CACHE = "model.layers.78.self_attn.indexer.k_cache"
WEIGHT_PATTERN = "quant_model_weights-0009[123]-of-00095.safetensors"
WEIGHT_FILES = tuple(
    f"quant_model_weights-{number:05d}-of-00095.safetensors"
    for number in (91, 92, 93)
)


def weight_sources(checkpoint):
    """Check the fixed checkpoint's MTP shard selection using JSON only."""
    path = Path(checkpoint) / "quant_model_weights.safetensors.index.json"
    weights = json.loads(path.read_text())["weight_map"]
    selected = {filename for name, filename in weights.items()
                if name.startswith("model.layers.78.")}
    if selected != set(WEIGHT_FILES):
        raise ValueError("Expected MTP weights in checkpoint shards 91-93")
    if (weights.get("rot.weight") != "rot.safetensors"
            or weights.get("model.layers.78.shared_head.head.weight")
            not in selected):
        raise ValueError("MTP requires rot.weight and its checkpoint head")
    return (*WEIGHT_FILES, "rot.safetensors")


def draft_config(target_config):
    """Keep target parallel settings and isolate draft registration state."""
    from vllm.config import CompilationMode, CUDAGraphMode, SpeculativeConfig

    parallel = target_config.parallel_config
    if (parallel.tensor_parallel_size != 16
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1):
        raise ValueError("CANN MTP requires TP16/PP1/DP1/PCP1/DCP1")
    if target_config.speculative_config is not None:
        raise ValueError("The target must not own a speculative model")
    hf = target_config.model_config.hf_config
    # The depth is not checked against a literal: a truncated backbone is a
    # legitimate configuration for an arm that measures the ratio between two
    # others, and what this guard is for is the model family and its single
    # shared-index MTP layer.
    if (hf.model_type != "glm_moe_dsa"
            or hf.num_nextn_predict_layers != 1
            or not hf.index_share_for_mtp_iteration):
        raise ValueError("Expected GLM-5.2 with one shared-index MTP layer")
    config = copy.copy(target_config)
    config.speculative_config = SpeculativeConfig(
        method="mtp", num_speculative_tokens=3,
        draft_tensor_parallel_size=16, enforce_eager=True,
        parallel_drafting=False,
        target_model_config=target_config.model_config,
        target_parallel_config=parallel,
    )
    config.speculative_config.draft_model_config.enforce_eager = True
    config.model_config = copy.copy(target_config.model_config)
    config.model_config.runner_type = "draft"
    config.model_config.enforce_eager = True
    config.attention_config = copy.copy(target_config.attention_config)
    config.attention_config.backend = None
    config.compilation_config = copy.copy(target_config.compilation_config)
    config.compilation_config.mode = CompilationMode.NONE
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    config.compilation_config.cudagraph_capture_sizes = []
    config.compilation_config.max_cudagraph_capture_size = 0
    config.compilation_config.compile_sizes = []
    config.compilation_config.static_forward_context = {}
    config.compilation_config.static_all_moe_layers = []
    config.additional_config = copy.deepcopy(target_config.additional_config)
    compilation = config.additional_config.setdefault(
        "ascend_compilation_config", {},
    )
    compilation.update(
        enable_static_kernel=False, enable_super_kernel=False,
    )
    config.quant_config = copy.copy(target_config.quant_config)
    config.load_config = copy.copy(target_config.load_config)
    config.load_config.load_format = "safetensors"
    config.speculative_config.draft_load_config = config.load_config
    return config


@dataclass
class CannMtpStep:
    logits: object
    token_ids: object
    pre_norm_hidden: object
    recycle_hidden: object
    indices: object
    start_position: int


class CannMtp:
    """Independent MTP state using the target's reserved layer-78 storage.

    Row p consumes token[p+1] with normalized, unrotated parent hidden[p].
    External indices select a chained draft's seed set; the official
    model still appends that row's latent, RoPE and BF16 index key.
    """

    def __init__(self, target, *, shard_projections=False):
        import torch
        from vllm.config import set_current_vllm_config

        self.target = target
        self.device = target.device
        self.capacity = target.capacity
        self.layout = target.layout
        self.position = 0
        self.graph = None
        self.conditioned_graph = None
        self.shard_projections = shard_projections
        self.hoist_metadata = target.hoist_metadata
        self.attention_kernel = (
            "sfa" if target.attention_kernel == "shared_fia"
            else target.attention_kernel
        )
        self.checkpoint = target.checkpoint
        self.weight_files = weight_sources(self.checkpoint)
        self.config = draft_config(target.config)
        with torch.npu.device(self.device), torch.inference_mode(), (
            set_current_vllm_config(self.config)
        ):
            if self.config.model_config.hf_config.num_hidden_layers < LAYER:
                # A truncated backbone's MTP layer index is its own depth,
                # where the checkpoint's is 78, so the completeness check
                # cannot be satisfied. It is checked here rather than in
                # `cann_model`, because the draft model loads in this object
                # and a ContextVar set around the target's load does not reach
                # it.
                from vllm.model_executor.model_loader.mtp_validation import (
                    disable_mtp_completeness_check,
                )

                with disable_mtp_completeness_check():
                    self.model = self._load_model()
            else:
                self.model = self._load_model()
            self._bind_cache()
            CannModel._allocate_inputs(self)

    def _load_model(self):
        import torch
        from vllm.compilation.backends import set_model_tag
        from vllm.model_executor.model_loader.default_loader import (
            DefaultModelLoader,
        )
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )
        from vllm.utils.torch_utils import set_default_torch_dtype

        model_config = self.config.speculative_config.draft_model_config
        loader = DefaultModelLoader(self.config.load_config)
        with set_default_torch_dtype(model_config.dtype), (
            torch.device(self.device)
        ), set_model_tag("eagle_head"):
            model = loader.create_model(self.config, model_config)
            model.allow_patterns_overrides = [WEIGHT_PATTERN]
            model.secondary_weights = [DefaultModelLoader.Source(
                self.checkpoint, None, fall_back_to_pt=False,
                allow_patterns_overrides=["rot.safetensors"],
            )]
            loader.load_weights(model, model_config)
            # `has_own_lm_head` is gone from this vllm's `DeepSeekMTP`,
            # which says the same thing with `defer_lm_head`. What the check
            # is for is that the MTP head was built at all, so ask for its own
            # parameters by name: the two norms and the projection.
            names = {name for name, _ in model.named_parameters()}
            own = {"enorm.weight", "hnorm.weight", "eh_proj.weight"}
            if not all(any(name.endswith(suffix) for name in names)
                       for suffix in own):
                raise ValueError("Official loader did not load the MTP head")
            process_weights_after_loading(model, model_config, self.device)
        model.model.embed_tokens = self.target.model.model.embed_tokens
        # This vllm defers the MTP's output head and calls it `head`, to be
        # filled from the target's -- `eagle/utils.py` does `sh.head =
        # target_lm_head`. The engine shares the embedding the same way, so
        # share the head too rather than reach for a `lm_head` that the
        # deferred predictor does not have.
        head = self.target.model.lm_head
        for layer in model.model.layers.values():
            layer.head = head
        if self.target.fp32_head:
            from .cann_head import install_fp32_head

            install_fp32_head(head, model.model.logits_processor)
        if self.shard_projections:
            from .cann_mtp_projection import shard_mtp_projections

            shard_mtp_projections(model)
        return model.eval()

    def _bind_cache(self):
        import torch
        from vllm_ascend.attention.indexer import (
            AscendSFAIndexerMetadataBuilder,
        )
        from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
        from vllm_ascend.core.kv_cache_interface import (
            AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec,
        )
        from vllm_ascend.patch.worker.patch_bind_kv_cache import (
            bind_kv_cache_to_layers,
        )

        views = self.target.views
        latent, rope, index = (views[name] for name in (
            "mla_latent.78", "mla_rope.78", "index.78",
        ))
        if any(value.dtype != torch.bfloat16
               for value in (latent, rope, index)):
            raise ValueError("All layer-78 cache regions must be BF16")
        self.cache = {
            "latent": latent.view(-1, 512),
            "rope": rope.view(-1, 64),
            "index_keys": index.view(-1, 128),
        }
        context = self.config.compilation_config.static_forward_context
        bind_kv_cache_to_layers(
            {MAIN_CACHE: (latent, rope), INDEX_CACHE: (index,)}, context,
        )
        if self.attention_kernel == "gather_fia":
            from .cann_attention import install_gather_fia

            install_gather_fia(context[MAIN_CACHE].impl, self.target.graph_width)
        main_spec = AscendMLAAttentionSpec(
            block_size=128, num_kv_heads=1, head_size=576,
            num_heads=4, dtype=torch.bfloat16, cache_dtype_str="auto",
        )
        index_spec = AscendSFAIndexerCacheSpec(
            block_size=128, num_kv_heads=1, head_size=128,
            dtype=torch.bfloat16, cache_dtype_str="auto", scale_dim=0,
        )
        self.builders = [
            ([MAIN_CACHE], AscendSFAMetadataBuilder(
                main_spec, [MAIN_CACHE], self.config, self.device,
            )),
            ([INDEX_CACHE], AscendSFAIndexerMetadataBuilder(
                index_spec, [INDEX_CACHE], self.config, self.device,
            )),
        ]
        self.attention = self.model.model.layers["78"].mtp_block.self_attn
        self.indices_buffer = self.attention.mla_attn.topk_indices_buffer
        rows = self.config.scheduler_config.max_num_batched_tokens
        if self.indices_buffer.shape != (rows, TOPK):
            raise ValueError("Expected a private MTP index buffer")

    def inject(self, prefix):
        """Install P-1 MTP rows after the target has received P main rows."""
        import torch

        values = {key: value for key, value in prefix.items()
                  if value is not None}
        if values.keys() != self.cache.keys():
            raise ValueError("MTP requires latent, RoPE and BF16 index keys")
        length = values["latent"].shape[0]
        if length != self.target.position - 1:
            raise ValueError("MTP injection must contain target P-1 rows")
        if not 0 <= length < self.capacity:
            raise ValueError("MTP prefix leaves no writable row")
        for name, destination in self.cache.items():
            if (values[name].shape != destination[:length].shape
                    or values[name].dtype != destination.dtype):
                raise ValueError(f"Invalid MTP {name} prefix")
        with torch.npu.device(self.device), torch.inference_mode():
            for name, destination in self.cache.items():
                destination[:length].copy_(values[name])
        self.position = length
        self.model.model.set_skip_topk(False)

    def enable_graph(self, *, conditioned_width=0):
        """Capture shared-index M1 chains after uncaptured TP warmup.

        A positive ``conditioned_width`` also captures the target-conditioned
        pass, which selects its own index set instead of replaying a shared
        one. It must be the verify width: the pass carries one row per
        verified token, so a graph captured at any other width is never
        replayed.
        """
        import torch
        from .cann_mtp_graph import CannMtpGraph

        with torch.npu.device(self.device):
            if self.graph is None:
                self.graph = CannMtpGraph(self, 1)
            if conditioned_width and self.conditioned_graph is None:
                self.conditioned_graph = CannMtpGraph(self, conditioned_width)

    def close(self):
        """Release the draft graphs before shutting down the target worker."""
        if self.graph is not None:
            self.graph.close()
            self.graph = None
        if self.conditioned_graph is not None:
            self.conditioned_graph.close()
            self.conditioned_graph = None

    def _graph(self, width, shared_indices):
        """Name the captured graph for this call, or None for eager.

        A chain replays only against a shared index set; the
        target-conditioned pass replays only when it selects its own. Each
        asks the graph it would use whether that graph was captured at this
        width, rather than naming the width twice -- the literal 4 that used
        to stand here kept the conditioned pass eager at every other verify
        width even once it was captured at the right one.
        """
        candidate = self.graph if shared_indices else self.conditioned_graph
        if candidate is not None and candidate.width == width:
            return candidate
        return None

    def forward(self, tokens, parent, *, external_indices=None,
                full_logits=True):
        import torch
        from vllm.config import set_current_vllm_config
        from vllm_ascend.ascend_forward_context import (
            set_ascend_forward_context,
        )
        from vllm_ascend.attention.attention_v1 import AscendAttentionState

        tokens = device_tokens(tokens, self.device)
        width = tokens.shape[0]
        if width not in (1, 2, 4, 6, 8, 12, 16, 24, 32):
            raise ValueError("MTP requires 1, 2, 4, 8 or 16 tokens")
        if (parent.shape != (width, HIDDEN)
                or parent.dtype != torch.bfloat16
                or parent.device != self.device):
            raise ValueError("MTP parent must be device BF16 [M,6144]")
        if self.position + width > self.capacity:
            raise ValueError("MTP exceeds cache capacity")
        if self.attention_kernel == "gather_fia" and self.position < 2047:
            raise ValueError("Gather/FIA requires 2048 valid selected rows")
        if external_indices is not None and (
            external_indices.shape != (width, TOPK)
            or external_indices.dtype not in (torch.int32, torch.int64)
            or external_indices.device != self.device
        ):
            raise ValueError("Shared indices must be device integer [M,2048]")
        start = self.position
        with torch.npu.device(self.device), torch.inference_mode(), (
            set_current_vllm_config(self.config)
        ):
            graph = self._graph(width, external_indices is not None)
            metadata = CannModel._metadata(self, width,
                                           graphed=graph is not None)
            metadata[MAIN_CACHE].attn_state = AscendAttentionState.SpecDecoding
            self.input_ids.copy_(tokens)
            self.model.model.set_skip_topk(external_indices is not None)
            if external_indices is not None:
                self.indices_buffer[:width].copy_(external_indices)
            if graph is not None:
                pre_norm, recycle, logits, token_ids, indices = (
                    graph.forward(metadata, parent)
                )
            else:
                with set_ascend_forward_context(
                    metadata, self.config, num_tokens=width,
                    num_actual_tokens=width, model_instance=self.model,
                    is_draft_model=True,
                ):
                    pre_norm, recycle = self.model(
                        input_ids=self.input_ids, positions=self.positions,
                        hidden_states=parent, spec_step_idx=0,
                    )
                    logits = self.model.compute_logits(
                        pre_norm, spec_step_idx=0,
                    )
                    token_ids = logits.argmax(dim=-1)
                indices = self.indices_buffer[:width].clone()
        self.position += width
        return CannMtpStep(
            logits if full_logits else None, token_ids, pre_norm, recycle,
            indices, start,
        )

    def truncate(self, length):
        if type(length) is not int or not 0 <= length <= self.position:
            raise ValueError("Truncation requires an already written prefix")
        self.position = length

    def reset(self):
        self.position = 0
        self.model.model.set_skip_topk(False)
