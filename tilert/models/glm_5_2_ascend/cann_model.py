"""GLM-5.2 execution through the pinned vLLM-Ascend model and CANN."""

from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
import re

from tilert.pd_vllm.ascend_layout import (
    FULL_INDEX_LAYERS, INT8_INDEX_LAYERS, PAGE_SIZE, cache_layout,
)


VOCAB_SIZE = 154880
# Pinned sequence-length slots, one per forward a round can have in flight:
# the widest round is 32 draft steps plus the verify, the conditioned pass and
# the commit, and the host reads nothing until all of them are enqueued.
LENGTH_SLOTS = 64


def _token_values(tokens):
    import torch

    values = tuple(tokens) if isinstance(tokens, (list, tuple)) else (tokens,)
    has_tensors = False
    for value in values:
        if torch.is_tensor(value):
            has_tensors = True
        elif type(value) is not int or not 0 <= value < VOCAB_SIZE:
            raise ValueError("Decode requires GLM-5.2 vocabulary tokens")
    return values, has_tensors


def device_tokens(tokens, device):
    """Normalise decode tokens to one device integer tensor.

    An element may be a host integer, which a caller supplies and we check,
    or a device scalar from ``logits.argmax``, which is a vocabulary index
    by construction. Accepting both lets a round propose entirely on the
    device while still naming its pending token on the host.
    """
    import torch

    if torch.is_tensor(tokens):
        return tokens.reshape(-1).to(torch.int64)
    values, has_tensors = _token_values(tokens)
    if not has_tensors:
        return torch.tensor(values, dtype=torch.int64, device=device)
    return torch.stack([
        torch.as_tensor(value, dtype=torch.int64, device=device).reshape(())
        for value in values
    ])


def rope_buffers():
    """Name the module holding the process-wide MLA cos/sin buffers."""
    from vllm_ascend.ops import rotary_embedding

    return rotary_embedding


def rope_destinations(values, width):
    """Name the rope slices a replay reads outside the shared MLA buffers.

    A builder configured for speculative decoding keeps independent RoPE
    storage and copies the shared gather into it, so hoisted metadata still
    points at that copy and the copy has to be repeated every round.
    """
    rope = rope_buffers()
    seen = {(rope._cos_mla[:width].data_ptr(),
             rope._sin_mla[:width].data_ptr())}
    destinations = []
    for value in values:
        if value.cos is None or value.sin is None:
            continue
        key = (value.cos.data_ptr(), value.sin.data_ptr())
        if key in seen:
            continue
        seen.add(key)
        destinations.append((value.cos, value.sin))
    return destinations


def refresh_rope(positions, width, destinations):
    """Gather this round's cos/sin into the addresses a replay captured.

    One gather serves every builder group because they all alias the same
    module-global buffers. ``index_select`` writes the destination directly,
    which the builders reach in two dispatches instead of four.
    """
    import torch

    rope = rope_buffers()
    columns = rope._cos_cache.shape[-1]
    torch.index_select(rope._cos_cache, 0, positions,
                       out=rope._cos_mla[:width].view(width, columns))
    torch.index_select(rope._sin_cache, 0, positions,
                       out=rope._sin_mla[:width].view(width, columns))
    for cos, sin in destinations:
        cos.copy_(rope._cos_mla[:width])
        sin.copy_(rope._sin_mla[:width])


@dataclass
class CannStep:
    logits: object
    last_hidden: object
    token_ids: object


def create_config(checkpoint, capacity=9344, gpu_memory_utilization=0.85,
                  *, compile_target=False, super_kernel=False,
                  expert_parallel=False, graph_width=4, index_topk=None,
                  mtp_tokens=0, layers=None, tp_size=None):
    """Construct TP8+EP8 or TP16 config, optionally compiling the M4 target backbone.

    ``expert_parallel`` reshards the 256 routed experts over the active ranks
    (32 full experts per rank at TP8+EP8, 16 at EP16, intermediate stays 2048).
    """
    if super_kernel and not compile_target:
        raise ValueError("SuperKernel requires the compiled M4 target")
    if tp_size is None:
        tp_size = int(os.environ.get("WORLD_SIZE", "8" if expert_parallel else "16"))

    from vllm.config import CompilationMode, CUDAGraphMode
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform

    current_platform.pre_register_and_update()
    return EngineArgs(
        model=str(Path(checkpoint).resolve()), trust_remote_code=True,
        dtype="bfloat16", quantization="ascend",
        tensor_parallel_size=tp_size, pipeline_parallel_size=1,
        data_parallel_size=1,
        distributed_executor_backend="external_launcher",
        attention_config={"indexer_kv_dtype": "int8"},
        # Sparse attention costs 51 us a call before any position plus 17.61
        # a thousand picks, measured on one die, so the checkpoint's 2048 is
        # 36 of its 87 us. Overriding the config rather than editing it keeps
        # the checkpoint's hash stable, which the run's own record checks.
        hf_overrides={
            **({"index_topk": index_topk} if index_topk else {}),
            # A shortened backbone fits the residue a tenant leaves. The round
            # scales with the layer count, so absolute TPOT is not the full
            # model's -- but the ratio between two arms is, and a ratio is what
            # every lever here is measured as. Without this the engine needs
            # 47 GiB a die and waits for the machine to empty.
            **({"num_hidden_layers": layers} if layers else {}),
        },
        # Random weights, for an arm that measures timing rather than output.
        # A truncated backbone cannot load the real checkpoint: the loader
        # walks every tensor the index lists and raises on the first layer the
        # model does not have, and `is_pp_missing_parameter` does not cover it
        # at one pipeline stage.
        load_format="dummy" if layers else "auto",
        safetensors_load_strategy="lazy",
        kv_cache_dtype="auto", block_size=PAGE_SIZE,
        # AscendCompiler filters static token buckets by this sequence limit.
        max_num_seqs=max(4, graph_width) if super_kernel else 1,
        max_num_batched_tokens=max(16, graph_width),
        max_model_len=capacity,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=False, enable_chunked_prefill=True,
        enable_expert_parallel=expert_parallel,
        enforce_eager=not compile_target, async_scheduling=False,
        compilation_config={
            "mode": CompilationMode.VLLM_COMPILE,
            "cudagraph_mode": CUDAGraphMode.FULL_DECODE_ONLY,
            # Several sizes, not one. Replay cost rises with the captured
            # width while every operator in the graph is flat in the row
            # count, cold and warm alike -- so a round that verifies six rows
            # in a graph captured for twenty-four pays for eighteen rows of
            # nothing. With a list, each round replays the smallest graph that
            # covers its draft, and an adaptive drafter's proposals are mostly
            # short. One size costs 0.66 ms a row by the round model where the
            # operators cost between 0 and -0.02.
            "cudagraph_capture_sizes": (
                sorted(
                    {size for size in (2, 4, 8, 16, 24, 32, 48, 64)
                     if size <= graph_width} | {graph_width}
                )
                if os.environ.get("TRT_MULTI_CAPTURE_SIZES") == "1"
                else [graph_width]
            ),
            "max_cudagraph_capture_size": graph_width,
            "custom_ops": ["all"],
        } if compile_target else {},
        speculative_config=(
            {"method": "mtp", "num_speculative_tokens": mtp_tokens,
             "model": str(Path(checkpoint).resolve())}
            if mtp_tokens else None),
        kv_transfer_config=None,
        additional_config={
            "enable_flashcomm1": False, "enable_dsa_cp": False,
            "enable_shared_expert_dp": False,
            "enable_prefill_mc2": bool(expert_parallel),
            "multistream_overlap_shared_expert": bool(
                int(os.environ.get("TRT_OVERLAP_SHARED", "0"))
            ),
            "enable_fused_mc2": int(os.environ.get(
                "TRT_FUSED_MC2",
                "0" if tp_size == 8 else ("2" if expert_parallel else "0"),
            )),
            "ascend_compilation_config": {
                "enable_static_kernel": super_kernel,
                "enable_super_kernel": (
                    super_kernel
                    and os.environ.get("VLLM_ASCEND_ENABLE_SUPER_KERNEL", "1")
                    == "1"
                ),
                # Preserve BF16 norm rounding before INT8 quantization.
                "fuse_norm_quant": False,
            },
        },
    ).create_engine_config()


def _choose_collective_engine(aiv):
    """Expand collectives on the vector cores rather than the device AI CPU.

    A3 defaults to AI_CPU. At decode sizes the vector path is 1.6 to 1.9 times
    faster -- sixteen ranks over twelve rows of 6144 bf16: all_reduce 300.56 us
    to 183.40, reduce_scatter 318.09 to 177.41, all_gather 326.74 to 175.40 --
    and the gain is the per-call floor rather than transport, since the only
    case that barely moves is the one actually carrying a large payload.

    The variable is read once when HCCL loads, so this has to run before any
    communicator exists and cannot be changed afterwards; setting it from
    inside the process works, measured at 181.50 us against 290.44 unset.

    Determinism comes with it. The vector path reduces in its own order and is
    not reproducible on its own -- two runs of the same sixteen-rank all_reduce
    gave different digests -- while the AI CPU path is. `HCCL_DETERMINISTIC`
    restores reproducibility for 6 to 8 percent: 189 us against 175
    non-deterministic and 284 to 350 on the AI CPU. Bit-exactness is what every
    comparison in this project is checked against, so it is not optional.
    """
    if not aiv:
        return
    os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")
    os.environ.setdefault("HCCL_DETERMINISTIC", "true")


class CannModel:
    """One request, TP16, full-P injection and fixed-width target forwards.

    ``rank`` is the logical TP rank; ``device`` selects the process's NPU.
    The caller supplies all 16 visible devices without a visibility rotation.
    Worker construction and loading execute device code. No scheduler,
    speculative model, profiling pass, or graph capture is created here.
    """

    def __init__(
        self, checkpoint, device, rank, init_method="env://", capacity=9344,
        gpu_memory_utilization=0.85, *, compile_target=False,
        super_kernel=False, attention_kernel="sfa", fp32_head=False,
        expert_parallel=False, onesided_reduce=None, graph_width=4,
        index_topk=None, plain_moe_combine=False, routed_kernel=False,
        mla_tail=False, routed_shared_kernel=False, native_attention=False,
        aiv_collectives=True, rope_graph=False,
        gather_fia_max_width=None,
        mtp_tokens=0, layers=None,
        hoist_metadata=False, hoist_forward_context=False,
        packed_host_inputs=False,
    ):
        world_size = int(os.environ.get("WORLD_SIZE", "16"))
        if world_size not in (8, 16):
            raise ValueError("Launch 8 or 16 ranks")
        if type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError("GLM-5.2 requires a logical TP rank in [0, 16)")
        if world_size == 8:
            expert_parallel = True
        routed_kernel = routed_kernel or routed_shared_kernel
        if routed_kernel and expert_parallel:
            raise ValueError("The routed kernel requires TP16 with EP off")
        if native_attention and (mla_tail or attention_kernel != "sfa"):
            raise ValueError("Native attention includes the MLA tail and requires SFA")
        self._mla_tail_enabled = mla_tail
        self._native_attention_enabled = native_attention
        self._routed_workspace_pool = None
        self._routed_shared_workspace_pool = None
        self._mla_tail_workspace_pool = None
        self._native_attention_workspace_pool = None
        self._restore_routed = None
        self._restore_routed_shared = []
        self._restore_mla_tail = []
        self._restore_native_attention = []
        _choose_collective_engine(aiv_collectives)
        self.rope_graph = rope_graph
        self._rope_graphs = {}
        self._head_graphs = {}
        # None keeps the bound where the measurement is bit-exact; a caller
        # raising it accepts a bf16 unit in the last place on the attention
        # output, which is a question about tokens rather than floats.
        self.gather_fia_max_width = gather_fia_max_width
        if capacity < 128:
            raise ValueError("Capacity must be at least one 128-token page")
        if attention_kernel not in ("sfa", "gather_fia", "shared_fia"):
            raise ValueError("Unknown attention kernel")
        if attention_kernel == "shared_fia" and graph_width not in (8, 16, 32):
            raise ValueError("shared_fia supports target widths 8, 16 and 32")
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

        import torch
        import torch_npu  # noqa: F401
        from vllm.config import set_current_vllm_config
        from vllm_ascend.worker.worker import NPUWorker

        if hoist_forward_context:
            # Only the private vLLM-Ascend overlay carries this entry point, so
            # importing it unconditionally would make every default run depend
            # on the overlay being ahead of the installed package.
            from vllm_ascend.ascend_forward_context import hoist_mc2_mask

            hoist_mc2_mask(True)

        self.device = torch.device(device)
        if self.device.type != "npu" or self.device.index not in range(16):
            raise ValueError("device must name one of the 16 visible NPUs")
        self.rank = rank
        self.tp_size = world_size
        self.capacity = capacity
        self.position = 0
        self.attention_kernel = attention_kernel
        self.graph_width = graph_width
        self.hoist_metadata = hoist_metadata
        self.packed_host_inputs = packed_host_inputs
        self.fp32_head = fp32_head
        # A config-store endpoint turns the TP sum into a one-sided push: the
        # vendor collective is rendezvous rather than transfer. The decode
        # path measures 37.9 us a site over 146 sites, not the 63.5 us this
        # once recorded, so the headroom against the push's 22.3 us is 2.28 ms
        # a round. [critical-path-20261020]
        self.onesided_endpoint = onesided_reduce
        self._onesided = None
        self._restore_combine = None
        self.checkpoint = str(Path(checkpoint).resolve())
        torch.npu.set_device(self.device)
        self.config = create_config(
            self.checkpoint, capacity, gpu_memory_utilization,
            compile_target=compile_target, super_kernel=super_kernel,
            expert_parallel=expert_parallel, graph_width=graph_width,
            index_topk=index_topk, mtp_tokens=mtp_tokens, layers=layers,
            tp_size=world_size,
        )
        # A tree's rows need a per-row index set, and the operator's selection
        # width is what the checkpoint's `index_topk` sets -- the argument only
        # overrides it, so the effective value is the one in the config.
        self.index_picks = (
            index_topk
            or self.config.model_config.hf_config.index_topk
        )
        self.tree_mask = None
        self._restore_tree_mask = None
        self._attention_names = ()
        self._shared_attention = None
        with set_current_vllm_config(self.config):
            self.worker = NPUWorker(
                self.config, local_rank=self.device.index, rank=rank,
                distributed_init_method=init_method,
                is_driver_worker=rank == 0,
            )
            self.worker.init_device()
            if self.worker.device != self.device:
                raise ValueError("Worker remapped the requested physical NPU")
            if routed_kernel:
                # Before the weights load, because the hook has to see them
                # in the checkpoint's own packing and `load_model` converts
                # them to fractal NZ on the way in.
                from .routed_expert import install_before_conversion
                from vllm_ascend.quantization.methods.w4a8.w4a8 import (
                    AscendW4A8DynamicFusedMoEMethod,
                )

                self._routed_kernel_hook = install_before_conversion(
                    AscendW4A8DynamicFusedMoEMethod,
                    **({"weight_layout": "nz", "reuse_dispatch": True}
                       if routed_shared_kernel else {}))
            self.worker.load_model()
            self.runner = self.worker.model_runner
            self.model = self.runner.get_model()
            if self.fp32_head:
                from .cann_head import install_fp32_head

                install_fp32_head(self.model.lm_head,
                                  self.model.logits_processor)
            if self.runner.drafter is not None and mtp_tokens == 0:
                raise ValueError("Target must not load a draft model")
            self._bind_cache()
        self._allocate_inputs()
        self._install_tp_sum()
        if plain_moe_combine:
            from .moe_combine import install

            self._restore_combine = install(self.model)
        if routed_kernel:
            self._install_routed_kernel()
        if routed_shared_kernel:
            self._install_routed_shared_kernel()
        self._prepare_fused_workspaces(self.graph_width)

    def _install_routed_kernel(self):
        """Send the routed experts to the hand-written kernel.

        Installed by wrapping the quant method's weight conversion, because
        the kernel reads the checkpoint's own nibble packing and the
        conversion to fractal NZ has already happened by the time this object
        exists. The wrapper has to be in place before the model loads, so this
        raises if it is not -- a silent miss would run the vendor path and
        report the kernel's number.
        """
        restore, installed, pool = self._routed_kernel_hook
        if not installed:
            raise RuntimeError("the routed kernel hook took no layer; it must "
                               "be installed before the model loads")
        self._restore_routed = restore
        self._routed_workspace_pool = pool

    def _install_routed_shared_kernel(self):
        from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner

        from .routed_shared import WorkspacePool, install, load

        load()
        pool = WorkspacePool()
        runners = [module for module in self.model.modules()
                   if isinstance(module, AscendMoERunner)]
        if not runners:
            raise RuntimeError("No Ascend MoE runner found for routed/shared fusion")
        weights = self._routed_workspace_pool.layer_weights
        try:
            for runner in runners:
                routed_weights = weights[id(runner.routed_experts)]
                self._restore_routed_shared.append(
                    install(runner, routed_weights, pool),
                )
        except Exception:
            for restore in reversed(self._restore_routed_shared):
                restore()
            self._restore_routed_shared.clear()
            raise
        self._routed_shared_workspace_pool = pool

    def _prepare_fused_workspaces(self, width):
        """Reserve native storage before the wrapper enters graph capture."""
        import torch

        torch.npu.synchronize(self.device)
        pools = (
            (self._routed_workspace_pool, (6144,), torch.int8),
            (self._routed_shared_workspace_pool, (6144,), torch.int8),
            (self._mla_tail_workspace_pool,
             (64 // self.tp_size, 512), torch.bfloat16),
            (self._native_attention_workspace_pool, (6144,), torch.bfloat16),
        )
        for pool, shape, dtype in pools:
            if pool is None:
                continue
            for size in {1, width}:
                sample = torch.empty(size, *shape, dtype=dtype,
                                     device=self.device)
                pool.get(sample)
                if size == width:
                    pool.reserve_capture(sample)
        if self._native_attention_workspace_pool is not None:
            for size in {1, width}:
                self._native_attention_workspace_pool.live_slots(
                    self._input_buffers[size][2],
                )
        torch.npu.synchronize(self.device)

    def _install_tp_sum(self):
        from vllm.distributed import get_tp_group

        self._tp_sum_group = get_tp_group()
        self._tp_sum_original = self._tp_sum_group._all_reduce_out_place
        self._tp_sum_had_override = (
            "_all_reduce_out_place" in vars(self._tp_sum_group)
        )
        self._tp_sum_group._all_reduce_out_place = self._tp_reduce
        self.tp_reduction = "float32"
        if self.onesided_endpoint is not None:
            from .onesided_reduce import install

            self._onesided = install(self, self.onesided_endpoint)

    def _tp_reduce(self, value):
        import torch

        if value.dtype == torch.bfloat16:
            return self._tp_sum_original(value.float()).bfloat16()
        return self._tp_sum_original(value)

    def _restore_tp_sum(self):
        if self._tp_sum_group is None:
            return
        if self._tp_sum_had_override:
            self._tp_sum_group._all_reduce_out_place = self._tp_sum_original
        else:
            del self._tp_sum_group._all_reduce_out_place
        self._tp_sum_group = None
        self.tp_reduction = "original"

    def shutdown(self):
        """Restore the TP group's original reduction and stop the worker."""
        import torch

        if hasattr(getattr(torch.ops, "tilert", None),
                   "set_active_tokens_ptr"):
            torch.ops.tilert.set_active_tokens_ptr(
                self._dyn_lens_dev[self.graph_width], False, False,
            )
        for graph, _ in self._head_graphs.values():
            graph.reset()
        self._head_graphs.clear()
        for captured in self._rope_graphs.values():
            captured.graph.reset()
        self._rope_graphs.clear()
        if self.graph is not None:
            self.graph.close()
        self._restore_tp_sum()
        if self._restore_combine is not None:
            self._restore_combine()
            self._restore_combine = None
        if self._onesided is not None:
            self._onesided.close()
            self._onesided = None
        for restore in reversed(self._restore_routed_shared):
            restore()
        self._restore_routed_shared.clear()
        if self._restore_routed is not None:
            self._restore_routed()
            self._restore_routed = None
        for restore in reversed(self._restore_mla_tail):
            restore()
        self._restore_mla_tail.clear()
        for restore in reversed(self._restore_native_attention):
            restore()
        self._restore_native_attention.clear()
        self.worker.shutdown()

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

        self.layout = cache_layout(self.capacity)
        self.storage = torch.empty(
            self.layout["nbytes"], dtype=torch.uint8, device=self.device,
        )
        # Zero-initialize cache storage so masked/unwritten rows never hold NaN.
        self.storage.zero_()
        self.views = {}
        for name, region in self.layout["regions"].items():
            start = region["offset"]
            self.views[name] = self.storage[
                start:start + region["nbytes"]
            ].view(getattr(torch, region["dtype"])).view(region["shape"])
        self.cache = []
        for layer in range(78):
            entry = {
                "latent": self.views[f"mla_latent.{layer}"].view(-1, 512),
                "rope": self.views[f"mla_rope.{layer}"].view(-1, 64),
            }
            if layer in FULL_INDEX_LAYERS:
                entry["index_keys"] = self.views[
                    f"index.{layer}"
                ].view(-1, 128)
            if layer in INT8_INDEX_LAYERS:
                entry["index_scales"] = self.views[
                    f"index_scale.{layer}"
                ].view(-1)
            self.cache.append(entry)

        groups, caches = {}, {}
        main_layers, index_layers = set(), set()
        attention_names = []
        for name, spec in self.runner.get_kv_cache_spec().items():
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            if match is None or int(match[1]) >= 78:
                raise ValueError(f"Unexpected main cache layer: {name}")
            layer = int(match[1])
            if isinstance(spec, AscendSFAIndexerCacheSpec):
                index_layers.add(layer)
                values = (self.views[f"index.{layer}"],)
                if layer in INT8_INDEX_LAYERS:
                    values += (
                        self.views[f"index_scale.{layer}"].unsqueeze(-1),
                    )
                builder_type = AscendSFAIndexerMetadataBuilder
            elif isinstance(spec, AscendMLAAttentionSpec):
                main_layers.add(layer)
                attention_names.append(name)
                values = (
                    self.views[f"mla_latent.{layer}"],
                    self.views[f"mla_rope.{layer}"],
                )
                builder_type = AscendSFAMetadataBuilder
            else:
                raise ValueError(f"Unexpected KV specification: {spec}")
            caches[name] = values
            groups.setdefault((builder_type, spec), []).append(name)
        # Against the backbone's own depth, not a literal 78: a truncated
        # backbone binds fewer caches and the check's intent is that every
        # layer the model has is bound, not that the model has 78.
        depth = self.config.model_config.hf_config.num_hidden_layers
        wanted_index = {layer for layer in FULL_INDEX_LAYERS if layer < depth}
        if main_layers != set(range(depth)) or index_layers != wanted_index:
            raise ValueError(
                f"Expected {depth} main and {len(wanted_index)} indexer cache "
                f"layers, found {len(main_layers)} and {len(index_layers)}")
        bind_kv_cache_to_layers(
            caches, self.config.compilation_config.static_forward_context,
        )
        if self.attention_kernel == "gather_fia":
            from .cann_attention import install_gather_fia

            context = self.config.compilation_config.static_forward_context
            for name in attention_names:
                install_gather_fia(context[name].impl, self.graph_width,
                                   max_width=self.gather_fia_max_width)
        elif self.attention_kernel == "shared_fia":
            from .cann_attention import SharedKVAttention

            context = self.config.compilation_config.static_forward_context
            self._shared_attention = SharedKVAttention(
                self.graph_width, self.layout["capacity_tokens"], self.device,
            )
            for name in attention_names:
                self._shared_attention.bind(context[name].impl)
        self._attention_names = tuple(attention_names)
        if self._mla_tail_enabled:
            from .mla_tail import WorkspacePool, install, load

            load()
            self._mla_tail_workspace_pool = WorkspacePool()
            context = self.config.compilation_config.static_forward_context
            self._restore_mla_tail = [
                install(context[name].impl, self._mla_tail_workspace_pool)
                for name in attention_names
            ]
        if self._native_attention_enabled:
            from .attention_native import WorkspacePool, install, load

            load()
            self._native_attention_workspace_pool = WorkspacePool()
            context = self.config.compilation_config.static_forward_context
            try:
                for name in attention_names:
                    self._restore_native_attention.append(install(
                        context[name], self._native_attention_workspace_pool,
                    ))
            except Exception:
                for restore in reversed(self._restore_native_attention):
                    restore()
                self._restore_native_attention.clear()
                self._native_attention_workspace_pool = None
                raise
        self.builders = [
            (names, kind(spec, names, self.config, self.device))
            for (kind, spec), names in groups.items()
        ]
        self.cache_names = sorted(caches)

    def _allocate_inputs(self):
        import torch

        widths = (1, 2, 4, 6, 8, 12, 16, 24, 32)
        self._host_inputs = None
        if getattr(self, "packed_host_inputs", False):
            from .host_inputs import HostInputs

            self._host_inputs = HostInputs(
                self.device, widths, slots=LENGTH_SLOTS,
            )
        self.graph = None
        self._hoisted = {}
        self._input_buffers = {}
        for width in widths:
            if self._host_inputs is None:
                tokens = torch.empty(
                    width, dtype=torch.int32, device=self.device,
                )
                positions = torch.empty(
                    width, dtype=torch.long, device=self.device,
                )
                slots = torch.empty(
                    width, dtype=torch.int32, device=self.device,
                )
            else:
                tokens, positions, slots, _ = self._host_inputs.views(width)
            query_cpu = torch.tensor([0, width], dtype=torch.int32)
            self._input_buffers[width] = (
                tokens, positions, slots, query_cpu, query_cpu.to(self.device),
            )
        (self.input_ids, self.positions, self.slots,
         self.query_cpu, self.query) = self._input_buffers[1]
        self.seq_lens = torch.empty(1, dtype=torch.int32, device=self.device)
        # A copy from pageable host memory drains the stream before it stages,
        # and a round enqueues every forward before it reads anything, so each
        # forward's copy pays for the work queued ahead of it: measured at 139
        # microseconds behind one draft graph and 1141 behind the target graph,
        # against 48 to 71 from pinned memory. A slot per enqueued forward,
        # because the host writes the next one long before the engine reads
        # this one.
        self._lengths = torch.zeros(
            LENGTH_SLOTS, dtype=torch.int32).pin_memory()
        self._length_slot = 0
        self.block_table = torch.arange(
            self.layout["capacity_tokens"] // PAGE_SIZE,
            dtype=torch.int32, device=self.device,
        ).unsqueeze(0)
        self._dynamic_mc2_mask = (
            os.environ.get("TRT_DYNAMIC_MC2_MASK", "1") == "1"
        )
        self._dynamic_sfa_len = (
            os.environ.get("TRT_DYNAMIC_SFA_LEN", "1") == "1"
        )
        self._dynamic_sfa_max = int(
            os.environ.get("TRT_DYNAMIC_SFA_MAX", "32")
        )
        self._dynamic_idx_max = int(
            os.environ.get("TRT_DYNAMIC_IDX_MAX", "16")
        )
        self._mc2_host_masks = {}
        self._last_mc2_active = {}
        self._dyn_lens_dev = {}
        for width in widths:
            dyn_buf = torch.zeros(2, 8, dtype=torch.int32, device=self.device)
            dyn_buf[0, 0] = width
            dyn_buf[0, 1] = width
            self._dyn_lens_dev[width] = dyn_buf
        self._sfa_len_slots = tuple(
            torch.zeros(2, 8, dtype=torch.int32).pin_memory()
            for _ in range(LENGTH_SLOTS)
        )
        self._sfa_len_idx = 0
        if self._dynamic_mc2_mask:
            for width in widths:
                table = torch.zeros(
                    width + 1, width, dtype=torch.bool, pin_memory=True,
                )
                for k in range(1, width + 1):
                    table[k, :k] = True
                self._mc2_host_masks[width] = tuple(table.unbind())

    def inject(self, prefixes):
        """Copy all P main rows; the next consumed token has position P."""
        import torch

        if len(prefixes) != 78:
            raise ValueError("Full-P injection requires all 78 main layers")
        length = prefixes[0]["latent"].shape[0]
        if not 0 <= length < self.capacity:
            raise ValueError("Prefix leaves no room for a decode token")
        for layer, (source, target) in enumerate(zip(prefixes, self.cache)):
            source = {key: value for key, value in source.items()
                      if value is not None}
            if source.keys() != target.keys():
                raise ValueError(f"Layer {layer}: cache fields differ")
            for name, destination in target.items():
                value = source[name]
                if (value.shape != destination[:length].shape
                        or value.dtype != destination.dtype):
                    raise ValueError(f"Layer {layer}: invalid {name} cache")
        with torch.npu.device(self.device), torch.inference_mode():
            for source, target in zip(prefixes, self.cache):
                for name, destination in target.items():
                    destination[:length].copy_(source[name])
        self.position = length

    def _metadata(self, width=1, *, graphed=False, depths=None,
                  host_values=None):
        import torch
        from vllm_ascend.attention.attention_v1 import AscendAttentionState
        from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

        (self.input_ids, self.positions, self.slots,
         self.query_cpu, self.query) = self._input_buffers[width]
        host_inputs = getattr(self, "_host_inputs", None)
        if host_inputs is not None:
            self.seq_lens = host_inputs.views(width)[3]
        if host_values is not None:
            length_cpu, positions_cpu = host_inputs.stage(
                host_values, self.position, depths,
            )
        else:
            slot = self._length_slot % LENGTH_SLOTS
            self._length_slot += 1
            length_cpu = self._lengths[slot:slot + 1]
            length_cpu[0] = self.position + width
            slots_cpu = torch.arange(
                self.position, self.position + width, dtype=torch.long,
            )
            # A tree stores rows linearly and assigns RoPE positions by depth.
            positions_cpu = slots_cpu if depths is None else torch.tensor(
                [self.position + depth for depth in depths], dtype=torch.long,
            )
            self.seq_lens.copy_(length_cpu, non_blocking=True)
            if width == 1:
                self.slots.fill_(self.position)
                self.positions.fill_(self.position)
            else:
                self.slots.copy_(slots_cpu)
                self.positions.copy_(positions_cpu)
        computed_cpu = torch.tensor([self.position], dtype=torch.int32)
        hoisted = self._hoisted.get(width) if graphed else None
        if hoisted is not None:
            if (self.graph is not None and self.graph.capture_count
                    and host_values is not None and depths is None
                    and width > 1):
                k = width
                p0 = host_values[0]
                while k > 1 and host_values[k - 1] == p0:
                    k -= 1
                active = 1 if k == 1 else min(width, k + 1)
            else:
                active = width
            self._active_rows = active
            if (self._dynamic_mc2_mask and self.graph is not None
                    and self.graph.capture_count
                    and self._last_mc2_active.get(width, width) != active):
                from vllm_ascend.ascend_forward_context import (
                    _hoisted_mc2_masks, get_mc2_mask,
                )
                mask = (
                    _hoisted_mc2_masks.get((width, width))
                    if _hoisted_mc2_masks is not None
                    else None
                )
                if mask is None:
                    reserved = get_mc2_mask()
                    if reserved is not None:
                        mask = reserved[:width]
                if mask is not None:
                    mask.copy_(
                        self._mc2_host_masks[width][active],
                        non_blocking=True,
                    )
                self._last_mc2_active[width] = active
            if self._dynamic_sfa_len:
                sfa_a = active if active <= self._dynamic_sfa_max else width
                idx_a = active if active <= self._dynamic_idx_max else width
                s_idx = self._sfa_len_idx % LENGTH_SLOTS
                self._sfa_len_idx += 1
                s_len = self._sfa_len_slots[s_idx]
                s_len[0, 0] = sfa_a
                s_len[1, 0] = self.position + sfa_a
                s_len[0, 1] = idx_a
                s_len[1, 1] = self.position + idx_a
                self._dyn_lens_dev[width].copy_(s_len, non_blocking=True)
            metadata, values, _ = hoisted
            for value in values:
                value.max_seq_len = self.position + width
                value.seq_lens_cpu = length_cpu
            return metadata
        state = (AscendAttentionState.DecodeOnly if width == 1
                 else AscendAttentionState.ChunkedPrefill)
        common = AscendCommonAttentionMetadata(
            query_start_loc=self.query, query_start_loc_cpu=self.query_cpu,
            seq_lens=self.seq_lens, seq_lens_cpu=length_cpu,
            seq_lens_cpu_upper_bound=length_cpu,
            num_computed_tokens_cpu=computed_cpu,
            _seq_lens_cpu=length_cpu,
            _num_computed_tokens_cpu=computed_cpu,
            num_reqs=1, num_actual_tokens=width, max_query_len=width,
            max_seq_len=self.position + width,
            block_table_tensor=self.block_table, slot_mapping=self.slots,
            positions=self.positions, positions_cpu=positions_cpu,
            causal=True,
            is_prefilling=torch.tensor([False]),
            decode_token_per_req=1, actual_seq_lengths_q=[width],
            attn_state=state, num_input_tokens=width,
        )
        result = {}
        for names, builder in self.builders:
            metadata = builder.build(0, common)
            if (hasattr(metadata, "actual_seq_lengths_query")
                    and self._dynamic_idx_max > 0):
                c8_meta = replace(
                    metadata,
                    cum_query_lens=metadata.cum_query_lens.clone(),
                    seq_lens=metadata.seq_lens.clone(),
                    actual_seq_lengths_query=(
                        metadata.actual_seq_lengths_query.clone()
                    ),
                    actual_seq_lengths_key=(
                        metadata.actual_seq_lengths_key.clone()
                    ),
                )
                c8_meta._is_c8_idx = True
                for name in names:
                    m = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
                    result[name] = (
                        c8_meta
                        if m and int(m[1]) in INT8_INDEX_LAYERS
                        else metadata
                    )
            else:
                result.update((name, metadata) for name in names)
        if graphed and self.hoist_metadata:
            values = list({id(value): value for value in result.values()}
                          .values())
            if self._dynamic_sfa_len:
                dyn = self._dyn_lens_dev[width]
                pos_w = self.position + width
                s_idx = self._sfa_len_idx % LENGTH_SLOTS
                self._sfa_len_idx += 1
                s_len = self._sfa_len_slots[s_idx]
                s_len[0, 0] = width
                s_len[1, 0] = pos_w
                s_len[0, 1] = width
                s_len[1, 1] = pos_w
                dyn.copy_(s_len, non_blocking=True)
                if hasattr(getattr(torch.ops, "tilert", None),
                           "set_active_tokens_ptr"):
                    en_shmem = (
                        os.environ.get("TRT_DYNAMIC_SHMEM", "1") == "1"
                    )
                    en_idx = (
                        os.environ.get("TRT_DYNAMIC_IDXPREP", "1") == "1"
                    )
                    torch.ops.tilert.set_active_tokens_ptr(
                        dyn, en_shmem, en_idx,
                    )
                q_view = dyn[0, :1]
                kv_view = dyn[1, :1]
                iq_view = dyn[0, 1:2]
                ikv_view = dyn[1, 1:2]
                for value in values:
                    if hasattr(value, "actual_seq_lengths_query"):
                        if getattr(value, "_is_c8_idx", False):
                            value.cum_query_lens = iq_view
                            value.seq_lens = ikv_view
                            value.actual_seq_lengths_query = iq_view
                            value.actual_seq_lengths_key = ikv_view
                    else:
                        value.cum_query_lens = q_view
                        value.seq_lens = kv_view
            self._hoisted[width] = (
                result, values, rope_destinations(values, width),
            )
        return result

    def refresh_hoisted_rope(self, width, positions):
        """Rewrite rope values for a replay that reuses hoisted metadata.

        Nothing else the metadata holds depends on the round: every other
        field is either a view of a buffer the caller refreshes by H2D copy
        or is invariant for the captured width.

        The refresh itself is 530 eager `copy_` calls a round -- one pair a
        speculative builder's private RoPE storage -- so it is captured when
        asked, which leaves the work and removes the dispatch.
        """
        hoisted = self._hoisted.get(width)
        if hoisted is None:
            return
        if not self.rope_graph:
            refresh_rope(positions, width, hoisted[2])
            return
        captured = self._rope_graphs.get(width)
        slots_i64 = (
            self.graph.slots_i64
            if self.graph is not None and self.graph.width == width
            else None
        )
        if captured is None or (
            slots_i64 is not None and not captured.handles_slots
        ):
            from .rope_graph import RopeGraph

            captured = RopeGraph(
                positions, width, hoisted[2], self.slots, slots_i64
            )
            self._rope_graphs[width] = captured
            if self.graph is not None and self.graph.width == width:
                self.graph._rope_handles_slots = captured.handles_slots
        captured.refresh()

    def enable_graph(self, width=1):
        """Capture one query width; other widths retain eager execution."""
        from .cann_graph import CannGraph

        if self.graph is None:
            self.graph = CannGraph(self, width)
        elif self.graph.width != width:
            raise ValueError("The model already captures another width")

    def _prepare_inputs(self, tokens, *, depths=None):
        """Stage persistent model inputs and build their consumer metadata."""
        import torch

        host_values = None
        if (getattr(self, "_host_inputs", None) is not None
                and not torch.is_tensor(tokens)):
            values, has_tensors = _token_values(tokens)
            if not has_tensors:
                host_values = values
        if host_values is None:
            tokens = device_tokens(tokens, self.device)
            width = tokens.shape[0]
        else:
            width = len(host_values)
        if width not in (1, 2, 4, 6, 8, 12, 16, 24, 32):
            raise ValueError("Unsupported verification width")
        if self.position + width > self.capacity:
            raise ValueError("Decode exceeds cache capacity")
        uses_fia = self.attention_kernel == "gather_fia" or (
            self.attention_kernel == "shared_fia" and width == self.graph_width
        )
        if uses_fia and self.position < 2047:
            raise ValueError("FIA requires 2048 valid selected rows")
        use_graph = self.graph is not None and self.graph.width == width
        metadata = self._metadata(
            width, graphed=use_graph, depths=depths, host_values=host_values,
        )
        if use_graph:
            self.refresh_hoisted_rope(width, self.positions)
        if host_values is None:
            self.input_ids.copy_(tokens)
        return width, metadata

    def verify(self, tokens, *, full_logits=True, depths=None):
        """Consume causal rows, with optional tree depths for RoPE."""
        import torch
        from vllm.config import set_current_vllm_config
        from vllm_ascend.ascend_forward_context import (
            set_ascend_forward_context,
        )

        with torch.npu.device(self.device), torch.inference_mode():
            width, metadata = self._prepare_inputs(tokens, depths=depths)
            use_graph = self.graph is not None and self.graph.width == width
            if use_graph and self.graph.capture_count:
                hidden = self.graph.forward(metadata)
                processor = getattr(self.model, "logits_processor", None)
                if (not full_logits and self.fp32_head
                        and self._onesided is not None
                        and processor is not None
                        and hasattr(getattr(torch.ops, "tilert", None),
                                    "onesided_top_tokens")):
                    if hasattr(self.model.compute_logits, "__wrapped__"):
                        processor._top_tokens_mode = True
                        try:
                            token_ids = self.model.compute_logits(hidden)
                        finally:
                            processor._top_tokens_mode = False
                    else:
                        head_entry = self._head_graphs.get(width)
                        if head_entry is None:
                            processor._top_tokens_mode = True
                            try:
                                self.model.compute_logits(hidden)
                                torch.npu.synchronize()
                                head_graph = torch.npu.NPUGraph()
                                with torch.npu.graph(head_graph):
                                    token_ids = self.model.compute_logits(
                                        hidden,
                                    )
                                torch.npu.synchronize()
                            finally:
                                processor._top_tokens_mode = False
                            head_entry = (head_graph, token_ids)
                            self._head_graphs[width] = head_entry
                        head_graph, token_ids = head_entry
                        head_graph.replay()
                    logits = None
                else:
                    logits = self.model.compute_logits(hidden)
                    token_ids = logits.argmax(dim=-1)
                self.position += width
                return CannStep(
                    logits if full_logits else None, hidden, token_ids,
                )
            with set_current_vllm_config(self.config), (
                set_ascend_forward_context(
                    metadata, self.config, num_tokens=width,
                    num_actual_tokens=width,
                    model_instance=self.model,
                )
            ):
                hidden = (self.model(self.input_ids, self.positions)
                          if not use_graph
                          else self.graph.forward(metadata))
                processor = getattr(self.model, "logits_processor", None)
                if (not full_logits and self.fp32_head
                        and self._onesided is not None
                        and processor is not None
                        and hasattr(getattr(torch.ops, "tilert", None),
                                    "onesided_top_tokens")):
                    processor._top_tokens_mode = True
                    try:
                        token_ids = self.model.compute_logits(hidden)
                    finally:
                        processor._top_tokens_mode = False
                    logits = None
                else:
                    logits = self.model.compute_logits(hidden)
                    token_ids = logits.argmax(dim=-1)
            self.position += width
            return CannStep(
                logits if full_logits else None, hidden, token_ids,
            )

    def install_tree_mask(self):
        """Give a draft tree the visibility its branches need.

        `shared_fia` already composes the indexer's selection with a causal
        mask, so a tree only has to replace the causal half -- the indexer
        keeps choosing the prefix at the width the checkpoint sets. The other
        kernels take `sparse_indices` instead, where the tree has to name
        every position a row may see, which means the whole prefix has to fit
        in `index_topk`.

        Idempotent: the round executor asks once a request.
        """
        if self.tree_mask is None:
            if self._shared_attention is not None:
                from .tree_mask import ancestor_bitmap

                capacity = self.layout["capacity_tokens"]
                shared = self._shared_attention

                def write(parents, prefix_length, device):
                    shared.visibility = ancestor_bitmap(
                        parents, prefix_length, capacity, device,
                    )
                    return shared.visibility

                self.tree_mask = write
            else:
                from .tree_attention import install

                context = self.config.compilation_config.static_forward_context
                self.tree_mask, self._restore_tree_mask = install(
                    [context[name].impl for name in self._attention_names],
                    self.index_picks,
                )
        return self.tree_mask

    def indexer_buffers(self):
        """The selected positions each indexer layer last wrote.

        Only one layer in `index_topk_freq` runs the indexer and the rest
        reuse its result, so the buffers alias; de-duplicating by storage
        gives the distinct selections. What they are wanted for is whether
        those positions cluster, which decides whether an eight-token sparse
        block covers them without attending to much that was never selected.
        """
        buffers = {}
        # `named_children` rather than `items`: the pinned vendor holds the
        # layers in a `ModuleList`, which has no `items`, so this accessor
        # raised on every call and `--dump-indices` had never run.
        for name, layer in self.model.model.layers.named_children():
            attention = getattr(layer, "self_attn", None)
            indices = getattr(getattr(attention, "mla_attn", None),
                              "topk_indices_buffer", None)
            if indices is not None:
                buffers.setdefault(indices.data_ptr(), (name, indices))
        return [value for value in buffers.values()]

    def host_tokens(self, *groups):
        """Read several token groups to the host in one transfer.

        A round issues every kernel before calling this, so this is its only
        synchronisation. The host needs these values to commit cache lengths
        and to deliver text.
        """
        import torch

        tensors = [device_tokens(group, self.device) for group in groups]
        read = (tensors[0] if len(tensors) == 1 else torch.cat(tensors)).tolist()
        result, start = [], 0
        for tensor in tensors:
            width = tensor.shape[0]
            result.append(tuple(int(value) for value in
                                read[start:start + width]))
            start += width
        return tuple(result)

    def step(self, token):
        return self.verify((token,), full_logits=False)

    def predict(self, normalized_parent, *, full_logits=False):
        """Project transferred hidden[P-1] without changing the KV length."""
        import torch
        from vllm.config import set_current_vllm_config

        if (normalized_parent.shape != (1, 6144)
                or normalized_parent.dtype != torch.bfloat16
                or normalized_parent.device != self.device):
            raise ValueError("Parent must be normalized device BF16 [1,6144]")
        with torch.npu.device(self.device), torch.inference_mode(), (
            set_current_vllm_config(self.config)
        ):
            logits = self.model.compute_logits(normalized_parent)
            return CannStep(logits if full_logits else None, normalized_parent,
                            logits.argmax(dim=-1))

    def compact(self, start, rows):
        """Gather written rows `start + rows[j]` into `start + j`.

        A chain's accepted prefix is already the rows it occupies, so nothing
        needed this until a tree: there the accepted path is scattered across
        the verification rows, and the cache the next round attends to has to
        hold those tokens contiguously.

        `rows[j] >= j` because a tree's parents precede its children, so a
        node at depth `j` cannot sit before row `j` -- which makes the gather
        safe to do in place and in order. The drafter expands best-first, so
        the likeliest path usually is `0, 1, 2, ...` and this does nothing.
        """
        import torch

        if rows != sorted(rows) or any(row < j for j, row in enumerate(rows)):
            raise ValueError("A path's rows must rise and not precede depth")
        if rows == list(range(len(rows))):
            return
        with torch.npu.device(self.device), torch.inference_mode():
            index = torch.tensor([start + row for row in rows],
                                 dtype=torch.long, device=self.device)
            for layer in self.cache:
                for destination in layer.values():
                    destination[start:start + len(rows)] = (
                        destination.index_select(0, index)
                    )

    def truncate(self, length):
        """Retain written rows [0, length); later writes replace the tail."""
        if type(length) is not int or not 0 <= length <= self.position:
            raise ValueError("Truncation requires an already written prefix")
        self.position = length

    def reset(self):
        """Exclude prior rows from attention without replaying the prompt."""
        self.position = 0

    def source_metadata(self):
        import inspect
        from vllm_ascend.ascend_forward_context import (
            set_ascend_forward_context,
        )

        objects = [type(self), type(self.worker), type(self.runner),
                   type(self.model), type(self.model.model),
                   set_ascend_forward_context]
        objects.extend(type(builder) for _, builder in self.builders)
        if getattr(self, "_host_inputs", None) is not None:
            objects.append(type(self._host_inputs))
        if self.attention_kernel == "gather_fia":
            from .cann_attention import gathered_fia

            objects.append(gathered_fia)
        elif self.attention_kernel == "shared_fia":
            from .cann_attention import SharedKVAttention

            objects.append(SharedKVAttention)
        elif self.attention_kernel == "sfa":
            from vllm_ascend.device.device_op import DeviceOperator

            objects.append(DeviceOperator.execute_sparse_flash_attention_process)
        if self.fp32_head:
            from .cann_head import fp32_projection

            objects.append(fp32_projection)
        native_libraries = []
        if self._routed_workspace_pool is not None:
            from . import routed_expert
            from vllm_ascend.quantization.methods.w4a8.w4a8 import (
                AscendW4A8DynamicFusedMoEMethod,
            )

            objects.extend((routed_expert, AscendW4A8DynamicFusedMoEMethod))
            native_libraries.append(
                routed_expert.LIBRARY_DIR / "routed_int4_torch.so",
            )
        if self._routed_shared_workspace_pool is not None:
            from . import routed_shared
            from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
            from vllm_ascend.ops.fused_moe.routed_experts import (
                AscendRoutedExperts,
            )

            objects.extend((routed_shared, AscendMoERunner, AscendRoutedExperts))
            native_libraries.extend(routed_shared.library_paths())
        if self._mla_tail_workspace_pool is not None:
            from . import mla_tail

            objects.append(mla_tail)
            native_libraries.append(mla_tail.LIBRARY_DIR / "mla_tail_torch.so")
        if self._native_attention_workspace_pool is not None:
            from . import attention_native, mla_tail

            objects.extend((attention_native, mla_tail))
            native_libraries.extend(attention_native.library_paths())
        if (self._routed_shared_workspace_pool is not None
                or self._mla_tail_workspace_pool is not None
                or self._native_attention_workspace_pool is not None):
            from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import (
                AscendW8A8DynamicLinearMethod,
            )

            objects.append(AscendW8A8DynamicLinearMethod)
        if (self._mla_tail_workspace_pool is not None
                or self._native_attention_workspace_pool is not None):
            context = self.config.compilation_config.static_forward_context
            objects.extend(type(layer.impl) for layer in context.values()
                           if hasattr(layer, "impl"))
        if not self.config.model_config.enforce_eager:
            from vllm.compilation.decorators import support_torch_compile
            from vllm.compilation.wrapper import (
                TorchCompileWithNoGuardsWrapper,
            )
            from vllm_ascend.compilation.compiler_interface import (
                AscendCompiler,
            )

            objects.extend((AscendCompiler, support_torch_compile,
                            TorchCompileWithNoGuardsWrapper))
        if self.graph is not None:
            objects.append(type(self.graph))
        paths = {Path(inspect.getfile(inspect.unwrap(value))).resolve()
                 for value in objects}
        paths.update(path.resolve() for path in native_libraries)
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(paths)}
