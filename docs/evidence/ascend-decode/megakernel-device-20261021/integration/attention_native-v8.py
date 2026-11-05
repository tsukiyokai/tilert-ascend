"""Run TP16 backbone attention through the native indexed or shared entry.

Weights are packed after cache binding. Workspaces belong to a device/stream
and exact row/cache geometry; reserve them under the forward context before
graph capture. The model retains cross-layer top-k and the original TP sum.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from types import MethodType

from .mla_tail import pack_output_weights, pack_value_weights


HIDDEN = 6144
MAX_TOKENS = 32
_ROOT = Path(__file__).resolve().parents[3] / "tests/glm52/whole_layer"
INDEXED_LIBRARY = Path(os.environ.get(
    "TILERT_INDEXED_ATTENTION_LIBRARY",
    _ROOT / "attention_indexed_build-scale-order/whole_layer_indexed_attention.so",
))
SHARED_LIBRARY = Path(os.environ.get(
    "TILERT_SHARED_ATTENTION_LIBRARY",
    _ROOT / "attention_full_build-dequant/whole_layer_full_attention.so",
))


def library_paths():
    return INDEXED_LIBRARY, SHARED_LIBRARY


def load():
    import torch

    for path in library_paths():
        torch.ops.load_library(str(path))


def pack_indexer_weights(query, key):
    """Pack logical WQ [2048,4096] and merged WK [160,6144]."""
    import torch

    _require_tensor(query, torch.int8, (2048, 4096), "indexer WQ")
    _require_tensor(key, torch.bfloat16, (160, HIDDEN), "indexer WK")
    query = query.reshape(2048, 16, 256).permute(1, 0, 2)
    key = key.T.reshape(HIDDEN, 5, 32).permute(1, 0, 2)
    return query.contiguous(), key.contiguous()


def _require_tensor(value, dtype, shape, name):
    if value.dtype != dtype or tuple(value.shape) != shape:
        raise ValueError(f"{name} requires {dtype} {shape}")


def _nd(value):
    import torch_npu

    return torch_npu.npu_format_cast(value.detach().contiguous(), 2)


def _float_scale(value, count, name):
    import torch

    if value.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"{name} requires FP32 or BF16 channel scales")
    result = _nd(value.float())
    _require_tensor(result, torch.float32, (count,), name)
    return result


def _indexer_backend(indexer):
    from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
    from vllm_ascend.ops.mla import IndexerWrapper

    if (type(indexer) is not IndexerWrapper
            or type(indexer.impl) is not AscendSFAIndexerBackend):
        raise ValueError("Native attention requires IndexerWrapper with AscendSFAIndexerBackend")
    return indexer.impl


def _validate_impl(impl):
    import torch
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, PreprocessType
    from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import (
        AscendW8A8DynamicLinearMethod,
    )

    parallel = impl.vllm_config.parallel_config
    expected = {
        "tp_size": 16, "local_num_heads": 4, "q_lora_rank": 2048,
        "kv_lora_rank": 512, "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64, "v_head_dim": 256,
    }
    if (type(impl) is not AscendSFAImpl
            or any(getattr(impl, name) != value
                   for name, value in expected.items())
            or impl.preprocess_type != PreprocessType.PROLOG_V3
            or impl._quant_type is not AscendW8A8DynamicLinearMethod
            or impl.enable_sp or impl.is_pcp_decode_sharded
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or impl.uses_packed_sfa_main_cache
            or impl.kv_sharing_target_layer_name is not None
            or impl.rl_weight_update_enabled
            or impl.vllm_config.kv_transfer_config is not None
            or getattr(impl.vllm_config, "lora_config", None) is not None
            or getattr(impl.vllm_config.model_config, "enable_sleep_mode", False)):
        raise ValueError("Native attention requires fixed TP16 BF16-cache Prolog V3 without CP/SP or weight reload")
    projection = impl.o_proj
    scheme = getattr(projection.quant_method, "quant_method", None)
    if (type(scheme) is not AscendW8A8DynamicLinearMethod
            or scheme.act_quant_type != torch.int8
            or projection.tp_size != 16 or not projection.input_is_parallel
            or getattr(projection, "bias", None) is not None
            or impl.g_proj is not None):
        raise ValueError("Native attention requires an ungated TP16 W8A8_DYNAMIC output projection")
    if impl.indexer is None:
        if not impl.skip_topk or impl.topk_indices_buffer is None:
            raise ValueError("Shared attention requires persistent index-cache selections")
        return "shared"
    indexer = _indexer_backend(impl.indexer)
    scheme = getattr(indexer.wq_b.quant_method, "quant_method", None)
    if (type(scheme) is not AscendW8A8DynamicLinearMethod
            or scheme.act_quant_type != torch.int8
            or indexer.n_head != 32 or indexer.head_dim != 128
            or indexer.q_lora_rank != 2048 or indexer.topk_tokens != 2048
            or indexer.qk_rope_head_dim != 64 or indexer.is_rope_neox_style
            or indexer.k_norm.eps != 1e-6
            or indexer.enable_sparse_li_c4
            or indexer._pcp_active or indexer._dsa_cp_active
            or indexer._is_pcp_decode_sharded
            or getattr(indexer.wq_b, "bias", None) is not None
            or getattr(indexer.wk_weights_proj, "bias", None) is not None):
        raise ValueError("Native indexer requires replicated W8A8 Q, BF16 WK, RoPE64 and 2048 selections")
    if indexer.enable_sparse_li_c8:
        if (indexer.c8_k_cache_dtype != torch.int8
                or indexer.c8_k_scale_cache_dtype != torch.float16):
            raise ValueError("Native C8 indexer requires INT8 keys and FP16 scales")
        return "int8"
    return "bf16"


def _pack_weights(impl, mode):
    import torch
    import torch_npu

    main = [impl.weight_dq.detach(), impl.weight_uq_qr.detach(),
            _nd(impl.W_UK_T), impl.weight_dkv_kr.detach(),
            _nd(impl.dequant_scale_w_dq),
            _nd(impl.dequant_scale_w_uq_qr),
            _nd(impl.dequant_scale_w_dkv_kr),
            _nd(impl.q_a_layernorm.weight), _nd(impl.kv_a_layernorm.weight)]
    shapes = ((HIDDEN, 2048), (2048, 1024), (4, 192, 512),
              (HIDDEN, 576), (1, 2048), (1, 1024), (1, 576),
              (2048,), (512,))
    types = (torch.int8, torch.int8, torch.bfloat16, torch.int8,
             torch.float32, torch.float32, torch.float32,
             torch.bfloat16, torch.bfloat16)
    for index, (value, dtype, shape) in enumerate(zip(main, types, shapes)):
        _require_tensor(value, dtype, shape, f"main weight {index}")
        if torch_npu.get_npu_format(value) != (29 if index in (0, 1, 3) else 2):
            raise ValueError(f"main weight {index} has the wrong physical format")
    tail = [_nd(pack_value_weights(_nd(impl.W_UV))),
            _nd(pack_output_weights(_nd(impl.o_proj.weight))),
            _float_scale(impl.o_proj.weight_scale, HIDDEN, "output scale")]
    front = []
    if mode != "shared":
        indexer = _indexer_backend(impl.indexer)
        query, key = pack_indexer_weights(
            _nd(indexer.wq_b.weight), _nd(indexer.wk_weights_proj.weight),
        )
        hadamard = (indexer.q_hadamard if mode == "int8" else torch.zeros(
            128, 128, dtype=torch.bfloat16, device=query.device,
        ))
        query_scale = indexer.wq_b.weight_scale
        if query_scale.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("indexer scale requires original FP32 or BF16 channel scales")
        _require_tensor(query_scale, query_scale.dtype, (4096,), "indexer scale")
        front = [_nd(query),
                 _nd(query_scale),
                 _nd(key), _nd(indexer.k_norm.weight),
                 _nd(indexer.k_norm.bias), _nd(hadamard)]
        for value, dtype, shape, name in (
                (front[3], torch.float32, (128,), "indexer gamma"),
                (front[4], torch.float32, (128,), "indexer beta"),
                (front[5], torch.bfloat16, (128, 128), "indexer Hadamard")):
            _require_tensor(value, dtype, shape, name)
    return main, front, tail


def _cache_tensors(layer, mode):
    import torch

    main = layer.kv_cache
    if not isinstance(main, (tuple, list)) or len(main) != 2:
        raise ValueError("Native attention requires the bound main cache pair")
    pages = main[0].shape[0]
    _require_tensor(main[0], torch.bfloat16, (pages, 128, 1, 512), "main cache")
    _require_tensor(main[1], torch.bfloat16, (pages, 128, 1, 64), "RoPE cache")
    if mode == "shared":
        return list(main)
    index = layer.impl.indexer.k_cache.kv_cache
    if not isinstance(index, (tuple, list)) or len(index) != (2 if mode == "int8" else 1):
        raise ValueError("Native indexer requires its own bound cache tuple")
    pages = index[0].shape[0]
    dtype = torch.int8 if mode == "int8" else torch.bfloat16
    _require_tensor(index[0], dtype, (pages, 128, 1, 128), "indexer cache")
    result = [*main, index[0]]
    if mode == "int8":
        scale = index[1]
        if scale.dtype != torch.float16 or scale.numel() != pages * 128:
            raise ValueError("Native indexer cache scales require FP16 per slot")
        result.append(scale.view(pages, 128))
    return result


def _metadata_values(metadata, rows, indexed=False):
    """Reject padded, multi-request, or custom-selector calls without GM reads."""
    import torch

    if (metadata is None or metadata.num_actual_tokens != rows
            or getattr(metadata, "num_input_tokens", rows) != rows
            or getattr(metadata, "topk_selector", None) is not None):
        return None
    query = (metadata.actual_seq_lengths_query if indexed
             else metadata.cum_query_lens)
    key = metadata.actual_seq_lengths_key if indexed else metadata.seq_lens
    table, slots = metadata.block_table, metadata.slot_mapping
    if (not isinstance(query, torch.Tensor) or not isinstance(key, torch.Tensor)
            or query.dtype != torch.int32 or key.dtype != torch.int32
            or tuple(query.shape) != (1,) or tuple(key.shape) != (1,)
            or table.dtype != torch.int32 or table.ndim != 2
            or table.shape[0] != 1 or table.shape[1] < 1
            or slots.dtype not in (torch.int32, torch.int64)
            or tuple(slots.shape) != (rows,)
            or metadata.sin.dtype != torch.bfloat16
            or metadata.cos.dtype != torch.bfloat16
            or metadata.sin.numel() != rows * 64
            or metadata.cos.numel() != rows * 64
            or any(not value.is_contiguous() for value in (
                query, key, table, slots, metadata.sin, metadata.cos))):
        return None
    return [metadata.sin.view(rows, 64), metadata.cos.view(rows, 64),
            slots, query, key, table]


@dataclass(frozen=True)
class Geometry:
    mode: str
    pages: int
    table_pages: int
    index_pages: int = 0
    index_table_pages: int = 0


def geometry_for(mode, caches, main, index=None):
    if mode == "shared":
        return Geometry(mode, caches[0].shape[0], main[5].shape[1])
    if mode not in ("int8", "bf16") or index is None:
        raise ValueError("Indexed geometry requires int8/bf16 mode and metadata")
    return Geometry(mode, caches[0].shape[0], main[5].shape[1],
                    caches[2].shape[0], index[5].shape[1])


class WorkspacePool:
    """Own exact-M workspaces and live slot casts across sequential layers."""

    def __init__(self, blocks=0):
        self.blocks = blocks
        self.layers = {}
        self.entries = {}
        self.capture_reserves = {}
        self.slot_entries = {}
        self.slot_reserves = {}
        self.captured_slot_sources = set()

    def _prepare(self, reference, geometry):
        import torch

        rows = reference.shape[0]
        codes = torch.empty(rows, HIDDEN, dtype=torch.int8,
                            device=reference.device)
        if geometry.mode == "shared":
            scratch, tiling, blocks, sync = (
                torch.ops.tilert.shared_full_attention_prepare(
                    codes, geometry.pages, geometry.table_pages, self.blocks,
                )
            )
            topk = None
        else:
            scratch, tiling, offsets, blocks, sync = (
                torch.ops.tilert.indexed_attention_prepare(
                    codes, geometry.pages, geometry.table_pages,
                    geometry.index_pages, geometry.index_table_pages,
                    geometry.mode == "int8", self.blocks,
                )
            )
            topk = scratch.narrow(0, offsets["topk"], rows * 2048 * 4)
            topk = topk.view(torch.int32).view(rows, 1, 2048)
        partial = torch.empty(rows, HIDDEN, dtype=torch.bfloat16,
                              device=reference.device)
        return scratch, tiling, blocks, sync, partial, topk

    def reserve_capture(self, reference):
        """Resolve bound geometry under the capture's current forward context."""
        from vllm.forward_context import get_forward_context

        rows = reference.shape[0]
        if not 1 <= rows <= MAX_TOKENS:
            raise ValueError("Native attention requires 1..32 rows")
        metadata = get_forward_context().attn_metadata
        self.captured_slot_sources.clear()
        for name, (layer, mode) in self.layers.items():
            main = _metadata_values(metadata[name], rows)
            if main is None:
                continue
            index = None
            if mode != "shared":
                index = _metadata_values(
                    layer.impl._get_indexer_attn_metadata(), rows, indexed=True,
                )
            if mode != "shared" and index is None:
                continue
            geometry = geometry_for(mode, _cache_tensors(layer, mode), main, index)
            sources = [values[2] for values in (main, index) if values is not None]
            self.reserve(reference, geometry, sources)

    def reserve(self, reference, geometry, slot_sources):
        """Reserve one geometry and independent slot producers before capture."""
        import torch

        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Reserve native attention outside graph capture")
        if not 1 <= reference.shape[0] <= MAX_TOKENS:
            raise ValueError("Native attention requires 1..32 rows")
        self.captured_slot_sources.clear()
        key = (reference.device, reference.shape[0], geometry)
        if key not in self.capture_reserves:
            self.capture_reserves[key] = self._prepare(reference, geometry)
        for source in slot_sources:
            signature = self._slot_signature(source)
            if source.dtype != torch.int64 and signature not in self.slot_reserves:
                self.slot_reserves[signature] = torch.empty_like(source, dtype=torch.int64)

    def get(self, reference, geometry):
        import torch

        stream = torch.npu.current_stream(reference.device).npu_stream
        key = (reference.device, stream, reference.shape[0], geometry)
        if key not in self.entries:
            reserve = (reference.device, reference.shape[0], geometry)
            if torch.npu.is_current_stream_capturing():
                if reserve not in self.capture_reserves:
                    raise RuntimeError("Reserve native attention geometry before capture")
                self.entries[key] = self.capture_reserves.pop(reserve)
            else:
                self.entries[key] = self._prepare(reference, geometry)
        return self.entries[key]

    @staticmethod
    def _slot_signature(source):
        return (source.device, source.data_ptr(), tuple(source.shape),
                tuple(source.stride()), source.dtype)

    def live_slots(self, source):
        """Capture one producer per slot source; refresh every eager use."""
        import torch

        if source.dtype == torch.int64:
            return source
        stream = torch.npu.current_stream(source.device).npu_stream
        signature = self._slot_signature(source)
        key = (stream, *signature)
        capturing = torch.npu.is_current_stream_capturing()
        if key not in self.slot_entries:
            if capturing:
                if signature not in self.slot_reserves:
                    raise RuntimeError("Reserve native attention slot sources before capture")
                self.slot_entries[key] = self.slot_reserves.pop(signature)
            else:
                self.slot_entries[key] = torch.empty_like(source, dtype=torch.int64)
        destination = self.slot_entries[key]
        if not capturing or key not in self.captured_slot_sources:
            destination.copy_(source)
            if capturing:
                self.captured_slot_sources.add(key)
        return destination


def run_body(hidden, main_weights, front_weights, tail_weights, caches,
             main_metadata, index_metadata, pool, *, mode, scale,
             epsilon_cq, epsilon_ckv, topk=None):
    """Return the rank partial and ordered top-k using the engine's pool path.

    Metadata lists contain sine, cosine, live slots, cumulative query/key
    lengths, and page table. An indexed call supplies four cache tensors,
    including an empty FP16 scale tensor for BF16 keys. TP reduction and
    cross-layer top-k publication belong to the caller.
    """
    import torch
    import torch_npu

    geometry = geometry_for(mode, caches, main_metadata, index_metadata)
    scratch, tiling, blocks, sync, partial, selected = pool.get(hidden, geometry)
    main = list(main_metadata)
    main[2] = pool.live_slots(main[2])
    codes, token_scale = torch_npu.npu_dynamic_quant(hidden)
    token_scale = token_scale.view(hidden.shape[0], 1)
    scalars = (scale, epsilon_cq, epsilon_ckv)
    if mode == "shared":
        torch.ops.tilert.shared_full_attention_out(
            codes, token_scale, *main_weights, *main[:3], *caches, topk,
            *main[3:], *tail_weights, None, scratch, tiling, partial,
            *scalars, blocks, sync,
        )
        selected = topk
    else:
        index = list(index_metadata)
        index[2] = pool.live_slots(index[2])
        torch.ops.tilert.indexed_attention_out(
            hidden, codes, token_scale, main_weights, front_weights,
            tail_weights, None, main, index, caches, scratch, tiling, partial,
            *scalars, blocks, sync,
        )
    return partial, selected


def install(attention_layer, pool):
    """Bind loaded backbone attention; retain MTP and unsupported call paths."""
    import torch
    import torch_npu
    from vllm.distributed import tensor_model_parallel_all_reduce

    impl = attention_layer.impl
    if impl._is_mtp_layer:
        return lambda: None
    mode = _validate_impl(impl)
    caches = _cache_tensors(attention_layer, mode)
    if any(not value.is_contiguous() or torch_npu.get_npu_format(value) != 2
           for value in caches):
        raise ValueError("Native attention caches require physical ND storage")
    main_weights, front_weights, tail_weights = _pack_weights(impl, mode)
    empty_scale = torch.empty(0, dtype=torch.float16, device=caches[0].device)
    name = impl.layer_name
    if name in pool.layers:
        raise ValueError(f"Native attention is already installed for {name}")
    pool.layers[name] = (attention_layer, mode)
    original = impl.forward

    def forward(self, layer_name, hidden_states, kv_cache, attn_metadata,
                output=None):
        rows = hidden_states.shape[0] if hidden_states.ndim else 0
        if (hidden_states.ndim != 2 or hidden_states.shape[1] != HIDDEN
                or hidden_states.dtype != torch.bfloat16
                or not 1 <= rows <= MAX_TOKENS
                or output is None or tuple(output.shape) != (rows, HIDDEN)
                or output.dtype != torch.bfloat16
                or not hidden_states.is_contiguous() or not output.is_contiguous()
                or self._is_mtp_layer or self.layerwise_kv_cache_hook is not None
                or self.skip_topk != (mode == "shared")):
            return original(layer_name, hidden_states, kv_cache,
                            attn_metadata, output)
        main = _metadata_values(attn_metadata, rows)
        if main is None:
            return original(layer_name, hidden_states, kv_cache,
                            attn_metadata, output)
        index = None
        if mode != "shared":
            index = _metadata_values(self._get_indexer_attn_metadata(), rows,
                                     indexed=True)
        if mode != "shared" and index is None:
            return original(layer_name, hidden_states, kv_cache,
                            attn_metadata, output)
        bound = _cache_tensors(attention_layer, mode)
        if (not isinstance(kv_cache, (tuple, list)) or len(kv_cache) != 2
                or any(value.data_ptr() != expected.data_ptr()
                       for value, expected in zip(kv_cache, bound[:2]))):
            return original(layer_name, hidden_states, kv_cache,
                            attn_metadata, output)
        selected = None
        if mode == "shared":
            selected = self._get_indexcache_topk_indices(rows)
        elif mode == "bf16":
            bound.append(empty_scale)
        partial, selected = run_body(
            hidden_states, main_weights, front_weights, tail_weights, bound,
            main, index, pool, mode=mode, topk=selected, scale=self.scale,
            epsilon_cq=self.q_a_layernorm.variance_epsilon,
            epsilon_ckv=self.kv_a_layernorm.variance_epsilon,
        )
        if mode != "shared" and self.use_index_cache:
            self._update_indexcache_topk_indices(selected)
        if self.o_proj.reduce_results:
            partial = tensor_model_parallel_all_reduce(partial)
        output.copy_(partial)
        return output

    impl.forward = MethodType(forward, impl)

    def restore():
        impl.forward = original
        pool.layers.pop(name, None)

    return restore
