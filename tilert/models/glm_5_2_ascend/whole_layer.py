"""Install one native launch per supported GLM Decoder MoE layer.

Installation follows weight postprocessing and KV-cache binding. Capture must
be preceded by ``reserve_capture`` and followed by ``validate_capture``. The
model owns weights, caches and shared TopK; this module retains their owners and
owns the packed copies, exact-width scratch and SHMEM windows. Close the model's
graphs before calling ``close_after_graphs`` on every TP rank.
"""

from collections import Counter
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import MethodType

from . import attention_native as attention
from .routed_shared import _shared_weights
from .whole_layer_abi import (
    LocalRecords,
    MetadataLibrary,
    PreparedLayer,
    TreeLibrary,
    Window,
)


HIDDEN = 6144
BLOCKS = 24
WIDTHS = (1, 2, 12, 32)
FFN_NAMESPACE = "tilert_ffn_original_shared_gu_static"
LIBRARY_HASHES = {
    "indexed": "8deff7507f538da70fba91bab8e4979f38e6f5848d8cd442b392b82159f908f6",
    "shared": "",
    "ffn": "381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2",
    "metadata": "609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f",
    "tree": "7ec3e8fdd6155d99d98a9b69d2d4b02dc43c9417695f7632ad6559f42ecb96b3",
}
SOURCE_HASHES = {
    "decoder": "96f4f755f62caa85869fb884209e7f8869882ddb6efbd40c930778b27bd3e39d",
    "runner": "d4dfab36e66bd0a2ab9d78b9a867335adfc8cf0c418c93b34f5ccef63df00e27",
    "lookup": "20b69fe514bebd6723a1378f89a564c8fec4144c30244101f629c3f32c188183",
    "scale": "efdcc610c05a44952c2548e0d8b0cd48ca7704c3e2cf8247f8f08f5e765b830b",
}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Libraries:
    indexed: Path
    shared: Path
    ffn: Path
    metadata: Path
    tree: Path

    @classmethod
    def from_json(cls, path):
        """Read five library paths, relative to the JSON file when necessary."""
        path = Path(path).resolve()
        values = json.loads(path.read_text())
        if set(values) != set(LIBRARY_HASHES):
            raise ValueError("Whole-layer libraries require exactly five paths")
        return cls(**{name: (path.parent / value).resolve()
                      for name, value in values.items()})

    def verify(self):
        for name, expected in LIBRARY_HASHES.items():
            if _digest(getattr(self, name)) != expected:
                raise ValueError(f"The frozen whole-layer {name} library changed")

    def paths(self):
        return tuple(Path(getattr(self, name)) for name in LIBRARY_HASHES)


def _source(value, name):
    if _digest(inspect.getfile(value)) != SOURCE_HASHES[name]:
        raise RuntimeError(f"The supported {name} source changed")


def tensor_identity(value):
    """Describe an owner without reading its device contents."""
    return (value.device, value.data_ptr(), tuple(value.shape),
            tuple(value.stride()), value.storage_offset(), value.dtype)


def unpack_scale(packed, columns):
    """Decode the original zero-extended FP32 bits in each INT64 carrier."""
    import torch

    if packed.dtype != torch.int64 or tuple(packed.shape) not in (
            (256, columns), (256, 1, columns)):
        raise ValueError("Routed scales require INT64 [256,C] carriers")
    words = packed.detach().cpu().contiguous().reshape(256, columns)
    if not bool(((words >= 0) & (words <= 0xffffffff)).all()):
        raise ValueError("Routed scale carriers must zero-extend FP32 bits")
    values = words.to(torch.int32).view(torch.float32)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Routed scales must decode to finite FP32 values")
    return values


def _require(value, dtype, shape, name, device=None, physical=2):
    import torch_npu

    if (value.dtype != dtype or tuple(value.shape) != tuple(shape)
            or not value.is_contiguous()
            or value.device.type != "npu"
            or (device is not None and value.device != device)
            or torch_npu.get_npu_format(value) != physical):
        raise ValueError(f"{name} has an unsupported dtype, shape or storage")


def _validate_layer(layer, config):
    import torch
    from vllm.model_executor.layers.fused_moe.runner import moe_runner
    from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
        ZeroExpertRouter,
    )
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
    from vllm_ascend.ops.fused_moe.shared_experts import SharedExpertMLPPath

    if type(layer) is not DeepseekV2DecoderLayer:
        raise ValueError("Whole-layer installation requires DeepseekV2DecoderLayer")
    _source(type(layer), "decoder")
    if (getattr(layer.forward, "__func__", None) is not type(layer).forward
            or layer.use_mha or layer.use_sequence_parallel_moe):
        raise ValueError("The Decoder forward must be original MLA without SP")
    runner = layer.mlp.experts
    _source(type(runner), "runner")
    _source(moe_runner, "lookup")
    moe = runner.moe_config
    if (moe.tp_size != 16 or moe.ep_size != 1 or moe.dp_size != 1
            or moe.pcp_size != 1 or moe.is_sequence_parallel
            or config.parallel_config.enable_eplb
            or runner.routed_input_transform is not None
            or runner.routed_output_transform is not None
            or runner._should_reduce_routed_before_combine()
            or isinstance(runner.router, ZeroExpertRouter)):
        raise ValueError("Require untransformed TP16/EP1/DP1/PCP1 MoE")
    if (moe.hidden_dim != HIDDEN or moe.hidden_dim_unpadded != HIDDEN
            or moe.intermediate_size_per_partition != 128
            or moe.intermediate_size_per_partition_unpadded != 128
            or moe.num_experts != 256 or moe.num_local_experts != 256
            or moe.experts_per_token != 8 or moe.in_dtype != torch.bfloat16
            or moe.has_bias or moe.is_lora_enabled or moe.has_hash_routing
            or moe.skip_final_all_reduce or runner.routed_scaling_factor != 2.5
            or runner.enable_dbo or runner.shared_expert_gate is not None):
        raise ValueError("Require BF16 H6144/I128/E256/K8 MoE with scale 2.5")
    expected = {
        "hidden_size": HIDDEN, "n_routed_experts": 256,
        "num_experts_per_tok": 8, "n_group": 1, "topk_group": 1,
        "routed_scaling_factor": 2.5, "scoring_func": "sigmoid",
        "norm_topk_prob": True,
    }
    if any(getattr(config.model_config.hf_config, name) != value
           for name, value in expected.items()):
        raise ValueError("The model router differs from the native arithmetic")
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
    for norm in (layer.input_layernorm, layer.post_attention_layernorm):
        if norm.bias is not None:
            raise ValueError("Whole-layer RMSNorm must be unbiased")
        _require(norm.weight, torch.bfloat16, (HIDDEN,), "RMSNorm gamma")
    for module in (layer.self_attn, layer.mlp, layer.input_layernorm,
                   layer.post_attention_layernorm):
        if module._forward_hooks or module._forward_pre_hooks:
            raise ValueError("Whole-layer execution cannot bypass child hooks")
    return runner


def _lookup_runner(runner):
    from vllm.model_executor.layers.fused_moe.runner import moe_runner

    selected = moe_runner.get_layer_from_name(
        moe_runner._resolve_layer_name(runner._encode_layer_name()))
    if selected is not runner:
        raise RuntimeError("The original MoE lookup selected another layer")


def _runtime_ready(runner):
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm_ascend.ops.fused_moe.moe_comm_method import MoECommType

    return (_EXTRA_CTX.moe_comm_type == MoECommType.ALLGATHER
            and not runner._fused_output_is_reduced)


class _Weights:
    def __init__(self, layer, config):
        import torch

        self.runner = _validate_layer(layer, config)
        self.attention = layer.self_attn.mla_attn.mla_attn
        self.impl = self.attention.impl
        self.mode = attention._validate_impl(self.impl)
        if (self.impl._is_mtp_layer or not self.impl.o_proj.reduce_results
                or self.impl.layerwise_kv_cache_hook is not None):
            raise ValueError("Require backbone attention with its TP reduction")
        self.main, self.front, self.tail = attention._pack_weights(
            self.impl, self.mode)
        self.caches = attention._cache_tensors(self.attention, self.mode)
        self.device = self.caches[0].device
        for value in self.caches:
            _require(value, value.dtype, value.shape, "KV cache", self.device)
        topk = layer.self_attn.mla_attn.topk_indices_buffer
        if (topk is None or self.impl.topk_indices_buffer is not topk
                or topk.ndim != 2 or topk.shape[0] < max(WIDTHS)
                or topk.shape[1] != 2048):
            raise ValueError("The layer must retain its shared [capacity,2048] TopK")
        _require(topk, torch.int32, topk.shape, "Shared TopK", self.device)
        self.topk = topk
        routed = self.runner.routed_experts
        _source(type(routed.quant_method.quant_method), "scale")
        scales = [attention._nd(unpack_scale(getattr(routed, name), columns)
                                .to(self.device))
                  for name, columns in (("w13_weight_scale", 256),
                                        ("w2_weight_scale", HIDDEN))]
        self.routed = [routed.w13_weight, routed.w13_scale_bias, scales[0],
                       routed.w2_weight, routed.w2_scale_bias, scales[1]]
        for index, shape in ((0, (256, HIDDEN, 32)),
                             (3, (256, 128, 768))):
            _require(self.routed[index], torch.int32, shape,
                     "Routed INT4 carrier", self.device, physical=29)
        for index, columns in ((1, 256), (2, 256), (4, HIDDEN), (5, HIDDEN)):
            value = self.routed[index]
            _require(value, torch.float32, (256, columns),
                     "Routed scale/correction", self.device)
        self.shared = list(_shared_weights(self.runner.ascend_shared_experts))
        self.input_gamma = attention._nd(layer.input_layernorm.weight)
        self.post = [attention._nd(layer.post_attention_layernorm.weight),
                     attention._nd(attention._nd(layer.mlp.gate.weight).t()),
                     layer.mlp.gate.e_score_correction_bias.detach()]
        _require(self.post[1], torch.bfloat16, (HIDDEN, 256),
                 "Router weight", self.device)
        _require(self.post[2], torch.float32, (256,),
                 "Router correction", self.device)
        self.scalars = (self.impl.scale,
                        self.impl.q_a_layernorm.variance_epsilon,
                        self.impl.kv_a_layernorm.variance_epsilon,
                        layer.input_layernorm.variance_epsilon,
                        layer.post_attention_layernorm.variance_epsilon)
        self.owners = (*self.main, *self.front, *self.tail, *self.routed,
                       *self.shared, self.input_gamma, *self.post,
                       *self.caches, topk)
        self.identities = tuple(map(tensor_identity, self.owners))

    def check_owners(self):
        if tuple(map(tensor_identity, self.owners)) != self.identities:
            raise RuntimeError("Whole-layer weight or cache storage changed")
        bound = attention._cache_tensors(self.attention, self.mode)
        if tuple(map(tensor_identity, bound)) != tuple(
                map(tensor_identity, self.caches)):
            raise RuntimeError("The model rebound a native KV cache")
        if self.impl.topk_indices_buffer is not self.topk:
            raise RuntimeError("The model replaced the persistent TopK owner")

    def metadata(self, rows):
        from vllm.forward_context import get_forward_context

        values = get_forward_context().attn_metadata
        if not isinstance(values, dict):
            return None
        main = attention._metadata_values(values.get(self.impl.layer_name), rows)
        index = None
        if self.mode != "shared":
            index = attention._metadata_values(
                self.impl._get_indexer_attn_metadata(), rows, indexed=True)
        if main is None or (self.mode != "shared" and index is None):
            return None
        for fields in (main, index):
            if fields is not None and any(value.device != self.device
                                          for value in fields):
                raise ValueError("Native metadata must remain on the layer device")
        return main, index


class _Arena:
    """One layer, exact M, metadata geometry and stream; no replay allocation."""

    def __init__(self, owner, reference, metadata):
        import torch

        self.owner = owner
        weights = owner.weights
        self.rows = reference.shape[0]
        self.geometry = attention.geometry_for(weights.mode, weights.caches,
                                               *metadata)

        def empty(shape, dtype):
            return attention._nd(torch.empty(shape, dtype=dtype,
                                             device=reference.device))

        self.norm_outputs = [empty((self.rows, HIDDEN), torch.bfloat16),
                             empty((self.rows, HIDDEN), torch.bfloat16),
                             empty((self.rows,), torch.float32),
                             empty((self.rows, HIDDEN), torch.int8),
                             empty((self.rows, 1), torch.float32)]
        geometry = self.geometry
        if weights.mode == "shared":
            self.scratch, self.tiling, blocks, self.sync = (
                torch.ops.tilert.shared_full_attention_prepare(
                    self.norm_outputs[3], geometry.pages,
                    geometry.table_pages, BLOCKS))
            self.selected = weights.topk[:self.rows].view(self.rows, 1, 2048)
        else:
            self.scratch, self.tiling, offsets, blocks, self.sync = (
                torch.ops.tilert.indexed_attention_prepare(
                    self.norm_outputs[3], geometry.pages, geometry.table_pages,
                    geometry.index_pages, geometry.index_table_pages,
                    weights.mode == "int8", BLOCKS))
            self.selected = self.scratch.narrow(
                0, offsets["topk"], self.rows * 2048 * 4
            ).view(torch.int32).view(self.rows, 1, 2048)
        self.partial = empty((self.rows, HIDDEN), torch.bfloat16)
        self.bridge, self.body, sync = owner.manager.ffn.prepare(
            self.partial, BLOCKS, 1)
        if (blocks != BLOCKS or not sync or self.sync != sync
                or len(self.bridge) != 12 or len(self.body) != 21):
            raise RuntimeError("Native parent workspaces disagree on geometry or sync")
        self.attention_sum = empty((self.rows, HIDDEN), torch.bfloat16)
        self.output = empty((self.rows, HIDDEN), torch.bfloat16)
        self.bindings = {}
        self.slot_buffers = {}
        self.reserve_metadata(metadata)

    def reserve_metadata(self, metadata):
        import torch

        for values in metadata:
            if values is None:
                continue
            # Format 0 -> ND can retag the shared RoPE storage descriptor.
            values[:2] = [attention._nd(value) for value in values[:2]]
            source = values[2]
            if source.dtype == torch.int32:
                self.slot_buffers.setdefault(
                    tensor_identity(source), torch.empty_like(
                        source, dtype=torch.int64))

    def bind(self, hidden, residual, metadata, window):
        import torch

        weights = self.owner.weights
        fields = []
        for values in metadata:
            if values is None:
                fields.append(None)
                continue
            values = list(values)
            values[:2] = [attention._nd(value) for value in values[:2]]
            source = values[2]
            if source.dtype == torch.int32:
                key = tensor_identity(source)
                if key not in self.slot_buffers:
                    if torch.npu.is_current_stream_capturing():
                        raise RuntimeError("The capture changed its reserved slot source")
                    self.slot_buffers[key] = torch.empty_like(source,
                                                              dtype=torch.int64)
                values[2] = self.slot_buffers[key]
                values[2].copy_(source)
            fields.append(values)
        main, index = fields
        norm = [hidden, residual, weights.input_gamma, *self.norm_outputs]
        caches = [*weights.caches, *([None] * (4 - len(weights.caches)))]
        addresses = [*norm[3:4], *norm[6:8], *weights.main,
                     *(weights.front or [None] * 6), *weights.tail, None,
                     *main, *(index or [None] * 6), *caches,
                     self.scratch, self.tiling, self.partial]
        geometry = self.geometry
        dimensions = (self.rows, geometry.pages, geometry.table_pages,
                      geometry.index_pages or geometry.pages,
                      geometry.index_table_pages or geometry.table_pages,
                      BLOCKS, int(weights.mode == "int8"),
                      int(bool(weights.front)
                          and weights.front[1].dtype == torch.bfloat16), 0)
        groups = (norm, addresses, weights.post, weights.routed, self.bridge,
                  self.body, weights.shared)
        key = tuple(tensor_identity(value) for group in groups
                    for value in group if value is not None)
        if key not in self.bindings:
            local = LocalRecords(
                self.owner.manager.metadata_library, *groups,
                self.selected if weights.mode == "shared" else None,
                dimensions, weights.scalars, self.sync,
                weights.mode != "shared")
            self.bindings[key] = PreparedLayer(
                local, window, self.attention_sum, self.output)
        return self.bindings[key]

    def tensors(self):
        return (*self.norm_outputs, self.scratch, self.tiling, self.partial,
                *self.bridge, *self.body, self.attention_sum, self.output,
                *self.slot_buffers.values())


class _Layer:
    def __init__(self, manager, layer):
        self.manager, self.layer = manager, layer
        self.weights = _Weights(layer, manager.config)
        self.name = self.weights.impl.layer_name
        self.original = layer.forward
        self.entries = {}
        self.reserves = {}
        self.counters = Counter()

    def reserve(self, reference):
        metadata = self.weights.metadata(reference.shape[0])
        if metadata is None:
            raise ValueError(f"{self.name}: target capture metadata is unsupported")
        geometry = attention.geometry_for(self.weights.mode,
                                           self.weights.caches, *metadata)
        key = reference.shape[0], geometry
        if key not in self.reserves:
            self.reserves[key] = _Arena(self, reference, metadata)
        self.counters["prepared"] += 1

    def forward(self, unused_layer, positions, hidden_states, residual,
                llama_4_scaling=None):
        import torch

        if self.manager.closed:
            raise RuntimeError("The whole-layer manager has been closed")
        rows = hidden_states.shape[0] if hidden_states.ndim == 2 else 0
        if rows not in WIDTHS:
            self.counters["other_width_fallback"] += 1
            return self.original(positions, hidden_states, residual,
                                 llama_4_scaling)
        if (residual is None or llama_4_scaling is not None
                or tuple(positions.shape) != (rows,)
                or not _runtime_ready(self.weights.runner)):
            raise ValueError(f"{self.name}: unsupported native Decoder inputs or MoE state")
        for value in (hidden_states, residual):
            _require(value, torch.bfloat16, (rows, HIDDEN),
                     "Decoder input/residual", self.weights.device)
        self.weights.check_owners()
        metadata = self.weights.metadata(rows)
        if metadata is None:
            raise ValueError(f"{self.name}: unsupported native metadata at M={rows}")
        geometry = attention.geometry_for(self.weights.mode,
                                           self.weights.caches, *metadata)
        stream = torch.npu.current_stream(hidden_states.device)
        capturing = torch.npu.is_current_stream_capturing()
        window = self.manager.window(stream, capturing)
        key = stream.npu_stream, rows, geometry
        if key not in self.entries:
            if capturing:
                reserve = rows, geometry
                if reserve not in self.reserves:
                    raise RuntimeError(f"{self.name}: capture has no reserved arena")
                self.entries[key] = self.reserves.pop(reserve)
                self.counters["reserve_claimed"] += 1
            else:
                self.entries[key] = _Arena(self, hidden_states, metadata)
        arena = self.entries[key]
        prepared = arena.bind(hidden_states, residual, metadata, window)
        _lookup_runner(self.weights.runner)
        prepared.launch()
        if self.weights.mode != "shared" and self.weights.impl.use_index_cache:
            self.weights.impl._update_indexcache_topk_indices(arena.selected)
        self.counters["submitted"] += 1
        self.counters[f"submitted_m{rows}"] += 1
        if capturing:
            self.manager.capture_layer(self.name, rows)
        return arena.output, arena.bridge[1]

    def restore(self):
        self.layer.forward = self.original


class WholeLayerManager:
    """Own a real TP16 model installation and its two in-kernel reductions.

    All ranks must install the same layers and reserve widths in the same
    order. The caller must retain this object through graph teardown. The
    collective endpoint must be distinct from any other SHMEM context.
    """

    def __init__(self, config, libraries, *, device, rank, ranks, endpoint,
                 target_width=12, heap_bytes=1 << 30):
        import torch
        import torch_npu
        from vllm.distributed import get_tp_group

        if (target_width not in WIDTHS or ranks != 16
                or not config.model_config.enforce_eager
                or getattr(config, "lora_config", None) is not None):
            raise ValueError("Whole-layer requires eager Python capture, TP16 and a supported width")
        group = get_tp_group()
        if group.world_size != ranks or group.rank_in_group != rank:
            raise ValueError("SHMEM ranks must match the actual model TP group")
        if (not torch.__version__.startswith("2.12.")
                or not torch_npu.__version__.startswith("2.12.")):
            raise ValueError("The frozen whole-layer bindings require Torch/NPU 2.12")
        if (os.environ.get("HCCL_OP_EXPANSION_MODE") != "AIV"
                or os.environ.get("HCCL_DETERMINISTIC", "").lower() != "true"
                or os.environ.get("HCCL_ALGO")
                or os.environ.get("HCCL_ALGO_CONFIG")):
            raise ValueError("Require deterministic AIV HCCL without an algorithm override")
        libraries.verify()
        self.config, self.libraries = config, libraries
        self.device, self.rank, self.ranks = torch.device(device), rank, ranks
        self.target_width = target_width
        self.group = group
        for path in (libraries.indexed, libraries.shared, libraries.ffn):
            torch.ops.load_library(str(path))
        self.ffn = getattr(torch.ops, FFN_NAMESPACE)
        self.metadata_library = MetadataLibrary(libraries.metadata,
                                                LIBRARY_HASHES["metadata"])
        self.tree = TreeLibrary(libraries.tree, LIBRARY_HASHES["tree"])
        self.layers = {}
        self.dense = []
        self.windows = {}
        self.capture_owner = None
        self.capture_stream = None
        self.capture_width = None
        self.capture_seen = set()
        self.capture_history = []
        self.closed = False
        self.initialized = False
        self.endpoint, self.heap_bytes = endpoint, heap_bytes

    def install_model(self, model):
        """Install every MoE Decoder in a loaded backbone, then initialize SHMEM."""
        import torch

        if self.layers or self.initialized:
            raise RuntimeError("The whole-layer model is already installed")
        hf = self.config.model_config.hf_config
        layers = model.model.layers
        values = list(layers.values()) if hasattr(layers, "values") else list(layers)
        frequency = getattr(hf, "moe_layer_freq", 1)
        expected = [layer for layer in values
                    if layer.layer_idx >= hf.first_k_dense_replace
                    and layer.layer_idx % frequency == 0]
        if not expected:
            raise ValueError("The loaded backbone has no eligible MoE Decoder")
        prepared = [_Layer(self, layer) for layer in expected]
        names = [item.name for item in prepared]
        if len(set(names)) != len(names):
            raise ValueError("Whole-layer Decoder names are not unique")
        peers = [None] * self.ranks
        torch.distributed.all_gather_object(peers, names,
                                            group=self.group.cpu_group)
        if any(peer != names for peer in peers):
            raise RuntimeError("TP ranks selected different native Decoder layers")
        self.tree.initialize(self.device.index, self.rank, self.ranks,
                             self.endpoint, self.heap_bytes)
        self.initialized = True
        for item in prepared:
            self.layers[item.name] = item
            item.layer.forward = MethodType(item.forward, item.layer)
        self.dense = [layer.layer_idx for layer in values if layer not in expected]
        # Eager and graph streams have separate symmetric storage and epochs.
        self.window(torch.npu.current_stream(self.device), False)
        return self

    def window(self, stream, capturing):
        key = stream.npu_stream
        if key not in self.windows:
            if capturing:
                raise RuntimeError("Reserve the actual capture-stream window before capture")
            if not self.initialized:
                raise RuntimeError("Install the complete model before allocating windows")
            self.windows[key] = Window(self.tree, max(WIDTHS))
        return self.windows[key]

    def reserve_capture(self, reference):
        """Allocate before official ACLGraphWrapper enters its default stream."""
        import torch
        from vllm_ascend.compilation.updatable_graph import UpdatableGraph

        rows = reference.shape[0]
        if rows not in WIDTHS or torch.npu.is_current_stream_capturing():
            raise ValueError("Reserve a supported width outside capture")
        if self.capture_width is not None:
            raise RuntimeError("Validate the preceding capture before reserving another")
        if self.capture_owner is None:
            graph = UpdatableGraph()
            manager = torch.npu.graph(graph)
            self.capture_owner = manager, graph
            self.capture_stream = manager.capture_stream
        with torch.npu.stream(self.capture_stream):
            self.window(self.capture_stream, False)
        for layer in self.layers.values():
            layer.reserve(reference)
        self.capture_width = rows
        self.capture_seen = set()

    def capture_layer(self, name, rows):
        import torch

        if (rows != self.capture_width
                or torch.npu.current_stream(self.device).npu_stream
                != self.capture_stream.npu_stream
                or name in self.capture_seen):
            raise RuntimeError("Whole-layer capture changed width, stream or layer order")
        self.capture_seen.add(name)

    def validate_capture(self, rows):
        """Require every installed MoE layer once before the first graph replay."""
        if (rows != self.capture_width
                or self.capture_seen != set(self.layers)):
            missing = sorted(set(self.layers) - self.capture_seen)
            raise RuntimeError(f"Whole-layer capture omitted native layers: {missing}")
        self.capture_history.append({"rows": rows,
                                     "layers": tuple(self.layers)})
        self.capture_width = None

    def coverage(self):
        """Report host submissions; captured replay does not execute Python."""
        storage = {}
        for layer in self.layers.values():
            for arena in (*layer.entries.values(), *layer.reserves.values()):
                for tensor in arena.tensors():
                    value = tensor.untyped_storage()
                    storage[(tensor.device, value.data_ptr())] = value.nbytes()
        return {"installed": tuple(self.layers), "dense_layers": self.dense,
                "target_width": self.target_width,
                "layers": {name: dict(layer.counters)
                           for name, layer in self.layers.items()},
                "captures": list(self.capture_history),
                "workspace_bytes": sum(storage.values()),
                "communication_window_bytes": sum(window.bytes
                                                   for window in self.windows.values())}

    def close_after_graphs(self):
        """After resetting every graph, drain ranks, release windows and restore."""
        import torch

        if self.closed:
            return
        torch.npu.synchronize(self.device)
        if self.initialized:
            torch.distributed.barrier(group=self.group.cpu_group)
        for layer in self.layers.values():
            layer.restore()
        for window in reversed(tuple(self.windows.values())):
            window.close()
        self.windows.clear()
        if self.initialized:
            self.tree.finalize()
        self.capture_owner = None
        self.layers.clear()
        self.closed = True
