from collections import Counter
import ctypes
from dataclasses import dataclass
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import struct
from types import MethodType

from . import attention_native as attention
from .routed_shared import _shared_weights
from .whole_layer_abi import (
    LocalRecords,
    MetadataLibrary,
    PreparedChain,
    PreparedLayer,
    TreeLibrary,
    Window,
)


HIDDEN = 6144
BLOCKS = 24
WIDTHS = (1, 2, 4, 6, 8, 12, 16, 24, 32)
FFN_NAMESPACE = "tilert_ffn_original_shared_gu_static"
LIBRARY_HASHES = {
    "indexed": {
        "fea3c0c6c8d1c1daa97dddc1c954ca1f61ffc1a8f579aeccb11a078bb211f403",
        "8deff750464e23bb871302c8626b86d37b60c28a00991d550f95dfcfd47ecb6f",
        "aa10bf55d65f34a9aa6d1627703794303dc7b1ce1e3119856ce6a6abf3b9d6f9",
    },
    "shared": {
        "64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c",
        "0e985f72fda85389bcc6c11915cef5255a71475633089a82bea8639fd59e1d5d",
    },
    "ffn": {
        "8c031c70c2ccef93a7425ef06ff457923e139889903caaeea21cdbe4ed21cf42",
        "eb565d83aceb23dc00be837042dfb009d0cbc58c9dff531dfec3df34bfb969ac",
        "358034c6b45ad37a270944b0918ceb91084d25e4ed1621a1745a3edbf7400ed9",
        "381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2",
        "9a22fae0d4c2aedde083b2d413d6706e33ce5b941508b7d7b7a86814acd611f4",
        "168f1c812bfc9a0a3774d6436004a7d416e18aa17efb18f63e631d6005af9a54",
        "74838084ddcc204cbb01e62d2d612716dc23e3f5e0a6b33232470674e155a199",
        "8ec2d71bc522efbf287bf70cb24bccc6e0c4b1c1556a0335d5529232c4676b0f",
    },
    "metadata": {
        "421775a060c02bbb112d94f17601522d10606d8b506476d900bb03e438e6d5c3",
        "cb5aaf655bab06b7b5e2304acb5a053030de121df4ba8f8735354bd6e0e52708",
        "3349e9978c38a9f34b77895325edda331d165d4be2aaaad09edf2f87f097370f",
        "c8f5cf5a4b54eaffe7d21fafd796e3431a4dc0de16ede4e624d1cd39cfc7cf5e",
        "609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f",
        "d9083176f84ca43205b0ec46206d5ee4a3f37c94557baa96cd4e8df76f5c2e66",
        "b8eadb136b308801921430fc1760f2c45a4f4c6ba5b93de54a846bcbb26bfea0",
        "c573611d5cf22c7e9446b5c356c19560bb7be9a0b1910d7588ffed3616069673",
    },
    "tree": {
        "bd380142f6c125c44b9ab5f79e3934241e16d38f5ae206e08333c56b926af6cc",
        "8db9ecf2bef487b90b8611c164b760494d4a9d25d1e6aa9afd9ace20ccb966cd",
        "f0d69486c76a9d5d1b4e7e5407066209ee51818e3a02540c24dbd64dbe720715",
        "ae3c46867c9923cf1646a6e77af4804a64d3f145c7359494a0f9cc01826c73c9",
        "3fed5d1704e880be62f5f3f96de285eae27c47cb9b8bc770454d1b22d8c9b434",
        "bf7f17c12eeda21169c933a90d091c5b1d55d25ebb748ae1fa14d6d8f1395d10",
        "15f2d17d9cd6cf9eb14c3abf4d9cde1d63773515e35b29ddf37716cedc60aea1",
        "11fa8bfb0ea274fd5473e2687960c487dfa298c492c639799abeed0af1c88ba7",
        "af13b91eb626ab9ece7582153b09caacdf7001898fc9341a781eeee44ef5cc9f",
        "7ec3e8fdd6155d99d98a9b69d2d4b02dc43c9417695f7632ad6559f42ecb96b3",
        "4671998072ae15e8efe69cecf5a3cd88757a431b72ff713dae4d596ca3d9191e",
        "06aef9dd18c662c04679cb976e636ff8804c5feacf7d1331862cacda74d718fa",
        "4905da958f67a17a02096d638616b63092c9dc0bfb6449987fa267152c43c447",
        "21ebf23d459e0819ba09b761f9ab066a65ea6cc68902085abfef2f34c852c0bf",
    },
}
SOURCE_HASHES = {
    "decoder": {
        "96f4f755f62caa85869fb884209e7f8869882ddb6efbd40c930778b27bd3e39d",
        "5cd8c8883b18cac91979781a8b191703781971d8c59cb9823167cd3e0dfc2408",
    },
    "runner": {
        "d4dfab36e66bd0a2ab9d78b9a867335adfc8cf0c418c93b34f5ccef63df00e27",
    },
    "lookup": {
        "20b69fe514bebd6723a1378f89a564c8fec4144c30244101f629c3f32c188183",
        "8c3860e0149dcbe2e338393d14f4972c267c93082112d87144593e7a30a9c8f5",
    },
    "scale": {
        "efdcc610c05a44952c2548e0d8b0cd48ca7704c3e2cf8247f8f08f5e765b830b",
        "d06549f45637300df3e4915823f4f71a913ce7059d1e3a4c475ab1f225e1a28a",
        "47444be0a5017d012401685d89134200f39c5e706f7941797fa7ecfccf1bba0d",
    },
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
        path = Path(path).resolve()
        values = json.loads(path.read_text())
        if set(values) != set(LIBRARY_HASHES):
            raise ValueError("Whole-layer libraries require exactly five paths")
        return cls(**{name: (path.parent / value).resolve()
                      for name, value in values.items()})

    def tree_manifest(self):
        manifest_path = self.tree.with_suffix(".manifest.json")
        if manifest_path.is_file():
            return json.loads(manifest_path.read_text())
        return None

    def verify(self):
        from .ops.build_megakernel import ROOT as mk_root, effective_source_digest, source_digest

        for name, expected in LIBRARY_HASHES.items():
            if name == "tree":
                manifest = self.tree_manifest()
                if manifest is not None:
                    if _digest(self.tree) != manifest.get("sha256"):
                        raise ValueError("The built whole-layer tree library changed")
                    if "enable_kernel_prof" in manifest:
                        flag_prof = "-DENABLE_KERNEL_PROF=1" in (manifest.get("flags") or ())
                        if bool(manifest.get("enable_kernel_prof")) != flag_prof:
                            raise ValueError("The built whole-layer tree prof flag changed")
                    if mk_root.is_dir():
                        if "effective_source_sha256" in manifest:
                            eff = effective_source_digest(manifest.get("flags", ()))
                            if manifest.get("effective_source_sha256") != eff:
                                raise ValueError("The built whole-layer tree source changed")
                        elif manifest.get("source_sha256") != source_digest():
                            raise ValueError("The built whole-layer tree source changed")
                    continue
            allowed = {expected} if isinstance(expected, str) else set(expected)
            if _digest(getattr(self, name)) not in allowed:
                raise ValueError(f"The frozen whole-layer {name} library changed")

    def paths(self):
        return tuple(Path(getattr(self, name)) for name in LIBRARY_HASHES)


def _source(value, name):
    expected = SOURCE_HASHES[name]
    allowed = {expected} if isinstance(expected, str) else set(expected)
    if _digest(inspect.getfile(value)) not in allowed:
        raise RuntimeError(f"The supported {name} source changed")


def tensor_identity(value):
    return (value.device, value.data_ptr(), tuple(value.shape),
            tuple(value.stride()), value.storage_offset(), value.dtype)


def unpack_scale(packed, columns, experts=256):
    import torch

    if packed.dtype != torch.int64 or tuple(packed.shape) not in (
            (experts, columns), (experts, 1, columns)):
        raise ValueError(f"Routed scales require INT64 [{experts},C] carriers")
    words = packed.detach().cpu().contiguous().reshape(experts, columns)
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
    is_tp16 = (moe.tp_size == 16 and moe.ep_size == 1
               and moe.intermediate_size_per_partition == 128
               and moe.num_local_experts == 256)
    is_tp8_ep8 = (moe.tp_size == 1 and moe.ep_size == 8
                  and moe.intermediate_size_per_partition == 2048
                  and moe.num_local_experts == 32)
    if ((not is_tp16 and not is_tp8_ep8) or moe.dp_size != 1
            or moe.pcp_size != 1 or moe.is_sequence_parallel
            or config.parallel_config.enable_eplb
            or runner.routed_input_transform is not None
            or runner.routed_output_transform is not None
            or runner._should_reduce_routed_before_combine()
            or isinstance(runner.router, ZeroExpertRouter)):
        raise ValueError("Require untransformed TP16/EP1 or TP8+EP8 MoE")
    if (moe.hidden_dim != HIDDEN or moe.hidden_dim_unpadded != HIDDEN
            or moe.intermediate_size_per_partition_unpadded != moe.intermediate_size_per_partition
            or moe.num_experts != 256
            or moe.experts_per_token != 8 or moe.in_dtype != torch.bfloat16
            or moe.has_bias or moe.is_lora_enabled or moe.has_hash_routing
            or moe.skip_final_all_reduce or runner.routed_scaling_factor != 2.5
            or runner.enable_dbo or runner.shared_expert_gate is not None):
        raise ValueError("Require BF16 H6144/E256/K8 MoE with scale 2.5")
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
        if norm.bias is not None and not getattr(norm, "bias_loaded", True):
            norm.bias = None
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

    if runner.moe_config.ep_size == 8:
        return _EXTRA_CTX.moe_comm_type in (
            MoECommType.ALLGATHER, MoECommType.MC2, MoECommType.ALLTOALL
        )
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
        if getattr(layer.self_attn.mla_attn, "topk_indices_buffer", None) is None:
            layer.self_attn.mla_attn.topk_indices_buffer = self.impl.topk_indices_buffer
        topk = layer.self_attn.mla_attn.topk_indices_buffer
        if (topk is None or self.impl.topk_indices_buffer is not topk
                or topk.ndim != 2 or topk.shape[0] < max(WIDTHS)
                or topk.shape[1] != 2048):
            raise ValueError("The layer must retain its shared [capacity,2048] TopK")
        _require(topk, torch.int32, topk.shape, "Shared TopK", self.device)
        self.topk = topk
        routed = self.runner.routed_experts
        _source(type(routed.quant_method.quant_method), "scale")
        moe = self.runner.moe_config
        local_experts = moe.num_local_experts
        inter = moe.intermediate_size_per_partition
        gu_cols = 2 * inter
        scales = [attention._nd(unpack_scale(getattr(routed, name), columns, local_experts)
                                .to(self.device))
                  for name, columns in (("w13_weight_scale", gu_cols),
                                        ("w2_weight_scale", HIDDEN))]
        self.routed = [routed.w13_weight, routed.w13_scale_bias, scales[0],
                       routed.w2_weight, routed.w2_scale_bias, scales[1]]
        for index, shape in ((0, (local_experts, HIDDEN, gu_cols // 8)),
                             (3, (local_experts, inter, HIDDEN // 8))):
            _require(self.routed[index], torch.int32, shape,
                     "Routed INT4 carrier", self.device, physical=29)
        for index, columns in ((1, gu_cols), (2, gu_cols), (4, HIDDEN), (5, HIDDEN)):
            value = self.routed[index]
            _require(value, torch.float32, (local_experts, columns),
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

    def __init__(self, owner, reference, metadata):
        import torch

        self.owner = owner
        weights = owner.weights
        self.rows = reference.shape[0]
        self.geometry = attention.geometry_for(weights.mode, weights.caches,
                                               *metadata)

        def empty(shape, dtype):
            return attention._nd(torch.zeros(shape, dtype=dtype,
                                             device=reference.device))

        geometry = self.geometry
        pool_key = (self.rows, weights.mode, geometry, owner.layer.layer_idx % 2)
        arena_pool = getattr(owner.manager, "arena_pool", None)
        if arena_pool is None:
            arena_pool = owner.manager.arena_pool = {}
        if pool_key not in arena_pool:
            norm_outputs = [empty((self.rows, HIDDEN), torch.bfloat16),
                            empty((self.rows, HIDDEN), torch.bfloat16),
                            empty((self.rows,), torch.float32),
                            empty((self.rows, HIDDEN), torch.int8),
                            empty((self.rows, 1), torch.float32)]
            if weights.mode == "shared":
                scratch, tiling, blocks, sync_a = (
                    torch.ops.tilert.shared_full_attention_prepare(
                        norm_outputs[3], geometry.pages,
                        geometry.table_pages, BLOCKS))
                offsets = None
            else:
                scratch, tiling, offsets, blocks, sync_a = (
                    torch.ops.tilert.indexed_attention_prepare(
                        norm_outputs[3], geometry.pages, geometry.table_pages,
                        geometry.index_pages, geometry.index_table_pages,
                        weights.mode == "int8", BLOCKS))
            partial = empty((self.rows, HIDDEN), torch.bfloat16)
            bridge, body, sync = owner.manager.ffn.prepare(partial, BLOCKS, 1)
            if (blocks != BLOCKS or not sync or sync_a != sync
                    or len(bridge) != 12 or len(body) != 21):
                raise RuntimeError("Native parent workspaces disagree on geometry or sync")
            attention_sum = empty((self.rows, HIDDEN), torch.bfloat16)
            output = empty((self.rows, HIDDEN), torch.bfloat16)
            arena_pool[pool_key] = (
                norm_outputs, scratch, tiling, offsets, sync_a,
                partial, bridge, body, attention_sum, output,
            )
        (self.norm_outputs, self.scratch, self.tiling, offsets, self.sync,
         self.partial, self.bridge, self.body, self.attention_sum,
         self.output) = arena_pool[pool_key]
        if weights.mode == "shared":
            self.selected = weights.topk[:self.rows].view(self.rows, 1, 2048)
        else:
            self.selected = self.scratch.narrow(
                0, offsets["topk"], self.rows * 2048 * 4
            ).view(torch.int32).view(self.rows, 1, 2048)
        self.bindings = {}
        self.slot_buffers = {}
        self.reserve_metadata(metadata)

    def reserve_metadata(self, metadata):
        import torch

        for values in metadata:
            if values is None:
                continue
            values[:2] = [attention._nd(value) for value in values[:2]]
            source = values[2]
            if source.dtype == torch.int32:
                key = tensor_identity(source)
                buf = self.owner.manager.slot_buffers.setdefault(
                    key, torch.empty_like(source, dtype=torch.int64))
                self.slot_buffers.setdefault(key, buf)

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
                    self.slot_buffers[key] = (
                        self.owner.manager.slot_buffers.setdefault(
                            key, torch.empty_like(source, dtype=torch.int64))
                    )
                values[2] = self.slot_buffers[key]
                if key not in self.owner.manager.slot_refreshed:
                    values[2].copy_(source)
                    if self.owner.manager.chain_active:
                        self.owner.manager.slot_refreshed.add(key)
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
            shared_topk = (
                self.selected if weights.mode == "shared"
                else (weights.topk[:self.rows].view(self.rows, 1, 2048)
                      if weights.impl.use_index_cache
                      and self.owner.manager.tree.has_chain
                      else None)
            )
            local = LocalRecords(
                self.owner.manager.metadata_library, *groups,
                shared_topk,
                dimensions, weights.scalars, self.sync,
                weights.mode != "shared")
            self.bindings[key] = PreparedLayer(
                local, window, self.attention_sum, self.output)
        return self.bindings[key]

    def tensors(self):
        return (*self.norm_outputs, self.scratch, self.tiling, self.partial,
                *self.bridge, *self.body, self.attention_sum, self.output,
                *self.slot_buffers.values())


def _pack_dense_mlp_weights(layer):
    import torch
    import torch_npu

    gate = layer.mlp.gate_up_proj
    down = layer.mlp.down_proj
    w_gu = torch_npu.npu_format_cast(gate.weight.detach(), 2)
    w_dn = torch_npu.npu_format_cast(down.weight.detach(), 2)
    if w_gu.shape != (6144, 3072) or w_gu.dtype != torch.int8:
        raise ValueError(f"Dense gate_up shape/dtype mismatch: {w_gu.shape} {w_gu.dtype}")
    if w_dn.shape != (1536, 6144) or w_dn.dtype != torch.int8:
        raise ValueError(f"Dense down shape/dtype mismatch: {w_dn.shape} {w_dn.dtype}")
    p_gu = (
        w_gu.reshape(6, 1024, 24, 128)
        .permute(2, 0, 1, 3)
        .contiguous()
        .view(24, 6, 2, 512, 4, 32)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
    )
    p_gu = torch_npu.npu_format_cast(p_gu, 2)
    p_dn = (
        w_dn.reshape(6, 256, 24, 256)
        .permute(2, 0, 1, 3)
        .contiguous()
        .view(24, 6, 2, 128, 8, 32)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
    )
    p_dn = torch_npu.npu_format_cast(p_dn, 2)
    s_gu = gate.weight_scale.detach().float().contiguous()
    s_dn = down.weight_scale.detach().float().contiguous()
    if s_gu.shape != (3072,) or s_dn.shape != (6144,):
        raise ValueError(f"Dense scale shapes mismatch: {s_gu.shape} {s_dn.shape}")
    return (p_gu, s_gu, p_dn, s_dn)


def _apply_attention_nz(weights, f0_nz, tail_nz):
    if f0_nz and weights.front:
        f0 = weights.front[0]
        f0.copy_(f0.view(16, 2048, 8, 32).permute(0, 2, 1, 3).contiguous().view_as(f0))
    if tail_nz:
        t0 = weights.tail[0]
        t0.copy_(t0.view(16, 512, 8, 16).permute(0, 2, 1, 3).contiguous().view_as(t0))
        t1 = weights.tail[1]
        t1.copy_(t1.view(24, 2048, 8, 32).permute(0, 2, 1, 3).contiguous().view_as(t1))


def _apply_r4_moe_transforms(layers_list, device, flags, orig_gates):
    import torch

    flag_set = set(flags)
    f0_nz = "-DK11040_F0_NZ=1" in flag_set
    tail_nz = "-DK11040_TAIL_NZ=1" in flag_set
    shared_nz = "-DK11150_SHARED_W8_NZ=1" in flag_set
    gate_contig = "-DK11088_GATE_CONTIG=1" in flag_set
    down_contig = "-DK11220_DOWN_CONTIG_256=1" in flag_set
    preload_gate = "-DK112726_BRIDGE_PRELOAD_GATE=1" in flag_set

    with torch.npu.device(device), torch.inference_mode():
        seen_attn = set()
        seen_shared = set()
        seen_post = set()
        for item in layers_list:
            w = item.weights
            if f0_nz and w.front:
                p = int(w.front[0].data_ptr())
                if p not in seen_attn:
                    seen_attn.add(p)
                    f0 = w.front[0]
                    f0.copy_(f0.view(16, 2048, 8, 32).permute(0, 2, 1, 3).contiguous().view_as(f0))
            if tail_nz:
                p0, p1 = int(w.tail[0].data_ptr()), int(w.tail[1].data_ptr())
                if p0 not in seen_attn:
                    seen_attn.add(p0)
                    t0 = w.tail[0]
                    t0.copy_(t0.view(16, 512, 8, 16).permute(0, 2, 1, 3).contiguous().view_as(t0))
                if p1 not in seen_attn:
                    seen_attn.add(p1)
                    t1 = w.tail[1]
                    t1.copy_(t1.view(24, 2048, 8, 32).permute(0, 2, 1, 3).contiguous().view_as(t1))
            if shared_nz and w.shared:
                pg = int(w.shared[0].data_ptr())
                if pg not in seen_shared:
                    seen_shared.add(pg)
                    g = w.shared[0]
                    g.copy_(
                        g.view(24, 2, 512, 4, 32)
                        .permute(0, 1, 3, 2, 4)
                        .contiguous()
                        .view_as(g)
                    )
                pd = int(w.shared[2].data_ptr())
                if pd not in seen_shared:
                    seen_shared.add(pd)
                    d = w.shared[2]
                    d.copy_(
                        d.view(24, 2, 128, 8, 32)
                        .permute(0, 1, 3, 2, 4)
                        .contiguous()
                        .view_as(d)
                    )
            if preload_gate and w.post:
                pp = int(w.post[1].data_ptr())
                if pp not in seen_post:
                    seen_post.add(pp)
                    orig_gates[item.name] = w.post[1].detach().cpu().clone()
                    gw = w.post[1]
                    gw.copy_(
                        gw.reshape(HIDDEN, 16, 16)
                        .permute(1, 0, 2)
                        .contiguous()
                        .view(HIDDEN, 256)
                    )
        if gate_contig:
            acl = ctypes.CDLL("libascendcl.so")
            tmp = torch.empty((32 * 12582912,), dtype=torch.uint8, device=device)
            seen_g = set()
            for item in layers_list:
                if not item.weights.routed:
                    continue
                r0 = item.weights.routed[0]
                p0 = int(r0.data_ptr())
                if p0 in seen_g:
                    continue
                seen_g.add(p0)
                nbytes = r0.numel() * r0.element_size()
                cnt = nbytes // 12582912
                sub = tmp[:nbytes]
                torch.npu.synchronize()
                rc = acl.aclrtMemcpy(
                    ctypes.c_void_p(int(sub.data_ptr())),
                    nbytes,
                    ctypes.c_void_p(p0),
                    nbytes,
                    3,
                )
                if rc != 0:
                    raise RuntimeError("aclrtMemcpy failed packing routed gate")
                sub.copy_(
                    sub.view(cnt, 16, 4, 4, 49152)
                    .permute(0, 1, 3, 2, 4)
                    .contiguous()
                    .view(-1)
                )
                torch.npu.synchronize()
                rc = acl.aclrtMemcpy(
                    ctypes.c_void_p(p0),
                    nbytes,
                    ctypes.c_void_p(int(sub.data_ptr())),
                    nbytes,
                    3,
                )
                if rc != 0:
                    raise RuntimeError("aclrtMemcpy failed writing routed gate")
            del tmp
        if down_contig:
            acl = ctypes.CDLL("libascendcl.so")
            tmp = torch.empty((32 * 6291456,), dtype=torch.uint8, device=device)
            seen_d = set()
            for item in layers_list:
                if not item.weights.routed:
                    continue
                r3 = item.weights.routed[3]
                p3 = int(r3.data_ptr())
                if p3 in seen_d:
                    continue
                seen_d.add(p3)
                nbytes = r3.numel() * r3.element_size()
                cnt = nbytes // 6291456
                sub = tmp[:nbytes]
                torch.npu.synchronize()
                rc = acl.aclrtMemcpy(
                    ctypes.c_void_p(int(sub.data_ptr())),
                    nbytes,
                    ctypes.c_void_p(p3),
                    nbytes,
                    3,
                )
                if rc != 0:
                    raise RuntimeError("aclrtMemcpy failed packing routed down")
                sub.copy_(
                    sub.view(cnt, 24, 4, 2, 32768)
                    .permute(0, 1, 3, 2, 4)
                    .contiguous()
                    .view(-1)
                )
                torch.npu.synchronize()
                rc = acl.aclrtMemcpy(
                    ctypes.c_void_p(p3),
                    nbytes,
                    ctypes.c_void_p(int(sub.data_ptr())),
                    nbytes,
                    3,
                )
                if rc != 0:
                    raise RuntimeError("aclrtMemcpy failed writing routed down")
            del tmp
        torch.npu.synchronize()


def reclaim_hbm(*, target=None, draft=None, mgr=None):
    import torch

    for model_obj in (target, draft):
        if model_obj is None:
            continue
        re = getattr(getattr(model_obj, "attention", None), "rotary_emb", None)
        if re is not None:
            for nm in ("cos_sin_cache", "_rope_derived_cos_cache", "_rope_derived_sin_cache"):
                t = getattr(re, nm, None)
                if torch.is_tensor(t) and t.shape[0] > 16384:
                    re._buffers[nm] = t[:16384].contiguous().clone()
        blds = getattr(model_obj, "builders", None)
        if blds:
            for b in blds:
                obj = b[1] if isinstance(b, (list, tuple)) and len(b) > 1 else b
                if getattr(obj, "chunked_prefill_workspace", None) is not None:
                    obj.chunked_prefill_workspace = None
    og = getattr(mgr, "_orig_gates", None)
    if isinstance(og, dict):
        for k in list(og.keys()):
            v = og[k]
            if torch.is_tensor(v) and str(v.device).startswith("npu"):
                og[k] = v.cpu()
    gc.collect()
    torch.npu.empty_cache()


class _DenseWeights:
    def __init__(self, layer, moe0_weights, device, f0_nz=False, tail_nz=False):
        import torch

        self.attention = layer.self_attn.mla_attn.mla_attn
        self.impl = self.attention.impl
        self.mode = attention._validate_impl(self.impl)
        with torch.npu.device(device), torch.inference_mode():
            self.main, self.front, self.tail = attention._pack_weights(self.impl, self.mode)
            _apply_attention_nz(self, f0_nz, tail_nz)
            self.dense_mlp = _pack_dense_mlp_weights(layer)
            torch.npu.synchronize()
        self.caches = attention._cache_tensors(self.attention, self.mode)
        self.device = self.caches[0].device
        if getattr(layer.self_attn.mla_attn, "topk_indices_buffer", None) is None:
            layer.self_attn.mla_attn.topk_indices_buffer = self.impl.topk_indices_buffer
        self.topk = layer.self_attn.mla_attn.topk_indices_buffer
        self.routed = moe0_weights.routed
        self.shared = moe0_weights.shared
        self.input_gamma = attention._nd(layer.input_layernorm.weight)
        self.post = [
            attention._nd(layer.post_attention_layernorm.weight),
            moe0_weights.post[1],
            moe0_weights.post[2],
        ]
        self.scalars = (
            self.impl.scale,
            self.impl.q_a_layernorm.variance_epsilon,
            self.impl.kv_a_layernorm.variance_epsilon,
            layer.input_layernorm.variance_epsilon,
            layer.post_attention_layernorm.variance_epsilon,
        )
        self.owners = (
            *self.main,
            *self.front,
            *self.tail,
            *self.routed,
            *self.shared,
            self.input_gamma,
            *self.post,
            *self.caches,
            self.topk,
            *self.dense_mlp,
        )
        self.identities = tuple(map(tensor_identity, self.owners))

    def check_owners(self):
        if tuple(map(tensor_identity, self.owners)) != self.identities:
            raise RuntimeError("Dense whole-layer weight or cache storage changed")

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
        return main, index


class _DenseLayer:
    def __init__(self, manager, layer, moe0_weights, device, f0_nz=False, tail_nz=False):
        self.manager = manager
        self.layer = layer
        self.weights = _DenseWeights(layer, moe0_weights, device, f0_nz=f0_nz, tail_nz=tail_nz)
        self.name = self.weights.impl.layer_name
        self.original = layer.forward
        self.entries = {}
        self.reserves = {}
        self.counters = Counter()

    def reserve(self, reference):
        metadata = self.weights.metadata(reference.shape[0])
        if metadata is None:
            raise ValueError(f"{self.name}: target capture metadata is unsupported")
        geometry = attention.geometry_for(self.weights.mode, self.weights.caches, *metadata)
        key = reference.shape[0], geometry
        if key not in self.reserves:
            self.reserves[key] = _Arena(self, reference, metadata)
        self.counters["prepared"] += 1

    def forward(self, unused_layer, positions, hidden_states, residual, llama_4_scaling=None):
        import torch

        if self.manager.closed:
            raise RuntimeError("The whole-layer manager has been closed")
        rows = hidden_states.shape[0] if hidden_states.ndim == 2 else 0
        if rows not in WIDTHS:
            self.counters["other_width_fallback"] += 1
            return self.original(positions, hidden_states, residual, llama_4_scaling)
        pre_normed = residual is None
        if pre_normed:
            residual = hidden_states
            hidden_states = self.layer.input_layernorm(hidden_states)
        self.weights.check_owners()
        metadata = self.weights.metadata(rows)
        if metadata is None:
            raise ValueError(f"{self.name}: unsupported native metadata at M={rows}")
        geometry = attention.geometry_for(self.weights.mode, self.weights.caches, *metadata)
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
        if self.manager.chain_enabled and self.name == self.manager.layer_order[0]:
            self.manager.begin_chain()
        prepared = arena.bind(hidden_states, residual, metadata, window)
        prepared.local.indexed = (
            (1 if self.weights.mode != "shared" else 0)
            | 2
            | (4 if pre_normed else 0)
        )
        if self.manager.chain_enabled:
            self.manager.record_step(self.name, prepared, rows, capturing)
        else:
            prepared.launch()
        self.counters["submitted"] += 1
        self.counters[f"submitted_m{rows}"] += 1
        if capturing:
            self.manager.capture_layer(self.name, rows)
        return arena.output, arena.bridge[1]

    def restore(self):
        self.layer.forward = self.original


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
        if self.manager.chain_enabled and self.name == self.manager.layer_order[0]:
            self.manager.begin_chain()
        prepared = arena.bind(hidden_states, residual, metadata, window)
        _lookup_runner(self.weights.runner)
        if self.manager.chain_enabled:
            self.manager.record_step(self.name, prepared, rows, capturing)
        else:
            prepared.launch()
            if (self.weights.mode != "shared"
                    and self.weights.impl.use_index_cache
                    and not self.manager.tree.has_chain):
                self.weights.impl._update_indexcache_topk_indices(arena.selected)
        self.counters["submitted"] += 1
        self.counters[f"submitted_m{rows}"] += 1
        if capturing:
            self.manager.capture_layer(self.name, rows)
        return arena.output, arena.bridge[1]

    def restore(self):
        self.layer.forward = self.original


class WholeLayerManager:

    def __init__(self, config, libraries, *, device, rank, ranks, endpoint,
                 target_width=12, heap_bytes=1 << 30, chain_mode=True):
        import torch
        import torch_npu
        from vllm.distributed import get_tp_group

        if (target_width not in WIDTHS or ranks not in (8, 16)
                or not config.model_config.enforce_eager
                or getattr(config, "lora_config", None) is not None):
            raise ValueError(
                "Whole-layer requires eager Python capture, TP8/TP16 and a supported width"
            )
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
        self.tree_manifest = libraries.tree_manifest()
        tree_expected = None if self.tree_manifest is not None else LIBRARY_HASHES["tree"]
        self.tree = TreeLibrary(libraries.tree, tree_expected)
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
        self.chain_mode = bool(
            chain_mode
            and os.environ.get("TILERT_WHOLE_LAYER_CHAIN", "1") == "1"
        )
        self.chain_enabled = False
        self.chain_active = False
        self.layer_order = ()
        self.pending_steps = []
        self.slot_buffers = {}
        self.slot_refreshed = set()
        self.chain_reserves = {}
        self.chains = {}
        self.copy_stream = None
        self.tree_mask_ptr = 0
        self.peer_weights_ptr = 0
        self.peer_weights_buf = None
        self._orig_gates = {}

    @property
    def kernel_prof_enabled(self) -> bool:
        if self.tree_manifest is None:
            return False
        flags = self.tree_manifest.get("flags") or ()
        return bool(
            self.tree_manifest.get("enable_kernel_prof", False)
            or "-DENABLE_KERNEL_PROF=1" in flags
        )

    def capture_prof_acc(
        self,
        width: int | None = None,
        *,
        sync: bool = True,
        cpu: bool = True,
    ):
        import torch

        if not self.layer_order or self.layer_order[-1] not in self.layers:
            raise RuntimeError("Whole-layer model has no installed layers")
        if self.capture_stream is None:
            raise RuntimeError("Whole-layer model has no capture stream")
        if width is not None:
            target_rows = int(width)
        elif self.capture_history:
            target_rows = int(self.capture_history[-1]["rows"])
        else:
            target_rows = int(self.target_width)
        last_layer = self.layers[self.layer_order[-1]]
        stream_id = self.capture_stream.npu_stream
        matches = [
            arena
            for (entry_stream, entry_rows, _geom), arena in last_layer.entries.items()
            if entry_stream == stream_id and entry_rows == target_rows
        ]
        if not matches:
            raise RuntimeError(
                f"No capture-stream arena found for {last_layer.name} at width={target_rows}"
            )
        raws = []
        for arena in matches:
            body = getattr(arena, "body", None)
            if body is None or len(body) <= 6 or body[6] is None:
                raise RuntimeError(
                    f"Capture-stream arena missing body[6] scratch for {last_layer.name}"
                )
            raws.append(body[6])
        ptrs = {int(r.data_ptr()) if hasattr(r, "data_ptr") else id(r) for r in raws}
        if len(ptrs) > 1:
            raise RuntimeError(
                f"Ambiguous capture-stream arenas ({len(matches)}) for "
                f"{last_layer.name} at width={target_rows}"
            )
        raw = raws[0]
        nbytes = raw.numel() * raw.element_size()
        if nbytes < 512:
            raise RuntimeError("Arena body[6] scratch is smaller than 512 bytes")
        if nbytes % 8 != 0:
            raise RuntimeError(f"Arena body[6] scratch size {nbytes} is not 8-byte aligned")
        if sync and getattr(getattr(raw, "device", None), "type", None) == "npu":
            torch.npu.synchronize(self.device)
        words = raw if raw.dtype == torch.int64 else raw.view(torch.int64)
        prof = words.reshape(-1)[40:64]
        return prof.detach().cpu().clone() if cpu else prof

    def _allocate_chain_buffer(self):
        import torch

        chain_bytes = len(self.layer_order) * 1656
        raw = torch.empty(chain_bytes + 64, dtype=torch.uint8,
                          device=self.device)
        offset = (-int(raw.data_ptr())) % 64
        return raw[offset:offset + chain_bytes]

    def install_model(self, model):
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
        flags = list(self.tree_manifest.get("flags", ())) if self.tree_manifest else []
        flag_set = set(flags)
        use_dense78 = bool(
            self.chain_mode
            and self.tree.has_chain
            and self.tree._set_peer_weights_ptr is not None
            and ("-DK11220_DENSE78_COLD=1" in flag_set or self.tree_manifest is None)
        )
        dense_items = []
        if use_dense78:
            dense_raw = sorted(
                (layer for layer in values if layer.layer_idx < hf.first_k_dense_replace),
                key=lambda item: item.layer_idx,
            )
            dense_items = [
                _DenseLayer(
                    self,
                    lyr,
                    prepared[0].weights,
                    self.device,
                    f0_nz="-DK11040_F0_NZ=1" in flag_set,
                    tail_nz="-DK11040_TAIL_NZ=1" in flag_set,
                )
                for lyr in dense_raw
            ]
        all_items = [*dense_items, *prepared]
        names = [item.name for item in all_items]
        if len(set(names)) != len(names):
            raise ValueError("Whole-layer Decoder names are not unique")
        peers = [None] * self.ranks
        torch.distributed.all_gather_object(peers, names,
                                            group=self.group.cpu_group)
        if any(peer != names for peer in peers):
            raise RuntimeError("TP ranks selected different native Decoder layers")
        if flags:
            _apply_r4_moe_transforms(prepared, self.device, flags, self._orig_gates)
        if dense_items:
            total_bytes = len(all_items) * 640
            peer_raw = torch.zeros(total_bytes + 64, dtype=torch.uint8, device=self.device)
            peer_off = (-int(peer_raw.data_ptr())) % 64
            peer_sub = peer_raw[peer_off:peer_off + total_bytes]
            peer_cpu = bytearray(total_bytes)
            for li, item in enumerate(dense_items):
                p_gu, s_gu, p_dn, s_dn = item.weights.dense_mlp
                struct.pack_into(
                    "<4Q",
                    peer_cpu,
                    li * 640 + 128,
                    int(p_gu.data_ptr()),
                    int(s_gu.data_ptr()),
                    int(p_dn.data_ptr()),
                    int(s_dn.data_ptr()),
                )
            with torch.npu.device(self.device), torch.inference_mode():
                peer_sub.copy_(torch.frombuffer(peer_cpu, dtype=torch.uint8))
                torch.npu.synchronize()
            self.peer_weights_buf = peer_raw
            self.peer_weights_ptr = int(peer_sub.data_ptr())
        self.tree.initialize(self.device.index, self.rank, self.ranks,
                             self.endpoint, self.heap_bytes)
        self.initialized = True
        for item in all_items:
            self.layers[item.name] = item
            item.layer.forward = MethodType(item.forward, item.layer)
        self.dense = ([] if dense_items else
                      [layer.layer_idx for layer in values if layer not in expected])
        self.layer_order = tuple(names)
        indices = [item.layer.layer_idx for item in all_items]
        contiguous = indices == list(range(indices[0], indices[0] + len(indices)))
        self.chain_enabled = bool(self.chain_mode and self.tree.has_chain
                                  and contiguous)
        if self.chain_enabled:
            self.copy_stream = torch.npu.Stream(device=self.device)
        self.window(torch.npu.current_stream(self.device), False)
        return self

    def set_tree_mask_ptr(self, ptr):
        self.tree_mask_ptr = int(ptr)

    def begin_chain(self):
        self.chain_active = True
        self.slot_refreshed.clear()
        self.pending_steps = []

    def record_step(self, name, prepared, rows, capturing):
        expected_index = len(self.pending_steps)
        if (not self.chain_active
                or expected_index >= len(self.layer_order)
                or self.layer_order[expected_index] != name):
            self.chain_active = False
            self.pending_steps.clear()
            raise RuntimeError("Chain execution encountered an unexpected layer order")
        self.pending_steps.append(prepared)
        if len(self.pending_steps) == len(self.layer_order):
            steps = tuple(self.pending_steps)
            self.pending_steps.clear()
            self.chain_active = False
            self.slot_refreshed.clear()
            chain_key = (
                "capture" if capturing else "eager",
                rows,
                self.tree_mask_ptr,
                self.peer_weights_ptr,
                *steps,
            )
            if chain_key not in self.chains:
                if capturing:
                    if rows not in self.chain_reserves:
                        raise RuntimeError("Capture has no reserved chain descriptor buffer")
                    steps_device = self.chain_reserves.pop(rows)
                    self.chains[chain_key] = PreparedChain(
                        steps,
                        steps_device,
                        self.copy_stream,
                        mask_ptr=self.tree_mask_ptr,
                        peer_weights_ptr=self.peer_weights_ptr,
                    )
                else:
                    steps_device = self._allocate_chain_buffer()
                    self.chains[chain_key] = PreparedChain(
                        steps,
                        steps_device,
                        mask_ptr=self.tree_mask_ptr,
                        peer_weights_ptr=self.peer_weights_ptr,
                    )
            self.chains[chain_key].launch()

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
        if self.chain_enabled and rows not in self.chain_reserves:
            self.chain_reserves[rows] = self._allocate_chain_buffer()
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
        if (rows != self.capture_width
                or self.capture_seen != set(self.layers)):
            missing = sorted(set(self.layers) - self.capture_seen)
            raise RuntimeError(f"Whole-layer capture omitted native layers: {missing}")
        self.capture_history.append({"rows": rows,
                                     "layers": tuple(self.layers)})
        self.capture_width = None

    def coverage(self):
        storage = {}
        for layer in self.layers.values():
            for arena in (*layer.entries.values(), *layer.reserves.values()):
                for tensor in arena.tensors():
                    value = tensor.untyped_storage()
                    storage[(tensor.device, value.data_ptr())] = value.nbytes()
        for tensor in (*self.chain_reserves.values(),
                       *(chain.steps_device for chain in self.chains.values())):
            value = tensor.untyped_storage()
            storage[(tensor.device, value.data_ptr())] = value.nbytes()
        return {"installed": tuple(self.layers), "dense_layers": self.dense,
                "target_width": self.target_width,
                "chain_enabled": self.chain_enabled,
                "layers": {name: dict(layer.counters)
                           for name, layer in self.layers.items()},
                "captures": list(self.capture_history),
                "workspace_bytes": sum(storage.values()),
                "communication_window_bytes": sum(window.bytes
                                                   for window in self.windows.values())}

    def close_after_graphs(self):
        import torch

        if self.closed:
            return
        torch.npu.synchronize(self.device)
        if self.initialized:
            torch.distributed.barrier(group=self.group.cpu_group)
        for layer in self.layers.values():
            layer.restore()
        self.chains.clear()
        if hasattr(self, 'arena_pool'): self.arena_pool.clear()
        self.chain_reserves.clear()
        self.slot_buffers.clear()
        self.slot_refreshed.clear()
        self.pending_steps.clear()
        self.copy_stream = None
        self.peer_weights_buf = None
        self.peer_weights_ptr = 0
        self.tree_mask_ptr = 0
        self._orig_gates.clear()
        for window in reversed(tuple(self.windows.values())):
            window.close()
        self.windows.clear()
        if self.initialized:
            self.tree.finalize()
        self.capture_owner = None
        self.layers.clear()
        self.closed = True
