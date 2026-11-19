from __future__ import annotations

from contextlib import ExitStack
import ctypes
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import struct
from typing import TYPE_CHECKING

from .abi import DraftWeights, LayerWeights, Libraries, ModelArena, _alloc_aligned
from .config import (
    DRAFT_LAYER,
    FULL_INDEX_LAYERS,
    HIDDEN_SIZE,
    INT8_INDEX_LAYERS,
    K_NORM_EPS,
    MAIN_LAYERS,
    RMS_NORM_EPS,
    RuntimeConfig,
    TOTAL_LAYERS,
    VOCAB_SIZE,
)

if TYPE_CHECKING:
    import torch

HEADS = 8
LATENT = 512
VALUE = 256
HIDDEN = HIDDEN_SIZE


@dataclass(frozen=True)
class QuantizedWeight:

    values: torch.Tensor
    scale: torch.Tensor
    bits: int
    scale_bias: torch.Tensor | None = None

    @property
    def out_features(self) -> int:
        return self.values.shape[0] * (8 // self.bits)

    @property
    def in_features(self) -> int:
        return self.values.shape[1]

    def unpack(self) -> torch.Tensor:
        import torch

        if self.bits == 8:
            return self.values
        low = ((self.values & 15) ^ 8) - 8
        high = self.values >> 4
        return torch.stack((low, high), dim=1).reshape(
            self.out_features, self.in_features
        )

    def dequantize(self) -> torch.Tensor:
        return self.unpack().float() * self.scale[:, None]

    def reference(
        self, activation: torch.Tensor, activation_scale: torch.Tensor
    ) -> torch.Tensor:
        import torch

        product = activation.to(torch.int32) @ self.unpack().to(torch.int32).T
        return product.float() * self.scale * activation_scale.reshape(-1, 1)


@dataclass(frozen=True)
class ExpertWeights:
    gate: QuantizedWeight
    up: QuantizedWeight
    down: QuantizedWeight


class ModelSlimCheckpoint:

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        index = self.directory / "quant_model_weights.safetensors.index.json"
        self.weight_map = json.loads(index.read_text())["weight_map"]
        description = self.directory / "quant_model_description.json"
        self.quantization = json.loads(description.read_text())

    def tensor(self, name: str, selection=None) -> torch.Tensor:
        from safetensors import safe_open

        path = self.directory / self.weight_map[name]
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            if selection is None:
                value = checkpoint.get_tensor(name)
            else:
                value = checkpoint.get_slice(name)[selection]
            return value.clone().contiguous()

    def tensor_opt(self, name: str, selection=None) -> torch.Tensor | None:
        if name not in self.weight_map:
            return None
        return self.tensor(name, selection)

    def tensors(self, names):
        from safetensors import safe_open

        with ExitStack() as stack:
            shards = {}
            for name in names:
                filename = self.weight_map[name]
                if filename not in shards:
                    shards[filename] = stack.enter_context(
                        safe_open(self.directory / filename, framework="pt", device="cpu")
                    )
                value = shards[filename].get_tensor(name)
                yield name, value.clone().contiguous()

    def shard_tensors(self, names, selections=None) -> dict[str, torch.Tensor]:
        from safetensors import safe_open

        selections = selections or {}
        by_shard: dict[str, list[str]] = {}
        for name in names:
            by_shard.setdefault(self.weight_map[name], []).append(name)
        loaded: dict[str, torch.Tensor] = {}
        for filename, shard_names in by_shard.items():
            with safe_open(self.directory / filename, framework="pt", device="cpu") as f:
                for name in shard_names:
                    sel = selections.get(name)
                    val = f.get_tensor(name) if sel is None else f.get_slice(name)[sel]
                    loaded[name] = val.clone().contiguous()
        return loaded

    def shape(self, name: str) -> tuple[int, ...]:
        from safetensors import safe_open

        path = self.directory / self.weight_map[name]
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            return tuple(checkpoint.get_slice(name).get_shape())

    def float_tensor(self, name: str, selection=None) -> torch.Tensor:
        return self.tensor(name, selection)

    def w8a8_shard(
        self,
        name: str,
        output_range: tuple[int, int] | None = None,
        input_range: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qw = self.linear(name, output_range=output_range, input_range=input_range)
        if qw.bits != 8:
            raise ValueError(f"{name}: expected W8A8_DYNAMIC, got W{qw.bits}A8")
        return qw.values, qw.scale

    def linear(
        self,
        name: str,
        output_range: tuple[int, int] | None = None,
        input_range: tuple[int, int] | None = None,
    ) -> QuantizedWeight:
        import torch

        kind = self.quantization[name + ".weight"]
        if kind not in ("W4A8_DYNAMIC", "W8A8_DYNAMIC"):
            raise ValueError(f"{name}: unsupported quantization {kind}")
        bits = 4 if kind == "W4A8_DYNAMIC" else 8
        packed_rows, columns = self.shape(name + ".weight")
        packing = 8 // bits
        rows = packed_rows * packing
        out_start, out_end = output_range or (0, rows)
        in_start, in_end = input_range or (0, columns)
        if not (0 <= out_start < out_end <= rows):
            raise ValueError(f"{name}: output range exceeds {rows} channels")
        if not (0 <= in_start < in_end <= columns):
            raise ValueError(f"{name}: input range exceeds {columns} channels")
        if out_start % packing or out_end % packing:
            raise ValueError(f"{name}: W4 output boundaries must be even")

        output = slice(out_start, out_end)
        packed_output = slice(out_start // packing, out_end // packing)
        need = [name + ".weight", name + ".weight_scale", name + ".weight_offset"]
        sels = {
            name + ".weight": (packed_output, slice(in_start, in_end)),
            name + ".weight_scale": output,
            name + ".weight_offset": output,
        }
        if bits == 4:
            partitions = self.shape(name + ".scale_bias")[1]
            partition_size = columns // partitions
            if in_start % partition_size or in_end % partition_size:
                raise ValueError(
                    f"{name}: input boundaries must align to "
                    f"{partition_size}-channel scale_bias partitions"
                )
            need.append(name + ".scale_bias")
            sels[name + ".scale_bias"] = (
                output,
                slice(in_start // partition_size, in_end // partition_size),
            )
        loaded = self.shard_tensors(need, sels)
        offset = loaded[name + ".weight_offset"]
        if torch.count_nonzero(offset).item():
            raise ValueError(f"{name}: nonzero weight offsets are unsupported")
        values = loaded[name + ".weight"]
        scale = loaded[name + ".weight_scale"].flatten().float()
        scale_bias = None
        if bits == 4:
            scale_bias = loaded[name + ".scale_bias"].float().sum(dim=1)
        return QuantizedWeight(values, scale, bits, scale_bias)

    def expert(
        self,
        layer: int,
        expert: int,
        rank: int | None = None,
        tp_size: int = 1,
    ) -> ExpertWeights:
        prefix = f"model.layers.{layer}.mlp.experts.{expert}"
        if tp_size == 1:
            return ExpertWeights(
                gate=self.linear(prefix + ".gate_proj"),
                up=self.linear(prefix + ".up_proj"),
                down=self.linear(prefix + ".down_proj"),
            )
        if tp_size != 16:
            raise ValueError(f"Unsupported expert tp_size={tp_size}; expected 1 or 16")
        if rank is None or not 0 <= rank < 16:
            raise ValueError("TP16 rank must be between 0 and 15")
        channels = (rank * 128, (rank + 1) * 128)
        return ExpertWeights(
            gate=self.linear(prefix + ".gate_proj", output_range=channels),
            up=self.linear(prefix + ".up_proj", output_range=channels),
            down=self.linear(prefix + ".down_proj", input_range=channels),
        )


def unpack_scale(packed, columns, experts=256):
    import torch

    if packed.dtype != torch.int64 or tuple(packed.shape) not in (
        (experts, columns),
        (experts, 1, columns),
    ):
        raise ValueError(f"Routed scales require INT64 [{experts},C] carriers")
    words = packed.detach().cpu().contiguous().reshape(experts, columns)
    if not bool(((words >= 0) & (words <= 0xFFFFFFFF)).all()):
        raise ValueError("Routed scale carriers must zero-extend FP32 bits")
    values = words.to(torch.int32).view(torch.float32)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Routed scales must decode to finite FP32 values")
    return values


def pack_value_weights(weight: torch.Tensor) -> torch.Tensor:
    import torch

    if (
        weight.dim() != 3
        or weight.shape[0] not in (4, 8)
        or weight.shape[1:] != (LATENT, VALUE)
        or weight.dtype != torch.bfloat16
    ):
        raise ValueError("MLA value weights require BF16 [4|8,512,256]")
    heads = weight.shape[0]
    tile = (heads * VALUE) // 16
    return (
        weight.reshape(heads, LATENT, VALUE // tile, tile)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(16, LATENT, tile)
    )


def pack_output_weights(weight: torch.Tensor) -> torch.Tensor:
    import torch

    if (
        weight.dim() != 2
        or weight.shape[0] not in (4 * VALUE, 8 * VALUE)
        or weight.shape[1] != HIDDEN
        or weight.dtype != torch.int8
    ):
        raise ValueError("MLA output weights require INT8 [1024|2048,6144]")
    return weight.reshape(weight.shape[0], 24, 256).permute(1, 0, 2).contiguous()


def pack_shared_weights(
    gate_up: torch.Tensor, down: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    if gate_up.dtype != torch.int8 or down.dtype != torch.int8:
        raise ValueError("Shared weights require INT8")
    if gate_up.shape == (HIDDEN, 256) and down.shape == (128, HIDDEN):
        gate = gate_up.reshape(6, 1024, 4, 64).permute(0, 2, 1, 3)
        output = down.reshape(128, 24, 256).permute(1, 0, 2)
        return gate.contiguous().reshape(24, 1024, 64), output.contiguous()
    if gate_up.shape == (HIDDEN, 512) and down.shape == (256, HIDDEN):
        gate = gate_up.reshape(6, 1024, 4, 128).permute(0, 2, 1, 3)
        output = down.reshape(256, 24, 256).permute(1, 0, 2)
        return gate.contiguous().reshape(24, 1024, 128), output.contiguous()
    raise ValueError(
        "Shared weights require INT8 [6144,256]/[128,6144] or [6144,512]/[256,6144]"
    )


def pack_indexer_weights(
    query: torch.Tensor, key: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    if query.dtype != torch.int8 or tuple(query.shape) != (2048, 4096):
        raise ValueError(f"indexer WQ requires INT8 (2048, 4096), got {query.shape}")
    if key.dtype != torch.bfloat16 or tuple(key.shape) != (160, HIDDEN):
        raise ValueError(f"indexer WK requires BF16 (160, {HIDDEN}), got {key.shape}")
    packed_q = query.reshape(2048, 16, 256).permute(1, 0, 2).contiguous()
    packed_k = key.T.reshape(HIDDEN, 5, 32).permute(1, 0, 2).contiguous()
    return packed_q, packed_k


def _pack_dense_mlp_weights(
    w_gu: torch.Tensor,
    s_gu: torch.Tensor,
    w_dn: torch.Tensor,
    s_dn: torch.Tensor,
    world_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    import torch

    d_dim = 12288 // int(world_size)
    gu_cols = 2 * d_dim
    gu_aic_cols = gu_cols // 24
    gu_g32 = gu_aic_cols // 32
    dn_kc = d_dim // 256
    if w_gu.shape != (HIDDEN, gu_cols) or w_gu.dtype != torch.int8:
        raise ValueError(f"Dense gate_up shape/dtype mismatch: {w_gu.shape} {w_gu.dtype}")
    if w_dn.shape != (d_dim, HIDDEN) or w_dn.dtype != torch.int8:
        raise ValueError(f"Dense down shape/dtype mismatch: {w_dn.shape} {w_dn.dtype}")
    p_gu = (
        w_gu.reshape(6, 1024, 24, gu_aic_cols)
        .permute(2, 0, 1, 3)
        .contiguous()
        .view(24, 6, 2, 512, gu_g32, 32)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
    )
    p_dn = (
        w_dn.reshape(dn_kc, 256, 24, 256)
        .permute(2, 0, 1, 3)
        .contiguous()
        .view(24, dn_kc, 2, 128, 8, 32)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
    )
    s_gu_f = s_gu.detach().float().reshape(-1).contiguous()
    s_dn_f = s_dn.detach().float().reshape(-1).contiguous()
    if s_gu_f.shape != (gu_cols,) or s_dn_f.shape != (HIDDEN,):
        raise ValueError(f"Dense scale shapes mismatch: {s_gu_f.shape} {s_dn_f.shape}")
    return p_gu, s_gu_f, p_dn, s_dn_f


def load_rot_shard(rot_path: Path, rank: int, world_size: int = 8) -> torch.Tensor:
    from safetensors import safe_open

    path = Path(rot_path)
    if path.is_dir():
        path = path / "rot.safetensors"
    with safe_open(path, framework="pt", device="cpu") as f:
        rows = f.get_slice("rot.weight").get_shape()[0]
        per_rank = rows // world_size
        shard = f.get_slice("rot.weight")[rank * per_rank : (rank + 1) * per_rank, :]
        return shard.clone().t().contiguous()


def build_rope_caches(
    max_positions: int = 16384,
    rope_theta: float = 8000000.0,
    rope_dim: int = 64,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    dev = torch.device(device)
    inv_freq = 1.0 / (
        float(rope_theta)
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=dev) / rope_dim)
    )
    positions = torch.arange(int(max_positions), dtype=torch.float32, device=dev)
    freqs = torch.einsum("i,j -> ij", positions, inv_freq)
    cos_sin = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)
    half = cos_sin.shape[-1] // 2
    cos_part, sin_part = (
        cos_sin.view(-1, 2, half).repeat(1, 1, 2).chunk(2, dim=1)
    )
    cos_cache = cos_part.squeeze(1).contiguous()
    sin_cache = sin_part.squeeze(1).contiguous()
    return cos_cache, sin_cache


def hadamard_128(device: str | torch.device = "cpu") -> torch.Tensor:
    import torch

    signs = [
        [1 if (row & col).bit_count() % 2 == 0 else -1 for col in range(128)]
        for row in range(128)
    ]
    return torch.tensor(signs, dtype=torch.bfloat16, device=device) / (128**0.5)


class _NdSlabBuilder:

    def __init__(self, align: int = 512):
        self.align = int(align)
        self.specs: dict[str, tuple[tuple[int, ...], torch.dtype, int, int]] = {}
        self.total_bytes = 0

    def reserve(self, key: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        import torch

        numel = 1
        for dim in shape:
            numel *= int(dim)
        nbytes = numel * torch.empty((), dtype=dtype).element_size()
        offset = self.total_bytes
        self.specs[key] = (tuple(int(d) for d in shape), dtype, offset, nbytes)
        aligned = ((nbytes + self.align - 1) // self.align) * self.align
        self.total_bytes += aligned

    def allocate(self, device) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        raw, slab = _alloc_aligned(self.total_bytes, self.align, device, zero=False)
        views = {}
        for key, (shape, dtype, offset, nbytes) in self.specs.items():
            views[key] = slab[offset : offset + nbytes].view(dtype).view(shape)
        return raw, slab, views


def _to_nz(tensor_cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    dev_t = tensor_cpu.to(device=device).contiguous()
    if device.type != "npu":
        return dev_t
    import torch_npu

    nz = torch_npu.npu_format_cast(dev_t, 29)
    del dev_t
    return nz


def _apply_r4_routed_contig(
    tensor_nz_i32: torch.Tensor,
    tmp_u8: torch.Tensor,
    per_expert_bytes: int,
    permute_shape: tuple[int, ...],
    acl,
) -> None:
    import torch

    nbytes = tensor_nz_i32.numel() * tensor_nz_i32.element_size()
    cnt = nbytes // per_expert_bytes
    buf_a = tmp_u8[:per_expert_bytes]
    buf_b = tmp_u8[per_expert_bytes : 2 * per_expert_bytes]
    ptr = int(tensor_nz_i32.data_ptr())
    a_ptr = int(buf_a.data_ptr())
    b_ptr = int(buf_b.data_ptr())
    perm_dst = tuple(permute_shape[i] for i in (0, 2, 1, 3))
    torch.npu.synchronize()
    for e in range(cnt):
        e_ptr = ptr + e * per_expert_bytes
        rc = acl.aclrtMemcpy(
            ctypes.c_void_p(a_ptr), per_expert_bytes, ctypes.c_void_p(e_ptr), per_expert_bytes, 3
        )
        if rc != 0:
            raise RuntimeError(f"aclrtMemcpy D2D read failed with {rc}")
        buf_b.view(*perm_dst).copy_(buf_a.view(*permute_shape).permute(0, 2, 1, 3))
        torch.npu.synchronize()
        rc = acl.aclrtMemcpy(
            ctypes.c_void_p(e_ptr), per_expert_bytes, ctypes.c_void_p(b_ptr), per_expert_bytes, 3
        )
        if rc != 0:
            raise RuntimeError(f"aclrtMemcpy D2D write failed with {rc}")


def _check_zero_offset(loaded: dict[str, torch.Tensor], name: str) -> None:
    import torch

    if torch.count_nonzero(loaded[name]).item():
        raise ValueError(f"{name}: nonzero weight offsets are unsupported")


def _load_attention_weights(
    ckpt: ModelSlimCheckpoint,
    layer: int,
    rank: int,
    world_size: int,
    device: torch.device,
    nd_views: dict[str, torch.Tensor],
    nz_holders: list[torch.Tensor],
    q_hadamard: torch.Tensor,
    q_hadamard_zero: torch.Tensor,
    *,
    f0_nz: bool,
    tail_nz: bool,
) -> tuple[tuple, tuple, tuple, torch.Tensor]:
    import torch

    prefix = f"model.layers.{layer}"
    attn = prefix + ".self_attn"
    heads = 64 // world_size
    q_b_slice = slice(rank * (heads * 256), (rank + 1) * (heads * 256))
    kv_b_slice = slice(rank * (heads * 448), (rank + 1) * (heads * 448))
    o_slice = slice(rank * (heads * 256), (rank + 1) * (heads * 256))

    names = [
        prefix + ".input_layernorm.weight",
        attn + ".q_a_proj.weight",
        attn + ".q_a_proj.weight_scale",
        attn + ".q_a_proj.weight_offset",
        attn + ".q_b_proj.weight",
        attn + ".q_b_proj.weight_scale",
        attn + ".q_b_proj.weight_offset",
        attn + ".kv_a_proj_with_mqa.weight",
        attn + ".kv_a_proj_with_mqa.weight_scale",
        attn + ".kv_a_proj_with_mqa.weight_offset",
        attn + ".kv_b_proj.weight",
        attn + ".q_a_layernorm.weight",
        attn + ".kv_a_layernorm.weight",
        attn + ".o_proj.weight",
        attn + ".o_proj.weight_scale",
        attn + ".o_proj.weight_offset",
    ]
    sels = {
        attn + ".q_b_proj.weight": (q_b_slice, slice(None)),
        attn + ".q_b_proj.weight_scale": q_b_slice,
        attn + ".q_b_proj.weight_offset": q_b_slice,
        attn + ".kv_b_proj.weight": (kv_b_slice, slice(None)),
        attn + ".o_proj.weight": (slice(None), o_slice),
    }
    has_idx = layer in FULL_INDEX_LAYERS
    if has_idx:
        idx_p = attn + ".indexer"
        names.extend(
            [
                idx_p + ".wq_b.weight",
                idx_p + ".wq_b.weight_scale",
                idx_p + ".wq_b.weight_offset",
                idx_p + ".wk.weight",
                idx_p + ".weights_proj.weight",
                idx_p + ".k_norm.weight",
                idx_p + ".k_norm.bias",
            ]
        )
    loaded = ckpt.shard_tensors(names, sels)
    for off_name in (
        attn + ".q_a_proj.weight_offset",
        attn + ".q_b_proj.weight_offset",
        attn + ".kv_a_proj_with_mqa.weight_offset",
        attn + ".o_proj.weight_offset",
    ):
        _check_zero_offset(loaded, off_name)
    if has_idx:
        _check_zero_offset(loaded, attn + ".indexer.wq_b.weight_offset")

    w_dq_nz = _to_nz(loaded[attn + ".q_a_proj.weight"].t().contiguous(), device)
    w_uq_nz = _to_nz(loaded[attn + ".q_b_proj.weight"].t().contiguous(), device)
    w_dkv_nz = _to_nz(
        loaded[attn + ".kv_a_proj_with_mqa.weight"].t().contiguous(), device
    )
    nz_holders.extend((w_dq_nz, w_uq_nz, w_dkv_nz))

    kv_b = loaded[attn + ".kv_b_proj.weight"].t().contiguous().view(512, heads, 448)
    w_uk, w_uv = kv_b.split((192, 256), dim=-1)
    w_uk_t = w_uk.permute(1, 2, 0).contiguous()
    w_uv_p = pack_value_weights(w_uv.transpose(0, 1).contiguous())
    if tail_nz:
        uv_tile = (heads * VALUE) // 16
        uv_n16 = uv_tile // 16
        w_uv_p = (
            w_uv_p.view(16, 512, uv_n16, 16)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(16, 512, uv_tile)
        )
    w_o_p = pack_output_weights(loaded[attn + ".o_proj.weight"].t().contiguous())
    if tail_nz:
        o_in = heads * VALUE
        w_o_p = (
            w_o_p.view(24, o_in, 8, 32)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(24, o_in, 256)
        )

    pfx = f"l{layer}"
    nd_views[f"{pfx}.input_gamma"].copy_(loaded[prefix + ".input_layernorm.weight"])
    nd_views[f"{pfx}.w_uk_t"].copy_(w_uk_t)
    nd_views[f"{pfx}.s_dq"].copy_(
        loaded[attn + ".q_a_proj.weight_scale"].float().view(1, 2048)
    )
    nd_views[f"{pfx}.s_uq_qr"].copy_(
        loaded[attn + ".q_b_proj.weight_scale"].float().view(1, heads * 256)
    )
    nd_views[f"{pfx}.s_dkv_kr"].copy_(
        loaded[attn + ".kv_a_proj_with_mqa.weight_scale"].float().view(1, 576)
    )
    nd_views[f"{pfx}.q_a_norm"].copy_(loaded[attn + ".q_a_layernorm.weight"])
    nd_views[f"{pfx}.kv_a_norm"].copy_(loaded[attn + ".kv_a_layernorm.weight"])
    nd_views[f"{pfx}.tail_uv"].copy_(w_uv_p)
    nd_views[f"{pfx}.tail_o"].copy_(w_o_p)
    nd_views[f"{pfx}.tail_s"].copy_(loaded[attn + ".o_proj.weight_scale"].float().reshape(HIDDEN))

    main = (
        w_dq_nz,
        w_uq_nz,
        nd_views[f"{pfx}.w_uk_t"],
        w_dkv_nz,
        nd_views[f"{pfx}.s_dq"],
        nd_views[f"{pfx}.s_uq_qr"],
        nd_views[f"{pfx}.s_dkv_kr"],
        nd_views[f"{pfx}.q_a_norm"],
        nd_views[f"{pfx}.kv_a_norm"],
    )
    tail = (
        nd_views[f"{pfx}.tail_uv"],
        nd_views[f"{pfx}.tail_o"],
        nd_views[f"{pfx}.tail_s"],
    )
    front = ()
    if has_idx:
        idx_p = attn + ".indexer"
        wq_t = loaded[idx_p + ".wq_b.weight"].t().contiguous()
        wk_cat = torch.cat(
            (loaded[idx_p + ".wk.weight"], loaded[idx_p + ".weights_proj.weight"]), dim=0
        ).contiguous()
        fq, fk = pack_indexer_weights(wq_t, wk_cat)
        if f0_nz:
            fq = (
                fq.view(16, 2048, 8, 32)
                .permute(0, 2, 1, 3)
                .contiguous()
                .view(16, 2048, 256)
            )
        nd_views[f"{pfx}.front_q"].copy_(fq)
        nd_views[f"{pfx}.front_qs"].copy_(
            loaded[idx_p + ".wq_b.weight_scale"].to(torch.bfloat16).reshape(4096)
        )
        nd_views[f"{pfx}.front_k"].copy_(fk)
        nd_views[f"{pfx}.front_knw"].copy_(loaded[idx_p + ".k_norm.weight"].float())
        nd_views[f"{pfx}.front_knb"].copy_(loaded[idx_p + ".k_norm.bias"].float())
        had = q_hadamard if layer in INT8_INDEX_LAYERS else q_hadamard_zero
        front = (
            nd_views[f"{pfx}.front_q"],
            nd_views[f"{pfx}.front_qs"],
            nd_views[f"{pfx}.front_k"],
            nd_views[f"{pfx}.front_knw"],
            nd_views[f"{pfx}.front_knb"],
            had,
        )
    input_gamma = nd_views[f"{pfx}.input_gamma"]
    del loaded
    return main, front, tail, input_gamma


def _load_moe_body(
    ckpt: ModelSlimCheckpoint,
    layer: int,
    rank: int,
    world_size: int,
    local_experts: int,
    device: torch.device,
    nd_views: dict[str, torch.Tensor],
    nz_holders: list[torch.Tensor],
    w13_cpu: torch.Tensor,
    w2_cpu: torch.Tensor,
    tmp_contig: torch.Tensor | None,
    acl,
    *,
    shared_nz: bool,
    gate_contig: bool,
    down_contig: bool,
    preload_gate: bool,
) -> tuple[tuple, tuple, tuple]:
    import torch

    prefix = f"model.layers.{layer}"
    sh_p = prefix + ".mlp.shared_experts"
    sh_dim = 2048 // world_size
    sh_out = slice(rank * sh_dim, (rank + 1) * sh_dim)
    names = [
        prefix + ".post_attention_layernorm.weight",
        prefix + ".mlp.gate.weight",
        prefix + ".mlp.gate.e_score_correction_bias",
        sh_p + ".gate_proj.weight",
        sh_p + ".gate_proj.weight_scale",
        sh_p + ".gate_proj.weight_offset",
        sh_p + ".up_proj.weight",
        sh_p + ".up_proj.weight_scale",
        sh_p + ".up_proj.weight_offset",
        sh_p + ".down_proj.weight",
        sh_p + ".down_proj.weight_scale",
        sh_p + ".down_proj.weight_offset",
    ]
    sels = {
        sh_p + ".gate_proj.weight": (sh_out, slice(None)),
        sh_p + ".gate_proj.weight_scale": sh_out,
        sh_p + ".gate_proj.weight_offset": sh_out,
        sh_p + ".up_proj.weight": (sh_out, slice(None)),
        sh_p + ".up_proj.weight_scale": sh_out,
        sh_p + ".up_proj.weight_offset": sh_out,
        sh_p + ".down_proj.weight": (slice(None), sh_out),
    }
    exp_start = rank * local_experts
    for e_idx in range(exp_start, exp_start + local_experts):
        ep = f"{prefix}.mlp.experts.{e_idx}"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            names.extend(
                [
                    f"{ep}.{proj}.weight",
                    f"{ep}.{proj}.weight_scale",
                    f"{ep}.{proj}.weight_offset",
                    f"{ep}.{proj}.scale_bias",
                ]
            )
    loaded = ckpt.shard_tensors(names, sels)
    for off_name in (
        sh_p + ".gate_proj.weight_offset",
        sh_p + ".up_proj.weight_offset",
        sh_p + ".down_proj.weight_offset",
    ):
        _check_zero_offset(loaded, off_name)

    pfx = f"l{layer}"
    nd_views[f"{pfx}.post_gamma"].copy_(loaded[prefix + ".post_attention_layernorm.weight"])
    gw = loaded[prefix + ".mlp.gate.weight"].to(torch.bfloat16).t().contiguous()
    if preload_gate:
        gw = gw.reshape(HIDDEN, 16, 16).permute(1, 0, 2).contiguous().view(HIDDEN, 256)
    nd_views[f"{pfx}.router_w"].copy_(gw)
    nd_views[f"{pfx}.router_b"].copy_(
        loaded[prefix + ".mlp.gate.e_score_correction_bias"].float()
    )
    post = (
        nd_views[f"{pfx}.post_gamma"],
        nd_views[f"{pfx}.router_w"],
        nd_views[f"{pfx}.router_b"],
    )

    sh_gu_w = (
        torch.cat(
            (loaded[sh_p + ".gate_proj.weight"], loaded[sh_p + ".up_proj.weight"]),
            dim=0,
        )
        .t()
        .contiguous()
    )
    sh_dn_w = loaded[sh_p + ".down_proj.weight"].t().contiguous()
    sh_p_gu, sh_p_dn = pack_shared_weights(sh_gu_w, sh_dn_w)
    if shared_nz:
        sh_gu_tile = sh_dim // 2
        sh_gu_g32 = sh_gu_tile // 32
        sh_dn_k128 = sh_dim // 128
        sh_p_gu = (
            sh_p_gu.view(24, 2, 512, sh_gu_g32, 32)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(24, 1024, sh_gu_tile)
        )
        sh_p_dn = (
            sh_p_dn.view(24, sh_dn_k128, 128, 8, 32)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(24, sh_dim, 256)
        )
    sh_s_gu = torch.cat(
        (
            loaded[sh_p + ".gate_proj.weight_scale"].reshape(-1),
            loaded[sh_p + ".up_proj.weight_scale"].reshape(-1),
        ),
        dim=0,
    ).float()
    sh_s_dn = loaded[sh_p + ".down_proj.weight_scale"].reshape(-1).float()
    nd_views[f"{pfx}.sh_gu"].copy_(sh_p_gu)
    nd_views[f"{pfx}.sh_s_gu"].copy_(sh_s_gu)
    nd_views[f"{pfx}.sh_dn"].copy_(sh_p_dn)
    nd_views[f"{pfx}.sh_s_dn"].copy_(sh_s_dn)
    shared = (
        nd_views[f"{pfx}.sh_gu"],
        nd_views[f"{pfx}.sh_s_gu"],
        nd_views[f"{pfx}.sh_dn"],
        nd_views[f"{pfx}.sh_s_dn"],
    )

    r1_cpu = torch.empty((local_experts, 4096), dtype=torch.float32)
    r2_cpu = torch.empty((local_experts, 4096), dtype=torch.float32)
    r4_cpu = torch.empty((local_experts, HIDDEN), dtype=torch.float32)
    r5_cpu = torch.empty((local_experts, HIDDEN), dtype=torch.float32)
    for local_i, e_idx in enumerate(range(exp_start, exp_start + local_experts)):
        ep = f"{prefix}.mlp.experts.{e_idx}"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            _check_zero_offset(loaded, f"{ep}.{proj}.weight_offset")
        w13_cpu[local_i, :, :1024].copy_(loaded[f"{ep}.gate_proj.weight"].t())
        w13_cpu[local_i, :, 1024:].copy_(loaded[f"{ep}.up_proj.weight"].t())
        w2_cpu[local_i].copy_(loaded[f"{ep}.down_proj.weight"].t())
        r1_cpu[local_i, :2048].copy_(loaded[f"{ep}.gate_proj.scale_bias"].float().reshape(-1))
        r1_cpu[local_i, 2048:].copy_(loaded[f"{ep}.up_proj.scale_bias"].float().reshape(-1))
        r2_cpu[local_i, :2048].copy_(
            loaded[f"{ep}.gate_proj.weight_scale"].float().reshape(-1)
        )
        r2_cpu[local_i, 2048:].copy_(loaded[f"{ep}.up_proj.weight_scale"].float().reshape(-1))
        r4_cpu[local_i].copy_(loaded[f"{ep}.down_proj.scale_bias"].float().sum(dim=1))
        r5_cpu[local_i].copy_(loaded[f"{ep}.down_proj.weight_scale"].float().reshape(-1))
    del loaded

    nd_views[f"{pfx}.r1"].copy_(r1_cpu)
    nd_views[f"{pfx}.r2"].copy_(r2_cpu)
    nd_views[f"{pfx}.r4"].copy_(r4_cpu)
    nd_views[f"{pfx}.r5"].copy_(r5_cpu)
    del r1_cpu, r2_cpu, r4_cpu, r5_cpu

    r0_nz = _to_nz(w13_cpu, device).view(torch.int32)
    if gate_contig and device.type == "npu":
        _apply_r4_routed_contig(r0_nz, tmp_contig, 12582912, (16, 4, 4, 49152), acl)
    r3_nz = _to_nz(w2_cpu, device).view(torch.int32)
    if down_contig and device.type == "npu":
        _apply_r4_routed_contig(r3_nz, tmp_contig, 6291456, (24, 4, 2, 32768), acl)
    nz_holders.extend((r0_nz, r3_nz))

    routed = (
        r0_nz,
        nd_views[f"{pfx}.r1"],
        nd_views[f"{pfx}.r2"],
        r3_nz,
        nd_views[f"{pfx}.r4"],
        nd_views[f"{pfx}.r5"],
    )
    return post, shared, routed


def _static_routing_prior(
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    post_gamma: torch.Tensor,
    x_probe: torch.Tensor,
    world_size: int,
    local_experts: int,
) -> torch.Tensor:
    import torch

    gw = gate_weight.detach().cpu().float()
    gb = gate_bias.detach().cpu().float().reshape(-1)
    pg = post_gamma.detach().cpu().float().reshape(1, -1)
    scores = torch.sigmoid((x_probe * pg) @ gw.t()) + gb
    g_scores = scores.view(-1, world_size, local_experts)
    top4_g = torch.topk(torch.topk(g_scores, 2, dim=-1).values.sum(dim=-1), 4, dim=-1).indices
    mask = torch.zeros_like(g_scores, dtype=torch.bool).scatter_(
        1, top4_g.unsqueeze(-1).expand(-1, -1, local_experts), True
    )
    masked_s = scores.masked_fill(~mask.view(-1, world_size * local_experts), -1e9)
    idx = torch.topk(masked_s, 8, dim=-1).indices
    freq = torch.bincount(idx.reshape(-1), minlength=world_size * local_experts).float()
    return freq / freq.sum().clamp(min=1.0)


def _plan_moe_mirrors(
    ckpt: ModelSlimCheckpoint,
    rank: int,
    world_size: int,
    local_experts: int,
    *,
    mirror_64: bool,
    total_budget: int,
) -> dict[int, tuple[list[int], list[int], list[int]]]:
    import torch

    gen = torch.Generator(device="cpu").manual_seed(0)
    x_probe = torch.randn((1024, HIDDEN), generator=gen, dtype=torch.float32)
    freqs: dict[int, torch.Tensor] = {}
    ranked: dict[int, list[list[int]]] = {}
    for li in range(3, MAIN_LAYERS):
        p = f"model.layers.{li}"
        loaded = ckpt.shard_tensors(
            [
                f"{p}.mlp.gate.weight",
                f"{p}.mlp.gate.e_score_correction_bias",
                f"{p}.post_attention_layernorm.weight",
            ]
        )
        freq = _static_routing_prior(
            loaded[f"{p}.mlp.gate.weight"],
            loaded[f"{p}.mlp.gate.e_score_correction_bias"],
            loaded[f"{p}.post_attention_layernorm.weight"],
            x_probe,
            world_size,
            local_experts,
        )
        freqs[li] = freq
        ranked[li] = [
            [
                hr * local_experts + int(le)
                for le in torch.topk(
                    freq[hr * local_experts : (hr + 1) * local_experts], local_experts
                ).indices.tolist()
            ]
            for hr in range(world_size)
        ]

    min_s = 4 if mirror_64 else 6
    max_s = 8 if mirror_64 else 6
    slots = {li: min_s for li in range(3, MAIN_LAYERS)}
    rem = total_budget - sum(slots.values())
    while rem > 0:
        best_li = -1
        best_gain = -1.0
        for li in range(3, MAIN_LAYERS):
            s = slots[li]
            if s < max_s:
                gain = sum(float(freqs[li][ranked[li][hr][s]]) for hr in range(world_size))
                if gain > best_gain:
                    best_gain = gain
                    best_li = li
        if best_li < 0:
            break
        slots[best_li] += 1
        rem -= 1

    plan: dict[int, tuple[list[int], list[int], list[int]]] = {}
    for li in range(3, MAIN_LAYERS):
        mslots = slots[li]
        cands: list[int] = []
        masks: list[int] = []
        by_rank: list[list[int]] = [[] for _ in range(world_size)]
        for k in range(mslots):
            step = 1 + (k % (world_size - 1))
            for hr in range(world_size):
                ge = ranked[li][hr][k]
                mr = (hr + step) % world_size
                cands.append(ge)
                masks.append((1 << hr) | (1 << mr))
                by_rank[mr].append(ge)
        plan[li] = (cands, masks, by_rank[rank])
    return plan


def _load_moe_mirrors(
    ckpt: ModelSlimCheckpoint,
    layer: int,
    my_hosted: list[int],
    device: torch.device,
    m_views: dict[str, torch.Tensor],
    nz_holders: list[torch.Tensor],
    mw13_cpu: torch.Tensor,
    mw2_cpu: torch.Tensor,
    tmp_contig: torch.Tensor | None,
    acl,
    *,
    gate_contig: bool,
    down_contig: bool,
) -> tuple[int, int, int, int, int, int]:
    import torch

    prefix = f"model.layers.{layer}"
    names: list[str] = []
    for e_idx in my_hosted:
        ep = f"{prefix}.mlp.experts.{e_idx}"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            names.extend(
                [
                    f"{ep}.{proj}.weight",
                    f"{ep}.{proj}.weight_scale",
                    f"{ep}.{proj}.weight_offset",
                    f"{ep}.{proj}.scale_bias",
                ]
            )
    loaded = ckpt.shard_tensors(names)
    ms = len(my_hosted)
    pfx = f"l{layer}"
    mr1 = torch.empty((ms, 4096), dtype=torch.float32)
    mr2 = torch.empty((ms, 4096), dtype=torch.float32)
    mr4 = torch.empty((ms, HIDDEN), dtype=torch.float32)
    mr5 = torch.empty((ms, HIDDEN), dtype=torch.float32)
    for mi, e_idx in enumerate(my_hosted):
        ep = f"{prefix}.mlp.experts.{e_idx}"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            _check_zero_offset(loaded, f"{ep}.{proj}.weight_offset")
        mw13_cpu[mi, :, :1024].copy_(loaded[f"{ep}.gate_proj.weight"].t())
        mw13_cpu[mi, :, 1024:].copy_(loaded[f"{ep}.up_proj.weight"].t())
        mw2_cpu[mi].copy_(loaded[f"{ep}.down_proj.weight"].t())
        mr1[mi, :2048].copy_(loaded[f"{ep}.gate_proj.scale_bias"].float().reshape(-1))
        mr1[mi, 2048:].copy_(loaded[f"{ep}.up_proj.scale_bias"].float().reshape(-1))
        mr2[mi, :2048].copy_(loaded[f"{ep}.gate_proj.weight_scale"].float().reshape(-1))
        mr2[mi, 2048:].copy_(loaded[f"{ep}.up_proj.weight_scale"].float().reshape(-1))
        mr4[mi].copy_(loaded[f"{ep}.down_proj.scale_bias"].float().sum(dim=1))
        mr5[mi].copy_(loaded[f"{ep}.down_proj.weight_scale"].float().reshape(-1))
    del loaded
    m_views[f"{pfx}.m_r1"].copy_(mr1)
    m_views[f"{pfx}.m_r2"].copy_(mr2)
    m_views[f"{pfx}.m_r4"].copy_(mr4)
    m_views[f"{pfx}.m_r5"].copy_(mr5)
    del mr1, mr2, mr4, mr5
    mr0_nz = _to_nz(mw13_cpu[:ms], device).view(torch.int32)
    if gate_contig and device.type == "npu":
        _apply_r4_routed_contig(mr0_nz, tmp_contig, 12582912, (16, 4, 4, 49152), acl)
    mr3_nz = _to_nz(mw2_cpu[:ms], device).view(torch.int32)
    if down_contig and device.type == "npu":
        _apply_r4_routed_contig(mr3_nz, tmp_contig, 6291456, (24, 4, 2, 32768), acl)
    nz_holders.extend((mr0_nz, mr3_nz))
    return (
        int(mr0_nz.data_ptr()),
        int(m_views[f"{pfx}.m_r1"].data_ptr()),
        int(m_views[f"{pfx}.m_r2"].data_ptr()),
        int(mr3_nz.data_ptr()),
        int(m_views[f"{pfx}.m_r4"].data_ptr()),
        int(m_views[f"{pfx}.m_r5"].data_ptr()),
    )


def _pack_layer_mirror_meta(
    peer_cpu: bytearray,
    layer: int,
    cands: list[int],
    masks: list[int],
    mw_ptrs: tuple[int, int, int, int, int, int],
    my_hosted: list[int],
    *,
    mirror_64: bool,
) -> None:
    off = layer * 640
    n_cands = len(cands)
    n_mslots = len(my_hosted)
    for idx in range(min(32, n_cands)):
        peer_cpu[off + idx] = cands[idx]
        peer_cpu[off + 32 + idx] = masks[idx]
    struct.pack_into("<6Q", peer_cpu, off + 64, *mw_ptrs)
    for ms, ge in enumerate(my_hosted[:8]):
        peer_cpu[off + 112 + ms] = ge
    ext_flag = (
        (1 << 17) if (mirror_64 and n_cands > 32) else ((1 << 16) if n_cands > 32 else 0)
    )
    hdr = n_cands | (n_mslots << 8) | ext_flag
    struct.pack_into("<Q", peer_cpu, off + 120, hdr)
    mask_ext_off = 224 if mirror_64 else 208
    for idx in range(32, n_cands):
        peer_cpu[off + 192 + (idx - 32)] = cands[idx]
        peer_cpu[off + mask_ext_off + (idx - 32)] = masks[idx]


def _reserve_model_nd_slots(
    builder: _NdSlabBuilder,
    local_vocab: int,
    max_rope_positions: int,
    local_experts: int,
    world_size: int = 8,
    *,
    load_draft: bool,
) -> None:
    import torch

    heads = 64 // int(world_size)
    d_dim = 12288 // int(world_size)
    sh_dim = 2048 // int(world_size)
    col_dim = HIDDEN // int(world_size)

    builder.reserve("embed_w", (local_vocab, HIDDEN), torch.bfloat16)
    builder.reserve("norm_w", (HIDDEN,), torch.bfloat16)
    builder.reserve("lm_head_w", (local_vocab, HIDDEN), torch.bfloat16)
    builder.reserve("cos_cache", (max_rope_positions, 64), torch.bfloat16)
    builder.reserve("sin_cache", (max_rope_positions, 64), torch.bfloat16)
    builder.reserve("q_hadamard", (128, 128), torch.bfloat16)
    builder.reserve("q_hadamard_zero", (128, 128), torch.bfloat16)

    layer_limit = TOTAL_LAYERS if load_draft else MAIN_LAYERS
    for layer in range(layer_limit):
        pfx = f"l{layer}"
        builder.reserve(f"{pfx}.input_gamma", (HIDDEN,), torch.bfloat16)
        builder.reserve(f"{pfx}.w_uk_t", (heads, 192, LATENT), torch.bfloat16)
        builder.reserve(f"{pfx}.s_dq", (1, 2048), torch.float32)
        builder.reserve(f"{pfx}.s_uq_qr", (1, heads * 256), torch.float32)
        builder.reserve(f"{pfx}.s_dkv_kr", (1, 576), torch.float32)
        builder.reserve(f"{pfx}.q_a_norm", (2048,), torch.bfloat16)
        builder.reserve(f"{pfx}.kv_a_norm", (LATENT,), torch.bfloat16)
        builder.reserve(f"{pfx}.tail_uv", (16, LATENT, (heads * VALUE) // 16), torch.bfloat16)
        builder.reserve(f"{pfx}.tail_o", (24, heads * VALUE, VALUE), torch.int8)
        builder.reserve(f"{pfx}.tail_s", (HIDDEN,), torch.float32)
        if layer in FULL_INDEX_LAYERS:
            builder.reserve(f"{pfx}.front_q", (16, 2048, 256), torch.int8)
            builder.reserve(f"{pfx}.front_qs", (4096,), torch.bfloat16)
            builder.reserve(f"{pfx}.front_k", (5, HIDDEN, 32), torch.bfloat16)
            builder.reserve(f"{pfx}.front_knw", (128,), torch.float32)
            builder.reserve(f"{pfx}.front_knb", (128,), torch.float32)
        if layer < 3:
            builder.reserve(f"{pfx}.post_gamma", (HIDDEN,), torch.bfloat16)
            gu_n32 = (2 * d_dim) // (24 * 32)
            builder.reserve(f"{pfx}.dense_p_gu", (24, 6, 2, gu_n32, 512, 32), torch.int8)
            builder.reserve(f"{pfx}.dense_s_gu", (2 * d_dim,), torch.float32)
            builder.reserve(f"{pfx}.dense_p_dn", (24, d_dim // 256, 2, 8, 128, 32), torch.int8)
            builder.reserve(f"{pfx}.dense_s_dn", (HIDDEN,), torch.float32)
        else:
            builder.reserve(f"{pfx}.post_gamma", (HIDDEN,), torch.bfloat16)
            builder.reserve(f"{pfx}.router_w", (HIDDEN, 256), torch.bfloat16)
            builder.reserve(f"{pfx}.router_b", (256,), torch.float32)
            builder.reserve(f"{pfx}.sh_gu", (24, 1024, sh_dim // 2), torch.int8)
            builder.reserve(f"{pfx}.sh_s_gu", (2 * sh_dim,), torch.float32)
            builder.reserve(f"{pfx}.sh_dn", (24, sh_dim, VALUE), torch.int8)
            builder.reserve(f"{pfx}.sh_s_dn", (HIDDEN,), torch.float32)
            builder.reserve(f"{pfx}.r1", (local_experts, 4096), torch.float32)
            builder.reserve(f"{pfx}.r2", (local_experts, 4096), torch.float32)
            builder.reserve(f"{pfx}.r4", (local_experts, HIDDEN), torch.float32)
            builder.reserve(f"{pfx}.r5", (local_experts, HIDDEN), torch.float32)

    if load_draft:
        builder.reserve("l78.head_w_shard", (local_vocab, HIDDEN), torch.bfloat16)
        builder.reserve("l78.rot_w_shard", (HIDDEN, col_dim), torch.bfloat16)
        builder.reserve("l78.w_eh_shard", (2 * HIDDEN, col_dim), torch.bfloat16)
        builder.reserve("l78.enorm_w", (HIDDEN,), torch.bfloat16)
        builder.reserve("l78.hnorm_w", (HIDDEN,), torch.bfloat16)
        builder.reserve("l78.shared_head_norm_w", (HIDDEN,), torch.bfloat16)


def build_model_arena(
    cfg: RuntimeConfig,
    libraries: Libraries | None = None,
    *,
    load_draft: bool = True,
) -> ModelArena:
    import torch

    rank = int(cfg.rank)
    world_size = int(cfg.world_size)
    local_experts = int(cfg.local_experts)
    device = torch.device(cfg.device)
    if device.type == "npu":
        torch.npu.set_device(device)
        torch.npu.config.allow_internal_format = True

    flags = (
        libraries.tree_flags()
        if libraries is not None
        else (
            "-DK11040_F0_NZ=1",
            "-DK11040_TAIL_NZ=1",
            "-DK11088_GATE_CONTIG=1",
            "-DK11150_SHARED_W8_NZ=1",
            "-DK11220_DOWN_CONTIG_256=1",
            "-DK112726_BRIDGE_PRELOAD_GATE=1",
        )
    )
    flag_set = set(flags)
    f0_nz = "-DK11040_F0_NZ=1" in flag_set
    tail_nz = "-DK11040_TAIL_NZ=1" in flag_set
    shared_nz = "-DK11150_SHARED_W8_NZ=1" in flag_set
    gate_contig = "-DK11088_GATE_CONTIG=1" in flag_set
    down_contig = "-DK11220_DOWN_CONTIG_256=1" in flag_set
    preload_gate = "-DK112726_BRIDGE_PRELOAD_GATE=1" in flag_set
    mirror_v4 = "-DK11220_FAST_MIRROR_V4=1" in flag_set
    mirror_p1_v3 = "-DK11220_FAST_P1_V3=1" in flag_set
    mirror_64 = "-DK112600_MIRROR_64=1" in flag_set
    ckpt = ModelSlimCheckpoint(cfg.model_dir)
    mirror_ok = (
        mirror_v4 and (mirror_p1_v3 or mirror_64) and world_size == 8 and local_experts == 32
    )
    mirror_plan = (
        _plan_moe_mirrors(
            ckpt,
            rank,
            world_size,
            local_experts,
            mirror_64=mirror_64,
            total_budget=(468 if mirror_64 else 450),
        )
        if mirror_ok
        else {}
    )

    local_vocab = VOCAB_SIZE // world_size
    vocab_slice = slice(rank * local_vocab, (rank + 1) * local_vocab)
    builder = _NdSlabBuilder(align=512)
    _reserve_model_nd_slots(
        builder,
        local_vocab,
        cfg.max_rope_positions,
        local_experts,
        world_size=world_size,
        load_draft=load_draft,
    )
    nd_raw, nd_arena, nd_views = builder.allocate(device)
    nz_holders: list[torch.Tensor] = []

    cos_dev, sin_dev = build_rope_caches(cfg.max_rope_positions, device=device)
    nd_views["cos_cache"].copy_(cos_dev)
    nd_views["sin_cache"].copy_(sin_dev)
    del cos_dev, sin_dev

    nd_views["q_hadamard"].copy_(hadamard_128(device=device))
    nd_views["q_hadamard_zero"].zero_()
    q_hadamard = nd_views["q_hadamard"]
    q_hadamard_zero = nd_views["q_hadamard_zero"]

    top_tensors = ckpt.shard_tensors(
        ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"],
        {
            "model.embed_tokens.weight": (vocab_slice, slice(None)),
            "lm_head.weight": (vocab_slice, slice(None)),
        },
    )
    nd_views["embed_w"].copy_(top_tensors["model.embed_tokens.weight"])
    nd_views["norm_w"].copy_(top_tensors["model.norm.weight"])
    nd_views["lm_head_w"].copy_(top_tensors["lm_head.weight"])
    del top_tensors

    acl = ctypes.CDLL("libascendcl.so") if device.type == "npu" else None
    tmp_contig = (
        torch.empty((2 * 12582912,), dtype=torch.uint8, device=device)
        if device.type == "npu"
        else None
    )
    w13_cpu = torch.empty((local_experts, HIDDEN, 2048), dtype=torch.int8)
    w2_cpu = torch.empty((local_experts, 2048, 3072), dtype=torch.int8)
    m_views: dict[str, torch.Tensor] = {}
    mw13_cpu = None
    mw2_cpu = None
    if mirror_plan:
        m_builder = _NdSlabBuilder(align=512)
        max_ms = max(len(v[2]) for v in mirror_plan.values())
        for ml, (_, _, my_hosted) in mirror_plan.items():
            mpfx = f"l{ml}"
            ms_cnt = len(my_hosted)
            m_builder.reserve(f"{mpfx}.m_r1", (ms_cnt, 4096), torch.float32)
            m_builder.reserve(f"{mpfx}.m_r2", (ms_cnt, 4096), torch.float32)
            m_builder.reserve(f"{mpfx}.m_r4", (ms_cnt, HIDDEN), torch.float32)
            m_builder.reserve(f"{mpfx}.m_r5", (ms_cnt, HIDDEN), torch.float32)
        m_nd_raw, _, m_views = m_builder.allocate(device)
        nz_holders.append(m_nd_raw)
        mw13_cpu = torch.empty((max_ms, HIDDEN, 2048), dtype=torch.int8)
        mw2_cpu = torch.empty((max_ms, 2048, 3072), dtype=torch.int8)
    mirror_meta_specs: dict[int, tuple[list[int], list[int], list[int], tuple]] = {}

    shared_topk = torch.zeros((32, 2048), dtype=torch.int32, device=device)
    topk_bufs = [shared_topk]
    scalars = (0.0625, RMS_NORM_EPS, RMS_NORM_EPS, RMS_NORM_EPS, RMS_NORM_EPS)
    dense_temp: dict[int, tuple] = {}
    layers_out: list[LayerWeights] = []

    for layer in range(MAIN_LAYERS):
        main, front, tail, input_gamma = _load_attention_weights(
            ckpt,
            layer,
            rank,
            world_size,
            device,
            nd_views,
            nz_holders,
            q_hadamard,
            q_hadamard_zero,
            f0_nz=f0_nz,
            tail_nz=tail_nz,
        )
        mode = (
            "int8"
            if layer in INT8_INDEX_LAYERS
            else ("bf16" if layer in FULL_INDEX_LAYERS else "shared")
        )
        if layer < 3:
            prefix = f"model.layers.{layer}"
            d_dim = 12288 // world_size
            d_out = slice(rank * d_dim, (rank + 1) * d_dim)
            d_loaded = ckpt.shard_tensors(
                [
                    prefix + ".post_attention_layernorm.weight",
                    prefix + ".mlp.gate_proj.weight",
                    prefix + ".mlp.gate_proj.weight_scale",
                    prefix + ".mlp.gate_proj.weight_offset",
                    prefix + ".mlp.up_proj.weight",
                    prefix + ".mlp.up_proj.weight_scale",
                    prefix + ".mlp.up_proj.weight_offset",
                    prefix + ".mlp.down_proj.weight",
                    prefix + ".mlp.down_proj.weight_scale",
                    prefix + ".mlp.down_proj.weight_offset",
                ],
                {
                    prefix + ".mlp.gate_proj.weight": (d_out, slice(None)),
                    prefix + ".mlp.gate_proj.weight_scale": d_out,
                    prefix + ".mlp.gate_proj.weight_offset": d_out,
                    prefix + ".mlp.up_proj.weight": (d_out, slice(None)),
                    prefix + ".mlp.up_proj.weight_scale": d_out,
                    prefix + ".mlp.up_proj.weight_offset": d_out,
                    prefix + ".mlp.down_proj.weight": (slice(None), d_out),
                },
            )
            for off_k in (
                prefix + ".mlp.gate_proj.weight_offset",
                prefix + ".mlp.up_proj.weight_offset",
                prefix + ".mlp.down_proj.weight_offset",
            ):
                _check_zero_offset(d_loaded, off_k)
            nd_views[f"l{layer}.post_gamma"].copy_(
                d_loaded[prefix + ".post_attention_layernorm.weight"]
            )
            w_gu = (
                torch.cat(
                    (
                        d_loaded[prefix + ".mlp.gate_proj.weight"],
                        d_loaded[prefix + ".mlp.up_proj.weight"],
                    ),
                    dim=0,
                )
                .t()
                .contiguous()
            )
            s_gu = torch.cat(
                (
                    d_loaded[prefix + ".mlp.gate_proj.weight_scale"].reshape(-1),
                    d_loaded[prefix + ".mlp.up_proj.weight_scale"].reshape(-1),
                ),
                dim=0,
            )
            w_dn = d_loaded[prefix + ".mlp.down_proj.weight"].t().contiguous()
            s_dn = d_loaded[prefix + ".mlp.down_proj.weight_scale"].reshape(-1)
            p_gu, s_gu_f, p_dn, s_dn_f = _pack_dense_mlp_weights(
                w_gu, s_gu, w_dn, s_dn, world_size=world_size
            )
            nd_views[f"l{layer}.dense_p_gu"].copy_(p_gu)
            nd_views[f"l{layer}.dense_s_gu"].copy_(s_gu_f)
            nd_views[f"l{layer}.dense_p_dn"].copy_(p_dn)
            nd_views[f"l{layer}.dense_s_dn"].copy_(s_dn_f)
            dense_mlp = (
                nd_views[f"l{layer}.dense_p_gu"],
                nd_views[f"l{layer}.dense_s_gu"],
                nd_views[f"l{layer}.dense_p_dn"],
                nd_views[f"l{layer}.dense_s_dn"],
            )
            dense_temp[layer] = (
                main,
                front,
                tail,
                input_gamma,
                nd_views[f"l{layer}.post_gamma"],
                dense_mlp,
            )
            del d_loaded, w_gu, s_gu, w_dn, s_dn, p_gu, s_gu_f, p_dn, s_dn_f
            layers_out.append(None)
        else:
            post, shared, routed = _load_moe_body(
                ckpt,
                layer,
                rank,
                world_size,
                local_experts,
                device,
                nd_views,
                nz_holders,
                w13_cpu,
                w2_cpu,
                tmp_contig,
                acl,
                shared_nz=shared_nz,
                gate_contig=gate_contig,
                down_contig=down_contig,
                preload_gate=preload_gate,
            )
            if layer in mirror_plan:
                cands, masks, my_hosted = mirror_plan[layer]
                mw_ptrs = _load_moe_mirrors(
                    ckpt,
                    layer,
                    my_hosted,
                    device,
                    m_views,
                    nz_holders,
                    mw13_cpu,
                    mw2_cpu,
                    tmp_contig,
                    acl,
                    gate_contig=gate_contig,
                    down_contig=down_contig,
                )
                mirror_meta_specs[layer] = (cands, masks, my_hosted, mw_ptrs)
            owners = (*main, *front, *tail, *routed, *shared, input_gamma, *post, shared_topk)
            layers_out.append(
                LayerWeights(
                    layer_idx=layer,
                    mode=mode,
                    main=main,
                    front=front,
                    tail=tail,
                    routed=routed,
                    shared=shared,
                    input_gamma=input_gamma,
                    post=post,
                    dense_mlp=None,
                    scalars=scalars,
                    topk=shared_topk,
                    owners=owners,
                )
            )

    moe0 = layers_out[3]
    for layer in range(3):
        main, front, tail, input_gamma, post_gamma, dense_mlp = dense_temp[layer]
        post = (post_gamma, moe0.post[1], moe0.post[2])
        mode = "int8" if layer in INT8_INDEX_LAYERS else "bf16"
        owners = (
            *main,
            *front,
            *tail,
            *moe0.routed,
            *moe0.shared,
            input_gamma,
            *post,
            shared_topk,
            *dense_mlp,
        )
        layers_out[layer] = LayerWeights(
            layer_idx=layer,
            mode=mode,
            main=main,
            front=front,
            tail=tail,
            routed=moe0.routed,
            shared=moe0.shared,
            input_gamma=input_gamma,
            post=post,
            dense_mlp=dense_mlp,
            scalars=scalars,
            topk=shared_topk,
            owners=owners,
        )

    peer_total_bytes = MAIN_LAYERS * 640
    peer_raw, peer_sub = _alloc_aligned(peer_total_bytes, 64, device, zero=True)
    peer_cpu = bytearray(peer_total_bytes)
    for li in range(3):
        p_gu, s_gu, p_dn, s_dn = layers_out[li].dense_mlp
        struct.pack_into(
            "<4Q",
            peer_cpu,
            li * 640 + 128,
            int(p_gu.data_ptr()),
            int(s_gu.data_ptr()),
            int(p_dn.data_ptr()),
            int(s_dn.data_ptr()),
        )
    for li, (cands, masks, my_hosted, mw_ptrs) in mirror_meta_specs.items():
        _pack_layer_mirror_meta(
            peer_cpu, li, cands, masks, mw_ptrs, my_hosted, mirror_64=mirror_64
        )
    peer_sub.copy_(torch.frombuffer(peer_cpu, dtype=torch.uint8))
    peer_weights_ptr = int(peer_sub.data_ptr())

    tree_mask_raw, tree_mask_u8 = _alloc_aligned(128, 64, device, zero=True)
    tree_mask_i32 = tree_mask_u8.view(torch.int32)

    draft_weights: DraftWeights | None = None
    if load_draft:
        from .dspine.calibrated_l78 import TpCalibratedL78Weights

        l78_main, l78_front, l78_tail, l78_in_gamma = _load_attention_weights(
            ckpt,
            DRAFT_LAYER,
            rank,
            world_size,
            device,
            nd_views,
            nz_holders,
            q_hadamard,
            q_hadamard_zero,
            f0_nz=True,
            tail_nz=True,
        )
        l78_post, l78_shared, l78_routed = _load_moe_body(
            ckpt,
            DRAFT_LAYER,
            rank,
            world_size,
            local_experts,
            device,
            nd_views,
            nz_holders,
            w13_cpu,
            w2_cpu,
            tmp_contig,
            acl,
            shared_nz=True,
            gate_contig=True,
            down_contig=True,
            preload_gate=True,
        )
        p78 = f"model.layers.{DRAFT_LAYER}"
        col_dim = HIDDEN // world_size
        col_slice = slice(rank * col_dim, (rank + 1) * col_dim)
        d_extra = ckpt.shard_tensors(
            [
                p78 + ".enorm.weight",
                p78 + ".hnorm.weight",
                p78 + ".eh_proj.weight",
                p78 + ".shared_head.norm.weight",
                p78 + ".shared_head.head.weight",
            ],
            {
                p78 + ".eh_proj.weight": (col_slice, slice(None)),
                p78 + ".shared_head.head.weight": (vocab_slice, slice(None)),
            },
        )
        nd_views["l78.head_w_shard"].copy_(d_extra[p78 + ".shared_head.head.weight"])
        nd_views["l78.w_eh_shard"].copy_(
            d_extra[p78 + ".eh_proj.weight"].t().contiguous()
        )
        nd_views["l78.enorm_w"].copy_(d_extra[p78 + ".enorm.weight"])
        nd_views["l78.hnorm_w"].copy_(d_extra[p78 + ".hnorm.weight"])
        nd_views["l78.shared_head_norm_w"].copy_(d_extra[p78 + ".shared_head.norm.weight"])
        rot_src = cfg.rot_path if cfg.rot_path is not None else cfg.model_dir
        nd_views["l78.rot_w_shard"].copy_(load_rot_shard(rot_src, rank, world_size))
        del d_extra

        cal_wt = TpCalibratedL78Weights.from_file(
            cfg.cal_l78_path, rank=rank, dev=device, world_size=world_size
        )
        head_q8_nz, head_scale = cal_wt.quantize_head_q8_nz(
            nd_views["l78.head_w_shard"], valid_vocab=local_vocab, inv_temp=cfg.inv_temp
        )
        nz_holders.append(head_q8_nz)
        if cal_wt.cal_uq_qr_q8_nz is not None:
            nz_holders.append(cal_wt.cal_uq_qr_q8_nz)
        routed_folded = tuple(cal_wt.fold_routed_weights(list(l78_routed)))
        draft_weights = DraftWeights(
            cal_wt=cal_wt,
            head_q8_nz=head_q8_nz,
            head_scale=head_scale,
            head_w_shard=nd_views["l78.head_w_shard"],
            input_gamma=l78_in_gamma,
            main=l78_main,
            front=l78_front,
            tail=l78_tail,
            post=l78_post,
            routed=l78_routed,
            routed_folded=routed_folded,
            shared=l78_shared,
            scalars=scalars,
            rot_w_shard=nd_views["l78.rot_w_shard"],
            w_eh_shard=nd_views["l78.w_eh_shard"],
            enorm_w=nd_views["l78.enorm_w"],
            hnorm_w=nd_views["l78.hnorm_w"],
            shared_head_norm_w=nd_views["l78.shared_head_norm_w"],
        )

    del w13_cpu, w2_cpu, mw13_cpu, mw2_cpu, tmp_contig
    gc.collect()
    if device.type == "npu":
        torch.npu.synchronize(device)
        torch.npu.empty_cache()

    return ModelArena(
        device=device,
        rank=rank,
        world_size=world_size,
        nd_raw=nd_raw,
        nd_arena=nd_arena,
        nz_holders=tuple(nz_holders),
        embed_w=nd_views["embed_w"],
        norm_w=nd_views["norm_w"],
        lm_head_w=nd_views["lm_head_w"],
        cos_cache=nd_views["cos_cache"],
        sin_cache=nd_views["sin_cache"],
        layers=tuple(layers_out),
        draft=draft_weights,
        peer_weights_buf=peer_raw,
        peer_weights_ptr=peer_weights_ptr,
        tree_mask_raw=tree_mask_raw,
        tree_mask_i32=tree_mask_i32,
        topk_bufs=tuple(topk_bufs),
    )


def _check_scalar_eps() -> float:
    return K_NORM_EPS
