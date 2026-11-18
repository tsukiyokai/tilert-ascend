from contextlib import contextmanager
import copy
import ctypes
import gc
import hashlib
import json
from pathlib import Path
import shutil
import struct
import time
from types import MethodType

import torch

from .build_dspine_ops import build_dspine_ops, default_build_dir

try:
    import torch_npu
except ImportError:
    torch_npu = None


def _load_ops_cdll(so_path: Path, rank: int) -> ctypes.CDLL:
    sha = hashlib.sha256(so_path.read_bytes()).hexdigest()[:10]
    ver_so = so_path.with_name(f"tilert_dspine_ops_{sha}.so")
    if rank == 0 and not ver_so.exists():
        shutil.copy2(so_path, ver_so)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    lib = ctypes.CDLL(str(ver_so), ctypes.RTLD_LOCAL)

    lib.tilert_get_flushed_stream.argtypes = []
    lib.tilert_get_flushed_stream.restype = ctypes.c_void_p

    lib.tilert_kv_compact_launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int32,
    ]
    lib.tilert_kv_compact_launch.restype = ctypes.c_int

    lib.tilert_prolog_pack_launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.tilert_prolog_pack_launch.restype = ctypes.c_int

    lib.tilert_dual_rmsnorm_slice_launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
    ]
    lib.tilert_dual_rmsnorm_slice_launch.restype = ctypes.c_int

    lib.tilert_logits_topk_lse_bf16_launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.tilert_logits_topk_lse_bf16_launch.restype = ctypes.c_int

    lib.tilert_dspine_prolog_pack_launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.tilert_dspine_prolog_pack_launch.restype = ctypes.c_int

    lib.tilert_dspine_prep_commit_seed_v3_launch.argtypes = [
        ctypes.c_void_p,
        *([ctypes.c_uint64] * 16),
        ctypes.c_int32,
    ]
    lib.tilert_dspine_prep_commit_seed_v3_launch.restype = ctypes.c_int

    lib.tilert_dspine_beam_step_v3_launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        *([ctypes.c_int32] * 8),
        ctypes.c_float,
        ctypes.c_float,
        *([ctypes.c_uint64] * 26),
    ]
    lib.tilert_dspine_beam_step_v3_launch.restype = ctypes.c_int

    lib.tilert_dspine_build_tree_v3_launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        *([ctypes.c_int32] * 6),
        ctypes.c_float,
        ctypes.c_float,
        *([ctypes.c_uint64] * 21),
    ]
    lib.tilert_dspine_build_tree_v3_launch.restype = ctypes.c_int
    return lib


def load_dspine_ops(rank: int, out_dir: str | Path | None = None) -> tuple[Path, ctypes.CDLL]:
    out = default_build_dir() if out_dir is None else Path(out_dir)
    so_path = out / "tilert_dspine_ops.so"
    if not so_path.exists():
        if rank == 0:
            build_dspine_ops(out)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    return so_path, _load_ops_cdll(so_path, rank)


@contextmanager
def exact_rows(target):
    original = target._metadata

    def exact_meta(self, width_arg, *, graphed=False, depths=None, host_values=None):
        ea = getattr(self, "_exact_active", None)
        res = original(width_arg, graphed=graphed, depths=depths, host_values=host_values)
        if ea is not None and graphed and self._hoisted.get(width_arg) is not None:
            self._active_rows = ea
            if self._dynamic_mc2_mask and self.graph is not None and self.graph.capture_count:
                if self._last_mc2_active.get(width_arg, width_arg) != ea:
                    from vllm_ascend.ascend_forward_context import (
                        _hoisted_mc2_masks,
                        get_mc2_mask,
                    )

                    mask = (
                        _hoisted_mc2_masks.get((width_arg, width_arg))
                        if _hoisted_mc2_masks is not None
                        else None
                    )
                    if mask is None:
                        reserved = get_mc2_mask()
                        if reserved is not None:
                            mask = reserved[:width_arg]
                    if mask is not None:
                        mask.copy_(self._mc2_host_masks[width_arg][ea], non_blocking=True)
                    self._last_mc2_active[width_arg] = ea
            if self._dynamic_sfa_len:
                s_idx = self._sfa_len_idx % len(self._sfa_len_slots)
                self._sfa_len_idx += 1
                s_len = self._sfa_len_slots[s_idx]
                s_len[0, 0] = ea
                s_len[1, 0] = self.position + ea
                s_len[0, 1] = width_arg
                s_len[1, 1] = self.position + width_arg
                self._dyn_lens_dev[width_arg].copy_(s_len, non_blocking=True)
        return res

    target._metadata = MethodType(exact_meta, target)
    try:
        yield
    finally:
        target._exact_active = None
        target._metadata = original


def _bind_draft_klib(so_path: Path, ver: int):
    lib = ctypes.CDLL(str(so_path), ctypes.RTLD_LOCAL)
    fn_m = getattr(lib, f"layer_tp_tree_set_mask_ptr_K{ver}")
    fn_m.argtypes = [ctypes.c_uint64]
    fn_m.restype = None
    fn_i = getattr(lib, f"layer_tp_tree_indexed_launch_K{ver}")
    fn_s = getattr(lib, f"layer_tp_tree_shared_launch_K{ver}")
    fn_kvc = getattr(lib, f"layer_tp_tree_kv_commit_launch_K{ver}")
    for fn in (fn_i, fn_s, fn_kvc):
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        fn.restype = ctypes.c_int
    fn_m(0)
    return lib, fn_m, fn_i, fn_s, fn_kvc


class _DraftLayerWeights:
    def __init__(self, layer, mode: str, device: torch.device, *, f0_nz: bool = False):
        from .. import attention_native as an
        from ..routed_shared import _shared_weights
        from ..whole_layer import HIDDEN, unpack_scale

        self.attention = layer.self_attn.mla_attn.mla_attn
        self.impl = self.attention.impl
        self.mode = mode
        with torch.npu.device(device), torch.inference_mode():
            self.main, self.front, self.tail = an._pack_weights(self.impl, mode)
            if f0_nz and self.front:
                f0 = self.front[0]
                f0.copy_(f0.view(16, 2048, 8, 32).permute(0, 2, 1, 3).contiguous().view_as(f0))
            torch.npu.synchronize()
        self.caches = an._cache_tensors(self.attention, mode)
        self.device = self.caches[0].device
        runner = layer.mlp.experts
        routed = runner.routed_experts
        moe = runner.moe_config
        local_experts = moe.num_local_experts
        inter = moe.intermediate_size_per_partition
        gu_cols = 2 * inter
        scales = [
            an._nd(unpack_scale(getattr(routed, name), columns, local_experts).to(device))
            for name, columns in (("w13_weight_scale", gu_cols), ("w2_weight_scale", HIDDEN))
        ]
        self.routed = [
            routed.w13_weight,
            routed.w13_scale_bias,
            scales[0],
            routed.w2_weight,
            routed.w2_scale_bias,
            scales[1],
        ]
        self.shared = list(_shared_weights(runner.ascend_shared_experts))
        self.input_gamma = an._nd(layer.input_layernorm.weight)
        self.post = [
            an._nd(layer.post_attention_layernorm.weight),
            an._nd(an._nd(layer.mlp.gate.weight).t()),
            layer.mlp.gate.e_score_correction_bias.detach(),
        ]
        self.scalars = (
            self.impl.scale,
            self.impl.q_a_layernorm.variance_epsilon,
            self.impl.kv_a_layernorm.variance_epsilon,
            layer.input_layernorm.variance_epsilon,
            layer.post_attention_layernorm.variance_epsilon,
        )


def capture_shared_target_for_width(
    target,
    draft,
    width: int,
    case_prefix_main,
    case_mtp,
    golden_tokens: list[int],
    tree_mask_i32: torch.Tensor,
    channel_group,
) -> dict:
    from ..cann_graph import CannGraph
    from ..cann_model import rope_buffers
    from vllm_ascend.ascend_forward_context import _hoisted_mc2_masks, get_mc2_mask

    dev = target.device
    mgr = target._whole_layer
    mgr.set_tree_mask_ptr(int(tree_mask_i32.data_ptr()))
    tree_mask_i32.zero_()
    torch.npu.synchronize()

    target._hoisted.pop(width, None)
    target._last_mc2_active.pop(width, None)
    with torch.inference_mode():
        target._input_buffers[width][4].copy_(target._input_buffers[width][3].to(dev))
        if _hoisted_mc2_masks is not None and (width, width) in _hoisted_mc2_masks:
            _hoisted_mc2_masks[(width, width)].fill_(True)
    torch.ops.tilert.set_active_tokens_ptr(target._dyn_lens_dev[width], True, True)

    target.inject(case_prefix_main)
    draft.inject(case_mtp)
    torch.distributed.barrier(group=channel_group)

    if target.graph is not None:
        target.graph.reset()
        target.graph = None

    with torch.npu.device(dev), torch.inference_mode(), exact_rows(target):
        target.graph = CannGraph(target, width)
        for _ in range(3):
            tree_mask_i32.zero_()
            st_w = target.verify(golden_tokens[:width], full_logits=False)
            torch.npu.synchronize()
            if st_w.token_ids[:width].cpu().tolist() != golden_tokens[1 : width + 1]:
                raise RuntimeError(f"Target capture parity mismatch at width={width}")
            target.truncate(8192)

    for rg in getattr(target, "_rope_graphs", {}).values():
        rg.graph.reset()
    target._rope_graphs.clear()

    hoisted_vals = target._hoisted[width][1]
    hoisted_rope_dsts = target._hoisted[width][2]
    rb = rope_buffers()
    all_rope_pairs = [(rb._cos_mla[:width], rb._sin_mla[:width])] + list(hoisted_rope_dsts)
    rdsts_cpu = torch.zeros((1024,), dtype=torch.int64)
    for idx_r, (rcos, rsin) in enumerate(all_rope_pairs):
        rdsts_cpu[2 * idx_r] = int(rcos.data_ptr())
        rdsts_cpu[2 * idx_r + 1] = int(rsin.data_ptr())
    rope_dsts_dev = rdsts_cpu.to(dev)

    dyn_w_t = target._dyn_lens_dev[width]
    dyn_kv_ptrs = {int(dyn_w_t[1, :1].data_ptr()), int(dyn_w_t[1, 1:2].data_ptr())}
    full_seq_ptrs = {}
    q_restore_map = {}
    for v in hoisted_vals:
        if int(v.seq_lens.data_ptr()) not in dyn_kv_ptrs:
            full_seq_ptrs[v.seq_lens.data_ptr()] = v.seq_lens
        if hasattr(v, "actual_seq_lengths_key"):
            if int(v.actual_seq_lengths_key.data_ptr()) not in dyn_kv_ptrs:
                full_seq_ptrs[v.actual_seq_lengths_key.data_ptr()] = v.actual_seq_lengths_key
        for q_attr in ("cum_query_lens", "actual_seq_lengths_query"):
            tq = getattr(v, q_attr, None)
            if isinstance(tq, torch.Tensor) and tq.device.type == "npu" and tq.dtype == torch.int32:
                q_restore_map[int(tq.data_ptr())] = tq

    full_seq_tensors = list(full_seq_ptrs.values())
    fsptrs_cpu = torch.zeros((16,), dtype=torch.int64)
    for i, sl in enumerate(full_seq_tensors):
        fsptrs_cpu[i] = int(sl.data_ptr())
    full_seq_ptrs_dev = fsptrs_cpu.to(dev)

    mc2_mask = _hoisted_mc2_masks.get((width, width)) if _hoisted_mc2_masks is not None else None
    if mc2_mask is None:
        res_m = get_mc2_mask()
        if res_m is not None:
            mc2_mask = res_m[:width]
    mc2_mask_ptr = int(mc2_mask.data_ptr()) if mc2_mask is not None else 0
    has_mc2_mask = 1 if mc2_mask is not None else 0

    q_restore_pairs = [
        (dyn_w_t, dyn_w_t.clone()),
        *((tq, tq.clone()) for tq in q_restore_map.values()),
    ]
    if mc2_mask is not None:
        q_restore_pairs.append((mc2_mask, mc2_mask.clone()))

    t_ids, t_pos, t_slots = target._input_buffers[width][:3]
    head_graph, target_tokens = target._head_graphs[width]
    return {
        "width": width,
        "graph_width": width,
        "target_aclgraph": target.graph._entry.aclgraph,
        "target_hidden": target.graph._entry.output,
        "head_graph": head_graph,
        "target_tokens": target_tokens,
        "t_input_ids": t_ids,
        "t_positions": t_pos,
        "t_slots": t_slots,
        "t_slots_i64": target.graph.slots_i64,
        "dyn_w": dyn_w_t,
        "full_seq_ptrs_dev": full_seq_ptrs_dev,
        "num_full_seq_ptrs": len(full_seq_tensors),
        "mc2_mask_ptr": mc2_mask_ptr,
        "has_mc2_mask": has_mc2_mask,
        "rope_dsts_dev": rope_dsts_dev,
        "num_rope_dsts": len(all_rope_pairs),
        "q_restore_pairs": q_restore_pairs,
        "tree_mask_i32": tree_mask_i32,
    }


class _DraftEntry:
    pass


class CalibratedL78V3Runner:
    COMMIT_M = 16

    def __init__(
        self,
        target,
        draft,
        shared_tgt: dict,
        cal_wt,
        rank: int,
        channel_group,
        first_token: int,
        prefix_parent_hidden: torch.Tensor,
        b_scheds: list[list[int]],
        horizon_ctrl=None,
        inv_temp: float = 0.96,
        beta_margin: float | None = None,
        beta_depth: float | None = None,
        k_sfa: int = 1024,
        min_w_act: int = 10,
        t_base_ms: float = 47.0,
        c_w_ms: float = 0.72,
        p2_commit: bool = True,
        narrow_topk: bool = True,
        l78_build_dir: str | Path | None = None,
    ):
        from .calibrated_l78 import build_beam_specs

        self.target = target
        self.draft = draft
        self.st = shared_tgt
        self.w = int(shared_tgt["width"])
        self.wt = cal_wt
        self.rank = int(rank)
        self.dev = target.device
        self.channel_group = channel_group
        self.first_token = int(first_token)
        self.prefix_parent_hidden = prefix_parent_hidden
        self.l78_build_dir = l78_build_dir

        self.b_scheds = [tuple(int(x) for x in s) for s in b_scheds]
        if not 1 <= len(self.b_scheds) <= 3 or max(len(s) for s in self.b_scheds) > 14:
            raise ValueError("Unsupported DSpine beam schedule dimensions")
        self.b0 = int(self.b_scheds[0][0])
        self.max_b = max(max(s) for s in self.b_scheds)
        if not (
            1 <= self.b0 <= 4
            and 1 <= self.max_b <= 4
            and all(s[0] == self.b0 for s in self.b_scheds)
        ):
            raise ValueError("Beam schedules must share root beam width in [1, 4]")

        self.horizon_ctrl = horizon_ctrl
        self.inv_temp = float(inv_temp)
        self.beta_margin = float(cal_wt.beta_margin if beta_margin is None else beta_margin)
        self.beta_depth = float(cal_wt.beta_depth if beta_depth is None else beta_depth)
        self.p2_commit = bool(p2_commit)
        self.k_sfa = int(k_sfa) if self.p2_commit else 2048
        if self.k_sfa not in (512, 1024, 2048):
            raise ValueError(f"Unsupported k_sfa: {self.k_sfa}")
        self.min_w_act = int(min_w_act) if self.p2_commit else self.w
        self.t_base_ms = float(t_base_ms)
        self.c_w_ms = float(c_w_ms)
        self.narrow_topk = bool(narrow_topk) and self.p2_commit

        dev = self.dev
        self.sched_specs = []
        for sched in self.b_scheds:
            specs, depths = build_beam_specs(sched)
            dep_cpu = torch.zeros((256,), dtype=torch.int64)
            for i, dp in enumerate(depths):
                dep_cpu[i] = dp
            self.sched_specs.append((specs, len(depths), dep_cpu.to(dev)))

        self._init_runtime_buffers()

        head_cache = getattr(cal_wt, "_head_q8_cache", None)
        if head_cache is None:
            head_cache = {}
            cal_wt._head_q8_cache = head_cache
        t_key = round(self.inv_temp, 6)
        if t_key not in head_cache:
            head_cache[t_key] = cal_wt.quantize_head_q8_nz(
                self.head_w, self.head_valid_vocab, inv_temp=self.inv_temp
            )
        self.cal_head_w_q8_nz, self.cal_head_s8 = head_cache[t_key]

        self.seed_top_i = torch.zeros((8,), dtype=torch.int64, device=dev)
        self.seed_top_lp = torch.zeros((8,), dtype=torch.float32, device=dev)
        self.prefix_topk_buf = torch.zeros((2048,), dtype=torch.int32, device=dev)
        self.anc_slots_buf = torch.zeros((128,), dtype=torch.int32, device=dev)
        self.active_toks = torch.zeros((8,), dtype=torch.int64, device=dev)
        self.active_scores = torch.zeros((8,), dtype=torch.float32, device=dev)
        self.active_raw_lp = torch.zeros((8,), dtype=torch.float32, device=dev)
        self.active_margins = torch.zeros((8,), dtype=torch.float32, device=dev)
        self.active_pids = torch.zeros((8,), dtype=torch.int64, device=dev)
        self.active_recs = torch.zeros((self.max_b, 6144), dtype=torch.bfloat16, device=dev)

        self.rot_part = torch.zeros((self.max_b, 768), dtype=torch.bfloat16, device=dev)
        self.prolog_comm_in = torch.zeros((2 * self.max_b, 6144), dtype=torch.bfloat16, device=dev)
        self.prolog_comm_out = torch.zeros((2 * self.max_b, 6144), dtype=torch.bfloat16, device=dev)
        self.eh_in = torch.zeros((self.max_b, 1536), dtype=torch.bfloat16, device=dev)
        self.eh_comm_in = torch.zeros((self.max_b, 6144), dtype=torch.bfloat16, device=dev)

        self.pool_scores = torch.zeros((256,), dtype=torch.float32, device=dev)
        self.pool_raw_lp = torch.zeros((256,), dtype=torch.float32, device=dev)
        self.pool_tokens = torch.zeros((256,), dtype=torch.int64, device=dev)
        self.pool_parents = torch.zeros((256,), dtype=torch.int64, device=dev)

        self.tree_tokens = torch.zeros((self.w,), dtype=torch.int64, device=dev)
        self.tree_parents = torch.zeros((self.w,), dtype=torch.int64, device=dev)
        self.tree_depths = torch.zeros((self.w,), dtype=torch.int64, device=dev)

        self.draft_ctrl_cpu = torch.zeros((8,), dtype=torch.int64, pin_memory=True)
        self.draft_ctrl_dev = torch.zeros((8,), dtype=torch.int64, device=dev)
        self.compact_ctrl_cpu = torch.zeros((64,), dtype=torch.int64, pin_memory=True)
        self.compact_ctrl_dev = torch.zeros((64,), dtype=torch.int64, device=dev)

        cm = self.COMMIT_M
        self.commit_ttoks_pad32 = torch.zeros((32,), dtype=torch.int64, device=dev)
        self.commit_ctrl_cpu = torch.zeros((48,), dtype=torch.int64, pin_memory=True)
        self.commit_ctrl_dev = torch.zeros((48,), dtype=torch.int64, device=dev)
        self.commit_rot_shard = torch.zeros((cm, 768), dtype=torch.bfloat16, device=dev)
        self.commit_comm_in = torch.zeros((2 * cm, 6144), dtype=torch.bfloat16, device=dev)
        self.commit_comm_out = torch.zeros((2 * cm, 6144), dtype=torch.bfloat16, device=dev)
        self.commit_eh_in = torch.zeros((cm, 12288), dtype=torch.bfloat16, device=dev)
        self.commit_eh_shard = torch.zeros((cm, 768), dtype=torch.bfloat16, device=dev)
        self.commit_eh_comm = torch.zeros((cm, 6144), dtype=torch.bfloat16, device=dev)
        self.commit_hs_all = torch.zeros((cm + 4, 6144), dtype=torch.bfloat16, device=dev)
        self.kv_commit_topk = torch.zeros((cm, 1, 2048), dtype=torch.int32, device=dev)

        self.host_read_buf = torch.zeros((3 * self.w,), dtype=torch.int64, device=dev)

        all_widths = sorted({b for s in self.b_scheds for b in s})
        (
            self.cal_entries,
            self.entry_cal_idx_1,
            self.entry_kv_commit,
        ) = self._build_v3_entries(all_widths)
        self.sync_addr = int(self.entry_cal_idx_1.sync)
        with torch.inference_mode():
            for dst_q, ref_q in self.st.get("q_restore_pairs", ()):
                dst_q.copy_(ref_q)
        torch.ops.tilert.set_active_tokens_ptr(shared_tgt["dyn_w"], True, True)
        self._capture()

    def _init_runtime_buffers(self):
        from vllm.distributed import get_tp_group

        dev = self.dev
        draft = self.draft
        mtp_layer = draft.model.model.layers["78"]
        blk = mtp_layer.mtp_block
        mla_wrapper = blk.self_attn.mla_attn
        impl = mla_wrapper.mla_attn.impl
        self.mtp_layer = mtp_layer
        self.impl = impl
        tp_grp = get_tp_group().device_group

        def gather_col_shards(mod):
            w = mod.weight.data.contiguous()
            chunks = [torch.empty_like(w) for _ in range(8)]
            torch.distributed.all_gather(chunks, w, group=tp_grp)
            full = torch.cat(chunks, dim=0).contiguous()
            return tuple(c.contiguous() for c in chunks), full

        cache_rep = getattr(draft, "_dspine_rep_weights", None)
        if cache_rep is None:
            rot_shards, _ = gather_col_shards(draft.model.rot)
            _, eh_full = gather_col_shards(mtp_layer.eh_proj)
            cache_rep = (rot_shards, eh_full)
            draft._dspine_rep_weights = cache_rep
        rot_shards, eh_full = cache_rep

        self.rot_shard_t = rot_shards[self.rank].t().contiguous()
        self.w_eh_col_r = (
            eh_full[self.rank * 768 : (self.rank + 1) * 768, :].t().contiguous()
        )

        emb_mod = draft.model.model.embed_tokens
        self.emb_w = emb_mod.weight.contiguous()
        self.emb_vocab_start = int(emb_mod.shard_indices.org_vocab_start_index)
        self.emb_vocab_end = int(emb_mod.shard_indices.org_vocab_end_index)
        self.enorm_w = mtp_layer.enorm.weight.contiguous()
        self.hnorm_w = mtp_layer.hnorm.weight.contiguous()
        self.norm_eps = float(mtp_layer.enorm.variance_epsilon)

        lm_head = mtp_layer.shared_head.head
        self.head_w = lm_head.weight.contiguous()
        self.head_row_stride = int(self.head_w.shape[0])
        self.head_valid_vocab = (
            self.head_row_stride - int(lm_head.shard_indices.num_org_vocab_padding)
        )
        self.head_vocab_start = int(lm_head.shard_indices.org_vocab_start_index)

        self.gather_in = torch.zeros((8, 64), dtype=torch.float32, device=dev)
        self.gather_out = torch.zeros((8, 64), dtype=torch.float32, device=dev)
        self.logits_work = torch.zeros((1024,), dtype=torch.float32, device=dev)
        self.beam_work = torch.zeros((32,), dtype=torch.int64, device=dev)
        self.tree_work = torch.zeros((32,), dtype=torch.int64, device=dev)
        self.seed_rec_buf = torch.zeros((1, 6144), dtype=torch.bfloat16, device=dev)

        cache_tensors = [t for layer in self.target.cache for t in layer.values()]
        self.num_cache_tensors = len(cache_tensors)
        descs_cpu = torch.zeros((512,), dtype=torch.int64)
        for idx_t, t in enumerate(cache_tensors):
            row_bytes = t.stride(0) * t.element_size() if t.dim() > 1 else t.element_size()
            descs_cpu[2 * idx_t] = int(t.data_ptr())
            descs_cpu[2 * idx_t + 1] = int(row_bytes)
        self.descs_dev = descs_cpu.to(dev)

    def _ensure_l78_klibs(self):
        from .calibrated_l78 import build_draft_l78_k110419

        cached = getattr(self.wt, "_p2_klibs", None)
        if cached is not None:
            for h in cached:
                h[1](0)
            return cached
        if self.rank == 0:
            so_19, so_20 = build_draft_l78_k110419(out_dir=self.l78_build_dir)
            out_dir = so_19.parent
            so_15 = out_dir / "chain_K110415.so"
            for src_so in (so_15, so_19, so_20):
                sha = hashlib.sha256(src_so.read_bytes()).hexdigest()[:10]
                ver_so = src_so.with_name(f"{src_so.stem}_{sha}.so")
                if not ver_so.exists():
                    shutil.copy2(src_so, ver_so)
        torch.distributed.barrier(group=self.channel_group)
        so_19, so_20 = build_draft_l78_k110419(out_dir=self.l78_build_dir)
        out_dir = so_19.parent
        so_15 = out_dir / "chain_K110415.so"

        def ver_path(p):
            sha = hashlib.sha256(p.read_bytes()).hexdigest()[:10]
            return p.with_name(f"{p.stem}_{sha}.so")

        h15 = _bind_draft_klib(ver_path(so_15), 110415)
        h19 = _bind_draft_klib(ver_path(so_19), 110419)
        h20 = _bind_draft_klib(ver_path(so_20), 110420)
        self.wt._p2_klibs = (h15, h19, h20)
        return self.wt._p2_klibs

    def _build_v3_entries(self, widths: list[int]):
        from vllm.config import set_current_vllm_config
        from .. import attention_native as an
        from ..cann_model import CannModel
        from ..cann_mtp import INDEX_CACHE, MAIN_CACHE
        from ..whole_layer_abi import LocalRecords, Window

        h15, h19, h20 = self._ensure_l78_klibs()
        _, _, fn_i15, fn_s15, _ = h15
        _, _, _, fn_s19, fn_kvc19 = h19
        _, _, _, fn_s20, _ = h20
        if self.k_sfa == 2048:
            fn_s_rollout = fn_s15
        elif self.k_sfa == 1024:
            fn_s_rollout = fn_s19
        else:
            fn_s_rollout = fn_s20

        dev = self.dev
        draft = self.draft
        mgr = self.target._whole_layer

        bundle = getattr(self.wt, "_v3_shared_bundle", None)
        if bundle is None:
            with (
                torch.npu.device(dev),
                torch.inference_mode(),
                set_current_vllm_config(draft.config),
            ):
                old_pos = draft.position
                draft.position = 9300
                gmeta1 = CannModel._metadata(draft, 1, graphed=True)
                raw_mt = an._nd(gmeta1[MAIN_CACHE].block_table).clone()
                raw_it = an._nd(gmeta1[INDEX_CACHE].block_table).clone()
                draft.position = old_pos

            blk = self.mtp_layer.mtp_block
            idx_mode = an._validate_impl(self.impl)
            w_idx = _DraftLayerWeights(blk, idx_mode, dev, f0_nz=True)
            w_shr = _DraftLayerWeights(blk, "shared", dev, f0_nz=False)
            l78_win = getattr(draft, "_l78_win", None)
            if l78_win is None:
                l78_win = Window(mgr.tree, 32)
                draft._l78_win = l78_win

            with torch.npu.device(dev):
                cal_input_gamma = an._nd(self.wt.cal_input_gamma)
                cal_main = list(w_shr.main)
                if self.wt.cal_uq_qr_q8_nz is not None:
                    cal_main[1] = self.wt.cal_uq_qr_q8_nz
                    cal_main[5] = an._nd(self.wt.cal_uq_qr_scale)
                cal_main[2] = an._nd(self.wt.cal_w_uk_t)
                cal_main[7] = an._nd(self.wt.cal_q_a_norm)

                cal_t0 = an._nd(self.wt.cal_w_uv_packed.clone())
                cal_tail = list(w_shr.tail)
                cal_tail[0] = cal_t0
                if self.wt.cal_w_o_packed is not None:
                    cal_t1 = an._nd(self.wt.cal_w_o_packed.clone())
                    cal_tail[1] = cal_t1
                    cal_tail[2] = an._nd(self.wt.cal_w_o_scale)

                cal_post = [
                    an._nd(self.wt.cal_post_gamma),
                    an._nd(self.wt.cal_router_w_packed),
                    w_shr.post[2],
                ]
                cal_routed = self.wt.fold_routed_weights(w_shr.routed)
                cal_shared = list(w_shr.shared)
                if self.wt.cal_shared_packed is not None:
                    cal_shared = [an._nd(x) for x in self.wt.cal_shared_packed]
                weight_refs = (
                    w_idx, w_shr, l78_win,
                    cal_input_gamma, cal_main, cal_tail, cal_post, cal_routed, cal_shared,
                    raw_mt, raw_it,
                )

                def empty(shape, dtype):
                    return an._nd(torch.zeros(shape, dtype=dtype, device=dev))

                def make_entry(
                    m_rows: int, indexed: bool, fn_launch, use_cal: bool, custom_topk=None
                ):
                    em = _DraftEntry()
                    w_ref = w_idx if indexed else w_shr
                    em.weights = w_ref
                    em.m = m_rows
                    em.indexed = indexed
                    em.hs_in = empty((m_rows, 6144), torch.bfloat16)
                    em.zero_res = empty((m_rows, 6144), torch.bfloat16)
                    em.norm_outputs = [
                        empty((m_rows, 6144), torch.bfloat16),
                        empty((m_rows, 6144), torch.bfloat16),
                        empty((m_rows,), torch.float32),
                        empty((m_rows, 6144), torch.int8),
                        empty((m_rows, 1), torch.float32),
                    ]
                    em.sin_buf = empty((m_rows, 64), torch.bfloat16)
                    em.cos_buf = empty((m_rows, 64), torch.bfloat16)
                    em.slot_i64 = empty((m_rows,), torch.int64)
                    em.slot_i64.copy_(
                        torch.arange(9250, 9250 + m_rows, dtype=torch.int64, device=dev)
                    )
                    em.q_len = torch.full((1,), m_rows, dtype=torch.int32, device=dev)
                    em.kv_len = torch.zeros((1,), dtype=torch.int32, device=dev)
                    main_meta = [em.sin_buf, em.cos_buf, em.slot_i64, em.q_len, em.kv_len, raw_mt]
                    idx_meta = (
                        [em.sin_buf, em.cos_buf, em.slot_i64, em.q_len, em.kv_len, raw_it]
                        if indexed
                        else None
                    )
                    g = an.geometry_for(w_ref.mode, w_ref.caches, main_meta, idx_meta)
                    if not indexed:
                        (
                            em.scratch,
                            em.tiling,
                            blocks,
                            em.sync,
                        ) = torch.ops.tilert.shared_full_attention_prepare(
                            em.norm_outputs[3], g.pages, g.table_pages, 24
                        )
                    else:
                        (
                            em.scratch,
                            em.tiling,
                            _,
                            blocks,
                            em.sync,
                        ) = torch.ops.tilert.indexed_attention_prepare(
                            em.norm_outputs[3],
                            g.pages,
                            g.table_pages,
                            g.index_pages,
                            g.index_table_pages,
                            False,
                            24,
                        )
                    em.partial = empty((m_rows, 6144), torch.bfloat16)
                    em.bridge, em.body, sync_f = mgr.ffn.prepare(em.partial, 24, 1)
                    if blocks != 24 or em.sync != sync_f:
                        raise RuntimeError("Draft entry FFN workspace mismatch")
                    em.attention_sum = empty((m_rows, 6144), torch.bfloat16)
                    em.output = empty((m_rows, 6144), torch.bfloat16)

                    ig_use = cal_input_gamma if use_cal else w_ref.input_gamma
                    mn_use = cal_main if use_cal else w_ref.main
                    tl_use = cal_tail if use_cal else w_ref.tail
                    ps_use = cal_post if use_cal else w_ref.post
                    rt_use = cal_routed if use_cal else w_ref.routed
                    sh_use = cal_shared if use_cal else w_ref.shared

                    norm = [em.hs_in, em.zero_res, ig_use, *em.norm_outputs]
                    caches = [*w_ref.caches, *([None] * (4 - len(w_ref.caches)))]
                    addrs = [
                        *norm[3:4], *norm[6:8], *mn_use,
                        *(w_ref.front or [None] * 6), *tl_use, None,
                        *main_meta, *(idx_meta or [None] * 6), *caches,
                        em.scratch, em.tiling, em.partial,
                    ]
                    dims = (
                        m_rows, g.pages, g.table_pages,
                        g.index_pages or g.pages, g.index_table_pages or g.table_pages,
                        24, 0, int(bool(w_ref.front) and w_ref.front[1].dtype == torch.bfloat16), 0,
                    )
                    groups = (norm, addrs, ps_use, rt_use, em.bridge, em.body, sh_use)
                    shared_topk = (
                        custom_topk
                        if custom_topk is not None
                        else draft.indices_buffer[:m_rows].view(m_rows, 1, 2048)
                    )
                    em.local = LocalRecords(
                        mgr.metadata_library,
                        *groups,
                        shared_topk,
                        dims,
                        w_ref.scalars,
                        em.sync,
                        indexed,
                    )
                    comm_raw = struct.pack(
                        "<QQQQqii",
                        l78_win.address, l78_win.bytes,
                        em.attention_sum.data_ptr(), em.output.data_ptr(),
                        l78_win.capacity, l78_win.rank, l78_win.ranks,
                    )
                    em.comm_buf = ctypes.create_string_buffer(comm_raw, 48)
                    em.fn = fn_launch
                    return em

                e_cal_idx1 = make_entry(1, True, fn_i15, True)
                torch.npu.synchronize()
                bundle = ({}, e_cal_idx1, weight_refs, make_entry)
                self.wt._v3_shared_bundle = bundle

        base_shr, e_cal_idx1, weight_refs, make_entry = bundle
        self._cal_weight_refs = weight_refs
        out_shr = {}
        with torch.npu.device(dev):
            for m_rows in widths:
                if m_rows not in base_shr:
                    base_shr[m_rows] = make_entry(m_rows, False, fn_s_rollout, True)
                em_cl = copy.copy(base_shr[m_rows])
                em_cl.fn = fn_s_rollout
                out_shr[m_rows] = em_cl
            e_kvc = make_entry(
                self.COMMIT_M, True, fn_kvc19, False, custom_topk=self.kv_commit_topk
            )
            torch.npu.synchronize()
        return out_shr, e_cal_idx1, e_kvc

    def _q8_logits_rollout(self, x: torch.Tensor) -> torch.Tensor:
        act_q, act_s = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
        return torch_npu.npu_quant_matmul(
            act_q,
            self.cal_head_w_q8_nz,
            self.cal_head_s8,
            pertoken_scale=act_s,
            output_dtype=torch.bfloat16,
        )

    def _run_compact_body(self):
        stream_ptr = ctypes.c_void_p(torch.npu.current_stream(self.dev).npu_stream)
        st = self.wt.ops.tilert_kv_compact_launch(
            stream_ptr,
            self.descs_dev.data_ptr(),
            self.compact_ctrl_dev.data_ptr(),
            self.num_cache_tensors,
        )
        if st != 0:
            raise RuntimeError(f"kv_compact failed: {st}")

    def _exact_index_keys(self, entry):
        from vllm_ascend.ops.triton.rope import rope_forward_triton_siso

        idx_impl = self.impl.indexer.impl
        wk_proj = idx_impl.wk_weights_proj
        norm_x = entry.norm_outputs[0]
        rows = norm_x.shape[0]
        if rows == 1:
            kw, _ = wk_proj(norm_x)
        else:
            kw = torch.cat([wk_proj(norm_x[r : r + 1])[0] for r in range(rows)], dim=0)
        k_li = idx_impl.k_norm(kw[:, :128]).view(rows, 1, 128)
        k_li = rope_forward_triton_siso(
            k_li,
            entry.cos_buf.view(rows, 64),
            entry.sin_buf.view(rows, 64),
            rope_dim=64,
            is_neox_style=idx_impl.is_rope_neox_style,
        ).view(rows, 128)
        self.draft.cache["index_keys"].view(-1, 128).index_copy_(0, entry.slot_i64, k_li)

    def _run_commit_seed_body(self):
        from vllm_ascend.ops import rotary_embedding as rope_mod

        cm = self.COMMIT_M
        ekv = self.entry_kv_commit
        e1 = self.entry_cal_idx_1
        stream_ptr = ctypes.c_void_p(torch.npu.current_stream(self.dev).npu_stream)
        flushed = self.wt.ops.tilert_get_flushed_stream
        sin_ptr = rope_mod._sin_cache.data_ptr()
        cos_ptr = rope_mod._cos_cache.data_ptr()

        self.commit_ttoks_pad32[: self.w].copy_(self.st["target_tokens"][: self.w])
        row_idx = self.commit_ctrl_dev[3 : 3 + cm]
        par_m = self.st["target_hidden"].index_select(0, row_idx)
        for off in range(0, cm, 8):
            torch.mm(
                par_m[off : off + 8],
                self.rot_shard_t,
                out=self.commit_rot_shard[off : off + 8],
            )
        st = self.wt.ops.tilert_dspine_prolog_pack_launch(
            stream_ptr,
            self.commit_ctrl_dev.data_ptr(),
            self.commit_ttoks_pad32.data_ptr(),
            self.emb_w.data_ptr(),
            self.commit_rot_shard.data_ptr(),
            self.commit_comm_in.data_ptr(),
            cm,
            self.emb_vocab_start,
            self.emb_vocab_end,
            self.rank,
        )
        if st != 0:
            raise RuntimeError(f"dspine_prolog_pack failed: {st}")
        torch.ops.tilert.onesided_all_reduce_out(
            self.commit_comm_in[:cm], self.commit_comm_out[:cm]
        )
        torch.ops.tilert.onesided_all_reduce_out(
            self.commit_comm_in[cm : 2 * cm], self.commit_comm_out[cm : 2 * cm]
        )
        en = torch_npu.npu_rms_norm(
            self.commit_comm_out[:cm], self.enorm_w, self.norm_eps
        )[0]
        hn = torch_npu.npu_rms_norm(
            self.commit_comm_out[cm : 2 * cm], self.hnorm_w, self.norm_eps
        )[0]
        self.commit_eh_in[:, :6144].copy_(en)
        self.commit_eh_in[:, 6144:].copy_(hn)
        for off in range(0, cm, 8):
            torch.mm(
                self.commit_eh_in[off : off + 8],
                self.w_eh_col_r,
                out=self.commit_eh_shard[off : off + 8],
            )
        sl_s = self.rank * 768
        self.commit_eh_comm[:, sl_s : sl_s + 768].copy_(self.commit_eh_shard)
        torch.ops.tilert.onesided_all_reduce_out(
            self.commit_eh_comm, self.commit_hs_all[:cm]
        )

        st = self.wt.ops.tilert_dspine_prep_commit_seed_v3_launch(
            stream_ptr,
            self.commit_ctrl_dev.data_ptr(),
            self.commit_hs_all.data_ptr(),
            sin_ptr,
            cos_ptr,
            ekv.hs_in.data_ptr(),
            ekv.sin_buf.data_ptr(),
            ekv.cos_buf.data_ptr(),
            ekv.slot_i64.data_ptr(),
            ekv.q_len.data_ptr(),
            ekv.kv_len.data_ptr(),
            e1.hs_in.data_ptr(),
            e1.sin_buf.data_ptr(),
            e1.cos_buf.data_ptr(),
            e1.slot_i64.data_ptr(),
            e1.q_len.data_ptr(),
            e1.kv_len.data_ptr(),
            cm,
        )
        if st != 0:
            raise RuntimeError(f"prep_commit_seed_v3 failed: {st}")

        st_ekv = ekv.fn(
            flushed(),
            ctypes.c_uint64(ekv.local.sync),
            ekv.local.config,
            ekv.local.addresses,
            ekv.comm_buf,
        )
        if st_ekv != 0:
            raise RuntimeError("kv_commit kernel failed")
        self._exact_index_keys(ekv)
        st_e1 = e1.fn(
            flushed(),
            ctypes.c_uint64(e1.local.sync),
            e1.local.config,
            e1.local.addresses,
            e1.comm_buf,
        )
        if st_e1 != 0:
            raise RuntimeError("seed e1 kernel failed")
        self._exact_index_keys(e1)

        s0_gamma = (
            self.wt.logit_norm_seed0
            if self.wt.logit_norm_seed0 is not None
            else self.mtp_layer.shared_head.norm.weight
        )
        rec_0 = torch_npu.npu_rms_norm(e1.bridge[1] + e1.output, s0_gamma, self.norm_eps)[0]
        self.seed_rec_buf.copy_(rec_0)
        logits_0 = self._q8_logits_rollout(rec_0)
        k_seed = max(2, self.b0) if self.narrow_topk else 4
        st = self.wt.ops.tilert_logits_topk_lse_bf16_launch(
            stream_ptr,
            self.sync_addr,
            logits_0.data_ptr(),
            self.logits_work.data_ptr(),
            self.gather_in.data_ptr(),
            1,
            self.head_row_stride,
            self.head_valid_vocab,
            self.head_vocab_start,
            k_seed,
            self.rank,
        )
        if st != 0:
            raise RuntimeError(f"logits_topk_lse_bf16 failed: {st}")
        torch.ops.tilert.onesided_all_reduce_out(self.gather_in, self.gather_out)
        st = self.wt.ops.tilert_dspine_beam_step_v3_launch(
            stream_ptr,
            self.sync_addr,
            0,
            self.max_b,
            self.b0,
            k_seed,
            self.b0,
            0,
            self.k_sfa,
            0,
            self.beta_margin,
            self.beta_depth,
            self.gather_out.data_ptr(),
            0,
            self.seed_top_i.data_ptr(),
            self.seed_top_lp.data_ptr(),
            0,
            self.prefix_topk_buf.data_ptr(),
            0,
            self.draft.indices_buffer.data_ptr(),
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            self.beam_work.data_ptr(),
        )
        if st != 0:
            raise RuntimeError(f"beam_step_v3 seed0 failed: {st}")

    def _run_draft_level(self, lvl_idx: int):
        from vllm_ascend.ops import rotary_embedding as rope_mod

        specs, pool_sz, p_depths = self.sched_specs[lvl_idx]
        w = self.w
        b0 = self.b0
        eshr0 = self.cal_entries[b0]
        stream_ptr = ctypes.c_void_p(torch.npu.current_stream(self.dev).npu_stream)
        flushed = self.wt.ops.tilert_get_flushed_stream
        sin_ptr = rope_mod._sin_cache.data_ptr()
        cos_ptr = rope_mod._cos_cache.data_ptr()

        st = self.wt.ops.tilert_dspine_beam_step_v3_launch(
            stream_ptr,
            self.sync_addr,
            -1,
            b0,
            b0,
            b0,
            b0,
            0,
            self.k_sfa,
            0,
            self.beta_margin,
            self.beta_depth,
            self.gather_out.data_ptr(),
            self.draft_ctrl_dev.data_ptr(),
            self.seed_top_i.data_ptr(),
            self.seed_top_lp.data_ptr(),
            self.seed_rec_buf.data_ptr(),
            self.prefix_topk_buf.data_ptr(),
            self.anc_slots_buf.data_ptr(),
            self.draft.indices_buffer.data_ptr(),
            self.active_toks.data_ptr(),
            self.active_scores.data_ptr(),
            self.active_raw_lp.data_ptr(),
            self.active_margins.data_ptr(),
            self.active_pids.data_ptr(),
            self.active_recs[:b0].data_ptr(),
            0,
            self.pool_scores.data_ptr(),
            self.pool_raw_lp.data_ptr(),
            self.pool_tokens.data_ptr(),
            self.pool_parents.data_ptr(),
            sin_ptr,
            cos_ptr,
            eshr0.sin_buf.data_ptr(),
            eshr0.cos_buf.data_ptr(),
            eshr0.slot_i64.data_ptr(),
            eshr0.kv_len.data_ptr(),
            self.beam_work.data_ptr(),
        )
        if st != 0:
            raise RuntimeError(f"beam_step_v3 init failed: {st}")

        slot_cur = b0
        for d, cur_b, k_exp, nxt_b, p_off in specs:
            s_idx = self.wt.step_idx(d)
            eshr = self.cal_entries[cur_b]
            a_rec = self.active_recs[:cur_b]
            r_part = self.rot_part[:cur_b]
            p_in = self.prolog_comm_in[: 2 * cur_b]
            p_out = self.prolog_comm_out[: 2 * cur_b]
            e_in = self.eh_in[:cur_b]
            e_comm = self.eh_comm_in[:cur_b]

            torch.matmul(a_rec, self.wt.rot_w_step[s_idx], out=r_part)
            st = self.wt.ops.tilert_prolog_pack_launch(
                stream_ptr,
                self.active_toks.data_ptr(),
                self.emb_w.data_ptr(),
                r_part.data_ptr(),
                self.wt.rot_b_step[s_idx].data_ptr(),
                p_in.data_ptr(),
                cur_b,
                self.emb_vocab_start,
                self.emb_vocab_end,
                self.rank,
            )
            if st != 0:
                raise RuntimeError(f"prolog_pack failed: {st}")
            torch.ops.tilert.onesided_all_reduce_out(p_in, p_out)
            st = self.wt.ops.tilert_dual_rmsnorm_slice_launch(
                stream_ptr,
                p_out.data_ptr(),
                self.wt.enorm_step[s_idx].data_ptr(),
                self.wt.hnorm_step[s_idx].data_ptr(),
                e_in.data_ptr(),
                cur_b,
                self.rank,
                self.norm_eps,
            )
            if st != 0:
                raise RuntimeError(f"dual_rmsnorm_slice failed: {st}")
            torch.matmul(e_in, self.wt.w_eh_step[s_idx], out=e_comm)
            torch.ops.tilert.onesided_all_reduce_out(e_comm, eshr.hs_in)

            st_eshr = eshr.fn(
                flushed(),
                ctypes.c_uint64(eshr.local.sync),
                eshr.local.config,
                eshr.local.addresses,
                eshr.comm_buf,
            )
            if st_eshr != 0:
                raise RuntimeError("draft rollout kernel failed")

            recs_d = torch_npu.npu_rms_norm(
                eshr.bridge[1] + eshr.output, self.wt.logit_norm_step[s_idx], self.norm_eps
            )[0]
            logits_d = self._q8_logits_rollout(recs_d)
            k_top = max(2, k_exp) if self.narrow_topk else 4
            st = self.wt.ops.tilert_logits_topk_lse_bf16_launch(
                stream_ptr,
                self.sync_addr,
                logits_d.data_ptr(),
                self.logits_work.data_ptr(),
                self.gather_in.data_ptr(),
                cur_b,
                self.head_row_stride,
                self.head_valid_vocab,
                self.head_vocab_start,
                k_top,
                self.rank,
            )
            if st != 0:
                raise RuntimeError(f"logits_topk_lse_bf16 rollout failed: {st}")
            torch.ops.tilert.onesided_all_reduce_out(self.gather_in, self.gather_out)

            eshr_nxt = self.cal_entries[nxt_b] if nxt_b > 0 else eshr
            st = self.wt.ops.tilert_dspine_beam_step_v3_launch(
                stream_ptr,
                self.sync_addr,
                d,
                cur_b,
                k_exp,
                k_top,
                nxt_b,
                p_off,
                self.k_sfa,
                slot_cur,
                self.beta_margin,
                self.beta_depth,
                self.gather_out.data_ptr(),
                self.draft_ctrl_dev.data_ptr(),
                self.seed_top_i.data_ptr(),
                self.seed_top_lp.data_ptr(),
                self.seed_rec_buf.data_ptr(),
                self.prefix_topk_buf.data_ptr(),
                self.anc_slots_buf.data_ptr(),
                self.draft.indices_buffer.data_ptr(),
                self.active_toks.data_ptr(),
                self.active_scores.data_ptr(),
                self.active_raw_lp.data_ptr(),
                self.active_margins.data_ptr(),
                self.active_pids.data_ptr(),
                self.active_recs[: max(1, nxt_b)].data_ptr(),
                recs_d.data_ptr(),
                self.pool_scores.data_ptr(),
                self.pool_raw_lp.data_ptr(),
                self.pool_tokens.data_ptr(),
                self.pool_parents.data_ptr(),
                sin_ptr,
                cos_ptr,
                eshr_nxt.sin_buf.data_ptr(),
                eshr_nxt.cos_buf.data_ptr(),
                eshr_nxt.slot_i64.data_ptr(),
                eshr_nxt.kv_len.data_ptr(),
                self.beam_work.data_ptr(),
            )
            if st != 0:
                raise RuntimeError(f"beam_step_v3 step failed: {st}")
            slot_cur += nxt_b

        st = self.wt.ops.tilert_dspine_build_tree_v3_launch(
            stream_ptr,
            self.sync_addr,
            w,
            self.min_w_act,
            pool_sz,
            self.st["num_full_seq_ptrs"],
            self.st["has_mc2_mask"],
            self.st["num_rope_dsts"],
            self.t_base_ms,
            self.c_w_ms,
            self.draft_ctrl_dev.data_ptr(),
            self.pool_scores.data_ptr(),
            self.pool_raw_lp.data_ptr(),
            self.pool_tokens.data_ptr(),
            self.pool_parents.data_ptr(),
            p_depths.data_ptr(),
            self.tree_tokens.data_ptr(),
            self.tree_parents.data_ptr(),
            self.tree_depths.data_ptr(),
            self.st["tree_mask_i32"].data_ptr(),
            self.st["t_input_ids"].data_ptr(),
            self.st["t_positions"].data_ptr(),
            self.st["t_slots"].data_ptr(),
            self.st["t_slots_i64"].data_ptr(),
            self.st["dyn_w"].data_ptr(),
            self.st["full_seq_ptrs_dev"].data_ptr(),
            self.st["mc2_mask_ptr"],
            cos_ptr,
            sin_ptr,
            self.st["rope_dsts_dev"].data_ptr(),
            self.tree_work.data_ptr(),
        )
        if st != 0:
            raise RuntimeError(f"build_tree_v3 failed: {st}")
        self.host_read_buf[:w].copy_(self.tree_tokens)
        self.host_read_buf[w : 2 * w].copy_(self.tree_parents)

    def _capture(self):
        from vllm.config import set_current_vllm_config
        from vllm_ascend.worker.model_runner_v1 import graph_capture

        dev = self.dev
        self.draft_ctrl_cpu.zero_()
        self.draft_ctrl_cpu[0] = self.first_token
        self.draft_ctrl_cpu[1] = 8192
        self.draft_ctrl_dev.copy_(self.draft_ctrl_cpu)

        self.compact_ctrl_cpu.zero_()
        self.compact_ctrl_cpu[0] = 8192
        self.compact_ctrl_cpu[1] = 1
        self.compact_ctrl_cpu[2] = 8192
        self.compact_ctrl_dev.copy_(self.compact_ctrl_cpu)

        self.commit_ctrl_cpu.zero_()
        self.commit_ctrl_cpu[0] = 8191
        self.commit_ctrl_cpu[1] = 2
        self.commit_ctrl_cpu[2] = 2
        self.commit_ctrl_dev.copy_(self.commit_ctrl_cpu)

        with (
            torch.npu.device(dev),
            torch.inference_mode(),
            set_current_vllm_config(self.draft.config),
        ):
            self.st["target_tokens"][0:2].fill_(self.first_token)
            self.st["target_hidden"][0:2].copy_(self.prefix_parent_hidden.to(dev).expand(2, -1))
            for c_val in self.draft.cache.values():
                c_val[8191:].zero_()

            self._run_commit_seed_body()
            for lvl_i in range(len(self.sched_specs)):
                self._run_draft_level(lvl_i)
            self._run_compact_body()
            torch.npu.synchronize()

            pool = torch.npu.graph_pool_handle()
            caller = torch.npu.current_stream(dev)

            self.compact_graph = torch.npu.NPUGraph()
            with graph_capture(dev) as cap:
                with torch.npu.graph(self.compact_graph, pool=pool, stream=cap.stream):
                    self._run_compact_body()
            caller.wait_stream(cap.stream)

            self.commit_seed_graph = torch.npu.NPUGraph()
            with graph_capture(dev) as cap:
                with torch.npu.graph(self.commit_seed_graph, pool=pool, stream=cap.stream):
                    self._run_commit_seed_body()
            caller.wait_stream(cap.stream)

            self.draft_graphs = []
            for lvl_i in range(len(self.sched_specs)):
                dg = torch.npu.NPUGraph()
                with graph_capture(dev) as cap:
                    with torch.npu.graph(dg, pool=pool, stream=cap.stream):
                        self._run_draft_level(lvl_i)
                caller.wait_stream(cap.stream)
                self.draft_graphs.append(dg)
            torch.npu.synchronize()

        with torch.inference_mode():
            for dst_q, ref_q in self.st.get("q_restore_pairs", ()):
                dst_q.copy_(ref_q)
            for c_val in self.draft.cache.values():
                c_val[8191:].zero_()
            self.st["tree_mask_i32"].zero_()
        torch.npu.synchronize()

    def _replay_commit_seed(self, main_start: int, acc_rows: list[int]):
        count = len(acc_rows)
        cm = self.COMMIT_M
        self.commit_ctrl_cpu[0] = main_start
        self.commit_ctrl_cpu[1] = count
        self.commit_ctrl_cpu[2] = count
        last_r = acc_rows[-1]
        for j in range(cm):
            self.commit_ctrl_cpu[3 + j] = acc_rows[j] if j < count else last_r
        self.commit_ctrl_dev.copy_(self.commit_ctrl_cpu, non_blocking=True)
        self.commit_seed_graph.replay()

    def _seed_round0(self, pending_tok: int, parent_hidden: torch.Tensor, pos_minus_1: int):
        self.st["target_tokens"][0:1].fill_(pending_tok)
        self.st["target_hidden"][0:1].copy_(parent_hidden)
        self._replay_commit_seed(pos_minus_1, [0])

    def _commit_and_seed(self, main_start: int, acc_rows: list[int], need_seed: bool):
        count = len(acc_rows)
        did_compact = False
        if acc_rows != list(range(count)):
            did_compact = True
            self.compact_ctrl_cpu[0] = main_start
            self.compact_ctrl_cpu[1] = count
            for j in range(count):
                self.compact_ctrl_cpu[2 + j] = main_start + acc_rows[j]
            self.compact_ctrl_dev.copy_(self.compact_ctrl_cpu, non_blocking=True)
            self.compact_graph.replay()
        if need_seed:
            self._replay_commit_seed(main_start, acc_rows)
        return did_compact

    def _read_round_tensors(self):
        w = self.w
        self.host_read_buf[2 * w : 3 * w].copy_(self.st["target_tokens"][:w], non_blocking=True)
        vals = self.host_read_buf.tolist()
        return vals[:w], vals[w : 2 * w], vals[2 * w : 3 * w]

    def decode_full(self, case_b: dict, name: str, check_digests: bool = False):
        from ..cann_model import VOCAB_SIZE
        from ..mtp_round import MtpRoundExecutor
        from ..tree_resolve import accepted_path

        w = self.w
        target = self.target
        draft = self.draft
        with torch.npu.device(self.dev), torch.inference_mode(), exact_rows(target):
            for dst_q, ref_q in self.st.get("q_restore_pairs", ()):
                dst_q.copy_(ref_q)
            torch.ops.tilert.set_active_tokens_ptr(self.st["dyn_w"], True, True)
            target.inject(case_b["prefix"].main)
            draft.inject(case_b["mtp"])
            for c_val in draft.cache.values():
                c_val[8191:].zero_()
            parent_dev = case_b["prefix"].parent_hidden.to(self.dev)
            executor = MtpRoundExecutor(target, draft, width=w, max_new_tokens=1024, eos_tokens=())
            executor.start(case_b["first"], parent_dev)

            out_toks = [case_b["first"]]
            rounds_acc = []
            round_traces = []
            compact_cnt = 0
            lvl_counts = [0] * len(self.draft_graphs)
            h_ctrl = self.horizon_ctrl
            if h_ctrl is not None:
                h_ctrl.reset()

            torch.npu.synchronize()
            torch.distributed.barrier(group=self.channel_group)
            t_start = time.perf_counter()
            seeded = False
            draft_committed_pos = int(draft.position)
            seed_positions = [8191]
            while executor.state.finish_reason is None:
                plan = executor.state.begin()
                if h_ctrl is not None and len(self.draft_graphs) > 1:
                    lvl = min(h_ctrl.select_level(), len(self.draft_graphs) - 1)
                else:
                    lvl = 0
                lvl_counts[lvl] += 1
                dg = self.draft_graphs[lvl]

                if not seeded:
                    self._seed_round0(
                        plan.pending_token,
                        executor.seed_parent_hidden,
                        plan.main_start - 1,
                    )
                    seeded = True
                self.draft_ctrl_cpu[0] = plan.pending_token
                self.draft_ctrl_cpu[1] = plan.main_start
                self.draft_ctrl_dev.copy_(self.draft_ctrl_cpu, non_blocking=True)
                dg.replay()
                self.st["target_aclgraph"].replay()
                self.st["head_graph"].replay()
                nodes, lineage, predictions = self._read_round_tensors()
                path = accepted_path(nodes, lineage, predictions)
                along = [nodes[idx] for idx in path]
                predicted = [predictions[idx] for idx in path]
                bonus = predictions[path[-1]]
                along += [(bonus + 1) % VOCAB_SIZE] * (w - len(along))
                predicted += [bonus] * (w - len(predicted))
                commit = executor.state.resolve(plan, along, predicted)
                acc_rows = path[: commit.committed]
                if self._commit_and_seed(plan.main_start, acc_rows, True):
                    compact_cnt += 1
                draft_committed_pos = int(commit.draft.retained)
                seed_positions.append(plan.main_start + len(acc_rows) - 1)
                target.position = commit.main.retained
                draft.position = commit.draft.retained

                h_lvl = int(h_ctrl.update(commit.committed)) if h_ctrl is not None else 0
                rounds_acc.append(commit.committed)
                out_toks.extend(commit.accepted_tokens)
                if check_digests:
                    w_act = int(self.st["dyn_w"][0, 0].item())
                    kv_q = len(acc_rows) - 1
                    round_traces.append(
                        {
                            "round_idx": len(rounds_acc) - 1,
                            "width": w,
                            "min_w_act": self.min_w_act,
                            "w_act": w_act,
                            "mc2_mask": [i < w_act for i in range(w)],
                            "dcci_issued": True,
                            "collective_rows": self.COMMIT_M,
                            "l78_gemm_chunk_rows": 8,
                            "kv_commit_q_len": kv_q,
                            "kv_commit_wrote": kv_q > 0,
                            "accepted": commit.committed,
                            "final_pos": int(target.position),
                            "horizon_level": h_lvl,
                            "done": executor.state.finish_reason is not None,
                        }
                    )

            torch.npu.synchronize()
            torch.distributed.barrier(group=self.channel_group)
            dt_ms = (time.perf_counter() - t_start) * 1000.0

            for dst_q, ref_q in self.st.get("q_restore_pairs", ()):
                dst_q.copy_(ref_q)

            dt_list = [None] * 8
            torch.distributed.all_gather_object(dt_list, dt_ms, group=self.channel_group)
            tpot_list = [x / 1023.0 for x in dt_list]
            max_tpot = round(max(tpot_list), 4)
            out_sha = hashlib.sha256(json.dumps(out_toks).encode()).hexdigest()
            sum_acc = sum(rounds_acc)
            tok_ok = len(out_toks) == 1024 and sum_acc == 1023 and out_sha == case_b["ref_sha"]

            res = {
                "name": name,
                "case": case_b["slug"],
                "w": w,
                "rounds": len(rounds_acc),
                "sum_accepted": sum_acc,
                "mean_accepted": round(sum_acc / len(rounds_acc), 3),
                "compact_rounds": compact_cnt,
                "lvl_counts": lvl_counts,
                "max_tpot_ms": max_tpot,
                "rank_dt_ms": dt_list,
                "rank_tpots_ms": [round(x, 4) for x in tpot_list],
                "ms_per_round": round(max(tpot_list) * 1023.0 / len(rounds_acc), 3),
                "accepted_per_round": list(rounds_acc),
                "output_token_ids": list(out_toks),
                "output_sha256": out_sha,
                "tokens_ok": tok_ok,
                "target_pos": int(target.position),
                "draft_pos": int(draft.position),
                "draft_committed_pos": draft_committed_pos,
                "seed_positions": seed_positions,
            }
            if check_digests:
                target_dig = case_b["digest_fn"](target.cache, target.position)
                draft_dig = case_b["digest_fn"]([draft.cache], draft_committed_pos)
                main_ok = target_dig == case_b["ref_state"]
                all_checks = [None] * 8
                torch.distributed.all_gather_object(
                    all_checks,
                    {
                        "tok": tok_ok,
                        "main": main_ok,
                        "target_digest": target_dig,
                        "draft_digest": draft_dig,
                    },
                    group=self.channel_group,
                )
                same_target = all(
                    c["target_digest"] == all_checks[0]["target_digest"] for c in all_checks
                )
                same_draft = all(
                    c["draft_digest"] == all_checks[0]["draft_digest"] for c in all_checks
                )
                res["all_ranks_target_exact"] = all(c["tok"] and c["main"] for c in all_checks)
                res["all_ranks_cache_parity"] = same_target and same_draft
                res["target_digest"] = all_checks[0]["target_digest"]
                res["draft_digest"] = all_checks[0]["draft_digest"]
                res["local_target_digest"] = target_dig
                res["local_draft_digest"] = draft_dig
                res["round_traces"] = round_traces
                if not res["all_ranks_target_exact"] or not res["all_ranks_cache_parity"]:
                    raise RuntimeError(f"Verification mismatch: {all_checks}")
            elif not tok_ok:
                raise RuntimeError(f"Rank {self.rank} token parity mismatch: {out_sha}")
            return res

    def close(self):
        torch.npu.synchronize()
        tree_mask = self.st.get("tree_mask_i32")
        if tree_mask is not None:
            tree_mask.zero_()
        mgr = getattr(self.target, "_whole_layer", None)
        if mgr is not None:
            mgr.set_tree_mask_ptr(0)
        torch.npu.synchronize()
        for g_attr in ("compact_graph", "commit_seed_graph"):
            g_obj = getattr(self, g_attr, None)
            if g_obj is not None:
                g_obj.reset()
                setattr(self, g_attr, None)
        for dg in getattr(self, "draft_graphs", ()):
            dg.reset()
        self.draft_graphs = []
        if hasattr(self.wt, "_v3_shared_bundle"):
            delattr(self.wt, "_v3_shared_bundle")
        self._cal_weight_refs = None
        l78_win = getattr(self.draft, "_l78_win", None)
        if l78_win is not None and getattr(l78_win, "handle", 0) != 0:
            l78_win.close()
            self.draft._l78_win = None
        gc.collect()
