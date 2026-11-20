import copy
import ctypes
import gc
import hashlib
import math
from pathlib import Path
import shutil
import struct

from ..abi import LocalRecords, _alloc_aligned
from ..config import HIDDEN_SIZE, K_NORM_EPS, RMS_NORM_EPS, VOCAB_SIZE
from .build_dspine_ops import build_dspine_ops, default_build_dir
from .calibrated_l78 import (
    L78_VARIANT_SPECS,
    build_beam_specs,
    build_draft_l78_k110419,
)

MAX_COMMIT_M = 16


def _load_ops_cdll(so_path: Path, rank: int) -> ctypes.CDLL:
    import torch

    sha = hashlib.sha256(so_path.read_bytes()).hexdigest()[:10]
    ver_so = so_path.with_name(f"tilert_dspine_ops_{sha}.so")
    if rank == 0 and not ver_so.exists():
        tmp = ver_so.with_name(f".{ver_so.name}.{rank}.tmp")
        shutil.copy2(so_path, tmp)
        tmp.replace(ver_so)
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

    lib.tilert_dspine_prolog_prepare_meta_launch.argtypes = [
        ctypes.c_void_p,
        *([ctypes.c_uint64] * 11),
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.tilert_dspine_prolog_prepare_meta_launch.restype = ctypes.c_int

    lib.tilert_dspine_resolve_and_advance_launch.argtypes = [
        ctypes.c_void_p,
        *([ctypes.c_uint64] * 9),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.tilert_dspine_resolve_and_advance_launch.restype = ctypes.c_int

    lib.tilert_dspine_embed_lookup_launch.argtypes = [
        ctypes.c_void_p,
        *([ctypes.c_uint64] * 4),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.tilert_dspine_embed_lookup_launch.restype = ctypes.c_int

    lib.tilert_dspine_exact_index_keys_launch.argtypes = [
        ctypes.c_void_p,
        *([ctypes.c_uint64] * 11),
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_int32,
    ]
    lib.tilert_dspine_exact_index_keys_launch.restype = ctypes.c_int

    lib.tilert_step3c_prolog_pack_guarded_launch.argtypes = [
        ctypes.c_void_p,
        *([ctypes.c_uint64] * 6),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.tilert_step3c_prolog_pack_guarded_launch.restype = ctypes.c_int

    lib.tilert_step3c_dual_rmsnorm_slice_guarded_launch.argtypes = [
        ctypes.c_void_p,
        *([ctypes.c_uint64] * 5),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
    ]
    lib.tilert_step3c_dual_rmsnorm_slice_guarded_launch.restype = ctypes.c_int

    lib.tilert_logits_topk_lse_bf16_guarded_launch.argtypes = [
        ctypes.c_void_p,
        *([ctypes.c_uint64] * 5),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.tilert_logits_topk_lse_bf16_guarded_launch.restype = ctypes.c_int

    lib.tilert_dspine_add_rmsnorm_quant_launch.argtypes = [
        ctypes.c_void_p,
        *([ctypes.c_uint64] * 7),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
    ]
    lib.tilert_dspine_add_rmsnorm_quant_launch.restype = ctypes.c_int

    lib.tilert_dspine_prolog_pack_compact_launch.argtypes = [
        ctypes.c_void_p,
        *([ctypes.c_uint64] * 5),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.tilert_dspine_prolog_pack_compact_launch.restype = ctypes.c_int

    lib.tilert_dspine_alloc_comm_win.argtypes = []
    lib.tilert_dspine_alloc_comm_win.restype = ctypes.c_uint64

    lib.tilert_dspine_free_comm_win.argtypes = [ctypes.c_uint64]
    lib.tilert_dspine_free_comm_win.restype = None

    lib.tilert_dspine_draft_head_topk_launch.argtypes = [
        ctypes.c_void_p,
        *([ctypes.c_uint64] * 15),
        *([ctypes.c_int32] * 6),
        ctypes.c_float,
    ]
    lib.tilert_dspine_draft_head_topk_launch.restype = ctypes.c_int
    return lib


def load_dspine_ops(
    rank: int, out_dir: str | Path | None = None, world_size: int = 8
) -> tuple[Path, ctypes.CDLL]:
    import torch

    out = default_build_dir() if out_dir is None else Path(out_dir)
    so_path = out / "tilert_dspine_ops.so"
    if not so_path.exists():
        if rank == 0:
            if int(world_size) == 8:
                build_dspine_ops(out)
            else:
                build_dspine_ops(out, world_size=int(world_size))
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    return so_path, _load_ops_cdll(so_path, rank)


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
    fn_exact = getattr(lib, f"layer_tp_tree_set_exact_idx_ptrs_K{ver}", None)
    if fn_exact is not None:
        fn_exact.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_float,
        ]
        fn_exact.restype = None
    fn_prep = getattr(lib, f"layer_tp_tree_prepare_records_K{ver}", None)
    if fn_prep is not None:
        u64_ptr = ctypes.POINTER(ctypes.c_uint64)
        fn_prep.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_float),
            u64_ptr,
            u64_ptr,
            ctypes.c_uint64,
            u64_ptr,
            u64_ptr,
            u64_ptr,
            u64_ptr,
            u64_ptr,
        ]
        fn_prep.restype = ctypes.c_int
    fn_m(0)
    return lib, fn_m, fn_i, fn_s, fn_kvc


class _DraftEntry:
    pass


class CalibratedL78V3Runner:
    COMMIT_M = MAX_COMMIT_M

    def __init__(
        self,
        arena,
        prefix,
        metadata_lib,
        ffn_ops,
        l78_win,
        shared_tgt: dict,
        channel_group,
        first_token: int,
        b_scheds: list[list[int]],
        horizon_ctrl,
        *,
        inv_temp: float = 0.95,
        beta_margin: float | None = None,
        beta_depth: float | None = None,
        k_sfa: int = 1024,
        min_w_act: int = 10,
        t_base_ms: float = 41.5,
        c_w_ms: float = 0.68,
        narrow_topk: bool = False,
        l78_build_dir: str | Path | None = None,
    ):
        import torch

        self.arena = arena
        self.prefix = prefix
        self.metadata_lib = metadata_lib
        self.ffn_ops = ffn_ops
        self.l78_win = l78_win
        self.st = shared_tgt
        self.w = int(shared_tgt["width"])
        self.dw = arena.draft
        self.wt = self.dw.cal_wt
        self.rank = int(arena.rank)
        if self.wt.ops is None:
            _, self.wt.ops = load_dspine_ops(self.rank)
        self.world_size = int(arena.world_size)
        self.dev = arena.device
        self.channel_group = channel_group
        self.first_token = int(first_token)
        self.l78_build_dir = l78_build_dir

        self.b_scheds = [tuple(int(x) for x in s) for s in b_scheds]
        if not 1 <= len(self.b_scheds) <= 3 or max(len(s) for s in self.b_scheds) > 14:
            raise ValueError("Unsupported DSpine beam schedule dimensions")
        self.b0 = int(self.b_scheds[0][0])
        self.max_steps = max(len(s) for s in self.b_scheds)
        self.max_b = max(max(s) for s in self.b_scheds)
        if not (
            1 <= self.b0 <= 4
            and 1 <= self.max_b <= 4
            and all(s[0] == self.b0 for s in self.b_scheds)
        ):
            raise ValueError("Beam schedules must share root beam width in [1, 4]")

        self.horizon_ctrl = horizon_ctrl
        self.inv_temp = float(inv_temp)
        self.beta_margin = float(self.wt.beta_margin if beta_margin is None else beta_margin)
        self.beta_depth = float(self.wt.beta_depth if beta_depth is None else beta_depth)
        self.k_sfa = int(k_sfa)
        if self.k_sfa not in (512, 1024, 2048):
            raise ValueError(f"Unsupported k_sfa: {self.k_sfa}")
        self.min_w_act = int(min_w_act)
        self.t_base_ms = float(t_base_ms)
        self.c_w_ms = float(c_w_ms)
        self.narrow_topk = bool(narrow_topk)

        dev = self.dev
        self._owners = []
        self.step_m_rows = [
            max(s[idx] for s in self.b_scheds if idx < len(s))
            for idx in range(self.max_steps)
        ]

        max_pool = 1
        max_depths = [0]
        for s in self.b_scheds:
            _, depths = build_beam_specs(s)
            if len(depths) > max_pool:
                max_pool = len(depths)
                max_depths = depths
        self.max_pool_size = max_pool
        dep_cpu = torch.zeros((256,), dtype=torch.int64)
        for i, dp in enumerate(max_depths):
            dep_cpu[i] = dp
        self.pool_depths_dev = self._alloc_from(dep_cpu, 64)

        self.step_table_stride = self.max_steps + 2
        tbl_cpu = torch.zeros((16, self.step_table_stride), dtype=torch.int64)
        for lvl, sched in enumerate(self.b_scheds):
            packed = 0
            for t, b in enumerate(sched):
                packed |= (int(b) & 0xF) << (4 * t)
            tbl_cpu[lvl, 0] = len(sched)
            tbl_cpu[lvl, 1] = packed
            for t, b in enumerate(sched):
                tbl_cpu[lvl, 2 + t] = int(b)
        self.step_table_dev = self._alloc_from(tbl_cpu, 64)

        _, sq_u8 = self._alloc_bytes(max(16, self.max_steps) * 4, 64, zero=True)
        self.step_qlen_dev = sq_u8.view(torch.int32)
        _, sq2_u8 = self._alloc_bytes(max(16, self.max_steps) * 4, 64, zero=True)
        self.step_2qlen_dev = sq2_u8.view(torch.int32)
        _, sqp_u8 = self._alloc_bytes(max(8, self.max_steps) * 8, 64, zero=True)
        self.step_qlen_ptrs_dev = sqp_u8.view(torch.int64)
        _, cqp_u8 = self._alloc_bytes(64, 64, zero=True)
        self.commit_qlen_ptrs_dev = cqp_u8.view(torch.int64)

        self.rot_shard_t = self.dw.rot_w_shard.contiguous()
        self.w_eh_col_r = self.dw.w_eh_shard.contiguous()
        self.emb_w = arena.embed_w.contiguous()
        local_vocab = VOCAB_SIZE // self.world_size
        self.emb_vocab_start = self.rank * local_vocab
        self.emb_vocab_end = (self.rank + 1) * local_vocab
        self.enorm_w = self.dw.enorm_w.contiguous()
        self.hnorm_w = self.dw.hnorm_w.contiguous()
        self.norm_eps = RMS_NORM_EPS

        self.head_w = self.dw.head_w_shard.contiguous()
        self.head_row_stride = int(self.head_w.shape[0])
        self.head_valid_vocab = local_vocab
        self.head_vocab_start = self.rank * local_vocab
        self.cal_head_w_q8_nz = self.dw.head_q8_nz
        self.cal_head_s8 = self.dw.head_scale
        self.comm_win = int(self.wt.ops.tilert_dspine_alloc_comm_win())
        if self.comm_win == 0:
            raise RuntimeError("tilert_dspine_alloc_comm_win failed")
        self._owns_comm_win = True
        self.c2b_recs_out = torch.zeros((self.max_b, HIDDEN_SIZE), dtype=torch.bfloat16, device=dev)
        self.c2b_act_q = torch.zeros((self.max_b, HIDDEN_SIZE), dtype=torch.int8, device=dev)
        self.c2b_act_s = torch.zeros((self.max_b,), dtype=torch.float32, device=dev)
        self.c2b_acc_i32 = torch.zeros(
            (self.max_b, self.head_valid_vocab), dtype=torch.int32, device=dev
        )

        self.wk_w_128 = (
            self.dw.front[2][:4]
            .permute(1, 0, 2)
            .reshape(HIDDEN_SIZE, 128)
            .t()
            .contiguous()
        )
        self.exact_idx_work = torch.zeros((self.COMMIT_M, 128), dtype=torch.float32, device=dev)

        self.shard_dim = HIDDEN_SIZE // self.world_size
        self.gather_in = torch.zeros(
            (max(8, self.world_size), 64), dtype=torch.float32, device=dev
        )
        self.gather_out = torch.zeros(
            (max(8, self.world_size), 64), dtype=torch.float32, device=dev
        )
        self.logits_work = torch.zeros((1024,), dtype=torch.float32, device=dev)
        self.beam_work = torch.zeros((32,), dtype=torch.int64, device=dev)
        self.tree_work = torch.zeros((32,), dtype=torch.int64, device=dev)
        self.seed_rec_buf = torch.zeros((1, HIDDEN_SIZE), dtype=torch.bfloat16, device=dev)
        self.draft_indices_buf = torch.zeros((32, 2048), dtype=torch.int32, device=dev)

        self.seed_top_i = torch.zeros((8,), dtype=torch.int64, device=dev)
        self.seed_top_lp = torch.zeros((8,), dtype=torch.float32, device=dev)
        self.prefix_topk_buf = torch.zeros((2048,), dtype=torch.int32, device=dev)
        self.anc_slots_buf = torch.zeros((128,), dtype=torch.int32, device=dev)
        self.active_toks = torch.zeros((8,), dtype=torch.int64, device=dev)
        self.active_scores = torch.zeros((8,), dtype=torch.float32, device=dev)
        self.active_raw_lp = torch.zeros((8,), dtype=torch.float32, device=dev)
        self.active_margins = torch.zeros((8,), dtype=torch.float32, device=dev)
        self.active_pids = torch.zeros((8,), dtype=torch.int64, device=dev)
        self.active_recs = torch.zeros((self.max_b, HIDDEN_SIZE), dtype=torch.bfloat16, device=dev)

        self.rot_part = torch.zeros((self.max_b, self.shard_dim), dtype=torch.bfloat16, device=dev)
        self.prolog_comm_in = torch.zeros(
            (32, HIDDEN_SIZE), dtype=torch.bfloat16, device=dev
        )
        self.prolog_comm_out = torch.zeros(
            (32, HIDDEN_SIZE), dtype=torch.bfloat16, device=dev
        )
        self.eh_in = torch.zeros((self.max_b, 2 * self.shard_dim), dtype=torch.bfloat16, device=dev)
        self.eh_comm_in = torch.zeros((32, HIDDEN_SIZE), dtype=torch.bfloat16, device=dev)

        self.pool_scores = torch.zeros((256,), dtype=torch.float32, device=dev)
        self.pool_raw_lp = torch.zeros((256,), dtype=torch.float32, device=dev)
        self.pool_tokens = torch.zeros((256,), dtype=torch.int64, device=dev)
        self.pool_parents = torch.zeros((256,), dtype=torch.int64, device=dev)

        _, tt_u8 = self._alloc_bytes(max(8, self.w) * 8, 64, zero=True)
        self.tree_tokens = tt_u8.view(torch.int64)[: self.w]
        _, tp_u8 = self._alloc_bytes(max(8, self.w) * 8, 64, zero=True)
        self.tree_parents = tp_u8.view(torch.int64)[: self.w]
        _, td_u8 = self._alloc_bytes(max(8, self.w) * 8, 64, zero=True)
        self.tree_depths = td_u8.view(torch.int64)[: self.w]

        _, dc_u8 = self._alloc_bytes(64, 64, zero=True)
        self.draft_ctrl_dev = dc_u8.view(torch.int64)
        _, cmp_u8 = self._alloc_bytes(64 * 8, 64, zero=True)
        self.compact_ctrl_dev = cmp_u8.view(torch.int64)

        cm = self.COMMIT_M
        self.commit_ttoks_pad32 = torch.zeros((32,), dtype=torch.int64, device=dev)
        _, cmt_u8 = self._alloc_bytes(48 * 8, 64, zero=True)
        self.commit_ctrl_dev = cmt_u8.view(torch.int64)
        self.commit_acc_i32 = cmt_u8[8:12].view(torch.int32)
        self.commit_2acc_i32 = cmt_u8[47 * 8 : 47 * 8 + 4].view(torch.int32)
        self.commit_rot_shard = torch.zeros((cm, self.shard_dim), dtype=torch.bfloat16, device=dev)
        self.commit_comm_in = torch.zeros((4 * cm, HIDDEN_SIZE), dtype=torch.bfloat16, device=dev)
        self.commit_comm_out = torch.zeros((4 * cm, HIDDEN_SIZE), dtype=torch.bfloat16, device=dev)
        self.commit_eh_in = torch.zeros((cm, 2 * HIDDEN_SIZE), dtype=torch.bfloat16, device=dev)
        self.commit_eh_shard = torch.zeros((cm, self.shard_dim), dtype=torch.bfloat16, device=dev)
        self.commit_eh_comm = torch.zeros((2 * cm, HIDDEN_SIZE), dtype=torch.bfloat16, device=dev)
        self.commit_hs_all = torch.zeros((2 * cm, HIDDEN_SIZE), dtype=torch.bfloat16, device=dev)
        self.kv_commit_topk = torch.zeros((cm, 1, 2048), dtype=torch.int32, device=dev)

        self._build_v3_entries()
        self.sync_addr = int(self.entry_cal_idx_1.sync)

        cqp_cpu = torch.tensor(
            [
                int(self.entry_kv_commit.q_len.data_ptr()),
                int(self.entry_cal_idx_1.q_len.data_ptr()),
            ],
            dtype=torch.int64,
        )
        self.commit_qlen_ptrs_dev[:2].copy_(cqp_cpu.to(dev))

        sqp_cpu = torch.tensor(
            [int(eshr.q_len.data_ptr()) for eshr in self.eshr_steps],
            dtype=torch.int64,
        )
        self.step_qlen_ptrs_dev[: self.max_steps].copy_(sqp_cpu.to(dev))
        self.compact_stream = torch.npu.Stream(device=dev)
        self._ev_cmp_fork = torch.npu.Event(enable_timing=False)
        self._ev_cmp_join = torch.npu.Event(enable_timing=False)

    def _alloc_bytes(self, nbytes: int, align: int, *, zero: bool = True):
        raw, view = _alloc_aligned(nbytes, align, self.dev, zero=zero)
        self._owners.append(raw)
        return raw, view

    def _alloc_from(self, cpu_tensor, align: int):
        nbytes = cpu_tensor.numel() * cpu_tensor.element_size()
        _, view = self._alloc_bytes(nbytes, align, zero=False)
        typed = view.view(cpu_tensor.dtype).view(cpu_tensor.shape)
        typed.copy_(cpu_tensor.to(self.dev))
        return typed

    def _ensure_l78_klibs(self):
        need_ver = L78_VARIANT_SPECS[self.k_sfa][0]
        klibs_by_ver = getattr(self.wt, "_p2_klibs_by_ver", None)
        if not isinstance(klibs_by_ver, dict):
            klibs_by_ver = {}
            self.wt._p2_klibs_by_ver = klibs_by_ver
        mtime = int(self.l78_build_dir.stat().st_mtime_ns) if self.l78_build_dir.exists() else 0
        tag = (id(_bind_draft_klib), mtime)
        if need_ver in klibs_by_ver and getattr(self.wt, "_p2_klib_tag", None) == tag:
            klibs_by_ver[need_ver][1](0)
            return klibs_by_ver
        ws = int(getattr(self, "world_size", 8))
        dist_ready = torch_dist_ready()
        if self.rank == 0 or not dist_ready:
            if ws == 8:
                so_19, _ = build_draft_l78_k110419(out_dir=self.l78_build_dir, k_sfa=self.k_sfa)
            else:
                so_19, _ = build_draft_l78_k110419(
                    out_dir=self.l78_build_dir, k_sfa=self.k_sfa, world_size=ws
                )
            out_dir = so_19.parent
            src_so = out_dir / f"chain_K{need_ver}.so"
            sha = hashlib.sha256(src_so.read_bytes()).hexdigest()[:10]
            ver_so = src_so.with_name(f"{src_so.stem}_{sha}.so")
            if not ver_so.exists():
                tmp = ver_so.with_name(f".{ver_so.name}.{self.rank}.tmp")
                shutil.copy2(src_so, tmp)
                tmp.replace(ver_so)
        if dist_ready:
            import torch

            torch.distributed.barrier(group=self.channel_group)
        if self.rank != 0 and dist_ready:
            src_so = Path(self.l78_build_dir) / f"chain_K{need_ver}.so"
            sha = hashlib.sha256(src_so.read_bytes()).hexdigest()[:10]
            ver_so = src_so.with_name(f"{src_so.stem}_{sha}.so")
        klibs_by_ver[need_ver] = _bind_draft_klib(ver_so, need_ver)
        self.wt._p2_klib_tag = (
            id(_bind_draft_klib),
            int(self.l78_build_dir.stat().st_mtime_ns) if self.l78_build_dir.exists() else 0,
        )
        return klibs_by_ver

    def _build_v3_entries(self):
        import torch

        klibs_by_ver = self._ensure_l78_klibs()
        need_ver = L78_VARIANT_SPECS[self.k_sfa][0]
        klib, _, fn_i, fn_s_rollout, fn_kvc = klibs_by_ver[need_ver]
        fn_exact = getattr(klib, f"layer_tp_tree_set_exact_idx_ptrs_K{need_ver}", None)
        if fn_exact is not None:
            fn_exact(
                int(self.wk_w_128.data_ptr()),
                int(self.dw.front[3].data_ptr()),
                int(self.dw.front[4].data_ptr()),
                int(self.prefix.draft_cache["index_keys"].data_ptr()),
                int(self.exact_idx_work.data_ptr()),
                float(K_NORM_EPS),
            )

        dev = self.dev
        dw = self.dw
        pages = int(self.prefix.pages)
        table_pages = int(self.prefix.block_table.shape[1])
        raw_mt = self.prefix.block_table
        raw_it = self.prefix.block_table

        cal_input_gamma = self.wt.cal_input_gamma
        cal_main = list(dw.main)
        if self.wt.cal_uq_qr_q8_nz is not None:
            cal_main[1] = self.wt.cal_uq_qr_q8_nz
            cal_main[5] = self.wt.cal_uq_qr_scale
        cal_main[2] = self.wt.cal_w_uk_t
        cal_main[7] = self.wt.cal_q_a_norm

        cal_tail = list(dw.tail)
        cal_tail[0] = self.wt.cal_w_uv_packed
        if self.wt.cal_w_o_packed is not None:
            cal_tail[1] = self.wt.cal_w_o_packed
            cal_tail[2] = self.wt.cal_w_o_scale

        cal_post = [self.wt.cal_post_gamma, self.wt.cal_router_w_packed, dw.post[2]]
        cal_routed = list(dw.routed_folded)
        cal_shared = (
            list(self.wt.cal_shared_packed)
            if self.wt.cal_shared_packed is not None
            else list(dw.shared)
        )

        caches_idx = [
            self.prefix.views["mla_latent.78"],
            self.prefix.views["mla_rope.78"],
            self.prefix.views["index.78"],
        ]
        caches_shr = [
            self.prefix.views["mla_latent.78"],
            self.prefix.views["mla_rope.78"],
        ]

        def empty(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=dev)

        def make_ws(m_rows: int, indexed: bool):
            ws = _DraftEntry()
            ws.m = m_rows
            ws.indexed = indexed
            ws.hs_in_full = empty((32, HIDDEN_SIZE), torch.bfloat16)
            ws.hs_in = ws.hs_in_full[:m_rows]
            ws.zero_res = empty((m_rows, HIDDEN_SIZE), torch.bfloat16)
            ws.norm_outputs = [
                empty((m_rows, HIDDEN_SIZE), torch.bfloat16),
                empty((m_rows, HIDDEN_SIZE), torch.bfloat16),
                empty((m_rows,), torch.float32),
                empty((m_rows, HIDDEN_SIZE), torch.int8),
                empty((m_rows, 1), torch.float32),
            ]
            ws.sin_buf = empty((m_rows, 64), torch.bfloat16)
            ws.cos_buf = empty((m_rows, 64), torch.bfloat16)
            ws.slot_i64 = empty((m_rows,), torch.int64)
            ws.slot_i64.copy_(torch.arange(9250, 9250 + m_rows, dtype=torch.int64, device=dev))
            ws.kv_len = torch.zeros((1,), dtype=torch.int32, device=dev)
            if not indexed:
                ws.scratch, ws.tiling, blocks, ws.sync = (
                    torch.ops.tilert.shared_full_attention_prepare(
                        ws.norm_outputs[3], pages, table_pages, 24
                    )
                )
            else:
                ws.scratch, ws.tiling, _, blocks, ws.sync = (
                    torch.ops.tilert.indexed_attention_prepare(
                        ws.norm_outputs[3], pages, table_pages, pages, table_pages, False, 24
                    )
                )
            ws.partial = empty((m_rows, HIDDEN_SIZE), torch.bfloat16)
            ws.bridge, ws.body, sync_f = self.ffn_ops.prepare(ws.partial, 24, 1)
            if blocks != 24 or ws.sync != sync_f:
                raise RuntimeError("Draft entry FFN workspace mismatch")
            ws.attention_sum = empty((m_rows, HIDDEN_SIZE), torch.bfloat16)
            ws.output = empty((m_rows, HIDDEN_SIZE), torch.bfloat16)
            return ws

        def bind_entry(
            ws,
            indexed: bool,
            fn_launch,
            use_cal: bool,
            q_len_tensor=None,
            custom_topk=None,
        ):
            em = copy.copy(ws)
            if q_len_tensor is None:
                _, q_u8 = self._alloc_bytes(64, 64, zero=False)
                em.q_len = q_u8[:4].view(torch.int32)
                em.q_len.fill_(ws.m)
            else:
                em.q_len = q_len_tensor
            main_meta = [em.sin_buf, em.cos_buf, em.slot_i64, em.q_len, em.kv_len, raw_mt]
            idx_meta = (
                [em.sin_buf, em.cos_buf, em.slot_i64, em.q_len, em.kv_len, raw_it]
                if indexed
                else None
            )
            ig_use = cal_input_gamma if use_cal else dw.input_gamma
            mn_use = cal_main if use_cal else list(dw.main)
            tl_use = cal_tail if use_cal else list(dw.tail)
            ps_use = cal_post if use_cal else list(dw.post)
            rt_use = cal_routed if use_cal else list(dw.routed)
            sh_use = cal_shared if use_cal else list(dw.shared)
            caches_use = caches_idx if indexed else caches_shr

            norm = [em.hs_in, em.zero_res, ig_use, *em.norm_outputs]
            caches = [*caches_use, *([None] * (4 - len(caches_use)))]
            front = list(dw.front) if indexed else [None] * 6
            addrs = [
                *norm[3:4], *norm[6:8], *mn_use,
                *front, *tl_use, None,
                *main_meta, *(idx_meta or [None] * 6), *caches,
                em.scratch, em.tiling, em.partial,
            ]
            dims = (
                ws.m,
                pages,
                table_pages,
                pages,
                table_pages,
                24,
                0,
                int(indexed and bool(dw.front) and dw.front[1].dtype == torch.bfloat16),
                0,
            )
            groups = (norm, addrs, ps_use, rt_use, em.bridge, em.body, sh_use)
            shared_topk = (
                custom_topk
                if custom_topk is not None
                else self.draft_indices_buf[: ws.m].view(ws.m, 1, 2048)
            )
            em.local = LocalRecords(
                self.metadata_lib,
                *groups,
                shared_topk,
                dims,
                dw.scalars,
                em.sync,
                indexed,
                prepare_fn=getattr(
                    klib, f"layer_tp_tree_prepare_records_K{need_ver}", None
                ),
            )
            comm_raw = struct.pack(
                "<QQQQqii",
                self.l78_win.address,
                self.l78_win.bytes,
                em.attention_sum.data_ptr(),
                em.output.data_ptr(),
                self.l78_win.capacity,
                self.l78_win.rank,
                self.l78_win.ranks,
            )
            em.comm_buf = ctypes.create_string_buffer(comm_raw, 48)
            em.fn = fn_launch
            return em

        self._ws_shr = {}
        for m_rows in sorted(set(self.step_m_rows)):
            self._ws_shr[m_rows] = make_ws(m_rows, False)
        self.eshr_steps = [
            bind_entry(
                self._ws_shr[m_rows],
                False,
                fn_s_rollout,
                True,
                q_len_tensor=self.step_qlen_dev[s : s + 1],
            )
            for s, m_rows in enumerate(self.step_m_rows)
        ]
        self._ws_idx1 = make_ws(1, True)
        self.entry_cal_idx_1 = bind_entry(self._ws_idx1, True, fn_i, True)
        self._ws_kvc = make_ws(self.COMMIT_M, True)
        self.entry_kv_commit = bind_entry(
            self._ws_kvc, True, fn_kvc, False, custom_topk=self.kv_commit_topk
        )
        torch.npu.synchronize(dev)

    def _q8_logits_rollout(self, x):
        import torch
        import torch_npu

        act_q, act_s = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
        return torch_npu.npu_quant_matmul(
            act_q,
            self.cal_head_w_q8_nz,
            self.cal_head_s8,
            pertoken_scale=act_s,
            output_dtype=torch.bfloat16,
        )

    def _exact_index_keys(self, entry):
        import torch

        stream_ptr = ctypes.c_void_p(torch.npu.current_stream(self.dev).npu_stream)
        st = self.wt.ops.tilert_dspine_exact_index_keys_launch(
            stream_ptr,
            self.sync_addr,
            entry.norm_outputs[0].data_ptr(),
            entry.q_len.data_ptr(),
            entry.slot_i64.data_ptr(),
            entry.cos_buf.data_ptr(),
            entry.sin_buf.data_ptr(),
            self.wk_w_128.data_ptr(),
            self.dw.front[3].data_ptr(),
            self.dw.front[4].data_ptr(),
            self.prefix.draft_cache["index_keys"].data_ptr(),
            self.exact_idx_work.data_ptr(),
            entry.m,
            K_NORM_EPS,
            0,
        )
        if st != 0:
            raise RuntimeError(f"dspine_exact_index_keys failed: {st}")

    def emit_prolog_prepare_meta(self, state_dev, receipt_ring_dev):
        import torch

        stream_ptr = ctypes.c_void_p(torch.npu.current_stream(self.dev).npu_stream)
        st = self.wt.ops.tilert_dspine_prolog_prepare_meta_launch(
            stream_ptr,
            state_dev.data_ptr(),
            self.draft_ctrl_dev.data_ptr(),
            self.compact_ctrl_dev.data_ptr(),
            self.commit_ctrl_dev.data_ptr(),
            self.st["dyn_w"].data_ptr(),
            self.st["mc2_mask_ptr"],
            self.step_qlen_dev.data_ptr(),
            self.step_qlen_ptrs_dev.data_ptr(),
            self.step_table_dev.data_ptr(),
            receipt_ring_dev.data_ptr(),
            self.commit_qlen_ptrs_dev.data_ptr(),
            self.w,
            self.max_steps,
        )
        if st != 0:
            raise RuntimeError(f"prolog_prepare_meta failed: {st}")
        torch.add(self.step_qlen_dev, self.step_qlen_dev, out=self.step_2qlen_dev)

    def emit_draft_stage(self):
        import torch
        import torch_npu

        b0 = self.b0
        eshr0 = self.eshr_steps[0]
        stream_ptr = ctypes.c_void_p(torch.npu.current_stream(self.dev).npu_stream)
        flushed = self.wt.ops.tilert_get_flushed_stream
        sin_ptr = self.arena.sin_cache.data_ptr()
        cos_ptr = self.arena.cos_cache.data_ptr()

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
            self.draft_indices_buf.data_ptr(),
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
        for s in range(self.max_steps):
            d = s + 1
            m_rows = self.step_m_rows[s]
            s_idx = self.wt.step_idx(d)
            eshr = self.eshr_steps[s]
            a_rec = self.active_recs[:m_rows]
            r_part = self.rot_part[:m_rows]
            e_in = self.eh_in[:m_rows]
            e_comm = self.eh_comm_in[:m_rows]
            ql_ptr = self.step_qlen_dev[s : s + 1].data_ptr()

            torch.matmul(a_rec, self.wt.rot_w_step[s_idx], out=r_part)
            st = self.wt.ops.tilert_step3c_prolog_pack_guarded_launch(
                stream_ptr,
                ql_ptr,
                self.active_toks.data_ptr(),
                self.emb_w.data_ptr(),
                r_part.data_ptr(),
                self.wt.rot_b_step[s_idx].data_ptr(),
                self.prolog_comm_in.data_ptr(),
                m_rows,
                self.emb_vocab_start,
                self.emb_vocab_end,
                self.rank,
            )
            if st != 0:
                raise RuntimeError(f"prolog_pack_guarded failed: {st}")
            torch.ops.tilert.onesided_all_reduce_out_guarded(
                self.prolog_comm_in, self.prolog_comm_out, self.step_2qlen_dev[s : s + 1]
            )
            st = self.wt.ops.tilert_step3c_dual_rmsnorm_slice_guarded_launch(
                stream_ptr,
                ql_ptr,
                self.prolog_comm_out.data_ptr(),
                self.wt.enorm_step[s_idx].data_ptr(),
                self.wt.hnorm_step[s_idx].data_ptr(),
                e_in.data_ptr(),
                m_rows,
                self.rank,
                self.norm_eps,
            )
            if st != 0:
                raise RuntimeError(f"dual_rmsnorm_slice_guarded failed: {st}")
            torch.matmul(e_in, self.wt.w_eh_step[s_idx], out=e_comm)
            torch.ops.tilert.onesided_all_reduce_out_guarded(
                self.eh_comm_in, eshr.hs_in_full, self.step_qlen_dev[s : s + 1]
            )

            st_eshr = eshr.fn(
                flushed(),
                ctypes.c_uint64(eshr.local.sync),
                eshr.local.config,
                eshr.local.addresses,
                eshr.comm_buf,
            )
            if st_eshr != 0:
                raise RuntimeError("draft rollout kernel failed")

            recs_d = self.c2b_recs_out[:m_rows]
            k_top = max(2, m_rows) if self.narrow_topk else 4
            st = self.wt.ops.tilert_dspine_draft_head_topk_launch(
                stream_ptr,
                self.sync_addr,
                ql_ptr,
                eshr.bridge[1].data_ptr(),
                eshr.output.data_ptr(),
                self.wt.logit_norm_step[s_idx].data_ptr(),
                self.cal_head_w_q8_nz.data_ptr(),
                self.cal_head_s8.data_ptr(),
                recs_d.data_ptr(),
                self.c2b_act_q.data_ptr(),
                self.c2b_act_s.data_ptr(),
                self.c2b_acc_i32.data_ptr(),
                self.logits_work.data_ptr(),
                self.gather_in.data_ptr(),
                self.gather_out.data_ptr(),
                self.comm_win,
                m_rows,
                self.head_valid_vocab,
                self.head_vocab_start,
                k_top,
                self.rank,
                self.world_size,
                self.norm_eps,
            )
            if st != 0:
                raise RuntimeError(f"dspine_draft_head_topk failed: {st}")

            nxt_b = self.step_m_rows[s + 1] if (s + 1) < self.max_steps else 0
            eshr_nxt = self.eshr_steps[s + 1] if (s + 1) < self.max_steps else eshr
            st = self.wt.ops.tilert_dspine_beam_step_v3_launch(
                stream_ptr,
                self.sync_addr,
                d,
                m_rows,
                m_rows,
                k_top,
                nxt_b,
                0,
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
                self.draft_indices_buf.data_ptr(),
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

        torch.ops.tilert.set_active_tokens_ptr(self.st["dyn_w"], True, True)
        st = self.wt.ops.tilert_dspine_build_tree_v3_launch(
            stream_ptr,
            self.sync_addr,
            self.w,
            self.min_w_act,
            self.max_pool_size,
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
            self.pool_depths_dev.data_ptr(),
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
        torch.ops.tilert.set_active_tokens_ptr(self.st["dyn_w"], True, True)

    def emit_resolve_and_advance(
        self,
        state_dev,
        receipt_ring_dev,
        receipt_toks_dev,
        max_pos: int,
        eos_token_id: int = -1,
    ):
        import torch

        stream_ptr = ctypes.c_void_p(torch.npu.current_stream(self.dev).npu_stream)
        hc = self.horizon_ctrl
        st = self.wt.ops.tilert_dspine_resolve_and_advance_launch(
            stream_ptr,
            state_dev.data_ptr(),
            self.st["dyn_w"].data_ptr(),
            self.tree_tokens.data_ptr(),
            self.tree_parents.data_ptr(),
            self.st["target_tokens"].data_ptr(),
            self.compact_ctrl_dev.data_ptr(),
            self.commit_ctrl_dev.data_ptr(),
            receipt_ring_dev.data_ptr(),
            receipt_toks_dev.data_ptr(),
            self.w,
            self.COMMIT_M,
            int(max_pos),
            int(eos_token_id),
            float(hc.alpha),
            float(hc.mid_thresh),
            float(hc.up_thresh),
            float(hc.down_thresh),
            int(hc.streak_acc),
            int(hc.drop_acc),
            int(hc.jump_acc),
            int(hc.num_levels),
        )
        if st != 0:
            raise RuntimeError(f"resolve_and_advance failed: {st}")

    def emit_compact_and_commit_seed(self, *, include_compact: bool = True):
        import torch
        import torch_npu

        cm = self.COMMIT_M
        ekv = self.entry_kv_commit
        e1 = self.entry_cal_idx_1
        main_stream = torch.npu.current_stream(self.dev)
        stream_ptr = ctypes.c_void_p(main_stream.npu_stream)
        flushed = self.wt.ops.tilert_get_flushed_stream
        sin_ptr = self.arena.sin_cache.data_ptr()
        cos_ptr = self.arena.cos_cache.data_ptr()

        if include_compact:
            self._ev_cmp_fork.record(main_stream)
            self.compact_stream.wait_event(self._ev_cmp_fork)
            with torch.npu.stream(self.compact_stream):
                cmp_ptr = ctypes.c_void_p(self.compact_stream.npu_stream)
                st = self.wt.ops.tilert_kv_compact_launch(
                    cmp_ptr,
                    self.prefix.descs_dev.data_ptr(),
                    self.compact_ctrl_dev.data_ptr(),
                    self.prefix.num_cache_tensors,
                )
            if st != 0:
                raise RuntimeError(f"kv_compact failed: {st}")
            self._ev_cmp_join.record(self.compact_stream)

        self.commit_ttoks_pad32[: self.w].copy_(self.st["target_tokens"][: self.w])
        row_idx = self.commit_ctrl_dev[3 : 3 + cm]
        par_m = self.st["target_hidden"].index_select(0, row_idx)
        for off in range(0, cm, 8):
            torch.mm(
                par_m[off : off + 8],
                self.rot_shard_t,
                out=self.commit_rot_shard[off : off + 8],
            )
        st = self.wt.ops.tilert_dspine_prolog_pack_compact_launch(
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
            raise RuntimeError(f"dspine_prolog_pack_compact failed: {st}")

        torch.ops.tilert.onesided_all_reduce_out_guarded(
            self.commit_comm_in[: 2 * cm],
            self.commit_comm_out[: 2 * cm],
            self.commit_2acc_i32,
        )
        en = torch_npu.npu_rms_norm(
            self.commit_comm_out[:cm], self.enorm_w, self.norm_eps
        )[0]
        hn = torch_npu.npu_rms_norm(
            self.commit_comm_out[cm : 2 * cm], self.hnorm_w, self.norm_eps
        )[0]
        self.commit_eh_in[:, :HIDDEN_SIZE].copy_(en)
        self.commit_eh_in[:, HIDDEN_SIZE:].copy_(hn)
        for off in range(0, cm, 8):
            torch.mm(
                self.commit_eh_in[off : off + 8],
                self.w_eh_col_r,
                out=self.commit_eh_shard[off : off + 8],
            )
        sl_s = self.rank * self.shard_dim
        self.commit_eh_comm[:cm, sl_s : sl_s + self.shard_dim].copy_(self.commit_eh_shard)
        torch.ops.tilert.onesided_all_reduce_out_guarded(
            self.commit_eh_comm, self.commit_hs_all, self.commit_acc_i32
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

        if include_compact:
            main_stream.wait_event(self._ev_cmp_join)
        st_ekv = ekv.fn(
            flushed(),
            ctypes.c_uint64(ekv.local.sync),
            ekv.local.config,
            ekv.local.addresses,
            ekv.comm_buf,
        )
        if st_ekv != 0:
            raise RuntimeError("kv_commit kernel failed")
        st_e1 = e1.fn(
            flushed(),
            ctypes.c_uint64(e1.local.sync),
            e1.local.config,
            e1.local.addresses,
            e1.comm_buf,
        )
        if st_e1 != 0:
            raise RuntimeError("seed e1 kernel failed")

        s0_gamma = (
            self.wt.logit_norm_seed0
            if self.wt.logit_norm_seed0 is not None
            else self.dw.shared_head_norm_w
        )
        k_seed = max(2, self.b0) if self.narrow_topk else 4
        st = self.wt.ops.tilert_dspine_draft_head_topk_launch(
            stream_ptr,
            self.sync_addr,
            e1.q_len.data_ptr(),
            e1.bridge[1].data_ptr(),
            e1.output.data_ptr(),
            s0_gamma.data_ptr(),
            self.cal_head_w_q8_nz.data_ptr(),
            self.cal_head_s8.data_ptr(),
            self.seed_rec_buf.data_ptr(),
            self.c2b_act_q.data_ptr(),
            self.c2b_act_s.data_ptr(),
            self.c2b_acc_i32.data_ptr(),
            self.logits_work.data_ptr(),
            self.gather_in.data_ptr(),
            self.gather_out.data_ptr(),
            self.comm_win,
            1,
            self.head_valid_vocab,
            self.head_vocab_start,
            k_seed,
            self.rank,
            self.world_size,
            self.norm_eps,
        )
        if st != 0:
            raise RuntimeError(f"dspine_draft_head_topk seed0 failed: {st}")
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
            self.draft_ctrl_dev.data_ptr(),
            self.seed_top_i.data_ptr(),
            self.seed_top_lp.data_ptr(),
            0,
            self.prefix_topk_buf.data_ptr(),
            0,
            self.draft_indices_buf.data_ptr(),
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            self.beam_work.data_ptr(),
        )
        if st != 0:
            raise RuntimeError(f"beam_step_v3 seed0 failed: {st}")

    def seed_round0_eager(self):
        import torch

        with torch.npu.device(self.dev), torch.inference_mode():
            self.draft_ctrl_dev.zero_()
            self.draft_ctrl_dev[0] = self.first_token
            self.draft_ctrl_dev[1] = 8192
            self.draft_ctrl_dev[2] = self.w
            self.commit_ctrl_dev.zero_()
            self.commit_ctrl_dev[0] = 8191
            self.commit_ctrl_dev[1] = 1
            self.commit_ctrl_dev[2] = 1
            self.commit_ctrl_dev[47] = self.COMMIT_M + 1
            self.st["target_tokens"][:2].fill_(self.first_token)
            self.st["target_hidden"][:2].copy_(self.prefix.parent_hidden.to(self.dev).expand(2, -1))
            self.emit_compact_and_commit_seed(include_compact=False)
            torch.npu.synchronize(self.dev)

    def run_serial_commit_pass(
        self,
        tokens: list[int],
        hidden_seq,
        pos_limit: int,
        seed_positions: list[int],
        prev_tail: dict,
    ) -> tuple[dict, dict[str, float], dict[str, float]]:
        import torch
        import torch_npu

        cm = self.COMMIT_M
        ekv = self.entry_kv_commit
        e1 = self.entry_cal_idx_1
        parent_h = self.prefix.parent_hidden.to(self.dev)
        total_need = pos_limit - 8191
        sin_ptr = self.arena.sin_cache.data_ptr()
        cos_ptr = self.arena.cos_cache.data_ptr()
        stream_ptr = ctypes.c_void_p(torch.npu.current_stream(self.dev).npu_stream)
        flushed = self.wt.ops.tilert_get_flushed_stream

        def run_single(entry, pos: int, tok: int, hid):
            self.commit_ctrl_dev.zero_()
            self.commit_ctrl_dev[0] = pos
            self.commit_ctrl_dev[1] = 1
            self.commit_ctrl_dev[2] = 1
            self.commit_ctrl_dev[47] = cm + 1
            self.commit_ttoks_pad32[0] = int(tok)
            torch.mm(hid, self.rot_shard_t, out=self.commit_rot_shard[:1])
            self.wt.ops.tilert_dspine_prolog_pack_launch(
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
            torch.ops.tilert.set_active_tokens_ptr(self.commit_acc_i32, True, False)
            torch.ops.tilert.onesided_all_reduce_out(
                self.commit_comm_in[: 2 * cm], self.commit_comm_out[: 2 * cm]
            )
            torch.ops.tilert.onesided_all_reduce_out(
                self.commit_comm_in[2 * cm : 4 * cm], self.commit_comm_out[2 * cm : 4 * cm]
            )
            en = torch_npu.npu_rms_norm(
                self.commit_comm_out[:cm], self.enorm_w, self.norm_eps
            )[0]
            hn = torch_npu.npu_rms_norm(
                self.commit_comm_out[2 * cm : 3 * cm], self.hnorm_w, self.norm_eps
            )[0]
            self.commit_eh_in[:, :HIDDEN_SIZE].copy_(en)
            self.commit_eh_in[:, HIDDEN_SIZE:].copy_(hn)
            torch.mm(self.commit_eh_in[:1], self.w_eh_col_r, out=self.commit_eh_shard[:1])
            sl_s = self.rank * self.shard_dim
            self.commit_eh_comm[:cm, sl_s : sl_s + self.shard_dim].copy_(
                self.commit_eh_shard
            )
            torch.ops.tilert.onesided_all_reduce_out(
                self.commit_eh_comm, self.commit_hs_all
            )
            torch.ops.tilert.set_active_tokens_ptr(self.st["dyn_w"], True, True)
            entry.hs_in[:1].copy_(self.commit_hs_all[:1])
            entry.sin_buf[:1].copy_(self.arena.sin_cache[pos : pos + 1])
            entry.cos_buf[:1].copy_(self.arena.cos_cache[pos : pos + 1])
            entry.slot_i64[:1].fill_(pos)
            entry.q_len.fill_(1)
            entry.kv_len.fill_(pos + 1)
            st_k = entry.fn(
                flushed(),
                ctypes.c_uint64(entry.local.sync),
                entry.local.config,
                entry.local.addresses,
                entry.comm_buf,
            )
            if st_k != 0:
                raise RuntimeError("serial commit kernel failed")

        with torch.npu.device(self.dev), torch.inference_mode():
            for i in range(total_need):
                pos = 8191 + i
                hid = parent_h if i == 0 else hidden_seq[i - 1 : i]
                run_single(ekv, pos, tokens[i], hid)
            torch.npu.synchronize(self.dev)
            seed_set = set(seed_positions)
            ekv_idx = [p - 8191 for p in range(8191, pos_limit) if p not in seed_set]
            if ekv_idx:
                ekv_raw = {
                    k: float(
                        (
                            v[8191:pos_limit].cpu()[ekv_idx].float()
                            - prev_tail[k][ekv_idx].float()
                        )
                        .abs()
                        .max()
                        .item()
                    )
                    for k, v in self.prefix.draft_cache.items()
                }
                ekv_diffs = {k: d for k, d in ekv_raw.items() if math.isnan(d) or d > 0.0}
            else:
                ekv_diffs = {}
            for pos in sorted(seed_set):
                if not 8191 <= pos < pos_limit:
                    continue
                i = pos - 8191
                hid = parent_h if i == 0 else hidden_seq[i - 1 : i]
                run_single(e1, pos, tokens[i], hid)
            torch.npu.synchronize(self.dev)
            diffs_raw = {
                k: float(
                    (v[8191:pos_limit].cpu().float() - prev_tail[k].float())
                    .abs()
                    .max()
                    .item()
                )
                for k, v in self.prefix.draft_cache.items()
            }
            diffs = {k: d for k, d in diffs_raw.items() if math.isnan(d) or d > 0.0}
            return self.prefix.draft_state(pos_limit), diffs, ekv_diffs

    def close(self):
        if getattr(self, "_owns_comm_win", False) and getattr(self, "comm_win", 0) != 0:
            self.wt.ops.tilert_dspine_free_comm_win(self.comm_win)
            self.comm_win = 0
            self._owns_comm_win = False
        self.eshr_steps = []
        self.entry_cal_idx_1 = None
        self.entry_kv_commit = None
        self._owners.clear()
        gc.collect()


def torch_dist_ready() -> bool:
    try:
        import torch

        return torch.distributed.is_initialized()
    except Exception:
        return False
