from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import torch
from torch import nn

from ..weights import pack_output_weights, pack_shared_weights, pack_value_weights
from .build_dspine_ops import build_dspine_ops

try:
    import torch_npu
except ImportError:
    torch_npu = None


def pack_router_weight(router_w: torch.Tensor) -> torch.Tensor:
    h, num_experts = int(router_w.shape[0]), int(router_w.shape[1])
    if num_experts % 16 != 0:
        raise ValueError("Router experts must be divisible by 16")
    return (
        router_w.reshape(h, 16, num_experts // 16)
        .permute(1, 0, 2)
        .contiguous()
        .view(h, num_experts)
    )


def quantize_linear_col_i8(weight_f32_kn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (
        (weight_f32_kn.abs().amax(dim=0) / 127.0)
        .clamp(min=1e-8)
        .to(torch.float32)
        .contiguous()
    )
    q8 = (
        torch.clamp(torch.round(weight_f32_kn / scale.unsqueeze(0)), -127, 127)
        .to(torch.int8)
        .contiguous()
    )
    return q8, scale


def build_beam_specs(
    b_sched: list[int] | tuple[int, ...],
) -> tuple[list[tuple[int, int, int, int, int]], list[int]]:
    sched = tuple(int(x) for x in b_sched)
    depths = [0] + [1] * sched[0]
    specs: list[tuple[int, int, int, int, int]] = []
    for si, cur_b in enumerate(sched):
        nxt_b = sched[si + 1] if si + 1 < len(sched) else 0
        k_exp = max(cur_b, nxt_b) if nxt_b > 0 else cur_b
        p_off = len(depths)
        specs.append((si + 1, cur_b, k_exp, nxt_b, p_off))
        depths.extend([si + 2] * (cur_b * k_exp))
    return specs, depths


class HorizonController:
    def __init__(
        self,
        init_ema: float = 6.8,
        alpha: float = 0.20,
        up_thresh: float = 7.30,
        down_thresh: float = 7.10,
    ):
        self.init_ema = float(init_ema)
        self.alpha = float(alpha)
        self.up_thresh = float(up_thresh)
        self.down_thresh = float(down_thresh)
        self.ema = self.init_ema
        self.high = False

    def reset(self) -> None:
        self.ema = self.init_ema
        self.high = False

    def select_high(self) -> bool:
        return self.high

    def update(self, accepted: int) -> bool:
        self.ema = (1.0 - self.alpha) * self.ema + self.alpha * float(accepted)
        if self.high:
            if self.ema < self.down_thresh:
                self.high = False
        elif self.ema >= self.up_thresh:
            self.high = True
        return self.high


class ZeroLagHorizonController:
    def __init__(
        self,
        num_levels: int = 3,
        init_level: int = 1,
        init_ema: float = 6.2,
        alpha: float = 0.28,
        mid_thresh: float = 5.40,
        up_thresh: float = 7.15,
        down_thresh: float = 4.90,
        drop_acc: int = 3,
        jump_acc: int = 9,
        streak_acc: int = 7,
    ):
        self.num_levels = max(2, int(num_levels))
        self.init_level = min(max(int(init_level), 0), self.num_levels - 1)
        self.init_ema = float(init_ema)
        self.alpha = float(alpha)
        self.mid_thresh = float(mid_thresh)
        self.up_thresh = float(up_thresh)
        self.down_thresh = float(down_thresh)
        self.drop_acc = int(drop_acc)
        self.jump_acc = int(jump_acc)
        self.streak_acc = int(streak_acc)
        self.ema = self.init_ema
        self.level = self.init_level
        self.hi_streak = 0
        self.lo_streak = 0

    def reset(self) -> None:
        self.ema = self.init_ema
        self.level = self.init_level
        self.hi_streak = 0
        self.lo_streak = 0

    def select_level(self) -> int:
        return self.level

    def select_high(self) -> bool:
        return self.level > 0

    def update(self, accepted: int) -> int:
        acc_f = float(accepted)
        self.ema = (1.0 - self.alpha) * self.ema + self.alpha * acc_f
        if accepted >= self.streak_acc:
            self.hi_streak += 1
            self.lo_streak = 0
        elif accepted <= self.drop_acc:
            self.lo_streak += 1
            self.hi_streak = 0
        else:
            self.hi_streak = 0
            self.lo_streak = 0

        top = self.num_levels - 1
        if accepted <= 2 or self.lo_streak >= 2:
            self.level = 0
            self.ema = min(self.ema, self.down_thresh)
        elif accepted >= self.jump_acc or self.hi_streak >= 2:
            self.level = top
            self.ema = max(self.ema, self.up_thresh)
        elif self.num_levels >= 3:
            if self.ema >= self.up_thresh:
                self.level = 2
            elif self.ema >= self.mid_thresh:
                self.level = 1
            elif self.ema < self.down_thresh:
                self.level = 0
        else:
            if self.ema >= self.up_thresh:
                self.level = 1
            elif self.ema < self.down_thresh:
                self.level = 0
        return self.level


def select_optimal_w_act(
    raw_lps: list[float] | torch.Tensor,
    width: int,
    min_w_act: int = 8,
    t_base_ms: float = 47.0,
    c_w_ms: float = 0.72,
) -> int:
    if min_w_act <= 0 or min_w_act >= width:
        return int(width)
    vals = raw_lps.tolist() if isinstance(raw_lps, torch.Tensor) else list(raw_lps)
    e_tau = 0.0
    best_tpot = 1.0e30
    best_w = int(width)
    for r in range(int(width)):
        if r == 0:
            p_r = 1.0
        else:
            rlp = min(max(float(vals[r]), -20.0), 0.0)
            p_r = math.exp(rlp)
        e_tau += p_r
        cand_w = r + 1
        if cand_w >= min_w_act and ((cand_w & 1) == 0 or cand_w == width):
            est_tpot = (float(t_base_ms) + float(c_w_ms) * float(cand_w)) / e_tau
            if est_tpot <= best_tpot:
                best_tpot = est_tpot
                best_w = cand_w
    return best_w


@dataclass(frozen=True)
class CalibratedL78Config:
    hidden_size: int = 6144
    q_lora_rank: int = 2048
    num_local_heads: int = 8
    qk_nope_dim: int = 192
    kv_lora_rank: int = 512
    v_head_dim: int = 256
    num_experts: int = 256
    eh_local_rows: int = 1536
    rot_local_cols: int = 768
    max_steps: int = 8
    rot_rank: int = 256
    eh_rank: int = 256
    kv_rank: int = 32
    router_rank: int = 64
    head_rank: int = 384
    rms_eps: float = 1e-5

    def to_dict(self) -> dict:
        return asdict(self)


class CalibratedL78Model(nn.Module):
    def __init__(self, cfg: CalibratedL78Config | None = None):
        super().__init__()
        self.cfg = cfg or CalibratedL78Config()
        c = self.cfg
        h = c.hidden_size
        d = c.max_steps

        self.rot_scale = nn.Parameter(torch.ones(d, h))
        self.rot_u = nn.Parameter(torch.zeros(d, h, c.rot_rank))
        self.rot_v = nn.Parameter(torch.randn(d, c.rot_rank, h) * (1.0 / math.sqrt(h)))
        self.rot_bias = nn.Parameter(torch.zeros(d, h))

        self.enorm_step = nn.Parameter(torch.ones(d, h))
        self.hnorm_step = nn.Parameter(torch.ones(d, h))

        self.eh_u = nn.Parameter(torch.zeros(d, c.eh_local_rows, c.eh_rank))
        self.eh_v = nn.Parameter(torch.randn(d, c.eh_rank, h) * (1.0 / math.sqrt(h)))

        self.cal_input_gamma = nn.Parameter(torch.ones(h))
        self.cal_q_a_norm = nn.Parameter(torch.ones(c.q_lora_rank))
        self.cal_post_gamma = nn.Parameter(torch.ones(h))

        self.uk_u = nn.Parameter(torch.zeros(c.num_local_heads, c.qk_nope_dim, c.kv_rank))
        self.uk_v = nn.Parameter(
            torch.randn(c.num_local_heads, c.kv_rank, c.kv_lora_rank)
            * (1.0 / math.sqrt(c.qk_nope_dim))
        )
        self.uv_u = nn.Parameter(torch.zeros(c.num_local_heads, c.kv_lora_rank, c.kv_rank))
        self.uv_v = nn.Parameter(
            torch.randn(c.num_local_heads, c.kv_rank, c.v_head_dim)
            * (1.0 / math.sqrt(c.kv_lora_rank))
        )

        self.rt_u = nn.Parameter(torch.zeros(h, c.router_rank))
        self.rt_v = nn.Parameter(torch.randn(c.router_rank, c.num_experts) * (1.0 / math.sqrt(h)))

        self.logit_norm_step = nn.Parameter(torch.ones(d, h))
        self.head_scale = nn.Parameter(torch.ones(h))
        self.head_p = nn.Parameter(torch.zeros(h, c.head_rank))
        self.head_q = nn.Parameter(torch.randn(c.head_rank, h) * (1.0 / math.sqrt(h)))

    def step_index(self, step: int) -> int:
        return min(max(step, 0), self.cfg.max_steps - 1)

    def fold_step_prolog(
        self,
        step: int,
        rot_col_r: torch.Tensor,
        w_eh_row_r: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.step_index(step)
        c = self.cfg
        rc_f = rot_col_r.to(torch.float32)
        base_rot = self.rot_scale[idx].to(torch.float32).unsqueeze(1) * rc_f
        delta_rot = (16.0 / c.rot_rank) * (
            self.rot_u[idx].to(torch.float32) @ (self.rot_v[idx].to(torch.float32) @ rc_f)
        )
        rot_w_d = (base_rot + delta_rot).to(torch.bfloat16).contiguous()
        rot_b_d = (
            (self.rot_bias[idx].to(torch.float32).unsqueeze(0) @ rc_f)
            .squeeze(0)
            .to(torch.bfloat16)
            .contiguous()
        )
        w_eh_d = (
            w_eh_row_r.to(torch.float32)
            + (16.0 / c.eh_rank)
            * (self.eh_u[idx].to(torch.float32) @ self.eh_v[idx].to(torch.float32))
        ).to(torch.bfloat16).contiguous()
        en_d = self.enorm_step[idx].to(torch.bfloat16).contiguous()
        hn_d = self.hnorm_step[idx].to(torch.bfloat16).contiguous()
        return rot_w_d, rot_b_d, en_d, hn_d, w_eh_d

    def fold_layer_weights(
        self,
        w_uk_t0: torch.Tensor,
        w_uv0: torch.Tensor,
        router_w0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.cfg
        w_uk = (
            w_uk_t0.to(torch.float32)
            + (4.0 / c.kv_rank) * torch.bmm(self.uk_u.float(), self.uk_v.float())
        ).to(torch.bfloat16).contiguous()
        w_uv = (
            w_uv0.to(torch.float32)
            + (4.0 / c.kv_rank) * torch.bmm(self.uv_u.float(), self.uv_v.float())
        ).to(torch.bfloat16).contiguous()
        r_w = (
            router_w0.to(torch.float32)
            + (8.0 / c.router_rank) * (self.rt_u.float() @ self.rt_v.float())
        ).to(torch.bfloat16).contiguous()
        return w_uk, w_uv, r_w

    def head_matrix_f32(self) -> torch.Tensor:
        c = self.cfg
        return (
            torch.diag(self.head_scale.to(torch.float32))
            + (16.0 / c.head_rank) * (self.head_p.to(torch.float32) @ self.head_q.to(torch.float32))
        )

    def apply_head_proj(self, recs: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        rf = recs.to(torch.float32)
        out = rf * self.head_scale.to(torch.float32).unsqueeze(0) + (16.0 / c.head_rank) * (
            (rf @ self.head_p.to(torch.float32)) @ self.head_q.to(torch.float32)
        )
        return out.to(recs.dtype)


class TpCalibratedL78Weights:
    def __init__(
        self,
        ckpt_blob: dict,
        rank: int,
        dev: torch.device,
        ops_so_path: str | Path | None = None,
    ):
        self.rank = int(rank)
        self.dev = dev
        self.version = int(ckpt_blob.get("version", 2))
        self.max_d = int(ckpt_blob["max_d"])
        self.metrics = ckpt_blob.get("metrics", {})
        tree_calib = ckpt_blob.get("tree_calib", {})
        self.beta_margin = float(tree_calib.get("beta_margin", 0.0))
        self.beta_depth = float(tree_calib.get("beta_depth", 0.0))
        self.ops = None
        if ops_so_path is not None:
            from .tp_drafter import _load_ops_cdll

            so_p = Path(ops_so_path)
            if not so_p.exists():
                if self.rank == 0:
                    build_dspine_ops(so_p.parent)
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            self.ops = _load_ops_cdll(so_p, self.rank)

        sh = ckpt_blob["shards"][self.rank]
        assert int(sh["tp_rank"]) == self.rank

        self.rot_w_step = sh["rot_w_step"].to(device=dev, dtype=torch.bfloat16).contiguous()
        self.rot_b_step = sh["rot_b_step"].to(device=dev, dtype=torch.bfloat16).contiguous()
        self.w_eh_step = sh["w_eh_step"].to(device=dev, dtype=torch.bfloat16).contiguous()
        self.enorm_step = ckpt_blob["enorm_step"].to(device=dev, dtype=torch.bfloat16).contiguous()
        self.hnorm_step = ckpt_blob["hnorm_step"].to(device=dev, dtype=torch.bfloat16).contiguous()
        self.logit_norm_step = (
            ckpt_blob["logit_norm_step"].to(device=dev, dtype=torch.bfloat16).contiguous()
        )
        s0_norm = ckpt_blob.get("logit_norm_seed0")
        self.logit_norm_seed0 = (
            s0_norm.to(device=dev, dtype=torch.bfloat16).contiguous()
            if s0_norm is not None
            else None
        )

        self.cal_input_gamma = (
            ckpt_blob["cal_input_gamma"].to(device=dev, dtype=torch.bfloat16).contiguous()
        )
        self.cal_q_a_norm = (
            ckpt_blob["cal_q_a_norm"].to(device=dev, dtype=torch.bfloat16).contiguous()
        )
        self.cal_post_gamma = (
            ckpt_blob["cal_post_gamma"].to(device=dev, dtype=torch.bfloat16).contiguous()
        )
        self.cal_router_w = (
            ckpt_blob["cal_router_w"].to(device=dev, dtype=torch.bfloat16).contiguous()
        )
        self.cal_router_w_packed = pack_router_weight(self.cal_router_w)

        self.cal_w_uk_t = sh["cal_w_uk_t"].to(device=dev, dtype=torch.bfloat16).contiguous()
        self.cal_w_uv = sh["cal_w_uv"].to(device=dev, dtype=torch.bfloat16).contiguous()
        w_uv_packed_cpu = pack_value_weights(sh["cal_w_uv"].to(torch.bfloat16).cpu())
        self.cal_w_uv_packed = w_uv_packed_cpu.to(device=dev, dtype=torch.bfloat16).contiguous()

        self.cal_uq_qr_q8_nz = None
        self.cal_uq_qr_scale = None
        if "cal_w_uq_qr" in sh:
            uq_q8_cpu, uq_s_cpu = quantize_linear_col_i8(sh["cal_w_uq_qr"].to(torch.float32))
            uq_dev = uq_q8_cpu.to(device=dev).contiguous()
            if torch_npu is not None and dev.type == "npu":
                torch.npu.config.allow_internal_format = True
                uq_dev = torch_npu.npu_format_cast(uq_dev, 29)
            self.cal_uq_qr_q8_nz = uq_dev
            self.cal_uq_qr_scale = (
                uq_s_cpu.view(1, -1).to(device=dev, dtype=torch.float32).contiguous()
            )

        self.cal_w_o_packed = None
        self.cal_w_o_scale = None
        if "cal_w_o" in sh:
            wo_q8_cpu, wo_s_cpu = quantize_linear_col_i8(sh["cal_w_o"].to(torch.float32))
            self.cal_w_o_packed = pack_output_weights(wo_q8_cpu).to(device=dev).contiguous()
            self.cal_w_o_scale = wo_s_cpu.to(device=dev, dtype=torch.float32).contiguous()

        self.cal_shared_packed = None
        if "cal_shared_gu" in sh and "cal_shared_down" in sh:
            gu_q8_cpu, gu_s_cpu = quantize_linear_col_i8(sh["cal_shared_gu"].to(torch.float32))
            dn_q8_cpu, dn_s_cpu = quantize_linear_col_i8(sh["cal_shared_down"].to(torch.float32))
            p_gu_cpu, p_dn_cpu = pack_shared_weights(gu_q8_cpu, dn_q8_cpu)
            self.cal_shared_packed = [
                p_gu_cpu.to(device=dev).contiguous(),
                gu_s_cpu.to(device=dev, dtype=torch.float32).contiguous(),
                p_dn_cpu.to(device=dev).contiguous(),
                dn_s_cpu.to(device=dev, dtype=torch.float32).contiguous(),
            ]

        self.cal_routed_w13_mul = (
            sh["cal_routed_w13_mul"].to(device=dev, dtype=torch.float32).contiguous()
            if "cal_routed_w13_mul" in sh
            else None
        )
        self.cal_routed_w2_mul = (
            sh["cal_routed_w2_mul"].to(device=dev, dtype=torch.float32).contiguous()
            if "cal_routed_w2_mul" in sh
            else None
        )

        self.m_head_f32 = ckpt_blob["m_head_f32"].to(dtype=torch.float32).contiguous()

    def fold_routed_weights(self, orig_routed: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.cal_routed_w13_mul is None or self.cal_routed_w2_mul is None:
            return list(orig_routed)
        m13 = self.cal_routed_w13_mul.to(device=orig_routed[1].device, dtype=torch.float32)
        m2 = self.cal_routed_w2_mul.to(device=orig_routed[4].device, dtype=torch.float32)
        return [
            orig_routed[0],
            (orig_routed[1].to(torch.float32) * m13).contiguous(),
            (orig_routed[2].to(torch.float32) * m13).contiguous(),
            orig_routed[3],
            (orig_routed[4].to(torch.float32) * m2).contiguous(),
            (orig_routed[5].to(torch.float32) * m2).contiguous(),
        ]

    @classmethod
    def from_file(
        cls,
        ckpt_path: str | Path,
        rank: int,
        dev: torch.device,
        ops_so_path: str | Path | None = None,
    ):
        blob = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
        return cls(blob, rank, dev, ops_so_path=ops_so_path)

    def step_idx(self, d_one_based: int) -> int:
        return min(max(d_one_based - 1, 0), self.max_d - 1)

    def fold_head_shard(
        self,
        head_w_shard: torch.Tensor,
        valid_vocab: int,
        inv_temp: float = 1.0,
    ) -> torch.Tensor:
        dev_h = head_w_shard.device
        src = head_w_shard
        if torch_npu is not None and dev_h.type == "npu":
            src = torch_npu.npu_format_cast(src, 2)
        mh_t = (
            (self.m_head_f32.to(device=dev_h, dtype=torch.float32) * float(inv_temp))
            .t()
            .contiguous()
        )
        wf = torch.zeros(src.shape, device=dev_h, dtype=torch.float32)
        for st in range(0, int(valid_vocab), 2048):
            ed = min(st + 2048, int(valid_vocab))
            wf[st:ed] = src[st:ed].to(torch.float32) @ mh_t
        del mh_t
        return wf

    def quantize_head_q8_nz(
        self,
        head_w_shard: torch.Tensor,
        valid_vocab: int,
        inv_temp: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dev_h = head_w_shard.device
        src = head_w_shard
        if torch_npu is not None and dev_h.type == "npu":
            src = torch_npu.npu_format_cast(src, 2)
        mh_t = (
            (self.m_head_f32.to(device=dev_h, dtype=torch.float32) * float(inv_temp))
            .t()
            .contiguous()
        )
        n_rows, hid = int(src.shape[0]), int(src.shape[1])
        q8_t = torch.zeros((hid, n_rows), device=dev_h, dtype=torch.int8)
        s8 = torch.full((n_rows,), 1e-8, device=dev_h, dtype=torch.float32)
        for st in range(0, int(valid_vocab), 2048):
            ed = min(st + 2048, int(valid_vocab))
            blk = src[st:ed].to(torch.float32) @ mh_t
            sb = (blk.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
            s8[st:ed] = sb
            qb = torch.clamp(torch.round(blk / sb[:, None]), -127.0, 127.0).to(torch.int8)
            q8_t[:, st:ed] = qb.t()
        del mh_t
        q8_t = q8_t.contiguous()
        s8 = s8.contiguous()
        if torch_npu is not None and dev_h.type == "npu":
            torch.npu.config.allow_internal_format = True
            return torch_npu.npu_format_cast(q8_t, 29), s8
        return q8_t, s8


_KV_COMMIT_DEVICE_SNIPPET = """
__aicore__ inline void RunKvCommit(
    TPipe& pipe, const Config& config, const Addresses& addresses)
{
    SetSysWorkspace((GM_ADDR)addresses.bridge[ffn_bridge::kSystem]);
    const auto rows = addresses.body.rows;
    const auto blocks = addresses.body.blocks;
    AscendC::GlobalTensor<int32_t> q_len_gm;
    q_len_gm.SetGlobalBuffer((__gm__ int32_t*)addresses.attention.main_metadata[3], 1);
    AscendC::DataCacheCleanAndInvalid<int32_t,
        AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(q_len_gm);
    AscendC::GlobalTensor<int32_t> kv_len_gm;
    kv_len_gm.SetGlobalBuffer((__gm__ int32_t*)addresses.attention.main_metadata[4], 1);
    AscendC::DataCacheCleanAndInvalid<int32_t,
        AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(kv_len_gm);
    AscendC::GlobalTensor<int64_t> slot_gm;
    slot_gm.SetGlobalBuffer((__gm__ int64_t*)addresses.attention.main_metadata[2], rows);
    AscendC::DataCacheCleanAndInvalid<int64_t,
        AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(slot_gm[0]);
    if (rows > 8) {
        AscendC::DataCacheCleanAndInvalid<int64_t,
            AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(slot_gm[8]);
    }
    const int32_t q_len = q_len_gm.GetValue(0);
    if (q_len <= 0) {
        return;
    }
    const uint32_t active_rows = (q_len <= rows)
        ? static_cast<uint32_t>(q_len) : static_cast<uint32_t>(rows);
    const auto& norm = addresses.norm;
    whole_layer_norm::device::Forward(pipe,
        (__gm__ bfloat16_t*)norm.input, (__gm__ bfloat16_t*)norm.residual,
        (__gm__ bfloat16_t*)norm.gamma, nullptr,
        (__gm__ bfloat16_t*)norm.normalized, (__gm__ bfloat16_t*)norm.retained,
        nullptr, (__gm__ int8_t*)norm.codes,
        (__gm__ float*)norm.scales, active_rows, blocks, config.input_epsilon);

    const auto a = IndexedArguments(addresses.attention);
    GM_ADDR memory = a.scratch;
    GM_ADDR topk_out = (addresses.shared_topk != 0)
        ? (GM_ADDR)addresses.shared_topk : (memory + config.indexed.topk);
    whole_layer_attention::Config c_attn = config.indexed.attention;
    c_attn.rows = active_rows;
    indexer_front::Config c_front = config.indexed.front;
    c_front.rows = active_rows;
    const whole_layer_attention::DeviceArgs attention{
        a.token_x, a.token_scale, a.main_weights[0], a.main_weights[1],
        a.main_weights[2], a.main_weights[3], a.main_weights[4],
        a.main_weights[5], a.main_weights[6], a.main_weights[7],
        a.main_weights[8], a.main_metadata[0], a.main_metadata[1],
        a.main_metadata[2], a.caches[0], a.caches[1], topk_out,
        a.main_metadata[3], a.main_metadata[4], a.main_metadata[5],
        memory + config.indexed.attention_offset, memory + config.indexed.latent};
    whole_layer_attention::PrologPhase(pipe, c_attn, attention,
        memory + config.indexed.query_norm, memory + config.indexed.query_norm_scale);

    const indexer_front::device::DeviceArgs front{
        a.hidden, memory + config.indexed.query_norm, memory + config.indexed.query_norm_scale,
        a.front_weights[0], a.front_weights[1], a.front_weights[2],
        a.front_weights[3], a.front_weights[4], a.indexer_metadata[1],
        a.indexer_metadata[0], a.front_weights[5], memory + config.indexed.front_offset,
        a.tiling, memory + config.indexed.query, memory + config.indexed.key,
        memory + config.indexed.weights,
        memory + config.indexed.query_scale, memory + config.indexed.key_scale,
        a.indexer_metadata[2], a.caches[2], a.caches[3],
        static_cast<int64_t>(config.indexed.indexer.pages) * whole_layer_indexer::kPage,
        config.indexed.attention.rows, static_cast<int32_t>(active_rows)};
    indexer_front::device::Forward(pipe, c_front, front, indexer_front::device::NoInputStage{});
}

template <int Ver = KERNEL_VER>
__global__ __mix__(1, 2) __schedmode__(1) __aicore__
void KvCommitKernel(uint64_t sync, Config config, Addresses addresses)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync);
    TPipe pipe;
    RunKvCommit(pipe, config, addresses);
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
}
"""

_KV_COMMIT_HOST_SNIPPET = """
extern "C" int layer_tp_tree_kv_commit_launch(void* stream, uint64_t sync,
    const layer_tp_tree_chain::Config* config,
    const layer_tp_tree_chain::Addresses* addresses,
    const layer_tp_tree_chain::Communication* communication)
{
    if (config == nullptr || addresses == nullptr || communication == nullptr ||
        !layer_tp_tree_chain::Valid(*config, *addresses, *communication)) {
        return -1;
    }
    layer_tp_tree_chain::KvCommitKernel<<<addresses->body.blocks, nullptr, stream>>>(
        sync, *config, *addresses);
    return 0;
}
"""


def _default_l78_src_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "ops/megakernel"


def _default_l78_out_dir() -> Path:
    env_dir = os.environ.get("TILERT_BUILD_DIR")
    if env_dir:
        return Path(env_dir) / "dspine_l78"
    return Path(__file__).resolve().parents[4] / "tests/.artifacts/builds/dspine_l78"


def _cann_root() -> Path:
    for candidate in (
        os.environ.get("ASCEND_TOOLKIT_HOME"),
        "/opt/ascend/cann-9.2.0",
        "/usr/local/Ascend/ascend-toolkit/latest",
    ):
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    return Path("/opt/ascend/cann-9.2.0")


def _site_root() -> Path:
    try:
        import torch

        return Path(torch.__file__).resolve().parents[1]
    except Exception:
        return Path("/usr/local/lib/python3.11/site-packages")


L78_VARIANT_SPECS: dict[int, tuple[int, str]] = {
    2048: (110415, "l78k2048"),
    1024: (110419, "l78k1024"),
    512: (110420, "l78k512"),
}
L78_PATCH_VERSION = "a2_idx_ktopk_v1"
L78_BASE_DEFS: tuple[str, ...] = (
    "-DSFA_MM2_PAIR_LOAD=1",
    "-DQUERY_GROUPS_NARROW=1",
    "-DBF16_QUERY12_GROUPS=1",
    "-DGU_WIDE=1",
    "-DATTENTION_HEADS=8",
    "-DFFN_OP_NS=tilert_ffn_original_shared_gu_static",
    "-DROUTED_ROW_BALANCED=1",
    "-DROUTED_SHARED_PIPELINE=1",
    "-DROUTED_INPUT_FRONT=1",
    "-DSHARED_W8_DYNAMIC_PERFORMANCE_QUANT=1",
    "-DROUTED_ACTIVE_LIST=1",
    "-DL78_PLAIN_RMSNORM=1",
    "-DL78_EXACT_INDEXER=1",
    "-DK11040_F0_NZ=1",
    "-DK11040_GU_FAST128=1",
)


def _l78_source_digest(src: Path) -> str:
    h = hashlib.sha256()
    for sub in ("local", "source", "vendor", "communication"):
        sub_dir = src / sub
        for p in sorted(sub_dir.rglob("*")):
            if p.is_file():
                h.update(str(p.relative_to(src)).encode("utf-8"))
                h.update(p.read_bytes())
    h.update(_KV_COMMIT_DEVICE_SNIPPET.encode("utf-8"))
    h.update(_KV_COMMIT_HOST_SNIPPET.encode("utf-8"))
    return h.hexdigest()


def _resolve_l78_ksfa(
    k_sfa: int | None,
    variants: tuple[int, ...] | list[int] | None,
) -> tuple[int, ...]:
    if k_sfa is not None:
        if isinstance(k_sfa, bool) or not isinstance(k_sfa, int) or k_sfa not in L78_VARIANT_SPECS:
            raise ValueError(f"Unsupported k_sfa for L78 build: {k_sfa!r}")
    if variants is not None:
        if not variants:
            raise ValueError("variants must not be empty")
        cleaned: list[int] = []
        for item in variants:
            if (
                isinstance(item, bool)
                or not isinstance(item, int)
                or item not in L78_VARIANT_SPECS
            ):
                raise ValueError(f"Unsupported L78 variant k_sfa: {item!r}")
            if item not in cleaned:
                cleaned.append(item)
        if k_sfa is not None and tuple(cleaned) != (k_sfa,):
            raise ValueError(f"Conflicting k_sfa={k_sfa} and variants={variants!r}")
        return tuple(cleaned)
    if k_sfa is not None:
        return (k_sfa,)
    return (2048, 1024, 512)


def _l78_variant_digest(src_digest: str, ver: int, k_val: int) -> str:
    payload = {
        "source_digest": src_digest,
        "ver": ver,
        "k_sfa": k_val,
        "defs": list(L78_BASE_DEFS),
        "patch_version": L78_PATCH_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _patch_l78_copied_sources(out: Path) -> None:
    attn_h = out / "local/attention/source/tests/glm52/whole_layer/attention_native.h"
    attn_txt = attn_h.read_text()
    if "base.sparseBlockCount = kTopK;" in attn_txt:
        attn_txt = attn_txt.replace(
            "base.sparseBlockCount = kTopK;",
            "base.sparseBlockCount = DRAFT_SFA_TOPK;",
            1,
        )
    elif "base.sparseBlockCount = DRAFT_SFA_TOPK;" not in attn_txt:
        raise RuntimeError("Missing sparseBlockCount anchor in attention_native.h")
    tiling_anchor = "auto tiling = SfaTiling(config);"
    if attn_txt.count(tiling_anchor) != 1:
        raise RuntimeError("Missing or ambiguous SfaTiling anchor in attention_native.h")
    attn_txt = attn_txt.replace(
        tiling_anchor,
        (
            f"{tiling_anchor}\n"
            "        if (!is_shared_only) {\n"
            "            tiling.baseParams.sparseBlockCount = kTopK;\n"
            "        }"
        ),
        1,
    )
    attn_h.write_text(attn_txt)

    kern_cpp = out / "source/kernel.cpp"
    kern_txt = kern_cpp.read_text()
    anchor_dev = "template <bool indexed, int Ver = KERNEL_VER>"
    anchor_host = 'extern "C" void layer_tp_tree_set_mask_ptr(uint64_t ptr)'
    if anchor_dev not in kern_txt or anchor_host not in kern_txt:
        raise RuntimeError("Missing kernel.cpp KV commit injection anchors")
    kern_txt = kern_txt.replace(anchor_dev, _KV_COMMIT_DEVICE_SNIPPET + "\n" + anchor_dev, 1)
    kern_txt = kern_txt.replace(anchor_host, _KV_COMMIT_HOST_SNIPPET + "\n" + anchor_host, 1)
    if "RunLayer(" in kern_txt and "if (q_len <= 0)" not in kern_txt:
        ql_anchor = "    const int64_t active_rows = (q_len > 0 && q_len <= rows) ? q_len : rows;"
        if ql_anchor not in kern_txt:
            raise RuntimeError("Missing RunLayer q_len anchor in kernel.cpp")
        kern_txt = kern_txt.replace(
            ql_anchor,
            f"    if (q_len <= 0) {{\n        return;\n    }}\n{ql_anchor}",
        )
    kern_cpp.write_text(kern_txt)

    wl_dir = out / "local/attention/source/tests/glm52/whole_layer"
    for fname, anchor in (
        (
            "attention_indexed_native.h",
            "    const uint32_t sfa_rows =\n"
            "        (q_len > 0 && q_len <= static_cast<int32_t>(c.attention.rows))",
        ),
        (
            "attention_full_native.h",
            "    const uint32_t sfa_rows =\n"
            "        (q_len > 0 && q_len <= static_cast<int32_t>(config.attention.rows))",
        ),
    ):
        hdr = wl_dir / fname
        if hdr.is_file():
            txt = hdr.read_text()
            if anchor in txt and "if (q_len <= 0)" not in txt:
                txt = txt.replace(
                    anchor,
                    f"    if (q_len <= 0) {{\n        return;\n    }}\n{anchor}",
                    1,
                )
                hdr.write_text(txt)


def _render_l78_build_script(
    out: Path,
    cann: Path,
    site: Path,
    target_ks: tuple[int, ...],
) -> str:
    asc_subdirs = (
        "asc", "asc/include", "asc/include/adv_api", "asc/include/basic_api",
        "asc/include/c_api", "asc/include/simt_api", "asc/include/utils",
        "asc/include/basic_api/reg_compute", "asc/impl/adv_api",
        "asc/impl/basic_api", "asc/impl/c_api", "asc/impl/simt_api",
        "asc/impl/utils", "asc/impl/basic_api/reg_compute",
        "tikcpp/tikcfw", "tikcpp/tikcfw/lib", "tikcpp/tikcfw/impl",
        "tikcpp/tikcfw/interface",
    )
    inc = " ".join(
        [
            f'-I {shlex.quote(str(out / "local/ffn/body/source/routed_int4"))}',
            f'-I {shlex.quote(str(cann / "include"))}',
            f'-I {shlex.quote(str(out / "local/ffn/bridge/vendor"))}',
            f'-I {shlex.quote(str(out / "local/ffn/source"))}',
            f'-I {shlex.quote(str(out / "vendor"))}',
            "-I /assets/shmem-lib/include",
            "-I /assets/shmem-lib/src/device",
            *(f'-I {shlex.quote(str(cann / "aarch64-linux" / s))}' for s in asc_subdirs),
        ]
    )
    torch_inc = " ".join(
        [
            f'-I {shlex.quote(str(site / "torch/include"))}',
            f'-I {shlex.quote(str(site / "torch/include/torch/csrc/api/include"))}',
            f'-I {shlex.quote(str(site / "torch_npu/include"))}',
            f'-I {shlex.quote(str(cann / "include"))}',
            "-I /assets/shmem-lib/include",
        ]
    )
    libs = " ".join(
        [
            "-L /assets/shmem-lib/lib -lshmem -lshmem_utils",
            f'-L {shlex.quote(str(cann / "aarch64-linux/lib64"))}',
            f'-L {shlex.quote(str(cann / "lib64"))}',
            "-lascendcl -lruntime -lunified_dlog -lc_sec "
            "-lregister -lprofapi -ltiling_api -lplatform",
            f'-L {shlex.quote(str(site / "torch/lib"))}',
            f'-L {shlex.quote(str(site / "torch_npu/lib"))}',
            "-ltorch -ltorch_cpu -lc10 -ltorch_npu",
        ]
    )
    base_defs_block = "\n".join(f"        {shlex.quote(d)}" for d in L78_BASE_DEFS)
    compile_lines = []
    for k_val in target_ks:
        ver, sfx = L78_VARIANT_SPECS[k_val]
        compile_lines.append(
            f"compile_variant {ver} {sfx} -DDRAFT_SFA_TOPK={k_val} &"
        )
    compile_block = "\n".join(compile_lines)
    return f"""set -euo pipefail
cann={shlex.quote(str(cann))}
out={shlex.quote(str(out))}
set +u
source "$cann/set_env.sh"
set -u
includes=({inc})
torch_includes=({torch_inc})
libs=({libs})
compile_variant() {{
    local ver=$1
    local sfx=$2
    shift 2
    local defs=(
        -DKERNEL_VER=$ver
{base_defs_block}
        -Dlayer_tp_tree_chain=layer_tp_tree_chain_K$ver
        -Dlayer_tp_tree_chain_launch=layer_tp_tree_chain_launch_K$ver
        -Dlayer_tp_tree_indexed_launch=layer_tp_tree_indexed_launch_K$ver
        -Dlayer_tp_tree_shared_launch=layer_tp_tree_shared_launch_K$ver
        -Dlayer_tp_tree_kv_commit_launch=layer_tp_tree_kv_commit_launch_K$ver
        -Dlayer_tp_tree_set_mask_ptr=layer_tp_tree_set_mask_ptr_K$ver
        -Dlayer_tp_tree_set_peer_weights_ptr=layer_tp_tree_set_peer_weights_ptr_K$ver
        "$@"
    )
    "$cann/bin/bisheng" -xcce --npu-arch=dav-2201 -std=c++17 -O2 -w \\
        -ffp-contract=off -fPIC "${{defs[@]}}" "${{includes[@]}}" \\
        -c "$out/source/kernel.cpp" -o "$out/kernel_$sfx.o" &
    c++ -std=c++17 -O2 -fPIC -D_GLIBCXX_USE_CXX11_ABI=1 \\
        "${{defs[@]}}" "${{torch_includes[@]}}" \\
        -c "$out/source/host.cpp" -o "$out/host_$sfx.o" &
    c++ -std=c++17 -O2 -fPIC -D_GLIBCXX_USE_CXX11_ABI=1 \\
        "${{defs[@]}}" "${{torch_includes[@]}}" \\
        -c "$out/communication/communication_host.cpp" -o "$out/comm_$sfx.o" &
    wait
    c++ -shared -Wl,--no-undefined -Wl,-Bsymbolic \\
        "$out/kernel_$sfx.o" "$out/host_$sfx.o" "$out/comm_$sfx.o" \\
        -o "$out/chain_K${{ver}}.so" "${{libs[@]}}"
    rm -f "$out/kernel_$sfx.o" "$out/host_$sfx.o" "$out/comm_$sfx.o"
}}
{compile_block}
wait
"""


def build_draft_l78_k110419(
    src_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    k_sfa: int | None = None,
    variants: tuple[int, ...] | list[int] | None = None,
) -> tuple[Path, Path]:
    src = _default_l78_src_dir() if src_dir is None else Path(src_dir)
    out = _default_l78_out_dir() if out_dir is None else Path(out_dir)
    target_ks = _resolve_l78_ksfa(k_sfa, variants)
    so_1024 = out / "chain_K110419.so"
    so_512 = out / "chain_K110420.so"
    src_digest = _l78_source_digest(src)
    manifest_path = out / "l78.manifest.json"
    prev_variants: dict[str, dict[str, object]] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict) and isinstance(loaded.get("variants"), dict):
            prev_variants = loaded["variants"]

    missing_or_stale: list[int] = []
    for k_val in target_ks:
        ver, _ = L78_VARIANT_SPECS[k_val]
        so_p = out / f"chain_K{ver}.so"
        entry = prev_variants.get(str(ver))
        expected_vdig = _l78_variant_digest(src_digest, ver, k_val)
        hit = (
            so_p.is_file()
            and isinstance(entry, dict)
            and entry.get("k_sfa") == k_val
            and entry.get("variant_digest") == expected_vdig
            and entry.get("so_sha256") == _file_sha256(so_p)
        )
        if not hit:
            missing_or_stale.append(k_val)

    if not missing_or_stale:
        return so_1024, so_512

    out.mkdir(parents=True, exist_ok=True)
    for sub in ("local", "source", "vendor", "communication"):
        dst_sub = out / sub
        if dst_sub.exists():
            shutil.rmtree(dst_sub)
        shutil.copytree(src / sub, dst_sub)

    _patch_l78_copied_sources(out)
    build_sh = _render_l78_build_script(out, _cann_root(), _site_root(), tuple(missing_or_stale))
    sh_path = out / "build_p2.sh"
    sh_path.write_text(build_sh, encoding="utf-8")
    subprocess.run(["bash", str(sh_path)], check=True)

    next_variants: dict[str, dict[str, object]] = {}
    for k_val, (ver, _) in L78_VARIANT_SPECS.items():
        so_p = out / f"chain_K{ver}.so"
        if not so_p.is_file():
            continue
        expected_vdig = _l78_variant_digest(src_digest, ver, k_val)
        if k_val in missing_or_stale:
            next_variants[str(ver)] = {
                "k_sfa": k_val,
                "variant_digest": expected_vdig,
                "so_sha256": _file_sha256(so_p),
            }
        else:
            prev_entry = prev_variants.get(str(ver))
            if (
                isinstance(prev_entry, dict)
                and prev_entry.get("variant_digest") == expected_vdig
                and prev_entry.get("so_sha256") == _file_sha256(so_p)
            ):
                next_variants[str(ver)] = prev_entry

    new_manifest = {"source_digest": src_digest, "variants": next_variants}
    tmp_manifest = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    tmp_manifest.write_text(json.dumps(new_manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)
    return so_1024, so_512
