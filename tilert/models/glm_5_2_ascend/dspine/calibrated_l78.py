from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import torch
from torch import nn

from ..ops.build_megakernel import UNSET_L78_MACROS, effective_tree_digest
from ..weights import pack_output_weights, pack_shared_weights, pack_value_weights
from .build_dspine_ops import DSPINE_WALK_SHARED_CPP, build_dspine_ops

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
        k_exp = max(2, cur_b, nxt_b) if nxt_b > 0 else max(2, cur_b)
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
        world_size: int = 8,
    ):
        self.rank = int(rank)
        self.world_size = int(world_size)
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
                    if self.world_size == 8:
                        build_dspine_ops(so_p.parent)
                    else:
                        build_dspine_ops(so_p.parent, world_size=self.world_size)
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            self.ops = _load_ops_cdll(so_p, self.rank)

        if self.world_size == 16 and len(ckpt_blob["shards"]) == 8:
            sh = ckpt_blob["shards"][self.rank // 2]
            sub = self.rank & 1
        else:
            sh = ckpt_blob["shards"][self.rank]
            assert int(sh["tp_rank"]) == self.rank
            sub = -1

        def _split_eh(t: torch.Tensor) -> torch.Tensor:
            if sub < 0:
                return t
            a = sub * 384
            b = a + 384
            return torch.cat((t[:, a:b, :], t[:, 768 + a : 768 + b, :]), dim=1)

        def _split_sh_gu(t: torch.Tensor) -> torch.Tensor:
            if sub < 0:
                return t
            a = sub * 128
            b = a + 128
            return torch.cat((t[:, a:b], t[:, 256 + a : 256 + b]), dim=1)

        if sub >= 0:
            a384 = sub * 384
            rot_w_raw = sh["rot_w_step"][:, :, a384 : a384 + 384]
            rot_b_raw = sh["rot_b_step"][:, a384 : a384 + 384]
        else:
            rot_w_raw = sh["rot_w_step"]
            rot_b_raw = sh["rot_b_step"]
        self.rot_w_step = rot_w_raw.to(device=dev, dtype=torch.bfloat16).contiguous()
        self.rot_b_step = rot_b_raw.to(device=dev, dtype=torch.bfloat16).contiguous()
        self.w_eh_step = (
            _split_eh(sh["w_eh_step"]).to(device=dev, dtype=torch.bfloat16).contiguous()
        )
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

        heads = 64 // self.world_size
        sh_dim = 2048 // self.world_size
        cal_uk_raw = sh["cal_w_uk_t"][sub * 4 : (sub + 1) * 4] if sub >= 0 else sh["cal_w_uk_t"]
        cal_uv_raw = sh["cal_w_uv"][sub * 4 : (sub + 1) * 4] if sub >= 0 else sh["cal_w_uv"]
        self.cal_w_uk_t = cal_uk_raw.to(device=dev, dtype=torch.bfloat16).contiguous()
        self.cal_w_uv = cal_uv_raw.to(device=dev, dtype=torch.bfloat16).contiguous()
        uv_tile = (heads * 256) // 16
        uv_n16 = uv_tile // 16
        w_uv_packed_cpu = (
            pack_value_weights(cal_uv_raw.to(torch.bfloat16).cpu())
            .view(16, 512, uv_n16, 16)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(16, 512, uv_tile)
        )
        self.cal_w_uv_packed = w_uv_packed_cpu.to(device=dev, dtype=torch.bfloat16).contiguous()

        self.cal_uq_qr_q8_nz = None
        self.cal_uq_qr_scale = None
        if "cal_w_uq_qr" in sh:
            cal_uq_raw = (
                sh["cal_w_uq_qr"][:, sub * 1024 : (sub + 1) * 1024]
                if sub >= 0
                else sh["cal_w_uq_qr"]
            )
            uq_q8_cpu, uq_s_cpu = quantize_linear_col_i8(cal_uq_raw.to(torch.float32))
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
            cal_wo_raw = (
                sh["cal_w_o"][sub * 1024 : (sub + 1) * 1024, :]
                if sub >= 0
                else sh["cal_w_o"]
            )
            wo_q8_cpu, wo_s_cpu = quantize_linear_col_i8(cal_wo_raw.to(torch.float32))
            o_in = heads * 256
            wo_p = (
                pack_output_weights(wo_q8_cpu)
                .view(24, o_in, 8, 32)
                .permute(0, 2, 1, 3)
                .contiguous()
                .view(24, o_in, 256)
            )
            self.cal_w_o_packed = wo_p.to(device=dev).contiguous()
            self.cal_w_o_scale = wo_s_cpu.to(device=dev, dtype=torch.float32).contiguous()

        self.cal_shared_packed = None
        if "cal_shared_gu" in sh and "cal_shared_down" in sh:
            cal_sgu_raw = _split_sh_gu(sh["cal_shared_gu"])
            cal_sdn_raw = (
                sh["cal_shared_down"][sub * 128 : (sub + 1) * 128, :]
                if sub >= 0
                else sh["cal_shared_down"]
            )
            gu_q8_cpu, gu_s_cpu = quantize_linear_col_i8(cal_sgu_raw.to(torch.float32))
            dn_q8_cpu, dn_s_cpu = quantize_linear_col_i8(cal_sdn_raw.to(torch.float32))
            p_gu_cpu, p_dn_cpu = pack_shared_weights(gu_q8_cpu, dn_q8_cpu)
            sh_gu_tile = sh_dim // 2
            sh_gu_g32 = sh_gu_tile // 32
            sh_dn_k128 = sh_dim // 128
            p_gu_cpu = (
                p_gu_cpu.view(24, 2, 512, sh_gu_g32, 32)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
                .view(24, 1024, sh_gu_tile)
            )
            p_dn_cpu = (
                p_dn_cpu.view(24, sh_dn_k128, 128, 8, 32)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
                .view(24, sh_dim, 256)
            )
            self.cal_shared_packed = [
                p_gu_cpu.to(device=dev).contiguous(),
                gu_s_cpu.to(device=dev, dtype=torch.float32).contiguous(),
                p_dn_cpu.to(device=dev).contiguous(),
                dn_s_cpu.to(device=dev, dtype=torch.float32).contiguous(),
            ]

        w13_mul = sh["cal_routed_w13_mul"] if "cal_routed_w13_mul" in sh else None
        if w13_mul is not None and sub >= 0:
            w13_mul = w13_mul[sub * 16 : (sub + 1) * 16]
        self.cal_routed_w13_mul = (
            w13_mul.to(device=dev, dtype=torch.float32).contiguous()
            if w13_mul is not None
            else None
        )
        w2_mul = sh["cal_routed_w2_mul"] if "cal_routed_w2_mul" in sh else None
        if w2_mul is not None and sub >= 0:
            w2_mul = w2_mul[sub * 16 : (sub + 1) * 16]
        self.cal_routed_w2_mul = (
            w2_mul.to(device=dev, dtype=torch.float32).contiguous()
            if w2_mul is not None
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
        world_size: int = 8,
    ):
        blob = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
        return cls(blob, rank, dev, ops_so_path=ops_so_path, world_size=world_size)

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
static uint64_t g_exact_wk_w_gm = 0ULL;
static uint64_t g_exact_knorm_w_gm = 0ULL;
static uint64_t g_exact_knorm_b_gm = 0ULL;
static uint64_t g_exact_kcache_gm = 0ULL;
static uint64_t g_exact_work_gm = 0ULL;
static float g_exact_eps = 1e-6F;
static uint64_t g_exact_seed_args_gm = 0ULL;

struct SeedTopkArgs {
    uint64_t comm_win_gm;
    uint64_t s0_gamma_gm;
    uint64_t cal_head_w_q8_nz_gm;
    uint64_t cal_head_s8_gm;
    uint64_t recs_out_gm;
    uint64_t c2b_act_q_gm;
    uint64_t c2b_act_s_gm;
    uint64_t c2b_acc_i32_gm;
    uint64_t logits_work_gm;
    uint64_t gather_in_gm;
    uint64_t gather_out_gm;
    uint64_t draft_ctrl_gm;
    uint64_t seed_top_i_gm;
    uint64_t seed_top_lp_gm;
    uint64_t prefix_topk_gm;
    uint64_t indices_buf_gm;
    uint64_t beam_work_gm;
    int32_t vocab_valid;
    int32_t vocab_start;
    int32_t k_seed;
    int32_t b0;
    int32_t max_b;
    int32_t k_sfa;
    int32_t rank;
    int32_t world_size;
    float norm_eps;
    uint32_t pf_bytes;
    uint64_t pad[10];
};
static_assert(sizeof(SeedTopkArgs) == 256, "SeedTopkArgs ABI mismatch");

__aicore__ inline void ExactDcciRange(uint64_t addr, int32_t bytes)
{
    if (addr == 0ULL || bytes <= 0) {
        return;
    }
    const uint64_t start = addr & ~63ULL;
    const uint64_t end = (addr + static_cast<uint64_t>(bytes) + 63ULL) & ~63ULL;
    for (uint64_t p = start; p < end; p += 64ULL) {
        AscendC::GlobalTensor<int64_t> g;
        g.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(p), 8);
        AscendC::DataCacheCleanAndInvalid<
            int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(g);
    }
}

template <AscendC::HardEvent Ev>
__aicore__ inline void ExactFence()
{
    const auto id = GetTPipePtr()->FetchEventID(Ev);
    AscendC::SetFlag<Ev>(id);
    AscendC::WaitFlag<Ev>(id);
}

__aicore__ inline void ExactPublish()
{
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
}

__aicore__ __attribute__((noinline)) void RunExactIndexKeysAiv(
    AscendC::TPipe& pipe,
    uint64_t q_len_gm,
    uint64_t slot_i64_gm,
    uint64_t cos_gm,
    uint64_t sin_gm,
    uint64_t k_norm_w_gm,
    uint64_t k_norm_b_gm,
    uint64_t k_cache_gm,
    uint64_t work_gm,
    int32_t rows_cap,
    float eps)
{
    if (work_gm == 0ULL) {
        return;
    }
    int32_t active_rows = rows_cap;
    if (q_len_gm != 0ULL) {
        ExactDcciRange(q_len_gm, 64);
        active_rows = *reinterpret_cast<__gm__ const int32_t*>(q_len_gm);
    }
    if (active_rows <= 0) {
        return;
    }
    if (active_rows > rows_cap) {
        active_rows = rows_cap;
    }
    if ASCEND_IS_AIV {
        const int32_t aiv = AscendC::GetBlockIdx();
        ExactDcciRange(slot_i64_gm, rows_cap * 8);
        ExactDcciRange(work_gm, active_rows * 128 * 4);

        AscendC::TBuf<AscendC::TPosition::VECCALC> b_f32, b_bf, b_slot;
        pipe.InitBuffer(b_f32, 11 * 128 * 4);
        pipe.InitBuffer(b_bf, 4 * 128 * 2);
        pipe.InitBuffer(b_slot, 64 * 8);

        auto ub_f32 = b_f32.Get<float>();
        auto ub_kw = ub_f32[0];
        auto ub_x = ub_f32[128];
        auto ub_diff = ub_f32[256];
        auto ub_sq = ub_f32[384];
        auto ub_gw = ub_f32[512];
        auto ub_gb = ub_f32[640];
        auto ub_tmp = ub_f32[768];
        auto ub_sum = ub_f32[896];
        auto ub_k = ub_f32[1024];
        auto ub_cos = ub_f32[1152];
        auto ub_sin = ub_f32[1280];

        auto ub_bf = b_bf.Get<bfloat16_t>();
        auto ub_kw_bf = ub_bf[0];
        auto ub_out_bf = ub_bf[128];
        auto ub_cos_bf = ub_bf[256];
        auto ub_sin_bf = ub_bf[384];
        auto ub_slot = b_slot.Get<int64_t>();

        AscendC::GlobalTensor<float> g_work, g_gw, g_gb;
        AscendC::GlobalTensor<bfloat16_t> g_cos, g_sin, g_kcache;
        AscendC::GlobalTensor<int64_t> g_slot;
        g_work.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(work_gm), rows_cap * 128);
        g_gw.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(k_norm_w_gm), 128);
        g_gb.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(k_norm_b_gm), 128);
        g_cos.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cos_gm), rows_cap * 64);
        g_sin.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(sin_gm), rows_cap * 64);
        g_kcache.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(k_cache_gm));
        g_slot.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slot_i64_gm), rows_cap);

        const int32_t copy_sl = ((rows_cap + 3) / 4) * 4;
        AscendC::DataCopy(ub_slot, g_slot, copy_sl);
        AscendC::DataCopy(ub_gw, g_gw, 128);
        AscendC::DataCopy(ub_gb, g_gb, 128);
        ExactFence<AscendC::HardEvent::MTE2_S>();
        ExactFence<AscendC::HardEvent::MTE2_V>();

        for (int32_t r = aiv; r < active_rows; r += 48) {
            const int64_t slot = ub_slot.GetValue(r);
            if (slot < 0) {
                continue;
            }
            AscendC::DataCopy(ub_kw, g_work[r * 128], 128);
            AscendC::DataCopy(ub_cos_bf, g_cos[r * 64], 64);
            AscendC::DataCopy(ub_sin_bf, g_sin[r * 64], 64);
            ExactFence<AscendC::HardEvent::MTE2_V>();

            AscendC::Cast(ub_kw_bf, ub_kw, AscendC::RoundMode::CAST_RINT, 128);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(ub_x, ub_kw_bf, AscendC::RoundMode::CAST_NONE, 128);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::ReduceSum(ub_sum, ub_x, ub_tmp, 128);
            ExactFence<AscendC::HardEvent::V_S>();
            const float mean = ub_sum.GetValue(0) * (1.0F / 128.0F);
            ExactFence<AscendC::HardEvent::S_V>();
            AscendC::Adds(ub_diff, ub_x, -mean, 128);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(ub_sq, ub_diff, ub_diff, 128);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceSum(ub_sum, ub_sq, ub_tmp, 128);
            ExactFence<AscendC::HardEvent::V_S>();
            const float var = ub_sum.GetValue(0) * (1.0F / 128.0F);
            const float inv_std = 1.0F / sqrt(var + eps);
            ExactFence<AscendC::HardEvent::S_V>();
            AscendC::Muls(ub_x, ub_diff, inv_std, 128);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(ub_x, ub_x, ub_gw, 128);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(ub_x, ub_x, ub_gb, 128);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(ub_out_bf, ub_x, AscendC::RoundMode::CAST_RINT, 128);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Cast(ub_k, ub_out_bf, AscendC::RoundMode::CAST_NONE, 64);
            AscendC::Cast(ub_cos, ub_cos_bf, AscendC::RoundMode::CAST_NONE, 64);
            AscendC::Cast(ub_sin, ub_sin_bf, AscendC::RoundMode::CAST_NONE, 64);
            ExactFence<AscendC::HardEvent::V_S>();
            for (int32_t j = 0; j < 32; ++j) {
                const float c = ub_cos.GetValue(j);
                const float s = ub_sin.GetValue(j);
                const float k0 = ub_k.GetValue(2 * j);
                const float k1 = ub_k.GetValue(2 * j + 1);
                ub_diff.SetValue(2 * j, k0 * c - k1 * s);
                ub_diff.SetValue(2 * j + 1, k1 * c + k0 * s);
            }
            ExactFence<AscendC::HardEvent::S_V>();
            AscendC::Cast(ub_out_bf, ub_diff, AscendC::RoundMode::CAST_RINT, 64);
            ExactFence<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopy(g_kcache[slot * 128], ub_out_bf, 128);
            ExactFence<AscendC::HardEvent::MTE3_V>();
            ExactFence<AscendC::HardEvent::MTE3_MTE2>();
        }
        ExactPublish();
        pipe.Reset();
    }
}

__aicore__ __attribute__((noinline)) void RunExactIndexKeysInline(
    AscendC::TPipe& pipe,
    uint64_t norm_x_gm,
    uint64_t q_len_gm,
    uint64_t slot_i64_gm,
    uint64_t cos_gm,
    uint64_t sin_gm,
    uint64_t wk_w_gm,
    uint64_t k_norm_w_gm,
    uint64_t k_norm_b_gm,
    uint64_t k_cache_gm,
    uint64_t work_gm,
    int32_t rows_cap,
    float eps)
{
    if (wk_w_gm == 0ULL || work_gm == 0ULL) {
        return;
    }
    int32_t active_rows = rows_cap;
    if (q_len_gm != 0ULL) {
        ExactDcciRange(q_len_gm, 64);
        active_rows = *reinterpret_cast<__gm__ const int32_t*>(q_len_gm);
    }
    if (active_rows <= 0) {
        return;
    }
    if (active_rows > rows_cap) {
        active_rows = rows_cap;
    }
    if ASCEND_IS_AIC {
        constexpr int32_t kTile = 16;
        constexpr int32_t kInner = 128;
        constexpr int32_t kHidden = 6144;
        constexpr int32_t kSteps = kHidden / kInner;
        constexpr int32_t kStage = kTile * kInner;
        constexpr int32_t kAcc = kTile * kTile;

        AscendC::TBuf<AscendC::TPosition::A1> a_stage;
        AscendC::TBuf<AscendC::TPosition::B1> b_stage;
        AscendC::TBuf<AscendC::TPosition::A2> a_op;
        AscendC::TBuf<AscendC::TPosition::B2> b_op;
        AscendC::TBuf<AscendC::TPosition::CO1> c_acc;
        pipe.InitBuffer(a_stage, 2 * kStage * sizeof(bfloat16_t));
        pipe.InitBuffer(b_stage, 2 * kStage * sizeof(bfloat16_t));
        pipe.InitBuffer(a_op, 2 * kStage * sizeof(bfloat16_t));
        pipe.InitBuffer(b_op, 2 * kStage * sizeof(bfloat16_t));
        pipe.InitBuffer(c_acc, kAcc * sizeof(float));

        auto a_l1 = a_stage.Get<bfloat16_t>();
        auto b_l1 = b_stage.Get<bfloat16_t>();
        auto a_l0 = a_op.Get<bfloat16_t>();
        auto b_l0 = b_op.Get<bfloat16_t>();
        auto c_l0 = c_acc.Get<float>();

        AscendC::GlobalTensor<bfloat16_t> a_gm, b_gm;
        AscendC::GlobalTensor<float> c_gm;
        a_gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(norm_x_gm), rows_cap * kHidden);
        b_gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(wk_w_gm), 128 * kHidden);
        c_gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(work_gm), rows_cap * 128);

        const int32_t blk = AscendC::GetBlockIdx();
        const int32_t row_tiles = (active_rows + 15) / 16;
        const int32_t total_tiles = row_tiles * 8;

        for (int32_t tile = blk; tile < total_tiles; tile += 24) {
            const int32_t row_tile = tile / 8;
            const int32_t col_tile = tile % 8;
            const int32_t r_start = row_tile * kTile;
            const int32_t r_rem = active_rows - r_start;
            const int32_t r_valid = (r_rem < kTile) ? r_rem : kTile;

            AscendC::Nd2NzParams copy_a{};
            copy_a.ndNum = 1;
            copy_a.nValue = r_valid;
            copy_a.dValue = kInner;
            copy_a.srcDValue = kHidden;
            copy_a.dstNzC0Stride = kTile;
            copy_a.dstNzNStride = 1;

            AscendC::Nd2NzParams copy_b{};
            copy_b.ndNum = 1;
            copy_b.nValue = kTile;
            copy_b.dValue = kInner;
            copy_b.srcDValue = kHidden;
            copy_b.dstNzC0Stride = kTile;
            copy_b.dstNzNStride = 1;

            AscendC::LoadData2DParams load_a{};
            load_a.repeatTimes = kInner / kTile;
            load_a.srcStride = 1;
            load_a.ifTranspose = false;

            AscendC::LoadData2DParams load_b{};
            load_b.repeatTimes = kInner / kTile;
            load_b.srcStride = 1;
            load_b.ifTranspose = false;

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);

            for (int32_t s = 0; s < kSteps; ++s) {
                const int32_t buf = s & 1;
                const event_t ev = buf ? EVENT_ID1 : EVENT_ID0;
                const int64_t k_off = static_cast<int64_t>(s) * kInner;
                const int64_t a_off = static_cast<int64_t>(r_start) * kHidden + k_off;
                const int64_t b_off =
                    static_cast<int64_t>(col_tile) * kTile * kHidden + k_off;

                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ev);
                AscendC::DataCopy(a_l1[buf * kStage], a_gm[a_off], copy_a);
                AscendC::DataCopy(b_l1[buf * kStage], b_gm[b_off], copy_b);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ev);

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ev);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ev);
                AscendC::LoadData(a_l0[buf * kStage], a_l1[buf * kStage], load_a);
                AscendC::LoadData(b_l0[buf * kStage], b_l1[buf * kStage], load_b);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ev);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ev);

                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ev);
                AscendC::MmadParams mad{};
                mad.m = kTile;
                mad.n = kTile;
                mad.k = kInner;
                mad.cmatrixInitVal = (s == 0);
                mad.cmatrixSource = false;
                AscendC::Mmad(c_l0, a_l0[buf * kStage], b_l0[buf * kStage], mad);
                AscendC::PipeBarrier<PIPE_M>();
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ev);
            }
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);

            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::FixpipeParamsV220 store{};
            store.nSize = kTile;
            store.mSize = r_valid;
            store.srcStride = kTile;
            store.dstStride = 128;
            store.ndNum = 1;
            store.quantPre = QuantMode_t::NoQuant;
            const int64_t c_off = static_cast<int64_t>(r_start) * 128 + col_tile * kTile;
            AscendC::Fixpipe(c_gm[c_off], c_l0, store);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }
        ExactPublish();
    }
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIV {
        const int32_t aiv = AscendC::GetBlockIdx();
        ExactDcciRange(slot_i64_gm, rows_cap * 8);
        ExactDcciRange(work_gm, active_rows * 128 * 4);

        AscendC::TBuf<AscendC::TPosition::VECCALC> b_f32, b_bf, b_slot;
        pipe.InitBuffer(b_f32, 11 * 128 * 4);
        pipe.InitBuffer(b_bf, 4 * 128 * 2);
        pipe.InitBuffer(b_slot, 64 * 8);

        auto ub_f32 = b_f32.Get<float>();
        auto ub_kw = ub_f32[0];
        auto ub_x = ub_f32[128];
        auto ub_diff = ub_f32[256];
        auto ub_sq = ub_f32[384];
        auto ub_gw = ub_f32[512];
        auto ub_gb = ub_f32[640];
        auto ub_tmp = ub_f32[768];
        auto ub_sum = ub_f32[896];
        auto ub_k = ub_f32[1024];
        auto ub_cos = ub_f32[1152];
        auto ub_sin = ub_f32[1280];

        auto ub_bf = b_bf.Get<bfloat16_t>();
        auto ub_kw_bf = ub_bf[0];
        auto ub_out_bf = ub_bf[128];
        auto ub_cos_bf = ub_bf[256];
        auto ub_sin_bf = ub_bf[384];
        auto ub_slot = b_slot.Get<int64_t>();

        AscendC::GlobalTensor<float> g_work, g_gw, g_gb;
        AscendC::GlobalTensor<bfloat16_t> g_cos, g_sin, g_kcache;
        AscendC::GlobalTensor<int64_t> g_slot;
        g_work.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(work_gm), rows_cap * 128);
        g_gw.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(k_norm_w_gm), 128);
        g_gb.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(k_norm_b_gm), 128);
        g_cos.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(cos_gm), rows_cap * 64);
        g_sin.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(sin_gm), rows_cap * 64);
        g_kcache.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(k_cache_gm));
        g_slot.SetGlobalBuffer(
            reinterpret_cast<__gm__ int64_t*>(slot_i64_gm), rows_cap);

        const int32_t copy_sl = ((rows_cap + 3) / 4) * 4;
        AscendC::DataCopy(ub_slot, g_slot, copy_sl);
        AscendC::DataCopy(ub_gw, g_gw, 128);
        AscendC::DataCopy(ub_gb, g_gb, 128);
        ExactFence<AscendC::HardEvent::MTE2_S>();
        ExactFence<AscendC::HardEvent::MTE2_V>();

        for (int32_t r = aiv; r < active_rows; r += 48) {
            const int64_t slot = ub_slot.GetValue(r);
            if (slot < 0) {
                continue;
            }
            AscendC::DataCopy(ub_kw, g_work[r * 128], 128);
            AscendC::DataCopy(ub_cos_bf, g_cos[r * 64], 64);
            AscendC::DataCopy(ub_sin_bf, g_sin[r * 64], 64);
            ExactFence<AscendC::HardEvent::MTE2_V>();

            AscendC::Cast(ub_kw_bf, ub_kw, AscendC::RoundMode::CAST_RINT, 128);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(ub_x, ub_kw_bf, AscendC::RoundMode::CAST_NONE, 128);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::ReduceSum(ub_sum, ub_x, ub_tmp, 128);
            ExactFence<AscendC::HardEvent::V_S>();
            const float mean = ub_sum.GetValue(0) * (1.0F / 128.0F);
            ExactFence<AscendC::HardEvent::S_V>();
            AscendC::Adds(ub_diff, ub_x, -mean, 128);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(ub_sq, ub_diff, ub_diff, 128);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceSum(ub_sum, ub_sq, ub_tmp, 128);
            ExactFence<AscendC::HardEvent::V_S>();
            const float var = ub_sum.GetValue(0) * (1.0F / 128.0F);
            const float inv_std = 1.0F / sqrt(var + eps);
            ExactFence<AscendC::HardEvent::S_V>();
            AscendC::Muls(ub_x, ub_diff, inv_std, 128);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(ub_x, ub_x, ub_gw, 128);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(ub_x, ub_x, ub_gb, 128);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(ub_out_bf, ub_x, AscendC::RoundMode::CAST_RINT, 128);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Cast(ub_k, ub_out_bf, AscendC::RoundMode::CAST_NONE, 64);
            AscendC::Cast(ub_cos, ub_cos_bf, AscendC::RoundMode::CAST_NONE, 64);
            AscendC::Cast(ub_sin, ub_sin_bf, AscendC::RoundMode::CAST_NONE, 64);
            ExactFence<AscendC::HardEvent::V_S>();
            for (int32_t j = 0; j < 32; ++j) {
                const float c = ub_cos.GetValue(j);
                const float s = ub_sin.GetValue(j);
                const float k0 = ub_k.GetValue(2 * j);
                const float k1 = ub_k.GetValue(2 * j + 1);
                ub_diff.SetValue(2 * j, k0 * c - k1 * s);
                ub_diff.SetValue(2 * j + 1, k1 * c + k0 * s);
            }
            ExactFence<AscendC::HardEvent::S_V>();
            AscendC::Cast(ub_out_bf, ub_diff, AscendC::RoundMode::CAST_RINT, 64);
            ExactFence<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopy(g_kcache[slot * 128], ub_out_bf, 128);
            ExactFence<AscendC::HardEvent::MTE3_V>();
            ExactFence<AscendC::HardEvent::MTE3_MTE2>();
        }
        ExactPublish();
    }
}

__aicore__ inline void RunKvCommit(
    TPipe& pipe, const Config& config, const Addresses& addresses,
    uint64_t wk_w_gm, uint64_t work_gm, float eps)
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
    RunExactIndexKeysInline(pipe,
        addresses.norm.normalized,
        addresses.attention.main_metadata[3],
        addresses.attention.main_metadata[2],
        addresses.attention.main_metadata[1],
        addresses.attention.main_metadata[0],
        wk_w_gm,
        addresses.attention.front_weights[3],
        addresses.attention.front_weights[4],
        addresses.attention.caches[2],
        work_gm,
        static_cast<int32_t>(rows),
        eps);
}

template <int Ver = KERNEL_VER>
__global__ __mix__(1, 2) __schedmode__(1) __aicore__
void KvCommitKernel(uint64_t sync, Config config, Addresses addresses,
    uint64_t wk_w_gm, uint64_t work_gm, float eps)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync);
    TPipe pipe;
    RunKvCommit(pipe, config, addresses, wk_w_gm, work_gm, eps);
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
}

template <bool Indexed, int Ver = KERNEL_VER>
__global__ __mix__(1, 2) __schedmode__(1) __aicore__
void LayerExactKernel(uint64_t sync, Config config, Addresses addresses,
    Communication communication, uint64_t wk_w_gm, uint64_t knorm_w_gm,
    uint64_t knorm_b_gm, uint64_t kcache_gm, uint64_t work_gm, float eps,
    uint64_t seed_args_gm = 0ULL)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    AscendC::GlobalTensor<int32_t> q_gm;
    q_gm.SetGlobalBuffer((__gm__ int32_t*)addresses.attention.main_metadata[3], 1);
    AscendC::DataCacheCleanAndInvalid<int32_t,
        AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(q_gm);
    if (q_gm.GetValue(0) <= 0) {
        return;
    }
    SetSyncBaseAddr(sync);
    TPipe pipe;
#ifdef L78_FUSE_HEAD_NORM
    if (seed_args_gm != 0ULL) {
        ExactDcciRange(seed_args_gm, static_cast<int32_t>(sizeof(SeedTopkArgs)));
        const SeedTopkArgs sargs = *reinterpret_cast<const SeedTopkArgs*>(seed_args_gm);
        uint32_t l78_red1_ep = 0U;
        if ASCEND_IS_AIV {
            if (communication.window != 0ULL) {
                l78_red1_ep = whole_layer_tree::ReadCommunicationEpoch(
                    reinterpret_cast<__gm__ uint8_t*>(communication.window)) + 1U;
            }
        }
        RunLayer(pipe, Indexed, config, addresses, communication, false, 0ULL,
            nullptr, 0ULL, nullptr, nullptr, false, nullptr, -1, -1, 0ULL, false, nullptr,
            l78_red1_ep, sargs.s0_gamma_gm, sargs.recs_out_gm,
            sargs.c2b_act_q_gm, sargs.c2b_act_s_gm,
            wk_w_gm, work_gm, static_cast<int32_t>(addresses.body.rows),
            sargs.cal_head_w_q8_nz_gm, sargs.pf_bytes);
        PipeBarrier<PIPE_ALL>();
        SyncAll<false>();
        pipe.Reset();
        RunExactIndexKeysAiv(pipe,
            addresses.attention.main_metadata[3],
            addresses.attention.main_metadata[2],
            addresses.attention.main_metadata[1],
            addresses.attention.main_metadata[0],
            knorm_w_gm,
            knorm_b_gm,
            kcache_gm,
            work_gm,
            static_cast<int32_t>(addresses.body.rows),
            eps);
        dspine_shared::RunDraftHeadTopk(
            pipe, 0ULL, addresses.bridge[1], communication.output,
            sargs.s0_gamma_gm, sargs.cal_head_w_q8_nz_gm, sargs.cal_head_s8_gm,
            sargs.recs_out_gm, sargs.c2b_act_q_gm, sargs.c2b_act_s_gm, sargs.c2b_acc_i32_gm,
            sargs.logits_work_gm, sargs.gather_in_gm, sargs.gather_out_gm, sargs.comm_win_gm,
            1, sargs.vocab_valid, sargs.vocab_start, sargs.k_seed,
            sargs.rank, sargs.world_size, sargs.norm_eps, 0U, true);
        dspine_shared::RunBeamStepV3(
            pipe, 0, sargs.max_b, sargs.b0, sargs.k_seed, sargs.b0, 0, sargs.k_sfa, 0,
            0.0F, 0.0F,
            sargs.gather_out_gm, sargs.draft_ctrl_gm, sargs.seed_top_i_gm, sargs.seed_top_lp_gm,
            0ULL, sargs.prefix_topk_gm, 0ULL, sargs.indices_buf_gm,
            0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
            0ULL, 0ULL, 0ULL, 0ULL,
            0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
            sargs.beam_work_gm);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
        return;
    }
#else
    (void)seed_args_gm;
#endif
    RunLayer(pipe, Indexed, config, addresses, communication, false, 0);
    PipeBarrier<PIPE_ALL>();
    pipe.Reset();
    RunExactIndexKeysInline(pipe,
        addresses.norm.normalized,
        addresses.attention.main_metadata[3],
        addresses.attention.main_metadata[2],
        addresses.attention.main_metadata[1],
        addresses.attention.main_metadata[0],
        wk_w_gm,
        knorm_w_gm,
        knorm_b_gm,
        kcache_gm,
        work_gm,
        static_cast<int32_t>(addresses.body.rows),
        eps);
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
}
"""

_KV_COMMIT_HOST_SNIPPET = """
extern "C" void layer_tp_tree_set_seed_args_ptr(uint64_t seed_args_gm)
{
    layer_tp_tree_chain::g_exact_seed_args_gm = seed_args_gm;
}

extern "C" void layer_tp_tree_set_exact_idx_ptrs(
    uint64_t wk_w_gm, uint64_t knorm_w_gm, uint64_t knorm_b_gm,
    uint64_t kcache_gm, uint64_t work_gm, float eps)
{
    layer_tp_tree_chain::g_exact_wk_w_gm = wk_w_gm;
    layer_tp_tree_chain::g_exact_knorm_w_gm = knorm_w_gm;
    layer_tp_tree_chain::g_exact_knorm_b_gm = knorm_b_gm;
    layer_tp_tree_chain::g_exact_kcache_gm = kcache_gm;
    layer_tp_tree_chain::g_exact_work_gm = work_gm;
    layer_tp_tree_chain::g_exact_eps = eps;
}

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
        sync, *config, *addresses,
        layer_tp_tree_chain::g_exact_wk_w_gm,
        layer_tp_tree_chain::g_exact_work_gm,
        layer_tp_tree_chain::g_exact_eps);
    return 0;
}
"""

_DRAFT_WALK_DEVICE_SNIPPET = """
struct DraftWalkArgs {
    uint64_t sync_addr;
    uint64_t draft_ctrl_gm;
    uint64_t step_qlen_gm;
    uint64_t comm_win_gm;
    uint64_t rot_w_step_nz_gm[32];
    uint64_t rot_b_step_gm[32];
    uint64_t w_eh_step_nz_gm[32];
    uint64_t enorm_step_gm[32];
    uint64_t hnorm_step_gm[32];
    uint64_t logit_norm_step_gm[32];
    uint64_t emb_w_gm;
    uint64_t cal_head_w_q8_nz_gm;
    uint64_t cal_head_s8_gm;
    uint64_t sin_cache_gm;
    uint64_t cos_cache_gm;
    uint64_t rot_part_gm;
    uint64_t prolog_comm_in_gm;
    uint64_t prolog_comm_out_gm;
    uint64_t eh_in_gm;
    uint64_t eh_comm_in_gm;
    uint64_t c2b_recs_out_gm;
    uint64_t c2b_act_q_gm;
    uint64_t c2b_act_s_gm;
    uint64_t c2b_acc_i32_gm;
    uint64_t logits_work_gm;
    uint64_t gather_in_gm;
    uint64_t gather_out_gm;
    uint64_t seed_top_i_gm;
    uint64_t seed_top_lp_gm;
    uint64_t seed_rec_gm;
    uint64_t prefix_topk_gm;
    uint64_t anc_slots_gm;
    uint64_t indices_buf_gm;
    uint64_t active_toks_gm;
    uint64_t active_scores_gm;
    uint64_t active_raw_lp_gm;
    uint64_t active_margins_gm;
    uint64_t active_pids_gm;
    uint64_t active_recs_gm;
    uint64_t pool_scores_gm;
    uint64_t pool_raw_lp_gm;
    uint64_t pool_tokens_gm;
    uint64_t pool_parents_gm;
    uint64_t pool_depths_gm;
    uint64_t beam_work_gm;
    uint64_t tree_work_gm;
    uint64_t tree_tokens_gm;
    uint64_t tree_parents_gm;
    uint64_t tree_depths_gm;
    uint64_t tree_mask_gm;
    uint64_t t_ids_gm;
    uint64_t t_pos_gm;
    uint64_t t_slots_gm;
    uint64_t t_slots_i64_gm;
    uint64_t dyn_w_gm;
    uint64_t full_seq_ptrs_gm;
    uint64_t mc2_mask_gm;
    uint64_t rope_dsts_gm;
    int32_t b0;
    int32_t max_steps;
    int32_t max_d;
    int32_t k_sfa;
    int32_t width;
    int32_t min_w_act;
    int32_t max_pool_size;
    int32_t num_seq_ptrs;
    int32_t num_q_ptrs;
    int32_t num_rope_dsts;
    int32_t emb_vocab_start;
    int32_t emb_vocab_end;
    int32_t head_valid_vocab;
    int32_t head_vocab_start;
    int32_t rank;
    int32_t world_size;
    int32_t mm2_k_inner;
    int32_t min_stop_step;
    float beta_margin;
    float beta_depth;
    float norm_eps;
    float t_base_ms;
    float c_w_ms;
    float log_p_stop;
    int32_t step_m_rows[32];
    int32_t step_k_top[32];
};
static_assert(sizeof(DraftWalkArgs) == 2304, "DraftWalkArgs ABI mismatch");

__aicore__ __attribute__((noinline)) void CubeMm1(
    AscendC::TPipe& pipe, uint64_t a_gm_addr, uint64_t b_nz_gm_addr, uint64_t c_gm_addr,
    int32_t eff)
{
    if ASCEND_IS_AIC {
        constexpr int32_t kTile = 16;
        constexpr int32_t kInner = 128;
        constexpr int32_t kHidden = 6144;
        constexpr int32_t kSteps = kHidden / kInner;
        constexpr int32_t kStage = kTile * kInner;
        constexpr int32_t kAcc = kTile * kTile;
        constexpr int32_t kNTotal = dspine_shared::kDspineShardDim / kTile;

        AscendC::TBuf<AscendC::TPosition::A1> a_stage;
        AscendC::TBuf<AscendC::TPosition::B1> b_stage;
        AscendC::TBuf<AscendC::TPosition::A2> a_op;
        AscendC::TBuf<AscendC::TPosition::B2> b_op;
        AscendC::TBuf<AscendC::TPosition::CO1> c_acc;
        pipe.InitBuffer(a_stage, 2 * kStage * sizeof(bfloat16_t));
        pipe.InitBuffer(b_stage, 2 * kStage * sizeof(bfloat16_t));
        pipe.InitBuffer(a_op, 2 * kStage * sizeof(bfloat16_t));
        pipe.InitBuffer(b_op, 2 * kStage * sizeof(bfloat16_t));
        pipe.InitBuffer(c_acc, kAcc * sizeof(float));

        auto a_l1 = a_stage.Get<bfloat16_t>();
        auto b_l1 = b_stage.Get<bfloat16_t>();
        auto a_l0 = a_op.Get<bfloat16_t>();
        auto b_l0 = b_op.Get<bfloat16_t>();
        auto c_l0 = c_acc.Get<float>();

        AscendC::GlobalTensor<bfloat16_t> a_gm, b_gm, c_gm;
        a_gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(a_gm_addr), eff * kHidden);
        b_gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(b_nz_gm_addr),
            kNTotal * kSteps * kStage);
        c_gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(c_gm_addr),
            eff * dspine_shared::kDspineShardDim);

        const int32_t cid = AscendC::GetBlockIdx();
        for (int32_t col_tile = cid; col_tile < kNTotal; col_tile += 24) {
            AscendC::Nd2NzParams copy_a{};
            copy_a.ndNum = 1;
            copy_a.nValue = eff;
            copy_a.dValue = kInner;
            copy_a.srcDValue = kHidden;
            copy_a.dstNzC0Stride = kTile;
            copy_a.dstNzNStride = 1;

            AscendC::LoadData2DParams load_ab{};
            load_ab.repeatTimes = kInner / kTile;
            load_ab.srcStride = 1;
            load_ab.ifTranspose = false;

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);

            for (int32_t s = 0; s < kSteps; ++s) {
                const int32_t buf = s & 1;
                const event_t ev = buf ? EVENT_ID1 : EVENT_ID0;
                const int64_t a_off = static_cast<int64_t>(s) * kInner;
                const int64_t b_off =
                    (static_cast<int64_t>(col_tile) * kSteps + s) * kStage;

                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ev);
                AscendC::DataCopy(a_l1[buf * kStage], a_gm[a_off], copy_a);
                AscendC::DataCopy(b_l1[buf * kStage], b_gm[b_off], kStage);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ev);

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ev);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ev);
                AscendC::LoadData(a_l0[buf * kStage], a_l1[buf * kStage], load_ab);
                AscendC::LoadData(b_l0[buf * kStage], b_l1[buf * kStage], load_ab);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ev);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ev);

                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ev);
                AscendC::MmadParams mad{};
                mad.m = kTile;
                mad.n = kTile;
                mad.k = kInner;
                mad.cmatrixInitVal = (s == 0);
                mad.cmatrixSource = false;
                AscendC::Mmad(c_l0, a_l0[buf * kStage], b_l0[buf * kStage], mad);
                AscendC::PipeBarrier<PIPE_M>();
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ev);
            }
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);

            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::FixpipeParamsV220 store{};
            store.nSize = kTile;
            store.mSize = eff;
            store.srcStride = kTile;
            store.dstStride = dspine_shared::kDspineShardDim;
            store.ndNum = 1;
            store.quantPre = QuantMode_t::F322BF16;
            AscendC::Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                c_gm[col_tile * kTile], c_l0, store);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        pipe.Reset();
    }
}

template <int32_t KInner, int32_t NSub>
__aicore__ __attribute__((noinline)) void CubeMm2Impl(
    AscendC::TPipe& pipe, uint64_t a_gm_addr, uint64_t b_nz_gm_addr, uint64_t c_gm_addr,
    int32_t eff)
{
    if ASCEND_IS_AIC {
        constexpr int32_t kTotal = dspine_shared::kDspineEhIn;
        constexpr int32_t kSteps = kTotal / KInner;
        constexpr int32_t k16 = KInner / 16;
        constexpr int32_t kAicCols = 256;
        constexpr int32_t kNChunks = kAicCols / NSub;
        constexpr int32_t kN1 = NSub / 16;
        constexpr int32_t kBChunk = NSub * KInner;

        AscendC::TBuf<AscendC::TPosition::A1> a_stage;
        AscendC::TBuf<AscendC::TPosition::B1> b_stage;
        AscendC::TBuf<AscendC::TPosition::A2> a_op;
        AscendC::TBuf<AscendC::TPosition::B2> b_op;
        AscendC::TBuf<AscendC::TPosition::CO1> c_acc;
        pipe.InitBuffer(a_stage, 16 * kTotal * sizeof(bfloat16_t));
        pipe.InitBuffer(b_stage, 2 * kBChunk * sizeof(bfloat16_t));
        pipe.InitBuffer(a_op, 16 * kTotal * sizeof(bfloat16_t));
        pipe.InitBuffer(b_op, 2 * kBChunk * sizeof(bfloat16_t));
        pipe.InitBuffer(c_acc, 16 * kAicCols * sizeof(float));

        auto a_l1 = a_stage.Get<bfloat16_t>();
        auto b_l1 = b_stage.Get<bfloat16_t>();
        auto a_l0 = a_op.Get<bfloat16_t>();
        auto b_l0 = b_op.Get<bfloat16_t>();
        auto c_l0 = c_acc.Get<float>();

        AscendC::GlobalTensor<bfloat16_t> a_gm, b_gm, c_gm;
        a_gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(a_gm_addr), eff * kTotal);
        b_gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(b_nz_gm_addr),
            24 * kSteps * kNChunks * kBChunk);
        c_gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(c_gm_addr), eff * 6144);

        const int32_t cid = AscendC::GetBlockIdx();
        if (cid < 24) {
            AscendC::Nd2NzParams copy_a{};
            copy_a.ndNum = 1;
            copy_a.nValue = eff;
            copy_a.dValue = kTotal;
            copy_a.srcDValue = kTotal;
            copy_a.dstNzC0Stride = 16;
            copy_a.dstNzNStride = 1;
            AscendC::DataCopy(a_l1, a_gm, copy_a);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);

            AscendC::LoadData2DParams load_a{};
            load_a.repeatTimes = kTotal / 16;
            load_a.srcStride = 1;
            load_a.ifTranspose = false;
            AscendC::LoadData(a_l0, a_l1, load_a);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID2);

            AscendC::LoadData2DParams load_b{};
            load_b.repeatTimes = kN1 * k16;
            load_b.srcStride = 1;
            load_b.ifTranspose = false;

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);

            for (int32_t kc = 0; kc < kSteps; ++kc) {
                for (int32_t nt = 0; nt < kNChunks; ++nt) {
                    const int32_t step_idx = kc * kNChunks + nt;
                    const int32_t buf = step_idx & 1;
                    const event_t ev = buf ? EVENT_ID1 : EVENT_ID0;
                    const int64_t b_off =
                        (static_cast<int64_t>(cid * kSteps + kc) * kNChunks + nt) * kBChunk;

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ev);
                    AscendC::DataCopy(b_l1[buf * kBChunk], b_gm[b_off], kBChunk);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ev);

                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ev);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ev);
                    AscendC::LoadData(b_l0[buf * kBChunk], b_l1[buf * kBChunk], load_b);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ev);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ev);

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ev);
                    AscendC::MmadParams mad{};
                    mad.m = 16;
                    mad.n = NSub;
                    mad.k = KInner;
                    mad.cmatrixInitVal = (kc == 0);
                    mad.cmatrixSource = false;
                    AscendC::Mmad(
                        c_l0[nt * 16 * NSub],
                        a_l0[kc * k16 * 256],
                        b_l0[buf * kBChunk],
                        mad);
                    AscendC::PipeBarrier<PIPE_M>();
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ev);
                }
            }
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);

            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::FixpipeParamsV220 store{};
            store.nSize = kAicCols;
            store.mSize = eff;
            store.srcStride = 16;
            store.dstStride = 6144;
            store.ndNum = 1;
            store.quantPre = QuantMode_t::F322BF16;
            AscendC::Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                c_gm[cid * kAicCols], c_l0, store);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        pipe.Reset();
    }
}

__aicore__ inline void CubeMm2(
    AscendC::TPipe& pipe, uint64_t a_gm_addr, uint64_t b_nz_gm_addr, uint64_t c_gm_addr,
    int32_t eff, int32_t k_inner)
{
    if (k_inner == 64) {
        CubeMm2Impl<64, 64>(pipe, a_gm_addr, b_nz_gm_addr, c_gm_addr, eff);
    } else if (k_inner == 256) {
        CubeMm2Impl<256, 32>(pipe, a_gm_addr, b_nz_gm_addr, c_gm_addr, eff);
    } else if (k_inner == 768) {
        CubeMm2Impl<768, 16>(pipe, a_gm_addr, b_nz_gm_addr, c_gm_addr, eff);
    } else {
        CubeMm2Impl<128, 64>(pipe, a_gm_addr, b_nz_gm_addr, c_gm_addr, eff);
    }
}

template <int Ver = KERNEL_VER>
__global__ __mix__(1, 2) __schedmode__(1) __aicore__
void DraftWalkKernel(
    uint64_t sync_base, uint64_t walk_args_gm, uint64_t walk_steps_gm, int32_t step_count)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);
    dspine_shared::DcciRange(walk_args_gm, static_cast<int32_t>(sizeof(DraftWalkArgs)));
    dspine_shared::DcciRange(
        walk_steps_gm, static_cast<int32_t>(sizeof(ChainStep)) * step_count);
    const DraftWalkArgs& args = *reinterpret_cast<const DraftWalkArgs*>(walk_args_gm);
    dspine_shared::DcciRange(args.draft_ctrl_gm, 64);
    if (reinterpret_cast<__gm__ const int64_t*>(args.draft_ctrl_gm)[2] <= 0) {
        return;
    }
    const ChainStep* steps = reinterpret_cast<const ChainStep*>(walk_steps_gm);
    uint64_t cur_sync = sync_base;
    uint32_t win_seq = 0U;
    const uint64_t init_l78_win = steps[0].communication.window;
    uint32_t l78_epoch = 0U;
    if ASCEND_IS_AIV {
        win_seq = dspine_shared::DspineReadSeq(args.comm_win_gm);
        if (init_l78_win != 0ULL) {
            l78_epoch = whole_layer_tree::ReadCommunicationEpoch(
                reinterpret_cast<__gm__ uint8_t*>(init_l78_win));
        }
    }
    TPipe pipe;
    const ChainStep& step0 = steps[0];
    dspine_shared::RunBeamStepV3(
        pipe, -1, args.b0, args.b0, args.b0, args.b0, 0, args.k_sfa, 0,
        args.beta_margin, args.beta_depth,
        args.gather_out_gm, args.draft_ctrl_gm, args.seed_top_i_gm, args.seed_top_lp_gm,
        args.seed_rec_gm, args.prefix_topk_gm, args.anc_slots_gm, args.indices_buf_gm,
        args.active_toks_gm, args.active_scores_gm, args.active_raw_lp_gm,
        args.active_margins_gm, args.active_pids_gm, args.active_recs_gm, 0ULL,
        args.pool_scores_gm, args.pool_raw_lp_gm, args.pool_tokens_gm, args.pool_parents_gm,
        args.sin_cache_gm, args.cos_cache_gm,
        step0.addresses.attention.main_metadata[0],
        step0.addresses.attention.main_metadata[1],
        step0.addresses.attention.main_metadata[2],
        step0.addresses.attention.main_metadata[4],
        args.beam_work_gm, true, -1.0e30F, 2, 16, false);
    if ASCEND_IS_AIV {
        pipe.Reset();
    }
    AscendC::SyncAll<false>();

    int32_t slot_cur = args.b0;
    for (int32_t s = 0; s < step_count; ++s) {
        const ChainStep& step = steps[s];
        const uint64_t ql_gm = step.addresses.attention.main_metadata[3];
        dspine_shared::DcciRange(ql_gm, 64);
        const int32_t eff = *reinterpret_cast<__gm__ const int32_t*>(ql_gm);
        if (eff <= 0) {
            break;
        }
        const int32_t s_idx = (s < args.max_d) ? s : (args.max_d - 1);

#ifdef L78_ONEHOP_COMM
        uint32_t ep_emb = 0U;
        if ASCEND_IS_AIV {
            ep_emb = ++win_seq;
        }
        CubeMm1(pipe, args.active_recs_gm, args.rot_w_step_nz_gm[s_idx], args.rot_part_gm, eff);
        if ASCEND_IS_AIV {
            dspine_shared::Inline1HopEmbPush(
                pipe, args.comm_win_gm, ep_emb, args.active_toks_gm, args.emb_w_gm,
                args.prolog_comm_out_gm, eff, args.emb_vocab_start, args.emb_vocab_end,
                args.rank);
            pipe.Reset();
        }
        AscendC::SyncAll<false>();

        const uint64_t h_sq_global_gm =
            args.prolog_comm_out_gm + 17ULL * 6144ULL * sizeof(bfloat16_t);
        uint32_t ep_sq = 0U;
        if ASCEND_IS_AIV {
            ep_sq = ++win_seq;
        }
        dspine_shared::Inline1HopEmbRecvRotSq(
            pipe, args.comm_win_gm, ep_emb, ep_sq, args.active_toks_gm,
            args.rot_part_gm, args.rot_b_step_gm[s_idx], args.prolog_comm_out_gm,
            h_sq_global_gm, eff, args.emb_vocab_start, args.emb_vocab_end, args.rank);
        if ASCEND_IS_AIV {
            pipe.Reset();
        }
        dspine_shared::RunStep3cDualRmsNormSlice(
            pipe, args.prolog_comm_out_gm, args.enorm_step_gm[s_idx],
            args.hnorm_step_gm[s_idx], args.eh_in_gm, eff, args.rank, args.norm_eps, true,
            h_sq_global_gm, args.rot_part_gm);
        if ASCEND_IS_AIV {
            pipe.Reset();
        }
        AscendC::SyncAll<false>();

        CubeMm2(
            pipe, args.eh_in_gm, args.w_eh_step_nz_gm[s_idx], args.eh_comm_in_gm,
            eff, args.mm2_k_inner);
        AscendC::SyncAll<false>();

        uint32_t ep_ag = 0U;
        if ASCEND_IS_AIV {
            ep_ag = ++win_seq;
        }
        dspine_shared::Inline1HopRsAgBf16(
            pipe, args.comm_win_gm, ep_ag, args.eh_comm_in_gm,
            step.addresses.norm.input, eff, args.rank);
        if ASCEND_IS_AIV {
            pipe.Reset();
        }
#else
        CubeMm1(pipe, args.active_recs_gm, args.rot_w_step_nz_gm[s_idx], args.rot_part_gm, eff);
        dspine_shared::RunStep3cPrologPackEmbedPart(
            pipe, args.active_toks_gm, args.emb_w_gm, args.prolog_comm_in_gm,
            eff, args.emb_vocab_start, args.emb_vocab_end, args.rank, true, true);
        if ASCEND_IS_AIV {
            pipe.Reset();
        }
        AscendC::SyncAll<false>();

        if ASCEND_IS_AIV {
            ++win_seq;
        }
        const uint64_t h_sq_local_gm =
            args.prolog_comm_out_gm + 16ULL * 6144ULL * sizeof(bfloat16_t);
        const uint64_t h_sq_global_gm =
            args.prolog_comm_out_gm + 17ULL * 6144ULL * sizeof(bfloat16_t);
        dspine_shared::RunStep3cPrologRotBiasAndSq(
            pipe, args.rot_part_gm, args.rot_b_step_gm[s_idx], h_sq_local_gm, eff);
        if ASCEND_IS_AIV {
            pipe.Reset();
        }
        dspine_shared::Inline8p2Ar(
            pipe, args.comm_win_gm, win_seq, args.prolog_comm_in_gm,
            args.prolog_comm_out_gm, eff * 6144, args.rank,
            h_sq_local_gm, h_sq_global_gm, eff);
        if ASCEND_IS_AIV {
            pipe.Reset();
        }
        dspine_shared::RunStep3cDualRmsNormSlice(
            pipe, args.prolog_comm_out_gm, args.enorm_step_gm[s_idx],
            args.hnorm_step_gm[s_idx], args.eh_in_gm, eff, args.rank, args.norm_eps, true,
            h_sq_global_gm, args.rot_part_gm);
        if ASCEND_IS_AIV {
            pipe.Reset();
        }
        AscendC::SyncAll<false>();

        CubeMm2(
            pipe, args.eh_in_gm, args.w_eh_step_nz_gm[s_idx], args.eh_comm_in_gm,
            eff, args.mm2_k_inner);
        AscendC::SyncAll<false>();

        if ASCEND_IS_AIV {
            ++win_seq;
        }
        dspine_shared::Inline8p2Ar(
            pipe, args.comm_win_gm, win_seq, args.eh_comm_in_gm,
            step.addresses.norm.input, eff * 6144, args.rank);
        if ASCEND_IS_AIV {
            pipe.Reset();
        }
        AscendC::SyncAll<false>();
#endif
        if (step.sync != cur_sync) {
            SetSyncBaseAddr(step.sync);
            cur_sync = step.sync;
        }
        uint32_t step_red1_ep = 0U;
        if ASCEND_IS_AIV {
            if (init_l78_win != 0ULL && step.communication.window == init_l78_win) {
                step_red1_ep = l78_epoch + 1U;
            }
            l78_epoch += 3U;
        }
#ifdef L78_FUSE_HEAD_NORM
        RunLayer(
            pipe, false, step.config, step.addresses, step.communication,
            false, 0ULL, nullptr, 0ULL, nullptr, nullptr, false, nullptr, -1, eff,
            0ULL, false, nullptr, step_red1_ep,
            args.logit_norm_step_gm[s_idx], args.c2b_recs_out_gm,
            args.c2b_act_q_gm, args.c2b_act_s_gm,
            0ULL, 0ULL, 0, 0ULL, 0U);
        const bool skip_head_stage1 = (eff >= 1 && eff <= 4);
#else
        RunLayer(
            pipe, false, step.config, step.addresses, step.communication,
            false, 0ULL, nullptr, 0ULL, nullptr, nullptr, false, nullptr, -1, eff,
            0ULL, false, nullptr, step_red1_ep);
        const bool skip_head_stage1 = false;
#endif
        dspine_shared::Publish();
        AscendC::SyncAll<false>();
        if (cur_sync != sync_base) {
            SetSyncBaseAddr(sync_base);
            cur_sync = sync_base;
        }

        if ASCEND_IS_AIV {
            ++win_seq;
        }
        dspine_shared::RunDraftHeadTopk(
            pipe, 0ULL, step.addresses.bridge[1], step.communication.output,
            args.logit_norm_step_gm[s_idx], args.cal_head_w_q8_nz_gm, args.cal_head_s8_gm,
            args.c2b_recs_out_gm, args.c2b_act_q_gm, args.c2b_act_s_gm, args.c2b_acc_i32_gm,
            args.logits_work_gm, args.gather_in_gm, args.gather_out_gm, args.comm_win_gm,
            eff, args.head_valid_vocab, args.head_vocab_start,
            args.step_k_top[s], args.rank, args.world_size, args.norm_eps, win_seq,
            skip_head_stage1);

        const int32_t d = s + 1;
        const int32_t m_rows = args.step_m_rows[s];
        const int32_t nxt_b = (s + 1 < step_count) ? args.step_m_rows[s + 1] : 0;
        const ChainStep& step_nxt = (s + 1 < step_count) ? steps[s + 1] : step;
        const bool prefix_tail_only = (d + 1 <= 64 && nxt_b <= args.b0);
        dspine_shared::RunBeamStepV3(
            pipe, d, m_rows, m_rows, args.step_k_top[s], nxt_b, 0, args.k_sfa, slot_cur,
            args.beta_margin, args.beta_depth,
            args.gather_out_gm, args.draft_ctrl_gm, args.seed_top_i_gm, args.seed_top_lp_gm,
            args.seed_rec_gm, args.prefix_topk_gm, args.anc_slots_gm, args.indices_buf_gm,
            args.active_toks_gm, args.active_scores_gm, args.active_raw_lp_gm,
            args.active_margins_gm, args.active_pids_gm, args.active_recs_gm,
            args.c2b_recs_out_gm,
            args.pool_scores_gm, args.pool_raw_lp_gm, args.pool_tokens_gm, args.pool_parents_gm,
            args.sin_cache_gm, args.cos_cache_gm,
            step_nxt.addresses.attention.main_metadata[0],
            step_nxt.addresses.attention.main_metadata[1],
            step_nxt.addresses.attention.main_metadata[2],
            step_nxt.addresses.attention.main_metadata[4],
            args.beam_work_gm, true, args.log_p_stop, args.min_stop_step, args.width,
            prefix_tail_only);
        if ASCEND_IS_AIV {
            pipe.Reset();
        }
        AscendC::SyncAll<false>();
        dspine_shared::DcciRange(args.draft_ctrl_gm, 64);
        if (reinterpret_cast<__gm__ const int64_t*>(args.draft_ctrl_gm)[5] != 0) {
            break;
        }
        slot_cur += nxt_b;
    }

    dspine_shared::RunBuildTreeV3(
        pipe, args.width, args.min_w_act, args.max_pool_size,
        args.num_seq_ptrs, args.num_q_ptrs, args.num_rope_dsts,
        args.t_base_ms, args.c_w_ms,
        args.draft_ctrl_gm, args.pool_scores_gm, args.pool_raw_lp_gm,
        args.pool_tokens_gm, args.pool_parents_gm, args.pool_depths_gm,
        args.tree_tokens_gm, args.tree_parents_gm, args.tree_depths_gm, args.tree_mask_gm,
        args.t_ids_gm, args.t_pos_gm, args.t_slots_gm, args.t_slots_i64_gm,
        args.dyn_w_gm, args.full_seq_ptrs_gm, args.mc2_mask_gm,
        args.cos_cache_gm, args.sin_cache_gm, args.rope_dsts_gm, args.tree_work_gm, true);
    if ASCEND_IS_AIV {
        pipe.Reset();
        if (AscendC::GetBlockIdx() == 0) {
            dspine_shared::DspineWriteSeq(args.comm_win_gm, win_seq);
        }
        dspine_shared::Publish();
    }
}
"""

_DRAFT_WALK_HOST_SNIPPET = """
extern "C" int layer_tp_tree_draft_walk_launch(
    void* stream, uint64_t sync_base, uint64_t walk_args_gm,
    const layer_tp_tree_chain::ChainStep* steps_host,
    const void* steps_device, int32_t step_count)
{
    if (sync_base == 0ULL || walk_args_gm == 0ULL || (walk_args_gm % 64ULL) != 0ULL ||
        steps_host == nullptr || steps_device == nullptr ||
        (reinterpret_cast<uintptr_t>(steps_device) % 64ULL) != 0ULL ||
        step_count <= 0 || step_count > 32) {
        return -1;
    }
    for (int32_t i = 0; i < step_count; ++i) {
        const auto& s = steps_host[i];
        if (s.sync == 0ULL || s.addresses.body.blocks != 24 ||
            s.addresses.shared_topk == 0ULL ||
            !layer_tp_tree_chain::Valid(s.config, s.addresses, s.communication)) {
            return -1;
        }
    }
    layer_tp_tree_chain::DraftWalkKernel<<<24, nullptr, stream>>>(
        sync_base, walk_args_gm, reinterpret_cast<uint64_t>(steps_device), step_count);
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
    "-DK11083_RED2_OPT=1",
    "-DK11090_FAST_SFA_VEC=1",
    "-DK11110_SFA_PIPE=1",
    "-DK11120_SFA_GATHER_OPT=1",
    "-DK11120_SKIP_DUP_PUB=1",
    "-DK11150_SIP_RELAY_RED2=1",
    "-DK11180_QUAD4_SIP_ALL=1",
    "-DK11180_TOK_OUT_BAL=1",
    "-DK11180_NORM_OPT=1",
    "-DK112110_SIP_RED1=1",
    "-DK112120_CLEAN_DCCI=1",
    "-DK112140_PIPE_KEY_ONLY=1",
    "-DK112120_SFA_COMPACT_WS=1",
    "-DK112166_SFA_QV_L1=1",
    "-DK112188_SFA_QVL1_ALL=1",
    "-DK112170_DEC_SIP_ATT=1",
    "-DK112174_FAST_MERGE_KV=1",
    "-DK112176_PHASE_DSB_OPT=1",
    "-DK112194_PROLOG_CUBE_BAL=1",
    "-DK112512_PROLOG_48AIV_V2=1",
    "-DK112702_NO_DDR_INTRADIE=1",
    "-DK112722_NO_DDR_EXTRA=1",
    "-DK112824_NO_R2_END_DSB=1",
    "-DK112830_FAST_POLL_RED2=1",
    "-DK112612_TRIM_BARRIERS=1",
    "-DK112600_EXACT48_RED2_OPT=1",
    "-DK112514_IDX_PIPE_V2=1",
    "-DK11050_AIC_PREFETCH=1",
    "-DK11040_TAIL_NZ=1",
    "-DK11140_DIRECT_MLA_TAIL=1",
    "-DK11150_MLA_TAIL_M32=1",
    "-DK112168_MLA_TAIL_OVERLAP=1",
    "-DK112510_MLA_TAIL_XCORE_V2=1",
    "-DK11150_SHARED_W8_NZ=1",
    "-DK11080_DIRECT_W4A8=1",
    "-DK11088_GATE_CONTIG=1",
    "-DK11220_DOWN_CONTIG_256=1",
    "-DK11220_MULTI_M16=1",
    "-DK11220_XTASK_B=1",
    "-DK11220_FAST_TOK_OUT=1",
    "-DK11220_FAST_ACT_P3=1",
    "-DK11220_BITMASK_OWN_P1=1",
    "-DK11220_SH_ROW_1D=1",
    "-DK11220_LOCAL_SH_RED2=1",
    "-DK11220_PF_RED2_FULL=1",
    "-DK11220_FAST_P1_V3=1",
    "-DK11220_MULTI_M32=1",
    "-DK112612_BRIDGE_FAST=1",
    "-DK112612_DOWN_PIPE2=1",
    "-DK112726_BRIDGE_PRELOAD_GATE=1",
    "-DL2_CACHE_HINT=1",
    "-DASCENDC_DUMP=0",
    "-DK112600_PIPE_C2V=1",
    "-DK112612_NO_FLAG9=1",
    "-DSKIP_CHAIN_KERNEL=1",
    "-DK112612_OVERLAP_SH_P4=1",
)


_L78_EXTRA_UNSET_RE = re.compile(
    r"^(?:K\d+[A-Z0-9_]*|L78_[A-Z0-9_]+|SKIP_[A-Z0-9_]+|"
    r"ENABLE_KERNEL_PROF|SFA_PACKED_SHADOW_KV|ATTENTION_INDEXED_PROFILE)$"
)


def _is_l78_build_unset_macro(name: str) -> bool:
    return name in UNSET_L78_MACROS or _L78_EXTRA_UNSET_RE.match(name) is not None


def _l78_effective_source_digest(src: Path, defs: Sequence[str]) -> str:
    return effective_tree_digest(
        src,
        defs,
        subdirs=("local", "source", "vendor", "communication"),
        extra_texts=(
            DSPINE_WALK_SHARED_CPP,
            _KV_COMMIT_DEVICE_SNIPPET,
            _KV_COMMIT_HOST_SNIPPET,
            _DRAFT_WALK_DEVICE_SNIPPET,
            _DRAFT_WALK_HOST_SNIPPET,
        ),
        unset_macro_fn=_is_l78_build_unset_macro,
    )


def _l78_source_digest(src: Path, world_size: int = 8) -> str:
    return _l78_effective_source_digest(src, _l78_defs_for_world(world_size))


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


def _l78_defs_for_world(world_size: int = 8) -> tuple[str, ...]:
    ws = int(world_size)
    if ws not in (8, 16):
        raise ValueError(f"Unsupported world_size={world_size!r}; expected 8 or 16")
    heads = 64 // ws
    local_exp = 256 // ws
    if ws == 16:
        drop = {"-DK112612_OVERLAP_SH_P4=1"}
        base = tuple(
            d
            for d in L78_BASE_DEFS
            if not d.startswith("-DATTENTION_HEADS=") and d not in drop
        )
        extra = (
            "-DK112140_SIP_RED1_MIN_ROWS=8",
            "-DK112184_IDX_FRONT_OPT=1",
            "-DK112192_DIRECT_IDX_Q14=1",
            "-DK112160_AIV_DSB_OPT=1",
            "-DK112162_MERGE_CONTIG=1",
            "-DK112612_DEFER_SH_SIG=1",
            "-DK112612_SPLIT_P1_OPT=1",
            "-DK112808_PULL_RED1=1",
            "-DK112808_PULL_RED2=1",
            "-DK112854_TP16_PROLOG_PAR=1",
            "-DK112860_KNOWN_RED1_EPOCH=1",
            "-DK113000_BRIDGE_GATE_OVERLAP=1",
            "-DK113000_WAVE16_RED2=1",
            "-DL78_ONEHOP_COMM=1",
            "-DL78_FUSE_HEAD_NORM=1",
        )
    else:
        base = tuple(d for d in L78_BASE_DEFS if not d.startswith("-DATTENTION_HEADS="))
        extra = ()
    return (
        *base,
        f"-DATTENTION_HEADS={heads}",
        f"-DWORLD_SIZE={ws}",
        f"-DROUTED_LOCAL_EXPERTS={local_exp}",
        *extra,
    )


def _l78_variant_digest(
    src_or_digest: str | Path,
    ver: int,
    k_val: int,
    world_size: int = 8,
) -> str:
    if isinstance(src_or_digest, Path):
        defs = (
            f"-DKERNEL_VER={ver}",
            f"-DDRAFT_SFA_TOPK={k_val}",
            *_l78_defs_for_world(world_size),
        )
        src_digest = _l78_effective_source_digest(src_or_digest, defs)
    else:
        src_digest = src_or_digest
    payload = {
        "source_digest": src_digest,
        "ver": ver,
        "k_sfa": k_val,
        "defs": list(_l78_defs_for_world(world_size)),
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
    dev_inject = (
        DSPINE_WALK_SHARED_CPP
        + "\n"
        + _KV_COMMIT_DEVICE_SNIPPET
        + "\n"
        + _DRAFT_WALK_DEVICE_SNIPPET
    )
    host_inject = _KV_COMMIT_HOST_SNIPPET + "\n" + _DRAFT_WALK_HOST_SNIPPET
    kern_txt = kern_txt.replace(anchor_dev, dev_inject + "\n" + anchor_dev, 1)
    kern_txt = kern_txt.replace(anchor_host, host_inject + "\n" + anchor_host, 1)
    idx_launch_old = (
        "    layer_tp_tree_chain::Kernel<true><<<addresses->body.blocks, nullptr, stream>>>(\n"
        "        sync, *config, *addresses, *communication);"
    )
    idx_launch_new = (
        "    if (layer_tp_tree_chain::g_exact_wk_w_gm != 0ULL) {\n"
        "        layer_tp_tree_chain::LayerExactKernel<true>"
        "<<<addresses->body.blocks, nullptr, stream>>>(\n"
        "            sync, *config, *addresses, *communication,\n"
        "            layer_tp_tree_chain::g_exact_wk_w_gm,\n"
        "            layer_tp_tree_chain::g_exact_knorm_w_gm,\n"
        "            layer_tp_tree_chain::g_exact_knorm_b_gm,\n"
        "            layer_tp_tree_chain::g_exact_kcache_gm,\n"
        "            layer_tp_tree_chain::g_exact_work_gm,\n"
        "            layer_tp_tree_chain::g_exact_eps,\n"
        "            layer_tp_tree_chain::g_exact_seed_args_gm);\n"
        "    } else {\n"
        "        layer_tp_tree_chain::Kernel<true><<<addresses->body.blocks, nullptr, stream>>>(\n"
        "            sync, *config, *addresses, *communication);\n"
        "    }"
    )
    if idx_launch_old in kern_txt:
        kern_txt = kern_txt.replace(idx_launch_old, idx_launch_new, 1)

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
    world_size: int = 8,
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
    base_defs_block = "\n".join(
        f"        {shlex.quote(d)}" for d in _l78_defs_for_world(world_size)
    )
    compile_lines = ["pids=()"]
    for k_val in target_ks:
        ver, sfx = L78_VARIANT_SPECS[k_val]
        compile_lines.append(
            f"compile_variant {ver} {sfx} -DDRAFT_SFA_TOPK={k_val} &"
        )
        compile_lines.append('pids+=("$!")')
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
        -Dlayer_tp_tree_draft_walk_launch=layer_tp_tree_draft_walk_launch_K$ver
        -Dlayer_tp_tree_set_mask_ptr=layer_tp_tree_set_mask_ptr_K$ver
        -Dlayer_tp_tree_set_exact_idx_ptrs=layer_tp_tree_set_exact_idx_ptrs_K$ver
        -Dlayer_tp_tree_set_seed_args_ptr=layer_tp_tree_set_seed_args_ptr_K$ver
        -Dlayer_tp_tree_set_peer_weights_ptr=layer_tp_tree_set_peer_weights_ptr_K$ver
        -Dlayer_tp_tree_prepare_records=layer_tp_tree_prepare_records_K$ver
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
for pid in "${{pids[@]}}"; do
    wait "$pid"
done
"""


def build_draft_l78_k110419(
    src_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    k_sfa: int | None = None,
    variants: tuple[int, ...] | list[int] | None = None,
    world_size: int = 8,
) -> tuple[Path, Path]:
    ws = int(world_size)
    src = _default_l78_src_dir() if src_dir is None else Path(src_dir)
    out = _default_l78_out_dir() if out_dir is None else Path(out_dir)
    target_ks = _resolve_l78_ksfa(k_sfa, variants)
    so_1024 = out / "chain_K110419.so"
    so_512 = out / "chain_K110420.so"
    src_digest = _l78_source_digest(src, ws)
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
        expected_vdig = _l78_variant_digest(src, ver, k_val, ws)
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
    build_sh = _render_l78_build_script(
        out, _cann_root(), _site_root(), tuple(missing_or_stale), ws
    )
    sh_path = out / "build_p2.sh"
    sh_path.write_text(build_sh, encoding="utf-8")
    subprocess.run(["bash", str(sh_path)], check=True)

    next_variants: dict[str, dict[str, object]] = {}
    for k_val, (ver, _) in L78_VARIANT_SPECS.items():
        so_p = out / f"chain_K{ver}.so"
        if not so_p.is_file():
            continue
        expected_vdig = _l78_variant_digest(src, ver, k_val, ws)
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
