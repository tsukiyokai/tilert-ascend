import argparse
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[4]
DEFAULT_OUT = REPO / "tests/.artifacts/builds/dspine_ops"

ASC_SUBDIRS = (
    "asc",
    "asc/include",
    "asc/include/adv_api",
    "asc/include/basic_api",
    "asc/include/c_api",
    "asc/include/simt_api",
    "asc/include/utils",
    "asc/include/basic_api/reg_compute",
    "asc/impl/adv_api",
    "asc/impl/basic_api",
    "asc/impl/c_api",
    "asc/impl/simt_api",
    "asc/impl/utils",
    "asc/impl/basic_api/reg_compute",
    "tikcpp/tikcfw",
    "tikcpp/tikcfw/lib",
    "tikcpp/tikcfw/impl",
    "tikcpp/tikcfw/interface",
)

KERNEL_CPP = """#include "kernel_operator.h"

namespace dspine_ops {
using namespace AscendC;

template <HardEvent Ev>
__aicore__ inline void Fence()
{
    const auto id = GetTPipePtr()->FetchEventID(Ev);
    SetFlag<Ev>(id);
    WaitFlag<Ev>(id);
}

__aicore__ inline void Publish()
{
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
}

__aicore__ inline void RunPrologPack(
    TPipe& pipe, uint64_t ctrl_gm, uint64_t target_toks_gm, uint64_t emb_w_gm,
    uint64_t rot_part_gm, uint64_t comm_in_gm,
    int32_t m_max, int32_t vocab_start, int32_t vocab_end, int32_t rank)
{
    if ASCEND_IS_AIV {
        TBuf<TPosition::VECCALC> b_ctrl, b_ttoks, b_bf16;
        pipe.InitBuffer(b_ctrl, 48 * 8);
        pipe.InitBuffer(b_ttoks, 32 * 8);
        pipe.InitBuffer(b_bf16, 768 * 2);
        auto ub_ctrl = b_ctrl.Get<int64_t>();
        auto ub_ttoks = b_ttoks.Get<int64_t>();
        auto ub_bf16 = b_bf16.Get<bfloat16_t>();

        GlobalTensor<int64_t> g_ctrl, g_ttoks;
        GlobalTensor<bfloat16_t> g_emb, g_rot, g_out;
        g_ctrl.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ctrl_gm), 48);
        g_ttoks.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(target_toks_gm), 32);
        g_emb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(emb_w_gm));
        g_rot.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(rot_part_gm), m_max * 768);
        g_out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(comm_in_gm), 2 * m_max * 6144);

        DataCopy(ub_ctrl, g_ctrl, 48);
        DataCopy(ub_ttoks, g_ttoks, 32);
        Fence<HardEvent::MTE2_S>();

        const int32_t count = static_cast<int32_t>(ub_ctrl.GetValue(2));
        const int32_t aiv = GetBlockIdx();

        for (int32_t task = aiv; task < m_max * 12; task += 48) {
            const int32_t r_idx = task / 12;
            const int64_t col = static_cast<int64_t>(task % 12) * 512;
            const int64_t row_id = ub_ctrl.GetValue(3 + r_idx);
            const int64_t tok = ub_ttoks.GetValue(row_id);
            if (r_idx < count && tok >= vocab_start && tok < vocab_end) {
                DataCopy(ub_bf16, g_emb[(tok - vocab_start) * 6144 + col], 512);
                Fence<HardEvent::MTE2_MTE3>();
            } else {
                Duplicate(ub_bf16.ReinterpretCast<int16_t>(), static_cast<int16_t>(0), 512);
                Fence<HardEvent::V_MTE3>();
            }
            DataCopy(g_out[r_idx * 6144 + col], ub_bf16, 512);
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::MTE3_MTE2>();
        }

        for (int32_t task = aiv; task < m_max * 8; task += 48) {
            const int32_t r_idx = task / 8;
            const int32_t s_rk = task % 8;
            const int64_t dst_off = static_cast<int64_t>(m_max + r_idx) * 6144
                + static_cast<int64_t>(s_rk) * 768;
            if (r_idx < count && s_rk == rank) {
                DataCopy(ub_bf16, g_rot[r_idx * 768], 768);
                Fence<HardEvent::MTE2_MTE3>();
            } else {
                Duplicate(ub_bf16.ReinterpretCast<int16_t>(), static_cast<int16_t>(0), 768);
                Fence<HardEvent::V_MTE3>();
            }
            DataCopy(g_out[dst_off], ub_bf16, 768);
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::MTE3_MTE2>();
        }
        Publish();
    }
}

__aicore__ inline void InsertTop4(
    float val, int64_t idx,
    float& v0, float& v1, float& v2, float& v3,
    int64_t& i0, int64_t& i1, int64_t& i2, int64_t& i3)
{
    if (val <= v3) {
        return;
    }
    if (val > v0) {
        v3 = v2; i3 = i2;
        v2 = v1; i2 = i1;
        v1 = v0; i1 = i0;
        v0 = val; i0 = idx;
    } else if (val > v1) {
        v3 = v2; i3 = i2;
        v2 = v1; i2 = i1;
        v1 = val; i1 = idx;
    } else if (val > v2) {
        v3 = v2; i3 = i2;
        v2 = val; i2 = idx;
    } else {
        v3 = val; i3 = idx;
    }
}

__aicore__ inline void DecodeGatheredTopKV3(
    LocalTensor<float>& ub_gat,
    LocalTensor<float>& ub_mr,
    LocalTensor<float>& ub_sr,
    LocalTensor<float>& ub_sum,
    LocalTensor<float>& ub_work,
    int32_t m_rows,
    int32_t k,
    LocalTensor<int64_t>& ub_out_i,
    LocalTensor<float>& ub_out_lp)
{
    const int32_t cols = 2 + 2 * k;
    for (int32_t m = 0; m < m_rows; ++m) {
        float tv0 = -1.0e30F, tv1 = -1.0e30F, tv2 = -1.0e30F, tv3 = -1.0e30F;
        int64_t ti0 = 0, ti1 = 0, ti2 = 0, ti3 = 0;
        float max_all = -1.0e30F;
        for (int32_t r = 0; r < 8; ++r) {
            const int32_t base = r * 64 + m * cols;
            const float mr = ub_gat.GetValue(base + 0);
            if (mr > max_all) {
                max_all = mr;
            }
            for (int32_t j = 0; j < k; ++j) {
                InsertTop4(
                    ub_gat.GetValue(base + 2 + j),
                    static_cast<int64_t>(ub_gat.GetValue(base + 2 + k + j)),
                    tv0, tv1, tv2, tv3, ti0, ti1, ti2, ti3);
            }
        }
        Fence<HardEvent::S_V>();
        Duplicate(ub_mr, -1.0e30F, 8);
        Duplicate(ub_sr, 0.0F, 8);
        Fence<HardEvent::V_S>();
        for (int32_t r = 0; r < 8; ++r) {
            const int32_t base = r * 64 + m * cols;
            ub_mr.SetValue(r, ub_gat.GetValue(base + 0));
            ub_sr.SetValue(r, ub_gat.GetValue(base + 1));
        }
        Fence<HardEvent::S_V>();
        Adds(ub_mr, ub_mr, -max_all, 8);
        PipeBarrier<PIPE_V>();
        Maxs(ub_mr, ub_mr, -104.0F, 8);
        PipeBarrier<PIPE_V>();
        Exp(ub_mr, ub_mr, 8);
        PipeBarrier<PIPE_V>();
        Mul(ub_mr, ub_mr, ub_sr, 8);
        PipeBarrier<PIPE_V>();
        ReduceSum(ub_sum, ub_mr, ub_work, 8);
        PipeBarrier<PIPE_V>();
        Ln(ub_sum, ub_sum, 8);
        Fence<HardEvent::V_S>();
        const float lse = max_all + ub_sum.GetValue(0);
        const float lp0 = (tv0 - lse > -2.0e-4F) ? -2.0e-4F : (tv0 - lse);
        const float lp1 = (tv1 - lse > -2.0e-4F) ? -2.0e-4F : (tv1 - lse);
        const float lp2 = (tv2 - lse > -2.0e-4F) ? -2.0e-4F : (tv2 - lse);
        const float lp3 = (tv3 - lse > -2.0e-4F) ? -2.0e-4F : (tv3 - lse);
        ub_out_i.SetValue(m * 4 + 0, ti0);
        ub_out_i.SetValue(m * 4 + 1, ti1);
        ub_out_i.SetValue(m * 4 + 2, ti2);
        ub_out_i.SetValue(m * 4 + 3, ti3);
        ub_out_lp.SetValue(m * 4 + 0, lp0);
        ub_out_lp.SetValue(m * 4 + 1, lp1);
        ub_out_lp.SetValue(m * 4 + 2, lp2);
        ub_out_lp.SetValue(m * 4 + 3, lp3);
    }
}

__aicore__ inline void RunPrepCommitSeedV3(
    TPipe& pipe,
    uint64_t ctrl_gm,
    uint64_t hs_all_gm,
    uint64_t sin_cache_gm,
    uint64_t cos_cache_gm,
    uint64_t kv_hs_gm,
    uint64_t kv_sin_gm,
    uint64_t kv_cos_gm,
    uint64_t kv_slot_gm,
    uint64_t kv_qlen_gm,
    uint64_t kv_kvlen_gm,
    uint64_t seed_hs_gm,
    uint64_t seed_sin_gm,
    uint64_t seed_cos_gm,
    uint64_t seed_slot_gm,
    uint64_t seed_qlen_gm,
    uint64_t seed_kvlen_gm,
    int32_t kv_m_cap)
{
    if ASCEND_IS_AIV {
        const int32_t aiv = GetBlockIdx();
        TBuf<TPosition::VECCALC> b_ctrl, b_bf16, b_i64, b_i32;
        pipe.InitBuffer(b_ctrl, 48 * 8);
        pipe.InitBuffer(b_bf16, 512 * 2);
        pipe.InitBuffer(b_i64, 32 * 8);
        pipe.InitBuffer(b_i32, 32 * 4);
        auto ub_ctrl = b_ctrl.Get<int64_t>();
        auto ub_bf16 = b_bf16.Get<bfloat16_t>();
        auto ub_i64 = b_i64.Get<int64_t>();
        auto ub_i32 = b_i32.Get<int32_t>();

        GlobalTensor<int64_t> g_ctrl;
        g_ctrl.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ctrl_gm), 48);
        DataCopy(ub_ctrl, g_ctrl, 48);
        Fence<HardEvent::MTE2_S>();

        const int64_t main_start = ub_ctrl.GetValue(0);
        int32_t count = static_cast<int32_t>(ub_ctrl.GetValue(1));
        if (count < 1) {
            count = 1;
        }
        if (count > kv_m_cap + 1) {
            count = kv_m_cap + 1;
        }
        const int32_t kv_rows = count - 1;
        const int32_t seed_idx = count - 1;
        const int64_t seed_pos = main_start + static_cast<int64_t>(seed_idx);

        GlobalTensor<bfloat16_t> g_hs_all, g_kv_hs, g_seed_hs;
        GlobalTensor<bfloat16_t> g_sin_c, g_cos_c, g_kv_sin, g_kv_cos, g_s_sin, g_s_cos;
        g_hs_all.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(hs_all_gm), (kv_m_cap + 4) * 6144);
        g_kv_hs.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(kv_hs_gm), kv_m_cap * 6144);
        g_seed_hs.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(seed_hs_gm), 6144);
        g_sin_c.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(sin_cache_gm));
        g_cos_c.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cos_cache_gm));
        g_kv_sin.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(kv_sin_gm), kv_m_cap * 64);
        g_kv_cos.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(kv_cos_gm), kv_m_cap * 64);
        g_s_sin.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(seed_sin_gm), 64);
        g_s_cos.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(seed_cos_gm), 64);

        for (int32_t task = aiv; task < kv_m_cap * 12; task += 48) {
            const int32_t r = task / 12;
            const int64_t col = static_cast<int64_t>(task % 12) * 512;
            const int32_t src_r = (r < kv_rows) ? r : 0;
            DataCopy(ub_bf16, g_hs_all[src_r * 6144 + col], 512);
            Fence<HardEvent::MTE2_MTE3>();
            DataCopy(g_kv_hs[r * 6144 + col], ub_bf16, 512);
            Fence<HardEvent::MTE3_MTE2>();
        }

        if (aiv >= 12 && aiv < 24) {
            const int64_t col = static_cast<int64_t>(aiv - 12) * 512;
            DataCopy(ub_bf16, g_hs_all[seed_idx * 6144 + col], 512);
            Fence<HardEvent::MTE2_MTE3>();
            DataCopy(g_seed_hs[col], ub_bf16, 512);
            Fence<HardEvent::MTE3_MTE2>();
        }

        if (aiv >= 24 && aiv < 24 + kv_m_cap) {
            const int32_t r = aiv - 24;
            const int64_t p = (r < kv_rows) ? (main_start + r) : main_start;
            DataCopy(ub_bf16[0], g_sin_c[p * 64], 64);
            DataCopy(ub_bf16[64], g_cos_c[p * 64], 64);
            Fence<HardEvent::MTE2_MTE3>();
            DataCopy(g_kv_sin[r * 64], ub_bf16[0], 64);
            DataCopy(g_kv_cos[r * 64], ub_bf16[64], 64);
        }

        if (aiv == 47) {
            DataCopy(ub_bf16[0], g_sin_c[seed_pos * 64], 64);
            DataCopy(ub_bf16[64], g_cos_c[seed_pos * 64], 64);
            Fence<HardEvent::MTE2_MTE3>();
            DataCopy(g_s_sin, ub_bf16[0], 64);
            DataCopy(g_s_cos, ub_bf16[64], 64);
        }

        if (aiv == 0) {
            for (int32_t r = 0; r < kv_m_cap; ++r) {
                ub_i64.SetValue(r, (r < kv_rows) ? (main_start + r) : (9296LL + r));
            }
            ub_i64.SetValue(16, seed_pos);
            ub_i32.SetValue(0, kv_rows);
            ub_i32.SetValue(8, static_cast<int32_t>(main_start + kv_rows));
            ub_i32.SetValue(16, 1);
            ub_i32.SetValue(24, static_cast<int32_t>(seed_pos + 1));
            Fence<HardEvent::S_MTE3>();

            GlobalTensor<int64_t> g_kv_sl, g_s_sl;
            GlobalTensor<int32_t> g_kv_ql, g_kv_kvl, g_s_ql, g_s_kvl;
            g_kv_sl.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(kv_slot_gm), kv_m_cap);
            g_s_sl.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(seed_slot_gm), 1);
            g_kv_ql.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(kv_qlen_gm), 1);
            g_kv_kvl.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(kv_kvlen_gm), 1);
            g_s_ql.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seed_qlen_gm), 1);
            g_s_kvl.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seed_kvlen_gm), 1);

            DataCopyPad(g_kv_sl, ub_i64[0], DataCopyExtParams{
                1, static_cast<uint32_t>(kv_m_cap * sizeof(int64_t)), 0, 0, 0});
            DataCopyPad(g_s_sl, ub_i64[16], DataCopyExtParams{1, sizeof(int64_t), 0, 0, 0});
            DataCopyPad(g_kv_ql, ub_i32[0], DataCopyExtParams{1, sizeof(int32_t), 0, 0, 0});
            DataCopyPad(g_kv_kvl, ub_i32[8], DataCopyExtParams{1, sizeof(int32_t), 0, 0, 0});
            DataCopyPad(g_s_ql, ub_i32[16], DataCopyExtParams{1, sizeof(int32_t), 0, 0, 0});
            DataCopyPad(g_s_kvl, ub_i32[24], DataCopyExtParams{1, sizeof(int32_t), 0, 0, 0});
        }
        Publish();
    }
}

__aicore__ inline void RunBeamStepV3(
    TPipe& pipe,
    int32_t step,
    int32_t b_cur,
    int32_t k_expand,
    int32_t k_topk,
    int32_t b_next,
    int32_t pool_off,
    int32_t k_sfa,
    int32_t next_slot_cur,
    float beta_margin,
    float beta_depth,
    uint64_t gather_out_gm,
    uint64_t draft_ctrl_gm,
    uint64_t seed_top_i_gm,
    uint64_t seed_top_lp_gm,
    uint64_t seed_rec_gm,
    uint64_t prefix_topk_gm,
    uint64_t anc_slots_gm,
    uint64_t indices_buf_gm,
    uint64_t active_toks_gm,
    uint64_t active_scores_gm,
    uint64_t active_raw_lp_gm,
    uint64_t active_margins_gm,
    uint64_t active_pids_gm,
    uint64_t active_recs_gm,
    uint64_t recs_d_gm,
    uint64_t pool_scores_gm,
    uint64_t pool_raw_lp_gm,
    uint64_t pool_tokens_gm,
    uint64_t pool_parents_gm,
    uint64_t sin_cache_gm,
    uint64_t cos_cache_gm,
    uint64_t eshr_sin_gm,
    uint64_t eshr_cos_gm,
    uint64_t eshr_slot_gm,
    uint64_t eshr_kvlen_gm,
    uint64_t work_gm)
{
    if ASCEND_IS_AIV {
        const int32_t aiv = GetBlockIdx();
        TBuf<TPosition::VECCALC> b_gat, b_mr, b_sr, b_sum, b_wk, b_i64, b_f32, b_anc, b_bf16;
        pipe.InitBuffer(b_gat, 512 * 4);
        pipe.InitBuffer(b_mr, 64);
        pipe.InitBuffer(b_sr, 64);
        pipe.InitBuffer(b_sum, 64);
        pipe.InitBuffer(b_wk, 256);
        pipe.InitBuffer(b_i64, 256 * 8);
        pipe.InitBuffer(b_f32, 256 * 4);
        pipe.InitBuffer(b_anc, 256 * 4);
        pipe.InitBuffer(b_bf16, 2048 * 2);

        auto ub_gat = b_gat.Get<float>();
        auto ub_mr = b_mr.Get<float>();
        auto ub_sr = b_sr.Get<float>();
        auto ub_sum = b_sum.Get<float>();
        auto ub_wk = b_wk.Get<float>();
        auto ub_i64 = b_i64.Get<int64_t>();
        auto ub_f32 = b_f32.Get<float>();
        auto ub_anc = b_anc.Get<int32_t>();
        auto ub_bf16 = b_bf16.Get<bfloat16_t>();

        if (step == 0) {
            GlobalTensor<int32_t> g_idx, g_pre;
            g_idx.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices_buf_gm), 2048);
            g_pre.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(prefix_topk_gm), 2048);
            auto ub_idx = ub_bf16.ReinterpretCast<int32_t>();
            for (int32_t task = aiv; task < 32; task += 48) {
                const int64_t off = static_cast<int64_t>(task) * 64;
                DataCopy(ub_idx, g_idx[off], 64);
                Fence<HardEvent::MTE2_MTE3>();
                DataCopy(g_pre[off], ub_idx, 64);
                Fence<HardEvent::MTE3_MTE2>();
            }
            if (aiv == 0) {
                GlobalTensor<float> g_gat, g_slp;
                GlobalTensor<int64_t> g_si;
                g_gat.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gather_out_gm), 512);
                g_si.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(seed_top_i_gm), 8);
                g_slp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(seed_top_lp_gm), 8);
                DataCopy(ub_gat, g_gat, 512);
                Fence<HardEvent::MTE2_S>();
                auto ub_dec_i = ub_i64[64];
                auto ub_dec_lp = ub_f32[64];
                const int32_t k_dec = (k_topk >= b_next) ? k_topk : b_next;
                DecodeGatheredTopKV3(
                    ub_gat, ub_mr, ub_sr, ub_sum, ub_wk, 1, k_dec, ub_dec_i, ub_dec_lp);
                for (int32_t j = 0; j < 8; ++j) {
                    ub_i64.SetValue(j, j < b_next ? ub_dec_i.GetValue(j) : 0);
                    ub_f32.SetValue(j, j < b_next ? ub_dec_lp.GetValue(j) : 0.0F);
                }
                float mg0 = ub_dec_lp.GetValue(0) - ub_dec_lp.GetValue(1);
                if (mg0 < 0.0F) {
                    mg0 = 0.0F;
                }
                ub_f32.SetValue(7, mg0);
                Fence<HardEvent::S_MTE3>();
                DataCopy(g_si, ub_i64, 8);
                DataCopy(g_slp, ub_f32, 8);
            }
            Publish();
            return;
        }

        if (step == -1) {
            GlobalTensor<int64_t> g_ctrl;
            g_ctrl.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(draft_ctrl_gm), 8);
            DataCopy(ub_i64, g_ctrl, 8);
            Fence<HardEvent::MTE2_S>();
            const int64_t pending_tok = ub_i64.GetValue(0);
            const int64_t base_pos = ub_i64.GetValue(1);

            GlobalTensor<bfloat16_t> g_srec, g_arec, g_sin, g_cos, g_esin, g_ecos;
            g_srec.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(seed_rec_gm), 6144);
            g_arec.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t*>(active_recs_gm), b_next * 6144);
            g_sin.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(sin_cache_gm));
            g_cos.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cos_cache_gm));
            g_esin.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(eshr_sin_gm), b_next * 64);
            g_ecos.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(eshr_cos_gm), b_next * 64);

            for (int32_t task = aiv; task < 12; task += 48) {
                const int64_t col = static_cast<int64_t>(task) * 512;
                DataCopy(ub_bf16, g_srec[col], 512);
                Fence<HardEvent::MTE2_MTE3>();
                for (int32_t r = 0; r < b_next; ++r) {
                    DataCopy(g_arec[r * 6144 + col], ub_bf16, 512);
                }
                Fence<HardEvent::MTE3_MTE2>();
            }

            if (aiv == 1) {
                DataCopy(ub_bf16[0], g_sin[base_pos * 64], 64);
                DataCopy(ub_bf16[64], g_cos[base_pos * 64], 64);
                Fence<HardEvent::MTE2_MTE3>();
                for (int32_t r = 0; r < b_next; ++r) {
                    DataCopy(g_esin[r * 64], ub_bf16[0], 64);
                    DataCopy(g_ecos[r * 64], ub_bf16[64], 64);
                }
                Fence<HardEvent::MTE3_MTE2>();
            }

            GlobalTensor<int32_t> g_pre, g_idx;
            g_pre.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(prefix_topk_gm), 2048);
            g_idx.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t*>(indices_buf_gm), b_next * k_sfa);
            auto ub_idx = ub_bf16.ReinterpretCast<int32_t>();
            const int32_t chunks_per_row = k_sfa / 64;
            const int32_t total_chunks = b_next * chunks_per_row;
            for (int32_t task = aiv; task < total_chunks; task += 48) {
                const int32_t r = task / chunks_per_row;
                const int32_t ch = task % chunks_per_row;
                const int64_t pre_off = static_cast<int64_t>(ch) * 64;
                DataCopy(ub_idx, g_pre[pre_off], 64);
                if (ch == chunks_per_row - 1) {
                    Fence<HardEvent::MTE2_S>();
                    ub_idx.SetValue(63, 9250 + next_slot_cur + r);
                    Fence<HardEvent::S_MTE3>();
                } else {
                    Fence<HardEvent::MTE2_MTE3>();
                }
                DataCopy(g_idx[r * k_sfa + pre_off], ub_idx, 64);
                Fence<HardEvent::MTE3_MTE2>();
            }

            if (aiv == 0) {
                GlobalTensor<int64_t> g_si, g_atoks, g_apids, g_ptoks, g_ppars, g_eslot;
                GlobalTensor<float> g_slp, g_ascs, g_arlp, g_amgs, g_pscs, g_prlp;
                GlobalTensor<int32_t> g_anc, g_ekv;
                g_si.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(seed_top_i_gm), 8);
                g_slp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(seed_top_lp_gm), 8);
                g_atoks.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(active_toks_gm), 8);
                g_ascs.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(active_scores_gm), 8);
                g_arlp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(active_raw_lp_gm), 8);
                g_amgs.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(active_margins_gm), 8);
                g_apids.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(active_pids_gm), 8);
                g_pscs.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pool_scores_gm));
                g_prlp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pool_raw_lp_gm));
                g_ptoks.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(pool_tokens_gm));
                g_ppars.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(pool_parents_gm));
                g_anc.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(anc_slots_gm), 128);
                g_eslot.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(eshr_slot_gm), b_next);
                g_ekv.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(eshr_kvlen_gm), 1);

                DataCopy(ub_i64[16], g_si, 8);
                DataCopy(ub_f32[0], g_slp, 8);
                Fence<HardEvent::MTE2_S>();

                const float mg0 = ub_f32.GetValue(7);
                for (int32_t z = 0; z < 128; ++z) {
                    ub_anc.SetValue(z, 0);
                }

                ub_f32.SetValue(64, 1.0e6F);
                ub_f32.SetValue(80, 0.0F);
                ub_i64.SetValue(64, pending_tok);
                ub_i64.SetValue(80, -1);

                for (int32_t j = 0; j < 8; ++j) {
                    if (j < b_next) {
                        const int64_t tk = ub_i64.GetValue(16 + j);
                        const float rlp = ub_f32.GetValue(j);
                        const float sc = rlp + beta_margin * mg0 - beta_depth * 1.0F;
                        const int32_t sl = 9250 + next_slot_cur + j;
                        ub_anc.SetValue(j * 16 + 0, sl);
                        ub_i64.SetValue(32 + j, tk);
                        ub_f32.SetValue(16 + j, sc);
                        ub_f32.SetValue(24 + j, rlp);
                        ub_f32.SetValue(32 + j, mg0);
                        ub_i64.SetValue(40 + j, static_cast<int64_t>(1 + j));
                        ub_i64.SetValue(48 + j, static_cast<int64_t>(sl));
                        ub_f32.SetValue(64 + 1 + j, sc);
                        ub_f32.SetValue(80 + 1 + j, rlp);
                        ub_i64.SetValue(64 + 1 + j, tk);
                        ub_i64.SetValue(80 + 1 + j, 0);
                    } else {
                        ub_i64.SetValue(32 + j, 0);
                        ub_f32.SetValue(16 + j, 0.0F);
                        ub_f32.SetValue(24 + j, 0.0F);
                        ub_f32.SetValue(32 + j, 0.0F);
                        ub_i64.SetValue(40 + j, 0);
                        ub_i64.SetValue(48 + j, 9250);
                    }
                }
                auto ub_kv = ub_mr.ReinterpretCast<int32_t>();
                ub_kv.SetValue(0, 9295);
                Fence<HardEvent::S_MTE3>();

                const uint32_t n_init = static_cast<uint32_t>(1 + b_next);
                DataCopy(g_anc, ub_anc, 128);
                DataCopy(g_atoks, ub_i64[32], 8);
                DataCopy(g_ascs, ub_f32[16], 8);
                DataCopy(g_arlp, ub_f32[24], 8);
                DataCopy(g_amgs, ub_f32[32], 8);
                DataCopy(g_apids, ub_i64[40], 8);
                DataCopyPad(g_eslot, ub_i64[48], DataCopyExtParams{
                    1, static_cast<uint32_t>(b_next * sizeof(int64_t)), 0, 0, 0});
                DataCopyPad(g_ekv, ub_kv, DataCopyExtParams{1, sizeof(int32_t), 0, 0, 0});
                DataCopyPad(g_pscs[0], ub_f32[64], DataCopyExtParams{
                    1, static_cast<uint32_t>(n_init * sizeof(float)), 0, 0, 0});
                DataCopyPad(g_prlp[0], ub_f32[80], DataCopyExtParams{
                    1, static_cast<uint32_t>(n_init * sizeof(float)), 0, 0, 0});
                DataCopyPad(g_ptoks[0], ub_i64[64], DataCopyExtParams{
                    1, static_cast<uint32_t>(n_init * sizeof(int64_t)), 0, 0, 0});
                DataCopyPad(g_ppars[0], ub_i64[80], DataCopyExtParams{
                    1, static_cast<uint32_t>(n_init * sizeof(int64_t)), 0, 0, 0});
            }
            Publish();
            return;
        }

        const int32_t d = step;
        const int32_t n_cand = b_cur * k_expand;
        GlobalTensor<int64_t> g_work_i64;
        g_work_i64.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(work_gm), 8);
        GlobalTensor<int32_t> g_anc;
        g_anc.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(anc_slots_gm), 128);

        if (aiv == 0) {
            GlobalTensor<int64_t> g_ctrl, g_atoks, g_apids, g_ptoks, g_ppars, g_eslot;
            GlobalTensor<float> g_gat, g_ascs, g_arlp, g_amgs, g_pscs, g_prlp;
            GlobalTensor<int32_t> g_ekv;

            g_ctrl.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(draft_ctrl_gm), 8);
            g_gat.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gather_out_gm), 512);
            g_atoks.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(active_toks_gm), 8);
            g_ascs.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(active_scores_gm), 8);
            g_arlp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(active_raw_lp_gm), 8);
            g_amgs.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(active_margins_gm), 8);
            g_apids.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(active_pids_gm), 8);
            g_pscs.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pool_scores_gm));
            g_prlp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pool_raw_lp_gm));
            g_ptoks.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(pool_tokens_gm));
            g_ppars.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(pool_parents_gm));

            DataCopy(ub_i64[0], g_apids, 8);
            DataCopy(ub_i64[8], g_ctrl, 8);
            DataCopy(ub_f32[0], g_arlp, 8);
            DataCopy(ub_f32[8], g_amgs, 8);
            DataCopy(ub_gat, g_gat, 512);
            DataCopy(ub_anc[0], g_anc, 128);
            Fence<HardEvent::MTE2_S>();

            const int64_t base_pos = ub_i64.GetValue(9);
            auto ub_dec_i = ub_i64[64];
            auto ub_dec_lp = ub_f32[64];
            const int32_t k_dec = (k_topk >= k_expand) ? k_topk : k_expand;
            DecodeGatheredTopKV3(
                ub_gat, ub_mr, ub_sr, ub_sum, ub_wk, b_cur, k_dec, ub_dec_i, ub_dec_lp);

            auto ub_s_sc = ub_gat[0];
            auto ub_s_rlp = ub_gat[32];
            auto ub_s_mg = ub_gat[64];
            auto ub_s_tk = ub_i64[16];
            auto ub_s_pr = ub_i64[32];
            const float dep_f = static_cast<float>(d + 1);

            for (int32_t bi = 0; bi < b_cur; ++bi) {
                const float arlp = ub_f32.GetValue(bi);
                const float amg = ub_f32.GetValue(8 + bi);
                const int64_t apid = ub_i64.GetValue(bi);
                float par_mg = ub_dec_lp.GetValue(bi * 4 + 0) - ub_dec_lp.GetValue(bi * 4 + 1);
                if (par_mg < 0.0F) {
                    par_mg = 0.0F;
                }
                const float cum_mg = amg + par_mg;
                for (int32_t j = 0; j < k_expand; ++j) {
                    const int32_t idx = bi * k_expand + j;
                    const float rlp = arlp + ub_dec_lp.GetValue(bi * 4 + j);
                    const float sc = rlp + beta_margin * cum_mg - beta_depth * dep_f;
                    ub_s_sc.SetValue(idx, sc);
                    ub_s_rlp.SetValue(idx, rlp);
                    ub_s_mg.SetValue(idx, cum_mg);
                    ub_s_tk.SetValue(idx, ub_dec_i.GetValue(bi * 4 + j));
                    ub_s_pr.SetValue(idx, apid);
                }
            }
            Fence<HardEvent::S_MTE3>();
            DataCopyPad(g_pscs[pool_off], ub_s_sc, DataCopyExtParams{
                1, static_cast<uint32_t>(n_cand * sizeof(float)), 0, 0, 0});
            DataCopyPad(g_prlp[pool_off], ub_s_rlp, DataCopyExtParams{
                1, static_cast<uint32_t>(n_cand * sizeof(float)), 0, 0, 0});
            DataCopyPad(g_ptoks[pool_off], ub_s_tk, DataCopyExtParams{
                1, static_cast<uint32_t>(n_cand * sizeof(int64_t)), 0, 0, 0});
            DataCopyPad(g_ppars[pool_off], ub_s_pr, DataCopyExtParams{
                1, static_cast<uint32_t>(n_cand * sizeof(int64_t)), 0, 0, 0});
            Fence<HardEvent::MTE3_S>();

            if (b_next > 0) {
                g_eslot.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(eshr_slot_gm), b_next);
                g_ekv.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(eshr_kvlen_gm), 1);

                float ba0 = -1.0e30F, ba1 = -1.0e30F, ba2 = -1.0e30F, ba3 = -1.0e30F;
                int64_t bi0 = 0, bi1 = 0, bi2 = 0, bi3 = 0;
                for (int32_t i = 0; i < n_cand; ++i) {
                    const float adj = ub_s_sc.GetValue(i) - 1.0e-5F * static_cast<float>(i);
                    InsertTop4(
                        adj, static_cast<int64_t>(i), ba0, ba1, ba2, ba3, bi0, bi1, bi2, bi3);
                }
                auto ub_bi = ub_i64[48];
                ub_bi.SetValue(0, bi0);
                ub_bi.SetValue(1, bi1);
                ub_bi.SetValue(2, bi2);
                ub_bi.SetValue(3, bi3);

                auto ub_ntk = ub_i64[80];
                auto ub_npd = ub_i64[88];
                auto ub_w_out = ub_i64[96];
                auto ub_nslot = ub_i64[104];
                auto ub_nsc = ub_f32[16];
                auto ub_nrlp = ub_f32[24];
                auto ub_nmg = ub_f32[32];
                auto ub_anew = ub_anc[128];
                auto ub_kv = ub_mr.ReinterpretCast<int32_t>();
                const int64_t next_pos = base_pos + d;

                for (int32_t z = 0; z < 128; ++z) {
                    ub_anew.SetValue(z, 0);
                }
                for (int32_t j = 0; j < 8; ++j) {
                    if (j < b_next) {
                        const int32_t top_i = static_cast<int32_t>(ub_bi.GetValue(j));
                        const int32_t src_bi = top_i / k_expand;
                        const int32_t sl = 9250 + next_slot_cur + j;
                        for (int32_t t = 0; t < d; ++t) {
                            ub_anew.SetValue(j * 16 + t, ub_anc.GetValue(src_bi * 16 + t));
                        }
                        ub_anew.SetValue(j * 16 + d, sl);
                        ub_ntk.SetValue(j, ub_s_tk.GetValue(top_i));
                        ub_nsc.SetValue(j, ub_s_sc.GetValue(top_i));
                        ub_nrlp.SetValue(j, ub_s_rlp.GetValue(top_i));
                        ub_nmg.SetValue(j, ub_s_mg.GetValue(top_i));
                        ub_npd.SetValue(j, static_cast<int64_t>(pool_off + top_i));
                        ub_w_out.SetValue(j, static_cast<int64_t>(src_bi));
                        ub_nslot.SetValue(j, static_cast<int64_t>(sl));
                    } else {
                        ub_ntk.SetValue(j, 0);
                        ub_nsc.SetValue(j, 0.0F);
                        ub_nrlp.SetValue(j, 0.0F);
                        ub_nmg.SetValue(j, 0.0F);
                        ub_npd.SetValue(j, 0);
                        ub_w_out.SetValue(j, 0);
                        ub_nslot.SetValue(j, 9250);
                    }
                }
                ub_w_out.SetValue(4, next_pos);
                ub_kv.SetValue(0, 9295);
                Fence<HardEvent::S_MTE3>();
                DataCopy(g_anc, ub_anew, 128);
                DataCopy(g_atoks, ub_ntk, 8);
                DataCopy(g_ascs, ub_nsc, 8);
                DataCopy(g_arlp, ub_nrlp, 8);
                DataCopy(g_amgs, ub_nmg, 8);
                DataCopy(g_apids, ub_npd, 8);
                DataCopy(g_work_i64, ub_w_out, 8);
                DataCopyPad(g_eslot, ub_nslot, DataCopyExtParams{
                    1, static_cast<uint32_t>(b_next * sizeof(int64_t)), 0, 0, 0});
                DataCopyPad(g_ekv, ub_kv, DataCopyExtParams{1, sizeof(int32_t), 0, 0, 0});
            }
        }
        Publish();

        if (b_next > 0) {
            SyncAll<true>();
            DataCopy(ub_i64[0], g_work_i64, 8);
            DataCopy(ub_anc[0], g_anc, 128);
            Fence<HardEvent::MTE2_S>();
            const int64_t next_pos = ub_i64.GetValue(4);

            GlobalTensor<bfloat16_t> g_recd, g_arec, g_sin, g_cos, g_esin, g_ecos;
            g_recd.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t*>(recs_d_gm), b_cur * 6144);
            g_arec.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t*>(active_recs_gm), b_next * 6144);
            g_sin.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(sin_cache_gm));
            g_cos.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cos_cache_gm));
            g_esin.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(eshr_sin_gm), b_next * 64);
            g_ecos.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(eshr_cos_gm), b_next * 64);

            if (aiv < b_next * 12) {
                const int32_t dst_r = aiv / 12;
                const int64_t src_r = ub_i64.GetValue(dst_r);
                const int64_t col = static_cast<int64_t>(aiv % 12) * 512;
                DataCopy(ub_bf16, g_recd[src_r * 6144 + col], 512);
                Fence<HardEvent::MTE2_MTE3>();
                DataCopy(g_arec[dst_r * 6144 + col], ub_bf16, 512);
                Fence<HardEvent::MTE3_MTE2>();
            }

            if (aiv == 47) {
                DataCopy(ub_bf16[0], g_sin[next_pos * 64], 64);
                DataCopy(ub_bf16[64], g_cos[next_pos * 64], 64);
                Fence<HardEvent::MTE2_MTE3>();
                for (int32_t r = 0; r < b_next; ++r) {
                    DataCopy(g_esin[r * 64], ub_bf16[0], 64);
                    DataCopy(g_ecos[r * 64], ub_bf16[64], 64);
                }
                Fence<HardEvent::MTE3_MTE2>();
            }

            GlobalTensor<int32_t> g_pre, g_idx;
            g_pre.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(prefix_topk_gm), 2048);
            g_idx.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t*>(indices_buf_gm), b_next * k_sfa);
            auto ub_idx = ub_bf16.ReinterpretCast<int32_t>();
            const int32_t chunks_per_row = k_sfa / 64;
            const int32_t total_chunks = b_next * chunks_per_row;
            const int32_t n_causal = d + 1;
            for (int32_t task = aiv; task < total_chunks; task += 48) {
                const int32_t r = task / chunks_per_row;
                const int32_t ch = task % chunks_per_row;
                const int64_t pre_off = static_cast<int64_t>(ch) * 64;
                DataCopy(ub_idx, g_pre[pre_off], 64);
                if (ch == chunks_per_row - 1) {
                    Fence<HardEvent::MTE2_S>();
                    const int32_t start_u = 64 - n_causal;
                    for (int32_t u = 0; u < n_causal; ++u) {
                        ub_idx.SetValue(start_u + u, ub_anc.GetValue(r * 16 + u));
                    }
                    Fence<HardEvent::S_MTE3>();
                } else {
                    Fence<HardEvent::MTE2_MTE3>();
                }
                DataCopy(g_idx[r * k_sfa + pre_off], ub_idx, 64);
                Fence<HardEvent::MTE3_MTE2>();
            }
            Publish();
        }
    }
}

__aicore__ inline void RunBuildTreeV3(
    TPipe& pipe,
    int32_t width,
    int32_t min_w_act,
    int32_t pool_size,
    int32_t num_seq_ptrs,
    int32_t num_q_ptrs,
    int32_t num_rope_dsts,
    float t_base_ms,
    float c_w_ms,
    uint64_t draft_ctrl_gm,
    uint64_t pool_scores_gm,
    uint64_t pool_raw_lp_gm,
    uint64_t pool_tokens_gm,
    uint64_t pool_parents_gm,
    uint64_t pool_depths_gm,
    uint64_t tree_tokens_gm,
    uint64_t tree_parents_gm,
    uint64_t tree_depths_gm,
    uint64_t tree_mask_gm,
    uint64_t t_ids_gm,
    uint64_t t_pos_gm,
    uint64_t t_slots_gm,
    uint64_t t_slots_i64_gm,
    uint64_t dyn_w_gm,
    uint64_t seq_ptrs_gm,
    uint64_t q_ptrs_gm,
    uint64_t cos_cache_gm,
    uint64_t sin_cache_gm,
    uint64_t rope_dsts_gm,
    uint64_t work_gm)
{
    if ASCEND_IS_AIV {
        const int32_t aiv = GetBlockIdx();
        TBuf<TPosition::VECCALC> b_pos, b_rdsts, b_cos, b_sin;
        pipe.InitBuffer(b_pos, 32 * 8);
        pipe.InitBuffer(b_rdsts, 1024 * 8);
        pipe.InitBuffer(b_cos, 32 * 64 * 2);
        pipe.InitBuffer(b_sin, 32 * 64 * 2);
        auto ub_pos = b_pos.Get<int64_t>();
        auto ub_rdsts = b_rdsts.Get<uint64_t>();
        auto ub_cos = b_cos.Get<bfloat16_t>();
        auto ub_sin = b_sin.Get<bfloat16_t>();

        GlobalTensor<int64_t> g_work_pos;
        g_work_pos.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(work_gm), 32);

        if (aiv == 0) {
            TBuf<TPosition::VECCALC> b_ctrl, b_psc, b_prlp, b_ptk, b_ppr, b_pdp, b_sptrs;
            TBuf<TPosition::VECCALC> b_out_i64, b_out_i32, b_sel_f, b_sel_i;
            pipe.InitBuffer(b_ctrl, 64);
            pipe.InitBuffer(b_psc, 256 * 4);
            pipe.InitBuffer(b_prlp, 256 * 4);
            pipe.InitBuffer(b_ptk, 256 * 8);
            pipe.InitBuffer(b_ppr, 256 * 8);
            pipe.InitBuffer(b_pdp, 256 * 8);
            pipe.InitBuffer(b_sptrs, 32 * 8);
            pipe.InitBuffer(b_out_i64, 256 * 8);
            pipe.InitBuffer(b_out_i32, 256 * 4);
            pipe.InitBuffer(b_sel_f, 128 * 4);
            pipe.InitBuffer(b_sel_i, 512 * 4);

            auto ub_ctrl = b_ctrl.Get<int64_t>();
            auto ub_psc = b_psc.Get<float>();
            auto ub_prlp = b_prlp.Get<float>();
            auto ub_ptk = b_ptk.Get<int64_t>();
            auto ub_ppr = b_ppr.Get<int64_t>();
            auto ub_pdp = b_pdp.Get<int64_t>();
            auto ub_sptrs = b_sptrs.Get<uint64_t>();
            auto ub_i64 = b_out_i64.Get<int64_t>();
            auto ub_i32 = b_out_i32.Get<int32_t>();
            auto ub_sel_f = b_sel_f.Get<float>();
            auto ub_sel_i = b_sel_i.Get<int32_t>();

            auto ub_best_sc = ub_sel_f[0];
            auto ub_rlp_sel = ub_sel_f[32];
            auto ub_p_sel = ub_sel_f[64];
            auto ub_best_pid = ub_sel_i[0];
            auto ub_pid2node = ub_sel_i[64];
            auto ub_par = ub_sel_i[320];

            GlobalTensor<int64_t> g_ctrl, g_ptk, g_ppr, g_pdp;
            GlobalTensor<float> g_psc, g_prlp;
            GlobalTensor<uint64_t> g_sptrs;
            g_ctrl.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(draft_ctrl_gm), 8);
            g_psc.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pool_scores_gm), 256);
            g_prlp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pool_raw_lp_gm), 256);
            g_ptk.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(pool_tokens_gm), 256);
            g_ppr.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(pool_parents_gm), 256);
            g_pdp.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(pool_depths_gm), 256);
            g_sptrs.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(seq_ptrs_gm), 16);

            DataCopy(ub_ctrl, g_ctrl, 8);
            DataCopy(ub_psc, g_psc, 256);
            DataCopy(ub_prlp, g_prlp, 256);
            DataCopy(ub_ptk, g_ptk, 256);
            DataCopy(ub_ppr, g_ppr, 256);
            DataCopy(ub_pdp, g_pdp, 256);
            DataCopy(ub_sptrs[0], g_sptrs, 16);
            Fence<HardEvent::MTE2_S>();

            const int64_t pending_tok = ub_ctrl.GetValue(0);
            const int64_t base_pos = ub_ctrl.GetValue(1);

            for (int32_t i = pool_size - 1; i >= 1; --i) {
                const int32_t p = static_cast<int32_t>(ub_ppr.GetValue(i));
                const float child_need = ub_psc.GetValue(i) + 2.0e-4F;
                if (p > 0 && child_need > ub_psc.GetValue(p)) {
                    ub_psc.SetValue(p, child_need);
                }
            }

            for (int32_t j = 0; j < width; ++j) {
                ub_best_sc.SetValue(j, -1.0e30F);
                ub_best_pid.SetValue(j, 0);
            }
            for (int32_t i = 0; i < pool_size; ++i) {
                const float adj = (i == 0)
                    ? 1.0e6F : (ub_psc.GetValue(i) - 1.0e-5F * static_cast<float>(i));
                if (adj > ub_best_sc.GetValue(width - 1)) {
                    int32_t p = width - 1;
                    while (p > 0 && adj > ub_best_sc.GetValue(p - 1)) {
                        ub_best_sc.SetValue(p, ub_best_sc.GetValue(p - 1));
                        ub_best_pid.SetValue(p, ub_best_pid.GetValue(p - 1));
                        --p;
                    }
                    ub_best_sc.SetValue(p, adj);
                    ub_best_pid.SetValue(p, i);
                }
            }

            int32_t w_act = width;
            if (min_w_act > 0 && min_w_act < width) {
                for (int32_t r = 0; r < 32; ++r) {
                    if (r == 0) {
                        ub_rlp_sel.SetValue(r, 0.0F);
                    } else if (r < width) {
                        float rlp = ub_prlp.GetValue(ub_best_pid.GetValue(r));
                        if (rlp < -20.0F) {
                            rlp = -20.0F;
                        }
                        if (rlp > 0.0F) {
                            rlp = 0.0F;
                        }
                        ub_rlp_sel.SetValue(r, rlp);
                    } else {
                        ub_rlp_sel.SetValue(r, -20.0F);
                    }
                }
                Fence<HardEvent::S_V>();
                Exp(ub_p_sel, ub_rlp_sel, 32);
                Fence<HardEvent::V_S>();

                float e_tau = 0.0F;
                float best_tpot = 1.0e30F;
                int32_t best_w = width;
                for (int32_t r = 0; r < width; ++r) {
                    e_tau += ub_p_sel.GetValue(r);
                    const int32_t cand_w = r + 1;
                    if (cand_w >= min_w_act && ((cand_w & 1) == 0 || cand_w == width)) {
                        const float est_tpot =
                            (t_base_ms + c_w_ms * static_cast<float>(cand_w)) / e_tau;
                        if (est_tpot <= best_tpot) {
                            best_tpot = est_tpot;
                            best_w = cand_w;
                        }
                    }
                }
                w_act = best_w;
            }

            for (int32_t i = 1; i < w_act; ++i) {
                const int32_t key = ub_best_pid.GetValue(i);
                int32_t j = i - 1;
                while (j >= 0 && ub_best_pid.GetValue(j) > key) {
                    ub_best_pid.SetValue(j + 1, ub_best_pid.GetValue(j));
                    --j;
                }
                ub_best_pid.SetValue(j + 1, key);
            }

            for (int32_t i = 0; i < pool_size; ++i) {
                ub_pid2node.SetValue(i, 0);
            }
            for (int32_t j = 0; j < w_act; ++j) {
                ub_pid2node.SetValue(ub_best_pid.GetValue(j), j);
            }

            for (int32_t j = 0; j < 32; ++j) {
                ub_i64.SetValue(96 + j, base_pos);
            }
            for (int32_t j = 0; j < width; ++j) {
                if (j < w_act) {
                    const int32_t pid = ub_best_pid.GetValue(j);
                    const int64_t tok = ub_ptk.GetValue(pid);
                    const int64_t dep = ub_pdp.GetValue(pid);
                    const int32_t par = (j == 0)
                        ? -1 : ub_pid2node.GetValue(static_cast<int32_t>(ub_ppr.GetValue(pid)));
                    ub_par.SetValue(j, par);
                    ub_i64.SetValue(j, tok);
                    ub_i64.SetValue(32 + j, static_cast<int64_t>(par));
                    ub_i64.SetValue(64 + j, dep);
                    ub_i64.SetValue(96 + j, base_pos + dep);
                    ub_i64.SetValue(128 + j, base_pos + j);
                    ub_i32.SetValue(32 + j, static_cast<int32_t>(base_pos + j));
                    ub_i32.SetValue(96 + j, static_cast<int32_t>(tok));
                } else {
                    ub_par.SetValue(j, -1);
                    ub_i64.SetValue(j, pending_tok);
                    ub_i64.SetValue(32 + j, -1);
                    ub_i64.SetValue(64 + j, 0);
                    ub_i64.SetValue(96 + j, base_pos);
                    ub_i64.SetValue(128 + j, base_pos + j);
                    ub_i32.SetValue(32 + j, static_cast<int32_t>(base_pos + j));
                    ub_i32.SetValue(96 + j, static_cast<int32_t>(pending_tok));
                }
            }

            for (int32_t j = 0; j < 32; ++j) {
                if (j < w_act) {
                    uint32_t m = 0U;
                    int32_t c = j;
                    while (c >= 0) {
                        m |= (1U << static_cast<uint32_t>(c));
                        c = ub_par.GetValue(c);
                    }
                    ub_i32.SetValue(j, static_cast<int32_t>(m));
                } else if (j < width) {
                    ub_i32.SetValue(j, static_cast<int32_t>(1U << static_cast<uint32_t>(j)));
                } else {
                    ub_i32.SetValue(j, 0);
                }
            }

            const int32_t pos_act = static_cast<int32_t>(base_pos + w_act);
            const int32_t pos_full = static_cast<int32_t>(base_pos + width);
            for (int32_t j = 0; j < 16; ++j) {
                ub_i32.SetValue(64 + j, 0);
            }
            ub_i32.SetValue(64 + 0, w_act);
            ub_i32.SetValue(64 + 1, width);
            ub_i32.SetValue(64 + 8, pos_act);
            ub_i32.SetValue(64 + 9, pos_full);
            ub_i32.SetValue(80, pos_full);
            auto ub_mc2 = ub_i32[88].ReinterpretCast<uint8_t>();
            for (int32_t j = 0; j < 32; ++j) {
                ub_mc2.SetValue(j, (j < w_act) ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0));
            }

            Fence<HardEvent::S_MTE3>();
            GlobalTensor<int64_t> g_ttok, g_tpar, g_tdep, g_tpos, g_tslots64;
            GlobalTensor<int32_t> g_tids, g_msk, g_tslots, g_dyn;
            g_ttok.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(tree_tokens_gm), width);
            g_tpar.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(tree_parents_gm), width);
            g_tdep.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(tree_depths_gm), width);
            g_tids.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(t_ids_gm), width);
            g_tpos.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(t_pos_gm), width);
            g_msk.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tree_mask_gm), 32);
            g_tslots.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(t_slots_gm), width);
            g_tslots64.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(t_slots_i64_gm), width);
            g_dyn.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dyn_w_gm), 16);

            const uint32_t wb_i64 = static_cast<uint32_t>(width * sizeof(int64_t));
            const uint32_t wb_i32 = static_cast<uint32_t>(width * sizeof(int32_t));
            DataCopyPad(g_ttok, ub_i64[0], DataCopyExtParams{1, wb_i64, 0, 0, 0});
            DataCopyPad(g_tpar, ub_i64[32], DataCopyExtParams{1, wb_i64, 0, 0, 0});
            DataCopyPad(g_tdep, ub_i64[64], DataCopyExtParams{1, wb_i64, 0, 0, 0});
            DataCopyPad(g_tids, ub_i32[96], DataCopyExtParams{1, wb_i32, 0, 0, 0});
            DataCopyPad(g_tpos, ub_i64[96], DataCopyExtParams{1, wb_i64, 0, 0, 0});
            DataCopy(g_work_pos, ub_i64[96], 32);
            DataCopy(g_msk, ub_i32[0], 32);
            DataCopyPad(g_tslots, ub_i32[32], DataCopyExtParams{1, wb_i32, 0, 0, 0});
            DataCopyPad(g_tslots64, ub_i64[128], DataCopyExtParams{1, wb_i64, 0, 0, 0});
            DataCopy(g_dyn, ub_i32[64], 16);

            for (int32_t s = 0; s < num_seq_ptrs; ++s) {
                GlobalTensor<int32_t> g_sl;
                g_sl.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ub_sptrs.GetValue(s)), 1);
                DataCopyPad(g_sl, ub_i32[80], DataCopyExtParams{1, sizeof(int32_t), 0, 0, 0});
            }
            if (num_q_ptrs > 0 && q_ptrs_gm != 0) {
                GlobalTensor<uint8_t> g_mc2;
                g_mc2.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(q_ptrs_gm), width);
                DataCopyPad(
                    g_mc2, ub_mc2, DataCopyExtParams{1, static_cast<uint32_t>(width), 0, 0, 0});
            }
            Fence<HardEvent::MTE3_S>();
            DataCacheCleanAndInvalid<
                int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(g_dyn[0]);
            DataCacheCleanAndInvalid<
                int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(g_dyn[8]);
            for (int32_t s = 0; s < num_seq_ptrs; ++s) {
                GlobalTensor<int32_t> g_sl;
                g_sl.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ub_sptrs.GetValue(s)), 1);
                DataCacheCleanAndInvalid<
                    int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(g_sl);
            }
            if (num_q_ptrs > 0 && q_ptrs_gm != 0) {
                GlobalTensor<uint8_t> g_mc2;
                g_mc2.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(q_ptrs_gm), width);
                DataCacheCleanAndInvalid<
                    uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(g_mc2);
            }
        }
        Publish();

        if (num_rope_dsts > 0) {
            SyncAll<true>();
            const int32_t rdst_elems = (2 * num_rope_dsts + 3) & ~3;
            GlobalTensor<uint64_t> g_rdsts;
            g_rdsts.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(rope_dsts_gm), rdst_elems);
            DataCopy(ub_pos, g_work_pos, 32);
            DataCopy(ub_rdsts, g_rdsts, rdst_elems);
            Fence<HardEvent::MTE2_S>();

            GlobalTensor<bfloat16_t> g_cos_c, g_sin_c;
            g_cos_c.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cos_cache_gm));
            g_sin_c.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(sin_cache_gm));
            for (int32_t r = 0; r < width; ++r) {
                const int64_t p = ub_pos.GetValue(r);
                DataCopy(ub_cos[r * 64], g_cos_c[p * 64], 64);
                DataCopy(ub_sin[r * 64], g_sin_c[p * 64], 64);
            }
            Fence<HardEvent::MTE2_MTE3>();
            const int32_t copy_elems = width * 64;
            for (int32_t d = aiv; d < num_rope_dsts; d += 48) {
                GlobalTensor<bfloat16_t> g_dc, g_ds;
                g_dc.SetGlobalBuffer(
                    reinterpret_cast<__gm__ bfloat16_t*>(ub_rdsts.GetValue(2 * d)), copy_elems);
                g_ds.SetGlobalBuffer(
                    reinterpret_cast<__gm__ bfloat16_t*>(ub_rdsts.GetValue(2 * d + 1)),
                    copy_elems);
                DataCopy(g_dc, ub_cos, copy_elems);
                DataCopy(g_ds, ub_sin, copy_elems);
            }
            Publish();
        }
    }
}

__aicore__ inline float KthTop4(int32_t k, float v0, float v1, float v2, float v3)
{
    if (k >= 4) {
        return v3;
    }
    if (k == 3) {
        return v2;
    }
    if (k == 2) {
        return v1;
    }
    return v0;
}

__aicore__ inline void RunKvCompact(
    TPipe& pipe, uint64_t descs_gm, uint64_t ctrl_gm, int32_t num_tensors)
{
    if ASCEND_IS_AIV {
        TBuf<TPosition::VECCALC> b_descs, b_ctrl, b_rows, b_out16;
        pipe.InitBuffer(b_descs, 4096);
        pipe.InitBuffer(b_ctrl, 320);
        pipe.InitBuffer(b_rows, 32 * 1024);
        pipe.InitBuffer(b_out16, 128);
        auto ub_descs = b_descs.Get<uint64_t>();
        auto ub_ctrl = b_ctrl.Get<int64_t>();
        auto ub_rows = b_rows.Get<uint16_t>();
        auto ub_out16 = b_out16.Get<uint16_t>();

        const int32_t desc_elems = (2 * num_tensors + 3) & ~3;
        GlobalTensor<uint64_t> g_descs;
        GlobalTensor<int64_t> g_ctrl;
        g_descs.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(descs_gm), desc_elems);
        g_ctrl.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ctrl_gm), 40);
        DataCopy(ub_descs, g_descs, desc_elems);
        DataCopy(ub_ctrl, g_ctrl, 40);
        Fence<HardEvent::MTE2_S>();

        const int64_t dst_base = ub_ctrl.GetValue(0);
        const int64_t count = ub_ctrl.GetValue(1);
        if (count <= 0) {
            return;
        }

        const int32_t aiv = GetBlockIdx();
        for (int32_t t = aiv; t < num_tensors; t += 48) {
            const uint64_t base_ptr = ub_descs.GetValue(2 * t);
            const uint64_t row_bytes = ub_descs.GetValue(2 * t + 1);
            GlobalTensor<uint16_t> gm;
            gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(base_ptr));
            if (row_bytes >= 32U) {
                const int64_t elems = static_cast<int64_t>(row_bytes >> 1U);
                for (int64_t j = 0; j < count; ++j) {
                    const int64_t sr = ub_ctrl.GetValue(2 + j);
                    DataCopy(ub_rows[j * elems], gm[sr * elems], elems);
                }
                Fence<HardEvent::MTE2_MTE3>();
                DataCopy(gm[dst_base * elems], ub_rows, count * elems);
                Fence<HardEvent::MTE3_MTE2>();
            } else {
                const int64_t aligned_base = dst_base & ~15LL;
                DataCopy(ub_rows, gm[aligned_base], 64);
                Fence<HardEvent::MTE2_S>();
                for (int64_t i = 0; i < 64; ++i) {
                    ub_out16.SetValue(i, ub_rows.GetValue(i));
                }
                for (int64_t j = 0; j < count; ++j) {
                    const int64_t sr = ub_ctrl.GetValue(2 + j);
                    ub_out16.SetValue(
                        (dst_base - aligned_base) + j, ub_rows.GetValue(sr - aligned_base));
                }
                Fence<HardEvent::S_MTE3>();
                DataCopy(gm[aligned_base], ub_out16, 64);
                Fence<HardEvent::MTE3_MTE2>();
                Fence<HardEvent::MTE3_S>();
            }
        }
        Publish();
    }
}

__aicore__ inline void RunStep3cPrologPack(
    TPipe& pipe, uint64_t active_toks_gm, uint64_t emb_w_gm,
    uint64_t rot_part_gm, uint64_t rot_bias_gm, uint64_t comm_in_gm,
    int32_t b_rows, int32_t vocab_start, int32_t vocab_end, int32_t rank)
{
    if ASCEND_IS_AIV {
        TBuf<TPosition::VECCALC> b_toks, b_bf16, b_bias, b_f1, b_f2;
        pipe.InitBuffer(b_toks, 64);
        pipe.InitBuffer(b_bf16, 768 * 2);
        pipe.InitBuffer(b_bias, 768 * 2);
        pipe.InitBuffer(b_f1, 768 * 4);
        pipe.InitBuffer(b_f2, 768 * 4);
        auto ub_toks = b_toks.Get<int64_t>();
        auto ub_bf16 = b_bf16.Get<bfloat16_t>();
        auto ub_bias = b_bias.Get<bfloat16_t>();
        auto ub_f1 = b_f1.Get<float>();
        auto ub_f2 = b_f2.Get<float>();

        GlobalTensor<int64_t> g_toks;
        GlobalTensor<bfloat16_t> g_emb, g_rot, g_rbias, g_out;
        g_toks.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(active_toks_gm), 8);
        g_emb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(emb_w_gm));
        g_rot.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(rot_part_gm));
        g_rbias.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(rot_bias_gm), 768);
        g_out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(comm_in_gm), 2 * b_rows * 6144);

        DataCopy(ub_toks, g_toks, 8);
        Fence<HardEvent::MTE2_S>();

        const int32_t aiv = GetBlockIdx();
        if (aiv < b_rows * 12) {
            const int32_t r_idx = aiv / 12;
            const int64_t col = static_cast<int64_t>(aiv % 12) * 512;
            const int64_t tok = ub_toks.GetValue(r_idx);
            if (tok >= vocab_start && tok < vocab_end) {
                DataCopy(ub_bf16, g_emb[(tok - vocab_start) * 6144 + col], 512);
                Fence<HardEvent::MTE2_MTE3>();
            } else {
                Duplicate(ub_bf16.ReinterpretCast<int16_t>(), static_cast<int16_t>(0), 512);
                Fence<HardEvent::V_MTE3>();
            }
            DataCopy(g_out[r_idx * 6144 + col], ub_bf16, 512);
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::MTE3_MTE2>();
        }

        if (aiv < b_rows * 8) {
            const int32_t r_idx = aiv / 8;
            const int32_t s_rk = aiv % 8;
            const int64_t dst_off = static_cast<int64_t>(b_rows + r_idx) * 6144
                + static_cast<int64_t>(s_rk) * 768;
            if (s_rk == rank) {
                DataCopy(ub_bf16, g_rot[r_idx * 768], 768);
                DataCopy(ub_bias, g_rbias, 768);
                Fence<HardEvent::MTE2_V>();
                Cast(ub_f1, ub_bf16, RoundMode::CAST_NONE, 768);
                Cast(ub_f2, ub_bias, RoundMode::CAST_NONE, 768);
                PipeBarrier<PIPE_V>();
                Add(ub_f1, ub_f1, ub_f2, 768);
                PipeBarrier<PIPE_V>();
                Cast(ub_bf16, ub_f1, RoundMode::CAST_RINT, 768);
                Fence<HardEvent::V_MTE3>();
                DataCopy(g_out[dst_off], ub_bf16, 768);
            } else {
                Duplicate(ub_bf16.ReinterpretCast<int16_t>(), static_cast<int16_t>(0), 768);
                Fence<HardEvent::V_MTE3>();
                DataCopy(g_out[dst_off], ub_bf16, 768);
            }
        }
        Publish();
    }
}

__aicore__ inline void RunStep3cDualRmsNormSlice(
    TPipe& pipe, uint64_t comm_out_gm, uint64_t enorm_w_gm, uint64_t hnorm_w_gm,
    uint64_t eh_in_gm, int32_t b_rows, int32_t rank, float eps)
{
    if ASCEND_IS_AIV {
        const int32_t t = GetBlockIdx();
        if (t < 2 * b_rows) {
            TBuf<TPosition::VECCALC> b_x, b_g, b_xf, b_sq, b_work, b_gf, b_sum;
            pipe.InitBuffer(b_x, 6144 * 2);
            pipe.InitBuffer(b_g, 768 * 2);
            pipe.InitBuffer(b_xf, 6144 * 4);
            pipe.InitBuffer(b_sq, 6144 * 4);
            pipe.InitBuffer(b_work, 6144 * 4);
            pipe.InitBuffer(b_gf, 768 * 4);
            pipe.InitBuffer(b_sum, 64);
            auto ub_x = b_x.Get<bfloat16_t>();
            auto ub_g = b_g.Get<bfloat16_t>();
            auto ub_xf = b_xf.Get<float>();
            auto ub_sq = b_sq.Get<float>();
            auto ub_work = b_work.Get<float>();
            auto ub_gf = b_gf.Get<float>();
            auto ub_sum = b_sum.Get<float>();

            const bool is_rot = (t >= b_rows);
            const int32_t r_idx = is_rot ? (t - b_rows) : t;
            GlobalTensor<bfloat16_t> g_in, g_w, g_out;
            g_in.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t*>(comm_out_gm), 2 * b_rows * 6144);
            g_w.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t*>(is_rot ? hnorm_w_gm : enorm_w_gm), 6144);
            g_out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(eh_in_gm), b_rows * 1536);

            DataCopy(ub_x, g_in[t * 6144], 6144);
            DataCopy(ub_g, g_w[rank * 768], 768);
            Fence<HardEvent::MTE2_V>();

            Cast(ub_xf, ub_x, RoundMode::CAST_NONE, 6144);
            Cast(ub_gf, ub_g, RoundMode::CAST_NONE, 768);
            PipeBarrier<PIPE_V>();
            Mul(ub_sq, ub_xf, ub_xf, 6144);
            PipeBarrier<PIPE_V>();
            ReduceSum(ub_sum, ub_sq, ub_work, 6144);
            Fence<HardEvent::V_S>();
            const float mean_eps = ub_sum.GetValue(0) * (1.0F / 6144.0F) + eps;
            Fence<HardEvent::S_V>();
            Duplicate(ub_sum, mean_eps, 8);
            PipeBarrier<PIPE_V>();
            Rsqrt(ub_sum, ub_sum, 8);
            Fence<HardEvent::V_S>();
            const float rstd = ub_sum.GetValue(0);
            Fence<HardEvent::S_V>();

            Muls(ub_sq, ub_xf[rank * 768], rstd, 768);
            PipeBarrier<PIPE_V>();
            Mul(ub_sq, ub_sq, ub_gf, 768);
            PipeBarrier<PIPE_V>();
            Cast(ub_g, ub_sq, RoundMode::CAST_RINT, 768);
            Fence<HardEvent::V_MTE3>();
            DataCopy(g_out[r_idx * 1536 + (is_rot ? 768 : 0)], ub_g, 768);
        }
        Publish();
    }
}

template <bool IsBf16>
__aicore__ inline void RunLogitsTopkLseImpl(
    TPipe& pipe, uint64_t logits_gm, uint64_t work_gm, uint64_t gather_in_gm,
    int32_t m_rows, int32_t row_stride, int32_t valid_vocab, int32_t vocab_start,
    int32_t k, int32_t rank)
{
    if ASCEND_IS_AIV {
        const int32_t aiv = GetBlockIdx();
        const int32_t aivs_per_row = 48 / m_rows;
        const int32_t r_idx = aiv / aivs_per_row;
        const int32_t sub = aiv % aivs_per_row;
        const int32_t chunk = (((valid_vocab + aivs_per_row - 1) / aivs_per_row) + 63) & ~63;
        const int32_t col_begin = sub * chunk;
        const int32_t col_end =
            (col_begin + chunk < valid_vocab) ? (col_begin + chunk) : valid_vocab;
        const int32_t len = (col_end > col_begin) ? (col_end - col_begin) : 0;

        TBuf<TPosition::VECCALC> b_x, b_bf16, b_blk, b_tmp, b_work, b_sum, b_out;
        pipe.InitBuffer(b_x, 2048 * 4);
        pipe.InitBuffer(b_bf16, 2048 * 2);
        pipe.InitBuffer(b_blk, 512);
        pipe.InitBuffer(b_tmp, 2048 * 4);
        pipe.InitBuffer(b_work, 2048 * 4);
        pipe.InitBuffer(b_sum, 64);
        pipe.InitBuffer(b_out, 256);
        auto ub_x = b_x.Get<float>();
        auto ub_bf16 = b_bf16.Get<bfloat16_t>();
        auto ub_blk = b_blk.Get<float>();
        auto ub_tmp = b_tmp.Get<float>();
        auto ub_work = b_work.Get<float>();
        auto ub_sum = b_sum.Get<float>();
        auto ub_out = b_out.Get<float>();

        GlobalTensor<float> g_work, g_gather;
        g_work.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(work_gm), 48 * 16);
        g_gather.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gather_in_gm), 8 * 64);

        if (aiv < 8 && aiv != rank) {
            Duplicate(ub_out, 0.0F, 64);
            Fence<HardEvent::V_MTE3>();
            DataCopy(g_gather[aiv * 64], ub_out, 64);
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::MTE3_S>();
        }

        Duplicate(ub_x, -1.0e30F, chunk);
        PipeBarrier<PIPE_V>();
        float tv0 = -1.0e30F, tv1 = -1.0e30F, tv2 = -1.0e30F, tv3 = -1.0e30F;
        int64_t ti0 = 0, ti1 = 0, ti2 = 0, ti3 = 0;
        float l_sum = 0.0F;

        if (len > 0) {
            if constexpr (IsBf16) {
                GlobalTensor<bfloat16_t> g_logits;
                g_logits.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(logits_gm));
                DataCopyPad(ub_bf16, g_logits[static_cast<int64_t>(r_idx) * row_stride + col_begin],
                            DataCopyExtParams{
                                1, static_cast<uint32_t>(len * sizeof(bfloat16_t)), 0, 0, 0},
                            DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
                Fence<HardEvent::MTE2_V>();
                const int32_t len_align16 = (len + 15) & ~15;
                Cast(ub_x, ub_bf16, RoundMode::CAST_NONE, len_align16);
                Fence<HardEvent::V_S>();
                for (int32_t pad_j = len; pad_j < len_align16; ++pad_j) {
                    ub_x.SetValue(pad_j, -1.0e30F);
                }
            } else {
                GlobalTensor<float> g_logits;
                g_logits.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(logits_gm));
                Fence<HardEvent::V_MTE2>();
                DataCopyPad(ub_x, g_logits[static_cast<int64_t>(r_idx) * row_stride + col_begin],
                            DataCopyExtParams{
                                1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0},
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
                Fence<HardEvent::MTE2_S>();
                const int32_t len_align8 = (len + 7) & ~7;
                for (int32_t pad_j = len; pad_j < len_align8; ++pad_j) {
                    ub_x.SetValue(pad_j, -1.0e30F);
                }
            }
            Fence<HardEvent::S_V>();
            const int32_t n_rep = chunk / 64;
            WholeReduceMax<float>(ub_blk, ub_x, 64, n_rep, 1, 1, 8);
            Fence<HardEvent::V_S>();
            for (int32_t b = 0; b < n_rep; ++b) {
                if (ub_blk.GetValue(2 * b) > KthTop4(k, tv0, tv1, tv2, tv3)) {
                    const int32_t j_end = ((b + 1) * 64 < len) ? ((b + 1) * 64) : len;
                    for (int32_t j = b * 64; j < j_end; ++j) {
                        InsertTop4(
                            ub_x.GetValue(j),
                            static_cast<int64_t>(vocab_start + col_begin + j),
                                   tv0, tv1, tv2, tv3, ti0, ti1, ti2, ti3);
                    }
                }
            }
            Fence<HardEvent::S_V>();
            Adds(ub_tmp, ub_x, -tv0, chunk);
            PipeBarrier<PIPE_V>();
            Maxs(ub_tmp, ub_tmp, -104.0F, chunk);
            PipeBarrier<PIPE_V>();
            Exp(ub_tmp, ub_tmp, chunk);
            PipeBarrier<PIPE_V>();
            ReduceSum(ub_sum, ub_tmp, ub_work, chunk);
            Fence<HardEvent::V_S>();
            l_sum = ub_sum.GetValue(0);
        }

        ub_out.SetValue(0, tv0);
        ub_out.SetValue(1, l_sum);
        ub_out.SetValue(2, tv0);
        ub_out.SetValue(3, tv1);
        ub_out.SetValue(4, tv2);
        ub_out.SetValue(5, tv3);
        ub_out.SetValue(6, static_cast<float>(ti0));
        ub_out.SetValue(7, static_cast<float>(ti1));
        ub_out.SetValue(8, static_cast<float>(ti2));
        ub_out.SetValue(9, static_cast<float>(ti3));
        Fence<HardEvent::S_MTE3>();
        DataCopy(g_work[aiv * 16], ub_out, 16);
        Publish();
        SyncAll<true>();

        if (aiv == 0) {
            DataCopy(ub_work, g_work, 48 * 16);
            Fence<HardEvent::MTE2_S>();
            Fence<HardEvent::S_V>();
            Duplicate(ub_out, 0.0F, 64);
            Fence<HardEvent::V_S>();

            const int32_t cols = 2 + 2 * k;
            for (int32_t m = 0; m < m_rows; ++m) {
                float rv0 = -1.0e30F, rv1 = -1.0e30F, rv2 = -1.0e30F, rv3 = -1.0e30F;
                int64_t ri0 = 0, ri1 = 0, ri2 = 0, ri3 = 0;
                for (int32_t s = 0; s < aivs_per_row; ++s) {
                    const int32_t base = (m * aivs_per_row + s) * 16;
                    for (int32_t j = 0; j < k; ++j) {
                        InsertTop4(ub_work.GetValue(base + 2 + j),
                                   static_cast<int64_t>(ub_work.GetValue(base + 6 + j)),
                                   rv0, rv1, rv2, rv3, ri0, ri1, ri2, ri3);
                    }
                }
                const float r_max = rv0;
                Fence<HardEvent::S_V>();
                Duplicate(ub_blk, -1.0e30F, 64);
                Duplicate(ub_tmp, 0.0F, 64);
                Fence<HardEvent::V_S>();
                for (int32_t s = 0; s < aivs_per_row; ++s) {
                    const int32_t base = (m * aivs_per_row + s) * 16;
                    ub_blk.SetValue(s, ub_work.GetValue(base + 0));
                    ub_tmp.SetValue(s, ub_work.GetValue(base + 1));
                }
                Fence<HardEvent::S_V>();
                Adds(ub_blk, ub_blk, -r_max, 64);
                PipeBarrier<PIPE_V>();
                Maxs(ub_blk, ub_blk, -104.0F, 64);
                PipeBarrier<PIPE_V>();
                Exp(ub_blk, ub_blk, 64);
                PipeBarrier<PIPE_V>();
                Mul(ub_blk, ub_blk, ub_tmp, 64);
                PipeBarrier<PIPE_V>();
                ReduceSum(ub_sum, ub_blk, ub_x, 64);
                Fence<HardEvent::V_S>();
                const float r_sum = ub_sum.GetValue(0);

                const int32_t out_base = m * cols;
                ub_out.SetValue(out_base + 0, r_max);
                ub_out.SetValue(out_base + 1, r_sum);
                if (k >= 1) {
                    ub_out.SetValue(out_base + 2 + 0, rv0);
                    ub_out.SetValue(out_base + 2 + k + 0, static_cast<float>(ri0));
                }
                if (k >= 2) {
                    ub_out.SetValue(out_base + 2 + 1, rv1);
                    ub_out.SetValue(out_base + 2 + k + 1, static_cast<float>(ri1));
                }
                if (k >= 3) {
                    ub_out.SetValue(out_base + 2 + 2, rv2);
                    ub_out.SetValue(out_base + 2 + k + 2, static_cast<float>(ri2));
                }
                if (k >= 4) {
                    ub_out.SetValue(out_base + 2 + 3, rv3);
                    ub_out.SetValue(out_base + 2 + k + 3, static_cast<float>(ri3));
                }
            }
            Fence<HardEvent::S_MTE3>();
            DataCopy(g_gather[rank * 64], ub_out, 64);
        }
        Publish();
    }
}

}

extern "C" __global__ __mix__(1, 2) __schedmode__(1) __aicore__
void dspine_prolog_pack_k(
    uint64_t ctrl_gm, uint64_t target_toks_gm, uint64_t emb_w_gm,
    uint64_t rot_part_gm, uint64_t comm_in_gm,
    int32_t m_max, int32_t vocab_start, int32_t vocab_end, int32_t rank)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    dspine_ops::RunPrologPack(
        pipe, ctrl_gm, target_toks_gm, emb_w_gm, rot_part_gm, comm_in_gm,
        m_max, vocab_start, vocab_end, rank);
}

extern "C" __global__ __mix__(1, 2) __schedmode__(1) __aicore__
void dspine_prep_commit_seed_v3_k(
    uint64_t ctrl_gm,
    uint64_t hs_all_gm,
    uint64_t sin_cache_gm,
    uint64_t cos_cache_gm,
    uint64_t kv_hs_gm,
    uint64_t kv_sin_gm,
    uint64_t kv_cos_gm,
    uint64_t kv_slot_gm,
    uint64_t kv_qlen_gm,
    uint64_t kv_kvlen_gm,
    uint64_t seed_hs_gm,
    uint64_t seed_sin_gm,
    uint64_t seed_cos_gm,
    uint64_t seed_slot_gm,
    uint64_t seed_qlen_gm,
    uint64_t seed_kvlen_gm,
    int32_t kv_m_cap)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    dspine_ops::RunPrepCommitSeedV3(
        pipe, ctrl_gm, hs_all_gm, sin_cache_gm, cos_cache_gm,
        kv_hs_gm, kv_sin_gm, kv_cos_gm, kv_slot_gm, kv_qlen_gm, kv_kvlen_gm,
        seed_hs_gm, seed_sin_gm, seed_cos_gm, seed_slot_gm, seed_qlen_gm, seed_kvlen_gm,
        kv_m_cap);
}

extern "C" __global__ __mix__(1, 2) __schedmode__(1) __aicore__
void dspine_beam_step_v3_k(
    uint64_t sync_base,
    int32_t step,
    int32_t b_cur,
    int32_t k_expand,
    int32_t k_topk,
    int32_t b_next,
    int32_t pool_off,
    int32_t k_sfa,
    int32_t next_slot_cur,
    float beta_margin,
    float beta_depth,
    uint64_t gather_out_gm,
    uint64_t draft_ctrl_gm,
    uint64_t seed_top_i_gm,
    uint64_t seed_top_lp_gm,
    uint64_t seed_rec_gm,
    uint64_t prefix_topk_gm,
    uint64_t anc_slots_gm,
    uint64_t indices_buf_gm,
    uint64_t active_toks_gm,
    uint64_t active_scores_gm,
    uint64_t active_raw_lp_gm,
    uint64_t active_margins_gm,
    uint64_t active_pids_gm,
    uint64_t active_recs_gm,
    uint64_t recs_d_gm,
    uint64_t pool_scores_gm,
    uint64_t pool_raw_lp_gm,
    uint64_t pool_tokens_gm,
    uint64_t pool_parents_gm,
    uint64_t sin_cache_gm,
    uint64_t cos_cache_gm,
    uint64_t eshr_sin_gm,
    uint64_t eshr_cos_gm,
    uint64_t eshr_slot_gm,
    uint64_t eshr_kvlen_gm,
    uint64_t work_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    AscendC::SetSyncBaseAddr(sync_base);
    AscendC::TPipe pipe;
    dspine_ops::RunBeamStepV3(
        pipe, step, b_cur, k_expand, k_topk, b_next, pool_off, k_sfa, next_slot_cur,
        beta_margin, beta_depth,
        gather_out_gm, draft_ctrl_gm, seed_top_i_gm, seed_top_lp_gm, seed_rec_gm,
        prefix_topk_gm, anc_slots_gm, indices_buf_gm,
        active_toks_gm, active_scores_gm, active_raw_lp_gm, active_margins_gm,
        active_pids_gm, active_recs_gm, recs_d_gm,
        pool_scores_gm, pool_raw_lp_gm, pool_tokens_gm, pool_parents_gm,
        sin_cache_gm, cos_cache_gm, eshr_sin_gm, eshr_cos_gm, eshr_slot_gm, eshr_kvlen_gm,
        work_gm);
}

extern "C" __global__ __mix__(1, 2) __schedmode__(1) __aicore__
void dspine_build_tree_v3_k(
    uint64_t sync_base,
    int32_t width,
    int32_t min_w_act,
    int32_t pool_size,
    int32_t num_seq_ptrs,
    int32_t num_q_ptrs,
    int32_t num_rope_dsts,
    float t_base_ms,
    float c_w_ms,
    uint64_t draft_ctrl_gm,
    uint64_t pool_scores_gm,
    uint64_t pool_raw_lp_gm,
    uint64_t pool_tokens_gm,
    uint64_t pool_parents_gm,
    uint64_t pool_depths_gm,
    uint64_t tree_tokens_gm,
    uint64_t tree_parents_gm,
    uint64_t tree_depths_gm,
    uint64_t tree_mask_gm,
    uint64_t t_ids_gm,
    uint64_t t_pos_gm,
    uint64_t t_slots_gm,
    uint64_t t_slots_i64_gm,
    uint64_t dyn_w_gm,
    uint64_t seq_ptrs_gm,
    uint64_t q_ptrs_gm,
    uint64_t cos_cache_gm,
    uint64_t sin_cache_gm,
    uint64_t rope_dsts_gm,
    uint64_t work_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    AscendC::SetSyncBaseAddr(sync_base);
    AscendC::TPipe pipe;
    dspine_ops::RunBuildTreeV3(
        pipe, width, min_w_act, pool_size, num_seq_ptrs, num_q_ptrs, num_rope_dsts,
        t_base_ms, c_w_ms, draft_ctrl_gm,
        pool_scores_gm, pool_raw_lp_gm, pool_tokens_gm, pool_parents_gm, pool_depths_gm,
        tree_tokens_gm, tree_parents_gm, tree_depths_gm, tree_mask_gm,
        t_ids_gm, t_pos_gm, t_slots_gm, t_slots_i64_gm,
        dyn_w_gm, seq_ptrs_gm, q_ptrs_gm, cos_cache_gm, sin_cache_gm, rope_dsts_gm, work_gm);
}

extern "C" int tilert_dspine_prolog_pack_raw(
    void* stream, uint64_t ctrl_gm, uint64_t target_toks_gm, uint64_t emb_w_gm,
    uint64_t rot_part_gm, uint64_t comm_in_gm,
    int32_t m_max, int32_t vocab_start, int32_t vocab_end, int32_t rank)
{
    dspine_prolog_pack_k<<<24, nullptr, stream>>>(
        ctrl_gm, target_toks_gm, emb_w_gm, rot_part_gm, comm_in_gm,
        m_max, vocab_start, vocab_end, rank);
    return 0;
}

extern "C" int tilert_dspine_prep_commit_seed_v3_raw(
    void* stream,
    uint64_t ctrl_gm,
    uint64_t hs_all_gm,
    uint64_t sin_cache_gm,
    uint64_t cos_cache_gm,
    uint64_t kv_hs_gm,
    uint64_t kv_sin_gm,
    uint64_t kv_cos_gm,
    uint64_t kv_slot_gm,
    uint64_t kv_qlen_gm,
    uint64_t kv_kvlen_gm,
    uint64_t seed_hs_gm,
    uint64_t seed_sin_gm,
    uint64_t seed_cos_gm,
    uint64_t seed_slot_gm,
    uint64_t seed_qlen_gm,
    uint64_t seed_kvlen_gm,
    int32_t kv_m_cap)
{
    dspine_prep_commit_seed_v3_k<<<24, nullptr, stream>>>(
        ctrl_gm, hs_all_gm, sin_cache_gm, cos_cache_gm,
        kv_hs_gm, kv_sin_gm, kv_cos_gm, kv_slot_gm, kv_qlen_gm, kv_kvlen_gm,
        seed_hs_gm, seed_sin_gm, seed_cos_gm, seed_slot_gm, seed_qlen_gm, seed_kvlen_gm,
        kv_m_cap);
    return 0;
}

extern "C" int tilert_dspine_beam_step_v3_raw(
    void* stream,
    uint64_t sync_base,
    int32_t step,
    int32_t b_cur,
    int32_t k_expand,
    int32_t k_topk,
    int32_t b_next,
    int32_t pool_off,
    int32_t k_sfa,
    int32_t next_slot_cur,
    float beta_margin,
    float beta_depth,
    uint64_t gather_out_gm,
    uint64_t draft_ctrl_gm,
    uint64_t seed_top_i_gm,
    uint64_t seed_top_lp_gm,
    uint64_t seed_rec_gm,
    uint64_t prefix_topk_gm,
    uint64_t anc_slots_gm,
    uint64_t indices_buf_gm,
    uint64_t active_toks_gm,
    uint64_t active_scores_gm,
    uint64_t active_raw_lp_gm,
    uint64_t active_margins_gm,
    uint64_t active_pids_gm,
    uint64_t active_recs_gm,
    uint64_t recs_d_gm,
    uint64_t pool_scores_gm,
    uint64_t pool_raw_lp_gm,
    uint64_t pool_tokens_gm,
    uint64_t pool_parents_gm,
    uint64_t sin_cache_gm,
    uint64_t cos_cache_gm,
    uint64_t eshr_sin_gm,
    uint64_t eshr_cos_gm,
    uint64_t eshr_slot_gm,
    uint64_t eshr_kvlen_gm,
    uint64_t work_gm)
{
    dspine_beam_step_v3_k<<<24, nullptr, stream>>>(
        sync_base, step, b_cur, k_expand, k_topk, b_next, pool_off, k_sfa, next_slot_cur,
        beta_margin, beta_depth,
        gather_out_gm, draft_ctrl_gm, seed_top_i_gm, seed_top_lp_gm, seed_rec_gm,
        prefix_topk_gm, anc_slots_gm, indices_buf_gm,
        active_toks_gm, active_scores_gm, active_raw_lp_gm, active_margins_gm,
        active_pids_gm, active_recs_gm, recs_d_gm,
        pool_scores_gm, pool_raw_lp_gm, pool_tokens_gm, pool_parents_gm,
        sin_cache_gm, cos_cache_gm, eshr_sin_gm, eshr_cos_gm, eshr_slot_gm, eshr_kvlen_gm,
        work_gm);
    return 0;
}

extern "C" int tilert_dspine_build_tree_v3_raw(
    void* stream,
    uint64_t sync_base,
    int32_t width,
    int32_t min_w_act,
    int32_t pool_size,
    int32_t num_seq_ptrs,
    int32_t num_q_ptrs,
    int32_t num_rope_dsts,
    float t_base_ms,
    float c_w_ms,
    uint64_t draft_ctrl_gm,
    uint64_t pool_scores_gm,
    uint64_t pool_raw_lp_gm,
    uint64_t pool_tokens_gm,
    uint64_t pool_parents_gm,
    uint64_t pool_depths_gm,
    uint64_t tree_tokens_gm,
    uint64_t tree_parents_gm,
    uint64_t tree_depths_gm,
    uint64_t tree_mask_gm,
    uint64_t t_ids_gm,
    uint64_t t_pos_gm,
    uint64_t t_slots_gm,
    uint64_t t_slots_i64_gm,
    uint64_t dyn_w_gm,
    uint64_t seq_ptrs_gm,
    uint64_t q_ptrs_gm,
    uint64_t cos_cache_gm,
    uint64_t sin_cache_gm,
    uint64_t rope_dsts_gm,
    uint64_t work_gm)
{
    dspine_build_tree_v3_k<<<24, nullptr, stream>>>(
        sync_base, width, min_w_act, pool_size, num_seq_ptrs, num_q_ptrs, num_rope_dsts,
        t_base_ms, c_w_ms, draft_ctrl_gm,
        pool_scores_gm, pool_raw_lp_gm, pool_tokens_gm, pool_parents_gm, pool_depths_gm,
        tree_tokens_gm, tree_parents_gm, tree_depths_gm, tree_mask_gm,
        t_ids_gm, t_pos_gm, t_slots_gm, t_slots_i64_gm,
        dyn_w_gm, seq_ptrs_gm, q_ptrs_gm, cos_cache_gm, sin_cache_gm, rope_dsts_gm, work_gm);
    return 0;
}

extern "C" __global__ __mix__(1, 2) __schedmode__(1) __aicore__
void step3c_kv_compact_k(uint64_t descs_gm, uint64_t ctrl_gm, int32_t num_tensors)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    dspine_ops::RunKvCompact(pipe, descs_gm, ctrl_gm, num_tensors);
}

extern "C" __global__ __mix__(1, 2) __schedmode__(1) __aicore__
void step3c_prolog_pack_k(
    uint64_t active_toks_gm, uint64_t emb_w_gm, uint64_t rot_part_gm,
    uint64_t rot_bias_gm, uint64_t comm_in_gm,
    int32_t b_rows, int32_t vocab_start, int32_t vocab_end, int32_t rank)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    dspine_ops::RunStep3cPrologPack(
        pipe, active_toks_gm, emb_w_gm, rot_part_gm, rot_bias_gm, comm_in_gm,
        b_rows, vocab_start, vocab_end, rank);
}

extern "C" __global__ __mix__(1, 2) __schedmode__(1) __aicore__
void step3c_dual_rmsnorm_slice_k(
    uint64_t comm_out_gm, uint64_t enorm_w_gm, uint64_t hnorm_w_gm,
    uint64_t eh_in_gm, int32_t b_rows, int32_t rank, float eps)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    dspine_ops::RunStep3cDualRmsNormSlice(
        pipe, comm_out_gm, enorm_w_gm, hnorm_w_gm, eh_in_gm, b_rows, rank, eps);
}

extern "C" __global__ __mix__(1, 2) __schedmode__(1) __aicore__
void step3c_logits_topk_lse_bf16_k(
    uint64_t sync_base, uint64_t logits_gm, uint64_t work_gm, uint64_t gather_in_gm,
    int32_t m_rows, int32_t row_stride, int32_t valid_vocab, int32_t vocab_start,
    int32_t k, int32_t rank)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    AscendC::SetSyncBaseAddr(sync_base);
    AscendC::TPipe pipe;
    dspine_ops::RunLogitsTopkLseImpl<true>(
        pipe, logits_gm, work_gm, gather_in_gm,
        m_rows, row_stride, valid_vocab, vocab_start, k, rank);
}

extern "C" int tilert_kv_compact_raw(
    void* stream, uint64_t descs_gm, uint64_t ctrl_gm, int32_t num_tensors)
{
    step3c_kv_compact_k<<<24, nullptr, stream>>>(descs_gm, ctrl_gm, num_tensors);
    return 0;
}

extern "C" int tilert_prolog_pack_raw(
    void* stream, uint64_t active_toks_gm, uint64_t emb_w_gm, uint64_t rot_part_gm,
    uint64_t rot_bias_gm, uint64_t comm_in_gm,
    int32_t b_rows, int32_t vocab_start, int32_t vocab_end, int32_t rank)
{
    step3c_prolog_pack_k<<<24, nullptr, stream>>>(
        active_toks_gm, emb_w_gm, rot_part_gm, rot_bias_gm, comm_in_gm,
        b_rows, vocab_start, vocab_end, rank);
    return 0;
}

extern "C" int tilert_dual_rmsnorm_slice_raw(
    void* stream, uint64_t comm_out_gm, uint64_t enorm_w_gm, uint64_t hnorm_w_gm,
    uint64_t eh_in_gm, int32_t b_rows, int32_t rank, float eps)
{
    step3c_dual_rmsnorm_slice_k<<<24, nullptr, stream>>>(
        comm_out_gm, enorm_w_gm, hnorm_w_gm, eh_in_gm, b_rows, rank, eps);
    return 0;
}

extern "C" int tilert_logits_topk_lse_bf16_raw(
    void* stream, uint64_t sync_base, uint64_t logits_gm, uint64_t work_gm, uint64_t gather_in_gm,
    int32_t m_rows, int32_t row_stride, int32_t valid_vocab, int32_t vocab_start,
    int32_t k, int32_t rank)
{
    step3c_logits_topk_lse_bf16_k<<<24, nullptr, stream>>>(
        sync_base, logits_gm, work_gm, gather_in_gm,
        m_rows, row_stride, valid_vocab, vocab_start, k, rank);
    return 0;
}
"""

HOST_CPP = """#include <cstdint>
#include <torch_npu/csrc/core/npu/NPUStream.h>

extern "C" {
int tilert_dspine_prolog_pack_raw(
    void* stream, uint64_t ctrl_gm, uint64_t target_toks_gm, uint64_t emb_w_gm,
    uint64_t rot_part_gm, uint64_t comm_in_gm,
    int32_t m_max, int32_t vocab_start, int32_t vocab_end, int32_t rank);

int tilert_dspine_prep_commit_seed_v3_raw(
    void* stream, uint64_t ctrl_gm, uint64_t hs_all_gm,
    uint64_t sin_cache_gm, uint64_t cos_cache_gm,
    uint64_t kv_hs_gm, uint64_t kv_sin_gm, uint64_t kv_cos_gm,
    uint64_t kv_slot_gm, uint64_t kv_qlen_gm, uint64_t kv_kvlen_gm,
    uint64_t seed_hs_gm, uint64_t seed_sin_gm, uint64_t seed_cos_gm,
    uint64_t seed_slot_gm, uint64_t seed_qlen_gm, uint64_t seed_kvlen_gm,
    int32_t kv_m_cap);
int tilert_dspine_beam_step_v3_raw(
    void* stream, uint64_t sync_base,
    int32_t step, int32_t b_cur, int32_t k_expand, int32_t k_topk, int32_t b_next, int32_t pool_off,
    int32_t k_sfa, int32_t next_slot_cur, float beta_margin, float beta_depth,
    uint64_t gather_out_gm, uint64_t draft_ctrl_gm,
    uint64_t seed_top_i_gm, uint64_t seed_top_lp_gm, uint64_t seed_rec_gm,
    uint64_t prefix_topk_gm, uint64_t anc_slots_gm, uint64_t indices_buf_gm,
    uint64_t active_toks_gm, uint64_t active_scores_gm,
    uint64_t active_raw_lp_gm, uint64_t active_margins_gm,
    uint64_t active_pids_gm, uint64_t active_recs_gm, uint64_t recs_d_gm,
    uint64_t pool_scores_gm, uint64_t pool_raw_lp_gm,
    uint64_t pool_tokens_gm, uint64_t pool_parents_gm,
    uint64_t sin_cache_gm, uint64_t cos_cache_gm,
    uint64_t eshr_sin_gm, uint64_t eshr_cos_gm, uint64_t eshr_slot_gm, uint64_t eshr_kvlen_gm,
    uint64_t work_gm);
int tilert_dspine_build_tree_v3_raw(
    void* stream, uint64_t sync_base,
    int32_t width, int32_t min_w_act, int32_t pool_size,
    int32_t num_seq_ptrs, int32_t num_q_ptrs, int32_t num_rope_dsts,
    float t_base_ms, float c_w_ms, uint64_t draft_ctrl_gm,
    uint64_t pool_scores_gm, uint64_t pool_raw_lp_gm,
    uint64_t pool_tokens_gm, uint64_t pool_parents_gm, uint64_t pool_depths_gm,
    uint64_t tree_tokens_gm, uint64_t tree_parents_gm,
    uint64_t tree_depths_gm, uint64_t tree_mask_gm,
    uint64_t t_ids_gm, uint64_t t_pos_gm, uint64_t t_slots_gm, uint64_t t_slots_i64_gm,
    uint64_t dyn_w_gm, uint64_t seq_ptrs_gm, uint64_t q_ptrs_gm,
    uint64_t cos_cache_gm, uint64_t sin_cache_gm, uint64_t rope_dsts_gm, uint64_t work_gm);

int tilert_kv_compact_raw(void* stream, uint64_t descs_gm, uint64_t ctrl_gm, int32_t num_tensors);
int tilert_prolog_pack_raw(
    void* stream, uint64_t active_toks_gm, uint64_t emb_w_gm, uint64_t rot_part_gm,
    uint64_t rot_bias_gm, uint64_t comm_in_gm,
    int32_t b_rows, int32_t vocab_start, int32_t vocab_end, int32_t rank);
int tilert_dual_rmsnorm_slice_raw(
    void* stream, uint64_t comm_out_gm, uint64_t enorm_w_gm, uint64_t hnorm_w_gm,
    uint64_t eh_in_gm, int32_t b_rows, int32_t rank, float eps);
int tilert_logits_topk_lse_bf16_raw(
    void* stream, uint64_t sync_base, uint64_t logits_gm, uint64_t work_gm, uint64_t gather_in_gm,
    int32_t m_rows, int32_t row_stride, int32_t valid_vocab, int32_t vocab_start,
    int32_t k, int32_t rank);

void* tilert_get_flushed_stream()
{
    return c10_npu::getCurrentNPUStream().stream(true);
}

int tilert_kv_compact_launch(void*, uint64_t descs_gm, uint64_t ctrl_gm, int32_t num_tensors)
{
    return tilert_kv_compact_raw(
        c10_npu::getCurrentNPUStream().stream(true), descs_gm, ctrl_gm, num_tensors);
}

int tilert_prolog_pack_launch(
    void*, uint64_t active_toks_gm, uint64_t emb_w_gm, uint64_t rot_part_gm,
    uint64_t rot_bias_gm, uint64_t comm_in_gm,
    int32_t b_rows, int32_t vocab_start, int32_t vocab_end, int32_t rank)
{
    return tilert_prolog_pack_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        active_toks_gm, emb_w_gm, rot_part_gm, rot_bias_gm, comm_in_gm,
        b_rows, vocab_start, vocab_end, rank);
}

int tilert_dual_rmsnorm_slice_launch(
    void*, uint64_t comm_out_gm, uint64_t enorm_w_gm, uint64_t hnorm_w_gm,
    uint64_t eh_in_gm, int32_t b_rows, int32_t rank, float eps)
{
    return tilert_dual_rmsnorm_slice_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        comm_out_gm, enorm_w_gm, hnorm_w_gm, eh_in_gm, b_rows, rank, eps);
}

int tilert_logits_topk_lse_bf16_launch(
    void*, uint64_t sync_base, uint64_t logits_gm, uint64_t work_gm, uint64_t gather_in_gm,
    int32_t m_rows, int32_t row_stride, int32_t valid_vocab, int32_t vocab_start,
    int32_t k, int32_t rank)
{
    return tilert_logits_topk_lse_bf16_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        sync_base, logits_gm, work_gm, gather_in_gm,
        m_rows, row_stride, valid_vocab, vocab_start, k, rank);
}

int tilert_dspine_prolog_pack_launch(
    void*, uint64_t ctrl_gm, uint64_t target_toks_gm, uint64_t emb_w_gm,
    uint64_t rot_part_gm, uint64_t comm_in_gm,
    int32_t m_max, int32_t vocab_start, int32_t vocab_end, int32_t rank)
{
    return tilert_dspine_prolog_pack_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        ctrl_gm, target_toks_gm, emb_w_gm, rot_part_gm, comm_in_gm,
        m_max, vocab_start, vocab_end, rank);
}

int tilert_dspine_prep_commit_seed_v3_launch(
    void*, uint64_t ctrl_gm, uint64_t hs_all_gm, uint64_t sin_cache_gm, uint64_t cos_cache_gm,
    uint64_t kv_hs_gm, uint64_t kv_sin_gm, uint64_t kv_cos_gm,
    uint64_t kv_slot_gm, uint64_t kv_qlen_gm, uint64_t kv_kvlen_gm,
    uint64_t seed_hs_gm, uint64_t seed_sin_gm, uint64_t seed_cos_gm,
    uint64_t seed_slot_gm, uint64_t seed_qlen_gm, uint64_t seed_kvlen_gm,
    int32_t kv_m_cap)
{
    return tilert_dspine_prep_commit_seed_v3_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        ctrl_gm, hs_all_gm, sin_cache_gm, cos_cache_gm,
        kv_hs_gm, kv_sin_gm, kv_cos_gm, kv_slot_gm, kv_qlen_gm, kv_kvlen_gm,
        seed_hs_gm, seed_sin_gm, seed_cos_gm, seed_slot_gm, seed_qlen_gm, seed_kvlen_gm,
        kv_m_cap);
}

int tilert_dspine_beam_step_v3_launch(
    void*, uint64_t sync_base,
    int32_t step, int32_t b_cur, int32_t k_expand, int32_t k_topk, int32_t b_next, int32_t pool_off,
    int32_t k_sfa, int32_t next_slot_cur, float beta_margin, float beta_depth,
    uint64_t gather_out_gm, uint64_t draft_ctrl_gm,
    uint64_t seed_top_i_gm, uint64_t seed_top_lp_gm, uint64_t seed_rec_gm,
    uint64_t prefix_topk_gm, uint64_t anc_slots_gm, uint64_t indices_buf_gm,
    uint64_t active_toks_gm, uint64_t active_scores_gm,
    uint64_t active_raw_lp_gm, uint64_t active_margins_gm,
    uint64_t active_pids_gm, uint64_t active_recs_gm, uint64_t recs_d_gm,
    uint64_t pool_scores_gm, uint64_t pool_raw_lp_gm,
    uint64_t pool_tokens_gm, uint64_t pool_parents_gm,
    uint64_t sin_cache_gm, uint64_t cos_cache_gm,
    uint64_t eshr_sin_gm, uint64_t eshr_cos_gm, uint64_t eshr_slot_gm, uint64_t eshr_kvlen_gm,
    uint64_t work_gm)
{
    return tilert_dspine_beam_step_v3_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        sync_base, step, b_cur, k_expand, k_topk, b_next, pool_off,
        k_sfa, next_slot_cur, beta_margin, beta_depth,
        gather_out_gm, draft_ctrl_gm, seed_top_i_gm, seed_top_lp_gm, seed_rec_gm,
        prefix_topk_gm, anc_slots_gm, indices_buf_gm,
        active_toks_gm, active_scores_gm, active_raw_lp_gm, active_margins_gm,
        active_pids_gm, active_recs_gm, recs_d_gm,
        pool_scores_gm, pool_raw_lp_gm, pool_tokens_gm, pool_parents_gm,
        sin_cache_gm, cos_cache_gm, eshr_sin_gm, eshr_cos_gm, eshr_slot_gm, eshr_kvlen_gm,
        work_gm);
}

int tilert_dspine_build_tree_v3_launch(
    void*, uint64_t sync_base,
    int32_t width, int32_t min_w_act, int32_t pool_size,
    int32_t num_seq_ptrs, int32_t num_q_ptrs, int32_t num_rope_dsts,
    float t_base_ms, float c_w_ms, uint64_t draft_ctrl_gm,
    uint64_t pool_scores_gm, uint64_t pool_raw_lp_gm,
    uint64_t pool_tokens_gm, uint64_t pool_parents_gm, uint64_t pool_depths_gm,
    uint64_t tree_tokens_gm, uint64_t tree_parents_gm,
    uint64_t tree_depths_gm, uint64_t tree_mask_gm,
    uint64_t t_ids_gm, uint64_t t_pos_gm, uint64_t t_slots_gm, uint64_t t_slots_i64_gm,
    uint64_t dyn_w_gm, uint64_t seq_ptrs_gm, uint64_t q_ptrs_gm,
    uint64_t cos_cache_gm, uint64_t sin_cache_gm, uint64_t rope_dsts_gm, uint64_t work_gm)
{
    return tilert_dspine_build_tree_v3_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        sync_base, width, min_w_act, pool_size, num_seq_ptrs, num_q_ptrs, num_rope_dsts,
        t_base_ms, c_w_ms, draft_ctrl_gm,
        pool_scores_gm, pool_raw_lp_gm, pool_tokens_gm, pool_parents_gm, pool_depths_gm,
        tree_tokens_gm, tree_parents_gm, tree_depths_gm, tree_mask_gm,
        t_ids_gm, t_pos_gm, t_slots_gm, t_slots_i64_gm,
        dyn_w_gm, seq_ptrs_gm, q_ptrs_gm, cos_cache_gm, sin_cache_gm, rope_dsts_gm, work_gm);
}
}"""


def default_build_dir() -> Path:
    env_dir = os.environ.get("TILERT_BUILD_DIR")
    return Path(env_dir) / "dspine_ops" if env_dir else DEFAULT_OUT


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


def build_dspine_ops(out_dir: str | Path | None = None) -> Path:
    out = default_build_dir() if out_dir is None else Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dspine_ops.cpp").write_text(KERNEL_CPP)
    (out / "dspine_host.cpp").write_text(HOST_CPP)

    cann = _cann_root()
    site = _site_root()
    inc = [
        "-I",
        str(cann / "include"),
        "-I",
        "/assets/shmem-lib/include",
        "-I",
        "/assets/shmem-lib/src/device",
    ]
    for sub in ASC_SUBDIRS:
        inc.extend(["-I", str(cann / "aarch64-linux" / sub)])
    host_inc = [
        "-I", str(site / "torch/include"),
        "-I", str(site / "torch/include/torch/csrc/api/include"),
        "-I", str(site / "torch_npu/include"),
        "-I", str(cann / "include"),
        "-I", "/assets/shmem-lib/include",
    ]
    libs = [
        "-L", "/assets/shmem-lib/lib", "-lshmem", "-lshmem_utils",
        "-L", str(cann / "aarch64-linux/lib64"), "-L", str(cann / "lib64"),
        "-L", str(site / "torch/lib"), "-L", str(site / "torch_npu/lib"),
        "-lascendcl", "-lruntime", "-lunified_dlog", "-lc_sec",
        "-lregister", "-lprofapi", "-ltiling_api", "-lplatform",
        "-ltorch", "-ltorch_cpu", "-lc10", "-ltorch_npu",
    ]
    k_cmd = [
        str(cann / "bin/bisheng"), "-xcce", "--npu-arch=dav-2201",
        "-std=c++17", "-O2", "-w", "-ffp-contract=off", "-fPIC",
        *inc, "-c", str(out / "dspine_ops.cpp"), "-o", str(out / "dspine_ops.o"),
    ]
    h_cmd = [
        "c++", "-std=c++17", "-O2", "-fPIC", "-D_GLIBCXX_USE_CXX11_ABI=1",
        *host_inc, "-c", str(out / "dspine_host.cpp"), "-o", str(out / "dspine_host.o"),
    ]
    so_path = out / "tilert_dspine_ops.so"
    link_cmd = [
        "c++", "-shared", "-Wl,--no-undefined", "-Wl,-Bsymbolic",
        str(out / "dspine_ops.o"), str(out / "dspine_host.o"), "-o", str(so_path), *libs,
    ]
    script = (
        f"set -euo pipefail\n"
        f"set +u; source {shlex.quote(str(cann / 'set_env.sh'))}; set -u\n"
        f"{' '.join(shlex.quote(x) for x in k_cmd)} &\n"
        f"{' '.join(shlex.quote(x) for x in h_cmd)} &\n"
        f"wait\n"
        f"{' '.join(shlex.quote(x) for x in link_cmd)}\n"
    )
    subprocess.run(["bash", "-c", script], check=True)
    sha = hashlib.sha256(so_path.read_bytes()).hexdigest()[:12]
    print(f"BUILD_DSPINE_OPS_OK {sha} {so_path}", flush=True)
    return so_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    build_dspine_ops(args.out)
