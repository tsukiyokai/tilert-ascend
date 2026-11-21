import argparse
from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
import shlex
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

_DSPINE_HEAD_CPP = """#if defined(__CCE_AICORE__) && !defined(__NPU_DEVICE__)
#define __NPU_DEVICE__ 1
#endif
#include "kernel_operator.h"
#include "shmem.h"

namespace dspine_ops {
using namespace AscendC;
"""

DSPINE_WALK_SHARED_CPP = """namespace dspine_shared {
using namespace AscendC;

#ifndef WORLD_SIZE
#define WORLD_SIZE 8
#endif

constexpr int32_t kDspineWorldSize = WORLD_SIZE;
constexpr int32_t kDspineShardDim = 6144 / WORLD_SIZE;
constexpr int32_t kDspineEhIn = 2 * kDspineShardDim;
constexpr int32_t kDspineVecRanks = (WORLD_SIZE > 8) ? 16 : 8;

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

__aicore__ inline void DcciRange(uint64_t addr, int32_t bytes)
{
    if (addr == 0ULL || bytes <= 0) {
        return;
    }
    const uint64_t start = addr & ~63ULL;
    const uint64_t end = (addr + static_cast<uint64_t>(bytes) + 63ULL) & ~63ULL;
    for (uint64_t p = start; p < end; p += 64ULL) {
        GlobalTensor<int64_t> g;
        g.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(p), 8);
        DataCacheCleanAndInvalid<
            int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(g);
    }
}

constexpr uint64_t kDspineFlagOff = 128ULL;
constexpr uint64_t kDspineFlagsPerPlane = 128ULL;
constexpr uint64_t kDspineDataOff = 65664ULL;
constexpr uint64_t kDspineSlotBytes = 131072ULL;
constexpr uint64_t kDspinePlaneBytes = 2097152ULL;

__aicore__ inline uint32_t DspineReadSeq(uint64_t win_gm)
{
    DcciRange(win_gm, 64);
    return *reinterpret_cast<__gm__ const uint32_t*>(win_gm);
}

__aicore__ inline void DspineWriteSeq(uint64_t win_gm, uint32_t seq)
{
    *reinterpret_cast<__gm__ uint32_t*>(win_gm) = seq;
    PipeBarrier<PIPE_ALL>();
    DcciRange(win_gm, 64);
}

__aicore__ inline __gm__ uint8_t* DspinePayloadPtr(
    uint64_t win_gm, uint32_t epoch, int32_t pe)
{
    return reinterpret_cast<__gm__ uint8_t*>(
        win_gm + kDspineDataOff
        + static_cast<uint64_t>(epoch & 7U) * kDspinePlaneBytes
        + static_cast<uint64_t>(pe) * kDspineSlotBytes);
}

__aicore__ inline __gm__ int32_t* DspineFlagPtr(
    uint64_t win_gm, uint32_t epoch, int32_t sender)
{
    return reinterpret_cast<__gm__ int32_t*>(
        win_gm + kDspineFlagOff
        + (static_cast<uint64_t>(epoch & 7U) * kDspineFlagsPerPlane
           + static_cast<uint64_t>(sender)) * 64ULL);
}

__aicore__ inline __gm__ int32_t* DspineArS1FlagPtr(
    uint64_t win_gm, uint32_t epoch, int32_t worker)
{
    return reinterpret_cast<__gm__ int32_t*>(
        win_gm + kDspineFlagOff
        + (static_cast<uint64_t>(epoch & 7U) * kDspineFlagsPerPlane
           + 16ULL + static_cast<uint64_t>(worker)) * 64ULL);
}

__aicore__ inline __gm__ int32_t* DspineArS2FlagPtr(
    uint64_t win_gm, uint32_t epoch, int32_t worker, int32_t lr)
{
    return reinterpret_cast<__gm__ int32_t*>(
        win_gm + kDspineFlagOff
        + (static_cast<uint64_t>(epoch & 7U) * kDspineFlagsPerPlane
           + 32ULL + static_cast<uint64_t>(worker) * 8ULL
           + static_cast<uint64_t>(lr)) * 64ULL);
}

__aicore__ inline void Inline8p2Ar(
    TPipe& pipe,
    uint64_t comm_win_gm,
    uint32_t epoch,
    uint64_t partial_gm,
    uint64_t out_gm,
    int32_t active_elements,
    int32_t rank,
    uint64_t extra_f32_in_gm = 0ULL,
    uint64_t extra_f32_out_gm = 0ULL,
    int32_t extra_f32_count = 0)
{
    if ASCEND_IS_AIV {
        const int32_t aiv = GetBlockIdx();
        const int32_t xpeer = rank ^ 8;
        const int32_t mod_base = rank & 8;
        const int32_t my_lr = rank & 7;
        const int32_t hi_half = (rank >> 3) & 1;

        const int32_t worker = aiv >> 3;
        const int32_t lr_off = aiv & 7;
        const int32_t peer_lr = (my_lr + lr_off) & 7;
        const int32_t peer = mod_base | peer_lr;
        const bool has_extra = (worker == 0 && extra_f32_count > 0 && extra_f32_count <= 8
            && extra_f32_in_gm != 0ULL && extra_f32_out_gm != 0ULL);

        const int32_t w_elems = active_elements / 6;
        const int32_t w_off = worker * w_elems;
        const int32_t c_elems = active_elements / 48;
        const int32_t c_off = aiv * c_elems;

        TBuf<TPosition::VECCALC> b_stage, b_acc, b_term, b_sq_raw, b_sq_loc, b_sq_s1, b_sq_acc;
        pipe.InitBuffer(b_stage, 2 * 8192 * sizeof(bfloat16_t));
        pipe.InitBuffer(b_acc, 8 * 1024 * sizeof(float));
        pipe.InitBuffer(b_term, 8192 * sizeof(float));
        pipe.InitBuffer(b_sq_raw, 64 * sizeof(float));
        pipe.InitBuffer(b_sq_loc, 8 * sizeof(float));
        pipe.InitBuffer(b_sq_s1, 8 * sizeof(float));
        pipe.InitBuffer(b_sq_acc, 64 * sizeof(float));
        auto stage_buf = b_stage.Get<bfloat16_t>();
        auto acc = b_acc.Get<float>();
        auto term = b_term.Get<float>();
        auto ub_sq_raw = b_sq_raw.Get<float>();
        auto ub_sq_loc = b_sq_loc.Get<float>();
        auto ub_sq_s1 = b_sq_s1.Get<float>();
        auto ub_sq_acc = b_sq_acc.Get<float>();

        __gm__ bfloat16_t* local_s1_bf16 = reinterpret_cast<__gm__ bfloat16_t*>(
            DspinePayloadPtr(comm_win_gm, epoch, 0));
        __gm__ bfloat16_t* remote_s1_bf16 = reinterpret_cast<__gm__ bfloat16_t*>(
            aclshmem_ptr(local_s1_bf16, xpeer));
        __gm__ float* local_s2_fp32 = reinterpret_cast<__gm__ float*>(
            reinterpret_cast<__gm__ uint8_t*>(local_s1_bf16) + kDspineSlotBytes);
        __gm__ float* local_s1_sq = reinterpret_cast<__gm__ float*>(
            DspinePayloadPtr(comm_win_gm, epoch, 7));
        __gm__ float* remote_s1_sq = has_extra
            ? reinterpret_cast<__gm__ float*>(aclshmem_ptr(local_s1_sq, xpeer)) : nullptr;
        __gm__ float* local_s2_sq = local_s1_sq + 1024;

        GlobalTensor<bfloat16_t> g_part, g_out;
        GlobalTensor<float> g_sq_in;
        g_part.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(partial_gm), active_elements);
        g_out.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(out_gm), active_elements);
        if (has_extra) {
            g_sq_in.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(extra_f32_in_gm), 64);
        }

        if (lr_off == 0) {
            GlobalTensor<bfloat16_t> g_s1_dst;
            g_s1_dst.SetGlobalBuffer(remote_s1_bf16 + w_off, w_elems);
            DataCopy(stage_buf, g_part[w_off], w_elems);
            if (has_extra) {
                DataCopy(ub_sq_raw, g_sq_in, 64);
            }
            Fence<HardEvent::MTE2_MTE3>();
            DataCopy(g_s1_dst, stage_buf, w_elems);
            if (has_extra) {
                Fence<HardEvent::MTE2_S>();
                for (int32_t r = 0; r < extra_f32_count; ++r) {
                    ub_sq_loc.SetValue(r, ub_sq_raw.GetValue(r * 16));
                }
                Fence<HardEvent::S_MTE3>();
                GlobalTensor<float> g_s1_sq_dst;
                g_s1_sq_dst.SetGlobalBuffer(remote_s1_sq, 8);
                DataCopy(g_s1_sq_dst, ub_sq_loc, 8);
            }
            Fence<HardEvent::MTE3_S>();
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::DDR>();
            __gm__ int32_t* rem_s1_sig = reinterpret_cast<__gm__ int32_t*>(
                aclshmem_ptr(DspineArS1FlagPtr(comm_win_gm, epoch, worker), xpeer));
            *rem_s1_sig = static_cast<int32_t>(epoch);
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(rem_s1_sig));
        }
        __gm__ volatile int32_t* loc_s1_sig = DspineArS1FlagPtr(comm_win_gm, epoch, worker);
        while (true) {
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(
                const_cast<__gm__ int32_t*>(loc_s1_sig)));
            if (*loc_s1_sig == static_cast<int32_t>(epoch)) {
                break;
            }
        }

        if (peer != rank) {
            __gm__ float* remote_s2_fp32 = reinterpret_cast<__gm__ float*>(
                aclshmem_ptr(local_s2_fp32, peer));
            GlobalTensor<bfloat16_t> g_s1_loc;
            g_s1_loc.SetGlobalBuffer(local_s1_bf16 + w_off, w_elems);
            DataCopy(stage_buf[0], g_part[w_off], w_elems);
            DataCopy(stage_buf[8192], g_s1_loc, w_elems);
            if (has_extra) {
                GlobalTensor<float> g_s1_sq_loc;
                g_s1_sq_loc.SetGlobalBuffer(local_s1_sq, 8);
                DataCopy(ub_sq_raw, g_sq_in, 64);
                DataCopy(ub_sq_s1, g_s1_sq_loc, 8);
            }
            Fence<HardEvent::MTE2_V>();
            if (has_extra) {
                Fence<HardEvent::MTE2_S>();
                for (int32_t r = 0; r < extra_f32_count; ++r) {
                    ub_sq_loc.SetValue(r, ub_sq_raw.GetValue(r * 16));
                }
                Fence<HardEvent::S_V>();
                Add(ub_sq_loc, hi_half ? ub_sq_s1 : ub_sq_loc,
                    hi_half ? ub_sq_loc : ub_sq_s1, 8);
            }
            Cast(acc, hi_half ? stage_buf[8192] : stage_buf[0],
                 RoundMode::CAST_NONE, w_elems);
            Cast(term, hi_half ? stage_buf[0] : stage_buf[8192],
                 RoundMode::CAST_NONE, w_elems);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, term, w_elems);
            Fence<HardEvent::V_MTE3>();
            GlobalTensor<float> g_s2_dst;
            g_s2_dst.SetGlobalBuffer(
                remote_s2_fp32 + static_cast<int64_t>(my_lr) * active_elements + w_off,
                w_elems);
            DataCopy(g_s2_dst, acc, w_elems);
            if (has_extra) {
                __gm__ float* remote_s2_sq = reinterpret_cast<__gm__ float*>(
                    aclshmem_ptr(local_s2_sq, peer));
                GlobalTensor<float> g_s2_sq_dst;
                g_s2_sq_dst.SetGlobalBuffer(remote_s2_sq + static_cast<int64_t>(my_lr) * 16, 8);
                DataCopy(g_s2_sq_dst, ub_sq_loc, 8);
            }
            Fence<HardEvent::MTE3_S>();
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::DDR>();
            __gm__ int32_t* rem_s2_sig = reinterpret_cast<__gm__ int32_t*>(
                aclshmem_ptr(DspineArS2FlagPtr(comm_win_gm, epoch, worker, my_lr), peer));
            *rem_s2_sig = static_cast<int32_t>(epoch);
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(rem_s2_sig));
        }

        for (int32_t lr = 0; lr < 8; ++lr) {
            if (lr != my_lr) {
                __gm__ volatile int32_t* loc_s2_sig =
                    DspineArS2FlagPtr(comm_win_gm, epoch, worker, lr);
                while (true) {
                    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(
                        const_cast<__gm__ int32_t*>(loc_s2_sig)));
                    if (*loc_s2_sig == static_cast<int32_t>(epoch)) {
                        break;
                    }
                }
                GlobalTensor<float> g_s2_src;
                g_s2_src.SetGlobalBuffer(
                    local_s2_fp32 + static_cast<int64_t>(lr) * active_elements + c_off,
                    c_elems);
                DataCopy(acc[lr * c_elems], g_s2_src, c_elems);
                if (aiv == 0 && has_extra) {
                    GlobalTensor<float> g_s2_sq_src;
                    g_s2_sq_src.SetGlobalBuffer(local_s2_sq + static_cast<int64_t>(lr) * 16, 8);
                    DataCopy(ub_sq_acc[lr * 8], g_s2_sq_src, 8);
                }
            }
        }
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<bfloat16_t> g_c_x;
        g_c_x.SetGlobalBuffer(local_s1_bf16 + c_off, c_elems);
        DataCopy(stage_buf[0], g_part[c_off], c_elems);
        DataCopy(stage_buf[8192], g_c_x, c_elems);
        if (aiv == 0 && has_extra) {
            GlobalTensor<float> g_s1_sq_loc;
            g_s1_sq_loc.SetGlobalBuffer(local_s1_sq, 8);
            DataCopy(ub_sq_s1, g_s1_sq_loc, 8);
        }
        Fence<HardEvent::MTE2_V>();

        Cast(term[0], hi_half ? stage_buf[8192] : stage_buf[0],
             RoundMode::CAST_NONE, c_elems);
        Cast(term[4096], hi_half ? stage_buf[0] : stage_buf[8192],
             RoundMode::CAST_NONE, c_elems);
        PipeBarrier<PIPE_V>();
        Add(acc[my_lr * c_elems], term[0], term[4096], c_elems);
        if (aiv == 0 && has_extra) {
            Add(ub_sq_acc[my_lr * 8], hi_half ? ub_sq_s1 : ub_sq_loc,
                hi_half ? ub_sq_loc : ub_sq_s1, 8);
        }
        PipeBarrier<PIPE_V>();

        Add(acc[0], acc[0], acc[4 * c_elems], c_elems);
        Add(acc[2 * c_elems], acc[2 * c_elems], acc[6 * c_elems], c_elems);
        Add(acc[1 * c_elems], acc[1 * c_elems], acc[5 * c_elems], c_elems);
        Add(acc[3 * c_elems], acc[3 * c_elems], acc[7 * c_elems], c_elems);
        if (aiv == 0 && has_extra) {
            Add(ub_sq_acc[0], ub_sq_acc[0], ub_sq_acc[4 * 8], 8);
            Add(ub_sq_acc[2 * 8], ub_sq_acc[2 * 8], ub_sq_acc[6 * 8], 8);
            Add(ub_sq_acc[1 * 8], ub_sq_acc[1 * 8], ub_sq_acc[5 * 8], 8);
            Add(ub_sq_acc[3 * 8], ub_sq_acc[3 * 8], ub_sq_acc[7 * 8], 8);
        }
        PipeBarrier<PIPE_V>();
        Add(acc[0], acc[0], acc[2 * c_elems], c_elems);
        Add(acc[1 * c_elems], acc[1 * c_elems], acc[3 * c_elems], c_elems);
        if (aiv == 0 && has_extra) {
            Add(ub_sq_acc[0], ub_sq_acc[0], ub_sq_acc[2 * 8], 8);
            Add(ub_sq_acc[1 * 8], ub_sq_acc[1 * 8], ub_sq_acc[3 * 8], 8);
        }
        PipeBarrier<PIPE_V>();
        Add(acc[0], acc[0], acc[1 * c_elems], c_elems);
        if (aiv == 0 && has_extra) {
            Add(ub_sq_acc[0], ub_sq_acc[0], ub_sq_acc[1 * 8], 8);
        }
        PipeBarrier<PIPE_V>();

        Cast(stage_buf, acc[0], RoundMode::CAST_RINT, c_elems);
        Fence<HardEvent::V_MTE3>();
        DataCopy(g_out[c_off], stage_buf, c_elems);
        if (aiv == 0 && has_extra) {
            GlobalTensor<float> g_sq_out;
            g_sq_out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(extra_f32_out_gm), 8);
            DataCopy(g_sq_out, ub_sq_acc[0], 8);
        }
        Fence<HardEvent::MTE3_V>();
        Fence<HardEvent::MTE3_MTE2>();
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
        SyncAll<true>();
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
        for (int32_t r = 0; r < kDspineWorldSize; ++r) {
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
        Duplicate(ub_mr, -1.0e30F, kDspineVecRanks);
        Duplicate(ub_sr, 0.0F, kDspineVecRanks);
        Fence<HardEvent::V_S>();
        for (int32_t r = 0; r < kDspineWorldSize; ++r) {
            const int32_t base = r * 64 + m * cols;
            ub_mr.SetValue(r, ub_gat.GetValue(base + 0));
            ub_sr.SetValue(r, ub_gat.GetValue(base + 1));
        }
        Fence<HardEvent::S_V>();
        Adds(ub_mr, ub_mr, -max_all, kDspineVecRanks);
        PipeBarrier<PIPE_V>();
        Maxs(ub_mr, ub_mr, -104.0F, kDspineVecRanks);
        PipeBarrier<PIPE_V>();
        Exp(ub_mr, ub_mr, kDspineVecRanks);
        PipeBarrier<PIPE_V>();
        Mul(ub_mr, ub_mr, ub_sr, kDspineVecRanks);
        PipeBarrier<PIPE_V>();
        ReduceSum(ub_sum, ub_mr, ub_work, kDspineVecRanks);
        PipeBarrier<PIPE_V>();
        Ln(ub_sum, ub_sum, kDspineVecRanks);
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
    uint64_t work_gm,
    bool intra_die = false,
    float log_p_stop = -1.0e30F,
    int32_t min_stop_step = 2,
    int32_t min_stop_pool = 16)
{
    if ASCEND_IS_AIV {
        if (draft_ctrl_gm != 0ULL) {
            DcciRange(draft_ctrl_gm, 8 * sizeof(int64_t));
            GlobalTensor<int64_t> g_dc;
            g_dc.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(draft_ctrl_gm), 8);
            if (g_dc.GetValue(2) <= 0) {
                return;
            }
            const uint64_t packed = static_cast<uint64_t>(g_dc.GetValue(7));
            if (packed != 0ULL) {
                const int32_t b0 = static_cast<int32_t>(packed & 0xFULL);
                if (step == -1) {
                    b_next = b0;
                } else if (step > 0) {
                    const int32_t b_cur_d = (step <= 14)
                        ? static_cast<int32_t>((packed >> (4 * (step - 1))) & 0xFULL) : 0;
                    if (b_cur_d <= 0) {
                        return;
                    }
                    const int32_t b_nxt_d = (step < 14)
                        ? static_cast<int32_t>((packed >> (4 * step)) & 0xFULL) : 0;
                    const int32_t k_exp_raw = (b_nxt_d > b_cur_d) ? b_nxt_d : b_cur_d;
                    const int32_t k_exp_d = (k_exp_raw > 2) ? k_exp_raw : 2;
                    int32_t p_off_d = 1 + b0;
                    int32_t s_cur_d = b0;
                    for (int32_t t = 0; t < step - 1; ++t) {
                        const int32_t bt = static_cast<int32_t>((packed >> (4 * t)) & 0xFULL);
                        const int32_t bn =
                            static_cast<int32_t>((packed >> (4 * (t + 1))) & 0xFULL);
                        const int32_t ke_raw = (bn > bt) ? bn : bt;
                        const int32_t ke = (ke_raw > 2) ? ke_raw : 2;
                        p_off_d += bt * ke;
                        s_cur_d += bn;
                    }
                    b_cur = b_cur_d;
                    b_next = b_nxt_d;
                    k_expand = k_exp_d;
                    pool_off = p_off_d;
                    next_slot_cur = s_cur_d;
                }
            }
        }
        const int32_t aiv = GetBlockIdx();
        TBuf<TPosition::VECCALC> b_gat, b_mr, b_sr, b_sum, b_wk, b_i64, b_f32, b_anc, b_bf16;
        pipe.InitBuffer(b_gat, 1024 * 4);
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
                g_gat.SetGlobalBuffer(
                    reinterpret_cast<__gm__ float*>(gather_out_gm), kDspineWorldSize * 64);
                g_si.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(seed_top_i_gm), 8);
                g_slp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(seed_top_lp_gm), 8);
                DataCopy(ub_gat, g_gat, kDspineWorldSize * 64);
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
            (void)intra_die;
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
                reinterpret_cast<__gm__ int32_t*>(indices_buf_gm), b_next * 2048);
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
                DataCopy(g_idx[r * 2048 + pre_off], ub_idx, 64);
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
            g_gat.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(gather_out_gm), kDspineWorldSize * 64);
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
            DataCopy(ub_gat, g_gat, kDspineWorldSize * 64);
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
                bool do_early_stop = false;
                if (log_p_stop > -99.0F && d >= min_stop_step
                    && (pool_off + n_cand) >= min_stop_pool) {
                    float max_rlp = -1.0e30F;
                    for (int32_t j = 0; j < b_next; ++j) {
                        const float rlp_j = ub_nrlp.GetValue(j);
                        if (rlp_j > max_rlp) {
                            max_rlp = rlp_j;
                        }
                    }
                    do_early_stop = (max_rlp < log_p_stop);
                }
                if (do_early_stop) {
                    ub_i64.SetValue(8 + 4, static_cast<int64_t>(pool_off + n_cand));
                    ub_i64.SetValue(8 + 5, 1LL);
                    Fence<HardEvent::S_MTE3>();
                    DataCopy(g_ctrl, ub_i64[8], 8);
                    Fence<HardEvent::MTE3_S>();
                    DcciRange(draft_ctrl_gm, 64);
                } else {
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
        }
        Publish();

        if (b_next > 0) {
            SyncAll<true>();
            bool stop_now = false;
            if (draft_ctrl_gm != 0ULL) {
                DcciRange(draft_ctrl_gm, 64);
                GlobalTensor<int64_t> g_dc_chk;
                g_dc_chk.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(draft_ctrl_gm), 8);
                stop_now = (g_dc_chk.GetValue(5) != 0);
            }
            if (!stop_now) {
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
                g_esin.SetGlobalBuffer(
                    reinterpret_cast<__gm__ bfloat16_t*>(eshr_sin_gm), b_next * 64);
                g_ecos.SetGlobalBuffer(
                    reinterpret_cast<__gm__ bfloat16_t*>(eshr_cos_gm), b_next * 64);

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
                    reinterpret_cast<__gm__ int32_t*>(indices_buf_gm), b_next * 2048);
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
                    DataCopy(g_idx[r * 2048 + pre_off], ub_idx, 64);
                    Fence<HardEvent::MTE3_MTE2>();
                }
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
    uint64_t work_gm,
    bool intra_die = false)
{
    if ASCEND_IS_AIV {
        if (draft_ctrl_gm != 0ULL) {
            DcciRange(draft_ctrl_gm, 8 * sizeof(int64_t));
            GlobalTensor<int64_t> g_dc;
            g_dc.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(draft_ctrl_gm), 8);
            if (g_dc.GetValue(2) <= 0) {
                return;
            }
        }
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
            int32_t eff_pool_size = pool_size;
            const uint64_t packed_sched = static_cast<uint64_t>(ub_ctrl.GetValue(7));
            if (packed_sched != 0ULL) {
                int32_t p_idx = 0;
                const int32_t b0 = static_cast<int32_t>(packed_sched & 0xFULL);
                ub_pdp.SetValue(p_idx++, 0);
                for (int32_t k = 0; k < b0; ++k) {
                    ub_pdp.SetValue(p_idx++, 1);
                }
                for (int32_t t = 0; t < 14; ++t) {
                    const int32_t bt = static_cast<int32_t>((packed_sched >> (4 * t)) & 0xFULL);
                    if (bt <= 0) {
                        break;
                    }
                    const int32_t bn = (t + 1 < 14)
                        ? static_cast<int32_t>((packed_sched >> (4 * (t + 1))) & 0xFULL) : 0;
                    const int32_t ke_raw = (bn > bt) ? bn : bt;
                    const int32_t ke = (ke_raw > 2) ? ke_raw : 2;
                    const int32_t cnt = bt * ke;
                    for (int32_t k = 0; k < cnt; ++k) {
                        ub_pdp.SetValue(p_idx++, t + 2);
                    }
                }
                eff_pool_size = p_idx;
            }
            const int64_t stop_pool = ub_ctrl.GetValue(4);
            if (stop_pool > 0 && static_cast<int32_t>(stop_pool) < eff_pool_size) {
                eff_pool_size = static_cast<int32_t>(stop_pool);
            }

            for (int32_t i = eff_pool_size - 1; i >= 1; --i) {
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
            for (int32_t i = 0; i < eff_pool_size; ++i) {
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

            for (int32_t i = 0; i < eff_pool_size; ++i) {
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
                ub_mc2.SetValue(
                    j, (j < w_act) ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0));
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
                int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(g_msk[0]);
            DataCacheCleanAndInvalid<
                int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(g_msk[16]);
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
        (void)intra_die;
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

__aicore__ inline void RunStep3cPrologPackEmbedPart(
    TPipe& pipe, uint64_t active_toks_gm, uint64_t emb_w_gm, uint64_t comm_in_gm,
    int32_t b_rows, int32_t vocab_start, int32_t vocab_end, int32_t rank,
    bool skip_rot_zero = false, bool intra_die = false)
{
    if ASCEND_IS_AIV {
        TBuf<TPosition::VECCALC> b_toks, b_bf16;
        pipe.InitBuffer(b_toks, 64);
        pipe.InitBuffer(b_bf16, 768 * 2);
        auto ub_toks = b_toks.Get<int64_t>();
        auto ub_bf16 = b_bf16.Get<bfloat16_t>();

        GlobalTensor<int64_t> g_toks;
        GlobalTensor<bfloat16_t> g_emb, g_out;
        g_toks.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(active_toks_gm), 8);
        g_emb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(emb_w_gm));
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

        if (!skip_rot_zero) {
            for (int32_t task = aiv; task < b_rows * kDspineWorldSize; task += 48) {
                const int32_t r_idx = task / kDspineWorldSize;
                const int32_t s_rk = task % kDspineWorldSize;
                if (s_rk != rank) {
                    const int64_t dst_off = static_cast<int64_t>(b_rows + r_idx) * 6144
                        + static_cast<int64_t>(s_rk) * kDspineShardDim;
                    Duplicate(
                        ub_bf16.ReinterpretCast<int16_t>(), static_cast<int16_t>(0),
                        kDspineShardDim);
                    Fence<HardEvent::V_MTE3>();
                    DataCopy(g_out[dst_off], ub_bf16, kDspineShardDim);
                    Fence<HardEvent::MTE3_V>();
                }
            }
        }
        (void)intra_die;
        Publish();
    }
}

__aicore__ inline void RunStep3cPrologPackRotPart(
    TPipe& pipe, uint64_t rot_part_gm, uint64_t rot_bias_gm, uint64_t comm_in_gm,
    int32_t b_rows, int32_t rank)
{
    if ASCEND_IS_AIV {
        const int32_t aiv = GetBlockIdx();
        if (aiv < b_rows) {
            TBuf<TPosition::VECCALC> b_bf16, b_bias, b_f1, b_f2;
            pipe.InitBuffer(b_bf16, 768 * 2);
            pipe.InitBuffer(b_bias, 768 * 2);
            pipe.InitBuffer(b_f1, 768 * 4);
            pipe.InitBuffer(b_f2, 768 * 4);
            auto ub_bf16 = b_bf16.Get<bfloat16_t>();
            auto ub_bias = b_bias.Get<bfloat16_t>();
            auto ub_f1 = b_f1.Get<float>();
            auto ub_f2 = b_f2.Get<float>();

            GlobalTensor<bfloat16_t> g_rot, g_rbias, g_out;
            g_rot.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(rot_part_gm));
            g_rbias.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t*>(rot_bias_gm), kDspineShardDim);
            g_out.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t*>(comm_in_gm), 2 * b_rows * 6144);

            DataCopy(ub_bias, g_rbias, kDspineShardDim);
            for (int32_t r_idx = aiv; r_idx < b_rows; r_idx += 48) {
                const int64_t dst_off = static_cast<int64_t>(b_rows + r_idx) * 6144
                    + static_cast<int64_t>(rank) * kDspineShardDim;
                DataCopy(ub_bf16, g_rot[r_idx * kDspineShardDim], kDspineShardDim);
                Fence<HardEvent::MTE2_V>();
                Cast(ub_f1, ub_bf16, RoundMode::CAST_NONE, kDspineShardDim);
                Cast(ub_f2, ub_bias, RoundMode::CAST_NONE, kDspineShardDim);
                PipeBarrier<PIPE_V>();
                Add(ub_f1, ub_f1, ub_f2, kDspineShardDim);
                PipeBarrier<PIPE_V>();
                Cast(ub_bf16, ub_f1, RoundMode::CAST_RINT, kDspineShardDim);
                Fence<HardEvent::V_MTE3>();
                DataCopy(g_out[dst_off], ub_bf16, kDspineShardDim);
                Fence<HardEvent::MTE3_V>();
                Fence<HardEvent::MTE3_MTE2>();
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
    RunStep3cPrologPackEmbedPart(
        pipe, active_toks_gm, emb_w_gm, comm_in_gm, b_rows, vocab_start, vocab_end, rank);
    if ASCEND_IS_AIV {
        pipe.Reset();
    }
    RunStep3cPrologPackRotPart(pipe, rot_part_gm, rot_bias_gm, comm_in_gm, b_rows, rank);
}

__aicore__ inline void InlineStep3cPrologRotPush(
    TPipe& pipe, uint64_t rot_part_gm, uint64_t rot_bias_gm, uint64_t comm_out_gm,
    uint64_t comm_win_gm, uint32_t epoch, int32_t b_rows, int32_t rank)
{
    if ASCEND_IS_AIV {
        const int32_t aiv = GetBlockIdx();
        TBuf<TPosition::VECCALC> b_bf16, b_bias, b_f1, b_f2;
        pipe.InitBuffer(b_bf16, 8 * kDspineShardDim * sizeof(bfloat16_t));
        pipe.InitBuffer(b_bias, kDspineShardDim * sizeof(bfloat16_t));
        pipe.InitBuffer(b_f1, kDspineShardDim * sizeof(float));
        pipe.InitBuffer(b_f2, kDspineShardDim * sizeof(float));
        auto ub_bf16 = b_bf16.Get<bfloat16_t>();
        auto ub_bias = b_bias.Get<bfloat16_t>();
        auto ub_f1 = b_f1.Get<float>();
        auto ub_f2 = b_f2.Get<float>();

        GlobalTensor<bfloat16_t> g_rot, g_rbias, g_out;
        g_rot.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(rot_part_gm));
        g_rbias.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(rot_bias_gm), kDspineShardDim);
        g_out.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(comm_out_gm), 2 * b_rows * 6144);

        if (aiv < b_rows) {
            DataCopy(ub_bias, g_rbias, kDspineShardDim);
            DataCopy(ub_bf16, g_rot[aiv * kDspineShardDim], kDspineShardDim);
            Fence<HardEvent::MTE2_V>();
            Cast(ub_f1, ub_bf16, RoundMode::CAST_NONE, kDspineShardDim);
            Cast(ub_f2, ub_bias, RoundMode::CAST_NONE, kDspineShardDim);
            PipeBarrier<PIPE_V>();
            Add(ub_f1, ub_f1, ub_f2, kDspineShardDim);
            PipeBarrier<PIPE_V>();
            Cast(ub_bf16, ub_f1, RoundMode::CAST_RINT, kDspineShardDim);
            Fence<HardEvent::V_MTE3>();
            const int64_t dst_off = static_cast<int64_t>(b_rows + aiv) * 6144
                + static_cast<int64_t>(rank) * kDspineShardDim;
            DataCopy(g_out[dst_off], ub_bf16, kDspineShardDim);
            DataCopy(g_rot[aiv * kDspineShardDim], ub_bf16, kDspineShardDim);
            Fence<HardEvent::MTE3_S>();
        }
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
        SyncAll<true>();

        if (aiv >= 1 && aiv < kDspineWorldSize) {
            const int32_t elems = b_rows * kDspineShardDim;
            DataCopy(ub_bf16, g_rot, elems);
            Fence<HardEvent::MTE2_MTE3>();
            const int32_t peer = (rank + aiv) % kDspineWorldSize;
            __gm__ bfloat16_t* rem_slot = reinterpret_cast<__gm__ bfloat16_t*>(
                aclshmem_ptr(DspinePayloadPtr(comm_win_gm, epoch, 15), peer));
            GlobalTensor<bfloat16_t> g_rem;
            g_rem.SetGlobalBuffer(rem_slot + static_cast<int64_t>(rank) * 4096, elems);
            DataCopy(g_rem, ub_bf16, elems);
            Fence<HardEvent::MTE3_S>();
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::DDR>();
            __gm__ int32_t* rem_sig = reinterpret_cast<__gm__ int32_t*>(
                aclshmem_ptr(DspineFlagPtr(comm_win_gm, epoch, rank), peer));
            *rem_sig = static_cast<int32_t>(epoch);
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(rem_sig));
        }
    }
}

__aicore__ inline void InlineStep3cPrologRotPull(
    TPipe& pipe, uint64_t comm_out_gm, uint64_t comm_win_gm, uint32_t epoch,
    int32_t b_rows, int32_t rank)
{
    if ASCEND_IS_AIV {
        const int32_t aiv = GetBlockIdx();
        if (aiv >= 1 && aiv < kDspineWorldSize) {
            const int32_t elems = b_rows * kDspineShardDim;
            TBuf<TPosition::VECCALC> b_bf16;
            pipe.InitBuffer(b_bf16, 8 * kDspineShardDim * sizeof(bfloat16_t));
            auto ub_bf16 = b_bf16.Get<bfloat16_t>();
            const int32_t peer = (rank + aiv) % kDspineWorldSize;
            __gm__ volatile int32_t* loc_sig = DspineFlagPtr(comm_win_gm, epoch, peer);
            while (true) {
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(
                    const_cast<__gm__ int32_t*>(loc_sig)));
                if (*loc_sig == static_cast<int32_t>(epoch)) {
                    break;
                }
            }
            __gm__ bfloat16_t* loc_slot = reinterpret_cast<__gm__ bfloat16_t*>(
                DspinePayloadPtr(comm_win_gm, epoch, 15));
            GlobalTensor<bfloat16_t> g_src, g_out;
            g_src.SetGlobalBuffer(loc_slot + static_cast<int64_t>(peer) * 4096, elems);
            g_out.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t*>(comm_out_gm), 2 * b_rows * 6144);
            DataCopy(ub_bf16, g_src, elems);
            Fence<HardEvent::MTE2_MTE3>();
            for (int32_t r_idx = 0; r_idx < b_rows; ++r_idx) {
                const int64_t dst_off = static_cast<int64_t>(b_rows + r_idx) * 6144
                    + static_cast<int64_t>(peer) * kDspineShardDim;
                DataCopy(g_out[dst_off], ub_bf16[r_idx * kDspineShardDim], kDspineShardDim);
            }
            Fence<HardEvent::MTE3_S>();
        }
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
        SyncAll<true>();
    }
}

__aicore__ inline void RunStep3cPrologRotBiasAndSq(
    TPipe& pipe, uint64_t rot_part_gm, uint64_t rot_bias_gm, uint64_t sq_out_gm,
    int32_t b_rows)
{
    if ASCEND_IS_AIV {
        const int32_t aiv = GetBlockIdx();
        if (aiv < b_rows) {
            TBuf<TPosition::VECCALC> b_bf16, b_bias, b_f1, b_f2, b_work, b_sum;
            pipe.InitBuffer(b_bf16, kDspineShardDim * sizeof(bfloat16_t));
            pipe.InitBuffer(b_bias, kDspineShardDim * sizeof(bfloat16_t));
            pipe.InitBuffer(b_f1, kDspineShardDim * sizeof(float));
            pipe.InitBuffer(b_f2, kDspineShardDim * sizeof(float));
            pipe.InitBuffer(b_work, kDspineShardDim * sizeof(float));
            pipe.InitBuffer(b_sum, 64);
            auto ub_bf16 = b_bf16.Get<bfloat16_t>();
            auto ub_bias = b_bias.Get<bfloat16_t>();
            auto ub_f1 = b_f1.Get<float>();
            auto ub_f2 = b_f2.Get<float>();
            auto ub_work = b_work.Get<float>();
            auto ub_sum = b_sum.Get<float>();

            GlobalTensor<bfloat16_t> g_rot, g_rbias;
            GlobalTensor<float> g_sq;
            g_rot.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(rot_part_gm));
            g_rbias.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t*>(rot_bias_gm), kDspineShardDim);
            g_sq.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sq_out_gm), 64);

            DataCopy(ub_bias, g_rbias, kDspineShardDim);
            DataCopy(ub_bf16, g_rot[aiv * kDspineShardDim], kDspineShardDim);
            Fence<HardEvent::MTE2_V>();
            Cast(ub_f1, ub_bf16, RoundMode::CAST_NONE, kDspineShardDim);
            Cast(ub_f2, ub_bias, RoundMode::CAST_NONE, kDspineShardDim);
            PipeBarrier<PIPE_V>();
            Add(ub_f1, ub_f1, ub_f2, kDspineShardDim);
            PipeBarrier<PIPE_V>();
            Cast(ub_bf16, ub_f1, RoundMode::CAST_RINT, kDspineShardDim);
            Fence<HardEvent::V_MTE3>();
            DataCopy(g_rot[aiv * kDspineShardDim], ub_bf16, kDspineShardDim);
            PipeBarrier<PIPE_V>();
            Cast(ub_f1, ub_bf16, RoundMode::CAST_NONE, kDspineShardDim);
            PipeBarrier<PIPE_V>();
            Mul(ub_f1, ub_f1, ub_f1, kDspineShardDim);
            PipeBarrier<PIPE_V>();
            ReduceSum(ub_sum, ub_f1, ub_work, kDspineShardDim);
            Fence<HardEvent::V_MTE3>();
            DataCopy(g_sq[aiv * 16], ub_sum, 8);
            Fence<HardEvent::MTE3_S>();
        }
        Publish();
        SyncAll<true>();
    }
}

__aicore__ inline void RunStep3cDualRmsNormSlice(
    TPipe& pipe, uint64_t comm_out_gm, uint64_t enorm_w_gm, uint64_t hnorm_w_gm,
    uint64_t eh_in_gm, int32_t b_rows, int32_t rank, float eps, bool intra_die = false,
    uint64_t h_sq_sum_gm = 0ULL, uint64_t h_local_gm = 0ULL)
{
    if ASCEND_IS_AIV {
        const int32_t t = GetBlockIdx();
        if (t < 2 * b_rows) {
            const bool is_rot = (t >= b_rows);
            const int32_t r_idx = is_rot ? (t - b_rows) : t;
            if (is_rot && h_sq_sum_gm != 0ULL && h_local_gm != 0ULL) {
                TBuf<TPosition::VECCALC> b_x, b_g, b_xf, b_gf, b_sq, b_sum;
                pipe.InitBuffer(b_x, kDspineShardDim * sizeof(bfloat16_t));
                pipe.InitBuffer(b_g, kDspineShardDim * sizeof(bfloat16_t));
                pipe.InitBuffer(b_xf, kDspineShardDim * sizeof(float));
                pipe.InitBuffer(b_gf, kDspineShardDim * sizeof(float));
                pipe.InitBuffer(b_sq, kDspineShardDim * sizeof(float));
                pipe.InitBuffer(b_sum, 64);
                auto ub_x = b_x.Get<bfloat16_t>();
                auto ub_g = b_g.Get<bfloat16_t>();
                auto ub_xf = b_xf.Get<float>();
                auto ub_gf = b_gf.Get<float>();
                auto ub_sq = b_sq.Get<float>();
                auto ub_sum = b_sum.Get<float>();

                GlobalTensor<bfloat16_t> g_h_loc, g_w, g_out;
                GlobalTensor<float> g_sq_sum;
                g_h_loc.SetGlobalBuffer(
                    reinterpret_cast<__gm__ bfloat16_t*>(h_local_gm), b_rows * kDspineShardDim);
                g_w.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(hnorm_w_gm), 6144);
                g_out.SetGlobalBuffer(
                    reinterpret_cast<__gm__ bfloat16_t*>(eh_in_gm), b_rows * kDspineEhIn);
                g_sq_sum.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(h_sq_sum_gm), 8);

                DataCopy(ub_x, g_h_loc[r_idx * kDspineShardDim], kDspineShardDim);
                DataCopy(ub_g, g_w[rank * kDspineShardDim], kDspineShardDim);
                DataCopy(ub_sum, g_sq_sum, 8);
                Fence<HardEvent::MTE2_S>();
                Fence<HardEvent::MTE2_V>();

                const float mean_eps = ub_sum.GetValue(r_idx) * (1.0F / 6144.0F) + eps;
                Fence<HardEvent::S_V>();
                Duplicate(ub_sum, mean_eps, 8);
                PipeBarrier<PIPE_V>();
                Rsqrt(ub_sum, ub_sum, 8);
                Fence<HardEvent::V_S>();
                const float rstd = ub_sum.GetValue(0);
                Fence<HardEvent::S_V>();

                Cast(ub_xf, ub_x, RoundMode::CAST_NONE, kDspineShardDim);
                Cast(ub_gf, ub_g, RoundMode::CAST_NONE, kDspineShardDim);
                PipeBarrier<PIPE_V>();
                Muls(ub_sq, ub_xf, rstd, kDspineShardDim);
                PipeBarrier<PIPE_V>();
                Mul(ub_sq, ub_sq, ub_gf, kDspineShardDim);
                PipeBarrier<PIPE_V>();
                Cast(ub_g, ub_sq, RoundMode::CAST_RINT, kDspineShardDim);
                Fence<HardEvent::V_MTE3>();
                DataCopy(g_out[r_idx * kDspineEhIn + kDspineShardDim], ub_g, kDspineShardDim);
            } else {
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

                GlobalTensor<bfloat16_t> g_in, g_w, g_out;
                g_in.SetGlobalBuffer(
                    reinterpret_cast<__gm__ bfloat16_t*>(comm_out_gm), 2 * b_rows * 6144);
                g_w.SetGlobalBuffer(
                    reinterpret_cast<__gm__ bfloat16_t*>(is_rot ? hnorm_w_gm : enorm_w_gm), 6144);
                g_out.SetGlobalBuffer(
                    reinterpret_cast<__gm__ bfloat16_t*>(eh_in_gm), b_rows * kDspineEhIn);

                DataCopy(ub_x, g_in[t * 6144], 6144);
                DataCopy(ub_g, g_w[rank * kDspineShardDim], kDspineShardDim);
                Fence<HardEvent::MTE2_V>();

                Cast(ub_xf, ub_x, RoundMode::CAST_NONE, 6144);
                Cast(ub_gf, ub_g, RoundMode::CAST_NONE, kDspineShardDim);
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

                Muls(ub_sq, ub_xf[rank * kDspineShardDim], rstd, kDspineShardDim);
                PipeBarrier<PIPE_V>();
                Mul(ub_sq, ub_sq, ub_gf, kDspineShardDim);
                PipeBarrier<PIPE_V>();
                Cast(ub_g, ub_sq, RoundMode::CAST_RINT, kDspineShardDim);
                Fence<HardEvent::V_MTE3>();
                DataCopy(
                    g_out[r_idx * kDspineEhIn + (is_rot ? kDspineShardDim : 0)],
                    ub_g, kDspineShardDim);
            }
        }
        (void)intra_die;
        Publish();
    }
}

__aicore__ inline void RunDraftHeadTopk(
    TPipe& pipe,
    uint64_t q_len_gm,
    uint64_t res_a_gm,
    uint64_t res_b_gm,
    uint64_t gamma_gm,
    uint64_t w_q8_nz_gm,
    uint64_t w_s8_gm,
    uint64_t recs_out_gm,
    uint64_t act_q_ws_gm,
    uint64_t act_s_ws_gm,
    uint64_t acc_i32_ws_gm,
    uint64_t work_gm,
    uint64_t gather_in_gm,
    uint64_t gather_out_gm,
    uint64_t comm_win_gm,
    int32_t m_rows,
    int32_t vocab_valid,
    int32_t vocab_start,
    int32_t k,
    int32_t rank,
    int32_t ranks,
    float eps,
    uint32_t explicit_epoch = 0U,
    bool skip_stage1 = false)
{
    int32_t eff = m_rows;
    if (q_len_gm != 0ULL) {
        DcciRange(q_len_gm, 64);
        const int32_t q = *reinterpret_cast<__gm__ const int32_t*>(q_len_gm);
        if (q <= 0) {
            return;
        }
        if (q < eff) {
            eff = q;
        }
    }
    constexpr int32_t kHidden = 6144;
    if (!skip_stage1) {
        if ASCEND_IS_AIV {
            const int32_t aiv = GetBlockIdx();
            if (aiv < eff) {
                TBuf<TPosition::VECCALC> b_f32, b_bf, b_i8;
                pipe.InitBuffer(b_f32, 3 * kHidden * 4 + 256 * 4);
                pipe.InitBuffer(b_bf, 3 * kHidden * 2);
                pipe.InitBuffer(b_i8, kHidden + 32);

                auto ub_f32 = b_f32.Get<float>();
                auto ub_x = ub_f32[0];
                auto ub_sq = ub_f32[kHidden];
                auto ub_tmp = ub_f32[2 * kHidden];
                auto ub_red = ub_f32[3 * kHidden];
                auto ub_sc = ub_f32[3 * kHidden + 128];

                auto ub_bf = b_bf.Get<bfloat16_t>();
                auto ub_x1 = ub_bf[0];
                auto ub_x2 = ub_bf[kHidden];
                auto ub_g = ub_bf[2 * kHidden];
                auto ub_q8 = b_i8.Get<int8_t>();

                GlobalTensor<bfloat16_t> g_a, g_b, g_g, g_recs;
                GlobalTensor<int8_t> g_q8;
                GlobalTensor<float> g_sc;
                g_a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(res_a_gm));
                g_b.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(res_b_gm));
                g_g.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gamma_gm), kHidden);
                g_recs.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(recs_out_gm));
                g_q8.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(act_q_ws_gm));
                g_sc.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(act_s_ws_gm));

                const int64_t row_off = static_cast<int64_t>(aiv) * kHidden;
                DataCopy(ub_x1, g_a[row_off], kHidden);
                if (res_b_gm != 0ULL) {
                    DataCopy(ub_x2, g_b[row_off], kHidden);
                }
                DataCopy(ub_g, g_g, kHidden);
                Fence<HardEvent::MTE2_V>();

                Cast(ub_x, ub_x1, RoundMode::CAST_NONE, kHidden);
                if (res_b_gm != 0ULL) {
                    Cast(ub_tmp, ub_x2, RoundMode::CAST_NONE, kHidden);
                    PipeBarrier<PIPE_V>();
                    Add(ub_x, ub_x, ub_tmp, kHidden);
                    PipeBarrier<PIPE_V>();
                    Cast(ub_x1, ub_x, RoundMode::CAST_RINT, kHidden);
                    PipeBarrier<PIPE_V>();
                    Cast(ub_x, ub_x1, RoundMode::CAST_NONE, kHidden);
                }
                PipeBarrier<PIPE_V>();
                Mul(ub_sq, ub_x, ub_x, kHidden);
                PipeBarrier<PIPE_V>();
                Add(ub_sq, ub_sq, ub_sq[4096], 2048);
                PipeBarrier<PIPE_V>();
                for (int32_t s = 2048; s >= 64; s >>= 1) {
                    Add(ub_sq, ub_sq, ub_sq[s], s);
                    PipeBarrier<PIPE_V>();
                }
                AscendCUtils::SetMask<float>(64);
                WholeReduceSum<float, false>(ub_sq, ub_sq, MASK_PLACEHOLDER, 1, 0, 1, 0);
                Fence<HardEvent::V_S>();
                const float rstd = 1.0F / sqrt(
                    ub_sq.GetValue(0) * (1.0F / static_cast<float>(kHidden)) + eps);
                Fence<HardEvent::S_V>();
                Muls(ub_x, ub_x, rstd, kHidden);
                Cast(ub_tmp, ub_g, RoundMode::CAST_NONE, kHidden);
                PipeBarrier<PIPE_V>();
                Mul(ub_x, ub_x, ub_tmp, kHidden);
                PipeBarrier<PIPE_V>();
                Cast(ub_x1, ub_x, RoundMode::CAST_RINT, kHidden);
                Fence<HardEvent::V_MTE3>();
                if (recs_out_gm != 0ULL) {
                    DataCopy(g_recs[row_off], ub_x1, kHidden);
                }
                PipeBarrier<PIPE_V>();
                Cast(ub_x, ub_x1, RoundMode::CAST_NONE, kHidden);
                PipeBarrier<PIPE_V>();
                Abs(ub_tmp, ub_x, kHidden);
                PipeBarrier<PIPE_V>();
                ReduceMax(ub_red, ub_tmp, ub_sq, kHidden);
                Fence<HardEvent::V_S>();
                const float max_v = ub_red.GetValue(0);
                const float sc = (max_v > 0.0F) ? (max_v / 127.0F) : 1.0F;
                ub_sc.SetValue(0, sc);
                Fence<HardEvent::S_V>();
                Duplicate(ub_tmp, sc, 64);
                Duplicate(ub_sq, 1.0F, 64);
                PipeBarrier<PIPE_V>();
                Div(ub_sq, ub_sq, ub_tmp, 64);
                PipeBarrier<PIPE_V>();
                Mul(ub_tmp, ub_x, ub_sq, 64, kHidden / 64, BinaryRepeatParams{1, 1, 0, 8, 8, 0});
                PipeBarrier<PIPE_V>();
                auto ub_i32 = ub_sq.ReinterpretCast<int32_t>();
                Cast(ub_i32, ub_tmp, RoundMode::CAST_RINT, kHidden);
                PipeBarrier<PIPE_V>();
                SetDeqScale(static_cast<half>(1.0F));
                PipeBarrier<PIPE_V>();
                auto ub_fp16 = ub_x2.ReinterpretCast<half>();
                Cast(ub_fp16, ub_i32, RoundMode::CAST_ROUND, kHidden);
                PipeBarrier<PIPE_V>();
                Cast(ub_q8, ub_fp16, RoundMode::CAST_TRUNC, kHidden);
                Fence<HardEvent::V_MTE3>();
                Fence<HardEvent::S_MTE3>();
                DataCopy(g_q8[row_off], ub_q8, kHidden);
                DataCopyExtParams copy_sc{1, 4, 0, 0, 0};
                DataCopyPad(g_sc[aiv], ub_sc, copy_sc);
            }
            Publish();
        }

        SyncAll<false>();
        pipe.Reset();
    }

    if ASCEND_IS_AIC {
        const int32_t cid = GetBlockIdx();
        const int32_t n32_total = (vocab_valid + 31) / 32;
        const int32_t acc_stride = n32_total * 32;
        const int32_t base_b = n32_total / 24;
        const int32_t rem_b = n32_total % 24;
        const int32_t my_b = base_b + (cid < rem_b ? 1 : 0);
        if (my_b > 0) {
            const int32_t b_start = cid * base_b + (cid < rem_b ? cid : rem_b);
            constexpr int32_t kStep = 512;
            constexpr int32_t kKSteps = kHidden / kStep;
            constexpr int32_t kN32Sub = 8;

            TBuf<TPosition::A1> a1_buf;
            TBuf<TPosition::B1> b1_buf0, b1_buf1;
            TBuf<TPosition::A2> l0a_buf0, l0a_buf1;
            TBuf<TPosition::B2> l0b_buf0, l0b_buf1;
            TBuf<TPosition::CO1> l0c_buf;
            pipe.InitBuffer(a1_buf, 16 * kHidden);
            pipe.InitBuffer(b1_buf0, kN32Sub * 32 * 512);
            pipe.InitBuffer(b1_buf1, kN32Sub * 32 * 512);
            pipe.InitBuffer(l0a_buf0, 16 * kStep);
            pipe.InitBuffer(l0a_buf1, 16 * kStep);
            pipe.InitBuffer(l0b_buf0, kN32Sub * 32 * 128);
            pipe.InitBuffer(l0b_buf1, kN32Sub * 32 * 128);
            pipe.InitBuffer(l0c_buf, my_b * 512 * sizeof(int32_t));

            auto a1 = a1_buf.Get<int8_t>();
            LocalTensor<int8_t> b1[2] = {b1_buf0.Get<int8_t>(), b1_buf1.Get<int8_t>()};
            LocalTensor<int8_t> l0a[2] = {l0a_buf0.Get<int8_t>(), l0a_buf1.Get<int8_t>()};
            LocalTensor<int8_t> l0b[2] = {l0b_buf0.Get<int8_t>(), l0b_buf1.Get<int8_t>()};
            auto l0c = l0c_buf.Get<int32_t>();

            GlobalTensor<int8_t> a_gm, b_gm;
            GlobalTensor<int32_t> c_gm;
            a_gm.SetGlobalBuffer(
                reinterpret_cast<__gm__ int8_t*>(act_q_ws_gm),
                static_cast<uint64_t>(eff) * kHidden);
            b_gm.SetGlobalBuffer(
                reinterpret_cast<__gm__ int8_t*>(w_q8_nz_gm),
                static_cast<uint64_t>(n32_total) * 384ULL * 512ULL);
            c_gm.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t*>(acc_i32_ws_gm),
                static_cast<uint64_t>(eff) * acc_stride);

            Nd2NzParams copy_a{};
            copy_a.ndNum = 1;
            copy_a.nValue = eff;
            copy_a.dValue = kHidden;
            copy_a.srcDValue = kHidden;
            copy_a.dstNzC0Stride = 16;
            copy_a.dstNzNStride = 1;
            DataCopy(a1, a_gm, copy_a);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);

            SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID5);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID6);

            const int32_t n_chunks = (my_b + kN32Sub - 1) / kN32Sub;
            const int32_t total_b_tiles = kKSteps * n_chunks;
            DataCopy(
                b1[0],
                b_gm[static_cast<uint64_t>(b_start) * 384ULL * 512ULL],
                DataCopyParams{
                    static_cast<uint16_t>((my_b < kN32Sub) ? my_b : kN32Sub), 512, 5632, 0});
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

            int32_t b_iter = 0;
            int32_t lb_iter = 0;
            for (int32_t kc = 0; kc < kKSteps; ++kc) {
                const int32_t ai = kc & 1;
                const event_t a_ev = ai ? EVENT_ID6 : EVENT_ID5;
                WaitFlag<HardEvent::M_MTE1>(a_ev);
                LoadData2DParams load_a{};
                load_a.startIndex = kc * 16;
                load_a.repeatTimes = 16;
                load_a.srcStride = 1;
                load_a.ifTranspose = false;
                LoadData(l0a[ai], a1, load_a);
                SetFlag<HardEvent::MTE1_M>(a_ev);

                for (int32_t nb_idx = 0; nb_idx < n_chunks; ++nb_idx) {
                    const int32_t nb = nb_idx * kN32Sub;
                    const int32_t n32_cnt = (my_b - nb < kN32Sub) ? (my_b - nb) : kN32Sub;
                    const int32_t bi = b_iter & 1;
                    const event_t b_ev = bi ? EVENT_ID1 : EVENT_ID0;
                    if (b_iter + 1 < total_b_tiles) {
                        const int32_t nxt_kc = (b_iter + 1) / n_chunks;
                        const int32_t nxt_nb = ((b_iter + 1) % n_chunks) * kN32Sub;
                        const int32_t nxt_cnt =
                            (my_b - nxt_nb < kN32Sub) ? (my_b - nxt_nb) : kN32Sub;
                        const int32_t nbi = (b_iter + 1) & 1;
                        const event_t nb_ev = nbi ? EVENT_ID1 : EVENT_ID0;
                        if (b_iter + 1 >= 2) {
                            WaitFlag<HardEvent::MTE1_MTE2>(nb_ev);
                        }
                        const uint64_t nxt_b_off =
                            static_cast<uint64_t>(b_start + nxt_nb) * 384ULL * 512ULL
                            + static_cast<uint64_t>(nxt_kc) * 32ULL * 512ULL;
                        DataCopy(
                            b1[nbi],
                            b_gm[nxt_b_off],
                            DataCopyParams{static_cast<uint16_t>(nxt_cnt), 512, 5632, 0});
                        SetFlag<HardEvent::MTE2_MTE1>(nb_ev);
                    }
                    WaitFlag<HardEvent::MTE2_MTE1>(b_ev);

                    LoadData2dTransposeParams trans(
                        0, static_cast<uint8_t>(n32_cnt), 16, 1, 0, inc);
                    for (int32_t sub = 0; sub < 4; ++sub) {
                        const int32_t lbi = lb_iter & 1;
                        const event_t lb_ev = lbi ? EVENT_ID4 : EVENT_ID3;
                        WaitFlag<HardEvent::M_MTE1>(lb_ev);
                        const uint32_t k_sub_off = static_cast<uint32_t>(sub) * 4096U;
                        for (int32_t i = 0; i < 4; ++i) {
                            LoadDataWithTranspose(
                                l0b[lbi][i * n32_cnt * 1024],
                                b1[bi][k_sub_off + static_cast<uint32_t>(i) * 1024U],
                                trans);
                        }
                        if (sub == 3 && b_iter + 2 < total_b_tiles) {
                            SetFlag<HardEvent::MTE1_MTE2>(b_ev);
                        }
                        SetFlag<HardEvent::MTE1_M>(lb_ev);
                        WaitFlag<HardEvent::MTE1_M>(lb_ev);
                        if (nb_idx == 0 && sub == 0) {
                            WaitFlag<HardEvent::MTE1_M>(a_ev);
                        }
                        MmadParams mad{};
                        mad.m = 16;
                        mad.n = n32_cnt * 32;
                        mad.k = 128;
                        mad.cmatrixSource = false;
                        mad.cmatrixInitVal = (kc == 0 && sub == 0);
                        Mmad(l0c[nb * 512], l0a[ai][sub * 2048], l0b[lbi], mad);
                        if (b_iter == total_b_tiles - 1 && sub == 3) {
                            PipeBarrier<PIPE_M>();
                        }
                        SetFlag<HardEvent::M_MTE1>(lb_ev);
                        if (nb_idx == n_chunks - 1 && sub == 3) {
                            SetFlag<HardEvent::M_MTE1>(a_ev);
                        }
                        ++lb_iter;
                    }
                    ++b_iter;
                }
            }
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID5);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID6);

            SetFlag<HardEvent::M_FIX>(EVENT_ID0);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
            for (int32_t nb = 0; nb < my_b; nb += kN32Sub) {
                const int32_t n32_cnt = (my_b - nb < kN32Sub) ? (my_b - nb) : kN32Sub;
                FixpipeParamsV220 fix{};
                fix.nSize = n32_cnt * 32;
                fix.mSize = eff;
                fix.srcStride = 16;
                fix.dstStride = acc_stride;
                fix.ndNum = 1;
                fix.quantPre = QuantMode_t::NoQuant;
                Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(
                    c_gm[(b_start + nb) * 32], l0c[nb * 512], fix);
            }
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        }
        Publish();
    }

    SyncAll<false>();
    pipe.Reset();
    if (work_gm == 0ULL) {
        return;
    }

    if ASCEND_IS_AIV {
        const int32_t aiv = GetBlockIdx();
        const int32_t aivs_per_row = 48 / eff;
        const int32_t r_idx = aiv / aivs_per_row;
        const int32_t sub = aiv % aivs_per_row;
        const int32_t chunk =
            (((vocab_valid + aivs_per_row - 1) / aivs_per_row) + 63) & ~63;
        const int32_t col_begin = sub * chunk;
        const int32_t col_end =
            (col_begin + chunk < vocab_valid) ? (col_begin + chunk) : vocab_valid;
        const int32_t len = (col_end > col_begin) ? (col_end - col_begin) : 0;

        TBuf<TPosition::VECCALC> b_i32, b_ws8, b_x, b_bf16, b_blk, b_tmp, b_work, b_sum, b_out;
        pipe.InitBuffer(b_i32, 2048 * 4);
        pipe.InitBuffer(b_ws8, 2048 * 4);
        pipe.InitBuffer(b_x, 2048 * 4);
        pipe.InitBuffer(b_bf16, 2048 * 2);
        pipe.InitBuffer(b_blk, 512);
        pipe.InitBuffer(b_tmp, 2048 * 4);
        pipe.InitBuffer(b_work, 2048 * 4);
        pipe.InitBuffer(b_sum, 64);
        pipe.InitBuffer(b_out, 512);
        auto ub_i32 = b_i32.Get<int32_t>();
        auto ub_ws8 = b_ws8.Get<float>();
        auto ub_x = b_x.Get<float>();
        auto ub_bf16 = b_bf16.Get<bfloat16_t>();
        auto ub_blk = b_blk.Get<float>();
        auto ub_tmp = b_tmp.Get<float>();
        auto ub_work = b_work.Get<float>();
        auto ub_sum = b_sum.Get<float>();
        auto ub_out = b_out.Get<float>();

        const int32_t acc_stride = ((vocab_valid + 31) / 32) * 32;
        GlobalTensor<int32_t> g_acc;
        GlobalTensor<float> g_ws8, g_act_s, g_work, g_gather;
        g_acc.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(acc_i32_ws_gm),
            static_cast<uint64_t>(eff) * acc_stride);
        g_ws8.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w_s8_gm), vocab_valid);
        g_act_s.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(act_s_ws_gm), eff);
        g_work.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(work_gm), 48 * 16);
        g_gather.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gather_in_gm), ranks * 64);

        DcciRange(act_s_ws_gm, eff * 4);
        DataCopyPad(
            ub_sum, g_act_s,
            DataCopyExtParams{1, static_cast<uint32_t>(eff * 4), 0, 0, 0},
            DataCopyPadExtParams<float>{false, 0, 0, 0});
        Fence<HardEvent::MTE2_S>();

        if (comm_win_gm == 0ULL && aiv < ranks && aiv != rank) {
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

        if (r_idx < eff && len > 0) {
            const float act_s = ub_sum.GetValue(r_idx);
            DataCopyPad(
                ub_i32, g_acc[static_cast<int64_t>(r_idx) * acc_stride + col_begin],
                DataCopyExtParams{1, static_cast<uint32_t>(len * 4), 0, 0, 0},
                DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            DataCopyPad(
                ub_ws8, g_ws8[col_begin],
                DataCopyExtParams{1, static_cast<uint32_t>(len * 4), 0, 0, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
            Fence<HardEvent::MTE2_V>();
            const int32_t len_a16 = (len + 15) & ~15;
            Cast(ub_tmp, ub_i32, RoundMode::CAST_RINT, len_a16);
            PipeBarrier<PIPE_V>();
            Mul(ub_tmp, ub_tmp, ub_ws8, len_a16);
            PipeBarrier<PIPE_V>();
            Muls(ub_tmp, ub_tmp, act_s, len_a16);
            PipeBarrier<PIPE_V>();
            Cast(ub_bf16, ub_tmp, RoundMode::CAST_RINT, len_a16);
            PipeBarrier<PIPE_V>();
            Cast(ub_x, ub_bf16, RoundMode::CAST_NONE, len_a16);
            Fence<HardEvent::V_S>();
            for (int32_t pad_j = len; pad_j < len_a16; ++pad_j) {
                ub_x.SetValue(pad_j, -1.0e30F);
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
        uint32_t epoch = explicit_epoch;
        if (comm_win_gm != 0ULL && explicit_epoch == 0U && aiv < ranks) {
            epoch = DspineReadSeq(comm_win_gm) + 1U;
        }
        Publish();
        SyncAll<true>();

        if (aiv < ranks) {
            DataCopy(ub_work, g_work, 48 * 16);
            Fence<HardEvent::MTE2_S>();
            Fence<HardEvent::S_V>();
            Duplicate(ub_out, 0.0F, 64);
            Fence<HardEvent::V_S>();

            const int32_t cols = 2 + 2 * k;
            for (int32_t m = 0; m < eff; ++m) {
                float rv0 = -1.0e30F, rv1 = -1.0e30F, rv2 = -1.0e30F, rv3 = -1.0e30F;
                int64_t ri0 = 0, ri1 = 0, ri2 = 0, ri3 = 0;
                for (int32_t s = 0; s < aivs_per_row; ++s) {
                    const int32_t base = (m * aivs_per_row + s) * 16;
                    for (int32_t j = 0; j < k; ++j) {
                        InsertTop4(
                            ub_work.GetValue(base + 2 + j),
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
            if (comm_win_gm == 0ULL) {
                if (aiv == 0) {
                    DataCopy(g_gather[rank * 64], ub_out, 64);
                }
                Publish();
                pipe.Reset();
                return;
            }
            if (aiv == 0) {
                GlobalTensor<float> g_gout;
                g_gout.SetGlobalBuffer(
                    reinterpret_cast<__gm__ float*>(gather_out_gm), ranks * 64);
                DataCopy(g_gout[rank * 64], ub_out, 64);
                Fence<HardEvent::MTE3_V>();
                Fence<HardEvent::MTE3_S>();
                if (explicit_epoch == 0U) {
                    DspineWriteSeq(comm_win_gm, epoch);
                }
                Publish();
            } else {
                const int32_t peer = (rank + aiv) % ranks;
                GlobalTensor<float> g_rem;
                g_rem.SetGlobalBuffer(
                    reinterpret_cast<__gm__ float*>(
                        aclshmem_ptr(DspinePayloadPtr(comm_win_gm, epoch, rank), peer)),
                    64);
                DataCopy(g_rem, ub_out, 64);
                Fence<HardEvent::MTE3_S>();
                Publish();
                __gm__ int32_t* remote_sig = reinterpret_cast<__gm__ int32_t*>(
                    aclshmem_ptr(DspineFlagPtr(comm_win_gm, epoch, rank), peer));
                *remote_sig = static_cast<int32_t>(epoch);
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(remote_sig));

                __gm__ volatile int32_t* sig = DspineFlagPtr(comm_win_gm, epoch, peer);
                while (true) {
                    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(
                        const_cast<__gm__ int32_t*>(sig)));
                    if (*sig == static_cast<int32_t>(epoch)) {
                        break;
                    }
                }
                GlobalTensor<float> g_pe, g_gout;
                g_pe.SetGlobalBuffer(
                    reinterpret_cast<__gm__ float*>(
                        DspinePayloadPtr(comm_win_gm, epoch, peer)),
                    64);
                g_gout.SetGlobalBuffer(
                    reinterpret_cast<__gm__ float*>(gather_out_gm), ranks * 64);
                DataCopy(ub_out[64], g_pe, 64);
                Fence<HardEvent::MTE2_MTE3>();
                DataCopy(g_gout[peer * 64], ub_out[64], 64);
                Fence<HardEvent::MTE3_S>();
                Publish();
            }
        }
        if (comm_win_gm == 0ULL) {
            Publish();
            pipe.Reset();
            return;
        }
        SyncAll<true>();
        pipe.Reset();
    }
}
}
"""

_DSPINE_BODY_CPP = """using namespace dspine_shared;


struct RoundState {
    int32_t q_len;
    int32_t done;
    int32_t eos_hit;
    int32_t round_idx;
    int32_t main_pos;
    int32_t draft_pos;
    int32_t horizon_level;
    int32_t max_pos;
    int32_t last_w_act;
    int32_t drain_q_len;
    int32_t hi_streak;
    int32_t lo_streak;
    int32_t eos_token_id;
    int32_t pad0;
    int64_t prev_accepted_total;
    uint64_t ema_acc;
    int64_t pending_token;
    int64_t pad1[6];
};

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
        g_rot.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(rot_part_gm), m_max * kDspineShardDim);
        g_out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(comm_in_gm), 4 * m_max * 6144);

        DcciRange(ctrl_gm, 48 * 8);
        DataCopy(ub_ctrl, g_ctrl, 48);
        Fence<HardEvent::MTE2_S>();

        const int32_t count = static_cast<int32_t>(ub_ctrl.GetValue(2));
        if (count <= 0) {
            return;
        }
        DataCopy(ub_ttoks, g_ttoks, 32);
        Fence<HardEvent::MTE2_S>();
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

        for (int32_t task = aiv; task < m_max * kDspineWorldSize; task += 48) {
            const int32_t r_idx = task / kDspineWorldSize;
            const int32_t s_rk = task % kDspineWorldSize;
            const int64_t dst_off = static_cast<int64_t>(2 * m_max + r_idx) * 6144
                + static_cast<int64_t>(s_rk) * kDspineShardDim;
            if (r_idx < count && s_rk == rank) {
                DataCopy(ub_bf16, g_rot[r_idx * kDspineShardDim], kDspineShardDim);
                Fence<HardEvent::MTE2_MTE3>();
            } else {
                Duplicate(
                    ub_bf16.ReinterpretCast<int16_t>(), static_cast<int16_t>(0), kDspineShardDim);
                Fence<HardEvent::V_MTE3>();
            }
            DataCopy(g_out[dst_off], ub_bf16, kDspineShardDim);
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::MTE3_MTE2>();
        }
        Publish();
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

        DcciRange(ctrl_gm, 48 * sizeof(int64_t));
        GlobalTensor<int64_t> g_ctrl;
        g_ctrl.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ctrl_gm), 48);
        DataCopy(ub_ctrl, g_ctrl, 48);
        Fence<HardEvent::MTE2_S>();

        const int64_t main_start = ub_ctrl.GetValue(0);
        int32_t count = static_cast<int32_t>(ub_ctrl.GetValue(1));
        if (count <= 0) {
            return;
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

        for (int32_t task = aiv; task < kv_rows * 12; task += 48) {
            const int32_t r = task / 12;
            const int64_t col = static_cast<int64_t>(task % 12) * 512;
            DataCopy(ub_bf16, g_hs_all[r * 6144 + col], 512);
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

        if (aiv >= 24 && aiv < 24 + kv_rows) {
            const int32_t r = aiv - 24;
            const int64_t p = main_start + r;
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
            Fence<HardEvent::MTE3_S>();
            DcciRange(kv_slot_gm, kv_m_cap * sizeof(int64_t));
            DcciRange(seed_slot_gm, sizeof(int64_t));
            DcciRange(kv_qlen_gm, sizeof(int32_t));
            DcciRange(kv_kvlen_gm, sizeof(int32_t));
            DcciRange(seed_qlen_gm, sizeof(int32_t));
            DcciRange(seed_kvlen_gm, sizeof(int32_t));
        }
        Publish();
    }
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
        g_gather.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(gather_in_gm), kDspineWorldSize * 64);

        if (aiv < kDspineWorldSize && aiv != rank) {
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

__aicore__ inline uint64_t F64FromU32(uint32_t v)
{
    if (v == 0U) {
        return 0ULL;
    }
    int32_t msb = 0;
    while ((v >> msb) > 1U) {
        ++msb;
    }
    const uint64_t exp = 1023ULL + static_cast<uint64_t>(msb);
    const uint64_t mant = (static_cast<uint64_t>(v) << (52 - msb)) & 0xFFFFFFFFFFFFFULL;
    return (exp << 52) | mant;
}

__aicore__ inline uint64_t F64Mul(uint64_t a, uint64_t b)
{
    if (a == 0ULL || b == 0ULL) {
        return 0ULL;
    }
    const int32_t ea = static_cast<int32_t>((a >> 52) & 0x7FFULL);
    const int32_t eb = static_cast<int32_t>((b >> 52) & 0x7FFULL);
    const uint64_t ma = (1ULL << 52) | (a & 0xFFFFFFFFFFFFFULL);
    const uint64_t mb = (1ULL << 52) | (b & 0xFFFFFFFFFFFFFULL);
    const uint64_t ma_l = ma & 0xFFFFFFFFULL;
    const uint64_t ma_h = ma >> 32;
    const uint64_t mb_l = mb & 0xFFFFFFFFULL;
    const uint64_t mb_h = mb >> 32;
    const uint64_t p00 = ma_l * mb_l;
    const uint64_t p01 = ma_l * mb_h;
    const uint64_t p10 = ma_h * mb_l;
    const uint64_t p11 = ma_h * mb_h;
    const uint64_t mid = p01 + (p00 >> 32);
    const uint64_t mid2 = (mid & 0xFFFFFFFFULL) + p10;
    const uint64_t lo64 = (p00 & 0xFFFFFFFFULL) | (mid2 << 32);
    const uint64_t hi64 = p11 + (mid >> 32) + (mid2 >> 32);
    int32_t er = ea + eb - 1023;
    uint64_t sig = 0;
    uint64_t guard = 0;
    bool sticky = false;
    if ((hi64 >> 41) != 0ULL) {
        er += 1;
        sig = (hi64 << 11) | (lo64 >> 53);
        guard = (lo64 >> 52) & 1ULL;
        sticky = (lo64 & ((1ULL << 52) - 1ULL)) != 0ULL;
    } else {
        sig = (hi64 << 12) | (lo64 >> 52);
        guard = (lo64 >> 51) & 1ULL;
        sticky = (lo64 & ((1ULL << 51) - 1ULL)) != 0ULL;
    }
    if (guard != 0ULL && (sticky || ((sig & 1ULL) != 0ULL))) {
        sig += 1ULL;
        if (sig >= (1ULL << 53)) {
            sig >>= 1;
            er += 1;
        }
    }
    return (static_cast<uint64_t>(er) << 52) | (sig & 0xFFFFFFFFFFFFFULL);
}

__aicore__ inline uint64_t F64Add(uint64_t a, uint64_t b)
{
    if (a == 0ULL) {
        return b;
    }
    if (b == 0ULL) {
        return a;
    }
    if (a < b) {
        const uint64_t t = a;
        a = b;
        b = t;
    }
    const int32_t ea = static_cast<int32_t>((a >> 52) & 0x7FFULL);
    const int32_t eb = static_cast<int32_t>((b >> 52) & 0x7FFULL);
    const int32_t diff = ea - eb;
    if (diff > 56) {
        return a;
    }
    const uint64_t ma = ((1ULL << 52) | (a & 0xFFFFFFFFFFFFFULL)) << 3;
    const uint64_t mb_raw = ((1ULL << 52) | (b & 0xFFFFFFFFFFFFFULL)) << 3;
    uint64_t sticky = 0ULL;
    if (diff > 0) {
        sticky = ((mb_raw & ((1ULL << diff) - 1ULL)) != 0ULL) ? 1ULL : 0ULL;
    }
    const uint64_t mb = (mb_raw >> diff) | sticky;
    uint64_t sum = ma + mb;
    int32_t er = ea;
    if (sum >= (1ULL << 56)) {
        const uint64_t s2 = sum & 1ULL;
        sum = (sum >> 1) | s2;
        er += 1;
    }
    uint64_t sig = sum >> 3;
    const uint64_t grs = sum & 7ULL;
    if (grs > 4ULL || (grs == 4ULL && ((sig & 1ULL) != 0ULL))) {
        sig += 1ULL;
        if (sig >= (1ULL << 53)) {
            sig >>= 1;
            er += 1;
        }
    }
    return (static_cast<uint64_t>(er) << 52) | (sig & 0xFFFFFFFFFFFFFULL);
}

__aicore__ inline void RunPrologPrepareMeta(
    uint64_t state_gm,
    uint64_t draft_ctrl_gm,
    uint64_t compact_ctrl_gm,
    uint64_t commit_ctrl_gm,
    uint64_t dyn_w_gm,
    uint64_t mc2_mask_gm,
    uint64_t step_qlen_gm,
    uint64_t step_qlen_ptrs_gm,
    uint64_t step_table_gm,
    uint64_t receipt_ring_gm,
    uint64_t commit_qlen_ptrs_gm,
    int32_t width,
    int32_t max_steps)
{
    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0) {
    DcciRange(state_gm, static_cast<int32_t>(sizeof(RoundState)));
    auto* st = reinterpret_cast<__gm__ RoundState*>(state_gm);
    const int32_t done = st->done;
    const int32_t r_idx = st->round_idx;
    const int32_t main_pos = st->main_pos;
    const int32_t h_lvl = st->horizon_level;
    const int32_t last_w_act = st->last_w_act;
    const int32_t eos_hit = st->eos_hit;
    const int64_t pending = st->pending_token;

    auto* d_ctrl = reinterpret_cast<__gm__ int64_t*>(draft_ctrl_gm);
    auto* cmp_ctrl = reinterpret_cast<__gm__ int64_t*>(compact_ctrl_gm);
    auto* cmt_ctrl = reinterpret_cast<__gm__ int64_t*>(commit_ctrl_gm);
    auto* dw = reinterpret_cast<__gm__ int32_t*>(dyn_w_gm);
    auto* mask = reinterpret_cast<__gm__ uint8_t*>(mc2_mask_gm);
    auto* s_ql = reinterpret_cast<__gm__ int32_t*>(step_qlen_gm);
    auto* s_ptrs = reinterpret_cast<__gm__ uint64_t*>(step_qlen_ptrs_gm);
    auto* s_tbl = reinterpret_cast<__gm__ int64_t*>(step_table_gm);
    auto* c_ptrs = reinterpret_cast<__gm__ uint64_t*>(commit_qlen_ptrs_gm);

    if (done != 0) {
        st->q_len = 0;
        st->drain_q_len = 0;
        if (draft_ctrl_gm != 0ULL) {
            d_ctrl[2] = 0;
        }
        if (dyn_w_gm != 0ULL) {
            for (int32_t i = 0; i < 16; ++i) {
                dw[i] = 0;
            }
        }
        if (mc2_mask_gm != 0ULL) {
            for (int32_t i = 0; i < width; ++i) {
                mask[i] = 0;
            }
        }
        if (compact_ctrl_gm != 0ULL) {
            cmp_ctrl[1] = 0;
        }
        if (commit_ctrl_gm != 0ULL) {
            cmt_ctrl[1] = 0;
            cmt_ctrl[2] = 0;
            cmt_ctrl[47] = 0;
        }
        if (step_qlen_ptrs_gm != 0ULL) {
            DcciRange(step_qlen_ptrs_gm, max_steps * 8);
        }
        for (int32_t s = 0; s < max_steps; ++s) {
            if (step_qlen_gm != 0ULL) {
                s_ql[s] = 0;
                s_ql[16 + s] = 0;
            }
            if (step_qlen_ptrs_gm != 0ULL && s_ptrs[s] != 0ULL) {
                *reinterpret_cast<__gm__ int32_t*>(s_ptrs[s]) = 0;
                PipeBarrier<PIPE_ALL>();
                DcciRange(s_ptrs[s], 64);
            }
        }
        if (commit_qlen_ptrs_gm != 0ULL) {
            DcciRange(commit_qlen_ptrs_gm, 16);
            for (int32_t i = 0; i < 2; ++i) {
                if (c_ptrs[i] != 0ULL) {
                    *reinterpret_cast<__gm__ int32_t*>(c_ptrs[i]) = 0;
                    PipeBarrier<PIPE_ALL>();
                    DcciRange(c_ptrs[i], 64);
                }
            }
        }
        if (receipt_ring_gm != 0ULL) {
            const int32_t slot_off = (r_idx & 3) * 16;
            auto* rec = reinterpret_cast<__gm__ int32_t*>(receipt_ring_gm) + slot_off;
            rec[0] = r_idx + 1;
            rec[1] = 0;
            rec[2] = main_pos;
            rec[3] = last_w_act;
            rec[4] = h_lvl;
            rec[5] = 0;
            rec[6] = 1;
            rec[7] = 0;
            rec[8] = eos_hit;
            rec[9] = st->draft_pos;
            rec[10] = 0;
            for (int32_t j = 11; j < 16; ++j) {
                rec[j] = 0;
            }
        }
    } else {
        st->q_len = width;
        st->drain_q_len = width;
        if (draft_ctrl_gm != 0ULL) {
            d_ctrl[0] = pending;
            d_ctrl[1] = static_cast<int64_t>(main_pos);
            d_ctrl[2] = static_cast<int64_t>(width);
            d_ctrl[3] = static_cast<int64_t>(h_lvl);
            d_ctrl[4] = 0;
            d_ctrl[5] = 0;
        }
        int32_t num_steps = max_steps;
        const int32_t stride = max_steps + 2;
        if (step_table_gm != 0ULL) {
            DcciRange(step_table_gm, 16 * stride * 8);
            const int64_t* row_tbl = nullptr;
            (void)row_tbl;
            const int32_t tbl_off = h_lvl * stride;
            num_steps = static_cast<int32_t>(s_tbl[tbl_off + 0]);
            if (draft_ctrl_gm != 0ULL) {
                d_ctrl[7] = s_tbl[tbl_off + 1];
            }
            if (step_qlen_ptrs_gm != 0ULL) {
                DcciRange(step_qlen_ptrs_gm, max_steps * 8);
            }
            for (int32_t s = 0; s < max_steps; ++s) {
                const int32_t val = (s < num_steps)
                    ? static_cast<int32_t>(s_tbl[tbl_off + 2 + s]) : 0;
                if (step_qlen_gm != 0ULL) {
                    s_ql[s] = val;
                    s_ql[16 + s] = val * 2;
                }
                if (step_qlen_ptrs_gm != 0ULL && s_ptrs[s] != 0ULL) {
                    *reinterpret_cast<__gm__ int32_t*>(s_ptrs[s]) = val;
                    PipeBarrier<PIPE_ALL>();
                    DcciRange(s_ptrs[s], 64);
                }
            }
        }
    }

    PipeBarrier<PIPE_ALL>();
    DcciRange(state_gm, static_cast<int32_t>(sizeof(RoundState)));
    if (draft_ctrl_gm != 0ULL) {
        DcciRange(draft_ctrl_gm, 64);
    }
    if (compact_ctrl_gm != 0ULL) {
        DcciRange(compact_ctrl_gm, 64);
    }
    if (commit_ctrl_gm != 0ULL) {
        DcciRange(commit_ctrl_gm, 64);
    }
    if (dyn_w_gm != 0ULL) {
        DcciRange(dyn_w_gm, 64);
    }
    if (mc2_mask_gm != 0ULL) {
        DcciRange(mc2_mask_gm, width);
    }
    if (step_qlen_gm != 0ULL && max_steps > 0) {
        DcciRange(step_qlen_gm, (16 + max_steps) * 4);
    }
    if (receipt_ring_gm != 0ULL) {
        DcciRange(receipt_ring_gm, 256);
    }
    Publish();
        }
    }
}

__aicore__ inline void RunResolveAndAdvance(
    uint64_t state_gm,
    uint64_t dyn_w_gm,
    uint64_t tree_tokens_gm,
    uint64_t tree_parents_gm,
    uint64_t target_tokens_gm,
    uint64_t compact_ctrl_gm,
    uint64_t commit_ctrl_gm,
    uint64_t receipt_ring_gm,
    uint64_t receipt_toks_gm,
    int32_t width,
    int32_t commit_m,
    int32_t max_pos,
    int32_t eos_token_id,
    uint64_t alpha_bits,
    uint64_t oma_bits,
    uint64_t mid_thresh_bits,
    uint64_t up_thresh_bits,
    uint64_t down_thresh_bits,
    int32_t streak_acc,
    int32_t drop_acc,
    int32_t jump_acc,
    int32_t num_levels)
{
    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0) {
    DcciRange(state_gm, static_cast<int32_t>(sizeof(RoundState)));
    DcciRange(dyn_w_gm, 64);
    DcciRange(tree_tokens_gm, width * 8);
    DcciRange(tree_parents_gm, width * 8);
    DcciRange(target_tokens_gm, width * 8);

    auto* st = reinterpret_cast<__gm__ RoundState*>(state_gm);
    if (st->q_len > 0) {

    const auto* dw = reinterpret_cast<__gm__ const int32_t*>(dyn_w_gm);
    const auto* t_toks = reinterpret_cast<__gm__ const int64_t*>(tree_tokens_gm);
    const auto* t_pars = reinterpret_cast<__gm__ const int64_t*>(tree_parents_gm);
    const auto* preds = reinterpret_cast<__gm__ const int64_t*>(target_tokens_gm);

    int32_t w_act = dw[0];
    if (w_act < 1) {
        w_act = 1;
    }
    if (w_act > width) {
        w_act = width;
    }
    if (w_act > 32) {
        w_act = 32;
    }

    bool reachable[32];
    int32_t depth[32];
    int32_t came_from[32];
    for (int32_t i = 0; i < 32; ++i) {
        reachable[i] = false;
        depth[i] = 0;
        came_from[i] = -1;
    }
    reachable[0] = true;
    int32_t best_node = 0;
    int32_t best_depth = 0;
    for (int32_t node = 1; node < w_act; ++node) {
        const int64_t par64 = t_pars[node];
        if (par64 >= 0 && par64 < static_cast<int64_t>(node)) {
            const int32_t par = static_cast<int32_t>(par64);
            if (reachable[par] && preds[par] == t_toks[node]) {
                reachable[node] = true;
                came_from[node] = par;
                depth[node] = depth[par] + 1;
                if (depth[node] > best_depth) {
                    best_depth = depth[node];
                    best_node = node;
                }
            }
        }
    }

    int32_t acc_rows[32];
    const int32_t path_len = best_depth + 1;
    int32_t cur = best_node;
    for (int32_t i = path_len - 1; i >= 0; --i) {
        acc_rows[i] = cur;
        cur = came_from[cur];
    }

    int64_t emitted[32];
    for (int32_t i = 0; i < path_len; ++i) {
        emitted[i] = preds[acc_rows[i]];
    }

    const int32_t main_start = st->main_pos;
    const int32_t eff_max_pos = (max_pos > 0) ? max_pos : st->max_pos;
    const int32_t rem = eff_max_pos - main_start;
    int32_t acc = (rem <= 0) ? 0 : ((path_len < rem) ? path_len : rem);
    int32_t eos_hit = 0;
    const int32_t eff_eos = (eos_token_id >= 0) ? eos_token_id : st->eos_token_id;
    if (eff_eos >= 0) {
        for (int32_t i = 0; i < acc; ++i) {
            if (emitted[i] == static_cast<int64_t>(eff_eos)) {
                acc = i + 1;
                eos_hit = 1;
                break;
            }
        }
    }
    const int32_t done = (eos_hit != 0 || (main_start + acc) >= eff_max_pos) ? 1 : 0;

    bool compact_needed = false;
    for (int32_t i = 0; i < acc; ++i) {
        if (acc_rows[i] != i) {
            compact_needed = true;
            break;
        }
    }

    if (compact_ctrl_gm != 0ULL) {
        auto* cmp = reinterpret_cast<__gm__ int64_t*>(compact_ctrl_gm);
        if (compact_needed && acc > 0) {
            cmp[0] = static_cast<int64_t>(main_start);
            cmp[1] = static_cast<int64_t>(acc);
            for (int32_t i = 0; i < acc; ++i) {
                cmp[2 + i] = static_cast<int64_t>(main_start + acc_rows[i]);
            }
        } else {
            cmp[1] = 0;
        }
    }

    if (commit_ctrl_gm != 0ULL) {
        auto* cmt = reinterpret_cast<__gm__ int64_t*>(commit_ctrl_gm);
        if (acc > 0) {
            cmt[0] = static_cast<int64_t>(main_start);
            cmt[1] = static_cast<int64_t>(acc);
            cmt[2] = static_cast<int64_t>(acc);
            cmt[47] = static_cast<int64_t>(commit_m + acc);
            const int32_t last_row = acc_rows[acc - 1];
            for (int32_t j = 0; j < commit_m; ++j) {
                cmt[3 + j] = static_cast<int64_t>((j < acc) ? acc_rows[j] : last_row);
            }
        } else {
            cmt[1] = 0;
            cmt[2] = 0;
            cmt[47] = 0;
        }
    }

    const uint64_t acc_f64 = F64FromU32(static_cast<uint32_t>(acc));
    uint64_t ema = F64Add(F64Mul(oma_bits, st->ema_acc), F64Mul(alpha_bits, acc_f64));
    int32_t hi = st->hi_streak;
    int32_t lo = st->lo_streak;
    int32_t lvl = st->horizon_level;
    if (acc >= streak_acc) {
        hi += 1;
        lo = 0;
    } else if (acc <= drop_acc) {
        lo += 1;
        hi = 0;
    } else {
        hi = 0;
        lo = 0;
    }
    const int32_t top = num_levels - 1;
    if (acc <= 2 || lo >= 2) {
        lvl = 0;
        if (down_thresh_bits < ema) {
            ema = down_thresh_bits;
        }
    } else if (acc >= jump_acc || hi >= 2) {
        lvl = top;
        if (up_thresh_bits > ema) {
            ema = up_thresh_bits;
        }
    } else if (num_levels >= 3) {
        if (ema >= up_thresh_bits) {
            lvl = 2;
        } else if (ema >= mid_thresh_bits) {
            lvl = 1;
        } else if (ema < down_thresh_bits) {
            lvl = 0;
        }
    } else {
        if (ema >= up_thresh_bits) {
            lvl = 1;
        } else if (ema < down_thresh_bits) {
            lvl = 0;
        }
    }

    const int32_t r_cur = st->round_idx;
    const int32_t new_pos = main_start + acc;
    st->main_pos = new_pos;
    st->draft_pos = new_pos;
    if (acc > 0) {
        st->pending_token = emitted[acc - 1];
    }
    st->ema_acc = ema;
    st->hi_streak = hi;
    st->lo_streak = lo;
    st->horizon_level = lvl;
    st->eos_hit = eos_hit;
    st->done = done;
    st->last_w_act = w_act;
    st->drain_q_len = (done != 0) ? 0 : width;
    st->prev_accepted_total += static_cast<int64_t>(acc);
    st->round_idx = r_cur + 1;

    if (receipt_ring_gm != 0ULL) {
        const int32_t slot_off = (r_cur & 3) * 16;
        auto* rec = reinterpret_cast<__gm__ int32_t*>(receipt_ring_gm) + slot_off;
        rec[0] = r_cur + 1;
        rec[1] = acc;
        rec[2] = new_pos;
        rec[3] = w_act;
        rec[4] = lvl;
        rec[5] = (acc > 0) ? (acc - 1) : 0;
        rec[6] = done;
        rec[7] = width;
        rec[8] = eos_hit;
        rec[9] = new_pos;
        rec[10] = compact_needed ? 1 : 0;
        for (int32_t j = 11; j < 16; ++j) {
            rec[j] = 0;
        }
    }
    if (receipt_toks_gm != 0ULL) {
        const int32_t tok_off = (r_cur & 3) * width;
        auto* rtok = reinterpret_cast<__gm__ int64_t*>(receipt_toks_gm) + tok_off;
        for (int32_t i = 0; i < width; ++i) {
            rtok[i] = (i < acc) ? emitted[i] : 0LL;
        }
    }

    PipeBarrier<PIPE_ALL>();
    DcciRange(state_gm, static_cast<int32_t>(sizeof(RoundState)));
    if (compact_ctrl_gm != 0ULL) {
        DcciRange(compact_ctrl_gm, 64 + width * 8);
    }
    if (commit_ctrl_gm != 0ULL) {
        DcciRange(commit_ctrl_gm, 48 * 8);
    }
    if (receipt_ring_gm != 0ULL) {
        DcciRange(receipt_ring_gm, 256);
    }
    if (receipt_toks_gm != 0ULL) {
        DcciRange(receipt_toks_gm, 4 * width * 8);
    }
    Publish();
    }
        }
    }
}

__aicore__ inline void RunEmbedLookup(
    TPipe& pipe,
    uint64_t t_ids_gm,
    uint64_t q_len_gm,
    uint64_t emb_w_gm,
    uint64_t partial_out_gm,
    int32_t width,
    int32_t hidden_dim,
    int32_t vocab_start,
    int32_t vocab_end)
{
    if ASCEND_IS_AIV {
    int32_t active = width;
    if (q_len_gm != 0ULL) {
        DcciRange(q_len_gm, 64);
        active = *reinterpret_cast<__gm__ const int32_t*>(q_len_gm);
    }
    if (active < 0) {
        active = 0;
    }
    if (active > width) {
        active = width;
    }
    const int32_t chunk = 512;
    const int32_t chunks_per_row = hidden_dim / chunk;
    const int32_t total_tasks = width * chunks_per_row;
    const int32_t aiv = GetBlockIdx();

    TBuf<TPosition::VECCALC> b_ids, b_bf16;
    pipe.InitBuffer(b_ids, 64 * 4);
    pipe.InitBuffer(b_bf16, chunk * 2);
    auto ub_ids = b_ids.Get<int32_t>();
    auto ub_bf16 = b_bf16.Get<bfloat16_t>();

    GlobalTensor<int32_t> g_ids;
    GlobalTensor<bfloat16_t> g_emb, g_out;
    g_ids.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(t_ids_gm), width);
    g_emb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(emb_w_gm));
    g_out.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t*>(partial_out_gm),
        static_cast<int64_t>(width) * hidden_dim);

    if (active > 0) {
        DcciRange(t_ids_gm, width * 4);
        const int32_t copy_ids = ((width + 7) / 8) * 8;
        DataCopy(ub_ids, g_ids, copy_ids);
        Fence<HardEvent::MTE2_S>();
    }

    for (int32_t task = aiv; task < total_tasks; task += 48) {
        const int32_t r = task / chunks_per_row;
        const int32_t c_idx = task % chunks_per_row;
        const int64_t col = static_cast<int64_t>(c_idx) * chunk;
        const int64_t dst_off = static_cast<int64_t>(r) * hidden_dim + col;
        bool valid_tok = false;
        int32_t tok = 0;
        if (r < active) {
            tok = ub_ids.GetValue(r);
            valid_tok = (tok >= vocab_start && tok < vocab_end);
        }
        if (valid_tok) {
            const int64_t src_off = static_cast<int64_t>(tok - vocab_start) * hidden_dim + col;
            DataCopy(ub_bf16, g_emb[src_off], chunk);
            Fence<HardEvent::MTE2_MTE3>();
        } else {
            Duplicate(ub_bf16.ReinterpretCast<int16_t>(), static_cast<int16_t>(0), chunk);
            Fence<HardEvent::V_MTE3>();
        }
        DataCopy(g_out[dst_off], ub_bf16, chunk);
        Fence<HardEvent::MTE3_V>();
        Fence<HardEvent::MTE3_MTE2>();
    }
    Publish();
    }
}

__aicore__ inline void RunExactIndexKeys(
    TPipe& pipe,
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
    float eps,
    int32_t is_neox)
{
    int32_t active_rows = rows_cap;
    if (q_len_gm != 0ULL) {
        DcciRange(q_len_gm, 64);
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

        TBuf<TPosition::A1> a_stage;
        TBuf<TPosition::B1> b_stage;
        TBuf<TPosition::A2> a_op;
        TBuf<TPosition::B2> b_op;
        TBuf<TPosition::CO1> c_acc;
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

        GlobalTensor<bfloat16_t> a_gm, b_gm;
        GlobalTensor<float> c_gm;
        a_gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(norm_x_gm), rows_cap * kHidden);
        b_gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(wk_w_gm), 128 * kHidden);
        c_gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(work_gm), rows_cap * 128);

        const int32_t blk = GetBlockIdx();
        const int32_t row_tiles = (active_rows + 15) / 16;
        const int32_t total_tiles = row_tiles * 8;

        for (int32_t tile = blk; tile < total_tiles; tile += 24) {
            const int32_t row_tile = tile / 8;
            const int32_t col_tile = tile % 8;
            const int32_t r_start = row_tile * kTile;
            const int32_t r_rem = active_rows - r_start;
            const int32_t r_valid = (r_rem < kTile) ? r_rem : kTile;

            Nd2NzParams copy_a{};
            copy_a.ndNum = 1;
            copy_a.nValue = r_valid;
            copy_a.dValue = kInner;
            copy_a.srcDValue = kHidden;
            copy_a.dstNzC0Stride = kTile;
            copy_a.dstNzNStride = 1;

            Nd2NzParams copy_b{};
            copy_b.ndNum = 1;
            copy_b.nValue = kTile;
            copy_b.dValue = kInner;
            copy_b.srcDValue = kHidden;
            copy_b.dstNzC0Stride = kTile;
            copy_b.dstNzNStride = 1;

            LoadData2DParams load_a{};
            load_a.repeatTimes = kInner / kTile;
            load_a.srcStride = 1;
            load_a.ifTranspose = false;

            LoadData2DParams load_b{};
            load_b.repeatTimes = kInner / kTile;
            load_b.srcStride = 1;
            load_b.ifTranspose = false;

            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID1);

            for (int32_t s = 0; s < kSteps; ++s) {
                const int32_t buf = s & 1;
                const event_t ev = buf ? EVENT_ID1 : EVENT_ID0;
                const int64_t k_off = static_cast<int64_t>(s) * kInner;
                const int64_t a_off = static_cast<int64_t>(r_start) * kHidden + k_off;
                const int64_t b_off = static_cast<int64_t>(col_tile) * kTile * kHidden + k_off;

                WaitFlag<HardEvent::MTE1_MTE2>(ev);
                DataCopy(a_l1[buf * kStage], a_gm[a_off], copy_a);
                DataCopy(b_l1[buf * kStage], b_gm[b_off], copy_b);
                SetFlag<HardEvent::MTE2_MTE1>(ev);

                WaitFlag<HardEvent::MTE2_MTE1>(ev);
                WaitFlag<HardEvent::M_MTE1>(ev);
                LoadData(a_l0[buf * kStage], a_l1[buf * kStage], load_a);
                LoadData(b_l0[buf * kStage], b_l1[buf * kStage], load_b);
                SetFlag<HardEvent::MTE1_MTE2>(ev);
                SetFlag<HardEvent::MTE1_M>(ev);

                WaitFlag<HardEvent::MTE1_M>(ev);
                MmadParams mad{};
                mad.m = kTile;
                mad.n = kTile;
                mad.k = kInner;
                mad.cmatrixInitVal = (s == 0);
                mad.cmatrixSource = false;
                Mmad(c_l0, a_l0[buf * kStage], b_l0[buf * kStage], mad);
                PipeBarrier<PIPE_M>();
                SetFlag<HardEvent::M_MTE1>(ev);
            }
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);

            SetFlag<HardEvent::M_FIX>(EVENT_ID0);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
            FixpipeParamsV220 store{};
            store.nSize = kTile;
            store.mSize = r_valid;
            store.srcStride = kTile;
            store.dstStride = 128;
            store.ndNum = 1;
            store.quantPre = QuantMode_t::NoQuant;
            const int64_t c_off = static_cast<int64_t>(r_start) * 128 + col_tile * kTile;
            Fixpipe(c_gm[c_off], c_l0, store);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        }
        Publish();
    }

    SyncAll<false>();

    if ASCEND_IS_AIV {
        const int32_t aiv = GetBlockIdx();
        DcciRange(slot_i64_gm, rows_cap * 8);
        DcciRange(work_gm, active_rows * 128 * 4);

        TBuf<TPosition::VECCALC> b_f32, b_bf, b_slot;
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

        GlobalTensor<float> g_work, g_gw, g_gb;
        GlobalTensor<bfloat16_t> g_cos, g_sin, g_kcache;
        GlobalTensor<int64_t> g_slot;
        g_work.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(work_gm), rows_cap * 128);
        g_gw.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(k_norm_w_gm), 128);
        g_gb.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(k_norm_b_gm), 128);
        g_cos.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cos_gm), rows_cap * 64);
        g_sin.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(sin_gm), rows_cap * 64);
        g_kcache.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(k_cache_gm));
        g_slot.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slot_i64_gm), rows_cap);

        const int32_t copy_sl = ((rows_cap + 3) / 4) * 4;
        DataCopy(ub_slot, g_slot, copy_sl);
        DataCopy(ub_gw, g_gw, 128);
        DataCopy(ub_gb, g_gb, 128);
        Fence<HardEvent::MTE2_S>();
        Fence<HardEvent::MTE2_V>();

        for (int32_t r = aiv; r < active_rows; r += 48) {
            const int64_t slot = ub_slot.GetValue(r);
            if (slot < 0) {
                continue;
            }
            DataCopy(ub_kw, g_work[r * 128], 128);
            DataCopy(ub_cos_bf, g_cos[r * 64], 64);
            DataCopy(ub_sin_bf, g_sin[r * 64], 64);
            Fence<HardEvent::MTE2_V>();

            Cast(ub_kw_bf, ub_kw, RoundMode::CAST_RINT, 128);
            PipeBarrier<PIPE_V>();
            Cast(ub_x, ub_kw_bf, RoundMode::CAST_NONE, 128);
            PipeBarrier<PIPE_V>();

            ReduceSum(ub_sum, ub_x, ub_tmp, 128);
            Fence<HardEvent::V_S>();
            const float mean = ub_sum.GetValue(0) * (1.0F / 128.0F);
            Fence<HardEvent::S_V>();
            Adds(ub_diff, ub_x, -mean, 128);
            PipeBarrier<PIPE_V>();
            Mul(ub_sq, ub_diff, ub_diff, 128);
            PipeBarrier<PIPE_V>();
            ReduceSum(ub_sum, ub_sq, ub_tmp, 128);
            Fence<HardEvent::V_S>();
            const float var = ub_sum.GetValue(0) * (1.0F / 128.0F);
            const float inv_std = 1.0F / sqrt(var + eps);
            Fence<HardEvent::S_V>();
            Muls(ub_x, ub_diff, inv_std, 128);
            PipeBarrier<PIPE_V>();
            Mul(ub_x, ub_x, ub_gw, 128);
            PipeBarrier<PIPE_V>();
            Add(ub_x, ub_x, ub_gb, 128);
            PipeBarrier<PIPE_V>();
            Cast(ub_out_bf, ub_x, RoundMode::CAST_RINT, 128);
            PipeBarrier<PIPE_V>();

            Cast(ub_k, ub_out_bf, RoundMode::CAST_NONE, 64);
            Cast(ub_cos, ub_cos_bf, RoundMode::CAST_NONE, 64);
            Cast(ub_sin, ub_sin_bf, RoundMode::CAST_NONE, 64);
            Fence<HardEvent::V_S>();
            for (int32_t j = 0; j < 32; ++j) {
                const float c = ub_cos.GetValue(j);
                const float s = ub_sin.GetValue(j);
                if (is_neox == 0) {
                    const float k0 = ub_k.GetValue(2 * j);
                    const float k1 = ub_k.GetValue(2 * j + 1);
                    ub_diff.SetValue(2 * j, k0 * c - k1 * s);
                    ub_diff.SetValue(2 * j + 1, k1 * c + k0 * s);
                } else {
                    const float k0 = ub_k.GetValue(j);
                    const float k1 = ub_k.GetValue(j + 32);
                    ub_diff.SetValue(j, k0 * c - k1 * s);
                    ub_diff.SetValue(j + 32, k1 * c + k0 * s);
                }
            }
            Fence<HardEvent::S_V>();
            Cast(ub_out_bf, ub_diff, RoundMode::CAST_RINT, 64);
            Fence<HardEvent::V_MTE3>();
            DataCopy(g_kcache[slot * 128], ub_out_bf, 128);
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::MTE3_MTE2>();
        }
        Publish();
    }
}

__aicore__ inline void RunStep3cPrologPackGuarded(
    TPipe& pipe,
    uint64_t q_len_gm,
    uint64_t active_toks_gm,
    uint64_t emb_w_gm,
    uint64_t rot_part_gm,
    uint64_t rot_bias_gm,
    uint64_t comm_in_gm,
    int32_t b_rows,
    int32_t vocab_start,
    int32_t vocab_end,
    int32_t rank)
{
    int32_t eff = b_rows;
    if (q_len_gm != 0ULL) {
        DcciRange(q_len_gm, 64);
        const int32_t q = *reinterpret_cast<__gm__ const int32_t*>(q_len_gm);
        if (q <= 0) {
            return;
        }
        if (q < eff) {
            eff = q;
        }
    }
    RunStep3cPrologPack(
        pipe, active_toks_gm, emb_w_gm, rot_part_gm, rot_bias_gm, comm_in_gm,
        eff, vocab_start, vocab_end, rank);
}

__aicore__ inline void RunStep3cDualRmsNormSliceGuarded(
    TPipe& pipe,
    uint64_t q_len_gm,
    uint64_t comm_out_gm,
    uint64_t enorm_w_gm,
    uint64_t hnorm_w_gm,
    uint64_t eh_in_gm,
    int32_t b_rows,
    int32_t rank,
    float eps)
{
    int32_t eff = b_rows;
    if (q_len_gm != 0ULL) {
        DcciRange(q_len_gm, 64);
        const int32_t q = *reinterpret_cast<__gm__ const int32_t*>(q_len_gm);
        if (q <= 0) {
            return;
        }
        if (q < eff) {
            eff = q;
        }
    }
    RunStep3cDualRmsNormSlice(
        pipe, comm_out_gm, enorm_w_gm, hnorm_w_gm, eh_in_gm, eff, rank, eps);
}

__aicore__ inline void RunLogitsTopkLseGuarded(
    TPipe& pipe,
    uint64_t q_len_gm,
    uint64_t logits_gm,
    uint64_t work_gm,
    uint64_t gather_in_gm,
    int32_t m_rows,
    int32_t row_stride,
    int32_t valid_vocab,
    int32_t vocab_start,
    int32_t k,
    int32_t rank)
{
    int32_t eff = m_rows;
    if (q_len_gm != 0ULL) {
        DcciRange(q_len_gm, 64);
        const int32_t q = *reinterpret_cast<__gm__ const int32_t*>(q_len_gm);
        if (q <= 0) {
            return;
        }
        if (q < eff) {
            eff = q;
        }
    }
    RunLogitsTopkLseImpl<true>(
        pipe, logits_gm, work_gm, gather_in_gm,
        eff, row_stride, valid_vocab, vocab_start, k, rank);
}

__aicore__ inline void RunAddRmsNormQuant(
    TPipe& pipe,
    uint64_t x_gm,
    uint64_t res_gm,
    uint64_t norm_w_gm,
    uint64_t y_out_gm,
    uint64_t q8_out_gm,
    uint64_t scale_out_gm,
    uint64_t q_len_gm,
    int32_t rows_cap,
    int32_t hidden_dim,
    float eps)
{
    if ASCEND_IS_AIV {
    int32_t active = rows_cap;
    if (q_len_gm != 0ULL) {
        DcciRange(q_len_gm, 64);
        active = *reinterpret_cast<__gm__ const int32_t*>(q_len_gm);
    }
    if (active > rows_cap) {
        active = rows_cap;
    }
    const int32_t aiv = GetBlockIdx();
    if (active > 0 && aiv < active) {

    TBuf<TPosition::VECCALC> b_f32, b_bf, b_i8;
    pipe.InitBuffer(b_f32, 3 * hidden_dim * 4 + 256 * 4);
    pipe.InitBuffer(b_bf, 3 * hidden_dim * 2);
    pipe.InitBuffer(b_i8, hidden_dim + 32);

    auto ub_f32 = b_f32.Get<float>();
    auto ub_x = ub_f32[0];
    auto ub_w = ub_f32[hidden_dim];
    auto ub_tmp = ub_f32[2 * hidden_dim];
    auto ub_red = ub_f32[3 * hidden_dim];
    auto ub_sc = ub_f32[3 * hidden_dim + 128];

    auto ub_bf = b_bf.Get<bfloat16_t>();
    auto ub_x_bf = ub_bf[0];
    auto ub_r_bf = ub_bf[hidden_dim];
    auto ub_w_bf = ub_bf[2 * hidden_dim];
    auto ub_q8 = b_i8.Get<int8_t>();

    GlobalTensor<bfloat16_t> g_x, g_r, g_w, g_y;
    GlobalTensor<int8_t> g_q8;
    GlobalTensor<float> g_sc;
    g_x.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x_gm));
    g_r.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(res_gm));
    g_w.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(norm_w_gm));
    g_y.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y_out_gm));
    g_q8.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(q8_out_gm));
    g_sc.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale_out_gm));

    DataCopy(ub_w_bf, g_w, hidden_dim);
    Fence<HardEvent::MTE2_V>();
    Cast(ub_w, ub_w_bf, RoundMode::CAST_NONE, hidden_dim);
    PipeBarrier<PIPE_V>();

    for (int32_t r = aiv; r < active; r += 48) {
        const int64_t row_off = static_cast<int64_t>(r) * hidden_dim;
        DataCopy(ub_x_bf, g_x[row_off], hidden_dim);
        if (res_gm != 0ULL) {
            DataCopy(ub_r_bf, g_r[row_off], hidden_dim);
        }
        Fence<HardEvent::MTE2_V>();
        Cast(ub_x, ub_x_bf, RoundMode::CAST_NONE, hidden_dim);
        if (res_gm != 0ULL) {
            Cast(ub_tmp, ub_r_bf, RoundMode::CAST_NONE, hidden_dim);
            PipeBarrier<PIPE_V>();
            Add(ub_x, ub_x, ub_tmp, hidden_dim);
        }
        PipeBarrier<PIPE_V>();
        Mul(ub_tmp, ub_x, ub_x, hidden_dim);
        PipeBarrier<PIPE_V>();
        ReduceSum(ub_red, ub_tmp, ub_w_bf.ReinterpretCast<float>(), hidden_dim);
        Fence<HardEvent::V_S>();
        const float mean_sq = ub_red.GetValue(0) / static_cast<float>(hidden_dim);
        const float rsqrt_v = 1.0F / sqrt(mean_sq + eps);
        Fence<HardEvent::S_V>();
        Muls(ub_x, ub_x, rsqrt_v, hidden_dim);
        PipeBarrier<PIPE_V>();
        Mul(ub_x, ub_x, ub_w, hidden_dim);
        PipeBarrier<PIPE_V>();
        if (y_out_gm != 0ULL) {
            Cast(ub_x_bf, ub_x, RoundMode::CAST_RINT, hidden_dim);
            Fence<HardEvent::V_MTE3>();
            DataCopy(g_y[row_off], ub_x_bf, hidden_dim);
            Fence<HardEvent::MTE3_V>();
        }
        if (q8_out_gm != 0ULL) {
            Abs(ub_tmp, ub_x, hidden_dim);
            PipeBarrier<PIPE_V>();
            ReduceMax(ub_red, ub_tmp, ub_r_bf.ReinterpretCast<float>(), hidden_dim);
            Fence<HardEvent::V_S>();
            const float max_v = ub_red.GetValue(0);
            const float sc = (max_v > 0.0F) ? (max_v / 127.0F) : 1.0F;
            const float inv_sc = 1.0F / sc;
            ub_sc.SetValue(0, sc);
            Fence<HardEvent::S_V>();
            Muls(ub_tmp, ub_x, inv_sc, hidden_dim);
            PipeBarrier<PIPE_V>();
            Cast(ub_r_bf.ReinterpretCast<half>(), ub_tmp, RoundMode::CAST_RINT, hidden_dim);
            PipeBarrier<PIPE_V>();
            Cast(ub_q8, ub_r_bf.ReinterpretCast<half>(), RoundMode::CAST_RINT, hidden_dim);
            Fence<HardEvent::V_MTE3>();
            Fence<HardEvent::S_MTE3>();
            DataCopy(g_q8[row_off], ub_q8, hidden_dim);
            if (scale_out_gm != 0ULL) {
                DataCopyExtParams copy_sc{1, 4, 0, 0, 0};
                DataCopyPad(g_sc[r], ub_sc, copy_sc);
            }
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::MTE3_MTE2>();
        }
    }
    Publish();
    }
    }
}

__aicore__ inline void RunPrologPackCompact(
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
        g_rot.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(rot_part_gm), m_max * kDspineShardDim);
        g_out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(comm_in_gm), 2 * m_max * 6144);

        DcciRange(ctrl_gm, 48 * 8);
        DataCopy(ub_ctrl, g_ctrl, 48);
        Fence<HardEvent::MTE2_S>();

        const int32_t count = static_cast<int32_t>(ub_ctrl.GetValue(2));
        if (count <= 0) {
            return;
        }
        DataCopy(ub_ttoks, g_ttoks, 32);
        Fence<HardEvent::MTE2_S>();
        const int32_t aiv = GetBlockIdx();

        for (int32_t task = aiv; task < count * 12; task += 48) {
            const int32_t r_idx = task / 12;
            const int64_t col = static_cast<int64_t>(task % 12) * 512;
            const int64_t row_id = ub_ctrl.GetValue(3 + r_idx);
            const int64_t tok = ub_ttoks.GetValue(row_id);
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

        for (int32_t task = aiv; task < count * kDspineWorldSize; task += 48) {
            const int32_t r_idx = task / kDspineWorldSize;
            const int32_t s_rk = task % kDspineWorldSize;
            const int64_t dst_off = static_cast<int64_t>(m_max + r_idx) * 6144
                + static_cast<int64_t>(s_rk) * kDspineShardDim;
            if (s_rk == rank) {
                DataCopy(ub_bf16, g_rot[r_idx * kDspineShardDim], kDspineShardDim);
                Fence<HardEvent::MTE2_MTE3>();
            } else {
                Duplicate(
                    ub_bf16.ReinterpretCast<int16_t>(), static_cast<int16_t>(0), kDspineShardDim);
                Fence<HardEvent::V_MTE3>();
            }
            DataCopy(g_out[dst_off], ub_bf16, kDspineShardDim);
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::MTE3_MTE2>();
        }
        Publish();
    }
}


__aicore__ inline void RmsNorm6144(
    LocalTensor<bfloat16_t>& ub_x,
    LocalTensor<float>& ub_xf,
    LocalTensor<float>& ub_sq,
    LocalTensor<float>& ub_gf,
    LocalTensor<float>& ub_sum,
    float eps,
    bool is_add_norm)
{
    Mul(ub_sq, ub_xf, ub_xf, 6144);
    PipeBarrier<PIPE_V>();
    float rstd = 0.0F;
    if (is_add_norm) {
        Muls(ub_sq, ub_sq, 1.0F / 6144.0F, 6144);
        PipeBarrier<PIPE_V>();
        Duplicate(ub_sum, 0.0F, 64);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams rp;
        rp.src0RepStride = 8;
        rp.src0BlkStride = 1;
        rp.src1RepStride = 0;
        rp.src1BlkStride = 1;
        rp.dstRepStride = 0;
        rp.dstBlkStride = 1;
        Add(ub_sum, ub_sq, ub_sum, 64, 96, rp);
        PipeBarrier<PIPE_V>();
        AscendCUtils::SetMask<float>(64);
        WholeReduceSum<float, false>(ub_sum, ub_sum, MASK_PLACEHOLDER, 1, 0, 1, 0);
        PipeBarrier<PIPE_V>();
        ResetMask();
        Adds(ub_sum, ub_sum, eps, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(ub_sum, ub_sum, 1);
        Duplicate(ub_sq, 1.0F, 1);
        PipeBarrier<PIPE_V>();
        Div(ub_sum, ub_sq, ub_sum, 1);
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_S>();
        rstd = ub_sum.GetValue(0);
        Fence<HardEvent::S_V>();
    } else {
        Add(ub_sq, ub_sq, ub_sq[4096], 2048);
        PipeBarrier<PIPE_V>();
        for (int32_t s = 2048; s >= 64; s >>= 1) {
            Add(ub_sq, ub_sq, ub_sq[s], s);
            PipeBarrier<PIPE_V>();
        }
        AscendCUtils::SetMask<float>(64);
        WholeReduceSum<float, false>(ub_sum, ub_sq, MASK_PLACEHOLDER, 1, 0, 1, 0);
        PipeBarrier<PIPE_V>();
        ResetMask();
        Fence<HardEvent::V_S>();
        rstd = 1.0F / sqrt(ub_sum.GetValue(0) * (1.0F / 6144.0F) + eps);
        Fence<HardEvent::S_V>();
    }
    Muls(ub_xf, ub_xf, rstd, 6144);
    PipeBarrier<PIPE_V>();
    if (is_add_norm) {
        Cast(ub_x, ub_xf, RoundMode::CAST_RINT, 6144);
        PipeBarrier<PIPE_V>();
        Cast(ub_xf, ub_x, RoundMode::CAST_NONE, 6144);
        PipeBarrier<PIPE_V>();
    }
    Mul(ub_xf, ub_xf, ub_gf, 6144);
    PipeBarrier<PIPE_V>();
    Cast(ub_x, ub_xf, RoundMode::CAST_RINT, 6144);
}

__aicore__ inline void RunTargetProlog(
    TPipe& pipe,
    uint64_t q_len_gm,
    uint64_t t_ids_gm,
    uint64_t emb_w_gm,
    uint64_t gamma_gm,
    uint64_t emb_out_gm,
    uint64_t l0_normed_gm,
    uint64_t comm_win_gm,
    int32_t width,
    int32_t vocab_size,
    int32_t vocab_start,
    int32_t vocab_end,
    int32_t rank,
    int32_t ranks,
    float eps)
{
    if ASCEND_IS_AIV {
        if (q_len_gm == 0ULL || comm_win_gm == 0ULL || width <= 0) {
            return;
        }
        DcciRange(q_len_gm, 64);
        int32_t active = *reinterpret_cast<__gm__ const int32_t*>(q_len_gm);
        if (active <= 0) {
            return;
        }
        if (active > width) {
            active = width;
        }
        const int32_t aiv = GetBlockIdx();
        const uint32_t seq = DspineReadSeq(comm_win_gm);
        const uint32_t epoch = seq + 1U;

        TBuf<TPosition::VECCALC> b_ids, b_row, b_g, b_xf, b_sq, b_gf, b_sum;
        pipe.InitBuffer(b_ids, 64 * 4);
        pipe.InitBuffer(b_row, 6144 * 2);
        pipe.InitBuffer(b_g, 6144 * 2);
        pipe.InitBuffer(b_xf, 6144 * 4);
        pipe.InitBuffer(b_sq, 6144 * 4);
        pipe.InitBuffer(b_gf, 6144 * 4);
        pipe.InitBuffer(b_sum, 64 * 4);

        auto ub_ids = b_ids.Get<int32_t>();
        auto ub_row = b_row.Get<bfloat16_t>();
        auto ub_g = b_g.Get<bfloat16_t>();
        auto ub_xf = b_xf.Get<float>();
        auto ub_sq = b_sq.Get<float>();
        auto ub_gf = b_gf.Get<float>();
        auto ub_sum = b_sum.Get<float>();

        GlobalTensor<int32_t> g_ids;
        GlobalTensor<bfloat16_t> g_emb, g_gamma, g_emb_out, g_l0_normed;
        g_ids.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(t_ids_gm), width);
        g_emb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(emb_w_gm));
        g_gamma.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gamma_gm), 6144);
        g_emb_out.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(emb_out_gm),
            static_cast<int64_t>(width) * 6144);
        g_l0_normed.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(l0_normed_gm),
            static_cast<int64_t>(width) * 6144);

        DcciRange(t_ids_gm, width * 4);
        const int32_t copy_ids = ((width + 7) / 8) * 8;
        DataCopy(ub_ids, g_ids, copy_ids);
        Fence<HardEvent::MTE2_S>();

        for (int32_t task = aiv; task < active * 12; task += 48) {
            const int32_t r = task / 12;
            const int64_t col = static_cast<int64_t>(task % 12) * 512;
            const int32_t tok = ub_ids.GetValue(r);
            if (tok >= vocab_start && tok < vocab_end) {
                const int64_t src_off =
                    static_cast<int64_t>(tok - vocab_start) * 6144 + col;
                const int64_t dst_off = static_cast<int64_t>(r) * 6144 + col;
                DataCopy(ub_row, g_emb[src_off], 512);
                Fence<HardEvent::MTE2_MTE3>();
                DataCopy(g_emb_out[dst_off], ub_row, 512);
                Fence<HardEvent::MTE3_MTE2>();
            }
        }
        Publish();
        SyncAll<true>();

        __gm__ bfloat16_t* plane_base = reinterpret_cast<__gm__ bfloat16_t*>(
            DspinePayloadPtr(comm_win_gm, epoch, 0));
        if (aiv == 0) {
            DspineWriteSeq(comm_win_gm, epoch);
            Publish();
        } else if (aiv < ranks) {
            const int32_t peer = (rank + aiv) % ranks;
            __gm__ bfloat16_t* rem_plane = reinterpret_cast<__gm__ bfloat16_t*>(
                aclshmem_ptr(plane_base, peer));
            GlobalTensor<bfloat16_t> g_rem;
            g_rem.SetGlobalBuffer(rem_plane, static_cast<int64_t>(width) * 6144);
            for (int32_t r = 0; r < active; ++r) {
                const int32_t tok = ub_ids.GetValue(r);
                if (tok >= vocab_start && tok < vocab_end) {
                    const int64_t row_off = static_cast<int64_t>(r) * 6144;
                    DataCopy(ub_row, g_emb_out[row_off], 6144);
                    Fence<HardEvent::MTE2_MTE3>();
                    DataCopy(g_rem[row_off], ub_row, 6144);
                    Fence<HardEvent::MTE3_MTE2>();
                }
            }
            Fence<HardEvent::MTE3_S>();
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::DDR>();
            __gm__ int32_t* remote_sig = reinterpret_cast<__gm__ int32_t*>(
                aclshmem_ptr(DspineFlagPtr(comm_win_gm, epoch, rank), peer));
            *remote_sig = static_cast<int32_t>(epoch);
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(remote_sig));

            __gm__ volatile int32_t* sig = DspineFlagPtr(comm_win_gm, epoch, peer);
            while (true) {
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(
                    const_cast<__gm__ int32_t*>(sig)));
                if (*sig == static_cast<int32_t>(epoch)) {
                    break;
                }
            }
        }
        SyncAll<true>();

        if (aiv < active) {
            GlobalTensor<bfloat16_t> g_plane;
            g_plane.SetGlobalBuffer(plane_base, static_cast<int64_t>(width) * 6144);
            DataCopy(ub_g, g_gamma, 6144);
            Fence<HardEvent::MTE2_V>();
            Cast(ub_gf, ub_g, RoundMode::CAST_NONE, 6144);
            PipeBarrier<PIPE_V>();

            for (int32_t r = aiv; r < active; r += 48) {
                const int32_t tok = ub_ids.GetValue(r);
                const int64_t row_off = static_cast<int64_t>(r) * 6144;
                if (tok >= vocab_start && tok < vocab_end) {
                    DataCopy(ub_row, g_emb_out[row_off], 6144);
                    Fence<HardEvent::MTE2_V>();
                } else if (tok >= 0 && tok < vocab_size) {
                    DataCopy(ub_row, g_plane[row_off], 6144);
                    Fence<HardEvent::MTE2_MTE3>();
                    Fence<HardEvent::MTE2_V>();
                    DataCopy(g_emb_out[row_off], ub_row, 6144);
                    Fence<HardEvent::MTE3_V>();
                } else {
                    Duplicate(
                        ub_row.ReinterpretCast<int16_t>(), static_cast<int16_t>(0), 6144);
                    Fence<HardEvent::V_MTE3>();
                    DataCopy(g_emb_out[row_off], ub_row, 6144);
                    Fence<HardEvent::MTE3_V>();
                    PipeBarrier<PIPE_V>();
                }
                Cast(ub_xf, ub_row, RoundMode::CAST_NONE, 6144);
                PipeBarrier<PIPE_V>();
                RmsNorm6144(ub_row, ub_xf, ub_sq, ub_gf, ub_sum, eps, false);
                Fence<HardEvent::V_MTE3>();
                DataCopy(g_l0_normed[row_off], ub_row, 6144);
                Fence<HardEvent::MTE3_MTE2>();
                Fence<HardEvent::MTE3_V>();
            }
        }
        Publish();
    }
}

__aicore__ inline void RunTargetAddRmsNorm(
    TPipe& pipe,
    uint64_t q_len_gm,
    uint64_t x1_gm,
    uint64_t x2_gm,
    uint64_t gamma_gm,
    uint64_t out_gm,
    int32_t width,
    float eps)
{
    if ASCEND_IS_AIV {
        int32_t active = width;
        if (q_len_gm != 0ULL) {
            DcciRange(q_len_gm, 64);
            active = *reinterpret_cast<__gm__ const int32_t*>(q_len_gm);
        }
        if (active <= 0) {
            return;
        }
        if (active > width) {
            active = width;
        }
        const int32_t aiv = GetBlockIdx();
        if (aiv < active) {
            TBuf<TPosition::VECCALC> b_x1, b_x2, b_g, b_xf, b_sq, b_gf, b_sum;
            pipe.InitBuffer(b_x1, 6144 * 2);
            pipe.InitBuffer(b_x2, 6144 * 2);
            pipe.InitBuffer(b_g, 6144 * 2);
            pipe.InitBuffer(b_xf, 6144 * 4);
            pipe.InitBuffer(b_sq, 6144 * 4);
            pipe.InitBuffer(b_gf, 6144 * 4);
            pipe.InitBuffer(b_sum, 64 * 4);

            auto ub_x1 = b_x1.Get<bfloat16_t>();
            auto ub_x2 = b_x2.Get<bfloat16_t>();
            auto ub_g = b_g.Get<bfloat16_t>();
            auto ub_xf = b_xf.Get<float>();
            auto ub_sq = b_sq.Get<float>();
            auto ub_gf = b_gf.Get<float>();
            auto ub_sum = b_sum.Get<float>();

            GlobalTensor<bfloat16_t> g_x1, g_x2, g_g, g_out;
            g_x1.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x1_gm));
            g_x2.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x2_gm));
            g_g.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gamma_gm), 6144);
            g_out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out_gm));

            DataCopy(ub_g, g_g, 6144);
            Fence<HardEvent::MTE2_V>();
            Cast(ub_gf, ub_g, RoundMode::CAST_NONE, 6144);
            PipeBarrier<PIPE_V>();

            for (int32_t r = aiv; r < active; r += 48) {
                const int64_t row_off = static_cast<int64_t>(r) * 6144;
                DataCopy(ub_x1, g_x1[row_off], 6144);
                DataCopy(ub_x2, g_x2[row_off], 6144);
                Fence<HardEvent::MTE2_V>();
                Cast(ub_xf, ub_x1, RoundMode::CAST_NONE, 6144);
                Cast(ub_sq, ub_x2, RoundMode::CAST_NONE, 6144);
                PipeBarrier<PIPE_V>();
                Add(ub_xf, ub_xf, ub_sq, 6144);
                PipeBarrier<PIPE_V>();
                Cast(ub_x1, ub_xf, RoundMode::CAST_RINT, 6144);
                PipeBarrier<PIPE_V>();
                Cast(ub_xf, ub_x1, RoundMode::CAST_NONE, 6144);
                PipeBarrier<PIPE_V>();
                RmsNorm6144(ub_x1, ub_xf, ub_sq, ub_gf, ub_sum, eps, true);
                Fence<HardEvent::V_MTE3>();
                DataCopy(g_out[row_off], ub_x1, 6144);
                Fence<HardEvent::MTE3_MTE2>();
                Fence<HardEvent::MTE3_V>();
            }
        }
        Publish();
    }
}

__aicore__ inline void RunTargetArgmax(
    TPipe& pipe,
    uint64_t q_len_gm,
    uint64_t logits_gm,
    uint64_t work_gm,
    uint64_t out_tokens_gm,
    uint64_t comm_win_gm,
    int32_t width,
    int32_t valid_vocab,
    int32_t vocab_start,
    int32_t rank,
    int32_t ranks)
{
    if ASCEND_IS_AIV {
        if (q_len_gm == 0ULL || comm_win_gm == 0ULL || width <= 0) {
            return;
        }
        DcciRange(q_len_gm, 64);
        int32_t eff = *reinterpret_cast<__gm__ const int32_t*>(q_len_gm);
        if (eff <= 0) {
            return;
        }
        if (eff > width) {
            eff = width;
        }
        const int32_t aiv = GetBlockIdx();
        const uint32_t seq = DspineReadSeq(comm_win_gm);
        const uint32_t epoch = seq + 1U;

        const int32_t aivs_per_row = (48 / eff >= 1) ? (48 / eff) : 1;
        const int32_t r_idx = aiv / aivs_per_row;
        const int32_t sub = aiv % aivs_per_row;
        const int32_t chunk =
            (((valid_vocab + aivs_per_row - 1) / aivs_per_row) + 63) & ~63;
        const int32_t col_begin = sub * chunk;
        const int32_t col_end =
            (col_begin + chunk < valid_vocab) ? (col_begin + chunk) : valid_vocab;
        const int32_t len = (r_idx < eff && col_end > col_begin) ? (col_end - col_begin) : 0;

        TBuf<TPosition::VECCALC> b_x, b_blk, b_work, b_out, b_all, b_toks;
        pipe.InitBuffer(b_x, 4096 * 4);
        pipe.InitBuffer(b_blk, 128 * 4);
        pipe.InitBuffer(b_work, 48 * 16 * 4);
        pipe.InitBuffer(b_out, 128 * 4);
        pipe.InitBuffer(b_all, 16 * 128 * 4);
        pipe.InitBuffer(b_toks, 64 * 8);

        auto ub_x = b_x.Get<float>();
        auto ub_blk = b_blk.Get<float>();
        auto ub_work = b_work.Get<float>();
        auto ub_out = b_out.Get<float>();
        auto ub_all = b_all.Get<float>();
        auto ub_toks = b_toks.Get<int64_t>();

        GlobalTensor<float> g_logits, g_work;
        GlobalTensor<int64_t> g_out_toks;
        g_logits.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(logits_gm),
            static_cast<uint64_t>(width) * valid_vocab);
        g_work.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(work_gm), 48 * 16);
        g_out_toks.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(out_tokens_gm), width);

        bool has_best = false;
        float best_val = 0.0F;
        int32_t best_idx = 0;
        for (int32_t off = 0; off < len; off += 4096) {
            const int32_t tile_len = (len - off < 4096) ? (len - off) : 4096;
            const int32_t tile_a64 = (tile_len + 63) & ~63;
            DataCopyPad(
                ub_x,
                g_logits[static_cast<int64_t>(r_idx) * valid_vocab + col_begin + off],
                DataCopyExtParams{1, static_cast<uint32_t>(tile_len * 4), 0, 0, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
            Fence<HardEvent::MTE2_S>();
            for (int32_t pad_j = tile_len; pad_j < tile_a64; ++pad_j) {
                ub_x.SetValue(pad_j, -3.4028235e38F);
            }
            Fence<HardEvent::S_V>();
            const int32_t n_rep = tile_a64 / 64;
            WholeReduceMax<float>(ub_blk, ub_x, 64, n_rep, 1, 1, 8);
            Fence<HardEvent::V_S>();
            for (int32_t b = 0; b < n_rep; ++b) {
                const float b_max = ub_blk.GetValue(2 * b);
                if (!has_best || b_max > best_val) {
                    const int32_t j_end =
                        ((b + 1) * 64 < tile_len) ? ((b + 1) * 64) : tile_len;
                    for (int32_t j = b * 64; j < j_end; ++j) {
                        const float v = ub_x.GetValue(j);
                        const int32_t gi = vocab_start + col_begin + off + j;
                        if (!has_best || v > best_val || (v == best_val && gi < best_idx)) {
                            has_best = true;
                            best_val = v;
                            best_idx = gi;
                        }
                    }
                }
            }
            Fence<HardEvent::S_MTE2>();
        }

        auto ub_out_i32 = ub_out.ReinterpretCast<int32_t>();
        ub_out.SetValue(0, best_val);
        ub_out_i32.SetValue(1, best_idx);
        ub_out_i32.SetValue(2, has_best ? 1 : 0);
        Fence<HardEvent::S_MTE3>();
        DataCopy(g_work[aiv * 16], ub_out, 16);
        Publish();
        SyncAll<true>();

        if (aiv == 0) {
            DataCopy(ub_work, g_work, 48 * 16);
            Fence<HardEvent::MTE2_S>();
            Duplicate(ub_out, 0.0F, 128);
            PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_S>();
            auto ub_work_i32 = ub_work.ReinterpretCast<int32_t>();
            for (int32_t m = 0; m < eff; ++m) {
                bool r_has = false;
                float r_val = 0.0F;
                int32_t r_idx_best = 0;
                for (int32_t s = 0; s < aivs_per_row; ++s) {
                    const int32_t base = (m * aivs_per_row + s) * 16;
                    if (ub_work_i32.GetValue(base + 2) == 0) {
                        continue;
                    }
                    const float v = ub_work.GetValue(base + 0);
                    const int32_t gi = ub_work_i32.GetValue(base + 1);
                    if (!r_has || v > r_val || (v == r_val && gi < r_idx_best)) {
                        r_has = true;
                        r_val = v;
                        r_idx_best = gi;
                    }
                }
                ub_out.SetValue(m, r_val);
                ub_out_i32.SetValue(64 + m, r_idx_best);
            }
            Fence<HardEvent::S_MTE3>();
            GlobalTensor<float> g_self_slot;
            g_self_slot.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(
                    DspinePayloadPtr(comm_win_gm, epoch, rank)),
                128);
            DataCopy(g_self_slot, ub_out, 128);
            Fence<HardEvent::MTE3_S>();
            DspineWriteSeq(comm_win_gm, epoch);
            Publish();
        }
        SyncAll<true>();

        if (aiv >= 1 && aiv < ranks) {
            const int32_t peer = (rank + aiv) % ranks;
            GlobalTensor<float> g_self_slot, g_rem_slot;
            g_self_slot.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(
                    DspinePayloadPtr(comm_win_gm, epoch, rank)),
                128);
            g_rem_slot.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(
                    aclshmem_ptr(DspinePayloadPtr(comm_win_gm, epoch, rank), peer)),
                128);
            DataCopy(ub_out, g_self_slot, 128);
            Fence<HardEvent::MTE2_MTE3>();
            DataCopy(g_rem_slot, ub_out, 128);
            Fence<HardEvent::MTE3_S>();
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::DDR>();
            __gm__ int32_t* remote_sig = reinterpret_cast<__gm__ int32_t*>(
                aclshmem_ptr(DspineFlagPtr(comm_win_gm, epoch, rank), peer));
            *remote_sig = static_cast<int32_t>(epoch);
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(remote_sig));

            __gm__ volatile int32_t* sig = DspineFlagPtr(comm_win_gm, epoch, peer);
            while (true) {
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(
                    const_cast<__gm__ int32_t*>(sig)));
                if (*sig == static_cast<int32_t>(epoch)) {
                    break;
                }
            }
            Publish();
        }
        SyncAll<true>();

        if (aiv == 0) {
            for (int32_t pe = 0; pe < ranks; ++pe) {
                GlobalTensor<float> g_pe;
                g_pe.SetGlobalBuffer(
                    reinterpret_cast<__gm__ float*>(
                        DspinePayloadPtr(comm_win_gm, epoch, pe)),
                    128);
                DataCopy(ub_all[pe * 128], g_pe, 128);
            }
            Fence<HardEvent::MTE2_S>();
            auto ub_all_i32 = ub_all.ReinterpretCast<int32_t>();
            for (int32_t m = 0; m < eff; ++m) {
                float r_val = ub_all.GetValue(m);
                int32_t r_idx_best = ub_all_i32.GetValue(64 + m);
                for (int32_t pe = 1; pe < ranks; ++pe) {
                    const float v = ub_all.GetValue(pe * 128 + m);
                    const int32_t gi = ub_all_i32.GetValue(pe * 128 + 64 + m);
                    if (v > r_val || (v == r_val && gi < r_idx_best)) {
                        r_val = v;
                        r_idx_best = gi;
                    }
                }
                ub_toks.SetValue(m, static_cast<int64_t>(r_idx_best));
            }
            for (int32_t m = eff; m < width; ++m) {
                ub_toks.SetValue(m, 0LL);
            }
            Fence<HardEvent::S_MTE3>();
            DataCopyPad(
                g_out_toks,
                ub_toks,
                DataCopyExtParams{1, static_cast<uint32_t>(width * 8), 0, 0, 0});
            Fence<HardEvent::MTE3_S>();
            DcciRange(out_tokens_gm, width * 8);
            Publish();
        }
    }
}

__aicore__ inline void RunCommitGather(
    TPipe& pipe,
    uint64_t ctrl_gm,
    uint64_t target_toks_gm,
    uint64_t target_hidden_gm,
    uint64_t ttoks_pad_gm,
    uint64_t par_out_gm,
    int32_t width,
    int32_t m_max)
{
    if ASCEND_IS_AIV {
        TBuf<TPosition::VECCALC> b_ctrl, b_toks, b_bf16;
        pipe.InitBuffer(b_ctrl, 48 * 8);
        pipe.InitBuffer(b_toks, 64 * 8);
        pipe.InitBuffer(b_bf16, 512 * 2);
        auto ub_ctrl = b_ctrl.Get<int64_t>();
        auto ub_toks = b_toks.Get<int64_t>();
        auto ub_bf16 = b_bf16.Get<bfloat16_t>();

        GlobalTensor<int64_t> g_ctrl, g_ttoks, g_ttoks_pad;
        GlobalTensor<bfloat16_t> g_hid, g_par;
        g_ctrl.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ctrl_gm), 48);
        g_ttoks.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(target_toks_gm), width);
        g_ttoks_pad.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ttoks_pad_gm), 32);
        g_hid.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(target_hidden_gm),
            static_cast<int64_t>(width) * 6144);
        g_par.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(par_out_gm),
            static_cast<int64_t>(m_max) * 6144);

        DcciRange(ctrl_gm, 48 * 8);
        DataCopy(ub_ctrl, g_ctrl, 48);
        Fence<HardEvent::MTE2_S>();

        int32_t count = static_cast<int32_t>(ub_ctrl.GetValue(2));
        if (count <= 0) {
            return;
        }
        if (count > m_max) {
            count = m_max;
        }
        const int32_t aiv = GetBlockIdx();
        if (aiv == 0) {
            DcciRange(target_toks_gm, width * 8);
            DataCopyPad(
                ub_toks,
                g_ttoks,
                DataCopyExtParams{1, static_cast<uint32_t>(width * 8), 0, 0, 0},
                DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
            Fence<HardEvent::MTE2_MTE3>();
            DataCopyPad(
                g_ttoks_pad,
                ub_toks,
                DataCopyExtParams{1, static_cast<uint32_t>(width * 8), 0, 0, 0});
            Fence<HardEvent::MTE3_S>();
            DcciRange(ttoks_pad_gm, width * 8);
        }

        for (int32_t task = aiv; task < count * 12; task += 48) {
            const int32_t r_idx = task / 12;
            const int64_t col = static_cast<int64_t>(task % 12) * 512;
            int64_t row_id = ub_ctrl.GetValue(3 + r_idx);
            if (row_id < 0) {
                row_id = 0;
            }
            if (row_id >= width) {
                row_id = width - 1;
            }
            DataCopy(ub_bf16, g_hid[row_id * 6144 + col], 512);
            Fence<HardEvent::MTE2_MTE3>();
            DataCopy(g_par[static_cast<int64_t>(r_idx) * 6144 + col], ub_bf16, 512);
            Fence<HardEvent::MTE3_MTE2>();
        }
        Publish();
    }
}

__aicore__ inline void RunCommitDualNorm(
    TPipe& pipe,
    uint64_t ctrl_gm,
    uint64_t comm_out_gm,
    uint64_t enorm_w_gm,
    uint64_t hnorm_w_gm,
    uint64_t eh_in_gm,
    int32_t m_max,
    float eps)
{
    if ASCEND_IS_AIV {
        TBuf<TPosition::VECCALC> b_ctrl, b_x, b_g, b_xf, b_sq, b_gf, b_sum;
        pipe.InitBuffer(b_ctrl, 48 * 8);
        pipe.InitBuffer(b_x, 6144 * 2);
        pipe.InitBuffer(b_g, 6144 * 2);
        pipe.InitBuffer(b_xf, 6144 * 4);
        pipe.InitBuffer(b_sq, 6144 * 4);
        pipe.InitBuffer(b_gf, 6144 * 4);
        pipe.InitBuffer(b_sum, 64 * 4);

        auto ub_ctrl = b_ctrl.Get<int64_t>();
        auto ub_x = b_x.Get<bfloat16_t>();
        auto ub_g = b_g.Get<bfloat16_t>();
        auto ub_xf = b_xf.Get<float>();
        auto ub_sq = b_sq.Get<float>();
        auto ub_gf = b_gf.Get<float>();
        auto ub_sum = b_sum.Get<float>();

        GlobalTensor<int64_t> g_ctrl;
        GlobalTensor<bfloat16_t> g_in, g_ew, g_hw, g_out;
        g_ctrl.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ctrl_gm), 48);
        g_in.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(comm_out_gm), 2 * m_max * 6144);
        g_ew.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(enorm_w_gm), 6144);
        g_hw.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(hnorm_w_gm), 6144);
        g_out.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(eh_in_gm), m_max * 12288);

        DcciRange(ctrl_gm, 48 * 8);
        DataCopy(ub_ctrl, g_ctrl, 48);
        Fence<HardEvent::MTE2_S>();

        int32_t count = static_cast<int32_t>(ub_ctrl.GetValue(2));
        if (count <= 0) {
            return;
        }
        if (count > m_max) {
            count = m_max;
        }
        const int32_t aiv = GetBlockIdx();
        for (int32_t t = aiv; t < 2 * count; t += 48) {
            const bool is_rot = (t >= count);
            const int32_t r_idx = is_rot ? (t - count) : t;
            const int32_t in_row = is_rot ? (m_max + r_idx) : r_idx;
            const int64_t out_off =
                static_cast<int64_t>(r_idx) * 12288 + (is_rot ? 6144 : 0);
            DataCopy(ub_x, g_in[static_cast<int64_t>(in_row) * 6144], 6144);
            DataCopy(ub_g, is_rot ? g_hw : g_ew, 6144);
            Fence<HardEvent::MTE2_V>();
            Cast(ub_xf, ub_x, RoundMode::CAST_NONE, 6144);
            Cast(ub_gf, ub_g, RoundMode::CAST_NONE, 6144);
            PipeBarrier<PIPE_V>();
            RmsNorm6144(ub_x, ub_xf, ub_sq, ub_gf, ub_sum, eps, false);
            Fence<HardEvent::V_MTE3>();
            DataCopy(g_out[out_off], ub_x, 6144);
            Fence<HardEvent::MTE3_MTE2>();
            Fence<HardEvent::MTE3_V>();
        }
        Publish();
    }
}

__aicore__ inline void RunCommitPostMm2Seed(
    TPipe& pipe,
    uint64_t ctrl_gm,
    uint64_t eh_shard_gm,
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
    uint64_t comm_win_gm,
    int32_t kv_m_cap,
    int32_t rank,
    int32_t ranks)
{
    if ASCEND_IS_AIV {
        if (comm_win_gm == 0ULL) {
            return;
        }
        TBuf<TPosition::VECCALC> b_ctrl, b_in, b_rem, b_rope, b_i64, b_i32;
        pipe.InitBuffer(b_ctrl, 48 * 8);
        pipe.InitBuffer(b_in, 12288 * 2);
        pipe.InitBuffer(b_rem, 12288 * 2);
        pipe.InitBuffer(b_rope, 128 * 2);
        pipe.InitBuffer(b_i64, 32 * 8);
        pipe.InitBuffer(b_i32, 32 * 4);

        auto ub_ctrl = b_ctrl.Get<int64_t>();
        auto ub_in = b_in.Get<bfloat16_t>();
        auto ub_rem = b_rem.Get<bfloat16_t>();
        auto ub_rope = b_rope.Get<bfloat16_t>();
        auto ub_i64 = b_i64.Get<int64_t>();
        auto ub_i32 = b_i32.Get<int32_t>();

        GlobalTensor<int64_t> g_ctrl;
        g_ctrl.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ctrl_gm), 48);
        DcciRange(ctrl_gm, 48 * 8);
        DataCopy(ub_ctrl, g_ctrl, 48);
        Fence<HardEvent::MTE2_S>();

        const int64_t main_start = ub_ctrl.GetValue(0);
        int32_t count = static_cast<int32_t>(ub_ctrl.GetValue(1));
        if (count <= 0) {
            return;
        }
        if (count > kv_m_cap) {
            count = kv_m_cap;
        }
        const int32_t aiv = GetBlockIdx();
        const uint32_t seq = DspineReadSeq(comm_win_gm);
        const uint32_t epoch = seq + 1U;
        SyncAll<true>();

        const int32_t kv_rows = count - 1;
        const int32_t seed_idx = count - 1;
        const int64_t seed_pos = main_start + static_cast<int64_t>(seed_idx);
        const int32_t total_elems = count * kDspineShardDim;
        const uint16_t blk_len = static_cast<uint16_t>(kDspineShardDim * 2 / 32);
        const uint16_t dst_str = static_cast<uint16_t>((6144 - kDspineShardDim) * 2 / 32);

        GlobalTensor<bfloat16_t> g_eh, g_kv_hs, g_seed_hs;
        g_eh.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(eh_shard_gm),
            kv_m_cap * kDspineShardDim);
        g_kv_hs.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(kv_hs_gm), kv_m_cap * 6144);
        g_seed_hs.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(seed_hs_gm), 6144);

        if (aiv == 0) {
            DspineWriteSeq(comm_win_gm, epoch);
            DataCopy(ub_in, g_eh, total_elems);
            Fence<HardEvent::MTE2_MTE3>();
            if (kv_rows > 0) {
                DataCopy(
                    g_kv_hs[static_cast<int64_t>(rank) * kDspineShardDim],
                    ub_in,
                    DataCopyParams{
                        static_cast<uint16_t>(kv_rows), blk_len, 0, dst_str});
            }
            DataCopy(
                g_seed_hs[static_cast<int64_t>(rank) * kDspineShardDim],
                ub_in[seed_idx * kDspineShardDim],
                kDspineShardDim);

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
            Fence<HardEvent::MTE3_S>();
            DcciRange(kv_slot_gm, kv_m_cap * sizeof(int64_t));
            DcciRange(seed_slot_gm, sizeof(int64_t));
            DcciRange(kv_qlen_gm, sizeof(int32_t));
            DcciRange(kv_kvlen_gm, sizeof(int32_t));
            DcciRange(seed_qlen_gm, sizeof(int32_t));
            DcciRange(seed_kvlen_gm, sizeof(int32_t));
        } else if (aiv < ranks) {
            const int32_t peer = (rank + aiv) % ranks;
            GlobalTensor<bfloat16_t> g_rem_slot, g_local_peer;
            g_rem_slot.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t*>(
                    aclshmem_ptr(DspinePayloadPtr(comm_win_gm, epoch, rank), peer)),
                total_elems);
            g_local_peer.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t*>(
                    DspinePayloadPtr(comm_win_gm, epoch, peer)),
                total_elems);
            DataCopy(ub_in, g_eh, total_elems);
            Fence<HardEvent::MTE2_MTE3>();
            DataCopy(g_rem_slot, ub_in, total_elems);
            Fence<HardEvent::MTE3_S>();
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::DDR>();
            __gm__ int32_t* remote_sig = reinterpret_cast<__gm__ int32_t*>(
                aclshmem_ptr(DspineFlagPtr(comm_win_gm, epoch, rank), peer));
            *remote_sig = static_cast<int32_t>(epoch);
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(remote_sig));

            __gm__ volatile int32_t* sig = DspineFlagPtr(comm_win_gm, epoch, peer);
            while (true) {
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(
                    const_cast<__gm__ int32_t*>(sig)));
                if (*sig == static_cast<int32_t>(epoch)) {
                    break;
                }
            }
            DataCopy(ub_rem, g_local_peer, total_elems);
            Fence<HardEvent::MTE2_MTE3>();
            if (kv_rows > 0) {
                DataCopy(
                    g_kv_hs[static_cast<int64_t>(peer) * kDspineShardDim],
                    ub_rem,
                    DataCopyParams{
                        static_cast<uint16_t>(kv_rows), blk_len, 0, dst_str});
            }
            DataCopy(
                g_seed_hs[static_cast<int64_t>(peer) * kDspineShardDim],
                ub_rem[seed_idx * kDspineShardDim],
                kDspineShardDim);
        }

        if (aiv >= 24 && aiv < 24 + kv_rows) {
            GlobalTensor<bfloat16_t> g_sin_c, g_cos_c, g_kv_sin, g_kv_cos;
            g_sin_c.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(sin_cache_gm));
            g_cos_c.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cos_cache_gm));
            g_kv_sin.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t*>(kv_sin_gm), kv_m_cap * 64);
            g_kv_cos.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t*>(kv_cos_gm), kv_m_cap * 64);
            const int32_t r = aiv - 24;
            const int64_t p = main_start + r;
            DataCopy(ub_rope[0], g_sin_c[p * 64], 64);
            DataCopy(ub_rope[64], g_cos_c[p * 64], 64);
            Fence<HardEvent::MTE2_MTE3>();
            DataCopy(g_kv_sin[r * 64], ub_rope[0], 64);
            DataCopy(g_kv_cos[r * 64], ub_rope[64], 64);
        }

        if (aiv == 47) {
            GlobalTensor<bfloat16_t> g_sin_c, g_cos_c, g_s_sin, g_s_cos;
            g_sin_c.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(sin_cache_gm));
            g_cos_c.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cos_cache_gm));
            g_s_sin.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(seed_sin_gm), 64);
            g_s_cos.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(seed_cos_gm), 64);
            DataCopy(ub_rope[0], g_sin_c[seed_pos * 64], 64);
            DataCopy(ub_rope[64], g_cos_c[seed_pos * 64], 64);
            Fence<HardEvent::MTE2_MTE3>();
            DataCopy(g_s_sin, ub_rope[0], 64);
            DataCopy(g_s_cos, ub_rope[64], 64);
        }

        Publish();
    }
}

template <int32_t KHidden>
__aicore__ inline void RunCommitMmK128(
    TPipe& pipe, uint64_t ctrl_gm, uint64_t a_gm_addr, uint64_t b_nz_gm_addr,
    uint64_t c_gm_addr, int32_t m_max)
{
    int32_t eff = m_max;
    if (ctrl_gm != 0ULL) {
        DcciRange(ctrl_gm, 48 * 8);
        eff = static_cast<int32_t>(
            reinterpret_cast<__gm__ const int64_t*>(ctrl_gm)[2]);
    }
    if (eff <= 0) {
        return;
    }
    if (eff > m_max) {
        eff = m_max;
    }
    if ASCEND_IS_AIC {
        constexpr int32_t kTile = 16;
        constexpr int32_t kInner = 128;
        constexpr int32_t kSteps = KHidden / kInner;
        constexpr int32_t kStage = kTile * kInner;
        constexpr int32_t kAcc = kTile * kTile;
        constexpr int32_t kNTotal = dspine_shared::kDspineShardDim / kTile;

        TBuf<TPosition::A1> a_stage;
        TBuf<TPosition::B1> b_stage;
        TBuf<TPosition::A2> a_op;
        TBuf<TPosition::B2> b_op;
        TBuf<TPosition::CO1> c_acc;
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

        GlobalTensor<bfloat16_t> a_gm, b_gm, c_gm;
        a_gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(a_gm_addr),
            static_cast<int64_t>(eff) * KHidden);
        b_gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(b_nz_gm_addr),
            static_cast<int64_t>(kNTotal) * kSteps * kStage);
        c_gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(c_gm_addr),
            static_cast<int64_t>(eff) * dspine_shared::kDspineShardDim);

        const int32_t cid = GetBlockIdx();
        for (int32_t r0 = 0; r0 < eff; r0 += kTile) {
            const int32_t r_valid = (eff - r0 < kTile) ? (eff - r0) : kTile;
            for (int32_t col_tile = cid; col_tile < kNTotal; col_tile += 24) {
                Nd2NzParams copy_a{};
                copy_a.ndNum = 1;
                copy_a.nValue = r_valid;
                copy_a.dValue = kInner;
                copy_a.srcDValue = KHidden;
                copy_a.dstNzC0Stride = kTile;
                copy_a.dstNzNStride = 1;

                LoadData2DParams load_ab{};
                load_ab.repeatTimes = kInner / kTile;
                load_ab.srcStride = 1;
                load_ab.ifTranspose = false;

                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID1);

                for (int32_t s = 0; s < kSteps; ++s) {
                    const int32_t buf = s & 1;
                    const event_t ev = buf ? EVENT_ID1 : EVENT_ID0;
                    const int64_t a_off =
                        static_cast<int64_t>(r0) * KHidden
                        + static_cast<int64_t>(s) * kInner;
                    const int64_t b_off =
                        (static_cast<int64_t>(col_tile) * kSteps + s) * kStage;

                    WaitFlag<HardEvent::MTE1_MTE2>(ev);
                    DataCopy(a_l1[buf * kStage], a_gm[a_off], copy_a);
                    DataCopy(b_l1[buf * kStage], b_gm[b_off], kStage);
                    SetFlag<HardEvent::MTE2_MTE1>(ev);

                    WaitFlag<HardEvent::MTE2_MTE1>(ev);
                    WaitFlag<HardEvent::M_MTE1>(ev);
                    LoadData(a_l0[buf * kStage], a_l1[buf * kStage], load_ab);
                    LoadData(b_l0[buf * kStage], b_l1[buf * kStage], load_ab);
                    SetFlag<HardEvent::MTE1_MTE2>(ev);
                    SetFlag<HardEvent::MTE1_M>(ev);

                    WaitFlag<HardEvent::MTE1_M>(ev);
                    MmadParams mad{};
                    mad.m = kTile;
                    mad.n = kTile;
                    mad.k = kInner;
                    mad.cmatrixInitVal = (s == 0);
                    mad.cmatrixSource = false;
                    Mmad(c_l0, a_l0[buf * kStage], b_l0[buf * kStage], mad);
                    PipeBarrier<PIPE_M>();
                    SetFlag<HardEvent::M_MTE1>(ev);
                }
                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
                WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);

                SetFlag<HardEvent::M_FIX>(EVENT_ID0);
                WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
                FixpipeParamsV220 store{};
                store.nSize = kTile;
                store.mSize = r_valid;
                store.srcStride = kTile;
                store.dstStride = dspine_shared::kDspineShardDim;
                store.ndNum = 1;
                store.quantPre = QuantMode_t::F322BF16;
                const int64_t c_off =
                    static_cast<int64_t>(r0) * dspine_shared::kDspineShardDim
                    + col_tile * kTile;
                Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                    c_gm[c_off], c_l0, store);
                SetFlag<HardEvent::FIX_M>(EVENT_ID0);
                WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
            }
        }
        dspine_shared::Publish();
        pipe.Reset();
    }
}
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void dspine_prolog_pack_k(
    uint64_t ctrl_gm, uint64_t target_toks_gm, uint64_t emb_w_gm,
    uint64_t rot_part_gm, uint64_t comm_in_gm,
    int32_t m_max, int32_t vocab_start, int32_t vocab_end, int32_t rank)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    dspine_ops::RunPrologPack(
        pipe, ctrl_gm, target_toks_gm, emb_w_gm, rot_part_gm, comm_in_gm,
        m_max, vocab_start, vocab_end, rank);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
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
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    dspine_ops::RunPrepCommitSeedV3(
        pipe, ctrl_gm, hs_all_gm, sin_cache_gm, cos_cache_gm,
        kv_hs_gm, kv_sin_gm, kv_cos_gm, kv_slot_gm, kv_qlen_gm, kv_kvlen_gm,
        seed_hs_gm, seed_sin_gm, seed_cos_gm, seed_slot_gm, seed_qlen_gm, seed_kvlen_gm,
        kv_m_cap);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
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
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
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

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
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
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
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
    dspine_prolog_pack_k<<<48, nullptr, stream>>>(
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
    dspine_prep_commit_seed_v3_k<<<48, nullptr, stream>>>(
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
    dspine_beam_step_v3_k<<<48, nullptr, stream>>>(
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
    dspine_build_tree_v3_k<<<48, nullptr, stream>>>(
        sync_base, width, min_w_act, pool_size, num_seq_ptrs, num_q_ptrs, num_rope_dsts,
        t_base_ms, c_w_ms, draft_ctrl_gm,
        pool_scores_gm, pool_raw_lp_gm, pool_tokens_gm, pool_parents_gm, pool_depths_gm,
        tree_tokens_gm, tree_parents_gm, tree_depths_gm, tree_mask_gm,
        t_ids_gm, t_pos_gm, t_slots_gm, t_slots_i64_gm,
        dyn_w_gm, seq_ptrs_gm, q_ptrs_gm, cos_cache_gm, sin_cache_gm, rope_dsts_gm, work_gm);
    return 0;
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void step3c_kv_compact_k(uint64_t descs_gm, uint64_t ctrl_gm, int32_t num_tensors)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    dspine_ops::RunKvCompact(pipe, descs_gm, ctrl_gm, num_tensors);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void step3c_prolog_pack_k(
    uint64_t active_toks_gm, uint64_t emb_w_gm, uint64_t rot_part_gm,
    uint64_t rot_bias_gm, uint64_t comm_in_gm,
    int32_t b_rows, int32_t vocab_start, int32_t vocab_end, int32_t rank)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    dspine_ops::RunStep3cPrologPack(
        pipe, active_toks_gm, emb_w_gm, rot_part_gm, rot_bias_gm, comm_in_gm,
        b_rows, vocab_start, vocab_end, rank);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void step3c_dual_rmsnorm_slice_k(
    uint64_t comm_out_gm, uint64_t enorm_w_gm, uint64_t hnorm_w_gm,
    uint64_t eh_in_gm, int32_t b_rows, int32_t rank, float eps)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    dspine_ops::RunStep3cDualRmsNormSlice(
        pipe, comm_out_gm, enorm_w_gm, hnorm_w_gm, eh_in_gm, b_rows, rank, eps);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void step3c_logits_topk_lse_bf16_k(
    uint64_t sync_base, uint64_t logits_gm, uint64_t work_gm, uint64_t gather_in_gm,
    int32_t m_rows, int32_t row_stride, int32_t valid_vocab, int32_t vocab_start,
    int32_t k, int32_t rank)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
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
    step3c_kv_compact_k<<<48, nullptr, stream>>>(descs_gm, ctrl_gm, num_tensors);
    return 0;
}

extern "C" int tilert_prolog_pack_raw(
    void* stream, uint64_t active_toks_gm, uint64_t emb_w_gm, uint64_t rot_part_gm,
    uint64_t rot_bias_gm, uint64_t comm_in_gm,
    int32_t b_rows, int32_t vocab_start, int32_t vocab_end, int32_t rank)
{
    step3c_prolog_pack_k<<<48, nullptr, stream>>>(
        active_toks_gm, emb_w_gm, rot_part_gm, rot_bias_gm, comm_in_gm,
        b_rows, vocab_start, vocab_end, rank);
    return 0;
}

extern "C" int tilert_dual_rmsnorm_slice_raw(
    void* stream, uint64_t comm_out_gm, uint64_t enorm_w_gm, uint64_t hnorm_w_gm,
    uint64_t eh_in_gm, int32_t b_rows, int32_t rank, float eps)
{
    step3c_dual_rmsnorm_slice_k<<<48, nullptr, stream>>>(
        comm_out_gm, enorm_w_gm, hnorm_w_gm, eh_in_gm, b_rows, rank, eps);
    return 0;
}

extern "C" int tilert_logits_topk_lse_bf16_raw(
    void* stream, uint64_t sync_base, uint64_t logits_gm, uint64_t work_gm, uint64_t gather_in_gm,
    int32_t m_rows, int32_t row_stride, int32_t valid_vocab, int32_t vocab_start,
    int32_t k, int32_t rank)
{
    step3c_logits_topk_lse_bf16_k<<<48, nullptr, stream>>>(
        sync_base, logits_gm, work_gm, gather_in_gm,
        m_rows, row_stride, valid_vocab, vocab_start, k, rank);
    return 0;
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void dspine_prolog_prepare_meta_k(
    uint64_t state_gm,
    uint64_t draft_ctrl_gm,
    uint64_t compact_ctrl_gm,
    uint64_t commit_ctrl_gm,
    uint64_t dyn_w_gm,
    uint64_t mc2_mask_gm,
    uint64_t step_qlen_gm,
    uint64_t step_qlen_ptrs_gm,
    uint64_t step_table_gm,
    uint64_t receipt_ring_gm,
    uint64_t commit_qlen_ptrs_gm,
    int32_t width,
    int32_t max_steps)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    dspine_ops::RunPrologPrepareMeta(
        state_gm, draft_ctrl_gm, compact_ctrl_gm, commit_ctrl_gm,
        dyn_w_gm, mc2_mask_gm, step_qlen_gm, step_qlen_ptrs_gm,
        step_table_gm, receipt_ring_gm, commit_qlen_ptrs_gm, width, max_steps);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void dspine_resolve_and_advance_k(
    uint64_t state_gm,
    uint64_t dyn_w_gm,
    uint64_t tree_tokens_gm,
    uint64_t tree_parents_gm,
    uint64_t target_tokens_gm,
    uint64_t compact_ctrl_gm,
    uint64_t commit_ctrl_gm,
    uint64_t receipt_ring_gm,
    uint64_t receipt_toks_gm,
    int32_t width,
    int32_t commit_m,
    int32_t max_pos,
    int32_t eos_token_id,
    uint64_t alpha_bits,
    uint64_t oma_bits,
    uint64_t mid_thresh_bits,
    uint64_t up_thresh_bits,
    uint64_t down_thresh_bits,
    int32_t streak_acc,
    int32_t drop_acc,
    int32_t jump_acc,
    int32_t num_levels)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    dspine_ops::RunResolveAndAdvance(
        state_gm, dyn_w_gm, tree_tokens_gm, tree_parents_gm, target_tokens_gm,
        compact_ctrl_gm, commit_ctrl_gm, receipt_ring_gm, receipt_toks_gm,
        width, commit_m, max_pos, eos_token_id,
        alpha_bits, oma_bits, mid_thresh_bits, up_thresh_bits, down_thresh_bits,
        streak_acc, drop_acc, jump_acc, num_levels);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void dspine_embed_lookup_k(
    uint64_t t_ids_gm,
    uint64_t q_len_gm,
    uint64_t emb_w_gm,
    uint64_t partial_out_gm,
    int32_t width,
    int32_t hidden_dim,
    int32_t vocab_start,
    int32_t vocab_end)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    dspine_ops::RunEmbedLookup(
        pipe, t_ids_gm, q_len_gm, emb_w_gm, partial_out_gm,
        width, hidden_dim, vocab_start, vocab_end);
}

extern "C" __global__ __mix__(1, 2) __schedmode__(1) __aicore__
void dspine_exact_index_keys_k(
    uint64_t sync_base,
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
    float eps,
    int32_t is_neox)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    AscendC::SetSyncBaseAddr(sync_base);
    AscendC::TPipe pipe;
    dspine_ops::RunExactIndexKeys(
        pipe, norm_x_gm, q_len_gm, slot_i64_gm, cos_gm, sin_gm,
        wk_w_gm, k_norm_w_gm, k_norm_b_gm, k_cache_gm, work_gm,
        rows_cap, eps, is_neox);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void dspine_step3c_prolog_pack_guarded_k(
    uint64_t q_len_gm,
    uint64_t active_toks_gm,
    uint64_t emb_w_gm,
    uint64_t rot_part_gm,
    uint64_t rot_bias_gm,
    uint64_t comm_in_gm,
    int32_t b_rows,
    int32_t vocab_start,
    int32_t vocab_end,
    int32_t rank)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    dspine_ops::RunStep3cPrologPackGuarded(
        pipe, q_len_gm, active_toks_gm, emb_w_gm, rot_part_gm, rot_bias_gm, comm_in_gm,
        b_rows, vocab_start, vocab_end, rank);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void dspine_step3c_dual_rmsnorm_slice_guarded_k(
    uint64_t q_len_gm,
    uint64_t comm_out_gm,
    uint64_t enorm_w_gm,
    uint64_t hnorm_w_gm,
    uint64_t eh_in_gm,
    int32_t b_rows,
    int32_t rank,
    float eps)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    dspine_ops::RunStep3cDualRmsNormSliceGuarded(
        pipe, q_len_gm, comm_out_gm, enorm_w_gm, hnorm_w_gm, eh_in_gm, b_rows, rank, eps);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void dspine_logits_topk_lse_bf16_guarded_k(
    uint64_t sync_base,
    uint64_t q_len_gm,
    uint64_t logits_gm,
    uint64_t work_gm,
    uint64_t gather_in_gm,
    int32_t m_rows,
    int32_t row_stride,
    int32_t valid_vocab,
    int32_t vocab_start,
    int32_t k,
    int32_t rank)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    AscendC::SetSyncBaseAddr(sync_base);
    AscendC::TPipe pipe;
    dspine_ops::RunLogitsTopkLseGuarded(
        pipe, q_len_gm, logits_gm, work_gm, gather_in_gm,
        m_rows, row_stride, valid_vocab, vocab_start, k, rank);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void dspine_add_rmsnorm_quant_k(
    uint64_t x_gm,
    uint64_t res_gm,
    uint64_t norm_w_gm,
    uint64_t y_out_gm,
    uint64_t q8_out_gm,
    uint64_t scale_out_gm,
    uint64_t q_len_gm,
    int32_t rows_cap,
    int32_t hidden_dim,
    float eps)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    dspine_ops::RunAddRmsNormQuant(
        pipe, x_gm, res_gm, norm_w_gm, y_out_gm, q8_out_gm, scale_out_gm,
        q_len_gm, rows_cap, hidden_dim, eps);
}

extern "C" int tilert_dspine_prolog_prepare_meta_raw(
    void* stream, uint64_t state_gm, uint64_t draft_ctrl_gm, uint64_t compact_ctrl_gm,
    uint64_t commit_ctrl_gm, uint64_t dyn_w_gm, uint64_t mc2_mask_gm,
    uint64_t step_qlen_gm, uint64_t step_qlen_ptrs_gm, uint64_t step_table_gm,
    uint64_t receipt_ring_gm, uint64_t commit_qlen_ptrs_gm, int32_t width, int32_t max_steps)
{
    dspine_prolog_prepare_meta_k<<<1, nullptr, stream>>>(
        state_gm, draft_ctrl_gm, compact_ctrl_gm, commit_ctrl_gm,
        dyn_w_gm, mc2_mask_gm, step_qlen_gm, step_qlen_ptrs_gm,
        step_table_gm, receipt_ring_gm, commit_qlen_ptrs_gm, width, max_steps);
    return 0;
}

extern "C" int tilert_dspine_resolve_and_advance_raw(
    void* stream, uint64_t state_gm, uint64_t dyn_w_gm, uint64_t tree_tokens_gm,
    uint64_t tree_parents_gm, uint64_t target_tokens_gm, uint64_t compact_ctrl_gm,
    uint64_t commit_ctrl_gm, uint64_t receipt_ring_gm, uint64_t receipt_toks_gm,
    int32_t width, int32_t commit_m, int32_t max_pos, int32_t eos_token_id,
    uint64_t alpha_bits, uint64_t oma_bits, uint64_t mid_thresh_bits,
    uint64_t up_thresh_bits, uint64_t down_thresh_bits,
    int32_t streak_acc, int32_t drop_acc, int32_t jump_acc, int32_t num_levels)
{
    dspine_resolve_and_advance_k<<<1, nullptr, stream>>>(
        state_gm, dyn_w_gm, tree_tokens_gm, tree_parents_gm, target_tokens_gm,
        compact_ctrl_gm, commit_ctrl_gm, receipt_ring_gm, receipt_toks_gm,
        width, commit_m, max_pos, eos_token_id,
        alpha_bits, oma_bits, mid_thresh_bits, up_thresh_bits, down_thresh_bits,
        streak_acc, drop_acc, jump_acc, num_levels);
    return 0;
}

extern "C" int tilert_dspine_embed_lookup_raw(
    void* stream, uint64_t t_ids_gm, uint64_t q_len_gm, uint64_t emb_w_gm,
    uint64_t partial_out_gm, int32_t width, int32_t hidden_dim,
    int32_t vocab_start, int32_t vocab_end)
{
    dspine_embed_lookup_k<<<48, nullptr, stream>>>(
        t_ids_gm, q_len_gm, emb_w_gm, partial_out_gm,
        width, hidden_dim, vocab_start, vocab_end);
    return 0;
}

extern "C" int tilert_dspine_exact_index_keys_raw(
    void* stream, uint64_t sync_base, uint64_t norm_x_gm, uint64_t q_len_gm,
    uint64_t slot_i64_gm, uint64_t cos_gm, uint64_t sin_gm, uint64_t wk_w_gm,
    uint64_t k_norm_w_gm, uint64_t k_norm_b_gm, uint64_t k_cache_gm, uint64_t work_gm,
    int32_t rows_cap, float eps, int32_t is_neox)
{
    dspine_exact_index_keys_k<<<24, nullptr, stream>>>(
        sync_base, norm_x_gm, q_len_gm, slot_i64_gm, cos_gm, sin_gm,
        wk_w_gm, k_norm_w_gm, k_norm_b_gm, k_cache_gm, work_gm,
        rows_cap, eps, is_neox);
    return 0;
}

extern "C" int tilert_step3c_prolog_pack_guarded_raw(
    void* stream, uint64_t q_len_gm, uint64_t active_toks_gm, uint64_t emb_w_gm,
    uint64_t rot_part_gm, uint64_t rot_bias_gm, uint64_t comm_in_gm,
    int32_t b_rows, int32_t vocab_start, int32_t vocab_end, int32_t rank)
{
    dspine_step3c_prolog_pack_guarded_k<<<48, nullptr, stream>>>(
        q_len_gm, active_toks_gm, emb_w_gm, rot_part_gm, rot_bias_gm, comm_in_gm,
        b_rows, vocab_start, vocab_end, rank);
    return 0;
}

extern "C" int tilert_step3c_dual_rmsnorm_slice_guarded_raw(
    void* stream, uint64_t q_len_gm, uint64_t comm_out_gm, uint64_t enorm_w_gm,
    uint64_t hnorm_w_gm, uint64_t eh_in_gm, int32_t b_rows, int32_t rank, float eps)
{
    dspine_step3c_dual_rmsnorm_slice_guarded_k<<<48, nullptr, stream>>>(
        q_len_gm, comm_out_gm, enorm_w_gm, hnorm_w_gm, eh_in_gm, b_rows, rank, eps);
    return 0;
}

extern "C" int tilert_logits_topk_lse_bf16_guarded_raw(
    void* stream, uint64_t sync_base, uint64_t q_len_gm, uint64_t logits_gm,
    uint64_t work_gm, uint64_t gather_in_gm, int32_t m_rows, int32_t row_stride,
    int32_t valid_vocab, int32_t vocab_start, int32_t k, int32_t rank)
{
    dspine_logits_topk_lse_bf16_guarded_k<<<48, nullptr, stream>>>(
        sync_base, q_len_gm, logits_gm, work_gm, gather_in_gm,
        m_rows, row_stride, valid_vocab, vocab_start, k, rank);
    return 0;
}

extern "C" int tilert_dspine_add_rmsnorm_quant_raw(
    void* stream, uint64_t x_gm, uint64_t res_gm, uint64_t norm_w_gm, uint64_t y_out_gm,
    uint64_t q8_out_gm, uint64_t scale_out_gm, uint64_t q_len_gm,
    int32_t rows_cap, int32_t hidden_dim, float eps)
{
    dspine_add_rmsnorm_quant_k<<<48, nullptr, stream>>>(
        x_gm, res_gm, norm_w_gm, y_out_gm, q8_out_gm, scale_out_gm,
        q_len_gm, rows_cap, hidden_dim, eps);
    return 0;
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void dspine_prolog_pack_compact_k(
    uint64_t ctrl_gm, uint64_t target_toks_gm, uint64_t emb_w_gm,
    uint64_t rot_part_gm, uint64_t comm_in_gm,
    int32_t m_max, int32_t vocab_start, int32_t vocab_end, int32_t rank)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    dspine_ops::RunPrologPackCompact(
        pipe, ctrl_gm, target_toks_gm, emb_w_gm, rot_part_gm, comm_in_gm,
        m_max, vocab_start, vocab_end, rank);
}

extern "C" int tilert_dspine_prolog_pack_compact_raw(
    void* stream, uint64_t ctrl_gm, uint64_t target_toks_gm, uint64_t emb_w_gm,
    uint64_t rot_part_gm, uint64_t comm_in_gm,
    int32_t m_max, int32_t vocab_start, int32_t vocab_end, int32_t rank)
{
    dspine_prolog_pack_compact_k<<<48, nullptr, stream>>>(
        ctrl_gm, target_toks_gm, emb_w_gm, rot_part_gm, comm_in_gm,
        m_max, vocab_start, vocab_end, rank);
    return 0;
}

extern "C" __global__ __mix__(1, 2) __schedmode__(1) __aicore__
void dspine_draft_head_topk_k(
    uint64_t sync_base,
    uint64_t q_len_gm,
    uint64_t res_a_gm,
    uint64_t res_b_gm,
    uint64_t gamma_gm,
    uint64_t w_q8_nz_gm,
    uint64_t w_s8_gm,
    uint64_t recs_out_gm,
    uint64_t act_q_ws_gm,
    uint64_t act_s_ws_gm,
    uint64_t acc_i32_ws_gm,
    uint64_t work_gm,
    uint64_t gather_in_gm,
    uint64_t gather_out_gm,
    uint64_t comm_win_gm,
    int32_t m_rows,
    int32_t vocab_valid,
    int32_t vocab_start,
    int32_t k,
    int32_t rank,
    int32_t ranks,
    float eps)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    AscendC::SetSyncBaseAddr(sync_base);
    AscendC::TPipe pipe;
    dspine_ops::RunDraftHeadTopk(
        pipe, q_len_gm, res_a_gm, res_b_gm, gamma_gm, w_q8_nz_gm, w_s8_gm,
        recs_out_gm, act_q_ws_gm, act_s_ws_gm, acc_i32_ws_gm, work_gm,
        gather_in_gm, gather_out_gm, comm_win_gm,
        m_rows, vocab_valid, vocab_start, k, rank, ranks, eps);
}

extern "C" int tilert_dspine_draft_head_topk_raw(
    void* stream, uint64_t sync_base, uint64_t q_len_gm,
    uint64_t res_a_gm, uint64_t res_b_gm, uint64_t gamma_gm,
    uint64_t w_q8_nz_gm, uint64_t w_s8_gm, uint64_t recs_out_gm,
    uint64_t act_q_ws_gm, uint64_t act_s_ws_gm, uint64_t acc_i32_ws_gm,
    uint64_t work_gm, uint64_t gather_in_gm, uint64_t gather_out_gm,
    uint64_t comm_win_gm, int32_t m_rows, int32_t vocab_valid,
    int32_t vocab_start, int32_t k, int32_t rank, int32_t ranks, float eps)
{
    dspine_draft_head_topk_k<<<24, nullptr, stream>>>(
        sync_base, q_len_gm, res_a_gm, res_b_gm, gamma_gm, w_q8_nz_gm, w_s8_gm,
        recs_out_gm, act_q_ws_gm, act_s_ws_gm, acc_i32_ws_gm, work_gm,
        gather_in_gm, gather_out_gm, comm_win_gm,
        m_rows, vocab_valid, vocab_start, k, rank, ranks, eps);
    return 0;
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void dspine_target_prolog_k(
    uint64_t sync_base, uint64_t q_len_gm, uint64_t t_ids_gm, uint64_t emb_w_gm,
    uint64_t gamma_gm, uint64_t emb_out_gm, uint64_t l0_normed_gm, uint64_t comm_win_gm,
    int32_t width, int32_t vocab_size, int32_t vocab_start, int32_t vocab_end,
    int32_t rank, int32_t ranks, float eps)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    AscendC::SetSyncBaseAddr(sync_base);
    AscendC::TPipe pipe;
    dspine_ops::RunTargetProlog(
        pipe, q_len_gm, t_ids_gm, emb_w_gm, gamma_gm, emb_out_gm, l0_normed_gm,
        comm_win_gm, width, vocab_size, vocab_start, vocab_end, rank, ranks, eps);
}

extern "C" int tilert_dspine_target_prolog_raw(
    void* stream, uint64_t sync_base, uint64_t q_len_gm, uint64_t t_ids_gm,
    uint64_t emb_w_gm, uint64_t gamma_gm, uint64_t emb_out_gm, uint64_t l0_normed_gm,
    uint64_t comm_win_gm, int32_t width, int32_t vocab_size,
    int32_t vocab_start, int32_t vocab_end, int32_t rank, int32_t ranks, float eps)
{
    dspine_target_prolog_k<<<48, nullptr, stream>>>(
        sync_base, q_len_gm, t_ids_gm, emb_w_gm, gamma_gm, emb_out_gm, l0_normed_gm,
        comm_win_gm, width, vocab_size, vocab_start, vocab_end, rank, ranks, eps);
    return 0;
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void dspine_target_add_rmsnorm_k(
    uint64_t q_len_gm, uint64_t x1_gm, uint64_t x2_gm, uint64_t gamma_gm,
    uint64_t out_gm, int32_t width, float eps)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    dspine_ops::RunTargetAddRmsNorm(
        pipe, q_len_gm, x1_gm, x2_gm, gamma_gm, out_gm, width, eps);
}

extern "C" int tilert_dspine_target_add_rmsnorm_raw(
    void* stream, uint64_t q_len_gm, uint64_t x1_gm, uint64_t x2_gm,
    uint64_t gamma_gm, uint64_t out_gm, int32_t width, float eps)
{
    dspine_target_add_rmsnorm_k<<<48, nullptr, stream>>>(
        q_len_gm, x1_gm, x2_gm, gamma_gm, out_gm, width, eps);
    return 0;
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void dspine_target_argmax_k(
    uint64_t sync_base, uint64_t q_len_gm, uint64_t logits_gm, uint64_t work_gm,
    uint64_t out_tokens_gm, uint64_t comm_win_gm, int32_t width,
    int32_t valid_vocab, int32_t vocab_start, int32_t rank, int32_t ranks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    AscendC::SetSyncBaseAddr(sync_base);
    AscendC::TPipe pipe;
    dspine_ops::RunTargetArgmax(
        pipe, q_len_gm, logits_gm, work_gm, out_tokens_gm, comm_win_gm,
        width, valid_vocab, vocab_start, rank, ranks);
}

extern "C" int tilert_dspine_target_argmax_raw(
    void* stream, uint64_t sync_base, uint64_t q_len_gm, uint64_t logits_gm,
    uint64_t work_gm, uint64_t out_tokens_gm, uint64_t comm_win_gm, int32_t width,
    int32_t valid_vocab, int32_t vocab_start, int32_t rank, int32_t ranks)
{
    dspine_target_argmax_k<<<48, nullptr, stream>>>(
        sync_base, q_len_gm, logits_gm, work_gm, out_tokens_gm, comm_win_gm,
        width, valid_vocab, vocab_start, rank, ranks);
    return 0;
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void dspine_commit_gather_k(
    uint64_t ctrl_gm, uint64_t target_toks_gm, uint64_t target_hidden_gm,
    uint64_t ttoks_pad_gm, uint64_t par_out_gm, int32_t width, int32_t m_max)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    dspine_ops::RunCommitGather(
        pipe, ctrl_gm, target_toks_gm, target_hidden_gm, ttoks_pad_gm,
        par_out_gm, width, m_max);
}

extern "C" int tilert_dspine_commit_gather_raw(
    void* stream, uint64_t ctrl_gm, uint64_t target_toks_gm, uint64_t target_hidden_gm,
    uint64_t ttoks_pad_gm, uint64_t par_out_gm, int32_t width, int32_t m_max)
{
    dspine_commit_gather_k<<<48, nullptr, stream>>>(
        ctrl_gm, target_toks_gm, target_hidden_gm, ttoks_pad_gm,
        par_out_gm, width, m_max);
    return 0;
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void dspine_commit_dual_norm_k(
    uint64_t ctrl_gm, uint64_t comm_out_gm, uint64_t enorm_w_gm, uint64_t hnorm_w_gm,
    uint64_t eh_in_gm, int32_t m_max, float eps)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    dspine_ops::RunCommitDualNorm(
        pipe, ctrl_gm, comm_out_gm, enorm_w_gm, hnorm_w_gm, eh_in_gm, m_max, eps);
}

extern "C" int tilert_dspine_commit_dual_norm_raw(
    void* stream, uint64_t ctrl_gm, uint64_t comm_out_gm, uint64_t enorm_w_gm,
    uint64_t hnorm_w_gm, uint64_t eh_in_gm, int32_t m_max, float eps)
{
    dspine_commit_dual_norm_k<<<48, nullptr, stream>>>(
        ctrl_gm, comm_out_gm, enorm_w_gm, hnorm_w_gm, eh_in_gm, m_max, eps);
    return 0;
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void dspine_commit_post_mm2_seed_k(
    uint64_t sync_base, uint64_t ctrl_gm, uint64_t eh_shard_gm,
    uint64_t sin_cache_gm, uint64_t cos_cache_gm,
    uint64_t kv_hs_gm, uint64_t kv_sin_gm, uint64_t kv_cos_gm,
    uint64_t kv_slot_gm, uint64_t kv_qlen_gm, uint64_t kv_kvlen_gm,
    uint64_t seed_hs_gm, uint64_t seed_sin_gm, uint64_t seed_cos_gm,
    uint64_t seed_slot_gm, uint64_t seed_qlen_gm, uint64_t seed_kvlen_gm,
    uint64_t comm_win_gm, int32_t kv_m_cap, int32_t rank, int32_t ranks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    AscendC::SetSyncBaseAddr(sync_base);
    AscendC::TPipe pipe;
    dspine_ops::RunCommitPostMm2Seed(
        pipe, ctrl_gm, eh_shard_gm, sin_cache_gm, cos_cache_gm,
        kv_hs_gm, kv_sin_gm, kv_cos_gm, kv_slot_gm, kv_qlen_gm, kv_kvlen_gm,
        seed_hs_gm, seed_sin_gm, seed_cos_gm, seed_slot_gm, seed_qlen_gm, seed_kvlen_gm,
        comm_win_gm, kv_m_cap, rank, ranks);
}

extern "C" int tilert_dspine_commit_post_mm2_seed_raw(
    void* stream, uint64_t sync_base, uint64_t ctrl_gm, uint64_t eh_shard_gm,
    uint64_t sin_cache_gm, uint64_t cos_cache_gm,
    uint64_t kv_hs_gm, uint64_t kv_sin_gm, uint64_t kv_cos_gm,
    uint64_t kv_slot_gm, uint64_t kv_qlen_gm, uint64_t kv_kvlen_gm,
    uint64_t seed_hs_gm, uint64_t seed_sin_gm, uint64_t seed_cos_gm,
    uint64_t seed_slot_gm, uint64_t seed_qlen_gm, uint64_t seed_kvlen_gm,
    uint64_t comm_win_gm, int32_t kv_m_cap, int32_t rank, int32_t ranks)
{
    dspine_commit_post_mm2_seed_k<<<48, nullptr, stream>>>(
        sync_base, ctrl_gm, eh_shard_gm, sin_cache_gm, cos_cache_gm,
        kv_hs_gm, kv_sin_gm, kv_cos_gm, kv_slot_gm, kv_qlen_gm, kv_kvlen_gm,
        seed_hs_gm, seed_sin_gm, seed_cos_gm, seed_slot_gm, seed_qlen_gm, seed_kvlen_gm,
        comm_win_gm, kv_m_cap, rank, ranks);
    return 0;
}

template <int32_t KHidden>
__global__ __mix__(1, 2) __schedmode__(1) __aicore__
void dspine_commit_mm_k128_k(
    uint64_t sync_base, uint64_t ctrl_gm, uint64_t a_gm_addr,
    uint64_t b_nz_gm_addr, uint64_t c_gm_addr, int32_t m_max)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    AscendC::SetSyncBaseAddr(sync_base);
    AscendC::TPipe pipe;
    dspine_ops::RunCommitMmK128<KHidden>(
        pipe, ctrl_gm, a_gm_addr, b_nz_gm_addr, c_gm_addr, m_max);
}

extern "C" int tilert_dspine_commit_mm_k128_raw(
    void* stream, uint64_t sync_base, uint64_t ctrl_gm, uint64_t a_gm_addr,
    uint64_t b_nz_gm_addr, uint64_t c_gm_addr, int32_t m_max, int32_t k_hidden)
{
    if (m_max <= 0) {
        return -1;
    }
    if (k_hidden == 6144) {
        dspine_commit_mm_k128_k<6144><<<24, nullptr, stream>>>(
            sync_base, ctrl_gm, a_gm_addr, b_nz_gm_addr, c_gm_addr, m_max);
        return 0;
    }
    if (k_hidden == 12288) {
        dspine_commit_mm_k128_k<12288><<<24, nullptr, stream>>>(
            sync_base, ctrl_gm, a_gm_addr, b_nz_gm_addr, c_gm_addr, m_max);
        return 0;
    }
    return -1;
}
"""

KERNEL_CPP = _DSPINE_HEAD_CPP + "\n" + DSPINE_WALK_SHARED_CPP + "\n" + _DSPINE_BODY_CPP

HOST_CPP = """#include <cstdint>
#include <cstring>
#include "acl/acl.h"
#include "shmem.h"
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
int tilert_dspine_prolog_prepare_meta_raw(
    void* stream, uint64_t state_gm, uint64_t draft_ctrl_gm, uint64_t compact_ctrl_gm,
    uint64_t commit_ctrl_gm, uint64_t dyn_w_gm, uint64_t mc2_mask_gm,
    uint64_t step_qlen_gm, uint64_t step_qlen_ptrs_gm, uint64_t step_table_gm,
    uint64_t receipt_ring_gm, uint64_t commit_qlen_ptrs_gm, int32_t width, int32_t max_steps);
int tilert_dspine_resolve_and_advance_raw(
    void* stream, uint64_t state_gm, uint64_t dyn_w_gm, uint64_t tree_tokens_gm,
    uint64_t tree_parents_gm, uint64_t target_tokens_gm, uint64_t compact_ctrl_gm,
    uint64_t commit_ctrl_gm, uint64_t receipt_ring_gm, uint64_t receipt_toks_gm,
    int32_t width, int32_t commit_m, int32_t max_pos, int32_t eos_token_id,
    uint64_t alpha_bits, uint64_t oma_bits, uint64_t mid_thresh_bits,
    uint64_t up_thresh_bits, uint64_t down_thresh_bits,
    int32_t streak_acc, int32_t drop_acc, int32_t jump_acc, int32_t num_levels);
int tilert_dspine_embed_lookup_raw(
    void* stream, uint64_t t_ids_gm, uint64_t q_len_gm, uint64_t emb_w_gm,
    uint64_t partial_out_gm, int32_t width, int32_t hidden_dim,
    int32_t vocab_start, int32_t vocab_end);
int tilert_dspine_exact_index_keys_raw(
    void* stream, uint64_t sync_base, uint64_t norm_x_gm, uint64_t q_len_gm,
    uint64_t slot_i64_gm, uint64_t cos_gm, uint64_t sin_gm, uint64_t wk_w_gm,
    uint64_t k_norm_w_gm, uint64_t k_norm_b_gm, uint64_t k_cache_gm, uint64_t work_gm,
    int32_t rows_cap, float eps, int32_t is_neox);
int tilert_step3c_prolog_pack_guarded_raw(
    void* stream, uint64_t q_len_gm, uint64_t active_toks_gm, uint64_t emb_w_gm,
    uint64_t rot_part_gm, uint64_t rot_bias_gm, uint64_t comm_in_gm,
    int32_t b_rows, int32_t vocab_start, int32_t vocab_end, int32_t rank);
int tilert_step3c_dual_rmsnorm_slice_guarded_raw(
    void* stream, uint64_t q_len_gm, uint64_t comm_out_gm, uint64_t enorm_w_gm,
    uint64_t hnorm_w_gm, uint64_t eh_in_gm, int32_t b_rows, int32_t rank, float eps);
int tilert_logits_topk_lse_bf16_guarded_raw(
    void* stream, uint64_t sync_base, uint64_t q_len_gm, uint64_t logits_gm,
    uint64_t work_gm, uint64_t gather_in_gm, int32_t m_rows, int32_t row_stride,
    int32_t valid_vocab, int32_t vocab_start, int32_t k, int32_t rank);
int tilert_dspine_add_rmsnorm_quant_raw(
    void* stream, uint64_t x_gm, uint64_t res_gm, uint64_t norm_w_gm, uint64_t y_out_gm,
    uint64_t q8_out_gm, uint64_t scale_out_gm, uint64_t q_len_gm,
    int32_t rows_cap, int32_t hidden_dim, float eps);
int tilert_dspine_prolog_pack_compact_raw(
    void* stream, uint64_t ctrl_gm, uint64_t target_toks_gm, uint64_t emb_w_gm,
    uint64_t rot_part_gm, uint64_t comm_in_gm,
    int32_t m_max, int32_t vocab_start, int32_t vocab_end, int32_t rank);
int tilert_dspine_draft_head_topk_raw(
    void* stream, uint64_t sync_base, uint64_t q_len_gm,
    uint64_t res_a_gm, uint64_t res_b_gm, uint64_t gamma_gm,
    uint64_t w_q8_nz_gm, uint64_t w_s8_gm, uint64_t recs_out_gm,
    uint64_t act_q_ws_gm, uint64_t act_s_ws_gm, uint64_t acc_i32_ws_gm,
    uint64_t work_gm, uint64_t gather_in_gm, uint64_t gather_out_gm,
    uint64_t comm_win_gm, int32_t m_rows, int32_t vocab_valid,
    int32_t vocab_start, int32_t k, int32_t rank, int32_t ranks, float eps);
int tilert_dspine_target_prolog_raw(
    void* stream, uint64_t sync_base, uint64_t q_len_gm, uint64_t t_ids_gm,
    uint64_t emb_w_gm, uint64_t gamma_gm, uint64_t emb_out_gm, uint64_t l0_normed_gm,
    uint64_t comm_win_gm, int32_t width, int32_t vocab_size,
    int32_t vocab_start, int32_t vocab_end, int32_t rank, int32_t ranks, float eps);
int tilert_dspine_target_add_rmsnorm_raw(
    void* stream, uint64_t q_len_gm, uint64_t x1_gm, uint64_t x2_gm,
    uint64_t gamma_gm, uint64_t out_gm, int32_t width, float eps);
int tilert_dspine_target_argmax_raw(
    void* stream, uint64_t sync_base, uint64_t q_len_gm, uint64_t logits_gm,
    uint64_t work_gm, uint64_t out_tokens_gm, uint64_t comm_win_gm, int32_t width,
    int32_t valid_vocab, int32_t vocab_start, int32_t rank, int32_t ranks);
int tilert_dspine_commit_gather_raw(
    void* stream, uint64_t ctrl_gm, uint64_t target_toks_gm, uint64_t target_hidden_gm,
    uint64_t ttoks_pad_gm, uint64_t par_out_gm, int32_t width, int32_t m_max);
int tilert_dspine_commit_dual_norm_raw(
    void* stream, uint64_t ctrl_gm, uint64_t comm_out_gm, uint64_t enorm_w_gm,
    uint64_t hnorm_w_gm, uint64_t eh_in_gm, int32_t m_max, float eps);
int tilert_dspine_commit_post_mm2_seed_raw(
    void* stream, uint64_t sync_base, uint64_t ctrl_gm, uint64_t eh_shard_gm,
    uint64_t sin_cache_gm, uint64_t cos_cache_gm,
    uint64_t kv_hs_gm, uint64_t kv_sin_gm, uint64_t kv_cos_gm,
    uint64_t kv_slot_gm, uint64_t kv_qlen_gm, uint64_t kv_kvlen_gm,
    uint64_t seed_hs_gm, uint64_t seed_sin_gm, uint64_t seed_cos_gm,
    uint64_t seed_slot_gm, uint64_t seed_qlen_gm, uint64_t seed_kvlen_gm,
    uint64_t comm_win_gm, int32_t kv_m_cap, int32_t rank, int32_t ranks);
int tilert_dspine_commit_mm_k128_raw(
    void* stream, uint64_t sync_base, uint64_t ctrl_gm, uint64_t a_gm_addr,
    uint64_t b_nz_gm_addr, uint64_t c_gm_addr, int32_t m_max, int32_t k_hidden);
int32_t rtGetC2cCtrlAddr(uint64_t* address, uint32_t* length);

static inline uint64_t ResolveSyncBase(uint64_t sync_base)
{
    if (sync_base != 0ULL) {
        return sync_base;
    }
    uint64_t ffts = 0ULL;
    uint32_t len = 0U;
    if (rtGetC2cCtrlAddr(&ffts, &len) == 0 && ffts != 0ULL) {
        return ffts;
    }
    return 0ULL;
}

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

int tilert_dspine_prolog_prepare_meta_launch(
    void*, uint64_t state_gm, uint64_t draft_ctrl_gm, uint64_t compact_ctrl_gm,
    uint64_t commit_ctrl_gm, uint64_t dyn_w_gm, uint64_t mc2_mask_gm,
    uint64_t step_qlen_gm, uint64_t step_qlen_ptrs_gm, uint64_t step_table_gm,
    uint64_t receipt_ring_gm, uint64_t commit_qlen_ptrs_gm, int32_t width, int32_t max_steps)
{
    return tilert_dspine_prolog_prepare_meta_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        state_gm, draft_ctrl_gm, compact_ctrl_gm, commit_ctrl_gm,
        dyn_w_gm, mc2_mask_gm, step_qlen_gm, step_qlen_ptrs_gm,
        step_table_gm, receipt_ring_gm, commit_qlen_ptrs_gm, width, max_steps);
}

int tilert_dspine_resolve_and_advance_launch(
    void*, uint64_t state_gm, uint64_t dyn_w_gm, uint64_t tree_tokens_gm,
    uint64_t tree_parents_gm, uint64_t target_tokens_gm, uint64_t compact_ctrl_gm,
    uint64_t commit_ctrl_gm, uint64_t receipt_ring_gm, uint64_t receipt_toks_gm,
    int32_t width, int32_t commit_m, int32_t max_pos, int32_t eos_token_id,
    double alpha, double mid_thresh, double up_thresh, double down_thresh,
    int32_t streak_acc, int32_t drop_acc, int32_t jump_acc, int32_t num_levels)
{
    uint64_t a_b = 0;
    uint64_t oma_b = 0;
    uint64_t m_b = 0;
    uint64_t u_b = 0;
    uint64_t d_b = 0;
    const double oma = 1.0 - alpha;
    std::memcpy(&a_b, &alpha, 8);
    std::memcpy(&oma_b, &oma, 8);
    std::memcpy(&m_b, &mid_thresh, 8);
    std::memcpy(&u_b, &up_thresh, 8);
    std::memcpy(&d_b, &down_thresh, 8);
    return tilert_dspine_resolve_and_advance_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        state_gm, dyn_w_gm, tree_tokens_gm, tree_parents_gm, target_tokens_gm,
        compact_ctrl_gm, commit_ctrl_gm, receipt_ring_gm, receipt_toks_gm,
        width, commit_m, max_pos, eos_token_id,
        a_b, oma_b, m_b, u_b, d_b, streak_acc, drop_acc, jump_acc, num_levels);
}

int tilert_dspine_embed_lookup_launch(
    void*, uint64_t t_ids_gm, uint64_t q_len_gm, uint64_t emb_w_gm,
    uint64_t partial_out_gm, int32_t width, int32_t hidden_dim,
    int32_t vocab_start, int32_t vocab_end)
{
    return tilert_dspine_embed_lookup_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        t_ids_gm, q_len_gm, emb_w_gm, partial_out_gm,
        width, hidden_dim, vocab_start, vocab_end);
}

int tilert_dspine_exact_index_keys_launch(
    void*, uint64_t sync_base, uint64_t norm_x_gm, uint64_t q_len_gm,
    uint64_t slot_i64_gm, uint64_t cos_gm, uint64_t sin_gm, uint64_t wk_w_gm,
    uint64_t k_norm_w_gm, uint64_t k_norm_b_gm, uint64_t k_cache_gm, uint64_t work_gm,
    int32_t rows_cap, float eps, int32_t is_neox)
{
    return tilert_dspine_exact_index_keys_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        ResolveSyncBase(sync_base), norm_x_gm, q_len_gm, slot_i64_gm, cos_gm, sin_gm,
        wk_w_gm, k_norm_w_gm, k_norm_b_gm, k_cache_gm, work_gm,
        rows_cap, eps, is_neox);
}

int tilert_step3c_prolog_pack_guarded_launch(
    void*, uint64_t q_len_gm, uint64_t active_toks_gm, uint64_t emb_w_gm,
    uint64_t rot_part_gm, uint64_t rot_bias_gm, uint64_t comm_in_gm,
    int32_t b_rows, int32_t vocab_start, int32_t vocab_end, int32_t rank)
{
    return tilert_step3c_prolog_pack_guarded_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        q_len_gm, active_toks_gm, emb_w_gm, rot_part_gm, rot_bias_gm, comm_in_gm,
        b_rows, vocab_start, vocab_end, rank);
}

int tilert_step3c_dual_rmsnorm_slice_guarded_launch(
    void*, uint64_t q_len_gm, uint64_t comm_out_gm, uint64_t enorm_w_gm,
    uint64_t hnorm_w_gm, uint64_t eh_in_gm, int32_t b_rows, int32_t rank, float eps)
{
    return tilert_step3c_dual_rmsnorm_slice_guarded_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        q_len_gm, comm_out_gm, enorm_w_gm, hnorm_w_gm, eh_in_gm, b_rows, rank, eps);
}

int tilert_logits_topk_lse_bf16_guarded_launch(
    void*, uint64_t sync_base, uint64_t q_len_gm, uint64_t logits_gm,
    uint64_t work_gm, uint64_t gather_in_gm, int32_t m_rows, int32_t row_stride,
    int32_t valid_vocab, int32_t vocab_start, int32_t k, int32_t rank)
{
    return tilert_logits_topk_lse_bf16_guarded_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        ResolveSyncBase(sync_base), q_len_gm, logits_gm, work_gm, gather_in_gm,
        m_rows, row_stride, valid_vocab, vocab_start, k, rank);
}

int tilert_dspine_add_rmsnorm_quant_launch(
    void*, uint64_t x_gm, uint64_t res_gm, uint64_t norm_w_gm, uint64_t y_out_gm,
    uint64_t q8_out_gm, uint64_t scale_out_gm, uint64_t q_len_gm,
    int32_t rows_cap, int32_t hidden_dim, float eps)
{
    return tilert_dspine_add_rmsnorm_quant_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        x_gm, res_gm, norm_w_gm, y_out_gm, q8_out_gm, scale_out_gm,
        q_len_gm, rows_cap, hidden_dim, eps);
}

int tilert_dspine_prolog_pack_compact_launch(
    void*, uint64_t ctrl_gm, uint64_t target_toks_gm, uint64_t emb_w_gm,
    uint64_t rot_part_gm, uint64_t comm_in_gm,
    int32_t m_max, int32_t vocab_start, int32_t vocab_end, int32_t rank)
{
    return tilert_dspine_prolog_pack_compact_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        ctrl_gm, target_toks_gm, emb_w_gm, rot_part_gm, comm_in_gm,
        m_max, vocab_start, vocab_end, rank);
}

uint64_t tilert_dspine_alloc_comm_win()
{
    if (aclshmemx_init_status() != ACLSHMEM_STATUS_IS_INITIALIZED) {
        return 0ULL;
    }
    constexpr size_t kBytes = 24ULL << 20;
    void* ptr = aclshmem_align(128, kBytes);
    if (ptr == nullptr) {
        return 0ULL;
    }
    if (aclrtMemset(ptr, kBytes, 0, kBytes) != ACL_SUCCESS) {
        aclshmem_free(ptr);
        return 0ULL;
    }
    aclshmem_barrier_all();
    return reinterpret_cast<uint64_t>(ptr);
}

void tilert_dspine_free_comm_win(uint64_t ptr)
{
    if (ptr != 0ULL && aclshmemx_init_status() == ACLSHMEM_STATUS_IS_INITIALIZED) {
        aclshmem_free(reinterpret_cast<void*>(ptr));
    }
}

int tilert_dspine_draft_head_topk_launch(
    void*, uint64_t sync_base, uint64_t q_len_gm,
    uint64_t res_a_gm, uint64_t res_b_gm, uint64_t gamma_gm,
    uint64_t w_q8_nz_gm, uint64_t w_s8_gm, uint64_t recs_out_gm,
    uint64_t act_q_ws_gm, uint64_t act_s_ws_gm, uint64_t acc_i32_ws_gm,
    uint64_t work_gm, uint64_t gather_in_gm, uint64_t gather_out_gm,
    uint64_t comm_win_gm, int32_t m_rows, int32_t vocab_valid,
    int32_t vocab_start, int32_t k, int32_t rank, int32_t ranks, float eps)
{
    return tilert_dspine_draft_head_topk_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        ResolveSyncBase(sync_base), q_len_gm,
        res_a_gm, res_b_gm, gamma_gm, w_q8_nz_gm, w_s8_gm,
        recs_out_gm, act_q_ws_gm, act_s_ws_gm, acc_i32_ws_gm, work_gm,
        gather_in_gm, gather_out_gm, comm_win_gm,
        m_rows, vocab_valid, vocab_start, k, rank, ranks, eps);
}

int tilert_dspine_target_prolog_launch(
    void*, uint64_t sync_base, uint64_t q_len_gm, uint64_t t_ids_gm,
    uint64_t emb_w_gm, uint64_t gamma_gm, uint64_t emb_out_gm, uint64_t l0_normed_gm,
    uint64_t comm_win_gm, int32_t width, int32_t vocab_size,
    int32_t vocab_start, int32_t vocab_end, int32_t rank, int32_t ranks, float eps)
{
    return tilert_dspine_target_prolog_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        ResolveSyncBase(sync_base), q_len_gm, t_ids_gm, emb_w_gm, gamma_gm,
        emb_out_gm, l0_normed_gm, comm_win_gm, width, vocab_size,
        vocab_start, vocab_end, rank, ranks, eps);
}

int tilert_dspine_target_add_rmsnorm_launch(
    void*, uint64_t q_len_gm, uint64_t x1_gm, uint64_t x2_gm,
    uint64_t gamma_gm, uint64_t out_gm, int32_t width, float eps)
{
    return tilert_dspine_target_add_rmsnorm_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        q_len_gm, x1_gm, x2_gm, gamma_gm, out_gm, width, eps);
}

int tilert_dspine_target_argmax_launch(
    void*, uint64_t sync_base, uint64_t q_len_gm, uint64_t logits_gm,
    uint64_t work_gm, uint64_t out_tokens_gm, uint64_t comm_win_gm, int32_t width,
    int32_t valid_vocab, int32_t vocab_start, int32_t rank, int32_t ranks)
{
    return tilert_dspine_target_argmax_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        ResolveSyncBase(sync_base), q_len_gm, logits_gm, work_gm, out_tokens_gm,
        comm_win_gm, width, valid_vocab, vocab_start, rank, ranks);
}

int tilert_dspine_commit_gather_launch(
    void*, uint64_t ctrl_gm, uint64_t target_toks_gm, uint64_t target_hidden_gm,
    uint64_t ttoks_pad_gm, uint64_t par_out_gm, int32_t width, int32_t m_max)
{
    return tilert_dspine_commit_gather_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        ctrl_gm, target_toks_gm, target_hidden_gm, ttoks_pad_gm, par_out_gm,
        width, m_max);
}

int tilert_dspine_commit_dual_norm_launch(
    void*, uint64_t ctrl_gm, uint64_t comm_out_gm, uint64_t enorm_w_gm,
    uint64_t hnorm_w_gm, uint64_t eh_in_gm, int32_t m_max, float eps)
{
    return tilert_dspine_commit_dual_norm_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        ctrl_gm, comm_out_gm, enorm_w_gm, hnorm_w_gm, eh_in_gm, m_max, eps);
}

int tilert_dspine_commit_post_mm2_seed_launch(
    void*, uint64_t sync_base, uint64_t ctrl_gm, uint64_t eh_shard_gm,
    uint64_t sin_cache_gm, uint64_t cos_cache_gm,
    uint64_t kv_hs_gm, uint64_t kv_sin_gm, uint64_t kv_cos_gm,
    uint64_t kv_slot_gm, uint64_t kv_qlen_gm, uint64_t kv_kvlen_gm,
    uint64_t seed_hs_gm, uint64_t seed_sin_gm, uint64_t seed_cos_gm,
    uint64_t seed_slot_gm, uint64_t seed_qlen_gm, uint64_t seed_kvlen_gm,
    uint64_t comm_win_gm, int32_t kv_m_cap, int32_t rank, int32_t ranks)
{
    return tilert_dspine_commit_post_mm2_seed_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        ResolveSyncBase(sync_base), ctrl_gm, eh_shard_gm, sin_cache_gm, cos_cache_gm,
        kv_hs_gm, kv_sin_gm, kv_cos_gm, kv_slot_gm, kv_qlen_gm, kv_kvlen_gm,
        seed_hs_gm, seed_sin_gm, seed_cos_gm, seed_slot_gm, seed_qlen_gm, seed_kvlen_gm,
        comm_win_gm, kv_m_cap, rank, ranks);
}

int tilert_dspine_commit_mm_k128_launch(
    void*, uint64_t sync_base, uint64_t ctrl_gm, uint64_t a_gm_addr,
    uint64_t b_nz_gm_addr, uint64_t c_gm_addr, int32_t m_max, int32_t k_hidden)
{
    return tilert_dspine_commit_mm_k128_raw(
        c10_npu::getCurrentNPUStream().stream(true),
        ResolveSyncBase(sync_base), ctrl_gm, a_gm_addr,
        b_nz_gm_addr, c_gm_addr, m_max, k_hidden);
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


def _dspine_ops_digest(k_flags: Sequence[str], h_flags: Sequence[str]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([list(k_flags), list(h_flags)]).encode("utf-8"))
    h.update(KERNEL_CPP.encode("utf-8"))
    h.update(HOST_CPP.encode("utf-8"))
    return h.hexdigest()


def build_dspine_ops(out_dir: str | Path | None = None, world_size: int = 8) -> Path:
    ws = int(world_size)
    if ws not in (8, 16):
        raise ValueError(f"Unsupported world_size={world_size!r}; expected 8 or 16")
    out = default_build_dir() if out_dir is None else Path(out_dir)
    so_path = out / "tilert_dspine_ops.so"
    manifest_path = out / "dspine_ops.manifest.json"
    k_flags = ("-DL2_CACHE_HINT=1", "-DASCENDC_DUMP=0", f"-DWORLD_SIZE={ws}")
    h_flags = ("-std=c++17", "-O2", "-fPIC", "-D_GLIBCXX_USE_CXX11_ABI=1")
    src_sha = _dspine_ops_digest(k_flags, h_flags)
    if so_path.is_file() and manifest_path.is_file():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if (
            isinstance(cached, dict)
            and cached.get("source_sha256") == src_sha
            and cached.get("sha256") == hashlib.sha256(so_path.read_bytes()).hexdigest()
        ):
            return so_path
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
        *k_flags,
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
        f"{' '.join(shlex.quote(x) for x in k_cmd)} & p1=$!\n"
        f"{' '.join(shlex.quote(x) for x in h_cmd)} & p2=$!\n"
        f"wait \"$p1\" \"$p2\"\n"
        f"{' '.join(shlex.quote(x) for x in link_cmd)}\n"
    )
    subprocess.run(["bash", "-c", script], check=True)
    for obj in (out / "dspine_ops.o", out / "dspine_host.o"):
        if obj.exists():
            obj.unlink()
    full_sha = hashlib.sha256(so_path.read_bytes()).hexdigest()
    manifest = {"source_sha256": src_sha, "sha256": full_sha}
    tmp_man = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    tmp_man.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_man, manifest_path)
    print(f"BUILD_DSPINE_OPS_OK {full_sha[:12]} {so_path}", flush=True)
    return so_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()
    build_dspine_ops(args.out, world_size=args.world_size)
