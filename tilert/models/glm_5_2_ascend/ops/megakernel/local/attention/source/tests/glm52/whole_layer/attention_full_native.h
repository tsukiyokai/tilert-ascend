#ifndef TESTS_GLM52_WHOLE_LAYER_ATTENTION_FULL_NATIVE_H
#define TESTS_GLM52_WHOLE_LAYER_ATTENTION_FULL_NATIVE_H


#ifndef ASCENDC_CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif
#include "attention_native.h"
#include "attention_full_contract.h"
#include "../mla_tail/mla_tail_native.h"

#ifndef K11240_TOP_NOINLINE
#ifdef K11240_OUTLINE_TOP
#define K11240_TOP_NOINLINE __attribute__((noinline))
#else
#define K11240_TOP_NOINLINE inline
#endif
#endif

namespace whole_layer_attention {

__aicore__ K11240_TOP_NOINLINE void FullForward(
    AscendC::TPipe& pipe, const FullConfig& config, DeviceArgs attention,
    GM_ADDR value_weight, GM_ADDR output_weight, GM_ADDR output_scale,
    GM_ADDR bias, int bias_kind, GM_ADDR scratch, GM_ADDR tail_tiling,
    GM_ADDR output,
    __gm__ uint8_t* comm_window = nullptr, int64_t comm_capacity = 0, int32_t comm_rank = 0, int32_t comm_ranks = 0,
    uint64_t tree_mask_gm = 0, __gm__ int64_t* attn_prof = nullptr,
    GM_ADDR resolved_topk_buf = nullptr, bool need_resolve = false, bool* inout_resolved_ok = nullptr,
    int32_t known_kv_len = -1, int32_t known_q_len = -1)
{
    int32_t q_len = known_q_len;
    if (q_len < 0) {
        AscendC::GlobalTensor<int32_t> query_lengths;
        query_lengths.SetGlobalBuffer((__gm__ int32_t*)attention.query_lengths, 1);
        q_len = query_lengths.GetValue(0);
    }
    const uint32_t sfa_rows =
        (q_len > 0 && q_len <= static_cast<int32_t>(config.attention.rows))
            ? static_cast<uint32_t>(q_len)
            : config.attention.rows;
    Config c_attn = config.attention;
    c_attn.rows = sfa_rows;
    GM_ADDR arena = scratch + config.arena_offset;
    attention.scratch = arena;
    attention.output = scratch;
#ifdef ENABLE_KERNEL_PROF
    if (attn_prof != nullptr) { if ASCEND_IS_AIV { if (AscendC::GetBlockIdx() == 0) { attn_prof[0] = AscendC::GetSystemCycle(); } } }
#endif
    PrologPhase(pipe, c_attn, attention, nullptr, nullptr);
#ifdef ENABLE_KERNEL_PROF
    if (attn_prof != nullptr) { if ASCEND_IS_AIV { if (AscendC::GetBlockIdx() == 0) {
        const int64_t t_p = AscendC::GetSystemCycle();
        attn_prof[1] = t_p;
        attn_prof[2] = t_p;
        attn_prof[3] = t_p;
    } } }
#endif
    GM_ADDR active_resolved_topk = nullptr;
#if defined(K112130_VEC_RESOLVED_TOPK) && defined(K11090_FAST_SFA_VEC)
    if (resolved_topk_buf != nullptr && inout_resolved_ok != nullptr) {
        if (need_resolve) {
            *inout_resolved_ok = ResolveTopk2048(
                pipe, attention.topk, resolved_topk_buf,
                attention.page_table, attention.kv_lengths,
                sfa_rows, c_attn.table_pages, tree_mask_gm, known_kv_len);
        }
        if (*inout_resolved_ok) {
            active_resolved_topk = resolved_topk_buf;
        }
    }
#endif
    SfaPhase(pipe, c_attn, attention, nullptr, tree_mask_gm, true, active_resolved_topk);
#ifdef ENABLE_KERNEL_PROF
    if (attn_prof != nullptr) { if ASCEND_IS_AIV { if (AscendC::GetBlockIdx() == 0) { attn_prof[4] = AscendC::GetSystemCycle(); } } }
#endif


    const uint32_t tail_epoch = mla_tail::device::Forward<kHeads>(pipe, (__gm__ bfloat16_t*)scratch,
        (__gm__ bfloat16_t*)value_weight, (__gm__ int8_t*)output_weight,
        (__gm__ float*)output_scale, bias, bias_kind, arena, tail_tiling,
        (__gm__ bfloat16_t*)output, sfa_rows, config.capacity,
        config.attention.blocks, config.tail, false,
        comm_window, comm_capacity, comm_rank, comm_ranks);
    if ASCEND_IS_AIV {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        AscendC::SyncAll<true>();
#ifdef TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H
        if (comm_window != nullptr) {
#ifdef K112120_CLEAN_DCCI
            const uint32_t epoch = tail_epoch;
#else
            const uint32_t epoch = whole_layer_tree::ReadCommunicationEpoch(comm_window);
#endif
            const auto comm_layout = whole_layer_tree::MakeCommunicationLayout(comm_capacity, comm_ranks);
            const int32_t off = static_cast<int32_t>(AscendC::GetBlockIdx());
            if (off > 0 && off < comm_ranks) {
                whole_layer_tree::SignalPeerReady(
                    comm_window, comm_layout, comm_ranks, epoch, comm_rank, (comm_rank + off) & 7);
            }
        }
#endif
        pipe.Reset();
#ifdef ENABLE_KERNEL_PROF
        if (attn_prof != nullptr && AscendC::GetBlockIdx() == 0) {
            attn_prof[5] = AscendC::GetSystemCycle();
            AscendC::GlobalTensor<int64_t> ap_gm;
            ap_gm.SetGlobalBuffer(attn_prof, 8);
            AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(ap_gm[0]);
        }
#endif
    }
}

}

#endif
