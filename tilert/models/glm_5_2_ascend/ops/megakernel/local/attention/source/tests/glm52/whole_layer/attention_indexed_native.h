#ifndef TESTS_GLM52_WHOLE_LAYER_ATTENTION_INDEXED_NATIVE_H
#define TESTS_GLM52_WHOLE_LAYER_ATTENTION_INDEXED_NATIVE_H

#ifndef ASCENDC_CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif
#include "attention_indexed_sfa.h"
#include "attention_native.h"
#include "indexer_front_native.h"
#include "indexer_native.h"
#include "attention_indexed_contract.h"
#ifdef SFA_PACKED_SHADOW_KV
#include "attention_indexed_shadow.h"
#endif

namespace whole_layer_indexed {

struct DeviceArgs {
    using Address = GM_ADDR;
    Address hidden, token_x, token_scale;
    Address main_weights[9], front_weights[6], tail_weights[3], bias;
    Address main_metadata[6], indexer_metadata[6], caches[kCacheCount];
    Address scratch, tiling, output;
};

struct PhaseTrace {
#ifdef ATTENTION_INDEXED_PROFILE
    int64_t stamps[kTraceMarks];

    __aicore__ inline void Mark(int index)
    {
        stamps[index] = AscendC::GetSystemCycle();
    }

    __aicore__ inline void Store(const Config& c, GM_ADDR memory)
    {
        int64_t kind = 0;
        if ASCEND_IS_AIV {
            kind = 1;
        }
        const int64_t index = AscendC::GetBlockIdx();
        const int64_t slot = kind == 0 ? index : c.attention.blocks + index;
        AscendC::GlobalTensor<int64_t> record;
        record.SetGlobalBuffer((__gm__ int64_t*)(memory + c.trace) +
            slot * kTraceWords, kTraceWords);
        for (uint64_t word = 0; word < kTraceWords; ++word) {
            record.SetValue(word, 0);
        }
        record.SetValue(0, 1);
        record.SetValue(1, kind);
        record.SetValue(2, index);
        record.SetValue(3, c.attention.rows);
        record.SetValue(4, c.attention.blocks);
        for (uint64_t mark = 0; mark < kTraceMarks; ++mark) {
            record.SetValue(kTraceStampOffset + mark, stamps[mark]);
        }
        for (uint64_t word = 0; word < kTraceWords; word += 8) {
            AscendC::DataCacheCleanAndInvalid<int64_t,
                AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(record[word]);
        }
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
#else
    __aicore__ inline void Mark(int) {}
    __aicore__ inline void Store(const Config&, GM_ADDR) {}
#endif
};

__aicore__ __attribute__((noinline)) void Forward(AscendC::TPipe& pipe, const Config& c,
                               const DeviceArgs& a, int bias_kind,
                               GM_ADDR shared_topk = nullptr,
                               __gm__ uint8_t* comm_window = nullptr, int64_t comm_capacity = 0, int32_t comm_rank = 0, int32_t comm_ranks = 0,
                               uint64_t tree_mask_gm = 0, __gm__ int64_t* attn_prof = nullptr,
                               GM_ADDR resolved_topk_buf = nullptr, bool* out_resolved_ok = nullptr,
                               int32_t known_kv_len = -1, int32_t known_q_len = -1,
                               uint64_t fused_rtk_hdr = 0ULL)
{
    GM_ADDR memory = a.scratch;
    GM_ADDR topk_out = (shared_topk != nullptr) ? shared_topk : (memory + c.topk);
    PhaseTrace trace;
    trace.Mark(0);
    int32_t q_len = known_q_len;
    if (q_len < 0) {
        AscendC::GlobalTensor<int32_t> query_lengths;
        query_lengths.SetGlobalBuffer((__gm__ int32_t*)a.main_metadata[3], 1);
        AscendC::DataCacheCleanAndInvalid<int32_t,
            AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(query_lengths);
        q_len = query_lengths.GetValue(0);
    }
    if (q_len <= 0) {
        return;
    }
    const uint32_t sfa_rows =
        (q_len <= static_cast<int32_t>(c.attention.rows))
            ? static_cast<uint32_t>(q_len)
            : c.attention.rows;
    whole_layer_attention::Config c_attn = c.attention;
    c_attn.rows = sfa_rows;
    indexer_front::Config c_front = c.front;
    c_front.rows = sfa_rows;
    whole_layer_indexer::Config c_indexer = c.indexer;
    c_indexer.rows = sfa_rows;

    const whole_layer_attention::DeviceArgs attention{
        a.token_x, a.token_scale, a.main_weights[0], a.main_weights[1],
        a.main_weights[2], a.main_weights[3], a.main_weights[4],
        a.main_weights[5], a.main_weights[6], a.main_weights[7],
        a.main_weights[8], a.main_metadata[0], a.main_metadata[1],
        a.main_metadata[2], a.caches[0], a.caches[1], topk_out,
        a.main_metadata[3], a.main_metadata[4], a.main_metadata[5],
        memory + c.attention_offset, memory + c.latent};
#ifdef ENABLE_KERNEL_PROF
    if (attn_prof != nullptr) { if ASCEND_IS_AIV { if (AscendC::GetBlockIdx() == 0) { attn_prof[0] = AscendC::GetSystemCycle(); } } }
#endif
    whole_layer_attention::PrologPhase(pipe, c_attn, attention,
        memory + c.query_norm, memory + c.query_norm_scale);
#ifdef ENABLE_KERNEL_PROF
    if (attn_prof != nullptr) { if ASCEND_IS_AIV { if (AscendC::GetBlockIdx() == 0) { attn_prof[1] = AscendC::GetSystemCycle(); } } }
#endif
    trace.Mark(1);

    const indexer_front::device::DeviceArgs front{
        a.hidden, memory + c.query_norm, memory + c.query_norm_scale,
        a.front_weights[0], a.front_weights[1], a.front_weights[2],
        a.front_weights[3], a.front_weights[4], a.indexer_metadata[1],
        a.indexer_metadata[0], a.front_weights[5], memory + c.front_offset,
        a.tiling, memory + c.query, memory + c.key, memory + c.weights,
        memory + c.query_scale, memory + c.key_scale,
        a.indexer_metadata[2], a.caches[2], a.caches[3],
        static_cast<int64_t>(c_indexer.pages) * whole_layer_indexer::kPage,
        c.attention.rows, q_len};
#ifdef SFA_PACKED_SHADOW_KV
    indexer_front::device::Forward(pipe, c_front, front,
        ShadowRows{a.caches[0], a.caches[1], a.main_metadata[2], a.caches[4]});
#else
    indexer_front::device::Forward(pipe, c_front, front, indexer_front::device::NoInputStage{});
#endif
#ifdef ENABLE_KERNEL_PROF
    if (attn_prof != nullptr) { if ASCEND_IS_AIV { if (AscendC::GetBlockIdx() == 0) { attn_prof[2] = AscendC::GetSystemCycle(); } } }
#endif
    trace.Mark(2);

    const whole_layer_indexer::DeviceArgs indexer{
        memory + c.query, memory + c.key, memory + c.weights,
        memory + c.query_scale, memory + c.key_scale, a.indexer_metadata[2],
        a.caches[2], a.caches[3], a.indexer_metadata[3], a.indexer_metadata[4],
        a.indexer_metadata[5], memory + c.indexer_offset, topk_out};
    GM_ADDR active_resolved_topk = nullptr;
    uint64_t idx_mask_arg = tree_mask_gm;
#if defined(K112162_FUSED_IDX_RTK) && defined(K112130_VEC_RESOLVED_TOPK) && defined(K11090_FAST_SFA_VEC)
    const uint32_t fuse_blks = (known_kv_len > 0) ? ((static_cast<uint32_t>(known_kv_len) + 127U) >> 7U) : 0U;
    const bool did_fuse_rtk =
        (resolved_topk_buf != nullptr) && (fused_rtk_hdr != 0ULL) &&
        c_indexer.quantized && (sfa_rows >= 8U) && (sfa_rows <= 32U) &&
        (c_attn.table_pages > 0U) && (known_kv_len > static_cast<int32_t>(sfa_rows)) &&
        (known_kv_len <= 65536) && (fuse_blks > 0U) && (fuse_blks <= 512U) &&
        (fuse_blks <= c_attn.table_pages);
    if (did_fuse_rtk) {
        idx_mask_arg = fused_rtk_hdr | 1ULL;
    }
#else
    const bool did_fuse_rtk = false;
#endif
    whole_layer_indexer::Forward(pipe, c_indexer, indexer, true, idx_mask_arg);
#if defined(K112130_VEC_RESOLVED_TOPK) && defined(K11090_FAST_SFA_VEC)
    if (did_fuse_rtk) {
        active_resolved_topk = resolved_topk_buf;
        if (out_resolved_ok != nullptr) {
            *out_resolved_ok = true;
        }
    } else if (resolved_topk_buf != nullptr) {
        const bool ok = whole_layer_attention::ResolveTopk2048(
            pipe, topk_out, resolved_topk_buf,
            attention.page_table, attention.kv_lengths,
            sfa_rows, c_attn.table_pages, tree_mask_gm, known_kv_len);
        if (ok) {
            active_resolved_topk = resolved_topk_buf;
        }
        if (out_resolved_ok != nullptr) {
            *out_resolved_ok = ok;
        }
    }
#endif
#ifdef ENABLE_KERNEL_PROF
    if (attn_prof != nullptr) { if ASCEND_IS_AIV { if (AscendC::GetBlockIdx() == 0) { attn_prof[3] = AscendC::GetSystemCycle(); } } }
#endif
    trace.Mark(3);

#ifdef SFA_PACKED_SHADOW_KV
    whole_layer_attention::SfaPhase(pipe, c_attn, attention, a.caches[4], tree_mask_gm, false, nullptr);
#else
    whole_layer_attention::SfaPhase(pipe, c_attn, attention, nullptr, tree_mask_gm, false, active_resolved_topk);
#endif
#ifdef ENABLE_KERNEL_PROF
    if (attn_prof != nullptr) { if ASCEND_IS_AIV { if (AscendC::GetBlockIdx() == 0) { attn_prof[4] = AscendC::GetSystemCycle(); } } }
#endif
    trace.Mark(4);

    const uint32_t tail_epoch = mla_tail::device::Forward<whole_layer_attention::kHeads>(
        pipe, (__gm__ bfloat16_t*)(memory + c.latent),
        (__gm__ bfloat16_t*)a.tail_weights[0], (__gm__ int8_t*)a.tail_weights[1],
        (__gm__ float*)a.tail_weights[2], a.bias, bias_kind,
        memory + c.tail_offset, a.tiling + indexer_front::kTilingBytes,
        (__gm__ bfloat16_t*)a.output, sfa_rows, c.front.capacity,
        c.attention.blocks, c.tail, false,
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
    }
#ifdef ENABLE_KERNEL_PROF
    if (attn_prof != nullptr) { if ASCEND_IS_AIV { if (AscendC::GetBlockIdx() == 0) {
        attn_prof[5] = AscendC::GetSystemCycle();
        AscendC::GlobalTensor<int64_t> ap_gm;
        ap_gm.SetGlobalBuffer(attn_prof, 8);
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(ap_gm[0]);
    } } }
#endif
    trace.Mark(5);
    trace.Store(c, memory);
}

}

#endif
