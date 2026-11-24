#ifndef TESTS_GLM52_ROUTED_INT4_DISPATCH_H
#define TESTS_GLM52_ROUTED_INT4_DISPATCH_H

#include "kernel_operator.h"
#include "routed_int4_contract.h"
#ifdef ROUTED_ACTIVE_LIST
#include "routed_int4_active_list.h"
#endif

namespace routed_int4::dispatch {
using namespace AscendC;

template <HardEvent event>
__aicore__ inline void Fence()
{
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
}

// This phase uses 15904 UB bytes per AIV. All GM destinations have one
// writer: AIV0 owns metadata, and token % (2*blocks) owns a restored row.
template <class Timing>
__aicore__ inline void Restore(
    TPipe& pipe, DispatchInput input, __gm__ int8_t* rows,
    __gm__ float* scales, __gm__ int32_t* group_count,
    __gm__ int32_t* group_start, __gm__ int32_t* source_row,
    __gm__ int32_t* source_slot, __gm__ float* row_weight,
    int64_t tokens, int64_t blocks, Timing& timing)
{
    if ASCEND_IS_AIV {
        TBuf<TPosition::VECCALC> inverse_buf, row_buf, scale_buf;
        TBuf<TPosition::VECCALC> counts64_buf, count_buf, start_buf;
        TBuf<TPosition::VECCALC> source_buf, slot_buf, probability_buf;
        TBuf<TPosition::VECCALC> narrowed_buf, weight_buf;
        pipe.InitBuffer(inverse_buf, kMaxRoutes * sizeof(int32_t));
        pipe.InitBuffer(row_buf, kHidden);
        pipe.InitBuffer(scale_buf, 32);
        pipe.InitBuffer(counts64_buf, kMaxExperts * sizeof(int64_t));
        pipe.InitBuffer(count_buf, kMaxExperts * sizeof(int32_t));
        pipe.InitBuffer(start_buf, kMaxExperts * sizeof(int32_t));
        pipe.InitBuffer(source_buf, kMaxRoutes * sizeof(int32_t));
        pipe.InitBuffer(slot_buf, kMaxRoutes * sizeof(int32_t));
        pipe.InitBuffer(probability_buf, kMaxRoutes * sizeof(float));
        pipe.InitBuffer(narrowed_buf, kMaxRoutes * sizeof(bfloat16_t));
        pipe.InitBuffer(weight_buf, kMaxRoutes * sizeof(float));
#ifdef ROUTED_ACTIVE_LIST
        TBuf<TPosition::VECCALC> active_buf;
        pipe.InitBuffer(active_buf, kActiveListBytes);
#endif
        auto inverse = inverse_buf.Get<int32_t>();
        auto row = row_buf.Get<int8_t>();
        auto scale = scale_buf.Get<float>();
        GlobalTensor<int32_t> inverse_gm;
        inverse_gm.SetGlobalBuffer((__gm__ int32_t*)input.inverse,
                                   tokens * kTopK);
        DataCopy(inverse, inverse_gm, tokens * kTopK);
        Fence<HardEvent::MTE2_S>();

        if (GetBlockIdx() == 0) {
            auto count64 = counts64_buf.Get<int64_t>();
            auto count = count_buf.Get<int32_t>();
            auto start = start_buf.Get<int32_t>();
            auto source = source_buf.Get<int32_t>();
            auto slot = slot_buf.Get<int32_t>();
            auto probability = probability_buf.Get<float>();
            auto narrowed = narrowed_buf.Get<bfloat16_t>();
            auto weight = weight_buf.Get<float>();
#ifdef ROUTED_ACTIVE_LIST
            auto active = active_buf.Get<int32_t>();
            GlobalTensor<int32_t> active_gm;
            active_gm.SetGlobalBuffer((__gm__ int32_t*)input.active_list,
                                      kActiveListElements);
            Duplicate(active, int32_t(0), kActiveListElements);
#endif
            GlobalTensor<int64_t> counts_gm;
            GlobalTensor<float> probability_gm, weight_gm;
            GlobalTensor<int32_t> count_gm, start_gm, source_gm, slot_gm;
            counts_gm.SetGlobalBuffer((__gm__ int64_t*)input.counts, kMaxExperts);
            probability_gm.SetGlobalBuffer((__gm__ float*)input.probabilities,
                                           tokens * kTopK);
            count_gm.SetGlobalBuffer(group_count, kMaxExperts);
            start_gm.SetGlobalBuffer(group_start, kMaxExperts);
            source_gm.SetGlobalBuffer(source_row, kMaxRoutes);
            slot_gm.SetGlobalBuffer(source_slot, kMaxRoutes);
            weight_gm.SetGlobalBuffer(row_weight, kMaxRoutes);
            DataCopy(count64, counts_gm, kMaxExperts);
            DataCopy(probability, probability_gm, tokens * kTopK);
            Fence<HardEvent::MTE2_V>();
            Cast(narrowed, probability, RoundMode::CAST_RINT, tokens * kTopK);
            PipeBarrier<PIPE_V>();
            Cast(probability, narrowed, RoundMode::CAST_NONE, tokens * kTopK);
            Fence<HardEvent::V_S>();
            int32_t prefix = 0;
#ifdef ROUTED_ACTIVE_LIST
            int32_t active_count = 0;
#endif
            for (int64_t expert = 0; expert < kMaxExperts; ++expert) {
                const int32_t size = static_cast<int32_t>(count64.GetValue(expert));
                count.SetValue(expert, size);
                start.SetValue(expert, prefix);
                prefix += size;
#ifdef ROUTED_ACTIVE_LIST
                active_count = AppendActiveExpert(
                    active, active_count, expert, size);
#endif
            }
            for (int64_t sorted = 0; sorted < kMaxRoutes; ++sorted) {
                source.SetValue(sorted, 0);
                slot.SetValue(sorted, 0);
                weight.SetValue(sorted, 0.0F);
            }
            for (int64_t original = 0; original < tokens * kTopK; ++original) {
                const int32_t sorted = inverse.GetValue(original);
                source.SetValue(sorted, original / kTopK);
                slot.SetValue(sorted, original % kTopK);
                weight.SetValue(sorted, probability.GetValue(original));
            }
#ifdef ROUTED_ACTIVE_LIST
            active.SetValue(0, active_count);
#endif
            Fence<HardEvent::S_MTE3>();
#ifdef ROUTED_ACTIVE_LIST
            DataCopy(active_gm, active, kActiveListElements);
#endif
            DataCopy(count_gm, count, kMaxExperts);
            DataCopy(start_gm, start, kMaxExperts);
            DataCopy(source_gm, source, kMaxRoutes);
            DataCopy(slot_gm, slot, kMaxRoutes);
            DataCopy(weight_gm, weight, kMaxRoutes);
            Fence<HardEvent::MTE3_S>();
        }

        GlobalTensor<int8_t> sorted_rows, restored_rows;
        GlobalTensor<float> sorted_scales, restored_scales;
        sorted_rows.SetGlobalBuffer((__gm__ int8_t*)input.rows,
                                    tokens * kTopK * kHidden);
        sorted_scales.SetGlobalBuffer((__gm__ float*)input.scales,
                                      tokens * kTopK);
        restored_rows.SetGlobalBuffer(rows, tokens * kHidden);
        restored_scales.SetGlobalBuffer(scales, tokens);
        for (int64_t token = GetBlockIdx(); token < tokens;
             token += 2 * blocks) {
            const int32_t sorted = inverse.GetValue(token * kTopK);
            DataCopy(row, sorted_rows[sorted * kHidden], kHidden);
            DataCopyPad(scale, sorted_scales[sorted],
                        DataCopyExtParams{1, sizeof(float), 0, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
            Fence<HardEvent::MTE2_MTE3>();
            DataCopy(restored_rows[token * kHidden], row, kHidden);
            // A 4-byte store preserves adjacent tokens owned by other AIVs.
            DataCopyPad(restored_scales[token], scale,
                        DataCopyExtParams{1, sizeof(float), 0, 0, 0});
            Fence<HardEvent::MTE3_MTE2>();
        }
    }
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
    timing.Barrier(0);
    pipe.Reset();
}

}  // namespace routed_int4::dispatch

#endif
