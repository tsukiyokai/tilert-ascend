#ifndef TESTS_GLM52_ROUTED_INT4_FRONT_H
#define TESTS_GLM52_ROUTED_INT4_FRONT_H

#include "kernel_operator.h"
#include "routed_int4_contract.h"
#ifdef ROUTED_ACTIVE_LIST
#include "routed_int4_active_list.h"
#endif

namespace routed_int4::front {
using namespace AscendC;

template <HardEvent event>
__aicore__ inline void Fence()
{
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
}

// AIV0 owns all metadata; token%(2*blocks)
// owns each A8 row and its exact four-byte scale store.
template <class Timing>
__aicore__ inline void Prepare(
    TPipe& pipe, DispatchInput input, __gm__ int8_t* rows,
    __gm__ float* scales, __gm__ int32_t* group_count,
    __gm__ int32_t* group_start, __gm__ int32_t* source_row,
    __gm__ int32_t* source_slot, __gm__ float* row_weight,
    int64_t tokens, int64_t blocks, Timing& timing)
{
    if ASCEND_IS_AIV {
        TBuf<TPosition::VECCALC> input_buf, work_buf, output_buf, scale_buf;
        TBuf<TPosition::VECCALC> input_narrow_buf;
        TBuf<TPosition::VECCALC> count_buf, start_buf, cursor_buf, choice_buf;
        TBuf<TPosition::VECCALC> source_buf, slot_buf, probability_buf;
        TBuf<TPosition::VECCALC> weight_buf, narrow_buf;
        pipe.InitBuffer(input_buf, kHidden * sizeof(float));
        pipe.InitBuffer(input_narrow_buf, kHidden * sizeof(bfloat16_t));
        pipe.InitBuffer(work_buf, kHidden * sizeof(float));
        pipe.InitBuffer(output_buf, kHidden);
        pipe.InitBuffer(scale_buf, 32);
        pipe.InitBuffer(count_buf, kMaxExperts * sizeof(int32_t));
        pipe.InitBuffer(start_buf, kMaxExperts * sizeof(int32_t));
        pipe.InitBuffer(cursor_buf, kMaxExperts * sizeof(int32_t));
        pipe.InitBuffer(choice_buf, kMaxRoutes * sizeof(int32_t));
        pipe.InitBuffer(source_buf, kMaxRoutes * sizeof(int32_t));
        pipe.InitBuffer(slot_buf, kMaxRoutes * sizeof(int32_t));
        pipe.InitBuffer(probability_buf, kMaxRoutes * sizeof(float));
        pipe.InitBuffer(weight_buf, kMaxRoutes * sizeof(float));
        pipe.InitBuffer(narrow_buf, kMaxRoutes * sizeof(bfloat16_t));
#ifdef ROUTED_ACTIVE_LIST
        TBuf<TPosition::VECCALC> active_buf;
        pipe.InitBuffer(active_buf, kActiveListBytes);
#endif

        if (GetBlockIdx() == 0) {
            auto count = count_buf.Get<int32_t>();
            auto start = start_buf.Get<int32_t>();
            auto cursor = cursor_buf.Get<int32_t>();
            auto choice = choice_buf.Get<int32_t>();
            auto source = source_buf.Get<int32_t>();
            auto slot = slot_buf.Get<int32_t>();
            auto probability = probability_buf.Get<float>();
            auto weight = weight_buf.Get<float>();
            auto narrow = narrow_buf.Get<bfloat16_t>();
#ifdef ROUTED_ACTIVE_LIST
            auto active = active_buf.Get<int32_t>();
            GlobalTensor<int32_t> active_gm;
            active_gm.SetGlobalBuffer((__gm__ int32_t*)input.active_list,
                                      kActiveListElements);
            Duplicate(active, int32_t(0), kActiveListElements);
#endif
            GlobalTensor<int32_t> choices, count_gm, start_gm, source_gm, slot_gm;
            GlobalTensor<float> probabilities, weight_gm;
            choices.SetGlobalBuffer((__gm__ int32_t*)input.expert_ids, tokens * kTopK);
            probabilities.SetGlobalBuffer((__gm__ float*)input.probabilities,
                                           tokens * kTopK);
            count_gm.SetGlobalBuffer(group_count, kMaxExperts);
            start_gm.SetGlobalBuffer(group_start, kMaxExperts);
            source_gm.SetGlobalBuffer(source_row, kMaxRoutes);
            slot_gm.SetGlobalBuffer(source_slot, kMaxRoutes);
            weight_gm.SetGlobalBuffer(row_weight, kMaxRoutes);
            DataCopy(choice, choices, tokens * kTopK);
            DataCopy(probability, probabilities, tokens * kTopK);
            Duplicate(count, int32_t(0), kMaxExperts);
            Duplicate(source, int32_t(0), kMaxRoutes);
            Duplicate(slot, int32_t(0), kMaxRoutes);
            Duplicate(weight, 0.0F, kMaxRoutes);
            Fence<HardEvent::MTE2_V>();
            Cast(narrow, probability, RoundMode::CAST_RINT, tokens * kTopK);
            PipeBarrier<PIPE_V>();
            Cast(probability, narrow, RoundMode::CAST_NONE, tokens * kTopK);
            Fence<HardEvent::V_S>();
            Fence<HardEvent::MTE2_S>();
            for (int64_t original = 0; original < tokens * kTopK; ++original) {
                const int32_t expert = choice.GetValue(original);
                count.SetValue(expert, count.GetValue(expert) + 1);
            }
            int32_t prefix = 0;
#ifdef ROUTED_ACTIVE_LIST
            int32_t active_count = 0;
#endif
            for (int64_t expert = 0; expert < kMaxExperts; ++expert) {
                start.SetValue(expert, prefix);
                cursor.SetValue(expert, prefix);
                prefix += count.GetValue(expert);
#ifdef ROUTED_ACTIVE_LIST
                active_count = AppendActiveExpert(
                    active, active_count, expert, count.GetValue(expert));
#endif
            }
            for (int64_t original = 0; original < tokens * kTopK; ++original) {
                const int32_t expert = choice.GetValue(original);
                const int32_t position = cursor.GetValue(expert);
                cursor.SetValue(expert, position + 1);
                source.SetValue(position, original / kTopK);
                slot.SetValue(position, original % kTopK);
                weight.SetValue(position, probability.GetValue(original));
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
        }

        auto values = input_buf.Get<float>();
        auto work = work_buf.Get<float>();
        auto output = output_buf.Get<int8_t>();
        auto scale = scale_buf.Get<float>();
        auto bf16_input = input_narrow_buf.Get<bfloat16_t>();
        auto half_output = input_narrow_buf.Get<half>();
        GlobalTensor<bfloat16_t> hidden;
        GlobalTensor<int8_t> output_gm;
        GlobalTensor<float> scale_gm;
        hidden.SetGlobalBuffer((__gm__ bfloat16_t*)input.bf16_rows, tokens * kHidden);
        output_gm.SetGlobalBuffer(rows, tokens * kHidden);
        scale_gm.SetGlobalBuffer(scales, tokens);
        for (int64_t token = GetBlockIdx(); token < tokens; token += 2 * blocks) {
            DataCopy(bf16_input, hidden[token * kHidden], kHidden);
            Fence<HardEvent::MTE2_V>();
            Cast(values, bf16_input, RoundMode::CAST_NONE, kHidden);
            PipeBarrier<PIPE_V>();
            Abs(work, values, kHidden);
            PipeBarrier<PIPE_V>();
            ReduceMax(scale, work, work, kHidden);
            Fence<HardEvent::V_S>();
            const float quantum = scale.GetValue(0) / 127.0F;
            Fence<HardEvent::S_V>();
            Duplicate(scale, quantum, 8);
            Duplicate(work, quantum, kHidden);
            PipeBarrier<PIPE_V>();
            // CANN InitRouting uses Div(x, amax/127), including zero rows.
            Div(work, values, work, kHidden);
            PipeBarrier<PIPE_V>();
            auto integer = work.ReinterpretCast<int32_t>();
            Cast(integer, work, RoundMode::CAST_RINT, kHidden);
            PipeBarrier<PIPE_V>();
            SetDeqScale(half(1.0F));
            Cast(half_output, integer, RoundMode::CAST_ROUND, kHidden);
            PipeBarrier<PIPE_V>();
            Cast(output, half_output, RoundMode::CAST_TRUNC, kHidden);
            Fence<HardEvent::V_MTE3>();
            DataCopy(output_gm[token * kHidden], output, kHidden);
            DataCopyPad(scale_gm[token], scale,
                        DataCopyExtParams{1, sizeof(float), 0, 0, 0});
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::MTE3_MTE2>();
        }
    }
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
    timing.Barrier(0);
    pipe.Reset();
}

}  // namespace routed_int4::front

#endif
