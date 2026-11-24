#ifndef TILERT_FFN_NORM_JOIN_NATIVE_H
#define TILERT_FFN_NORM_JOIN_NATIVE_H

// Included inside the frozen routed body's helper namespace.
__aicore__ inline void TokenOutputs(
    TPipe& pipe, Workspace& held, __gm__ RoutedProduct* product,
    __gm__ float* correction, __gm__ ScaleStorage* scales,
    __gm__ float* middle_scales, __gm__ int32_t* original_experts,
    __gm__ bfloat16_t* route_output, __gm__ bfloat16_t* routed_output,
    __gm__ bfloat16_t* shared_output, __gm__ bfloat16_t* scaled_output,
    __gm__ bfloat16_t* combined_output, int64_t tokens, int64_t blocks)
{
    using namespace ffn_norm_join;
    static_assert(kHidden == kOutputHidden && kTopK == kOutputTopK);
    static_assert(kOutputTile <= kVectorTile);
    static_assert(sizeof(RoutedProduct) == sizeof(int32_t));
    const int64_t routes = tokens * kTopK;
    if (GetBlockIdx() >= tokens * kOutputTiles) {
        return;
    }
    TBuf<TPosition::VECCALC> metadata;
    pipe.InitBuffer(metadata, kOutputMetadataBytes);
    auto words = metadata.Get<int32_t>();
    auto inverse = words;
    auto expert_ids = words[kOutputRoutes];
    auto activation = words[2 * kOutputRoutes].ReinterpretCast<float>();
    GlobalTensor<int32_t> ids;
    GlobalTensor<float> activation_gm;
    ids.SetGlobalBuffer(original_experts, routes);
    activation_gm.SetGlobalBuffer(middle_scales, routes);
    DataCopy(expert_ids, ids, routes);
    DataCopy(activation, activation_gm, routes);
    Fence<HardEvent::MTE2_S>();
    for (int64_t route = 0; route < routes; ++route) {
        const int64_t original = held.source.GetValue(route) * kTopK
                                 + held.slot.GetValue(route);
        inverse.SetValue(original, int32_t(route));
    }
    // Both the producer and consumer of inverse are scalar UB operations.
    DataSyncBarrier<MemDsbT::UB>();

    GlobalTensor<bfloat16_t> route_values, routed, shared, scaled, combined;
    route_values.SetGlobalBuffer(route_output, routes * kHidden);
    routed.SetGlobalBuffer(routed_output, tokens * kHidden);
    shared.SetGlobalBuffer(shared_output, tokens * kHidden);
    scaled.SetGlobalBuffer(scaled_output, tokens * kHidden);
    combined.SetGlobalBuffer(combined_output, tokens * kHidden);
    auto narrow = held.half_buf.Get<bfloat16_t>();
    constexpr int64_t span = kOutputTile;
    for (int64_t task = GetBlockIdx(); task < tokens * kOutputTiles;
         task += 2 * blocks) {
        const int64_t token = OutputToken(task);
        const int64_t column = OutputColumn(task);
        const int64_t output_offset = token * kHidden + column;
        Duplicate(held.other, 0.0F, span);
        PipeBarrier<PIPE_V>();
        for (int64_t slot = 0; slot < kTopK; ++slot) {
            const int64_t original = token * kTopK + slot;
            const int64_t expert = expert_ids.GetValue(original);
            const int64_t route = inverse.GetValue(original);
            const int64_t begin = held.start.GetValue(expert);
            const int64_t count = held.count.GetValue(expert);
            const int64_t row = ExpertRow(route, begin);
            GlobalTensor<RoutedProduct> dots;
            GlobalTensor<float> bias, weight_scales;
            dots.SetGlobalBuffer(product + 2 * begin * kHidden,
                                 2 * count * kHidden);
            bias.SetGlobalBuffer(correction + expert * kHidden, kHidden);
            weight_scales.SetGlobalBuffer(scales + expert * kHidden, kHidden);
            LoadScale(held, weight_scales, column, span);
            Recombine(held, dots, bias, row, count, kHidden, column, span);
            Muls(held.value, held.value, activation.GetValue(route), span);
            PipeBarrier<PIPE_V>();
            Cast(narrow, held.value, RoundMode::CAST_RINT, span);
            Fence<HardEvent::V_MTE3>();
            DataCopy(route_values[original * kHidden + column], narrow, span);
            Fence<HardEvent::MTE3_V>();

            // Route diagnostics are produced even for a zero probability.
            // Such routes do not participate in the weighted sum, including NaNs.
            const float probability = held.original_weight.GetValue(original);
            if (probability == 0.0F) {
                continue;
            }
            Cast(held.value, narrow, RoundMode::CAST_NONE, span);
            PipeBarrier<PIPE_V>();
            Muls(held.value, held.value, probability, span);
            PipeBarrier<PIPE_V>();
            if (slot == 0) {
                DataCopy(held.other, held.value, span);
            } else {
                Add(held.other, held.other, held.value, span);
            }
            PipeBarrier<PIPE_V>();
        }

        Cast(narrow, held.other, RoundMode::CAST_RINT, span);
        Fence<HardEvent::V_MTE3>();
        DataCopy(routed[output_offset], narrow, span);
        Fence<HardEvent::MTE3_V>();
        Cast(held.value, narrow, RoundMode::CAST_NONE, span);
        PipeBarrier<PIPE_V>();
        Muls(held.value, held.value, 2.5F, span);
        PipeBarrier<PIPE_V>();
        Cast(narrow, held.value, RoundMode::CAST_RINT, span);
        Fence<HardEvent::V_MTE3>();
        DataCopy(scaled[output_offset], narrow, span);
        Fence<HardEvent::MTE3_V>();
        Cast(held.value, narrow, RoundMode::CAST_NONE, span);
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE2>();
        Fence<HardEvent::MTE3_MTE2>();
        DataCopy(narrow, shared[output_offset], span);
        Fence<HardEvent::MTE2_V>();
        Cast(held.work, narrow, RoundMode::CAST_NONE, span);
        PipeBarrier<PIPE_V>();
        Add(held.value, held.value, held.work, span);
        PipeBarrier<PIPE_V>();
        Cast(narrow, held.value, RoundMode::CAST_RINT, span);
        Fence<HardEvent::V_MTE3>();
        DataCopy(combined[output_offset], narrow, span);
        Fence<HardEvent::MTE3_V>();
    }
}

#endif
