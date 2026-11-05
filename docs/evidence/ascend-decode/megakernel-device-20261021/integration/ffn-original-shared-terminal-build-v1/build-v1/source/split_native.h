#ifndef TILERT_FFN_TOKEN_MSD_SPLIT_NATIVE_H
#define TILERT_FFN_TOKEN_MSD_SPLIT_NATIVE_H

// Included inside the frozen routed body's helper namespace.
__aicore__ inline void SplitTokens(
    Workspace& held, __gm__ int8_t* rows, __gm__ int4b_t* stacked,
    __gm__ int32_t* original_experts, int64_t tokens, int64_t blocks)
{
    using namespace ffn_original_shared_terminal;
    static_assert(kHidden == kSplitHidden && kTopK == kSplitTopK);
    static_assert(kVectorTile == kSplitTile);
    static_assert(kSplitMetadataBytes <= (kVectorTile + kSlack) * sizeof(int32_t));
    static_assert(kSplitPackedBytes <= kVectorTile + kSlack);
    static_assert(kSplitPackedBytes <= (kVectorTile + kSlack) / 2 + kSlack);

    const int64_t tasks = tokens * kSplitTiles;
    if (GetBlockIdx() >= tasks) {
        return;
    }
    const int64_t routes = tokens * kTopK;
    auto inverse = held.staging;
    auto expert_ids = held.staging[kSplitMaxRoutes];
    GlobalTensor<int32_t> ids;
    ids.SetGlobalBuffer(original_experts, routes);
    DataCopy(expert_ids, ids, routes);
    Fence<HardEvent::MTE2_S>();
    for (int64_t route = 0; route < routes; ++route) {
        const int64_t original = held.source.GetValue(route) * kTopK
                                 + held.slot.GetValue(route);
        inverse.SetValue(original, int32_t(route));
    }
    DataSyncBarrier<MemDsbT::UB>();

    GlobalTensor<int8_t> input;
    GlobalTensor<int4b_t> output;
    input.SetGlobalBuffer(rows, tokens * kHidden);
    output.SetGlobalBuffer(stacked, 2 * routes * kHidden);
    auto low_packed = held.input.ReinterpretCast<int4b_t>();
    for (int64_t task = GetBlockIdx(); task < tasks; task += 2 * blocks) {
        const int64_t token = SplitToken(task);
        const int64_t column = SplitColumn(task);
        Fence<HardEvent::V_MTE2>();
        DataCopy(held.input, input[token * kHidden + column], kSplitTile);
        Fence<HardEvent::MTE2_V>();
        Cast(held.half_value, held.input, RoundMode::CAST_NONE, kSplitTile);
        PipeBarrier<PIPE_V>();
        Cast(held.value, held.half_value, RoundMode::CAST_NONE, kSplitTile);
        PipeBarrier<PIPE_V>();
        SplitHalves(held, kSplitTile);

        Cast(held.half_value, held.value, RoundMode::CAST_RINT, kSplitTile);
        PipeBarrier<PIPE_V>();
        Cast(held.packed, held.half_value, RoundMode::CAST_RINT, kSplitTile);
        PipeBarrier<PIPE_V>();
        Cast(held.half_value, held.work, RoundMode::CAST_RINT, kSplitTile);
        PipeBarrier<PIPE_V>();
        // The A8 input is fully consumed before its buffer holds packed low bits.
        Cast(low_packed, held.half_value, RoundMode::CAST_RINT, kSplitTile);
        Fence<HardEvent::V_MTE3>();
        for (int64_t slot = 0; slot < kTopK; ++slot) {
            const int64_t original = token * kTopK + slot;
            const int64_t route = inverse.GetValue(original);
            const int64_t expert = expert_ids.GetValue(original);
            const int64_t begin = held.start.GetValue(expert);
            const int64_t count = held.count.GetValue(expert);
            const int64_t high_row = StackedRow(begin, count, route, 0);
            const int64_t low_row = StackedRow(begin, count, route, 1);
            DataCopyPad(output[high_row * kHidden + column], held.packed,
                DataCopyExtParams{1, uint32_t(kSplitPackedBytes), 0, 0, 0});
            DataCopyPad(output[low_row * kHidden + column], low_packed,
                DataCopyExtParams{1, uint32_t(kSplitPackedBytes), 0, 0, 0});
        }
        Fence<HardEvent::MTE3_V>();
        Fence<HardEvent::MTE3_MTE2>();
    }
}

#endif
