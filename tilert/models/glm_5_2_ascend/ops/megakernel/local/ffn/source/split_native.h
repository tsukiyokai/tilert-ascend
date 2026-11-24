#ifndef TILERT_FFN_TOKEN_MSD_SPLIT_NATIVE_H
#define TILERT_FFN_TOKEN_MSD_SPLIT_NATIVE_H


__aicore__ inline void SplitTokens(
    Workspace& held, __gm__ int8_t* rows, __gm__ int4b_t* stacked,
    __gm__ int32_t* original_experts, int64_t tokens, int64_t blocks, int32_t expert_offset = 0,
    const int8_t* gexp_to_eidx = nullptr,
    const int32_t* raw_r = nullptr, const int32_t* start_arr = nullptr,
    const int32_t* cnt_arr = nullptr, const int32_t* inv_arr = nullptr,
    const int32_t* local_tok_arr = nullptr, int64_t pre_num_local_tokens = -1)
{
    using namespace ffn_original_shared_gu_static;
    static_assert(kHidden == kSplitHidden && kTopK == kSplitTopK);
    static_assert(kVectorTile == kSplitTile);
    static_assert(kSplitMetadataBytes <= (kVectorTile + kSlack) * sizeof(int32_t));
    static_assert(kSplitPackedBytes <= kVectorTile + kSlack);
    static_assert(kSplitPackedBytes <= (kVectorTile + kSlack) / 2 + kSlack);

    auto active_tokens = held.staging[2048];
    const int64_t num_local_tokens = (local_tok_arr != nullptr && pre_num_local_tokens >= 0)
        ? pre_num_local_tokens
        : active_tokens.GetValue(33);
    if (num_local_tokens <= 0) {
        return;
    }
    const int64_t routes = tokens * kTopK;
    auto expert_ids = held.slot;
    auto inverse = held.weight.ReinterpretCast<int32_t>();
    GlobalTensor<int8_t> input;
    input.SetGlobalBuffer(rows, tokens * kHidden);
    auto low_packed = held.input.ReinterpretCast<int4b_t>();
#ifdef K112612_SPLIT_P1_OPT
    GlobalTensor<int8_t> output_i8;
    output_i8.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(stacked), routes * kHidden);
    int64_t tpt = 3;
    int64_t s_tile = 2048;
    if (2 * blocks == 48) {
        if (num_local_tokens == 1) { tpt = 48; s_tile = 128; }
        else if (num_local_tokens == 2) { tpt = 24; s_tile = 256; }
        else if (num_local_tokens == 3) { tpt = 16; s_tile = 384; }
        else if (num_local_tokens == 4) { tpt = 12; s_tile = 512; }
        else if (num_local_tokens <= 6) { tpt = 8; s_tile = 768; }
        else if (num_local_tokens <= 8) { tpt = 6; s_tile = 1024; }
        else if (num_local_tokens <= 12) { tpt = 4; s_tile = 1536; }
        else if (num_local_tokens <= 16) { tpt = 3; s_tile = 2048; }
        else if (num_local_tokens <= 24) { tpt = 4; s_tile = 1536; }
        else { tpt = 3; s_tile = 2048; }
    }
    const int64_t s_packed_bytes = s_tile >> 1;
    const int64_t tasks = num_local_tokens * tpt;
    if (GetBlockIdx() >= tasks) {
        return;
    }
    for (int64_t task = GetBlockIdx(); task < tasks; task += 2 * blocks) {
        const uint32_t ut = static_cast<uint32_t>(task);
        uint32_t qt = 0U;
        if (tpt == 48) { qt = 0U; }
        else if (tpt == 24) { qt = (ut * 683U) >> 14U; }
        else if (tpt == 16) { qt = ut >> 4U; }
        else if (tpt == 12) { qt = (ut * 683U) >> 13U; }
        else if (tpt == 8) { qt = ut >> 3U; }
        else if (tpt == 6) { qt = (ut * 683U) >> 12U; }
        else if (tpt == 4) { qt = ut >> 2U; }
        else { qt = (ut * 171U) >> 9U; }
        const int64_t column = static_cast<int64_t>(ut - qt * static_cast<uint32_t>(tpt)) * s_tile;
        const int64_t token = (local_tok_arr != nullptr)
            ? static_cast<int64_t>(local_tok_arr[qt])
            : active_tokens.GetValue(static_cast<int64_t>(qt));
        Fence<HardEvent::V_MTE2>();
        DataCopy(held.input, input[token * kHidden + column], s_tile);
        Fence<HardEvent::MTE2_V>();
        Cast(held.half_value, held.input, RoundMode::CAST_NONE, s_tile);
        PipeBarrier<PIPE_V>();
        Cast(held.value, held.half_value, RoundMode::CAST_NONE, s_tile);
        PipeBarrier<PIPE_V>();
        SplitHalves(held, s_tile);
        Cast(held.half_value, held.value, RoundMode::CAST_RINT, s_tile);
        PipeBarrier<PIPE_V>();
        Cast(held.packed, held.half_value, RoundMode::CAST_RINT, s_tile);
        PipeBarrier<PIPE_V>();
        Cast(held.half_value, held.work, RoundMode::CAST_RINT, s_tile);
        PipeBarrier<PIPE_V>();
        Cast(low_packed, held.half_value, RoundMode::CAST_RINT, s_tile);
        Fence<HardEvent::V_MTE3>();
        for (int64_t slot = 0; slot < kTopK; ++slot) {
            const int64_t original = token * kTopK + slot;
            const int32_t raw_ge = (raw_r != nullptr) ? raw_r[original] : expert_ids.GetValue(original);
            const int64_t expert = (gexp_to_eidx != nullptr)
                ? ((static_cast<uint32_t>(raw_ge) < 256U) ? static_cast<int64_t>(gexp_to_eidx[raw_ge]) : -1)
                : static_cast<int64_t>(raw_ge - expert_offset);
            if (expert < 0 || expert >= kLocalExperts) {
                continue;
            }
            const int64_t route = (inv_arr != nullptr) ? inv_arr[original] : inverse.GetValue(original);
            const int64_t begin = (start_arr != nullptr) ? start_arr[expert] : held.start.GetValue(expert);
            const int64_t count = (cnt_arr != nullptr) ? cnt_arr[expert] : held.count.GetValue(expert);
            const int64_t high_row = StackedRow(begin, count, route, 0);
            const int64_t low_row = StackedRow(begin, count, route, 1);
            DataCopy(output_i8[(high_row * kHidden + column) >> 1],
                     held.packed.ReinterpretCast<int8_t>(), s_packed_bytes);
            DataCopy(output_i8[(low_row * kHidden + column) >> 1],
                     low_packed.ReinterpretCast<int8_t>(), s_packed_bytes);
        }
        Fence<HardEvent::MTE3_V>();
    }
#else
    const int64_t tasks = num_local_tokens * kSplitTiles;
    if (GetBlockIdx() >= tasks) {
        return;
    }
    GlobalTensor<int4b_t> output;
    output.SetGlobalBuffer(stacked, 2 * routes * kHidden);
    for (int64_t task = GetBlockIdx(); task < tasks; task += 2 * blocks) {
        const int64_t token = (local_tok_arr != nullptr)
            ? static_cast<int64_t>(local_tok_arr[SplitToken(task)])
            : active_tokens.GetValue(SplitToken(task));
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
        Cast(low_packed, held.half_value, RoundMode::CAST_RINT, kSplitTile);
        Fence<HardEvent::V_MTE3>();
        for (int64_t slot = 0; slot < kTopK; ++slot) {
            const int64_t original = token * kTopK + slot;
            const int32_t raw_ge = (raw_r != nullptr) ? raw_r[original] : expert_ids.GetValue(original);
            const int64_t expert = (gexp_to_eidx != nullptr)
                ? ((static_cast<uint32_t>(raw_ge) < 256U) ? static_cast<int64_t>(gexp_to_eidx[raw_ge]) : -1)
                : static_cast<int64_t>(raw_ge - expert_offset);
            if (expert < 0 || expert >= kLocalExperts) {
                continue;
            }
            const int64_t route = (inv_arr != nullptr) ? inv_arr[original] : inverse.GetValue(original);
            const int64_t begin = (start_arr != nullptr) ? start_arr[expert] : held.start.GetValue(expert);
            const int64_t count = (cnt_arr != nullptr) ? cnt_arr[expert] : held.count.GetValue(expert);
            const int64_t high_row = StackedRow(begin, count, route, 0);
            const int64_t low_row = StackedRow(begin, count, route, 1);
            DataCopyPad(output[high_row * kHidden + column], held.packed,
                DataCopyExtParams{1, uint32_t(kSplitPackedBytes), 0, 0, 0});
            DataCopyPad(output[low_row * kHidden + column], low_packed,
                DataCopyExtParams{1, uint32_t(kSplitPackedBytes), 0, 0, 0});
        }
        Fence<HardEvent::MTE3_V>();
    }
#endif
}

#endif
