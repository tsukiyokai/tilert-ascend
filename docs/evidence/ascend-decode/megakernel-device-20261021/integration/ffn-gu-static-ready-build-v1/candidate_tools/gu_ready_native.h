#ifndef TILERT_FFN_GU_STATIC_READY_PROTOCOL_NATIVE_H
#define TILERT_FFN_GU_STATIC_READY_PROTOCOL_NATIVE_H

#if !defined(ROUTED_ACTIVE_LIST) || !defined(ROUTED_ROW_BALANCED)
#error "Expert-ready activation requires the active-list balanced parent"
#endif

// Included after Activate, inside the routed-body helper namespace.
__aicore__ inline void PublishGate(int64_t& sequence)
{
    if (sequence >= ffn_gu_static_ready::kCredits) {
        CrossCoreWaitFlag(ffn_gu_static_ready::kFreeFlag);
    }
    CrossCoreSetFlag<2, PIPE_FIX>(ffn_gu_static_ready::kReadyFlag);
    ++sequence;
}

__aicore__ inline void DrainGate(int64_t sequence)
{
    const int64_t remaining = sequence < ffn_gu_static_ready::kCredits
        ? sequence : ffn_gu_static_ready::kCredits;
    for (int64_t credit = 0; credit < remaining; ++credit) {
        CrossCoreWaitFlag(ffn_gu_static_ready::kFreeFlag);
    }
}

__aicore__ inline void ConsumeGate(
    Workspace& held, const ffn_gu_static_ready::BodyArguments& arguments,
    int64_t active_count)
{
    for (int64_t ordinal = GetBlockIdx() / 2; ordinal < active_count;
         ordinal += arguments.blocks) {
        const int64_t expert = held.active.GetValue(kActiveHeaderElements + ordinal);
        const int64_t count = held.count.GetValue(expert);
        const int64_t begin = held.start.GetValue(expert);
        CrossCoreWaitFlag(ffn_gu_static_ready::kReadyFlag);
        Activate<true>(held, count, begin,
            (__gm__ RoutedProduct*)arguments.scratch[kGateProduct]
                + GateStorageRow(begin) * kGateUp,
            (__gm__ float*)arguments.weights[1] + expert * kGateUp,
            (__gm__ ScaleStorage*)arguments.weights[2] + expert * kGateUp,
            Nibbles((__gm__ int4b_t*)arguments.scratch[kMiddle],
                    2 * begin * kIntermediate),
            (__gm__ float*)arguments.scratch[kMiddleScale] + begin,
            GetSubBlockIdx(), 2);
        // Both AIVs acknowledge even when one owns no row of a one-route expert.
        Publish();
        CrossCoreSetFlag<2, PIPE_MTE3>(ffn_gu_static_ready::kFreeFlag);
    }
}
#endif
