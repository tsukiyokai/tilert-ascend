// Included in the candidate namespace after the accepted buffers and fences.
__aicore__ inline __gm__ int32_t* RsagReady(
    __gm__ uint8_t* window, uint64_t base, int32_t ranks,
    uint32_t epoch, int32_t sender)
{
    return reinterpret_cast<__gm__ int32_t*>(window + base
        + ((epoch & 1U) * ranks + sender) * kCommunicationCacheLine);
}

__aicore__ inline void ScatterInputs(
    CommunicationBuffers& buffers, __gm__ uint8_t* window,
    const CommunicationLayout& layout, __gm__ bfloat16_t* partial,
    int64_t rows, int32_t rank, int32_t ranks, uint32_t epoch,
    int64_t vector_index, int64_t vectors)
{
    for (int64_t offset = vector_index; offset < ranks; offset += vectors) {
        const int32_t owner = (rank + offset) % ranks;
        const auto owned = rsag_contract::OwnerRange(rows, ranks, owner);
        AscendC::GlobalTensor<bfloat16_t> source, destination;
        source.SetGlobalBuffer(partial + owned.begin, owned.count);
        destination.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
            window + layout.scatter + (epoch & 1U) * layout.scatter_plane_bytes
            + rank * layout.scatter_slot_bytes), owned.count);
        aclshmemx_mte_put_nbi(destination, source, buffers.narrow,
                              static_cast<uint32_t>(owned.count), owner, EVENT_ID0);
        // The library leaves its last MTE3 outstanding. Drain before UB reuse.
        aclshmem_quiet();
    }
}

__aicore__ inline void ReduceOwnedInputs(
    CommunicationBuffers& buffers, __gm__ uint8_t* window,
    const CommunicationLayout& layout, int64_t rows,
    int32_t rank, int32_t ranks, uint32_t epoch,
    int64_t vector_index, int64_t vectors)
{
    const auto owned = rsag_contract::OwnerRange(rows, ranks, rank);
    const TreePlan plan = GetTreePlan(ranks);
    for (int64_t begin = vector_index * kCommunicationChunk;
         begin < owned.count; begin += vectors * kCommunicationChunk) {
        const int64_t count = owned.count - begin < kCommunicationChunk
            ? owned.count - begin : kCommunicationChunk;
        int32_t depth = 0;
        for (int32_t leaf = 0; leaf < ranks; ++leaf) {
            const TreeStep step = GetTreeStep(plan, leaf);
            aclshmem_signal_wait_until(
                RsagReady(window, layout.scatter_ready, ranks, epoch, step.rank),
                ACLSHMEM_CMP_EQ, static_cast<int32_t>(epoch));
            AscendC::GlobalTensor<bfloat16_t> source;
            source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
                window + layout.scatter + (epoch & 1U) * layout.scatter_plane_bytes
                + step.rank * layout.scatter_slot_bytes) + begin, count);
            AscendC::DataCopy(buffers.narrow, source, count);
            CommunicationFence<AscendC::HardEvent::MTE2_V>();
            AscendC::Cast(buffers.accumulator[depth * kCommunicationChunk],
                          buffers.narrow, AscendC::RoundMode::CAST_NONE, count);
            ++depth;
            for (uint32_t merge = 0; merge < step.reductions; ++merge) {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(
                    buffers.accumulator[(depth - 2) * kCommunicationChunk],
                    buffers.accumulator[(depth - 2) * kCommunicationChunk],
                    buffers.accumulator[(depth - 1) * kCommunicationChunk], count);
                --depth;
            }
            CommunicationFence<AscendC::HardEvent::V_MTE2>();
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(buffers.narrow, buffers.accumulator,
                      AscendC::RoundMode::CAST_RINT, count);
        CommunicationFence<AscendC::HardEvent::V_MTE3>();
        AscendC::GlobalTensor<bfloat16_t> destination;
        destination.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
            window + layout.gather + (epoch & 1U) * layout.gather_plane_bytes)
            + owned.begin + begin, count);
        AscendC::DataCopy(destination, buffers.narrow, count);
        CommunicationFence<AscendC::HardEvent::MTE3_MTE2>();
    }
}

__aicore__ inline void BroadcastOwnedOutput(
    CommunicationBuffers& buffers, __gm__ uint8_t* window,
    const CommunicationLayout& layout, int64_t rows,
    int32_t rank, int32_t ranks, uint32_t epoch,
    int64_t vector_index, int64_t vectors)
{
    const auto owned = rsag_contract::OwnerRange(rows, ranks, rank);
    AscendC::GlobalTensor<bfloat16_t> slice;
    slice.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
        window + layout.gather + (epoch & 1U) * layout.gather_plane_bytes)
        + owned.begin, owned.count);
    for (int64_t offset = vector_index; offset < ranks - 1; offset += vectors) {
        // The self slice is already final. Never put it onto its own source.
        const int32_t peer = (rank + offset + 1) % ranks;
        aclshmemx_mte_put_nbi(slice, slice, buffers.narrow,
                              static_cast<uint32_t>(owned.count), peer, EVENT_ID0);
        aclshmem_quiet();
    }
}

__aicore__ inline void CopyGatheredOutput(
    CommunicationBuffers& buffers, __gm__ uint8_t* window,
    const CommunicationLayout& layout, __gm__ bfloat16_t* output,
    int64_t rows, int32_t ranks, uint32_t epoch,
    int64_t vector_index, int64_t vectors)
{
    const int64_t elements = rows * kCommunicationHidden;
    for (int64_t begin = vector_index * kCommunicationChunk;
         begin < elements; begin += vectors * kCommunicationChunk) {
        const int64_t count = elements - begin < kCommunicationChunk
            ? elements - begin : kCommunicationChunk;
        for (int32_t owner = 0; owner < ranks; ++owner) {
            const auto owned = rsag_contract::OwnerRange(rows, ranks, owner);
            if (owned.begin < begin + count && begin < owned.begin + owned.count) {
                aclshmem_signal_wait_until(
                    RsagReady(window, layout.gather_ready, ranks, epoch, owner),
                    ACLSHMEM_CMP_EQ, static_cast<int32_t>(epoch));
            }
        }
        AscendC::GlobalTensor<bfloat16_t> source, destination;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
            window + layout.gather + (epoch & 1U) * layout.gather_plane_bytes)
            + begin, count);
        destination.SetGlobalBuffer(output + begin, count);
        AscendC::DataCopy(buffers.narrow, source, count);
        CommunicationFence<AscendC::HardEvent::MTE2_MTE3>();
        AscendC::DataCopy(destination, buffers.narrow, count);
        CommunicationFence<AscendC::HardEvent::MTE3_MTE2>();
    }
}

__aicore__ inline void PublishRsagReady(
    __gm__ uint8_t* window, uint64_t base, int32_t rank, int32_t ranks,
    uint32_t epoch)
{
    for (int32_t offset = 0; offset < ranks; ++offset) {
        const int32_t peer = (rank + offset) % ranks;
        aclshmemx_signal_op(RsagReady(window, base, ranks, epoch, rank),
                            static_cast<int32_t>(epoch), ACLSHMEM_SIGNAL_SET, peer);
    }
}

// The same five joins are executed by every resident AIC and AIV.
__aicore__ inline void AllReduceBf16(
    CommunicationBuffers& buffers, __gm__ uint8_t* window,
    __gm__ bfloat16_t* partial, __gm__ bfloat16_t* output,
    int64_t rows, int64_t capacity, int32_t rank, int32_t ranks,
    int64_t vector_index, int64_t vectors)
{
    if ASCEND_IS_AIV {
        if (vector_index == 0) AdvanceCommunicationEpoch(window);
    }
    CommunicationPublish();
    AscendC::SyncAll<false>();  // 1: common local epoch, and input publication.
    const CommunicationLayout layout = MakeCommunicationLayout(capacity, ranks);
    uint32_t epoch = 0;
    if ASCEND_IS_AIV {
        epoch = ReadCommunicationEpoch(window);
        ScatterInputs(buffers, window, layout, partial, rows, rank, ranks,
                      epoch, vector_index, vectors);
    }
    CommunicationPublish();
    AscendC::SyncAll<false>();  // 2: every scatter put completed before ready.
    if ASCEND_IS_AIV {
        if (vector_index == 0)
            PublishRsagReady(window, layout.scatter_ready, rank, ranks, epoch);
        ReduceOwnedInputs(buffers, window, layout, rows, rank, ranks, epoch,
                           vector_index, vectors);
    }
    CommunicationPublish();
    AscendC::SyncAll<false>();  // 3: owned GM results visible to every local AIV.
    if ASCEND_IS_AIV {
        BroadcastOwnedOutput(buffers, window, layout, rows, rank, ranks, epoch,
                              vector_index, vectors);
    }
    CommunicationPublish();
    AscendC::SyncAll<false>();  // 4: every gather put completed before ready.
    if ASCEND_IS_AIV {
        if (vector_index == 0)
            PublishRsagReady(window, layout.gather_ready, rank, ranks, epoch);
        CopyGatheredOutput(buffers, window, layout, output, rows, ranks, epoch,
                            vector_index, vectors);
    }
    CommunicationPublish();
    AscendC::SyncAll<false>();  // 5: final output precedes the caller's consumer.
}
