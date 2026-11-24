// Included inside whole_layer_tree after the buffer and signal declarations.
__aicore__ inline void ReduceCommunicationPayload(
    CommunicationBuffers& buffers, __gm__ uint8_t* window,
    const CommunicationLayout& layout, __gm__ bfloat16_t* output,
    int64_t elements, int32_t ranks, uint32_t epoch,
    int64_t vector_index, int64_t vectors)
{
    const int32_t signal = static_cast<int32_t>(epoch);
    const int64_t tiles = elements / kCommunicationChunk;
    const TreePlan plan = GetTreePlan(ranks);
    for (int64_t tile = vector_index; tile < tiles; tile += vectors) {
        const int64_t begin = tile * kCommunicationChunk;
        int32_t depth = 0;
        for (int32_t leaf = 0; leaf < ranks; ++leaf) {
            const TreeStep step = GetTreeStep(plan, leaf);
            aclshmem_signal_wait_until(
                CommunicationReady(window, layout, ranks, epoch, step.rank),
                ACLSHMEM_CMP_EQ, signal);
            AscendC::GlobalTensor<bfloat16_t> source;
            source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
                window + layout.payload + (epoch & 1U) * layout.plane_bytes
                + step.rank * layout.rank_bytes) + begin, kCommunicationChunk);
            AscendC::DataCopy(buffers.narrow, source, kCommunicationChunk);
            CommunicationFence<AscendC::HardEvent::MTE2_V>();
            AscendC::Cast(buffers.accumulator[depth * kCommunicationChunk],
                          buffers.narrow, AscendC::RoundMode::CAST_NONE,
                          kCommunicationChunk);
            ++depth;
            for (uint32_t merge = 0; merge < step.reductions; ++merge) {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(
                    buffers.accumulator[(depth - 2) * kCommunicationChunk],
                    buffers.accumulator[(depth - 2) * kCommunicationChunk],
                    buffers.accumulator[(depth - 1) * kCommunicationChunk],
                    kCommunicationChunk);
                --depth;
            }
            // A later Cast may reuse a slot consumed by this leaf's additions.
            CommunicationFence<AscendC::HardEvent::V_MTE2>();
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(buffers.narrow, buffers.accumulator,
                      AscendC::RoundMode::CAST_RINT, kCommunicationChunk);
        CommunicationFence<AscendC::HardEvent::V_MTE3>();
        AscendC::GlobalTensor<bfloat16_t> destination;
        destination.SetGlobalBuffer(output + begin, kCommunicationChunk);
        AscendC::DataCopy(destination, buffers.narrow, kCommunicationChunk);
        CommunicationFence<AscendC::HardEvent::MTE3_MTE2>();
    }
}
