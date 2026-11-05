#ifndef TESTS_GLM52_WHOLE_LAYER_COMMUNICATION_DEVICE_H
#define TESTS_GLM52_WHOLE_LAYER_COMMUNICATION_DEVICE_H

#include "communication.h"
#include "kernel_operator.h"
#include "shmem.h"

namespace whole_layer {

// Call Init only on AIV, with 20 KiB available in the caller's TPipe arena.
// Finish all Matmul operations before this phase; its HardEvent 0 is reused.
class CommunicationBuffers {
 public:
    __aicore__ inline void Init(AscendC::TPipe& pipe)
    {
        pipe.InitBuffer(narrow_buffer_, kCommunicationChunk * 2);
        pipe.InitBuffer(accumulator_buffer_, kCommunicationChunk * 4);
        pipe.InitBuffer(term_buffer_, kCommunicationChunk * 4);
        narrow = narrow_buffer_.Get<bfloat16_t>();
        accumulator = accumulator_buffer_.Get<float>();
        term = term_buffer_.Get<float>();
    }

    AscendC::LocalTensor<bfloat16_t> narrow;
    AscendC::LocalTensor<float> accumulator;
    AscendC::LocalTensor<float> term;

 private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> narrow_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumulator_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> term_buffer_;
};

template <AscendC::HardEvent event>
__aicore__ inline void CommunicationFence()
{
    AscendC::SetFlag<event>(EVENT_ID0);
    AscendC::WaitFlag<event>(EVENT_ID0);
}

__aicore__ inline void CommunicationPublish()
{
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
}

__aicore__ inline uint32_t ReadCommunicationEpoch(__gm__ uint8_t* window)
{
    AscendC::GlobalTensor<uint32_t> counter;
    counter.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(window), 1);
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                     AscendC::DcciDst::CACHELINE_OUT>(counter);
    return counter.GetValue(0);
}

__aicore__ inline void AdvanceCommunicationEpoch(__gm__ uint8_t* window)
{
    const uint32_t next = ReadCommunicationEpoch(window) + uint32_t(1);
    AscendC::GlobalTensor<uint32_t> counter;
    counter.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(window), 1);
    counter.SetValue(0, next);
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                     AscendC::DcciDst::CACHELINE_OUT>(counter);
}

__aicore__ inline __gm__ int32_t* CommunicationReady(
    __gm__ uint8_t* window, const CommunicationLayout& layout,
    int32_t ranks, uint32_t epoch, int32_t sender)
{
    return reinterpret_cast<__gm__ int32_t*>(
        window + layout.ready
        + ((epoch & 1U) * ranks + sender) * kCommunicationCacheLine);
}

__aicore__ inline void PushCommunicationPayload(
    CommunicationBuffers& buffers, __gm__ uint8_t* window,
    const CommunicationLayout& layout, __gm__ bfloat16_t* partial,
    int64_t elements, int32_t rank, int32_t ranks, uint32_t epoch,
    int64_t vector_index, int64_t vectors)
{
    AscendC::GlobalTensor<bfloat16_t> source;
    AscendC::GlobalTensor<bfloat16_t> destination;
    source.SetGlobalBuffer(partial, elements);
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
        window + layout.payload + (epoch & 1U) * layout.plane_bytes
        + rank * layout.rank_bytes), elements);
    for (int64_t offset = vector_index; offset < ranks; offset += vectors) {
        const int32_t peer = (rank + offset) % ranks;
        // Explicit UB avoids the library's default MTE scratch at UB offset 0.
        aclshmemx_mte_put_nbi(destination, source, buffers.narrow,
                              static_cast<uint32_t>(elements), peer, EVENT_ID0);
        aclshmem_quiet();
    }
}

__aicore__ inline void ReduceCommunicationPayload(
    CommunicationBuffers& buffers, __gm__ uint8_t* window,
    const CommunicationLayout& layout, __gm__ bfloat16_t* output,
    int64_t elements, int32_t ranks, uint32_t epoch,
    int64_t vector_index, int64_t vectors)
{
    const int32_t signal = static_cast<int32_t>(epoch);
    const int64_t tiles = elements / kCommunicationChunk;
    for (int64_t tile = vector_index; tile < tiles; tile += vectors) {
        const int64_t begin = tile * kCommunicationChunk;
        for (int32_t sender = 0; sender < ranks; ++sender) {
            aclshmem_signal_wait_until(
                CommunicationReady(window, layout, ranks, epoch, sender),
                ACLSHMEM_CMP_EQ, signal);
            AscendC::GlobalTensor<bfloat16_t> source;
            source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
                window + layout.payload + (epoch & 1U) * layout.plane_bytes
                + sender * layout.rank_bytes) + begin, kCommunicationChunk);
            AscendC::DataCopy(buffers.narrow, source, kCommunicationChunk);
            CommunicationFence<AscendC::HardEvent::MTE2_V>();
            if (sender == 0) {
                AscendC::Cast(buffers.accumulator, buffers.narrow,
                              AscendC::RoundMode::CAST_NONE, kCommunicationChunk);
            } else {
                AscendC::Cast(buffers.term, buffers.narrow,
                              AscendC::RoundMode::CAST_NONE, kCommunicationChunk);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(buffers.accumulator, buffers.accumulator,
                             buffers.term, kCommunicationChunk);
            }
            CommunicationFence<AscendC::HardEvent::V_MTE2>();
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(buffers.narrow, buffers.accumulator,
                      AscendC::RoundMode::CAST_RINT, kCommunicationChunk);
        CommunicationFence<AscendC::HardEvent::V_MTE3>();
        AscendC::GlobalTensor<bfloat16_t> destination;
        destination.SetGlobalBuffer(output + begin, kCommunicationChunk);
        AscendC::DataCopy(destination, buffers.narrow, kCommunicationChunk);
        // The next tile loads into the same narrow buffer through MTE2.
        CommunicationFence<AscendC::HardEvent::MTE3_MTE2>();
    }
}

// All resident AIC and AIV cores call this function in the same phase order.
// AICs participate in the three SyncAll barriers and issue no SHMEM operation.
// vector_index is the flat AIV index; vectors is the participating AIV count.
// ranks, capacity, rows and the window's lifetime are validated by the caller.
__aicore__ inline void AllReduceBf16(
    CommunicationBuffers& buffers, __gm__ uint8_t* window,
    __gm__ bfloat16_t* partial, __gm__ bfloat16_t* output,
    int64_t rows, int64_t capacity, int32_t rank, int32_t ranks,
    int64_t vector_index, int64_t vectors)
{
    if ASCEND_IS_AIV {
        if (vector_index == 0) {
            AdvanceCommunicationEpoch(window);
        }
    }
    CommunicationPublish();
    AscendC::SyncAll<false>();
    const CommunicationLayout layout = MakeCommunicationLayout(capacity, ranks);
    uint32_t epoch = 0;
    if ASCEND_IS_AIV {
        epoch = ReadCommunicationEpoch(window);
        PushCommunicationPayload(buffers, window, layout, partial,
                                  rows * kCommunicationHidden, rank, ranks,
                                  epoch, vector_index, vectors);
    }
    // All AIVs have completed their writes before one AIV publishes ready.
    CommunicationPublish();
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIV {
        if (vector_index == 0) {
            for (int32_t offset = 0; offset < ranks; ++offset) {
                const int32_t peer = (rank + offset) % ranks;
                aclshmemx_signal_op(
                    CommunicationReady(window, layout, ranks, epoch, rank),
                    static_cast<int32_t>(epoch), ACLSHMEM_SIGNAL_SET, peer);
            }
        }
        ReduceCommunicationPayload(buffers, window, layout, output,
                                    rows * kCommunicationHidden, ranks, epoch,
                                    vector_index, vectors);
    }
    // No caller can advance the next epoch before every local reader finishes.
    CommunicationPublish();
    AscendC::SyncAll<false>();
}

}  // namespace whole_layer
#endif
