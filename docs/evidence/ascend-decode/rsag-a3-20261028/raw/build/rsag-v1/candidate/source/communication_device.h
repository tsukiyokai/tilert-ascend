#ifndef TRT_RSAG_CANDIDATE_COMMUNICATION_DEVICE_H
#define TRT_RSAG_CANDIDATE_COMMUNICATION_DEVICE_H

#include "communication.h"
#include "kernel_operator.h"
#include "shmem.h"
#include "tree_plan.h"

namespace trt_rsag_candidate {

// Call Init only on AIV, with 44 KiB available in the caller's TPipe arena.
// Finish all Matmul operations before this phase; its HardEvent 0 is reused.
class CommunicationBuffers {
 public:
    __aicore__ inline void Init(AscendC::TPipe& pipe)
    {
        pipe.InitBuffer(narrow_buffer_, kCommunicationChunk * 2);
        pipe.InitBuffer(accumulator_buffer_, kCommunicationChunk * 4 * 5);
        narrow = narrow_buffer_.Get<bfloat16_t>();
        accumulator = accumulator_buffer_.Get<float>();
        term = accumulator[kCommunicationChunk];
    }

    AscendC::LocalTensor<bfloat16_t> narrow;
    AscendC::LocalTensor<float> accumulator;
    AscendC::LocalTensor<float> term;

 private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> narrow_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumulator_buffer_;
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

#include "rsag_device.inc"

}  // namespace trt_rsag_candidate
#endif
