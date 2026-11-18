

#ifndef SERVICE_GATHER_SIN_COS_ARCH22_H
#define SERVICE_GATHER_SIN_COS_ARCH22_H

#include "mla_prolog_comm_arch22.h"

namespace MlaProlog {


template <typename T, typename O>
__aicore__ inline void GatherSinCos(LocalTensor<O>& cosLocal, LocalTensor<O>& sinLocal, const GlobalTensor<T>& cosGm,
                                    const GlobalTensor<T>& sinGm, int64_t tokenIndex, int64_t curVecToken,
                                    LocalTensor<uint8_t>& shareTmpUb, int64_t row, int64_t col)
{
    int64_t offset = tokenIndex * col;
    int64_t curDataSize = curVecToken * col;
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    if constexpr (IsSameType<O, float>::value) {
        auto tmpUbBf16 = shareTmpUb.ReinterpretCast<bfloat16_t>();
        DataCopy(tmpUbBf16, cosGm[offset], curDataSize);
        DataCopy(tmpUbBf16[curDataSize], sinGm[offset], curDataSize);
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(cosLocal, tmpUbBf16, RoundMode::CAST_NONE, curDataSize);
        Cast(sinLocal, tmpUbBf16[curDataSize], RoundMode::CAST_NONE, curDataSize);
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        DataCopy(cosLocal, cosGm[offset], curDataSize);
        DataCopy(sinLocal, sinGm[offset], curDataSize);
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    uint8_t blockNumPerRow = col / (ALIGN_BLOCK_SIZE / sizeof(O));
    Muls<O>(sinLocal, sinLocal, -1.0f, col >> 1, curVecToken, {1, 1, blockNumPerRow, blockNumPerRow});
    AscendC::PipeBarrier<PIPE_V>();
}

}

#endif
