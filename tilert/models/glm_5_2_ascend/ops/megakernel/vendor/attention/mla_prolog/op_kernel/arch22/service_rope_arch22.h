

#ifndef SERVICE_ROPE_ARCH22_H
#define SERVICE_ROPE_ARCH22_H

namespace MlaProlog {


template <typename C>
__aicore__ inline void RotaryPosEmb(const LocalTensor<C>& outputLocal, const LocalTensor<C>& inputLocal,
                                    const LocalTensor<C>& cosLocal, const LocalTensor<C>& sinLocal,
                                    const LocalTensor<uint8_t>& shareTmpUb, uint64_t row, uint64_t col,
                                    uint8_t sinCosRepStride)
{
    uint64_t cnt = row * col;
    uint64_t rsvdCnt = 0;
    LocalTensor<C> reArrLocal = shareTmpUb.ReinterpretCast<C>();
    LocalTensor<C> outputLocalSinTmp = shareTmpUb.ReinterpretCast<C>()[cnt];
    GatherMaskParams gatherMaskParams = {
        1,
        1,
        0,
        0
    };

    GatherMask(reArrLocal, inputLocal, 1, true, col * row, gatherMaskParams, rsvdCnt);

    GatherMask(reArrLocal[cnt >> 1], inputLocal, 2, true, col * row, gatherMaskParams, rsvdCnt);
    AscendC::PipeBarrier<PIPE_V>();
    uint8_t blockNumPerRow = col / (ALIGN_BLOCK_SIZE / sizeof(C));
    uint8_t blockNumPerRowHalf = blockNumPerRow >> 1;
    uint8_t blockNumSinCosRepStride = sinCosRepStride / (ALIGN_BLOCK_SIZE / sizeof(C));
    BinaryRepeatParams mulParams = {
        1,
        1,
        1,
        blockNumPerRow,
        blockNumPerRowHalf,
        blockNumSinCosRepStride
    };
    Mul(outputLocal, reArrLocal, cosLocal, col >> 1, row, mulParams);
    Mul(outputLocal[col >> 1], reArrLocal[cnt >> 1], cosLocal[col >> 1], col >> 1, row, mulParams);
    Mul(outputLocalSinTmp, reArrLocal[cnt >> 1], sinLocal, col >> 1, row, mulParams);
    Mul(outputLocalSinTmp[col >> 1], reArrLocal, sinLocal[col >> 1], col >> 1, row, mulParams);
    AscendC::PipeBarrier<PIPE_V>();
    Add(outputLocal, outputLocal, outputLocalSinTmp, cnt);
}
}
#endif
