

#ifndef MLA_PROLOG_VECTOR_COMM_ARCH22_H
#define MLA_PROLOG_VECTOR_COMM_ARCH22_H

#include "mla_prolog_comm_arch22.h"
namespace MlaProlog {

struct PerTileQuantParams {
    uint32_t tileSize;
    uint32_t tileNum;
    float alpha;
    uint32_t row;
    uint32_t col;
};

struct RmsNormParam {
    float reciprocal;
    float epsilon;
    uint32_t row;
    uint32_t col;
    float scale;
    uint16_t isScaleEnable;
};

struct Rectangle {
    uint32_t row;
    uint32_t col;
    uint32_t stride;
};


__aicore__ inline void VecMulMat(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                 const Rectangle& rectangleParams)
{
    uint32_t mask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t columnLoopCount = rectangleParams.col / mask;
    uint32_t remainCount = rectangleParams.col % mask;


    if (columnLoopCount < rectangleParams.row &&
        rectangleParams.stride < REPEAT_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) {
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = rectangleParams.stride / FP32_BLOCK_ELEMENT_NUM;
        repeatParams.src0RepStride = 0;
        repeatParams.src1RepStride = rectangleParams.stride / FP32_BLOCK_ELEMENT_NUM;
        uint32_t offset = 0;
        for (int i = 0; i < columnLoopCount; i++) {

            Mul(dstUb[offset], src0Ub[offset], src1Ub[offset], mask, rectangleParams.row, repeatParams);
            offset += mask;
        }
        if (remainCount > 0) {

            Mul(dstUb[offset], src0Ub[offset], src1Ub[offset], remainCount, rectangleParams.row, repeatParams);
        }
    } else {
        uint32_t offset = 0;
        for (int i = 0; i < rectangleParams.row; i++) {
            Mul(dstUb[offset], src0Ub, src1Ub[offset], rectangleParams.col);
            offset += rectangleParams.stride;
        }
    }
}


template <typename T>
__aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                               const Rectangle& rectangleParams)
{
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {

        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2;
    }


    uint32_t dRemain = rectangleParams.col % repeatElementNum;
    uint32_t dLoop = rectangleParams.col / repeatElementNum;

    if (rectangleParams.stride < REPEAT_STRIDE_UP_BOUND * blockElementNum) {

        BinaryRepeatParams repeatParams{1,
                                        1,
                                        0,
                                        (uint8_t)(rectangleParams.stride / blockElementNum),
                                        (uint8_t)(rectangleParams.stride / blockElementNum),
                                        1};


        if (dLoop <= rectangleParams.row) {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, rectangleParams.row, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            repeatParams.dstRepStride = 8;
            repeatParams.src0RepStride = 8;
            repeatParams.src1RepStride = 0;
            for (uint32_t i = 0; i < rectangleParams.row; i++) {
                Mul(dstUb[rectangleParams.stride * i], src0Ub[rectangleParams.stride * i], src1Ub[blockElementNum * i],
                    repeatElementNum, dLoop, repeatParams);
            }
        }


        if (dRemain > 0) {
            Mul(dstUb[repeatElementNum * dLoop], src0Ub[repeatElementNum * dLoop], src1Ub, dRemain, rectangleParams.row,
                repeatParams);
        }
    } else {


        BinaryRepeatParams repeatParams{1, 1, 0, 8, 8, 0};


        for (uint32_t i = 0; i < rectangleParams.row; i++) {

            Mul(dstUb[rectangleParams.stride * i], src0Ub[rectangleParams.stride * i], src1Ub[blockElementNum * i],
                repeatElementNum, dLoop, repeatParams);

            if (dRemain > 0) {
                Mul(dstUb[rectangleParams.stride * i + repeatElementNum * dLoop],
                    src0Ub[rectangleParams.stride * i + repeatElementNum * dLoop], src1Ub[blockElementNum * i], dRemain,
                    1, repeatParams);
            }
        }
    }
}


__aicore__ inline void RowMax(LocalTensor<float>& dstUb, LocalTensor<float>& srcUb, const Rectangle rectangleParams)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t remain = rectangleParams.col % dtypeMask;
    uint32_t blockCount = rectangleParams.col / dtypeMask;

    BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstRepStride = rectangleParams.stride / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.src0RepStride = rectangleParams.stride / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.src1RepStride = rectangleParams.stride / FP32_BLOCK_ELEMENT_NUM;
    if (blockCount > 0 && remain > 0) {
        Max(srcUb, srcUb, srcUb[dtypeMask * blockCount], remain, rectangleParams.row, repeatParamsMax);
        PipeBarrier<PIPE_V>();
    }

    for (uint32_t columnLoopCount = blockCount >> 1; columnLoopCount > 0;
         columnLoopCount = blockCount >> 1) {
        blockCount = (blockCount + 1) >> 1;
        for (uint32_t j = 0; j < columnLoopCount; j++) {
            Max(srcUb[dtypeMask * j], srcUb[dtypeMask * j], srcUb[dtypeMask * (j + blockCount)], dtypeMask,
                rectangleParams.row, repeatParamsMax);
        }
        PipeBarrier<PIPE_V>();
    }

    WholeReduceMax(dstUb, srcUb, (rectangleParams.col < dtypeMask) ? rectangleParams.col : dtypeMask,
                   rectangleParams.row, 1, 1, rectangleParams.stride / FP32_BLOCK_ELEMENT_NUM,
                   ReduceOrder::ORDER_ONLY_VALUE);
}


template <typename T>
__aicore__ inline void Dequant(const LocalTensor<float>& a22OutputLocal, const LocalTensor<T>& inputLocal,
                               const LocalTensor<float>& scaleLocal, const LocalTensor<float>& scale2Local,
                               const Rectangle& rectangleParams)
{
    uint64_t cnt = rectangleParams.col * rectangleParams.row;
    Cast(a22OutputLocal, inputLocal, RoundMode::CAST_RINT, cnt);
    AscendC::PipeBarrier<PIPE_V>();
    RowMuls(a22OutputLocal, a22OutputLocal, scale2Local, rectangleParams);
    AscendC::PipeBarrier<PIPE_V>();
    VecMulMat(a22OutputLocal, scaleLocal, a22OutputLocal, rectangleParams);
}


__aicore__ inline void DynamicQuant(const LocalTensor<float>& outputLocal, const LocalTensor<float>& scale,
                                    const LocalTensor<float>& inputLocal, const LocalTensor<float>& aMax,
                                    const LocalTensor<float>& rowMaxBrcb, const LocalTensor<uint8_t>& shareTmpUb,
                                    float alpha, uint64_t row, uint64_t col)
{
    int32_t aMaxSizeAlign = Align(static_cast<uint32_t>(row), FP32_BLOCK_ELEMENT_NUM);
    LocalTensor<float> aMaxAlpha = shareTmpUb.ReinterpretCast<float>();
    LocalTensor<float> dupTensor = aMaxAlpha[aMaxSizeAlign];
    LocalTensor<float> scaleReciprocal = dupTensor[aMaxSizeAlign];
    LocalTensor<float> scaleReciprocalBrcb = scaleReciprocal[aMaxSizeAlign];

    constexpr float maxInt8 = 127.0f;
    Duplicate(dupTensor, 1.00f, row);
    Muls(aMaxAlpha, aMax, alpha, row);
    PipeBarrier<PIPE_V>();
    Muls(scale, aMaxAlpha, 1.00f / maxInt8, row);
    PipeBarrier<PIPE_V>();
    Div(scaleReciprocal, dupTensor, scale, row);
    PipeBarrier<PIPE_V>();
    Brcb(scaleReciprocalBrcb, scaleReciprocal, CeilDivT(row, 8UL), {1, 8});
    PipeBarrier<PIPE_V>();
    RowMuls(outputLocal, inputLocal, scaleReciprocalBrcb, Rectangle{(uint32_t)row, (uint32_t)col, (uint32_t)col});
    PipeBarrier<PIPE_V>();
}


__aicore__ inline void RowClips(LocalTensor<float>& dstUb, const LocalTensor<float>& src0Ub,
                                const LocalTensor<float>& maxUb, const LocalTensor<float>& minUb, uint32_t dealRowCount,
                                uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 0;
    repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParams.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParams.src1RepStride = 1;
    uint32_t offset = 0;

    for (uint32_t i = 0; i < dLoop; i++) {
        Min(dstUb[offset], src0Ub[offset], maxUb, dtypeMask, dealRowCount, repeatParams);
        PipeBarrier<PIPE_V>();
        Max(dstUb[offset], dstUb[offset], minUb, dtypeMask, dealRowCount, repeatParams);
        offset += dtypeMask;
    }

    if (dRemain == 0) {
        return;
    }

    Min(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], maxUb, dRemain, dealRowCount, repeatParams);
    PipeBarrier<PIPE_V>();
    Max(dstUb[dLoop * dtypeMask], dstUb[dLoop * dtypeMask], minUb, dRemain, dealRowCount, repeatParams);
}

__aicore__ inline void PerTileClipWithAlpha(LocalTensor<float>& dstClipUb, LocalTensor<float>& aMax,
                                            LocalTensor<float>& aMaxBrcb, const LocalTensor<float>& srcUb,
                                            const LocalTensor<uint8_t>& shareTmpUb,
                                            const PerTileQuantParams& perTileQuantParams)
{
    constexpr uint32_t brcnNum = 8;
    uint32_t srcSize = perTileQuantParams.row * perTileQuantParams.col;
    uint32_t maxAlphaSize = perTileQuantParams.tileNum * perTileQuantParams.row * FP32_BLOCK_ELEMENT_NUM;

    LocalTensor<float> absSrcUb = shareTmpUb.template ReinterpretCast<float>();
    LocalTensor<float> maxAlpha = absSrcUb[Align(srcSize, FP32_BLOCK_ELEMENT_NUM)];
    LocalTensor<float> minAlpha = maxAlpha[Align(maxAlphaSize, FP32_BLOCK_ELEMENT_NUM)];

    Abs(absSrcUb, srcUb, srcSize);
    PipeBarrier<PIPE_V>();

    Rectangle rectangleParams{
        static_cast<uint32_t>(perTileQuantParams.tileNum * perTileQuantParams.row),
        static_cast<uint32_t>(perTileQuantParams.tileSize),
        static_cast<uint32_t>(perTileQuantParams.tileSize)
    };
    RowMax(aMax, absSrcUb, rectangleParams);
    PipeBarrier<PIPE_V>();
    Brcb(aMaxBrcb, aMax, (CeilDivT((perTileQuantParams.tileNum * perTileQuantParams.row), brcnNum)), {1, brcnNum});
    PipeBarrier<PIPE_V>();

    Muls(maxAlpha, aMaxBrcb, perTileQuantParams.alpha, maxAlphaSize);
    Muls(minAlpha, aMaxBrcb, -perTileQuantParams.alpha, maxAlphaSize);
    PipeBarrier<PIPE_V>();

    RowClips(dstClipUb, srcUb, maxAlpha, minAlpha, perTileQuantParams.tileNum * perTileQuantParams.row,
             perTileQuantParams.tileSize, perTileQuantParams.tileSize);
    PipeBarrier<PIPE_V>();
}


template <typename O>
__aicore__ inline void CastFP32ToINT8(const LocalTensor<O> outLocal, const LocalTensor<float>& inputLocal,
                                      const LocalTensor<uint8_t>& shareTmpUb, uint64_t cnt)
{
    LocalTensor<half> tmpHalf = shareTmpUb.ReinterpretCast<half>();
    LocalTensor<int32_t> int32 = shareTmpUb.ReinterpretCast<int32_t>();
    Cast(int32, inputLocal, RoundMode::CAST_RINT, cnt);
    PipeBarrier<PIPE_V>();
    SetDeqScale(static_cast<half>(1.0));
    PipeBarrier<PIPE_V>();
    Cast(tmpHalf, int32, RoundMode::CAST_ROUND, cnt);
    PipeBarrier<PIPE_V>();
    Cast(outLocal, tmpHalf, RoundMode::CAST_TRUNC, cnt);
}


template <typename T, typename C, typename O>
__aicore__ inline void QuantPerChannel(const LocalTensor<O>& outLocal, const LocalTensor<T>& inputLocal,
                                       const LocalTensor<C>& quantScaleLocal, const LocalTensor<uint8_t>& shareTmpUb,
                                       const Rectangle& rectangleParams)
{
    VecMulMat(inputLocal, quantScaleLocal, inputLocal, rectangleParams);
    AscendC::PipeBarrier<PIPE_V>();
    CastFP32ToINT8(outLocal, inputLocal, shareTmpUb, rectangleParams.row * rectangleParams.col);
}


template <typename T, typename C, typename O>
__aicore__ inline void QuantPerTensor(const LocalTensor<O>& outLocal, const LocalTensor<T>& inputLocal,
                                      const LocalTensor<C>& quantScaleLocal, const LocalTensor<uint8_t>& shareTmpUb,
                                      const Rectangle& rectangleParams)
{
    RowMuls(inputLocal, inputLocal, quantScaleLocal, rectangleParams);
    AscendC::PipeBarrier<PIPE_V>();
    CastFP32ToINT8(outLocal, inputLocal, shareTmpUb, rectangleParams.row * rectangleParams.col);
}

__aicore__ inline void QuantPerTile(const LocalTensor<int8_t>& outLocal, const LocalTensor<float>& inputLocal,
                                    const LocalTensor<uint8_t>& shareTmpUb,
                                    const PerTileQuantParams& perTileQuantParams)
{
    LocalTensor<float> scale =
        outLocal[perTileQuantParams.col * perTileQuantParams.row].template ReinterpretCast<float>();

    LocalTensor<float> clipOut = inputLocal;
    uint32_t aMaxSizeAlign =
        Align(static_cast<uint32_t>(perTileQuantParams.tileNum * perTileQuantParams.row), FP32_BLOCK_ELEMENT_NUM);
    uint32_t aMaxBrcbSize = perTileQuantParams.tileNum * perTileQuantParams.row * FP32_BLOCK_ELEMENT_NUM;
    LocalTensor<float> aMax = shareTmpUb.template ReinterpretCast<float>();
    LocalTensor<float> aMaxBrcb = aMax[aMaxSizeAlign];
    LocalTensor<uint8_t> sharedBuf = aMaxBrcb[aMaxBrcbSize].template ReinterpretCast<uint8_t>();

    PerTileClipWithAlpha(clipOut, aMax, aMaxBrcb, inputLocal, sharedBuf, perTileQuantParams);

    LocalTensor<float> quantOut = clipOut;
    DynamicQuant(quantOut, scale, clipOut, aMax, aMaxBrcb, sharedBuf, perTileQuantParams.alpha,
                 perTileQuantParams.tileNum * perTileQuantParams.row, perTileQuantParams.tileSize);
    PipeBarrier<PIPE_V>();
    CastFP32ToINT8(outLocal, quantOut, sharedBuf, perTileQuantParams.col * perTileQuantParams.row);
}


__aicore__ inline void DynamicQuant(const LocalTensor<float>& outputLocal, const LocalTensor<float>& scale,
                                    const LocalTensor<float>& inputLocal, const LocalTensor<float>& maxInt8Tensor,
                                    const LocalTensor<uint8_t>& shareTmpUb, uint64_t row, uint64_t col)
{
    constexpr uint64_t brcnNum = 8;
    uint64_t computeSize = col * row;
    LocalTensor<float> inputCopy = shareTmpUb.ReinterpretCast<float>();
    LocalTensor<float> rowMaxBrcb = inputCopy[Align(computeSize, (uint64_t)ALIGN_BLOCK_SIZE)];

    Abs(inputCopy, inputLocal, computeSize);
    AscendC::PipeBarrier<PIPE_V>();
    Rectangle rectangleParams{
        (uint32_t)row, (uint32_t)col,
        (uint32_t)col
    };

    RowMax(inputCopy, inputCopy, rectangleParams);
    AscendC::PipeBarrier<PIPE_V>();


    Div(scale, inputCopy, maxInt8Tensor, row);
    AscendC::PipeBarrier<PIPE_V>();


    Div(inputCopy, maxInt8Tensor, inputCopy, row);
    AscendC::PipeBarrier<PIPE_V>();

    Brcb(rowMaxBrcb, inputCopy, static_cast<uint8_t>(CeilDivT(row, brcnNum)), {1, brcnNum});
    AscendC::PipeBarrier<PIPE_V>();


    RowMuls(outputLocal, inputLocal, rowMaxBrcb, rectangleParams);
}

}
#endif
