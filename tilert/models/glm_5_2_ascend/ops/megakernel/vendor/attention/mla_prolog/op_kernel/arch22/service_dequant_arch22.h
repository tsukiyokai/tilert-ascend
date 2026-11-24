

#ifndef SERVICE_DEQUANT_ARCH22_H
#define SERVICE_DEQUANT_ARCH22_H

#include "mla_prolog_comm_arch22.h"
#include "mla_prolog_vector_comm_arch22.h"

namespace MlaProlog {


template <typename T, typename C, typename O>
__aicore__ inline void DequantSplitNQc(const GlobalTensor<O>& outputGm, const GlobalTensor<T>& inputGm,
                                       const GlobalTensor<C>& deqScaleQcQrWGm,
                                       const LocalTensor<C>& deQuantScaleQcQrLocal,
                                       const LocalTensor<uint8_t>& shareTmpUb, Rectangle dequantRowColStrideParams,
                                       uint32_t oriCol, uint32_t dstStride, uint32_t subBlockIdx_)
{
    int64_t a22Count = dequantRowColStrideParams.col * 1;
    LocalTensor<T> inputLocal = shareTmpUb.ReinterpretCast<T>();
    LocalTensor<C> scaleLocal = inputLocal[a22Count + 16].template ReinterpretCast<C>();
    LocalTensor<C> computeLocal = scaleLocal[a22Count + 16].template ReinterpretCast<C>();
    LocalTensor<O> outputLocal = computeLocal[a22Count + 16].template ReinterpretCast<O>();

    DataCopyParams inputCopyParams{
        static_cast<uint16_t>(1), static_cast<uint16_t>(a22Count * sizeof(T) / 32U),
        static_cast<uint16_t>((dequantRowColStrideParams.stride - dequantRowColStrideParams.col) * sizeof(T) / 32U), 0};

    DataCopyParams outputCopyParams{static_cast<uint16_t>(1), static_cast<uint16_t>(a22Count * sizeof(O) / 32U), 0,
                                    static_cast<uint16_t>(dstStride * sizeof(O) / 32U)};

    Rectangle rectangleParams{
        (uint32_t)1,
        (uint32_t)a22Count,
        (uint32_t)a22Count
    };

    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    int64_t scaleOffset = a22Count * subBlockIdx_;
    DataCopy(scaleLocal, deqScaleQcQrWGm[scaleOffset], dequantRowColStrideParams.col);
    for (int64_t rowOffset = 0; rowOffset < dequantRowColStrideParams.row; rowOffset++) {
        int64_t outputOffset = rowOffset * oriCol + subBlockIdx_ * a22Count;
        int64_t inputOffset = rowOffset * oriCol + subBlockIdx_ * a22Count;
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        DataCopy(inputLocal, inputGm[inputOffset], inputCopyParams);
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

        Dequant(computeLocal, inputLocal, scaleLocal, deQuantScaleQcQrLocal, rectangleParams);
        AscendC::PipeBarrier<PIPE_V>();

        Cast(outputLocal, computeLocal, RoundMode::CAST_RINT, a22Count);
        AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID2);

        AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(outputGm[outputOffset], outputLocal, a22Count);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}


template <typename T, typename C, typename O>
__aicore__ inline void DequantPerTokenQcSplit(const GlobalTensor<O>& outputGm, const GlobalTensor<T>& inputGm,
                                              const GlobalTensor<C>& deqScaleQcQrWGm,
                                              const LocalTensor<float> deQuantScaleQcQrLocal,
                                              const LocalTensor<uint8_t>& shareTmpUb, uint32_t headCount,
                                              uint32_t col, uint32_t stride, uint32_t startHead)
{
    int64_t count = static_cast<int64_t>(headCount) * static_cast<int64_t>(col);
    LocalTensor<T> inputLocal = shareTmpUb.ReinterpretCast<T>();
    LocalTensor<C> scaleLocal = inputLocal[count + 16].template ReinterpretCast<C>();
    LocalTensor<C> computeLocal = scaleLocal[count + 16].template ReinterpretCast<C>();
    LocalTensor<O> outputLocal = computeLocal[count + 16].template ReinterpretCast<O>();

    DataCopyParams copyParams{
        static_cast<uint16_t>(headCount),
        static_cast<uint16_t>(col * sizeof(T) / 32U),
        static_cast<uint16_t>((stride - col) * sizeof(T) / 32U), 0};

    Rectangle rectangleParams{1U, static_cast<uint32_t>(count), static_cast<uint32_t>(count)};

    int64_t inputOffset = static_cast<int64_t>(startHead) * static_cast<int64_t>(stride);
    int64_t outputOffset = static_cast<int64_t>(startHead) * static_cast<int64_t>(col);

    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    DataCopy(inputLocal, inputGm[inputOffset], copyParams);
    DataCopy(scaleLocal, deqScaleQcQrWGm[inputOffset], copyParams);
    AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

    Dequant(computeLocal, inputLocal, scaleLocal, deQuantScaleQcQrLocal, rectangleParams);
    AscendC::PipeBarrier<PIPE_V>();

    Cast(outputLocal, computeLocal, RoundMode::CAST_RINT, count);
    AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID2);

    AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    DataCopy(outputGm[outputOffset], outputLocal, count);
}

template <typename T, typename C, typename O>
__aicore__ inline void DequantPerTokenQc(const GlobalTensor<O>& outputGm, const GlobalTensor<T>& inputGm,
                                         const GlobalTensor<C>& deqScaleQcQrWGm,
                                         const LocalTensor<float> deQuantScaleQcQrLocal,
                                         const LocalTensor<uint8_t>& shareTmpUb, Rectangle dequantRowColStrideParams,
                                         uint32_t oriRow)
{
    int64_t count = dequantRowColStrideParams.col * dequantRowColStrideParams.row;

    LocalTensor<T> inputLocal = shareTmpUb.ReinterpretCast<T>();
    LocalTensor<C> scaleLocal = inputLocal[count + 16].template ReinterpretCast<C>();
    LocalTensor<C> computeLocal = scaleLocal[count + 16].template ReinterpretCast<C>();
    LocalTensor<O> outputLocal = computeLocal[count + 16].template ReinterpretCast<O>();

    DataCopyParams copyParams{
        static_cast<uint16_t>(dequantRowColStrideParams.row),
        static_cast<uint16_t>(dequantRowColStrideParams.col * sizeof(T) / 32U),
        static_cast<uint16_t>((dequantRowColStrideParams.stride - dequantRowColStrideParams.col) * sizeof(T) / 32U), 0};

    Rectangle rectangleParams{
        (uint32_t)1,
        (uint32_t)count,
        (uint32_t)count
    };

    for (int64_t rowOffset = 0; rowOffset < oriRow; rowOffset += dequantRowColStrideParams.row) {
        int64_t inputOffset = rowOffset * dequantRowColStrideParams.stride;
        int64_t outputOffset = rowOffset * dequantRowColStrideParams.col;

        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        DataCopy(inputLocal, inputGm[inputOffset], copyParams);
        DataCopy(scaleLocal, deqScaleQcQrWGm[inputOffset], copyParams);
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

        Dequant(computeLocal, inputLocal, scaleLocal, deQuantScaleQcQrLocal, rectangleParams);
        AscendC::PipeBarrier<PIPE_V>();

        Cast(outputLocal, computeLocal, RoundMode::CAST_RINT, count);
        AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID2);

        AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(outputGm[outputOffset], outputLocal, count);
    }
}


template <typename T, typename O>
__aicore__ inline void CastPerTokenQc(const GlobalTensor<O>& outputGm, const GlobalTensor<T>& inputGm,
                                      const LocalTensor<uint8_t>& shareTmpUb, Rectangle castRowColStrideParams,
                                      uint32_t oriRow)
{
    int64_t count = castRowColStrideParams.col * castRowColStrideParams.row;

    LocalTensor<T> inputLocal = shareTmpUb.ReinterpretCast<T>();
    LocalTensor<O> outputLocal = inputLocal[count + 16].template ReinterpretCast<O>();

    DataCopyParams copyParams{
        static_cast<uint16_t>(castRowColStrideParams.row),
        static_cast<uint16_t>(castRowColStrideParams.col * sizeof(T) / 32U),
        static_cast<uint16_t>((castRowColStrideParams.stride - castRowColStrideParams.col) * sizeof(T) / 32U), 0};

    for (int64_t rowOffset = 0; rowOffset < oriRow; rowOffset += castRowColStrideParams.row) {
        int64_t inputOffset = castRowColStrideParams.stride * rowOffset;
        int64_t outputOffset = castRowColStrideParams.col * rowOffset;

        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        DataCopy(inputLocal, inputGm[inputOffset], copyParams);
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

        Cast(outputLocal, inputLocal, RoundMode::CAST_RINT, count);
        AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID2);

        AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(outputGm[outputOffset], outputLocal, count);
    }
}

}
#endif
