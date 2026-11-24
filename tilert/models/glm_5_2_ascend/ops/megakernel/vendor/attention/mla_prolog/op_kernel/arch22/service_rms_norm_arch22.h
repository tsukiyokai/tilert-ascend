

#ifndef SERVICE_RMS_NORM_ARCH22_H
#define SERVICE_RMS_NORM_ARCH22_H

#include "mla_prolog_comm_arch22.h"
#include "mla_prolog_vector_comm_arch22.h"

#include "rms_norm.h"

namespace MlaProlog {


template <typename T, typename GammaType, typename C, typename O, typename U>
__aicore__ inline void RmsNormNormal(const LocalTensor<O>& outputLocal, const GlobalTensor<T>& inputGm,
                                     const LocalTensor<GammaType>& gammaLocal,
                                     const LocalTensor<U>& dequantScaleWDqLocal,
                                     const LocalTensor<U>& dequantScaleXLocal, const LocalTensor<uint8_t>& shareTmpUb,
                                     RmsNormParam& rmsNormParams)
{
    int64_t cnt = rmsNormParams.row * rmsNormParams.col;
    LocalTensor<C> xFp32Local = shareTmpUb.ReinterpretCast<C>();


    DataCopyExtParams copyParams{1, static_cast<uint32_t>(rmsNormParams.col * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    if constexpr (std::is_same<T, int32_t>::value) {
        LocalTensor<T> xInt32Local = shareTmpUb.ReinterpretCast<T>();
        DataCopyPad(xInt32Local, inputGm, copyParams, padParams);
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Rectangle rectangleParams{
            (uint32_t)rmsNormParams.row, (uint32_t)rmsNormParams.col,
            (uint32_t)rmsNormParams.col
        };
        Dequant(xFp32Local, xInt32Local, dequantScaleWDqLocal, dequantScaleXLocal, rectangleParams);
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        LocalTensor<T> inputLocal = xFp32Local[rmsNormParams.col].template ReinterpretCast<T>();
        DataCopyPad(inputLocal, inputGm, copyParams, padParams);

        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

        Cast(xFp32Local, inputLocal, RoundMode::CAST_NONE, cnt);
        AscendC::PipeBarrier<PIPE_V>();
    }
    LocalTensor<C> rmsnormShareUB = xFp32Local[rmsNormParams.col];
    AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);

    if constexpr (std::is_same<C, O>::value) {
        RmsNorm(outputLocal, xFp32Local, gammaLocal, rmsnormShareUB.template ReinterpretCast<uint8_t>(), rmsNormParams);
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        RmsNorm(xFp32Local, xFp32Local, gammaLocal, rmsnormShareUB.template ReinterpretCast<uint8_t>(), rmsNormParams);

        AscendC::PipeBarrier<PIPE_V>();
        Cast(outputLocal, xFp32Local, RoundMode::CAST_RINT, cnt);
        AscendC::PipeBarrier<PIPE_V>();
    }
}


template <typename T, typename GammaType, typename SmoothType, typename C, typename O, typename U>
__aicore__ inline void RmsNormDynamicQuant(const LocalTensor<O>& outputLocal, const LocalTensor<U>& outputScales,
                                           const GlobalTensor<T>& inputGm, const LocalTensor<GammaType>& gammaLocal,
                                           const LocalTensor<SmoothType>& smoothLocal,
                                           const LocalTensor<U>& dequantScaleWDqLocal,
                                           const LocalTensor<U>& dequantScaleXLocal,
                                           const LocalTensor<uint8_t>& shareTmpUb, RmsNormParam& rmsNormParams,
                                           bool enableSmoothScalesCq = true)
{
    int64_t cnt = rmsNormParams.row * rmsNormParams.col;
    LocalTensor<C> xFp32Local = shareTmpUb.ReinterpretCast<C>();
    RmsNormNormal<T, GammaType, C, C, U>(xFp32Local, inputGm, gammaLocal, dequantScaleWDqLocal, dequantScaleXLocal,
                                         shareTmpUb[rmsNormParams.col * sizeof(C)], rmsNormParams);
    AscendC::PipeBarrier<PIPE_V>();
    if (enableSmoothScalesCq) {
        Mul(xFp32Local, xFp32Local, smoothLocal, cnt);
    }
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<C> maxInt8Tensor = shareTmpUb.ReinterpretCast<C>()[cnt];
    int64_t rowAlign = Align(rmsNormParams.row, FP32_BLOCK_ELEMENT_NUM);
    Duplicate<C>(maxInt8Tensor, static_cast<C>(127.0), rowAlign);
    AscendC::PipeBarrier<PIPE_V>();
    DynamicQuant(xFp32Local, outputScales, xFp32Local, maxInt8Tensor, shareTmpUb[(cnt + rowAlign) * sizeof(C)],
                 rmsNormParams.row, rmsNormParams.col);
    AscendC::PipeBarrier<PIPE_V>();
    CastFP32ToINT8(outputLocal, xFp32Local, shareTmpUb, cnt);
    AscendC::PipeBarrier<PIPE_V>();
}

}

#endif
