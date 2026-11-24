

#ifndef RMS_NORM_H
#define RMS_NORM_H

#include "mla_prolog_comm_arch22.h"
namespace MlaProlog {


template <typename GammaType>
__aicore__ inline void RmsNorm(const LocalTensor<float>& outLocal, const LocalTensor<float>& inputLocal,
                               const LocalTensor<GammaType>& gammaLocal, const LocalTensor<uint8_t>& shareTmpUb,
                               const RmsNormParam& rmsNormParams)
{
    uint64_t cnt = rmsNormParams.row * rmsNormParams.col;
    LocalTensor<float> xSquareLocal = shareTmpUb.ReinterpretCast<float>();
    LocalTensor<float> xSumLocal = xSquareLocal[cnt];
    Mul(xSquareLocal, inputLocal, inputLocal, cnt);
    AscendC::PipeBarrier<PIPE_V>();


    uint64_t repeatTimesAdd = static_cast<uint64_t>(cnt) >> 6;

    BinaryRepeatParams addParams = {1, 1, 1, 0, 8, 0};
    Add(xSquareLocal, xSquareLocal[FP32_REPEAT_ELEMENT_NUM], xSquareLocal, FP32_REPEAT_ELEMENT_NUM, repeatTimesAdd - 1,
        addParams);
    AscendC::PipeBarrier<PIPE_V>();
    WholeReduceSum(xSumLocal, xSquareLocal, FP32_REPEAT_ELEMENT_NUM, 1, 8, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();


    Muls<float>(xSumLocal, xSumLocal, rmsNormParams.reciprocal, 1);
    AscendC::PipeBarrier<PIPE_V>();


    Adds<float>(xSumLocal, xSumLocal, rmsNormParams.epsilon, 1);
    AscendC::PipeBarrier<PIPE_V>();


    Sqrt(xSumLocal, xSumLocal, 1);
    AscendC::PipeBarrier<PIPE_V>();


    BrcbRepeatParams repeatParams = {1, 1};
    Brcb(xSquareLocal, xSumLocal, 1, repeatParams);
    AscendC::PipeBarrier<PIPE_V>();


    uint64_t mask[2] = {UINT64_MAX, UINT64_MAX};

    BinaryRepeatParams divParams = {1, 1, 0, 8, 8, 0};
    Div(inputLocal, inputLocal, xSquareLocal, mask, cnt / 64, divParams);

    AscendC::PipeBarrier<PIPE_V>();

    Cast(xSquareLocal, gammaLocal, RoundMode::CAST_NONE, cnt);

    AscendC::PipeBarrier<PIPE_V>();

    Mul(outLocal, inputLocal, xSquareLocal, cnt);

    if (unlikely(rmsNormParams.isScaleEnable)) {
        AscendC::PipeBarrier<PIPE_V>();
        Muls(outLocal, outLocal, rmsNormParams.scale, cnt);
    }
}
}
#endif
