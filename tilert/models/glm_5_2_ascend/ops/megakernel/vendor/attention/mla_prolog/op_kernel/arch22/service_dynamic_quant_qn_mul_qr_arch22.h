

#ifndef SERVICE_DYNAMIC_QUANT_ARCH22_H
#define SERVICE_DYNAMIC_QUANT_ARCH22_H

#include "mla_prolog_comm_arch22.h"
#include "mla_prolog_vector_comm_arch22.h"

namespace MlaProlog {
template <typename T, typename C>
__aicore__ inline void MulQr(const GlobalTensor<T>& outputGmRope, const GlobalTensor<T>& inputGmRope,
                             LocalTensor<T> outputLocalRope, const LocalTensor<T>& qrInputLocal,
                             const LocalTensor<C>& qrFp32Local, const LocalTensor<C>& reciprocalLocal,
                             const LocalTensor<C>& dequantScaleBrcbLocal, uint64_t row, uint64_t colRope,
                             uint64_t subRowRope, uint64_t qrOutputStrideRope, float quantScaleCkvRope,
                             uint32_t MUL_QR_INPUT_COPY_READY, uint32_t MUL_QR)
{
    constexpr uint64_t computeBlockAlign = (ALIGN_BLOCK_SIZE / sizeof(C));
    constexpr uint64_t inputBlockAlign = (ALIGN_BLOCK_SIZE / sizeof(T));

    uint64_t loopCntRope = CeilDivT(row, subRowRope);
    uint64_t lastSubRowRope = row - (loopCntRope - 1) * subRowRope;
    uint32_t computeSizeRope = subRowRope * colRope;


    DataCopyParams inputParamsRope{(uint16_t)row, (uint16_t)(colRope / inputBlockAlign),
                                   (uint16_t)((qrOutputStrideRope - colRope) / inputBlockAlign), 0};
    DataCopyParams outputParamsRope{(uint16_t)(subRowRope), (uint16_t)(colRope / inputBlockAlign), 0,
                                    (uint16_t)((qrOutputStrideRope - colRope) / inputBlockAlign)};

    uint64_t inputGmRopeOffset = 0;
    uint64_t dequantScaleOffset = 0;
    uint64_t inputLocalRopeOffset = 0;

    AscendC::WaitFlag<HardEvent::V_MTE2>(MUL_QR_INPUT_COPY_READY);
    DataCopy(qrInputLocal, inputGmRope, inputParamsRope);
    for (uint64_t loopIdx = 0; loopIdx < loopCntRope; loopIdx++) {
        if (loopIdx == (loopCntRope - 1) && subRowRope != lastSubRowRope) {
            subRowRope = lastSubRowRope;
            computeSizeRope = subRowRope * colRope;
            outputParamsRope.blockCount = subRowRope;
        }
        AscendC::SetFlag<HardEvent::MTE3_V>(MUL_QR);
        AscendC::WaitFlag<HardEvent::MTE3_V>(MUL_QR);

        AscendC::SetFlag<HardEvent::MTE2_V>(MUL_QR);
        AscendC::WaitFlag<HardEvent::MTE2_V>(MUL_QR);

        Cast(qrFp32Local, qrInputLocal[inputLocalRopeOffset], RoundMode::CAST_NONE, computeSizeRope);
        AscendC::PipeBarrier<PIPE_V>();
        Duplicate(reciprocalLocal, quantScaleCkvRope, subRowRope * computeBlockAlign);
        AscendC::PipeBarrier<PIPE_V>();

        Div(reciprocalLocal, reciprocalLocal, dequantScaleBrcbLocal[dequantScaleOffset],
            subRowRope * computeBlockAlign);
        AscendC::PipeBarrier<PIPE_V>();

        RowMuls(qrFp32Local, qrFp32Local, reciprocalLocal,
                Rectangle{(uint32_t)subRowRope, (uint32_t)colRope, (uint32_t)colRope});
        AscendC::PipeBarrier<PIPE_V>();
        Cast(outputLocalRope, qrFp32Local, RoundMode::CAST_RINT, computeSizeRope);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<HardEvent::V_MTE3>(MUL_QR);
        AscendC::WaitFlag<HardEvent::V_MTE3>(MUL_QR);

        DataCopy(outputGmRope[inputGmRopeOffset], outputLocalRope, outputParamsRope);
        inputLocalRopeOffset += subRowRope * colRope;
        inputGmRopeOffset += subRowRope * qrOutputStrideRope;
        dequantScaleOffset += subRowRope * computeBlockAlign;
    }
    AscendC::SetFlag<HardEvent::V_MTE2>(MUL_QR_INPUT_COPY_READY);
}

template <typename T, typename C, typename O>
__aicore__ inline void DynamicQuantMultiRow(const GlobalTensor<O>& outputGm, const LocalTensor<C>& scaleOutputLocal,
                                            const GlobalTensor<T>& inputGm, const LocalTensor<C> outputLocal,
                                            const LocalTensor<T>& inputHalf, const LocalTensor<C>& inputLocal,
                                            const LocalTensor<C>& maxInt8Tensor, const LocalTensor<uint8_t>& shareTmpUb,
                                            uint64_t row, uint64_t col, uint64_t subRow, uint64_t queryOutStride,
                                            uint32_t DYNAMIC_QUANT_INPUT_READY, uint32_t DYNAMIC_QUANT_OUTPUT_READY)
{
    constexpr uint64_t outputBlockAlign = (ALIGN_BLOCK_SIZE / sizeof(O));
    constexpr uint64_t inputBlockAlign = (ALIGN_BLOCK_SIZE / sizeof(T));

    DataCopyParams outParams{(uint16_t)subRow, (uint16_t)(col / outputBlockAlign), 0,
                             (uint16_t)((queryOutStride - col) / outputBlockAlign)};
    DataCopyParams inputParams{(uint16_t)subRow, (uint16_t)(col / inputBlockAlign),
                               (uint16_t)((queryOutStride - col) / inputBlockAlign), 0};

    uint32_t computeSize = col * subRow;
    uint64_t loopCnt = CeilDivT(row, subRow);
    uint64_t lastSubRow = row - (loopCnt - 1) * subRow;
    uint64_t inputGmOffset = 0;
    uint64_t scaleOffset = 0;

    for (uint64_t loopIdx = 0; loopIdx < loopCnt; loopIdx++) {
        if ((loopIdx == loopCnt - 1) && (lastSubRow != subRow)) {
            subRow = lastSubRow;
            computeSize = col * subRow;
            outParams.blockCount = subRow;
            inputParams.blockCount = subRow;
        }

        AscendC::WaitFlag<HardEvent::V_MTE2>(DYNAMIC_QUANT_INPUT_READY);
        DataCopy(inputHalf, inputGm[inputGmOffset], inputParams);
        AscendC::SetFlag<HardEvent::MTE2_V>(DYNAMIC_QUANT_INPUT_READY);
        AscendC::WaitFlag<HardEvent::MTE2_V>(DYNAMIC_QUANT_INPUT_READY);

        Cast(inputLocal, inputHalf, RoundMode::CAST_NONE, computeSize);
        AscendC::SetFlag<HardEvent::V_MTE2>(DYNAMIC_QUANT_INPUT_READY);
        AscendC::PipeBarrier<PIPE_V>();
        DynamicQuant(inputLocal, scaleOutputLocal[scaleOffset], inputLocal, maxInt8Tensor, shareTmpUb, subRow, col);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::WaitFlag<HardEvent::MTE3_V>(DYNAMIC_QUANT_OUTPUT_READY);
        LocalTensor<half> tmpf16Ub = outputLocal.template ReinterpretCast<half>();
        Cast(tmpf16Ub, inputLocal, AscendC::RoundMode::CAST_ROUND, computeSize);
        AscendC::PipeBarrier<PIPE_V>();

        LocalTensor<O> tmpMMResCastTensor = outputLocal.template ReinterpretCast<O>();
        Cast(tmpMMResCastTensor, tmpf16Ub, AscendC::RoundMode::CAST_ROUND, computeSize);
        AscendC::SetFlag<HardEvent::V_MTE3>(DYNAMIC_QUANT_OUTPUT_READY);
        AscendC::WaitFlag<HardEvent::V_MTE3>(DYNAMIC_QUANT_OUTPUT_READY);

        DataCopy(outputGm[inputGmOffset], tmpMMResCastTensor, outParams);
        AscendC::SetFlag<HardEvent::MTE3_V>(DYNAMIC_QUANT_OUTPUT_READY);
        inputGmOffset += subRow * queryOutStride;
        scaleOffset += subRow;
    }
}


template <typename T, typename C, typename O>
__aicore__ inline void DynamicQuantQnWithMulQr(

    const GlobalTensor<C>& scaleOutputGm, const GlobalTensor<O>& outputGm, const GlobalTensor<T>& outputGmRope,

    const GlobalTensor<T>& inputGm, LocalTensor<uint8_t>& shareTmpUb, uint64_t row, uint64_t col,
    uint64_t scaleOutStride, uint64_t queryOutStride,

    const GlobalTensor<T>& inputGmRope, float quantScaleCkvRope, uint64_t colRope, uint64_t qrOutputStrideRope,
    uint32_t cvRatio)
{
    if (col == 0 || row == 0) {
        return;
    }

    constexpr uint32_t MUL_QR = EVENT_ID1;
    constexpr uint32_t DYNAMIC_QUANT_INPUT_READY =
        EVENT_ID0;
    constexpr uint32_t MUL_QR_INPUT_COPY_READY = EVENT_ID3;
    constexpr uint32_t CALC_SCALE_FINISH = EVENT_ID0;
    constexpr uint32_t DYNAMIC_QUANT_OUTPUT_READY =
        EVENT_ID3;

    constexpr uint64_t inputBlockAlign = (ALIGN_BLOCK_SIZE / sizeof(T));
    constexpr uint64_t computeBlockAlign = (ALIGN_BLOCK_SIZE / sizeof(C));
    constexpr float maxInt8 = 127.0;

    uint64_t rowStepSize = 8 * cvRatio;

    uint64_t subRow = row < rowStepSize ? row : rowStepSize;
    uint32_t brcbCnt = Align(subRow, computeBlockAlign);
    uint32_t computeSize = subRow * col;


    uint64_t rowStepSizeRope = 64;
    uint32_t loadRopeGmSize = row * colRope;
    uint64_t subRowRope = row < rowStepSizeRope ? row : rowStepSizeRope;
    uint32_t computeSizeRope = subRowRope * colRope;


    LocalTensor<T> inputHalf = shareTmpUb.ReinterpretCast<T>();
    LocalTensor<T> qrInputLocal = inputHalf[computeSize];
    LocalTensor<T> outputLocalRope = qrInputLocal[inputBlockAlign + loadRopeGmSize];

    LocalTensor<C> inputLocal = outputLocalRope[inputBlockAlign + computeSizeRope].template ReinterpretCast<C>();
    LocalTensor<C> outputLocal = inputLocal[computeSize];
    LocalTensor<C> maxInt8Tensor = outputLocal[computeSize];
    LocalTensor<C> dynamicQuantUb = maxInt8Tensor[brcbCnt];
    LocalTensor<C> scaleOutputLocal = dynamicQuantUb[computeSize + brcbCnt * computeBlockAlign];
    LocalTensor<C> scaleBrcb = scaleOutputLocal[Align(row, computeBlockAlign)];


    LocalTensor<C> qrFp32Local = inputLocal;
    LocalTensor<C> reciprocalLocal = qrFp32Local[computeBlockAlign + computeSizeRope];


    Duplicate(maxInt8Tensor, static_cast<C>(maxInt8), brcbCnt);
    AscendC::PipeBarrier<PIPE_V>();

    DynamicQuantMultiRow(outputGm, scaleOutputLocal, inputGm, outputLocal, inputHalf, inputLocal, maxInt8Tensor,
                         dynamicQuantUb.template ReinterpretCast<uint8_t>(), row, col, subRow, queryOutStride,
                         DYNAMIC_QUANT_INPUT_READY, DYNAMIC_QUANT_OUTPUT_READY);

    Brcb(scaleBrcb, scaleOutputLocal, CeilDivT(row, computeBlockAlign), {1, computeBlockAlign});
    AscendC::PipeBarrier<PIPE_V>();


    DataCopyParams scaleOutCopyParams{(uint16_t)row, (uint16_t)sizeof(C), 0,
                                      (uint16_t)((scaleOutStride - 1) * sizeof(C))};

    AscendC::SetFlag<HardEvent::V_MTE3>(CALC_SCALE_FINISH);
    AscendC::WaitFlag<HardEvent::V_MTE3>(CALC_SCALE_FINISH);
    DataCopyPad(scaleOutputGm, scaleBrcb, scaleOutCopyParams);


    MulQr(outputGmRope, inputGmRope, outputLocalRope, qrInputLocal, qrFp32Local, reciprocalLocal, scaleBrcb, row,
          colRope, subRowRope, qrOutputStrideRope, quantScaleCkvRope, MUL_QR_INPUT_COPY_READY, MUL_QR);

    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

}

#endif
