

#ifndef SERVICE_ROTARY_POSITION_EMBEDDING_ARCH22_H
#define SERVICE_ROTARY_POSITION_EMBEDDING_ARCH22_H

#include "mla_prolog_comm_arch22.h"
#include "mla_prolog_vector_comm_arch22.h"
#include "service_rope_arch22.h"

namespace MlaProlog {

template <typename T, typename C>
__aicore__ inline void PreprocessRopeInput(const GlobalTensor<T>& inputGm, LocalTensor<uint8_t>& shareTmpUb,
                                           const Rectangle& ropeParams, LocalTensor<C>& kFp32Local,
                                           LocalTensor<T>& kLocal, LocalTensor<C>& ropeShareUB,
                                           LocalTensor<C>& kFp32OutputLocal, int64_t cnt)
{
    kLocal = shareTmpUb.ReinterpretCast<T>();

    int64_t baseOffset;
    if constexpr (std::is_same<T, int32_t>::value) {
        baseOffset = cnt;
    } else {
        baseOffset = cnt >> 1;
    }
    kFp32Local = shareTmpUb.ReinterpretCast<C>()[baseOffset];

    DataCopyExtParams copyParams{static_cast<uint16_t>(ropeParams.row),
                                 static_cast<uint32_t>(sizeof(T) * ropeParams.col),
                                 static_cast<uint32_t>(sizeof(T) * (ropeParams.stride - ropeParams.col)), 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    if constexpr (std::is_same<T, float>::value) {
        DataCopyPad(kFp32Local, inputGm, copyParams, padParams);
    } else {
        DataCopyPad(kLocal, inputGm, copyParams, padParams);
    }
    ropeShareUB = shareTmpUb.ReinterpretCast<C>()[baseOffset + cnt];

    kFp32OutputLocal = shareTmpUb.ReinterpretCast<C>()[baseOffset + cnt * 3];
}


template <typename T, typename C, typename O, bool a22EnableDequant = false, bool a22DoRope = true>
__aicore__ inline void RotaryPosEmbPerTensor(LocalTensor<O>& outputLocal, const GlobalTensor<T>& inputGm,
                                             const LocalTensor<C>& cosLocal, const LocalTensor<C>& sinLocal,
                                             LocalTensor<uint8_t>& shareTmpUb, Rectangle ropeParams,
                                             LocalTensor<float> channelDeqScaleLocal = LocalTensor<float>(),
                                             LocalTensor<float> scale = LocalTensor<float>())
{
    LocalTensor<T> kLocal;
    LocalTensor<C> kFp32Local;
    LocalTensor<C> ropeShareUB;
    LocalTensor<C> kFp32OutputLocal;
    int64_t cnt = ropeParams.row * ropeParams.col;
    PreprocessRopeInput<T, C>(inputGm, shareTmpUb, ropeParams, kFp32Local, kLocal, ropeShareUB, kFp32OutputLocal, cnt);
    AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    if constexpr (std::is_same<T, int32_t>::value || a22EnableDequant) {
        Rectangle rectangleParams{(uint32_t)1, (uint32_t)cnt, (uint32_t)cnt};
        if constexpr (std::is_same<T, float>::value) {
            Dequant(kFp32Local, kFp32Local, channelDeqScaleLocal, scale, rectangleParams);
        } else {
            Dequant(kFp32Local, kLocal, channelDeqScaleLocal, scale, rectangleParams);
        }
        AscendC::PipeBarrier<PIPE_V>();
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        Cast(kFp32Local, kLocal, RoundMode::CAST_NONE, cnt);
        AscendC::PipeBarrier<PIPE_V>();
    }
    if constexpr (a22DoRope) {
        if constexpr (std::is_same<O, C>::value) {
            RotaryPosEmb(outputLocal, kFp32Local, cosLocal, sinLocal, ropeShareUB.template ReinterpretCast<uint8_t>(),
                         ropeParams.row, ropeParams.col, 0);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            RotaryPosEmb(kFp32OutputLocal, kFp32Local, cosLocal, sinLocal,
                         ropeShareUB.template ReinterpretCast<uint8_t>(), ropeParams.row, ropeParams.col, 0);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(outputLocal, kFp32OutputLocal, RoundMode::CAST_RINT, cnt);
            AscendC::PipeBarrier<PIPE_V>();
        }
    } else {
        if constexpr (std::is_same<O, C>::value) {
            DataCopy(outputLocal, kFp32Local, cnt);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            Cast(outputLocal, kFp32Local, RoundMode::CAST_RINT, cnt);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}


template <typename T, typename C, typename O, bool a22EnableDequant = false, bool a22DoRope = true>
__aicore__ inline void RotaryPosEmbPerHead(LocalTensor<O>& outputLocal, const GlobalTensor<T>& inputGm,
                                           const LocalTensor<C>& cosLocal, const LocalTensor<C>& sinLocal,
                                           LocalTensor<uint8_t>& shareTmpUb, Rectangle ropeParams, int64_t strideScale,
                                           GlobalTensor<float> channelDeqScaleGm = GlobalTensor<float>(),
                                           LocalTensor<float> deQuantScale = LocalTensor<float>())
{

    if (ropeParams.row == 0U) {
        return;
    }

    int64_t cnt = ropeParams.col * ropeParams.row;
    LocalTensor<C> kFp32Local;
    LocalTensor<T> kLocal;
    LocalTensor<C> ropeShareUB;
    LocalTensor<C> kFp32OutputLocal;
    PreprocessRopeInput<T, C>(inputGm, shareTmpUb, ropeParams, kFp32Local, kLocal, ropeShareUB, kFp32OutputLocal, cnt);

    LocalTensor<C> scaleLocal = ropeShareUB;
    AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if constexpr (std::is_same<T, int32_t>::value || a22EnableDequant) {
        DataCopyExtParams copyParams1{static_cast<uint16_t>(1), static_cast<uint32_t>(ropeParams.col * sizeof(C)),
                                      static_cast<uint32_t>((strideScale - ropeParams.col) * sizeof(C)), 0, 0};
        DataCopyPadExtParams<C> padParams1{false, 0, 0, 0};
        DataCopyPad(scaleLocal, channelDeqScaleGm, copyParams1, padParams1);
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

        uint8_t blockNumPerRow = ropeParams.col / (ALIGN_BLOCK_SIZE / sizeof(C));

        Rectangle rectangleParams{(uint32_t)ropeParams.row, (uint32_t)ropeParams.col, (uint32_t)ropeParams.col};
        if constexpr (std::is_same<T, float>::value) {
            Dequant(kFp32Local, kFp32Local, scaleLocal, deQuantScale, rectangleParams);
        } else {
            Dequant(kFp32Local, kLocal, scaleLocal, deQuantScale, rectangleParams);
        }
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        Cast(kFp32Local, kLocal, RoundMode::CAST_NONE, cnt);
    }
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (a22DoRope) {
        RotaryPosEmb(kFp32OutputLocal, kFp32Local, cosLocal, sinLocal, ropeShareUB.template ReinterpretCast<uint8_t>(),
                     ropeParams.row, ropeParams.col, ropeParams.col);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (std::is_same<O, C>::value) {
            DataCopy(outputLocal, kFp32OutputLocal, cnt);
        } else {
            Cast(outputLocal, kFp32OutputLocal, RoundMode::CAST_RINT, cnt);
        }
    } else {
        if constexpr (std::is_same<O, C>::value) {
            DataCopy(outputLocal, kFp32Local, cnt);
        } else {
            Cast(outputLocal, kFp32Local, RoundMode::CAST_RINT, cnt);
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T, typename O>
__aicore__ inline void RopePostQuantPerChannel(LocalTensor<O>& outputLocal, LocalTensor<T>& inputLocal,
                                               LocalTensor<float>& quantScaleLocal, LocalTensor<uint8_t>& shareTmpUb,
                                               int64_t cnt)
{
    LocalTensor<float> inFp32;
    if constexpr (!std::is_same<T, float>::value) {
        inFp32 = shareTmpUb.ReinterpretCast<float>()[cnt];
        Cast(inFp32, inputLocal, RoundMode::CAST_NONE, cnt);
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        inFp32 = inputLocal;
    }
    Mul(inFp32, inFp32, quantScaleLocal, cnt);
    AscendC::PipeBarrier<PIPE_V>();
    CastFP32ToINT8(outputLocal, inFp32, shareTmpUb, cnt);
    AscendC::PipeBarrier<PIPE_V>();
}
}
#endif
