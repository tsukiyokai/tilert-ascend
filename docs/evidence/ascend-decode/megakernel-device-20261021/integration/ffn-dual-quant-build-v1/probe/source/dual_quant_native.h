#ifndef TILERT_FFN_DUAL_QUANT_INPUT_NATIVE_H
#define TILERT_FFN_DUAL_QUANT_INPUT_NATIVE_H

#include "../bridge/source/norm_native.h"

namespace ffn_dual_quant::device {
using namespace AscendC;
using whole_layer_norm::kHidden;
using whole_layer_norm::device::Fence;

template <HardEvent event>
__aicore__ inline void RouteFence()
{
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
}

class DualQuantRow {
public:
    __aicore__ inline void Init(TPipe& pipe)
    {
        pipe.InitBuffer(storage_, 15 * kHidden + 32);
        auto bytes = storage_.Get<uint8_t>();
        bf16_ = bytes.ReinterpretCast<bfloat16_t>();
        value_ = bytes[2 * kHidden].ReinterpretCast<float>();
        other_ = bytes[6 * kHidden].ReinterpretCast<float>();
        work_ = bytes[10 * kHidden].ReinterpretCast<float>();
        codes_ = bytes[14 * kHidden].ReinterpretCast<int8_t>();
        scalar_ = bytes[15 * kHidden].ReinterpretCast<float>();
    }

    __aicore__ inline void Process(__gm__ bfloat16_t* input,
        __gm__ int8_t* output, __gm__ float* scale,
        __gm__ int8_t* routed_output, __gm__ float* routed_scale)
    {
        GlobalTensor<bfloat16_t> source;
        GlobalTensor<int8_t> destination;
        GlobalTensor<float> scales;
        source.SetGlobalBuffer(input, kHidden);
        destination.SetGlobalBuffer(output, kHidden);
        scales.SetGlobalBuffer(scale, 1);
        GlobalTensor<int8_t> routed_destination;
        GlobalTensor<float> routed_scales;
        routed_destination.SetGlobalBuffer(routed_output, kHidden);
        routed_scales.SetGlobalBuffer(routed_scale, 1);
        DataCopy(bf16_, source, kHidden);
        Fence<HardEvent::MTE2_V>();
        Cast(value_, bf16_, RoundMode::CAST_NONE, kHidden);
        PipeBarrier<PIPE_V>();
        Abs(other_, value_, kHidden);
        Duplicate(scalar_, 0.0F, 8);
        PipeBarrier<PIPE_V>();
        ReduceMax(scalar_, other_, work_, kHidden, false);
        Fence<HardEvent::V_S>();
        const float maximum = scalar_.GetValue(0);

        const float quantum = maximum / 127.0F;
        RouteFence<HardEvent::S_V>();
        Duplicate(other_, quantum, 8);
        Duplicate(work_, quantum, kHidden);
        PipeBarrier<PIPE_V>();
        Div(work_, value_, work_, kHidden);
        PipeBarrier<PIPE_V>();
        auto routed_integer = work_.ReinterpretCast<int32_t>();
        Cast(routed_integer, work_, RoundMode::CAST_RINT, kHidden);
        PipeBarrier<PIPE_V>();
        SetDeqScale(half(1.0F));
        auto routed_half = bf16_.ReinterpretCast<half>();
        Cast(routed_half, routed_integer, RoundMode::CAST_ROUND, kHidden);
        PipeBarrier<PIPE_V>();
        Cast(codes_, routed_half, RoundMode::CAST_TRUNC, kHidden);
        RouteFence<HardEvent::V_MTE3>();
        DataCopy(routed_destination, codes_, kHidden);
        DataCopyPad(routed_scales, other_,
                    DataCopyExtParams{1, sizeof(float), 0, 0, 0});
        RouteFence<HardEvent::MTE3_V>();
        RouteFence<HardEvent::MTE3_MTE2>();

        Fence<HardEvent::S_V>();
        Duplicate(work_, maximum == 0.0F ? 1.0F : maximum, kHidden);
        Duplicate(other_, 127.0F, kHidden);
        PipeBarrier<PIPE_V>();
        Div(other_, other_, work_, kHidden);
        PipeBarrier<PIPE_V>();
        Mul(value_, value_, other_, kHidden);
        Muls(scalar_, scalar_, 1.0F / 127.0F, 8);
        PipeBarrier<PIPE_V>();
        Mins(value_, value_, 127.0F, kHidden);
        PipeBarrier<PIPE_V>();
        Maxs(value_, value_, -127.0F, kHidden);
        PipeBarrier<PIPE_V>();
        auto integers = work_.ReinterpretCast<int32_t>();
        Cast(integers, value_, RoundMode::CAST_RINT, kHidden);
        PipeBarrier<PIPE_V>();
        SetDeqScale(static_cast<half>(1.0F));
        PipeBarrier<PIPE_V>();
        auto fp16 = bf16_.ReinterpretCast<half>();
        Cast(fp16, integers, RoundMode::CAST_ROUND, kHidden);
        PipeBarrier<PIPE_V>();
        Cast(codes_, fp16, RoundMode::CAST_TRUNC, kHidden);
        Fence<HardEvent::V_MTE3>();
        DataCopy(destination, codes_, kHidden);
        RmsNorm::DataCopyCustom<float>(scales, scalar_, 1);
        Fence<HardEvent::MTE3_V>();
    }

private:
    TBuf<TPosition::VECCALC> storage_;
    LocalTensor<bfloat16_t> bf16_;
    LocalTensor<float> value_;
    LocalTensor<float> other_;
    LocalTensor<float> work_;
    LocalTensor<float> scalar_;
    LocalTensor<int8_t> codes_;
};

__aicore__ inline void Quantize(TPipe& pipe, __gm__ bfloat16_t* input,
    __gm__ int8_t* shared_output, __gm__ float* shared_scale,
    __gm__ int8_t* routed_output, __gm__ float* routed_scale, int64_t rows)
{
    if ASCEND_IS_AIV {
        const int64_t row = GetBlockIdx();
        if (row < rows) {
            DualQuantRow quant;
            quant.Init(pipe);
            quant.Process(input + row * kHidden,
                shared_output + row * kHidden, shared_scale + row,
                routed_output + row * kHidden, routed_scale + row);
        }
    }
}
}
#endif
