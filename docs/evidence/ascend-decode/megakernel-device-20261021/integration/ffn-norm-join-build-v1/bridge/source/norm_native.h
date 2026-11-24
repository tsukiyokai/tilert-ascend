#ifndef TILERT_WHOLE_LAYER_NORM_NATIVE_H
#define TILERT_WHOLE_LAYER_NORM_NATIVE_H

#include "kernel_operator.h"
#include "norm_contract.h"

// Fields consumed by the external single-N implementation. No serialized
// tiling blob crosses this interface; the structure is local to this source.
struct TileRTNormSingleNArgs {
    uint32_t num_col;
    uint32_t ub_factor;
    float epsilon;
    uint32_t nullptr_beta;
};

#define AddRMSNormBiasTilingData TileRTNormSingleNArgs
#include "add_rms_norm_bias_single_n.h"
#undef AddRMSNormBiasTilingData

namespace whole_layer_norm::device {
using namespace AscendC;

template <HardEvent Event>
__aicore__ inline void Fence()
{
    const auto event = GetTPipePtr()->FetchEventID(Event);
    SetFlag<Event>(event);
    WaitFlag<Event>(event);
}

class DynamicQuantRow {
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
        __gm__ int8_t* output, __gm__ float* scale)
    {
        GlobalTensor<bfloat16_t> source;
        GlobalTensor<int8_t> destination;
        GlobalTensor<float> scales;
        source.SetGlobalBuffer(input, kHidden);
        destination.SetGlobalBuffer(output, kHidden);
        scales.SetGlobalBuffer(scale, 1);
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

__aicore__ inline void Forward(
    TPipe& pipe, __gm__ bfloat16_t* input, __gm__ bfloat16_t* residual,
    __gm__ bfloat16_t* gamma, __gm__ bfloat16_t* beta,
    __gm__ bfloat16_t* normalized, __gm__ bfloat16_t* retained,
    __gm__ float* rstd, __gm__ int8_t* quantized,
    __gm__ float* scales, int64_t rows, int64_t blocks, float epsilon)
{
    if ASCEND_IS_AIV {
        const int64_t vector = GetBlockIdx();
        // These are the A2/A3 host tiler's BF16 single-N choices for H=6144.
        const TileRTNormSingleNArgs args{
            kHidden, beta == nullptr ? 12288U : 11264U,
            epsilon, beta == nullptr ? 1U : 0U};
        for (int64_t row = vector; row < rows; row += 2 * blocks) {
            // The vendor body adds GetBlockIdx() internally. Offset only
            // complete waves so its reduction and BF16 boundaries stay intact.
            const int64_t wave = row - vector;
            KernelAddRmsNormBiasSingleN<bfloat16_t> norm(&pipe);
            norm.Init(reinterpret_cast<GM_ADDR>(input + wave * kHidden),
                reinterpret_cast<GM_ADDR>(residual + wave * kHidden),
                reinterpret_cast<GM_ADDR>(gamma), reinterpret_cast<GM_ADDR>(beta),
                reinterpret_cast<GM_ADDR>(normalized + wave * kHidden),
                reinterpret_cast<GM_ADDR>(rstd + wave),
                reinterpret_cast<GM_ADDR>(retained + wave * kHidden), &args);
            norm.Process();
            PipeBarrier<PIPE_ALL>();
            pipe.Reset();
            if (quantized != nullptr) {
                DynamicQuantRow quant;
                quant.Init(pipe);
                quant.Process(normalized + row * kHidden,
                    quantized + row * kHidden, scales + row);
                PipeBarrier<PIPE_ALL>();
                pipe.Reset();
            }
        }
    }
}

}
#endif
