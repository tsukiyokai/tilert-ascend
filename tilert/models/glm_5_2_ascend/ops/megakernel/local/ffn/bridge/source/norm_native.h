#ifndef TILERT_WHOLE_LAYER_NORM_NATIVE_H
#define TILERT_WHOLE_LAYER_NORM_NATIVE_H

#include "kernel_operator.h"
#include "norm_contract.h"


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
        InitFromTensor(storage_.Get<uint8_t>());
    }

    __aicore__ inline void InitFromTensor(LocalTensor<uint8_t> bytes)
    {
        bf16_ = bytes.ReinterpretCast<bfloat16_t>();
        value_ = bytes[2 * kHidden].ReinterpretCast<float>();
        other_ = bytes[6 * kHidden].ReinterpretCast<float>();
        work_ = bytes[10 * kHidden].ReinterpretCast<float>();
        codes_ = bytes[14 * kHidden].ReinterpretCast<int8_t>();
        scalar_ = bytes[15 * kHidden].ReinterpretCast<float>();
    }

    __aicore__ inline void Process(__gm__ bfloat16_t* input,
        __gm__ int8_t* output, __gm__ float* scale,
        __gm__ int8_t* staged = nullptr, int64_t row = 0, int64_t capacity = 0,
        __gm__ int8_t* routed_out = nullptr, __gm__ float* routed_scale = nullptr)
    {
        GlobalTensor<bfloat16_t> source;
        GlobalTensor<int8_t> destination;
        GlobalTensor<float> scales;
        source.SetGlobalBuffer(input, kHidden);
        destination.SetGlobalBuffer(output, kHidden);
        scales.SetGlobalBuffer(scale, 1);
        if (input != nullptr) {
            DataCopy(bf16_, source, kHidden);
            Fence<HardEvent::MTE2_V>();
        }
        Cast(value_, bf16_, RoundMode::CAST_NONE, kHidden);
        PipeBarrier<PIPE_V>();
        Abs(other_, value_, kHidden);
        Duplicate(scalar_, 0.0F, 8);
        PipeBarrier<PIPE_V>();
        ReduceMax(scalar_, other_, work_, kHidden, false);
        Fence<HardEvent::V_S>();
        const float maximum = scalar_.GetValue(0);
        Fence<HardEvent::S_V>();
        Duplicate(work_, maximum == 0.0F ? 1.0F : maximum, 64);
        Duplicate(other_, 127.0F, 64);
        PipeBarrier<PIPE_V>();
        Div(other_, other_, work_, 64);
        PipeBarrier<PIPE_V>();
        Mul(work_, value_, other_, 64, kHidden / 64,
            BinaryRepeatParams{1, 1, 0, 8, 8, 0});
        Muls(scalar_, scalar_, 1.0F / 127.0F, 8);
        PipeBarrier<PIPE_V>();
        Mins(work_, work_, 127.0F, kHidden);
        PipeBarrier<PIPE_V>();
        Maxs(work_, work_, -127.0F, kHidden);
        PipeBarrier<PIPE_V>();
        auto integers = work_.ReinterpretCast<int32_t>();
        Cast(integers, work_, RoundMode::CAST_RINT, kHidden);
        PipeBarrier<PIPE_V>();
        SetDeqScale(static_cast<half>(1.0F));
        PipeBarrier<PIPE_V>();
        auto fp16 = bf16_.ReinterpretCast<half>();
        Cast(fp16, integers, RoundMode::CAST_ROUND, kHidden);
        PipeBarrier<PIPE_V>();
        Cast(codes_, fp16, RoundMode::CAST_TRUNC, kHidden);
        Fence<HardEvent::V_MTE3>();
        if (staged != nullptr) {
            constexpr int64_t kInnerTile = 1024;
            constexpr int64_t kInnerParts = kHidden / kInnerTile;
            GlobalTensor<int8_t> staged_dst;
            staged_dst.SetGlobalBuffer(staged, capacity * kHidden);
            DataCopyParams store_parts;
            store_parts.blockCount = kInnerParts;
            store_parts.blockLen = kInnerTile / 32;
            store_parts.srcStride = 0;
            store_parts.dstStride = (capacity - 1) * (kInnerTile / 32);
            DataCopy(staged_dst[row * kInnerTile], codes_, store_parts);
        } else {
            DataCopy(destination, codes_, kHidden);
        }
        RmsNorm::DataCopyCustom<float>(scales, scalar_, 1);
        Fence<HardEvent::MTE3_V>();
        if (routed_out != nullptr) {
            const float dynamic_scale = maximum == 0.0F ? 0.0F : (127.0F / maximum);
            const float r_scale_val = maximum == 0.0F ? 0.0F : (1.0F / dynamic_scale);
            auto r_codes = other_.ReinterpretCast<int8_t>();
            auto r_scalar = other_[kHidden / 4];
            Duplicate(r_scalar, r_scale_val, 8);
            Muls(work_, value_, dynamic_scale, kHidden);
            PipeBarrier<PIPE_V>();
            Cast(integers, work_, RoundMode::CAST_RINT, kHidden);
            PipeBarrier<PIPE_V>();
            SetDeqScale(static_cast<half>(1.0F));
            PipeBarrier<PIPE_V>();
            Cast(fp16, integers, RoundMode::CAST_ROUND, kHidden);
            PipeBarrier<PIPE_V>();
            Cast(r_codes, fp16, RoundMode::CAST_TRUNC, kHidden);
            Fence<HardEvent::V_MTE3>();
            GlobalTensor<int8_t> r_dst;
            GlobalTensor<float> r_scale;
            r_dst.SetGlobalBuffer(routed_out, kHidden);
            r_scale.SetGlobalBuffer(routed_scale, 1);
            DataCopy(r_dst, r_codes, kHidden);
            DataCopyPad(r_scale, r_scalar,
                        DataCopyExtParams{1, sizeof(float), 0, 0, 0});
            Fence<HardEvent::MTE3_V>();
        }
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
    __gm__ float* scales, int64_t rows, int64_t blocks, float epsilon,
    __gm__ int8_t* staged = nullptr, int64_t capacity = 0,
    __gm__ int8_t* routed_out = nullptr, __gm__ float* routed_scale = nullptr)
{
    if ASCEND_IS_AIV {
        const int64_t vector = GetBlockIdx();

        const TileRTNormSingleNArgs args{
            kHidden, beta == nullptr ? 12288U : 11264U,
            epsilon, beta == nullptr ? 1U : 0U};
        if (vector < rows) {
            KernelAddRmsNormBiasSingleN<bfloat16_t> norm(&pipe);
            norm.Init(reinterpret_cast<GM_ADDR>(input),
                reinterpret_cast<GM_ADDR>(residual),
                reinterpret_cast<GM_ADDR>(gamma), reinterpret_cast<GM_ADDR>(beta),
                normalized == nullptr ? nullptr : reinterpret_cast<GM_ADDR>(normalized),
                rstd == nullptr ? nullptr : reinterpret_cast<GM_ADDR>(rstd),
                reinterpret_cast<GM_ADDR>(retained), &args);
            norm.Process();
            if (quantized != nullptr) {
                PipeBarrier<PIPE_ALL>();
                DynamicQuantRow quant;
                quant.InitFromTensor(norm.unitBuf.Get<uint8_t>());
                quant.Process(nullptr,
                    quantized + vector * kHidden, scales + vector,
                    staged, vector, capacity,
                    routed_out != nullptr ? (routed_out + vector * kHidden) : nullptr,
                    routed_scale != nullptr ? (routed_scale + vector) : nullptr);
            }
            PipeBarrier<PIPE_ALL>();
            pipe.Reset();
        }
    }
#ifndef K112722_NO_DDR_EXTRA
#ifdef K112612_TRIM_BARRIERS
    if ASCEND_IS_AIV {
        DataSyncBarrier<MemDsbT::DDR>();
    }
#else
    DataSyncBarrier<MemDsbT::DDR>();
#endif
#endif
    SyncAll<false>();
}

}
#endif
