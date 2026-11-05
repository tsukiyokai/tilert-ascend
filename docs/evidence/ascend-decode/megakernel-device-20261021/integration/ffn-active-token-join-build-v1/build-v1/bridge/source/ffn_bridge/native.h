#ifndef TILERT_FFN_BRIDGE_NATIVE_H
#define TILERT_FFN_BRIDGE_NATIVE_H

#ifndef ASCENDC_CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "contract.h"
#include "../norm_native.h"

// Only fields read by the pinned without-group device body cross this seam.
// This is not the vendor host tiling ABI.
struct BridgeTopKTiling {
    int64_t perCoreRowCount = 1;
    int64_t lastCoreRowCount = 1;
    int64_t expertCount = ffn_bridge::kExperts;
    int64_t addBias = 1;
    int64_t outFlag = 0;
    int64_t k = ffn_bridge::kTopK;
    int64_t renorm = 1;
    int64_t normType = 1;
    float routedScalingFactor = 1.0F;
    float eps = 1.0e-20F;
};

#define MoeGatingTopKTilingData BridgeTopKTiling
#include "moe_gating_top_k_without_group.h"
#undef MoeGatingTopKTilingData

namespace ffn_bridge::device {
using namespace AscendC;
using namespace matmul;

template <HardEvent event>
__aicore__ inline void Fence()
{
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
}

__aicore__ inline void EndPhase(TPipe& pipe)
{
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
    SyncAll<false>();
    pipe.Reset();
}

using BF16 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using FP32 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using GateMatmul = Matmul<BF16, BF16, FP32, FP32>;

__aicore__ inline void Gate(TPipe& pipe, __gm__ bfloat16_t* input,
    __gm__ bfloat16_t* weight, __gm__ float* output, GM_ADDR tiling,
    int64_t rows, int64_t blocks)
{
    if ASCEND_IS_AIC {
        const int64_t core = GetBlockIdx();
        if (core >= kExperts / kGateTile) {
            return;
        }
        TCubeTiling cube;
        auto* words = reinterpret_cast<uint32_t*>(&cube);
        const auto* source = reinterpret_cast<__gm__ uint32_t*>(tiling);
        for (int64_t i = 0; i < sizeof(cube) / sizeof(uint32_t); ++i) {
            words[i] = source[i];
        }
        GateMatmul mm;
        mm.Init(&cube, &pipe);
        GlobalTensor<bfloat16_t> a, b;
        GlobalTensor<float> c;
        a.SetGlobalBuffer(input, rows * kHidden);
        b.SetGlobalBuffer(weight, kHidden * kExperts);
        c.SetGlobalBuffer(output, rows * kExperts);
        for (int64_t tile = core; tile < kExperts / kGateTile; tile += blocks) {
            const int64_t column = tile * kGateTile;
            mm.SetOrgShape(Capacity(rows), kExperts, kHidden);
            mm.SetSingleShape(rows, kGateTile, kHidden);
            mm.SetTensorA(a, false);
            mm.SetTensorB(b[column], false);
            mm.IterateAll(c[column]);
            mm.End();
        }
    }
}

__aicore__ inline void TopK(TPipe& pipe, GM_ADDR logits, GM_ADDR bias,
    GM_ADDR probabilities, GM_ADDR ids, GM_ADDR system, int64_t rows)
{
    if ASCEND_IS_AIV {
        // Every active AIV owns exactly one full row. Both tail counts are
        // one, so the standalone GetBlockNum convention cannot change it.
        if (GetBlockIdx() < rows) {
            BridgeTopKTiling tiling;
            MoeGatingTopK::MoeGatingTopKWithoutGroup<float> router;
            router.Init(logits, bias, probabilities, ids, logits, system,
                        &tiling, &pipe);
            router.Process();
        }
    }
}

__aicore__ inline void Quantize(TPipe& pipe, __gm__ bfloat16_t* input,
    __gm__ int8_t* output, __gm__ float* scales, int64_t rows)
{
    if ASCEND_IS_AIV {
        const int64_t row = GetBlockIdx();
        if (row < rows) {
            whole_layer_norm::device::DynamicQuantRow quant;
            quant.Init(pipe);
            quant.Process(input + row * kHidden, output + row * kHidden,
                          scales + row);
        }
    }
}

__aicore__ inline void Epilogue(TPipe& pipe, GM_ADDR routed, GM_ADDR shared,
    GM_ADDR scaled, GM_ADDR combined, int64_t rows, int64_t blocks)
{
    if ASCEND_IS_AIV {
        constexpr int64_t tile = 1024;
        TBuf<TPosition::VECCALC> buffer;
        pipe.InitBuffer(buffer, tile * 12);
        auto bytes = buffer.Get<uint8_t>();
        auto x = bytes.ReinterpretCast<bfloat16_t>();
        auto y = bytes[tile * 2].ReinterpretCast<bfloat16_t>();
        auto value = bytes[tile * 4].ReinterpretCast<float>();
        auto term = bytes[tile * 8].ReinterpretCast<float>();
        GlobalTensor<bfloat16_t> input, other, intermediate, output;
        input.SetGlobalBuffer((__gm__ bfloat16_t*)routed, rows * kHidden);
        other.SetGlobalBuffer((__gm__ bfloat16_t*)shared, rows * kHidden);
        intermediate.SetGlobalBuffer((__gm__ bfloat16_t*)scaled, rows * kHidden);
        output.SetGlobalBuffer((__gm__ bfloat16_t*)combined, rows * kHidden);
        for (int64_t index = GetBlockIdx(); index < rows * kHidden / tile;
             index += 2 * blocks) {
            const int64_t begin = index * tile;
            DataCopy(x, input[begin], tile);
            DataCopy(y, other[begin], tile);
            Fence<HardEvent::MTE2_V>();
            Cast(value, x, RoundMode::CAST_NONE, tile);
            Cast(term, y, RoundMode::CAST_NONE, tile);
            PipeBarrier<PIPE_V>();
            Muls(value, value, 2.5F, tile);
            PipeBarrier<PIPE_V>();
            Cast(x, value, RoundMode::CAST_RINT, tile);
            Fence<HardEvent::V_MTE3>();
            DataCopy(intermediate[begin], x, tile);
            PipeBarrier<PIPE_V>();
            Cast(value, x, RoundMode::CAST_NONE, tile);
            PipeBarrier<PIPE_V>();
            Add(value, value, term, tile);
            PipeBarrier<PIPE_V>();
            Cast(y, value, RoundMode::CAST_RINT, tile);
            Fence<HardEvent::V_MTE3>();
            DataCopy(output[begin], y, tile);
            Fence<HardEvent::MTE3_MTE2>();
        }
    }
}

}  // namespace ffn_bridge::device
#endif
