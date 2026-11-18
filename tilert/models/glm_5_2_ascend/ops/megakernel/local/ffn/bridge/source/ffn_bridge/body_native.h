#include "native.h"

namespace ffn_bridge {
struct Addresses { GM_ADDR values[kBuffers]; };
#ifdef K11240_OUTLINE_PHASES
__aicore__ __attribute__((noinline)) inline void Forward(AscendC::TPipe& pipe, GM_ADDR input,
#else
__aicore__ inline void Forward(AscendC::TPipe& pipe, GM_ADDR input,
#endif
    GM_ADDR residual, GM_ADDR gamma, GM_ADDR gate, GM_ADDR bias,
    Addresses a, int64_t rows, int64_t blocks, float epsilon,
    GM_ADDR restored_rows = nullptr, GM_ADDR restored_scales = nullptr,
    uint64_t shared_payload = 0, int64_t shared_capacity = 0,
    int64_t topk_k = ffn_bridge::kTopK)
{
    using namespace AscendC;
#ifdef K112612_BRIDGE_FAST
#ifdef K112726_BRIDGE_PRELOAD_GATE
    auto gate_pre = device::PreloadDirectBridgeGate16Aic(pipe, (__gm__ bfloat16_t*)gate, rows);
#endif
    whole_layer_norm::device::Forward(pipe,
        (__gm__ bfloat16_t*)input, (__gm__ bfloat16_t*)residual,
        (__gm__ bfloat16_t*)gamma, nullptr,
        (__gm__ bfloat16_t*)a.values[kNormalized],
        (__gm__ bfloat16_t*)a.values[kRetained],
#ifdef K11040_F0_NZ
        nullptr,
#else
        (__gm__ float*)a.values[kRstd],
#endif
        (__gm__ int8_t*)a.values[kSharedCodes],
        (__gm__ float*)a.values[kSharedScales],
        rows, blocks, epsilon,
        reinterpret_cast<__gm__ int8_t*>(shared_payload), shared_capacity,
        (__gm__ int8_t*)restored_rows, (__gm__ float*)restored_scales);
#ifdef K112726_BRIDGE_PRELOAD_GATE
    device::RunDirectBridgeGate16Aic(pipe, gate_pre, (__gm__ bfloat16_t*)a.values[kNormalized],
        (__gm__ float*)a.values[kLogits], rows);
#else
    device::Gate(pipe, (__gm__ bfloat16_t*)a.values[kNormalized],
        (__gm__ bfloat16_t*)gate, (__gm__ float*)a.values[kLogits],
        a.values[kTiling], rows, blocks);
#endif
    BridgeTopKTiling topk_tiling;
    topk_tiling.k = topk_k;
    MoeGatingTopK::MoeGatingTopKWithoutGroup<float> router;
    const int64_t idx = GetBlockIdx();
    if ASCEND_IS_AIV {
        if (idx < rows) {
            router.Init(a.values[kLogits], bias, a.values[kProbabilities],
                        a.values[kIds], a.values[kLogits], a.values[kSystem],
                        &topk_tiling, &pipe, idx);
            router.PreloadBias();
        }
    }
    PipeBarrier<PIPE_ALL>();
#if !defined(K112612_TRIM_BARRIERS) && !defined(K112722_NO_DDR_EXTRA)
    DataSyncBarrier<MemDsbT::DDR>();
#endif
    SyncAll<false>();
    if ASCEND_IS_AIC {
        pipe.Reset();
        return;
    }
    if ASCEND_IS_AIV {
        if (idx < rows) {
            router.ProcessPreloaded();
        } else {
            pipe.Reset();
        }
        PipeBarrier<PIPE_ALL>();
#if !defined(K112612_TRIM_BARRIERS) && !defined(K112722_NO_DDR_EXTRA)
        DataSyncBarrier<MemDsbT::DDR>();
#endif
        SyncAll<true>();
        pipe.Reset();
    }
#else
    whole_layer_norm::device::Forward(pipe,
        (__gm__ bfloat16_t*)input, (__gm__ bfloat16_t*)residual,
        (__gm__ bfloat16_t*)gamma, nullptr,
        (__gm__ bfloat16_t*)a.values[kNormalized],
        (__gm__ bfloat16_t*)a.values[kRetained],
#ifdef K11040_F0_NZ
        nullptr, nullptr, nullptr, rows, blocks, epsilon);
#else
        (__gm__ float*)a.values[kRstd], nullptr, nullptr, rows, blocks, epsilon);
#endif
    device::Gate(pipe, (__gm__ bfloat16_t*)a.values[kNormalized],
        (__gm__ bfloat16_t*)gate, (__gm__ float*)a.values[kLogits],
        a.values[kTiling], rows, blocks);
    device::Quantize(pipe, (__gm__ bfloat16_t*)a.values[kNormalized],
        (__gm__ int8_t*)a.values[kSharedCodes],
        (__gm__ float*)a.values[kSharedScales], rows,
        reinterpret_cast<__gm__ int8_t*>(shared_payload), shared_capacity,
        (__gm__ int8_t*)restored_rows, (__gm__ float*)restored_scales);
    BridgeTopKTiling topk_tiling;
    topk_tiling.k = topk_k;
    MoeGatingTopK::MoeGatingTopKWithoutGroup<float> router;
    const int64_t topk_first = (2 * rows <= 2 * blocks) ? rows : (2 * blocks - rows);
    const int64_t topk_row = GetBlockIdx() - topk_first;
    if ASCEND_IS_AIV {
        if (topk_row >= 0 && topk_row < rows) {
            router.Init(a.values[kLogits], bias, a.values[kProbabilities],
                        a.values[kIds], a.values[kLogits], a.values[kSystem],
                        &topk_tiling, &pipe, topk_row);
            router.PreloadBias();
        }
    }
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
    SyncAll<false>();
    if ASCEND_IS_AIC {
        pipe.Reset();
    }
    if ASCEND_IS_AIV {
        if (topk_row < 0 || topk_row >= rows) {
            pipe.Reset();
        } else {
            router.ProcessPreloaded();
        }
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
        SyncAll<true>();
        pipe.Reset();
    }
#endif
}
}
