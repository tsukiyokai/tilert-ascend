#include "native.h"

namespace {
using namespace AscendC;
using namespace ffn_bridge;

struct Addresses {
    GM_ADDR values[kBuffers];
};

__global__ __mix__(1, 2) __schedmode__(1) __aicore__
void Bridge(uint64_t sync_base, GM_ADDR input, GM_ADDR residual,
    GM_ADDR gamma, GM_ADDR gate, GM_ADDR bias, Addresses a,
    int64_t rows, int64_t blocks, float epsilon)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);
    SetSysWorkspace(a.values[kSystem]);
    TPipe pipe;
    whole_layer_norm::device::Forward(pipe,
        (__gm__ bfloat16_t*)input, (__gm__ bfloat16_t*)residual,
        (__gm__ bfloat16_t*)gamma, nullptr,
        (__gm__ bfloat16_t*)a.values[kNormalized],
        (__gm__ bfloat16_t*)a.values[kRetained],
        (__gm__ float*)a.values[kRstd], nullptr, nullptr, rows, blocks, epsilon);
    device::EndPhase(pipe);
    device::Gate(pipe, (__gm__ bfloat16_t*)a.values[kNormalized],
        (__gm__ bfloat16_t*)gate, (__gm__ float*)a.values[kLogits],
        a.values[kTiling], rows, blocks);
    device::Quantize(pipe, (__gm__ bfloat16_t*)a.values[kNormalized],
        (__gm__ int8_t*)a.values[kSharedCodes],
        (__gm__ float*)a.values[kSharedScales], rows);
    device::EndPhase(pipe);
    device::TopK(pipe, a.values[kLogits], bias, a.values[kProbabilities],
                 a.values[kIds], a.values[kSystem], rows);
    device::EndPhase(pipe);
}

__global__ __mix__(1, 2) __schedmode__(1) __aicore__
void FixedTopK(uint64_t sync_base, GM_ADDR logits, GM_ADDR bias,
    Addresses a, int64_t rows)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);
    SetSysWorkspace(a.values[kSystem]);
    TPipe pipe;
    device::TopK(pipe, logits, bias, a.values[kProbabilities], a.values[kIds],
                 a.values[kSystem], rows);
    device::EndPhase(pipe);
}

__global__ __mix__(1, 2) __schedmode__(1) __aicore__
void Finalize(uint64_t sync_base, GM_ADDR routed, GM_ADDR shared,
    Addresses a, int64_t rows, int64_t blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);
    SetSysWorkspace(a.values[kSystem]);
    TPipe pipe;
    device::Epilogue(pipe, routed, shared, a.values[kScaledRouted],
                     a.values[kCombined], rows, blocks);
    device::EndPhase(pipe);
}

Addresses Pack(void** scratch)
{
    Addresses result{};
    for (int64_t i = 0; i < kBuffers; ++i) {
        result.values[i] = (GM_ADDR)scratch[i];
    }
    return result;
}
}  // namespace

extern "C" void ffn_bridge_launch(void* stream, uint64_t sync_base,
    void* input, void* residual, void* gamma, void* gate, void* bias,
    void** scratch, int64_t rows, int64_t blocks, float epsilon)
{
    Bridge<<<blocks, nullptr, stream>>>(sync_base, (GM_ADDR)input,
        (GM_ADDR)residual, (GM_ADDR)gamma, (GM_ADDR)gate, (GM_ADDR)bias,
        Pack(scratch), rows, blocks, epsilon);
}

extern "C" void ffn_bridge_topk_launch(void* stream, uint64_t sync_base,
    void* logits, void* bias, void** scratch, int64_t rows, int64_t blocks)
{
    FixedTopK<<<blocks, nullptr, stream>>>(sync_base, (GM_ADDR)logits,
        (GM_ADDR)bias, Pack(scratch), rows);
}

extern "C" void ffn_bridge_epilogue_launch(void* stream, uint64_t sync_base,
    void* routed, void* shared, void** scratch, int64_t rows, int64_t blocks)
{
    Finalize<<<blocks, nullptr, stream>>>(sync_base, (GM_ADDR)routed,
        (GM_ADDR)shared, Pack(scratch), rows, blocks);
}
