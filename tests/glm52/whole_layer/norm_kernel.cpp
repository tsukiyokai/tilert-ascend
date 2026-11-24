#include "norm_native.h"

using namespace AscendC;

extern "C" __global__ __mix__(1, 2) __schedmode__(1)
__aicore__ void whole_layer_norm_fused(
    uint64_t sync_base, __gm__ bfloat16_t* input,
    __gm__ bfloat16_t* residual, __gm__ bfloat16_t* gamma,
    __gm__ bfloat16_t* beta, __gm__ bfloat16_t* normalized,
    __gm__ bfloat16_t* retained, __gm__ float* rstd,
    __gm__ int8_t* quantized, __gm__ float* scales,
    int64_t rows, int64_t blocks, float epsilon)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);
    TPipe pipe;
    whole_layer_norm::device::Forward(pipe, input, residual, gamma, beta,
        normalized, retained, rstd, quantized, scales, rows, blocks, epsilon);
}

void whole_layer_norm_launch(
    uint32_t blocks, void* stream, uint64_t sync_base,
    void* input, void* residual, void* gamma, void* beta,
    void* normalized, void* retained, float* rstd,
    void* quantized, float* scales, int64_t rows, float epsilon)
{
    whole_layer_norm_fused<<<blocks, nullptr, stream>>>(sync_base,
        (__gm__ bfloat16_t*)input, (__gm__ bfloat16_t*)residual,
        (__gm__ bfloat16_t*)gamma, (__gm__ bfloat16_t*)beta,
        (__gm__ bfloat16_t*)normalized, (__gm__ bfloat16_t*)retained, rstd,
        (__gm__ int8_t*)quantized, scales, rows, blocks, epsilon);
}
