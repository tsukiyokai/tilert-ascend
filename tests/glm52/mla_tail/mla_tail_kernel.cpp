// Standalone launch of the composable MLA projection tail.
#include "mla_tail_native.h"

using namespace AscendC;
using namespace mla_tail;

#define DEFINE_MLA_TAIL_KERNEL(NAME, HEADS)                                 \
extern "C" __global__ __mix__(1, 2) __schedmode__(1)                        \
__aicore__ void NAME(                                                       \
    uint64_t sync_base, __gm__ bfloat16_t* latent,                          \
    __gm__ bfloat16_t* value_weight, __gm__ int8_t* output_weight,          \
    __gm__ float* output_scale, __gm__ uint8_t* bias, int bias_kind,        \
    __gm__ uint8_t* scratch, __gm__ uint8_t* tiling,                        \
    __gm__ bfloat16_t* output, int64_t rows, int64_t capacity,              \
    int64_t blocks, uint64_t values_offset, uint64_t quantized_offset,      \
    uint64_t scales_offset, uint64_t accumulators_offset,                   \
    uint64_t system_offset)                                                 \
{                                                                           \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);                      \
    InitSocState();                                                         \
    SetSyncBaseAddr(sync_base);                                             \
    TPipe pipe;                                                             \
    const WorkspaceLayout layout{0, values_offset, quantized_offset,        \
        scales_offset, accumulators_offset, system_offset, 0};              \
    mla_tail::device::Forward<HEADS>(pipe, latent, value_weight,            \
        output_weight, output_scale, bias, bias_kind, scratch, tiling,      \
        output, rows, capacity, blocks, layout);                            \
}

DEFINE_MLA_TAIL_KERNEL(mla_tail_fused, 4)
DEFINE_MLA_TAIL_KERNEL(mla_tail_fused_h8, 8)

void mla_tail_launch(uint32_t blocks, void* stream, uint64_t sync_base,
                     void* latent, void* value_weight, void* output_weight,
                     float* output_scale, void* bias, int bias_kind,
                     void* scratch, void* tiling, void* output,
                     int64_t rows, int64_t capacity, int64_t heads)
{
    const WorkspaceLayout layout = MakeWorkspaceLayout(capacity, heads);
    if (heads == 8) {
        mla_tail_fused_h8<<<blocks, nullptr, stream>>>(
            sync_base, (__gm__ bfloat16_t*)latent,
            (__gm__ bfloat16_t*)value_weight,
            (__gm__ int8_t*)output_weight, output_scale,
            (__gm__ uint8_t*)bias, bias_kind,
            (__gm__ uint8_t*)scratch, (__gm__ uint8_t*)tiling,
            (__gm__ bfloat16_t*)output, rows, capacity, blocks,
            layout.values, layout.quantized, layout.token_scales,
            layout.accumulators, layout.system);
        return;
    }
    mla_tail_fused<<<blocks, nullptr, stream>>>(
        sync_base, (__gm__ bfloat16_t*)latent,
        (__gm__ bfloat16_t*)value_weight,
        (__gm__ int8_t*)output_weight, output_scale,
        (__gm__ uint8_t*)bias, bias_kind,
        (__gm__ uint8_t*)scratch, (__gm__ uint8_t*)tiling,
        (__gm__ bfloat16_t*)output, rows, capacity, blocks,
        layout.values, layout.quantized, layout.token_scales,
        layout.accumulators, layout.system);
}
