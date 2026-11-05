#include "mla_tail_dequant.h"

using namespace AscendC;
using namespace mla_tail;

extern "C" __global__ __mix__(1, 2) __schedmode__(1)
__aicore__ void mla_tail_batched_dequant(
    uint64_t sync_base, __gm__ bfloat16_t* latent,
    __gm__ bfloat16_t* value_weight, __gm__ int8_t* output_weight,
    __gm__ float* output_scale, __gm__ uint8_t* bias, int bias_kind,
    __gm__ uint8_t* scratch, __gm__ uint8_t* tiling,
    __gm__ bfloat16_t* output, int64_t rows, int64_t capacity, int64_t blocks,
    uint64_t values_offset, uint64_t quantized_offset,
    uint64_t scales_offset, uint64_t accumulators_offset, uint64_t system_offset)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);
    SetSysWorkspace(scratch + system_offset);
    TPipe pipe;
    auto* heads = reinterpret_cast<__gm__ bfloat16_t*>(scratch);
    auto* values = reinterpret_cast<__gm__ float*>(scratch + values_offset);
    auto* quantized = reinterpret_cast<__gm__ int8_t*>(scratch + quantized_offset);
    auto* scales = reinterpret_cast<__gm__ float*>(scratch + scales_offset);
    auto* accumulators =
        reinterpret_cast<__gm__ int32_t*>(scratch + accumulators_offset);
    device::VectorWorkspace held;
    if ASCEND_IS_AIV {
        held.Init(pipe);
        device::StageHeads(held, latent, heads, rows, capacity,
                            GetBlockIdx(), 2 * blocks);
        device::Publish();
    }
    SyncAll<false>();
    if ASCEND_IS_AIC {
        device::ProjectValues(pipe, heads, value_weight, values, tiling,
                               capacity, blocks);
        device::Publish();
    }
    SyncAll<false>();
    if ASCEND_IS_AIV {
        device::QuantizeValues(held, values, quantized, scales, capacity,
                                GetBlockIdx(), 2 * blocks);
        device::Publish();
    }
    SyncAll<false>();
    if ASCEND_IS_AIC {
        pipe.Reset();
        device::ProjectOutput(pipe, quantized, output_weight, accumulators,
                               tiling + kCubeTilingBytes, capacity, blocks);
        device::Publish();
    }
    SyncAll<false>();
    if ASCEND_IS_AIV {
        device::DequantizeOutputRows(pipe, accumulators, output_scale, scales,
            bias, bias_kind, output, rows, capacity, GetBlockIdx(), 2 * blocks);
        device::Publish();
    }
}

extern "C" void mla_tail_dequant_launch(
    uint32_t blocks, void* stream, uint64_t sync_base,
    void* latent, void* value_weight, void* output_weight,
    void* output_scale, void* bias, int bias_kind, void* scratch,
    void* tiling, void* output, int64_t rows, int64_t capacity)
{
    const WorkspaceLayout layout = MakeWorkspaceLayout(capacity);
    mla_tail_batched_dequant<<<blocks, nullptr, stream>>>(
        sync_base, (__gm__ bfloat16_t*)latent,
        (__gm__ bfloat16_t*)value_weight, (__gm__ int8_t*)output_weight,
        (__gm__ float*)output_scale, (__gm__ uint8_t*)bias, bias_kind,
        (__gm__ uint8_t*)scratch, (__gm__ uint8_t*)tiling,
        (__gm__ bfloat16_t*)output, rows, capacity, blocks, layout.values,
        layout.quantized, layout.token_scales, layout.accumulators, layout.system);
}
