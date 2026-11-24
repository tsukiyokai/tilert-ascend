#include "shared_w8_native.h"

using namespace AscendC;

extern "C" __global__ __mix__(1, 2) __schedmode__(1)
__aicore__ void shared_w8_fused(uint64_t sync_base,
    __gm__ int8_t* input, __gm__ float* input_scale,
    __gm__ int8_t* gate_weight, __gm__ float* gate_scale,
    __gm__ int8_t* down_weight, __gm__ float* down_scale,
    __gm__ uint8_t* scratch, __gm__ uint8_t* tiling,
    __gm__ bfloat16_t* output, int64_t rows, int64_t capacity, int64_t blocks,
    uint64_t partial_offset, uint64_t middle_offset, uint64_t scale_offset,
    uint64_t down_offset, uint64_t system_offset)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);
    const shared_w8::Layout layout{0, partial_offset, middle_offset, scale_offset,
                                  down_offset, system_offset, 0};
    SetSysWorkspace(scratch + layout.system);
    auto* staged = reinterpret_cast<__gm__ int8_t*>(scratch + layout.staged);
    auto* partial = reinterpret_cast<__gm__ int32_t*>(scratch + layout.partial);
    auto* middle = reinterpret_cast<__gm__ int8_t*>(scratch + layout.middle);
    auto* scales = reinterpret_cast<__gm__ float*>(scratch + layout.scales);
    auto* down = reinterpret_cast<__gm__ int32_t*>(scratch + layout.down);
    TPipe pipe;
    shared_w8::device::Buffers held;
    if ASCEND_IS_AIV {
        held.Init(pipe);
        shared_w8::device::StageInput(held, input, staged, rows, capacity, 2 * blocks);
        shared_w8::device::Publish();
    }
    SyncAll<false>();
    if ASCEND_IS_AIC {
        shared_w8::device::Project(pipe, staged, gate_weight, partial, tiling,
                                   capacity, blocks, true);
        shared_w8::device::Publish();
    }
    SyncAll<false>();
    if ASCEND_IS_AIV {
        shared_w8::device::Activate(held, partial, gate_scale, input_scale,
                                   middle, scales, rows, capacity, 2 * blocks);
        shared_w8::device::Publish();
    }
    SyncAll<false>();
    if ASCEND_IS_AIC {
        pipe.Reset();
        shared_w8::device::Project(pipe, middle, down_weight, down,
            tiling + shared_w8::kTilingStride, capacity, blocks, false);
        shared_w8::device::Publish();
    }
    SyncAll<false>();
    if ASCEND_IS_AIV {
        mla_tail::device::DequantizeOutputRows<
            mla_tail::device::ScaleOrder::CombinedScales>(
            pipe, down, down_scale, scales,
            nullptr, 0, output, rows, capacity, GetBlockIdx(), 2 * blocks);
        shared_w8::device::Publish();
    }
}

extern "C" void shared_w8_launch(uint32_t blocks, void* stream, uint64_t sync_base,
    void* input, void* input_scale, void* gate_weight, void* gate_scale,
    void* down_weight, void* down_scale, void* scratch, void* tiling,
    void* output, int64_t rows, int64_t capacity)
{
    const auto layout = shared_w8::MakeLayout(capacity);
    shared_w8_fused<<<blocks, nullptr, stream>>>(sync_base,
        (__gm__ int8_t*)input, (__gm__ float*)input_scale,
        (__gm__ int8_t*)gate_weight, (__gm__ float*)gate_scale,
        (__gm__ int8_t*)down_weight, (__gm__ float*)down_scale,
        (__gm__ uint8_t*)scratch, (__gm__ uint8_t*)tiling,
        (__gm__ bfloat16_t*)output, rows, capacity, blocks,
        layout.partial, layout.middle, layout.scales, layout.down, layout.system);
}
