#include "indexer_front_native.h"

using namespace AscendC;
using namespace indexer_front;

extern "C" __global__ __mix__(1, 2) __schedmode__(1)
__aicore__ void indexer_front_kernel(uint64_t sync_base, Config config,
                                    device::DeviceArgs args)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);
    TPipe pipe;
    device::Forward(pipe, config, args);
}

void indexer_front_launch(void* stream, uint64_t sync_base, Config config,
                          Addresses a)
{
    const device::DeviceArgs args{
        (GM_ADDR)a.hidden, (GM_ADDR)a.query_norm, (GM_ADDR)a.query_norm_scale,
        (GM_ADDR)a.query_weight, (GM_ADDR)a.query_weight_scale,
        (GM_ADDR)a.key_weight, (GM_ADDR)a.gamma, (GM_ADDR)a.beta,
        (GM_ADDR)a.cos, (GM_ADDR)a.sin, (GM_ADDR)a.hadamard,
        (GM_ADDR)a.scratch, (GM_ADDR)a.tiling, (GM_ADDR)a.query, (GM_ADDR)a.key,
        (GM_ADDR)a.weights, (GM_ADDR)a.query_scale, (GM_ADDR)a.key_scale};
    indexer_front_kernel<<<config.blocks, nullptr, stream>>>(sync_base, config, args);
}
