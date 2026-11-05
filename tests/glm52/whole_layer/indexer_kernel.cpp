#include "indexer_native.h"

using namespace AscendC;
using namespace whole_layer_indexer;

extern "C" __global__ __mix__(1, 2) __schedmode__(1) __aicore__
void whole_layer_indexer_kernel(
    uint64_t sync_base, Config config,
    GM_ADDR query, GM_ADDR new_keys, GM_ADDR weights, GM_ADDR query_scale,
    GM_ADDR new_key_scale, GM_ADDR slots, GM_ADDR key_cache, GM_ADDR scale_cache,
    GM_ADDR query_lengths, GM_ADDR key_lengths, GM_ADDR page_table,
    GM_ADDR scratch, GM_ADDR output)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);
    TPipe pipe;
    const DeviceArgs args{query, new_keys, weights, query_scale, new_key_scale,
        slots, key_cache, scale_cache, query_lengths, key_lengths, page_table,
        scratch, output};
    Forward(pipe, config, args);
}

void whole_layer_indexer_launch(
    void* stream, uint64_t sync_base, Config config,
    void* query, void* new_keys, void* weights, void* query_scale,
    void* new_key_scale, void* slots, void* key_cache, void* scale_cache,
    void* query_lengths, void* key_lengths, void* page_table,
    void* scratch, void* output)
{
    whole_layer_indexer_kernel<<<config.blocks, nullptr, stream>>>(sync_base, config,
        (GM_ADDR)query, (GM_ADDR)new_keys, (GM_ADDR)weights, (GM_ADDR)query_scale,
        (GM_ADDR)new_key_scale, (GM_ADDR)slots, (GM_ADDR)key_cache,
        (GM_ADDR)scale_cache, (GM_ADDR)query_lengths, (GM_ADDR)key_lengths,
        (GM_ADDR)page_table, (GM_ADDR)scratch, (GM_ADDR)output);
}
