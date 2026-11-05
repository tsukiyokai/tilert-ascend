#include "attention_indexed_native.h"

using namespace AscendC;
using namespace whole_layer_indexed;

extern "C" __global__ __mix__(1, 2) __schedmode__(1) __aicore__
void whole_layer_indexed_attention(uint64_t sync_base, Config config,
                                   DeviceArgs args, int bias_kind)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);
    TPipe pipe;
    Forward(pipe, config, args, bias_kind);
}

void whole_layer_indexed_attention_launch(
    void* stream, uint64_t sync_base, Config config, Addresses a, int bias_kind)
{
    DeviceArgs args{};
    args.hidden = (GM_ADDR)a.hidden;
    args.token_x = (GM_ADDR)a.token_x;
    args.token_scale = (GM_ADDR)a.token_scale;
    for (int index = 0; index < 9; ++index) {
        args.main_weights[index] = (GM_ADDR)a.main_weights[index];
    }
    for (int index = 0; index < 6; ++index) {
        args.front_weights[index] = (GM_ADDR)a.front_weights[index];
        args.main_metadata[index] = (GM_ADDR)a.main_metadata[index];
        args.indexer_metadata[index] = (GM_ADDR)a.indexer_metadata[index];
    }
    for (int index = 0; index < 3; ++index) {
        args.tail_weights[index] = (GM_ADDR)a.tail_weights[index];
    }
    for (int index = 0; index < 4; ++index) {
        args.caches[index] = (GM_ADDR)a.caches[index];
    }
    args.bias = (GM_ADDR)a.bias;
    args.scratch = (GM_ADDR)a.scratch;
    args.tiling = (GM_ADDR)a.tiling;
    args.output = (GM_ADDR)a.output;
    whole_layer_indexed_attention<<<config.attention.blocks, nullptr, stream>>>(
        sync_base, config, args, bias_kind);
}
