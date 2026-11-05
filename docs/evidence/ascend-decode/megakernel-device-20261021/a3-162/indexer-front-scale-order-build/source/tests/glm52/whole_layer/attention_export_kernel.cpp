#include "attention_native.h"
#include "attention_export_contract.h"

using namespace AscendC;
using namespace whole_layer_attention;

extern "C" __global__ __mix__(1, 2) __schedmode__(1) __aicore__
void whole_layer_export_attention(
    uint64_t sync_base, Config config, DeviceArgs args,
    GM_ADDR query_norm, GM_ADDR query_norm_scale)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);
    TPipe pipe;
    PrologPhase(pipe, config, args, query_norm, query_norm_scale);
    SfaPhase(pipe, config, args);
}

void whole_layer_export_attention_launch(
    void* stream, uint64_t sync_base, Config config,
    void* token_x, void* token_scale, void* weight_dq, void* weight_uq_qr,
    void* weight_uk, void* weight_dkv_kr, void* scale_dq, void* scale_uq_qr,
    void* scale_dkv_kr, void* gamma_cq, void* gamma_ckv, void* sin,
    void* cos, void* slots, void* kv_cache, void* rope_cache, void* topk,
    void* query_lengths, void* kv_lengths, void* page_table,
    void* scratch, void* output, void* query_norm, void* query_norm_scale)
{
    const DeviceArgs args{(GM_ADDR)token_x, (GM_ADDR)token_scale,
        (GM_ADDR)weight_dq, (GM_ADDR)weight_uq_qr, (GM_ADDR)weight_uk,
        (GM_ADDR)weight_dkv_kr, (GM_ADDR)scale_dq, (GM_ADDR)scale_uq_qr,
        (GM_ADDR)scale_dkv_kr, (GM_ADDR)gamma_cq, (GM_ADDR)gamma_ckv,
        (GM_ADDR)sin, (GM_ADDR)cos, (GM_ADDR)slots, (GM_ADDR)kv_cache,
        (GM_ADDR)rope_cache, (GM_ADDR)topk, (GM_ADDR)query_lengths,
        (GM_ADDR)kv_lengths, (GM_ADDR)page_table, (GM_ADDR)scratch,
        (GM_ADDR)output};
    whole_layer_export_attention<<<config.blocks, nullptr, stream>>>(
        sync_base, config, args, (GM_ADDR)query_norm,
        (GM_ADDR)query_norm_scale);
}
