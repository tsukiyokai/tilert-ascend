#include "attention_norm_native.h"

using namespace AscendC;
using namespace whole_layer_attention;

extern "C" __global__ __mix__(1, 2) __schedmode__(1) __aicore__
void whole_layer_norm_attention(
    uint64_t sync_base, NormConfig config, GM_ADDR input, GM_ADDR residual,
    GM_ADDR norm_gamma, GM_ADDR norm_beta, GM_ADDR retained,
    GM_ADDR weight_dq, GM_ADDR weight_uq_qr, GM_ADDR weight_uk,
    GM_ADDR weight_dkv_kr, GM_ADDR scale_dq, GM_ADDR scale_uq_qr,
    GM_ADDR scale_dkv_kr, GM_ADDR gamma_cq, GM_ADDR gamma_ckv,
    GM_ADDR sin, GM_ADDR cos, GM_ADDR slots, GM_ADDR kv_cache, GM_ADDR rope_cache,
    GM_ADDR topk, GM_ADDR query_lengths, GM_ADDR kv_lengths, GM_ADDR page_table,
    GM_ADDR value_weight, GM_ADDR output_weight, GM_ADDR output_scale,
    GM_ADDR bias, int bias_kind, GM_ADDR scratch, GM_ADDR tail_tiling, GM_ADDR output)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync_base);
    TPipe pipe;
    const DeviceArgs args{nullptr, nullptr, weight_dq, weight_uq_qr,
        weight_uk, weight_dkv_kr, scale_dq, scale_uq_qr, scale_dkv_kr,
        gamma_cq, gamma_ckv, sin, cos, slots, kv_cache, rope_cache, topk,
        query_lengths, kv_lengths, page_table, nullptr, nullptr};
    NormForward(pipe, config, args, input, residual, norm_gamma, norm_beta,
        retained, value_weight, output_weight, output_scale, bias, bias_kind,
        scratch, tail_tiling, output);
}

void whole_layer_norm_attention_launch(
    void* stream, uint64_t sync_base, NormConfig config,
    void* input, void* residual, void* norm_gamma, void* norm_beta, void* retained,
    void* weight_dq, void* weight_uq_qr, void* weight_uk, void* weight_dkv_kr,
    void* scale_dq, void* scale_uq_qr, void* scale_dkv_kr,
    void* gamma_cq, void* gamma_ckv, void* sin, void* cos, void* slots,
    void* kv_cache, void* rope_cache, void* topk, void* query_lengths,
    void* kv_lengths, void* page_table, void* value_weight, void* output_weight,
    void* output_scale, void* bias, int bias_kind, void* scratch,
    void* tail_tiling, void* output)
{
    whole_layer_norm_attention<<<config.full.attention.blocks, nullptr, stream>>>(
        sync_base, config, (GM_ADDR)input, (GM_ADDR)residual, (GM_ADDR)norm_gamma,
        (GM_ADDR)norm_beta, (GM_ADDR)retained,
        (GM_ADDR)weight_dq, (GM_ADDR)weight_uq_qr, (GM_ADDR)weight_uk,
        (GM_ADDR)weight_dkv_kr, (GM_ADDR)scale_dq, (GM_ADDR)scale_uq_qr,
        (GM_ADDR)scale_dkv_kr, (GM_ADDR)gamma_cq, (GM_ADDR)gamma_ckv,
        (GM_ADDR)sin, (GM_ADDR)cos, (GM_ADDR)slots, (GM_ADDR)kv_cache,
        (GM_ADDR)rope_cache, (GM_ADDR)topk, (GM_ADDR)query_lengths,
        (GM_ADDR)kv_lengths, (GM_ADDR)page_table, (GM_ADDR)value_weight,
        (GM_ADDR)output_weight, (GM_ADDR)output_scale, (GM_ADDR)bias,
        bias_kind, (GM_ADDR)scratch, (GM_ADDR)tail_tiling, (GM_ADDR)output);
}
