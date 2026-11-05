#ifndef TESTS_GLM52_WHOLE_LAYER_ATTENTION_NORM_CONTRACT_H
#define TESTS_GLM52_WHOLE_LAYER_ATTENTION_NORM_CONTRACT_H

#include "attention_full_contract.h"

namespace whole_layer_attention {

struct NormConfig {
    FullConfig full;
    float epsilon;
    uint32_t reserved;
    uint64_t scale_offset;
    uint64_t full_offset;
    uint64_t rstd_offset;
    uint64_t norm_temporary_bytes;
    uint64_t bytes;
};

inline NormConfig MakeNormConfig(const Config& attention, float epsilon)
{
    NormConfig config{};
    config.full = MakeFullConfig(attention);
    config.epsilon = epsilon;
    // Codes and scales survive the norm stage and remain outside all later
    // attention/tail arenas. The normalized BF16 values and rstd die earlier.
    config.scale_offset = Align(attention.rows * kHidden);
    config.full_offset = config.scale_offset + Align(attention.rows * 4ULL);
    config.rstd_offset = Align(attention.rows * kHidden * 2ULL);
    config.norm_temporary_bytes = config.rstd_offset + Align(attention.rows * 4ULL);
    const uint64_t arena = config.full.bytes > config.norm_temporary_bytes
        ? config.full.bytes : config.norm_temporary_bytes;
    config.bytes = config.full_offset + arena;
    return config;
}

}  // namespace whole_layer_attention

void whole_layer_norm_attention_launch(
    void* stream, uint64_t sync_base, whole_layer_attention::NormConfig config,
    void* input, void* residual, void* norm_gamma, void* norm_beta, void* retained,
    void* weight_dq, void* weight_uq_qr, void* weight_uk, void* weight_dkv_kr,
    void* scale_dq, void* scale_uq_qr, void* scale_dkv_kr,
    void* gamma_cq, void* gamma_ckv, void* sin, void* cos, void* slots,
    void* kv_cache, void* rope_cache, void* topk, void* query_lengths,
    void* kv_lengths, void* page_table, void* value_weight, void* output_weight,
    void* output_scale, void* bias, int bias_kind, void* scratch,
    void* tail_tiling, void* output);

#endif
