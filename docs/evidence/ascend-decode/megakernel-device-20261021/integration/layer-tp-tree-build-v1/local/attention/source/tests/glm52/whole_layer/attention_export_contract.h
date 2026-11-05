#ifndef TESTS_GLM52_WHOLE_LAYER_ATTENTION_EXPORT_CONTRACT_H
#define TESTS_GLM52_WHOLE_LAYER_ATTENTION_EXPORT_CONTRACT_H

#include "attention_contract.h"

void whole_layer_export_attention_launch(
    void* stream, uint64_t sync_base, whole_layer_attention::Config config,
    void* token_x, void* token_scale, void* weight_dq, void* weight_uq_qr,
    void* weight_uk, void* weight_dkv_kr, void* scale_dq, void* scale_uq_qr,
    void* scale_dkv_kr, void* gamma_cq, void* gamma_ckv, void* sin,
    void* cos, void* slots, void* kv_cache, void* rope_cache, void* topk,
    void* query_lengths, void* kv_lengths, void* page_table,
    void* scratch, void* output, void* query_norm, void* query_norm_scale);

#endif
