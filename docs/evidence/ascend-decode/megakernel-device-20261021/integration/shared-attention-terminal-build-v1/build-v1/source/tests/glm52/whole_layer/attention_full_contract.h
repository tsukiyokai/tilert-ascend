#ifndef TESTS_GLM52_WHOLE_LAYER_ATTENTION_FULL_CONTRACT_H
#define TESTS_GLM52_WHOLE_LAYER_ATTENTION_FULL_CONTRACT_H

#include "attention_contract.h"
#include "../mla_tail/mla_tail.h"

namespace whole_layer_attention_terminal {

struct FullConfig {
    Config attention;
    mla_tail_attention_terminal::WorkspaceLayout tail;
    uint32_t capacity;
    uint32_t reserved;
    uint64_t arena_offset;
    uint64_t bytes;
};

inline FullConfig MakeFullConfig(const Config& attention)
{
    FullConfig config{};
    config.attention = attention;
    config.capacity = attention.rows <= 16 ? 16 : 32;
    config.tail = mla_tail_attention_terminal::MakeWorkspaceLayout(config.capacity);
    // SFA's latent survives the attention arena and becomes the tail's input.
    config.arena_offset = Align(attention.rows * kHeads * kLatent * 2ULL);
    config.bytes = config.arena_offset +
        (attention.bytes > config.tail.bytes ? attention.bytes : config.tail.bytes);
    return config;
}

}  // namespace whole_layer_attention_terminal

void whole_layer_full_attention_terminal_launch(
    void* stream, uint64_t sync_base, whole_layer_attention_terminal::FullConfig config,
    void* token_x, void* token_scale, void* weight_dq, void* weight_uq_qr,
    void* weight_uk, void* weight_dkv_kr, void* scale_dq, void* scale_uq_qr,
    void* scale_dkv_kr, void* gamma_cq, void* gamma_ckv, void* sin,
    void* cos, void* slots, void* kv_cache, void* rope_cache, void* topk,
    void* query_lengths, void* kv_lengths, void* page_table,
    void* value_weight, void* output_weight, void* output_scale, void* bias,
    int bias_kind, void* scratch, void* tail_tiling, void* output);

#endif
