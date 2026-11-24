#ifndef TESTS_GLM52_WHOLE_LAYER_ATTENTION_INDEXED_CONTRACT_H
#define TESTS_GLM52_WHOLE_LAYER_ATTENTION_INDEXED_CONTRACT_H

#include "attention_contract.h"
#include "indexer_front_contract.h"
#include "indexer_contract.h"
#include "../mla_tail/mla_tail.h"

namespace whole_layer_indexed {

constexpr uint64_t kTilingBytes = indexer_front::kTilingBytes +
                                 mla_tail::kTilingBytes;

struct Config {
    whole_layer_attention::Config attention;
    indexer_front::Config front;
    whole_layer_indexer::Config indexer;
    mla_tail::WorkspaceLayout tail;
    uint64_t query_norm, query_norm_scale, query, key, weights;
    uint64_t query_scale, key_scale, topk, latent;
    uint64_t attention_offset, front_offset, indexer_offset, tail_offset, bytes;
};

inline Config MakeConfig(const whole_layer_attention::Config& attention,
                          uint32_t index_pages, uint32_t index_table_pages,
                          bool quantized, bool query_scale_bf16 = false)
{
    using whole_layer_attention::Align;
    Config c{};
    c.attention = attention;
    const uint64_t rows = attention.rows;
    c.front = indexer_front::MakeConfig(rows, attention.blocks, quantized,
                                       query_scale_bf16);
    c.indexer = whole_layer_indexer::MakeConfig(rows, index_pages,
        index_table_pages, attention.blocks, quantized);
    c.tail = mla_tail::MakeWorkspaceLayout(c.front.capacity);
    const uint64_t encoded_bytes = quantized ? 1 : 2;
    c.query_norm_scale = Align(rows * 2048);
    c.query = Align(c.query_norm_scale + rows * 4);
    c.key = Align(c.query + rows * 4096 * encoded_bytes);
    c.weights = Align(c.key + rows * 128 * encoded_bytes);
    c.query_scale = Align(c.weights + rows * 32 * 2);
    c.key_scale = Align(c.query_scale + rows * 32 * 2);
    c.topk = Align(c.key_scale + rows * 2);
    c.latent = Align(c.topk + rows * 2048 * 4);
    c.attention_offset = Align(c.latent + rows * 4 * 512 * 2);
    c.front_offset = Align(c.attention_offset + attention.bytes);
    c.indexer_offset = Align(c.front_offset + c.front.bytes);
    c.tail_offset = Align(c.indexer_offset + c.indexer.bytes);
    c.bytes = c.tail_offset + c.tail.bytes;
    return c;
}

struct Addresses {
    void* hidden;
    void* token_x;
    void* token_scale;
    void* main_weights[9];
    void* front_weights[6];
    void* tail_weights[3];
    void* bias;
    void* main_metadata[6];
    void* indexer_metadata[6];
    void* caches[4];
    void* scratch;
    void* tiling;
    void* output;
};

}  // namespace whole_layer_indexed

void whole_layer_indexed_attention_launch(
    void* stream, uint64_t sync_base, whole_layer_indexed::Config config,
    whole_layer_indexed::Addresses addresses, int bias_kind);

#endif
