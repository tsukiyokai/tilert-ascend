#ifndef TESTS_GLM52_WHOLE_LAYER_INDEXER_CONTRACT_H
#define TESTS_GLM52_WHOLE_LAYER_INDEXER_CONTRACT_H

#include <cstdint>

namespace whole_layer_indexer {

constexpr uint32_t kHeads = 32;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kPage = 128;
constexpr uint32_t kTopK = 2048;
constexpr uint32_t kMaximumRows = 32;

struct Config {
    uint32_t rows;
    uint32_t pages;
    uint32_t table_pages;
    uint32_t blocks;
    uint32_t quantized;
    uint64_t mm_bytes;
    uint64_t merge_offset;
    uint64_t merge_bytes;
    uint64_t params_offset;
    uint64_t params_bytes;
    uint64_t weights_offset;
    uint64_t weights_bytes;
    uint64_t bytes;
};

// The arch22 classes use private Cube primitives, not Matmul's system arena.
// Every workspace term is allocated for the launched block count, even when
// the selector assigns computation to fewer groups.
inline Config MakeConfig(uint32_t rows, uint32_t pages, uint32_t table_pages,
                         uint32_t blocks, bool quantized)
{
    Config config{};
    config.rows = rows;
    config.pages = pages;
    config.table_pages = table_pages;
    config.blocks = blocks;
    config.quantized = quantized;
    const uint64_t s1_base = quantized ? 4 : 8;
    const uint64_t mm_rows = quantized ? s1_base : s1_base * kHeads;
    const uint64_t s2_base = quantized ? 2048 : 512;
    config.mm_bytes = blocks * 2ULL * mm_rows * s2_base * sizeof(float);
    config.merge_offset = config.mm_bytes;
    config.merge_bytes = blocks * s1_base * 2 * 2 * kTopK * sizeof(float);
    config.params_offset = config.merge_offset + config.merge_bytes;
    config.params_bytes = blocks * s1_base * 2 * 16 * sizeof(int64_t);
    config.weights_offset = config.params_offset + config.params_bytes;
    config.weights_bytes = quantized
        ? blocks * 16ULL * s1_base * kHeads * 2 * sizeof(uint16_t) : 0;
    config.bytes = config.weights_offset + config.weights_bytes;
    return config;
}

}  // namespace whole_layer_indexer

void whole_layer_indexer_launch(
    void* stream, uint64_t sync_base, whole_layer_indexer::Config config,
    void* query, void* new_keys, void* weights, void* query_scale,
    void* new_key_scale, void* slots, void* key_cache, void* scale_cache,
    void* query_lengths, void* key_lengths, void* page_table,
    void* scratch, void* output);

#endif
