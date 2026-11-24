#ifndef TESTS_GLM52_WHOLE_LAYER_ATTENTION_CONTRACT_H
#define TESTS_GLM52_WHOLE_LAYER_ATTENTION_CONTRACT_H

#include <cstdint>

namespace whole_layer_attention_terminal {

constexpr uint32_t kHidden = 6144;
constexpr uint32_t kQueryRank = 2048;
constexpr uint32_t kHeads = 4;
constexpr uint32_t kNope = 192;
constexpr uint32_t kRope = 64;
constexpr uint32_t kLatent = 512;
constexpr uint32_t kPage = 128;
constexpr uint32_t kTopK = 2048;
constexpr uint32_t kMaximumRows = 32;
constexpr uint64_t kAlignment = 512;

struct Config {
    uint32_t rows;
    uint32_t pages;
    uint32_t table_pages;
    uint32_t blocks;
    float scale;
    float epsilon_cq;
    float epsilon_ckv;
    uint32_t s2_base;
    uint32_t mm_elements;
    uint32_t bmm_elements;
    uint64_t query_offset;
    uint64_t rope_offset;
    uint64_t work_offset;
    uint64_t prolog_bytes;
    uint64_t sfa_bytes;
    uint64_t bytes;
};

constexpr uint64_t Align(uint64_t value, uint64_t alignment = kAlignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

// Prolog and SFA workspace lifetimes are separated by an all-core barrier.
// Their shared arena excludes the query and query-RoPE tensors kept by SFA.
inline Config MakeConfig(uint32_t rows, uint32_t pages, uint32_t table_pages,
                         uint32_t blocks, float scale, float epsilon_cq,
                         float epsilon_ckv)
{
    Config config{};
    config.rows = rows;
    config.pages = pages;
    config.table_pages = table_pages;
    config.blocks = blocks;
    config.scale = scale;
    config.epsilon_cq = epsilon_cq;
    config.epsilon_ckv = epsilon_ckv;
    const uint32_t capacity = table_pages * kPage;
    config.s2_base = capacity < 512 ? capacity : 512;
    const uint32_t matrix_rows = static_cast<uint32_t>(Align(rows * kHeads, 16));
    config.mm_elements = matrix_rows * static_cast<uint32_t>(Align(config.s2_base, 32));
    config.bmm_elements = matrix_rows * kLatent;
    config.query_offset = 0;
    config.rope_offset = Align(rows * kHeads * kLatent * 2ULL);
    config.work_offset = Align(config.rope_offset + rows * kHeads * kRope * 2ULL);
    // Vendor host reserves BF16-sized query-norm storage even for W8 output.
    config.prolog_bytes = Align(rows * 20224ULL);
    // V-template: double-buffered QK/P/PV, four merged-KV tiles and valid IDs.
    config.sfa_bytes = Align(blocks * (12ULL * config.mm_elements +
        8ULL * config.bmm_elements + 512ULL * 576 * 4 * 2 + 2ULL * 128 * 4 * 4));
    const uint64_t work = config.prolog_bytes > config.sfa_bytes
        ? config.prolog_bytes : config.sfa_bytes;
    config.bytes = config.work_offset + work;
    return config;
}

}  // namespace whole_layer_attention_terminal

void whole_layer_attention_terminal_launch(
    void* stream, uint64_t sync_base, whole_layer_attention_terminal::Config config,
    void* token_x, void* token_scale, void* weight_dq, void* weight_uq_qr,
    void* weight_uk, void* weight_dkv_kr, void* scale_dq, void* scale_uq_qr,
    void* scale_dkv_kr, void* gamma_cq, void* gamma_ckv, void* sin,
    void* cos, void* slots, void* kv_cache, void* rope_cache, void* topk,
    void* query_lengths, void* kv_lengths, void* page_table,
    void* scratch, void* output);

#endif
