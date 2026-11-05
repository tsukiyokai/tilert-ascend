#ifndef TESTS_GLM52_INDEXER_FRONT_CONTRACT_H
#define TESTS_GLM52_INDEXER_FRONT_CONTRACT_H

#include <cstdint>

namespace indexer_front {

constexpr int64_t kHidden = 6144;
constexpr int64_t kQueryRank = 2048;
constexpr int64_t kHeads = 32;
constexpr int64_t kDim = 128;
constexpr int64_t kRope = 64;
constexpr int64_t kQueryColumns = kHeads * kDim;
constexpr int64_t kQueryTile = 256;
constexpr int64_t kKeyTile = 32;
constexpr int64_t kKeyColumns = kDim + kHeads;
constexpr int64_t kHadamardRows = 32;
constexpr uint64_t kTilingStride = 512;
constexpr uint64_t kTilingBytes = 3 * kTilingStride;
constexpr uint64_t kSystemBytes = 16ULL << 20;

constexpr uint64_t Align(uint64_t value)
{
    return (value + 511) / 512 * 512;
}

struct Config {
    int64_t rows, capacity, blocks, quantized, query_scale_bf16;
    uint64_t query_acc, key_acc, query_projection, key_projection;
    uint64_t key_normalized, query_rope, key_rope, norm_stats;
    uint64_t query_hadamard, key_hadamard, system, bytes;
};

constexpr Config MakeConfig(int64_t rows, int64_t blocks, bool quantized,
                            bool query_scale_bf16 = false)
{
    Config c{};
    c.rows = rows;
    c.capacity = rows <= 16 ? 16 : 32;
    c.blocks = blocks;
    c.quantized = quantized;
    c.query_scale_bf16 = query_scale_bf16;
    c.key_acc = Align(c.capacity * kQueryColumns * 4);
    c.query_projection = Align(c.key_acc + c.capacity * kKeyColumns * 4);
    c.key_projection = Align(c.query_projection + rows * kQueryColumns * 2);
    c.key_normalized = Align(c.key_projection + rows * kKeyColumns * 2);
    c.query_rope = Align(c.key_normalized + rows * kDim * 2);
    c.key_rope = Align(c.query_rope + rows * kQueryColumns * 2);
    c.norm_stats = Align(c.key_rope + rows * kDim * 2);
    c.query_hadamard = Align(c.norm_stats + rows * 2 * 4);
    c.key_hadamard = Align(c.query_hadamard + rows * kQueryColumns * 4);
    c.system = Align(c.key_hadamard + c.capacity * kDim * 4);
    c.bytes = c.system + kSystemBytes;
    return c;
}

struct Addresses {
    void* hidden;
    void* query_norm;
    void* query_norm_scale;
    void* query_weight;
    void* query_weight_scale;
    void* key_weight;
    void* gamma;
    void* beta;
    void* cos;
    void* sin;
    void* hadamard;
    void* scratch;
    void* tiling;
    void* query;
    void* key;
    void* weights;
    void* query_scale;
    void* key_scale;
};

}  // namespace indexer_front

extern "C" int indexer_front_build_tiling(int capacity, void* buffer);
void indexer_front_launch(void* stream, uint64_t sync_base,
                          indexer_front::Config config,
                          indexer_front::Addresses addresses);

#endif
