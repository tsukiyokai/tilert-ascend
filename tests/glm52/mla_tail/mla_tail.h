#ifndef TESTS_GLM52_MLA_TAIL_MLA_TAIL_H
#define TESTS_GLM52_MLA_TAIL_MLA_TAIL_H

#include <cstdint>

namespace mla_tail {

#ifndef ATTENTION_HEADS
constexpr int64_t kHeads = 4;
#else
constexpr int64_t kHeads = ATTENTION_HEADS;
#endif
constexpr int64_t kLatent = 512;
constexpr int64_t kValue = 256;
constexpr int64_t kProjectionInput = kHeads * kValue;
constexpr int64_t kHidden = 6144;
constexpr int64_t kValueTiles = 16;
constexpr int64_t kValueTile = kProjectionInput / kValueTiles;
constexpr int64_t kOutputTile = 256;
constexpr int64_t kOutputTiles = kHidden / kOutputTile;
constexpr int64_t kScaleStride = 8;
constexpr int64_t kMaximumRows = 32;
constexpr uint64_t kAlignment = 512;
constexpr uint64_t kCubeTilingBytes = 512;
constexpr uint64_t kTilingBytes = 2 * kCubeTilingBytes;
constexpr uint64_t kSystemWorkspaceBytes = 16ULL << 20;

struct WorkspaceLayout {
    uint64_t heads;
    uint64_t values;
    uint64_t quantized;
    uint64_t token_scales;
    uint64_t accumulators;
    uint64_t system;
    uint64_t bytes;
};

constexpr uint64_t Align(uint64_t value)
{
    return (value + kAlignment - 1) / kAlignment * kAlignment;
}

// Rows are padded to complete cube tiles. Every producer writes the padding.
constexpr WorkspaceLayout MakeWorkspaceLayout(int64_t capacity,
                                              int64_t heads = kHeads)
{
    const int64_t projection_input = heads * kValue;
    WorkspaceLayout layout{};
    layout.heads = 0;
    layout.values = Align(capacity * heads * kLatent * 2);
    layout.quantized = Align(layout.values + capacity * projection_input * 4);
    layout.token_scales = Align(layout.quantized + capacity * projection_input);
    layout.accumulators = Align(layout.token_scales + capacity * kScaleStride * 4);
    layout.system = Align(layout.accumulators + capacity * kHidden * 4);
    layout.bytes = layout.system + kSystemWorkspaceBytes;
    return layout;
}

}  // namespace mla_tail

extern "C" int mla_tail_build_tiling(int capacity, void* buffer,
                                     int heads = mla_tail::kHeads);

void mla_tail_launch(uint32_t blocks, void* stream, uint64_t sync_base,
                     void* latent, void* value_weight, void* output_weight,
                     float* output_scale, void* bias, int bias_kind,
                     void* scratch, void* tiling, void* output,
                     int64_t rows, int64_t capacity,
                     int64_t heads = mla_tail::kHeads);

#endif
