#ifndef TILERT_FFN_TOKEN_SPLIT_GEOMETRY_H
#define TILERT_FFN_TOKEN_SPLIT_GEOMETRY_H

#include <cstdint>

namespace ffn_token_split {

constexpr int64_t kOutputHidden = 6144;
constexpr int64_t kOutputTopK = 8;
constexpr int64_t kOutputMaximumRows = 32;
constexpr int64_t kOutputRoutes = kOutputMaximumRows * kOutputTopK;
constexpr int64_t kOutputTile = 1536;
constexpr int64_t kOutputTiles = kOutputHidden / kOutputTile;
constexpr int64_t kOutputMetadataBytes = 3 * kOutputRoutes * sizeof(int32_t);
constexpr int64_t kOutputResidentBytes = 112352 + 1056 + kOutputMetadataBytes;

#if defined(__CCE__)
#define TOKEN_OUTPUT_DEVICE __host__ __aicore__
#else
#define TOKEN_OUTPUT_DEVICE
#endif

TOKEN_OUTPUT_DEVICE constexpr int64_t OutputToken(int64_t task)
{
    return task / kOutputTiles;
}

TOKEN_OUTPUT_DEVICE constexpr int64_t OutputColumn(int64_t task)
{
    return (task % kOutputTiles) * kOutputTile;
}

TOKEN_OUTPUT_DEVICE constexpr int64_t ExpertRow(int64_t route, int64_t begin)
{
    return route - begin;
}

TOKEN_OUTPUT_DEVICE constexpr int64_t OutputOwner(int64_t token,
                                                 int64_t tile, int64_t blocks)
{
    return (token * kOutputTiles + tile) % (2 * blocks);
}

static_assert(kOutputHidden % kOutputTile == 0);
static_assert(kOutputTile % 256 == 0);
static_assert(kOutputMetadataBytes == 3072);
static_assert(kOutputResidentBytes == 116480);
static_assert(kOutputResidentBytes < 192 * 1024);

#undef TOKEN_OUTPUT_DEVICE

}  // namespace ffn_token_split

#endif
