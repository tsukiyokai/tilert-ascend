#ifndef TILERT_FFN_TOKEN_MSD_SPLIT_GEOMETRY_H
#define TILERT_FFN_TOKEN_MSD_SPLIT_GEOMETRY_H

#include <cstdint>

namespace ffn_gu_static_ready {

constexpr int64_t kSplitHidden = 6144;
constexpr int64_t kSplitTopK = 8;
constexpr int64_t kSplitMaxRows = 32;
constexpr int64_t kSplitMaxRoutes = kSplitMaxRows * kSplitTopK;
constexpr int64_t kSplitTile = 2048;
constexpr int64_t kSplitTiles = kSplitHidden / kSplitTile;
constexpr int64_t kSplitMetadataBytes = 2 * kSplitMaxRoutes * sizeof(int32_t);
constexpr int64_t kSplitPackedBytes = kSplitTile / 2;

#if defined(__CCE__)
#define TOKEN_SPLIT_DEVICE __host__ __aicore__
#else
#define TOKEN_SPLIT_DEVICE
#endif

TOKEN_SPLIT_DEVICE constexpr int64_t SplitToken(int64_t task)
{
    return task / kSplitTiles;
}

TOKEN_SPLIT_DEVICE constexpr int64_t SplitColumn(int64_t task)
{
    return (task % kSplitTiles) * kSplitTile;
}

TOKEN_SPLIT_DEVICE constexpr int64_t SplitOwner(int64_t task, int64_t blocks)
{
    return task % (2 * blocks);
}

TOKEN_SPLIT_DEVICE constexpr int64_t StackedRow(
    int64_t begin, int64_t count, int64_t route, int64_t half)
{
    return 2 * begin + (route - begin) + half * count;
}

static_assert(kSplitHidden % kSplitTile == 0);
static_assert(kSplitTile % 256 == 0);
static_assert(kSplitMetadataBytes == 2048);
static_assert(kSplitPackedBytes == 1024);

#undef TOKEN_SPLIT_DEVICE

}  // namespace ffn_gu_static_ready

#endif
