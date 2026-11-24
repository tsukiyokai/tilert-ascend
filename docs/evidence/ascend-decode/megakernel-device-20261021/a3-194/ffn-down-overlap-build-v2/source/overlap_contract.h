#ifndef TILERT_FFN_DOWN_OVERLAP_GEOMETRY_H
#define TILERT_FFN_DOWN_OVERLAP_GEOMETRY_H

#include <cstdint>

namespace ffn_down_overlap {

constexpr int64_t kDownHidden = 6144;
constexpr int64_t kDownInner = 128;
constexpr int64_t kDownMaximumRows = 64;
constexpr int64_t kDownTile = 256;
constexpr int64_t kDownHalfTile = kDownTile / 2;
constexpr int64_t kDownTilesPerExpert = kDownHidden / kDownTile;
constexpr int64_t kDownCredits = 4;
constexpr uint16_t kDownReadyFlag = 8;
constexpr uint16_t kDownFreeFlag = 9;

constexpr bool IsDownTiling(int64_t base_m, int64_t base_n, int64_t base_k,
                            int64_t single_m, int64_t single_n, int64_t single_k)
{
    return base_m == kDownMaximumRows && base_n == kDownTile
        && base_k == kDownInner && single_m == kDownMaximumRows
        && single_n == kDownHidden && single_k == kDownInner;
}

static_assert(kDownTilesPerExpert == 24);
static_assert(kDownHalfTile == 128);
static_assert(kDownCredits < 15);
static_assert(kDownReadyFlag > 7 && kDownFreeFlag < 11);
static_assert(IsDownTiling(64, 256, 128, 64, 6144, 128));
static_assert(!IsDownTiling(32, 256, 128, 64, 6144, 128));
static_assert(!IsDownTiling(64, 512, 128, 64, 6144, 128));
static_assert(!IsDownTiling(64, 256, 64, 64, 6144, 128));
static_assert(!IsDownTiling(64, 256, 128, 32, 6144, 128));

}  // namespace ffn_down_overlap

#endif
