#ifndef TESTS_GLM52_ROUTED_INT4_CONTRACT_H
#define TESTS_GLM52_ROUTED_INT4_CONTRACT_H

#include <cstddef>
#include <cstdint>

namespace routed_int4 {

constexpr int64_t kHidden = 6144;
constexpr int64_t kIntermediate = 128;
constexpr int64_t kGateUp = 2 * kIntermediate;
constexpr int64_t kMaxExperts = 256;
constexpr int64_t kMaxTokens = 32;
constexpr int64_t kTopK = 8;
constexpr int64_t kMaxRoutes = kMaxTokens * kTopK;
constexpr int64_t kMaxRowsPerExpert = kMaxTokens;
constexpr int64_t kMaxStackedRows = 2 * kMaxRowsPerExpert;
constexpr int64_t kVectorTile = 2048;
constexpr size_t kTilingBytes = 256;
constexpr size_t kMatmulWorkspaceBytes = 16u << 20;

#ifdef ROUTED_GU_PAD16
constexpr bool kPadGateRows = true;
constexpr int64_t kGateRowsPerRoute = 16;
#else
constexpr bool kPadGateRows = false;
constexpr int64_t kGateRowsPerRoute = 2;
#endif

#if defined(__CCE__)
#define ROUTED_HOST_DEVICE __host__ __aicore__
#else
#define ROUTED_HOST_DEVICE
#endif

ROUTED_HOST_DEVICE constexpr int64_t GateStorageRow(int64_t route_begin)
{
    return kGateRowsPerRoute * route_begin;
}

ROUTED_HOST_DEVICE constexpr int64_t GateMatrixRows(int64_t count)
{
    return kPadGateRows ? ((2 * count + 15) / 16) * 16 : 2 * count;
}

#undef ROUTED_HOST_DEVICE

#ifdef ROUTED_FIXPIPE_HALF
using ScaleStorage = uint64_t;
constexpr size_t kProductBytes = sizeof(uint16_t);
#else
using ScaleStorage = float;
constexpr size_t kProductBytes = sizeof(int32_t);
#endif

enum WeightLayout : int64_t {
    kWeightND = 0,
    kWeightNZ = 1,
};

constexpr bool IsWeightLayout(int64_t layout)
{
    return layout == kWeightND || layout == kWeightNZ;
}

// ND weights store B[K,N/2]. NZ weights store bytes [N/64,K/16,16,32]
// per expert, exposed as logical INT32 [K,N/8]. Both pack adjacent output
// columns low nibble first. Activations store A[M,K/2], with high MSD rows first and
// low MSD rows immediately after each group's high rows. A group with count
// c and route prefix s occupies rows [2*s, 2*(s+c)) by default. The GU pad16
// variant starts A/C at 16*s and reserves 16*c rows; only the first 2*c rows
// carry high/low values. Its MMA writes ceil(2*c/16)*16 rows. Padding A bytes
// are initialized once and may retain old values between replays; no consumer
// reads padding results. Down remains compact in every variant.
// Offsets below count physical bytes.
constexpr int64_t PackedWeightOffset(int64_t expert, int64_t inner,
                                     int64_t column, int64_t inputs,
                                     int64_t outputs)
{
    return (expert * inputs + inner) * (outputs / 2) + column / 2;
}

constexpr int64_t StackedOffset(int64_t route_begin, int64_t row,
                                int64_t inputs)
{
    return (2 * route_begin + row) * inputs / 2;
}

constexpr int64_t OriginalSlot(int64_t token, int64_t slot)
{
    return token * kTopK + slot;
}

enum ScratchIndex : size_t {
    kStacked,
    kGateProduct,
    kMiddle,
    kMiddleScale,
    kDownProduct,
    kRouteOutput,
    kGateTiling,
    kDownTiling,
    kMatmulWorkspace,
    kOutput,
    kScratchCount,
};

enum DispatchScratchIndex : size_t {
    kRestoredRows = kScratchCount,
    kRestoredScale,
    kGroupCount,
    kGroupStart,
    kSourceRow,
    kSourceSlot,
    kRowWeight,
    kDispatchScratchCount,
};

// Addresses of the unmodified vendor InitRouting outputs. inverse maps
// token * kTopK + slot to an expert-major row; counts includes empty experts.
struct DispatchInput {
    uint64_t rows = 0;
    uint64_t scales = 0;
    uint64_t inverse = 0;
    uint64_t counts = 0;
    uint64_t probabilities = 0;
};

constexpr size_t DispatchScratchBytes(int64_t tokens)
{
    return tokens * (kHidden + sizeof(float))
        + 2 * kMaxExperts * sizeof(int32_t)
        + kMaxRoutes * (2 * sizeof(int32_t) + sizeof(float));
}

constexpr size_t ScratchBytes(int64_t tokens)
{
    const int64_t routes = tokens * kTopK;
    return GateStorageRow(routes) * kHidden / 2
        + GateStorageRow(routes) * kGateUp * kProductBytes
        + 2 * routes * kIntermediate / 2
        + routes * sizeof(float)
        + 2 * routes * kHidden * kProductBytes
        + tokens * kTopK * kHidden * sizeof(uint16_t)
        + 2 * kTilingBytes + kMatmulWorkspaceBytes
        + tokens * kHidden * sizeof(uint16_t);
}

static_assert(kMaxRowsPerExpert >= kMaxTokens);
static_assert(kMaxStackedRows % 16 == 0);
static_assert(kVectorTile % 256 == 0);

}  // namespace routed_int4

#endif
