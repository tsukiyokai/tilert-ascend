#ifndef TESTS_GLM52_ROUTED_INT4_GU_SPLITK_H
#define TESTS_GLM52_ROUTED_INT4_GU_SPLITK_H

#include "routed_int4_contract.h"

#if defined(ROUTED_GU_SPLIT_K) && (defined(ROUTED_FIXPIPE_HALF) \
    || defined(ROUTED_GU_MDL) || defined(ROUTED_GU_PAD16) \
    || defined(ROUTED_DOWN_STRIPED))
#error "GU split-K requires the independent INT32 Norm path"
#endif
#if defined(ROUTED_GU_SPLIT_K) && !defined(ROUTED_ROW_BALANCED)
#error "GU split-K requires the row-balanced activation owner"
#endif

namespace routed_int4 {

constexpr int64_t kGateParts = 3;
constexpr int64_t kGatePartInner = kHidden / kGateParts;
constexpr int64_t kGatePartBlocks = 24;
constexpr int64_t kGatePartExperts = 8;
static_assert(kGatePartInner == 2048);
static_assert(kGatePartInner % 256 == 0);
static_assert(kHidden * 64 < (1LL << 31));
static_assert(kGateParts * kGateUp <= kVectorTile);

struct GatePart {
    int64_t active_expert;
    int64_t part;
};

#if defined(__CCE__)
#define ROUTED_SPLITK_HOST_DEVICE __host__ __aicore__
#else
#define ROUTED_SPLITK_HOST_DEVICE
#endif

ROUTED_SPLITK_HOST_DEVICE constexpr int64_t GateProductRows(int64_t routes)
{
    return kGateParts * GateStorageRow(routes);
}

ROUTED_SPLITK_HOST_DEVICE constexpr int64_t GatePartStride(
    int64_t tokens, int64_t blocks, int64_t active_experts)
{
    return blocks == kGatePartBlocks && active_experts > 0
        && active_experts <= kGatePartExperts
        ? GateStorageRow(tokens * kTopK) * kGateUp : 0;
}

ROUTED_SPLITK_HOST_DEVICE constexpr GatePart GatePartFor(
    int64_t core, int64_t active_experts)
{
    return core < kGateParts * active_experts
        ? GatePart{core / kGateParts, core % kGateParts}
        : GatePart{-1, 0};
}

// INT4 element offsets. The NZ N64-block stride remains full K6144.
ROUTED_SPLITK_HOST_DEVICE constexpr int64_t GatePartWeightOffset(
    int64_t part, bool nz)
{
    return part * kGatePartInner * (nz ? 64 : kGateUp);
}

#undef ROUTED_SPLITK_HOST_DEVICE

}  // namespace routed_int4

#endif
