#ifndef TESTS_GLM52_ROUTED_INT4_DOWN_STRIPES_H
#define TESTS_GLM52_ROUTED_INT4_DOWN_STRIPES_H

#include "routed_int4_contract.h"

#if defined(ROUTED_DOWN_STRIPED) && defined(ROUTED_FIXPIPE_HALF)
#error "ROUTED_DOWN_STRIPED requires INT32 products"
#endif

namespace routed_int4 {

constexpr int64_t kDownStripes = 3;
constexpr int64_t kDownStripeColumns = kHidden / kDownStripes;
constexpr int64_t kDownStripeBlocks = 24;
constexpr int64_t kDownStripeExperts = 8;
static_assert(kDownStripeColumns == 2048);
static_assert(kDownStripeColumns % 256 == 0);
static_assert(kDownStripeColumns % 64 == 0);

struct DownStripe {
    int64_t active_expert;
    int64_t column;
};

#if defined(__CCE__)
#define ROUTED_STRIPE_HOST_DEVICE __host__ __aicore__
#else
#define ROUTED_STRIPE_HOST_DEVICE
#endif

ROUTED_STRIPE_HOST_DEVICE constexpr bool DownStripesEnabled(
    int64_t blocks, int64_t active_experts)
{
    return blocks == kDownStripeBlocks && active_experts > 0
        && active_experts <= kDownStripeExperts;
}

ROUTED_STRIPE_HOST_DEVICE constexpr DownStripe DownStripeFor(
    int64_t core, int64_t active_experts)
{
    return core < kDownStripes * active_experts
        ? DownStripe{core / kDownStripes,
                     (core % kDownStripes) * kDownStripeColumns}
        : DownStripe{-1, 0};
}

// Offsets count INT4 elements. NZ stores complete K rows within each
// N64 block; ND preserves the full N6144 stride between K rows.
ROUTED_STRIPE_HOST_DEVICE constexpr int64_t DownWeightColumnOffset(
    int64_t column, bool nz)
{
    return nz ? column * kIntermediate : column;
}

#undef ROUTED_STRIPE_HOST_DEVICE

}  // namespace routed_int4

#endif
