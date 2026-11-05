#ifndef TRT_RSAG_CANDIDATE_TREE_PLAN_H
#define TRT_RSAG_CANDIDATE_TREE_PLAN_H

#include <cstdint>

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define COMMUNICATION_TREE_DEVICE __aicore__
#else
#define COMMUNICATION_TREE_DEVICE
#endif

namespace trt_rsag_candidate {

struct TreePlan {
    uint64_t low;
    uint64_t high;
};

struct TreeStep {
    uint32_t rank;
    uint32_t reductions;
};

// Each byte encodes one leaf followed by zero or more stack additions:
// low four bits select its rank; high four bits count adjacent subtree sums.
// The host admits only 2..16 ranks. The default is never a valid plan.
COMMUNICATION_TREE_DEVICE inline constexpr TreePlan GetTreePlan(int32_t ranks)
{
    switch (ranks) {
        case 2: return {0x0000000000001100ULL, 0x0000000000000000ULL};
        case 3: return {0x0000000000111200ULL, 0x0000000000000000ULL};
        case 4: return {0x0000000023011200ULL, 0x0000000000000000ULL};
        case 5: return {0x0000002301121400ULL, 0x0000000000000000ULL};
        case 6: return {0x0000231501121400ULL, 0x0000000000000000ULL};
        case 7: return {0x0023150126021400ULL, 0x0000000000000000ULL};
        case 8: return {0x3703150126021400ULL, 0x0000000000000000ULL};
        case 9: return {0x0315012602141800ULL, 0x0000000000000037ULL};
        case 10: return {0x1519012602141800ULL, 0x0000000000003703ULL};
        case 11: return {0x1901261a02141800ULL, 0x0000000000370315ULL};
        case 12: return {0x1901261a02141800ULL, 0x00000000371b0315ULL};
        case 13: return {0x01261a022c041800ULL, 0x000000371b031519ULL};
        case 14: return {0x01261a022c041800ULL, 0x0000371b032d0519ULL};
        case 15: return {0x3e061a022c041800ULL, 0x00371b032d051901ULL};
        case 16: return {0x3e061a022c041800ULL, 0x4f071b032d051901ULL};
        default: return {0, 0};
    }
}

COMMUNICATION_TREE_DEVICE inline constexpr TreeStep GetTreeStep(
    TreePlan plan, int32_t leaf)
{
    const uint64_t word = leaf < 8 ? plan.low : plan.high;
    const uint32_t byte = (word >> ((leaf & 7) * 8)) & 255U;
    return {byte & 15U, byte >> 4};
}

}  // namespace trt_rsag_candidate

#undef COMMUNICATION_TREE_DEVICE
#endif
