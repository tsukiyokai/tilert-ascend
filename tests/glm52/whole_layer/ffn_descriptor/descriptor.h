#ifndef TILERT_FFN_DESCRIPTOR_ARGUMENTS_H
#define TILERT_FFN_DESCRIPTOR_ARGUMENTS_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "contract.h"

namespace ffn_descriptor {

struct BodyArguments {
    uint64_t weights[kRoutedWeights];
    uint64_t scratch[kBodyBuffers];
    routed_int4::DispatchInput dispatch;
    routed_int4::pipeline::Input shared;
    int64_t rows;
    int64_t blocks;
};

static_assert(kRoutedWeights == 6 && kBodyBuffers == 20);
static_assert(sizeof(routed_int4::DispatchInput) == 56);
static_assert(sizeof(routed_int4::pipeline::Input) == 112);
static_assert(std::is_standard_layout_v<BodyArguments>);
static_assert(std::is_trivially_copyable_v<BodyArguments>);
static_assert(sizeof(BodyArguments) == 392);
static_assert(offsetof(BodyArguments, dispatch) == 208);
static_assert(offsetof(BodyArguments, shared) == 264);
static_assert(offsetof(BodyArguments, rows) == 376);

// Only addresses are captured here. The bridge produces their contents later.
inline BodyArguments MakeBodyArguments(void** weights, void** bridge,
    void** scratch, routed_int4::pipeline::Input shared, int64_t rows,
    int64_t blocks)
{
    BodyArguments result{};
    for (int64_t i = 0; i < kRoutedWeights; ++i) {
        result.weights[i] = reinterpret_cast<uint64_t>(weights[i]);
    }
    for (int64_t i = 0; i < kBodyBuffers; ++i) {
        result.scratch[i] = reinterpret_cast<uint64_t>(scratch[i]);
    }
    result.dispatch.bf16_rows =
        reinterpret_cast<uint64_t>(bridge[ffn_bridge::kNormalized]);
    result.dispatch.expert_ids =
        reinterpret_cast<uint64_t>(bridge[ffn_bridge::kIds]);
    result.dispatch.probabilities =
        reinterpret_cast<uint64_t>(bridge[ffn_bridge::kProbabilities]);
    result.shared = shared;
    result.rows = rows;
    result.blocks = blocks;
    return result;
}

}  // namespace ffn_descriptor

#endif
