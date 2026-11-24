#ifndef TILERT_FFN_TOKEN_TERMINAL_ARGUMENTS_H
#define TILERT_FFN_TOKEN_TERMINAL_ARGUMENTS_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "contract.h"

namespace ffn_token_terminal {

struct BodyArguments {
    uint64_t weights[kRoutedWeights];
    uint64_t scratch[kBodyBuffers];
    routed_int4::DispatchInput dispatch;
    routed_int4::pipeline::Input shared;
    int64_t rows;
    int64_t blocks;
};

static_assert(kRoutedWeights == 6 && kBodyBuffers == 21);
static_assert(sizeof(routed_int4::DispatchInput) == 64);
static_assert(sizeof(routed_int4::pipeline::Input) == 112);
static_assert(std::is_standard_layout_v<BodyArguments>);
static_assert(std::is_trivially_copyable_v<BodyArguments>);
static_assert(sizeof(BodyArguments) == 408);
static_assert(offsetof(BodyArguments, dispatch) == 216);
static_assert(offsetof(BodyArguments, shared) == 280);
static_assert(offsetof(BodyArguments, rows) == 392);

// The bridge and dispatch populate these addresses during device execution.
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
    result.dispatch.active_list =
        reinterpret_cast<uint64_t>(scratch[routed_int4::pipeline::kActiveList]);
    result.shared = shared;
    result.rows = rows;
    result.blocks = blocks;
    return result;
}

}  // namespace ffn_token_terminal

#endif
