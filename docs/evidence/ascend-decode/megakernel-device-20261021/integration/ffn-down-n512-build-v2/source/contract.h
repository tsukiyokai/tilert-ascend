#ifndef TILERT_FFN_DOWN_N512_CONTRACT_H
#define TILERT_FFN_DOWN_N512_CONTRACT_H

#include <cstdint>

#include "../bridge/source/ffn_bridge/contract.h"
#include "../body/source/routed_int4/routed_int4_pipeline.h"

namespace ffn_down_n512 {
constexpr int64_t kRoutedWeights = 6;
constexpr int64_t kSharedWeights = 4;
constexpr int64_t kBridgeBuffers = ffn_bridge::kBuffers;
constexpr int64_t kBodyBuffers = routed_int4::pipeline::kScratchCount;
static_assert(kBridgeBuffers == 12 && kBodyBuffers == 21);
static_assert(ffn_bridge::kExperts == routed_int4::kMaxExperts);
static_assert(ffn_bridge::kHidden == routed_int4::kHidden);
static_assert(ffn_bridge::kTopK == routed_int4::kTopK);
}  // namespace ffn_down_n512

extern "C" void ffn_down_n512_launch(void* stream, uint64_t sync,
    void* input, void* residual, void* gamma, void* gate, void* bias,
    void** routed_weights, void** bridge, void** body,
    routed_int4::pipeline::Input shared, int64_t rows, int64_t blocks,
    int64_t weight_layout, float epsilon);

#endif
