#ifndef TESTS_GLM52_ROUTED_INT4_PROFILE_H
#define TESTS_GLM52_ROUTED_INT4_PROFILE_H

#include <cstdint>

#include "routed_int4_contract.h"

namespace routed_int4::profile {

constexpr int64_t kVersion = 1;
constexpr int64_t kStages = 7;
constexpr int64_t kMarksPerStage = 3;
constexpr int64_t kRecordWords = 64;
constexpr int64_t kTimestampOffset = 8;
constexpr int64_t kScratchOffset = 11;
constexpr int64_t kTraceAddress = kScratchOffset + kDispatchScratchCount;
constexpr int64_t kAddressCount = kTraceAddress + 1;

// Host address array: six expert tensors (gate/up, correction, scale, down,
// correction, scale), five vendor dispatch outputs (rows, scales, inverse,
// counts, probabilities), all dispatch scratch tensors, and the trace.

}  // namespace routed_int4::profile

extern "C" void routed_int4_profile_launch(
    uint32_t blocks, void* stream, uint64_t sync_base,
    const uint64_t* addresses, int64_t tokens, int64_t weight_layout);

#endif
