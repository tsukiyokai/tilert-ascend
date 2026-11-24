#ifndef TESTS_GLM52_ROUTED_INT4_PIPELINE_H
#define TESTS_GLM52_ROUTED_INT4_PIPELINE_H

#include "routed_int4_contract.h"
#ifdef ROUTED_ACTIVE_LIST
#include "routed_int4_active_list.h"
#endif
#include "../shared_w8/shared_w8.h"

namespace routed_int4::pipeline {

enum ScratchIndex : size_t {
    kSharedPayload = kDispatchScratchCount,
    kSharedTiling,
    kSharedOutput,
#ifdef ROUTED_ACTIVE_LIST
    kActiveList,
#endif
    kScratchCount,
};

enum InputIndex : size_t {
    kInput,
    kInputScale,
    kGateWeight,
    kGateScale,
    kDownWeight,
    kDownScale,
    kInputCount,
};

constexpr int64_t Capacity(int64_t tokens)
{
    return tokens <= 16 ? 16 : 32;
}

constexpr size_t ExtraScratchBytes(int64_t tokens)
{
    return shared_w8::MakeLayout(Capacity(tokens)).system
        + 2 * shared_w8::kTilingStride + tokens * kHidden * sizeof(uint16_t)
#ifdef ROUTED_ACTIVE_LIST
        + kActiveListBytes
#endif
        ;
}

// The shared branch owns only payload. Every CUBE phase borrows the routed
// system workspace; CUBE projections never overlap each other.
struct Input {
    uint64_t input = 0;
    uint64_t input_scale = 0;
    uint64_t gate_weight = 0;
    uint64_t gate_scale = 0;
    uint64_t down_weight = 0;
    uint64_t down_scale = 0;
    uint64_t payload = 0;
    uint64_t tiling = 0;
    uint64_t output = 0;
    int64_t capacity = 0;
    uint64_t partial_offset = 0;
    uint64_t middle_offset = 0;
    uint64_t scale_offset = 0;
    uint64_t down_offset = 0;
};

constexpr size_t kExtraActiveUB =
#ifdef ROUTED_ACTIVE_LIST
    kActiveListBytes;
#else
    0;
#endif
constexpr size_t kRoutedResidentUB = 81952 + kExtraActiveUB;
constexpr size_t kSharedResidentUB = 6592;
constexpr size_t kSharedEpilogueUB = 23808;
#ifdef ROUTED_INPUT_FRONT
constexpr size_t kFrontPhaseUB = kSharedResidentUB + 76320 + kExtraActiveUB;
static_assert(kFrontPhaseUB == 82912 + kExtraActiveUB);
static_assert(kFrontPhaseUB <= 188 * 1024);
#endif
static_assert(kRoutedResidentUB + kSharedResidentUB + kSharedEpilogueUB
              <= 188 * 1024);
#ifdef ROUTED_ACTIVE_LIST
static_assert(kActiveList == 20 && kScratchCount == 21);
#else
static_assert(kScratchCount == 20);
#endif
static_assert(shared_w8::MakeLayout(16).system == 592384);
static_assert(shared_w8::MakeLayout(32).system == 1184768);

}  // namespace routed_int4::pipeline

#ifdef ROUTED_INPUT_FRONT
#define routed_shared_build_tiling ffn_dual_quant_body_build_shared_tiling
#endif
extern "C" int routed_shared_build_tiling(int capacity, void* destination);

#endif
