#ifndef TESTS_GLM52_MLA_TAIL_PROFILE_H
#define TESTS_GLM52_MLA_TAIL_PROFILE_H

#include <cstdint>

namespace mla_tail::profile {

constexpr int64_t kRecordWords = 64;
constexpr int64_t kStageCount = 5;
constexpr int64_t kMarksPerStage = 3;
constexpr int64_t kTimestampOffset = 8;
constexpr int64_t kVersion = 1;

}  // namespace mla_tail::profile

extern "C" void mla_tail_profile_launch(
    uint32_t blocks, void* stream, uint64_t sync_base,
    void* latent, void* value_weight, void* output_weight,
    void* output_scale, void* bias, int bias_kind, void* scratch,
    void* tiling, void* output, void* trace, int64_t rows, int64_t capacity);

#endif
