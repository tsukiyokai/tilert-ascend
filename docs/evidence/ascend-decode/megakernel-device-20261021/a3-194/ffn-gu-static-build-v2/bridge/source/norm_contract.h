#ifndef TILERT_WHOLE_LAYER_NORM_CONTRACT_H
#define TILERT_WHOLE_LAYER_NORM_CONTRACT_H

#include <cstdint>

namespace whole_layer_norm {
constexpr int64_t kHidden = 6144;
constexpr int64_t kMaximumRows = 32;
}

void whole_layer_norm_launch(
    uint32_t blocks, void* stream, uint64_t sync_base,
    void* input, void* residual, void* gamma, void* beta,
    void* normalized, void* retained, float* rstd,
    void* quantized, float* scales, int64_t rows, float epsilon);

#endif
