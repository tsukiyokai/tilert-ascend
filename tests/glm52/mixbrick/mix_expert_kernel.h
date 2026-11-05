#ifndef TESTS_GLM52_MIXBRICK_MIX_EXPERT_KERNEL_H
#define TESTS_GLM52_MIXBRICK_MIX_EXPERT_KERNEL_H

#include <cstdint>

void mix_expert_fused_launch(uint32_t blocks, void* stream, uint64_t sync_base,
                             void* x, void* gate_up, void* partial,
                             float* gu_scale, float* token_scale, void* out,
                             void* tiling, int64_t rows, int64_t stages,
                             void* middle, void* down, float* down_scale,
                             void* wide, void* narrow_out, void* down_tiling,
                             float* middle_scale, float* hidden,
                             float* maxima);

void mix_expert_staged_launch(uint32_t blocks, void* stream,
                              uint64_t sync_base, void* x, void* gate_up,
                              void* partial, float* gu_scale,
                              float* token_scale, void* out, void* tiling,
                              int64_t rows, float* hidden, float* maxima);

/** Fill a 200-byte buffer with the cube tiling for one projection shape. */
extern "C" int mix_build_tiling(int rows, int columns, int inner,
                                void* buffer);

#endif
