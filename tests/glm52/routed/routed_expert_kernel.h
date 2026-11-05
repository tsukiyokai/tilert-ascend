#ifndef TESTS_GLM52_ROUTED_ROUTED_EXPERT_KERNEL_H
#define TESTS_GLM52_ROUTED_ROUTED_EXPERT_KERNEL_H

#include <cstdint>

void routed_expert_launch(uint32_t blocks, void* stream, uint64_t sync_base,
                          void* rows, void* gate_up, float* gu_scale,
                          void* down, float* down_scale, float* token_scale,
                          void* partial, void* middle, float* middle_scale,
                          void* wide, void* out, void* gu_tiling,
                          void* down_tiling, int64_t experts);

/** Fill a 256-byte buffer with the cube tiling for one projection shape. */
extern "C" int routed_build_tiling(int rows, int columns, int inner,
                                   int width, void* buffer);

#endif
