#ifndef TESTS_GLM52_HANDOFF_GATHER_KERNEL_H
#define TESTS_GLM52_HANDOFF_GATHER_KERNEL_H

#include <cstdint>

void gather_demo(uint32_t block_dim, void* stream, float* cache, float* out,
                 int mode, int iterations);

#endif
