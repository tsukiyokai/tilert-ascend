#ifndef TESTS_GLM52_HANDOFF_FLOOR_KERNEL_H
#define TESTS_GLM52_HANDOFF_FLOOR_KERNEL_H

#include <cstdint>

void floor_demo(uint32_t block_dim, void* stream, float* source, float* sink,
                int mode, int iterations);

#endif
