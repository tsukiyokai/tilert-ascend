#ifndef TESTS_GLM52_HANDOFF_RENDEZVOUS_KERNEL_H
#define TESTS_GLM52_HANDOFF_RENDEZVOUS_KERNEL_H

#include <cstdint>

void rendezvous_demo(uint32_t block_dim, void* stream, float* sink, int mode,
                     int iterations);

#endif
