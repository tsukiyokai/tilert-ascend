#ifndef TESTS_GLM52_HANDSHAKE_HANDSHAKE_KERNEL_H
#define TESTS_GLM52_HANDSHAKE_HANDSHAKE_KERNEL_H

#include <cstdint>

void handshake_launch(void* stream, uint64_t sync_base, void* progress,
                      int64_t reverse_mode, int64_t pattern, int64_t rounds);

#endif
