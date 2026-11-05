#ifndef FUSED_PROJ_REDUCE_KERNEL_H
#define FUSED_PROJ_REDUCE_KERNEL_H

#include <cstdint>

void fused_proj_reduce_demo(uint32_t block_dim, void* stream, uint64_t fftsAddr, int8_t* activation, int8_t* weight_nz,
                            float* partial, float* out, uint8_t* window, uint8_t* tiling, int rows, int cols,
                            int rounds, int round_base);

#endif  // FUSED_PROJ_REDUCE_KERNEL_H
