#ifndef TP_PUSH_KERNEL_H
#define TP_PUSH_KERNEL_H

#include <cstdint>

void tp_push_demo(uint32_t block_dim, void* stream, uint64_t fftsAddr, float* partial, float* out, uint8_t* window,
                  int32_t* errors, int rows, int hidden, int rounds, int round_base, int stage,
                  const int32_t* active_ptr);
void tp_push_demo(uint32_t block_dim, void* stream, uint64_t fftsAddr, float* partial, float* out, uint8_t* window,
                  int32_t* errors, int rows, int hidden, int rounds, int round_base, int stage);
void tp_push_rmsnorm_quant(uint32_t block_dim, void* stream, uint64_t fftsAddr, float* partial, float* out,
                           uint8_t* window, int32_t* routed, int numel, int scale_bits, int round_base, int stage,
                           const int32_t* active_ptr, const void* residual_in, void* res_out, const void* gamma, const void* beta,
                           void* norm_out, void* quant_out, void* scale_out, float eps);

#endif  // TP_PUSH_KERNEL_H
