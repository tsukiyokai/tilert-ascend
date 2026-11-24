#ifndef TESTS_GLM52_EPILOGUE_MOE_EPILOGUE_KERNEL_H
#define TESTS_GLM52_EPILOGUE_MOE_EPILOGUE_KERNEL_H

#include <cstdint>

void moe_epilogue_fused_launch(uint32_t blocks, void* stream, void* routed,
                               void* shared, float* out, float routed_scale,
                               int64_t elements,
                               int iterations);

void moe_epilogue_stage_launch(uint32_t blocks, void* stream, void* routed,
                               void* shared, float* out, float routed_scale,
                               int stage, int64_t elements,
                               int iterations);

#endif
