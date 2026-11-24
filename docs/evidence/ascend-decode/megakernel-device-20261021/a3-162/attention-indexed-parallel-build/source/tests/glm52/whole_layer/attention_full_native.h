#ifndef TESTS_GLM52_WHOLE_LAYER_ATTENTION_FULL_NATIVE_H
#define TESTS_GLM52_WHOLE_LAYER_ATTENTION_FULL_NATIVE_H

// Select the same Matmul backend as the independently checked projection tail
// before either vendor include tree can include lib/matmul_intf.h.
#ifndef ASCENDC_CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif
#include "attention_native.h"
#include "attention_full_contract.h"
#include "../mla_tail/mla_tail_native.h"

namespace whole_layer_attention {

__aicore__ inline void FullForward(
    AscendC::TPipe& pipe, const FullConfig& config, DeviceArgs attention,
    GM_ADDR value_weight, GM_ADDR output_weight, GM_ADDR output_scale,
    GM_ADDR bias, int bias_kind, GM_ADDR scratch, GM_ADDR tail_tiling,
    GM_ADDR output)
{
    GM_ADDR arena = scratch + config.arena_offset;
    attention.scratch = arena;
    attention.output = scratch;
    Forward(pipe, config.attention, attention);
    // Forward has retired all SFA reads/writes and reset both AIC and AIV
    // resources. Its query/rope/work arena can now hold the entire tail.
    mla_tail::device::Forward(pipe, (__gm__ bfloat16_t*)scratch,
        (__gm__ bfloat16_t*)value_weight, (__gm__ int8_t*)output_weight,
        (__gm__ float*)output_scale, bias, bias_kind, arena, tail_tiling,
        (__gm__ bfloat16_t*)output, config.attention.rows, config.capacity,
        config.attention.blocks, config.tail);
    EndPhase(pipe);
}

}  // namespace whole_layer_attention

#endif
