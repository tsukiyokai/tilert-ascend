#ifndef TESTS_GLM52_WHOLE_LAYER_ATTENTION_NORM_NATIVE_H
#define TESTS_GLM52_WHOLE_LAYER_ATTENTION_NORM_NATIVE_H

#include "attention_full_native.h"
#include "attention_norm_contract.h"
#include "norm_native.h"

namespace whole_layer_attention {

__aicore__ inline void NormForward(
    AscendC::TPipe& pipe, const NormConfig& config, DeviceArgs attention,
    GM_ADDR input, GM_ADDR residual, GM_ADDR norm_gamma, GM_ADDR norm_beta,
    GM_ADDR retained, GM_ADDR value_weight, GM_ADDR output_weight,
    GM_ADDR output_scale, GM_ADDR bias, int bias_kind, GM_ADDR scratch,
    GM_ADDR tail_tiling, GM_ADDR output)
{
    GM_ADDR full_arena = scratch + config.full_offset;
    attention.token_x = scratch;
    attention.token_scale = scratch + config.scale_offset;
    whole_layer_norm::device::Forward(pipe,
        (__gm__ bfloat16_t*)input, (__gm__ bfloat16_t*)residual,
        (__gm__ bfloat16_t*)norm_gamma, (__gm__ bfloat16_t*)norm_beta,
        (__gm__ bfloat16_t*)full_arena, (__gm__ bfloat16_t*)retained,
        (__gm__ float*)(full_arena + config.rstd_offset),
        (__gm__ int8_t*)attention.token_x, (__gm__ float*)attention.token_scale,
        config.full.attention.rows, config.full.attention.blocks, config.epsilon);
    // Publish the norm outputs before Prolog reads their quantized form.
    // normalized/rstd are dead after this barrier; retained remains external.
    EndPhase(pipe);
    FullForward(pipe, config.full, attention, value_weight, output_weight,
        output_scale, bias, bias_kind, full_arena, tail_tiling, output);
}

}  // namespace whole_layer_attention

#endif
