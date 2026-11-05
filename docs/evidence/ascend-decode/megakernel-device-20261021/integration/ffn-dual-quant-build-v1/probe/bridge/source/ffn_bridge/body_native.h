#include "native.h"
#include "../../../source/dual_quant_native.h"

namespace ffn_bridge {
struct Addresses { GM_ADDR values[kBuffers]; };
__aicore__ inline void Forward(AscendC::TPipe& pipe, GM_ADDR input,
    GM_ADDR residual, GM_ADDR gamma, GM_ADDR gate, GM_ADDR bias,
    Addresses a, GM_ADDR routed_codes, GM_ADDR routed_scales,
    int64_t rows, int64_t blocks, float epsilon)
{
    using namespace AscendC;
    whole_layer_norm::device::Forward(pipe,
        (__gm__ bfloat16_t*)input, (__gm__ bfloat16_t*)residual,
        (__gm__ bfloat16_t*)gamma, nullptr,
        (__gm__ bfloat16_t*)a.values[kNormalized],
        (__gm__ bfloat16_t*)a.values[kRetained],
        (__gm__ float*)a.values[kRstd], nullptr, nullptr, rows, blocks, epsilon);
    device::EndPhase(pipe);
    device::Gate(pipe, (__gm__ bfloat16_t*)a.values[kNormalized],
        (__gm__ bfloat16_t*)gate, (__gm__ float*)a.values[kLogits],
        a.values[kTiling], rows, blocks);
    ffn_dual_quant::device::Quantize(pipe, (__gm__ bfloat16_t*)a.values[kNormalized],
        (__gm__ int8_t*)a.values[kSharedCodes],
        (__gm__ float*)a.values[kSharedScales],
        (__gm__ int8_t*)routed_codes,
        (__gm__ float*)routed_scales, rows);
    device::EndPhase(pipe);
    device::TopK(pipe, a.values[kLogits], bias, a.values[kProbabilities],
                 a.values[kIds], a.values[kSystem], rows);
    device::EndPhase(pipe);
}
}
