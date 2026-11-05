#include "native.h"

namespace ffn_bridge {
struct Addresses { GM_ADDR values[kBuffers]; };
__aicore__ inline void Forward(AscendC::TPipe& pipe, GM_ADDR input,
    GM_ADDR residual, GM_ADDR gamma, GM_ADDR gate, GM_ADDR bias,
    Addresses a, int64_t rows, int64_t blocks, float epsilon, ffn_trace::Clock& clock)
{
    using namespace AscendC;
    whole_layer_norm::device::Forward(pipe,
        (__gm__ bfloat16_t*)input, (__gm__ bfloat16_t*)residual,
        (__gm__ bfloat16_t*)gamma, nullptr,
        (__gm__ bfloat16_t*)a.values[kNormalized],
        (__gm__ bfloat16_t*)a.values[kRetained],
        (__gm__ float*)a.values[kRstd], nullptr, nullptr, rows, blocks, epsilon);
    device::EndPhase(pipe);
    clock.Mark(1);
    device::Gate(pipe, (__gm__ bfloat16_t*)a.values[kNormalized],
        (__gm__ bfloat16_t*)gate, (__gm__ float*)a.values[kLogits],
        a.values[kTiling], rows, blocks);
    device::Quantize(pipe, (__gm__ bfloat16_t*)a.values[kNormalized],
        (__gm__ int8_t*)a.values[kSharedCodes],
        (__gm__ float*)a.values[kSharedScales], rows);
    device::EndPhase(pipe);
    clock.Mark(2);
    device::TopK(pipe, a.values[kLogits], bias, a.values[kProbabilities],
                 a.values[kIds], a.values[kSystem], rows);
    device::EndPhase(pipe);
    clock.Mark(3);
}
}
