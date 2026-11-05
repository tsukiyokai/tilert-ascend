/**
 * Does the INT8-to-two-INT4 split compute what it should?
 *
 * The halves leave as INT32, so the packing of nibbles cannot be blamed for
 * an answer that is already wrong in arithmetic. `low` should be
 * `((x + 8) mod 16) - 8` and `high` the exact remainder `(x - low) / 16`.
 */
#if CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif

#include "split_probe_kernel.h"
#include "kernel_operator.h"
#if CUBE_ONLY
#include "lib/matmul_intf.h"
#endif

using namespace AscendC;

namespace {

#ifndef SPAN_K
#define SPAN_K 256
#endif
constexpr int64_t SPAN = SPAN_K;
constexpr int64_t SLACK = 64;
#ifndef LOW_AT
#define LOW_AT 256
#endif

}  // namespace

#if AS_MIX
extern "C" __global__ __mix__(1, 2) __aicore__ void split_probe(
    __gm__ int8_t* source,
#else
extern "C" __global__ __aicore__ void split_probe(__gm__ int8_t* source,
#endif
                                                  __gm__ int32_t* high_out,
                                                  __gm__ int32_t* low_out,
                                                  __gm__ float* trace,
                                                  __gm__ int4b_t* packed)
{
#if AS_MIX
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    if ASCEND_IS_AIC {
        return;
    }
#if GATED
    // The routed kernel runs its split on one subblock only. If a gated
    // vector op inherits the wrong mask or repeat state, both of that
    // kernel's defects follow.
    if (GetSubBlockIdx() != 0) {
        return;
    }
#endif
#endif
    TPipe pipe;
    TBuf<TPosition::VECCALC> narrow_buffer;
    TBuf<TPosition::VECCALC> value_buffer;
    TBuf<TPosition::VECCALC> other_buffer;
    TBuf<TPosition::VECCALC> work_buffer;
    TBuf<TPosition::VECCALC> half_buffer;
    TBuf<TPosition::VECCALC> wide_buffer;
    TBuf<TPosition::VECCALC> code_buffer;
    pipe.InitBuffer(narrow_buffer, (SPAN + SLACK) * sizeof(int8_t));
    pipe.InitBuffer(value_buffer, (SPAN + SLACK) * sizeof(float));
    pipe.InitBuffer(other_buffer, (SPAN + SLACK) * sizeof(float));
    pipe.InitBuffer(work_buffer, (SPAN + SLACK) * sizeof(float));
    pipe.InitBuffer(half_buffer, (SPAN + SLACK) * sizeof(half));
    pipe.InitBuffer(wide_buffer, (SPAN + SLACK) * sizeof(int32_t));
    pipe.InitBuffer(code_buffer, (SPAN + SLACK) / 2 + SLACK);
    LocalTensor<int8_t> narrow = narrow_buffer.Get<int8_t>();
    LocalTensor<float> value = value_buffer.Get<float>();
    LocalTensor<float> other = other_buffer.Get<float>();
    LocalTensor<float> work = work_buffer.Get<float>();
    LocalTensor<half> half_value = half_buffer.Get<half>();
    LocalTensor<int32_t> wide = wide_buffer.Get<int32_t>();
    LocalTensor<int4b_t> code = code_buffer.Get<int4b_t>();

    GlobalTensor<int8_t> source_global;
    GlobalTensor<int32_t> high_global;
    GlobalTensor<int32_t> low_global;
    GlobalTensor<float> trace_global;
    GlobalTensor<int4b_t> packed_global;
    source_global.SetGlobalBuffer(source, SPAN);
    high_global.SetGlobalBuffer(high_out, SPAN);
    low_global.SetGlobalBuffer(low_out, SPAN);
    trace_global.SetGlobalBuffer(trace, SPAN);
    // Where the second half goes. The routed kernel's own writes land at
    // offset zero and vanish at 12288; this is the remaining variable.
    packed_global.SetGlobalBuffer(packed, LOW_AT + SPAN);

    DataCopy(narrow, source_global, SPAN);
    PipeBarrier<PIPE_ALL>();
    Cast(half_value, narrow, RoundMode::CAST_NONE, SPAN);
    PipeBarrier<PIPE_V>();
    Cast(value, half_value, RoundMode::CAST_NONE, SPAN);
    PipeBarrier<PIPE_V>();

    Adds(work, value, 8.0F, SPAN);
    PipeBarrier<PIPE_V>();
    Muls(other, work, 1.0F / 16.0F, SPAN);
    PipeBarrier<PIPE_V>();
    Floor(other, other, SPAN);
    PipeBarrier<PIPE_V>();
    // What the floor produced, so a floor that is not one shows up here.
    DataCopy(trace_global, other, SPAN);
    Muls(other, other, -16.0F, SPAN);
    PipeBarrier<PIPE_V>();
    Add(work, work, other, SPAN);
    PipeBarrier<PIPE_V>();
    Adds(work, work, -8.0F, SPAN);
    PipeBarrier<PIPE_V>();
    Sub(value, value, work, SPAN);
    PipeBarrier<PIPE_V>();
    Muls(value, value, 1.0F / 16.0F, SPAN);
    PipeBarrier<PIPE_V>();

    Cast(wide, value, RoundMode::CAST_RINT, SPAN);
    PipeBarrier<PIPE_ALL>();
    DataCopy(high_global, wide, SPAN);
    PipeBarrier<PIPE_ALL>();
    Cast(wide, work, RoundMode::CAST_RINT, SPAN);
    PipeBarrier<PIPE_ALL>();
    DataCopy(low_global, wide, SPAN);
    PipeBarrier<PIPE_ALL>();

    // The same two halves again, this time packed as INT4, so the nibble
    // convention can be read off rather than assumed.
    Cast(half_value, value, RoundMode::CAST_RINT, SPAN);
    PipeBarrier<PIPE_V>();
    Cast(code, half_value, RoundMode::CAST_RINT, SPAN);
    PipeBarrier<PIPE_ALL>();
    DataCopy(packed_global, code, SPAN);
    PipeBarrier<PIPE_ALL>();
    Cast(half_value, work, RoundMode::CAST_RINT, SPAN);
    PipeBarrier<PIPE_V>();
    Cast(code, half_value, RoundMode::CAST_RINT, SPAN);
    PipeBarrier<PIPE_ALL>();
    DataCopy(packed_global[LOW_AT], code, SPAN);
}

void split_probe_launch(void* stream, void* source, void* high_out,
                        void* low_out, void* trace, void* packed)
{
    split_probe<<<1, nullptr, stream>>>((__gm__ int8_t*)source,
                                        (__gm__ int32_t*)high_out,
                                        (__gm__ int32_t*)low_out,
                                        (__gm__ float*)trace,
                                        (__gm__ int4b_t*)packed);
}
