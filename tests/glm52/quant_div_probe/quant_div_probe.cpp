#include "kernel_operator.h"

constexpr uint32_t kLanes = 64;

extern "C" __global__ __aicore__ void quant_div_probe(
    GM_ADDR input, GM_ADDR float_output, GM_ADDR integer_output)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace AscendC;
    TPipe pipe;
    TBuf<TPosition::VECCALC> scratch;
    pipe.InitBuffer(scratch, 7 * kLanes * sizeof(float));
    auto storage = scratch.Get<float>();
    auto x = storage;
    auto maximum = storage[kLanes];
    auto literal = storage[2 * kLanes];
    auto factor = storage[3 * kLanes];
    auto product = storage[4 * kLanes];
    auto rounded = storage[5 * kLanes].ReinterpretCast<int32_t>();
    auto literal_rounded = storage[6 * kLanes].ReinterpretCast<int32_t>();
    GlobalTensor<float> source;
    GlobalTensor<float> floats;
    GlobalTensor<int32_t> integers;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input), 3 * kLanes);
    floats.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(float_output),
                           2 * kLanes);
    integers.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(integer_output),
                             2 * kLanes);
    DataCopy(x, source, kLanes);
    DataCopy(maximum, source[kLanes], kLanes);
    DataCopy(literal, source[2 * kLanes], kLanes);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    Duplicate(factor, 127.0f, kLanes);
    PipeBarrier<PIPE_V>();
    Div(factor, factor, maximum, kLanes);
    PipeBarrier<PIPE_V>();
    Mul(product, x, factor, kLanes);
    PipeBarrier<PIPE_V>();
    Cast(rounded, product, RoundMode::CAST_RINT, kLanes);
    Cast(literal_rounded, literal, RoundMode::CAST_RINT, kLanes);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(floats, factor, kLanes);
    DataCopy(floats[kLanes], product, kLanes);
    DataCopy(integers, rounded, kLanes);
    DataCopy(integers[kLanes], literal_rounded, kLanes);
}

extern "C" __attribute__((visibility("default"))) void quant_div_probe_launch(
    void* stream, void* input, void* float_output, void* integer_output)
{
    quant_div_probe<<<1, nullptr, stream>>>(input, float_output, integer_output);
}
