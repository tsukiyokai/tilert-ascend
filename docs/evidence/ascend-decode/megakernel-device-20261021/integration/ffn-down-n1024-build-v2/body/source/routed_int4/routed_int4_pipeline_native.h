#ifndef TESTS_GLM52_ROUTED_INT4_PIPELINE_NATIVE_H
#define TESTS_GLM52_ROUTED_INT4_PIPELINE_NATIVE_H

#include "routed_int4_pipeline.h"
#include "../shared_w8/shared_w8_native.h"

namespace routed_int4::pipeline {

static_assert(shared_w8::kDynamicPerformanceQuant,
              "the shared pipeline requires CANN dynamic-performance quantization");

struct Stages {
    Input input;
    int64_t rows;
    int64_t blocks;
    shared_w8::device::Buffers held;

    __aicore__ inline Stages(Input value, int64_t tokens, int64_t groups)
        : input(value), rows(tokens), blocks(groups) {}

    template <typename T>
    __aicore__ inline __gm__ T* Payload(uint64_t offset = 0)
    {
        return reinterpret_cast<__gm__ T*>(input.payload + offset);
    }

    __aicore__ inline void StageInput(AscendC::TPipe& pipe)
    {
        if ASCEND_IS_AIV {
            // Input preparation resets the pipe after barrier 0. This scope
            // retains only the staged GM bytes across that reset.
            shared_w8::device::Buffers staging;
            staging.Init(pipe);
            shared_w8::device::StageInput(staging,
                reinterpret_cast<__gm__ int8_t*>(input.input), Payload<int8_t>(),
                rows, input.capacity, 2 * blocks);
            shared_w8::device::Publish();
        }
    }

    __aicore__ inline void InitResident(AscendC::TPipe& pipe)
    {
        if ASCEND_IS_AIV {
            held.Init(pipe);
        }
    }

    __aicore__ inline void Gate(AscendC::TPipe& pipe)
    {
        if ASCEND_IS_AIC {
            shared_w8::device::Project(pipe, Payload<int8_t>(),
                reinterpret_cast<__gm__ int8_t*>(input.gate_weight),
                Payload<int32_t>(input.partial_offset),
                reinterpret_cast<__gm__ uint8_t*>(input.tiling),
                input.capacity, blocks, true);
            shared_w8::device::Publish();
        }
    }

    __aicore__ inline void Activate()
    {
        if ASCEND_IS_AIV {
            shared_w8::device::Activate(held,
                Payload<int32_t>(input.partial_offset),
                reinterpret_cast<__gm__ float*>(input.gate_scale),
                reinterpret_cast<__gm__ float*>(input.input_scale),
                Payload<int8_t>(input.middle_offset),
                Payload<float>(input.scale_offset),
                rows, input.capacity, 2 * blocks);
            shared_w8::device::Publish();
        }
    }

    __aicore__ inline void Down(AscendC::TPipe& pipe)
    {
        if ASCEND_IS_AIC {
            // The preceding routed gate object has ended and left scope.
            pipe.Reset();
            shared_w8::device::Project(pipe, Payload<int8_t>(input.middle_offset),
                reinterpret_cast<__gm__ int8_t*>(input.down_weight),
                Payload<int32_t>(input.down_offset),
                reinterpret_cast<__gm__ uint8_t*>(
                    input.tiling + shared_w8::kTilingStride),
                input.capacity, blocks, false);
            shared_w8::device::Publish();
        }
    }

    __aicore__ inline void Store(AscendC::TPipe& pipe)
    {
        if ASCEND_IS_AIV {
            mla_tail::device::DequantizeOutputRows<
                mla_tail::device::ScaleOrder::CombinedScales>(
                pipe, Payload<int32_t>(input.down_offset),
                reinterpret_cast<__gm__ float*>(input.down_scale),
                Payload<float>(input.scale_offset), nullptr, 0,
                reinterpret_cast<__gm__ bfloat16_t*>(input.output),
                rows, input.capacity, AscendC::GetBlockIdx(), 2 * blocks);
            shared_w8::device::Publish();
        }
    }
};

}  // namespace routed_int4::pipeline

#endif
