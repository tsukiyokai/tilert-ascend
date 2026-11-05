#ifndef ASCENDC_CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif

#include "contract.h"
#include "descriptor.h"
#include "../bridge/source/ffn_bridge/body_native.h"
#include "../body/source/routed_int4/routed_body_native.h"

namespace ffn_gu_static_device {
using namespace AscendC;
using namespace routed_int4;

template <CubeFormat format>
__global__ __mix__(1, 2) __schedmode__(1) __aicore__
void Kernel(uint64_t sync, GM_ADDR input, GM_ADDR residual, GM_ADDR gamma,
    GM_ADDR gate, GM_ADDR bias, ffn_bridge::Addresses bridge,
    const ffn_gu_static::BodyArguments arguments,
    float epsilon)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync);
    SetSysWorkspace(bridge.values[ffn_bridge::kSystem]);
    TPipe pipe;
    ffn_bridge::Forward(pipe, input, residual, gamma, gate, bias,
                        bridge, arguments.rows, arguments.blocks, epsilon);

    // The bridge's final EndPhase has drained all CUBE work and reset UB.
    // Only its GM outputs survive into the unchanged routed/shared body.
    SetSysWorkspace((GM_ADDR)arguments.scratch[kMatmulWorkspace]);
    routed_body::Forward<format, true>(pipe, arguments,
        (__gm__ bfloat16_t*)bridge.values[ffn_bridge::kScaledRouted],
        (__gm__ bfloat16_t*)bridge.values[ffn_bridge::kCombined]);
    // All output consumers follow this global task on the same stream.
    // The callable body and its internal publication boundaries are unchanged.
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();

}
}  // namespace ffn_gu_static_device

extern "C" void ffn_gu_static_launch(void* stream, uint64_t sync,
    void* input, void* residual, void* gamma, void* gate, void* bias,
    void** routed_weights, void** bridge, void** body,
    routed_int4::pipeline::Input shared, int64_t rows, int64_t blocks,
    int64_t weight_layout, float epsilon)
{
    ffn_bridge::Addresses bridge_addresses{};
    for (int64_t i = 0; i < ffn_gu_static::kBridgeBuffers; ++i) {
        bridge_addresses.values[i] = (GM_ADDR)bridge[i];
    }
    const auto arguments = ffn_gu_static::MakeBodyArguments(
        routed_weights, bridge, body, shared, rows, blocks);
    if (weight_layout == routed_int4::kWeightNZ) {
        ffn_gu_static_device::Kernel<CubeFormat::NZ><<<blocks, nullptr, stream>>>(
            sync, (GM_ADDR)input, (GM_ADDR)residual, (GM_ADDR)gamma, (GM_ADDR)gate,
            (GM_ADDR)bias, bridge_addresses, arguments, epsilon);
    } else {
        ffn_gu_static_device::Kernel<CubeFormat::ND><<<blocks, nullptr, stream>>>(
            sync, (GM_ADDR)input, (GM_ADDR)residual, (GM_ADDR)gamma, (GM_ADDR)gate,
            (GM_ADDR)bias, bridge_addresses, arguments, epsilon);
    }
}
