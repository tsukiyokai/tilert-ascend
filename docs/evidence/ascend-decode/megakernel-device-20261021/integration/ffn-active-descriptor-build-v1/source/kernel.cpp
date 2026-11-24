#ifndef ASCENDC_CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif

#include "contract.h"
#include "descriptor.h"
#include "../bridge/source/ffn_bridge/body_native.h"
#include "../body/source/routed_int4/routed_body_native.h"

namespace ffn_active_descriptor_device {
using namespace AscendC;
using namespace routed_int4;

template <CubeFormat format>
__global__ __mix__(1, 2) __schedmode__(1) __aicore__
void Kernel(uint64_t sync, GM_ADDR input, GM_ADDR residual, GM_ADDR gamma,
    GM_ADDR gate, GM_ADDR bias, ffn_bridge::Addresses bridge,
    const ffn_active_descriptor::BodyArguments arguments,
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
    routed_body::Forward<format, true>(pipe, arguments);
    // NoTrace::FinishFinal is empty. This is the required publication seam,
    // after every body-local TBuf and matrix object has left scope.
    ffn_bridge::device::EndPhase(pipe);
    ffn_bridge::device::Epilogue(pipe,
        (GM_ADDR)arguments.scratch[kOutput],
        (GM_ADDR)arguments.scratch[pipeline::kSharedOutput],
        bridge.values[ffn_bridge::kScaledRouted],
        bridge.values[ffn_bridge::kCombined], arguments.rows, arguments.blocks);
    ffn_bridge::device::EndPhase(pipe);
}
}  // namespace ffn_active_descriptor_device

extern "C" void ffn_active_descriptor_launch(void* stream, uint64_t sync,
    void* input, void* residual, void* gamma, void* gate, void* bias,
    void** routed_weights, void** bridge, void** body,
    routed_int4::pipeline::Input shared, int64_t rows, int64_t blocks,
    int64_t weight_layout, float epsilon)
{
    ffn_bridge::Addresses bridge_addresses{};
    for (int64_t i = 0; i < ffn_active_descriptor::kBridgeBuffers; ++i) {
        bridge_addresses.values[i] = (GM_ADDR)bridge[i];
    }
    const auto arguments = ffn_active_descriptor::MakeBodyArguments(
        routed_weights, bridge, body, shared, rows, blocks);
    if (weight_layout == routed_int4::kWeightNZ) {
        ffn_active_descriptor_device::Kernel<CubeFormat::NZ><<<blocks, nullptr, stream>>>(
            sync, (GM_ADDR)input, (GM_ADDR)residual, (GM_ADDR)gamma, (GM_ADDR)gate,
            (GM_ADDR)bias, bridge_addresses, arguments, epsilon);
    } else {
        ffn_active_descriptor_device::Kernel<CubeFormat::ND><<<blocks, nullptr, stream>>>(
            sync, (GM_ADDR)input, (GM_ADDR)residual, (GM_ADDR)gamma, (GM_ADDR)gate,
            (GM_ADDR)bias, bridge_addresses, arguments, epsilon);
    }
}
