#ifndef ASCENDC_CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif

#include "contract.h"
#include "../bridge/source/ffn_bridge/body_native.h"
#include "../body/source/routed_int4/routed_body_native.h"

namespace ffn_composed_device {
using namespace AscendC;
using namespace routed_int4;

struct BodyAddresses {
    GM_ADDR values[ffn_composed::kBodyBuffers];
};

struct WeightAddresses {
    GM_ADDR values[ffn_composed::kRoutedWeights];
};

template <CubeFormat format>
__global__ __mix__(1, 2) __schedmode__(1) __aicore__
void Kernel(uint64_t sync, GM_ADDR input, GM_ADDR residual, GM_ADDR gamma,
    GM_ADDR gate, GM_ADDR bias, WeightAddresses w, ffn_bridge::Addresses bridge,
    BodyAddresses body, pipeline::Input shared, int64_t rows, int64_t blocks,
    float epsilon)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync);
    SetSysWorkspace(bridge.values[ffn_bridge::kSystem]);
    TPipe pipe;
    ffn_bridge::Forward(pipe, input, residual, gamma, gate, bias,
                        bridge, rows, blocks, epsilon);

    // The bridge's final EndPhase has drained all CUBE work and reset UB.
    // Only its GM outputs survive into the unchanged routed/shared body.
    SetSysWorkspace(body.values[kMatmulWorkspace]);
    DispatchInput dispatch;
    dispatch.bf16_rows = (uint64_t)bridge.values[ffn_bridge::kNormalized];
    dispatch.expert_ids = (uint64_t)bridge.values[ffn_bridge::kIds];
    dispatch.probabilities = (uint64_t)bridge.values[ffn_bridge::kProbabilities];
    routed_body::Forward<format, true>(pipe,
        (__gm__ int8_t*)body.values[kRestoredRows],
        (__gm__ int4b_t*)w.values[0], (__gm__ float*)w.values[1],
        (__gm__ ScaleStorage*)w.values[2], (__gm__ int4b_t*)w.values[3],
        (__gm__ float*)w.values[4], (__gm__ ScaleStorage*)w.values[5],
        (__gm__ float*)body.values[kRestoredScale],
        (__gm__ int32_t*)body.values[kGroupCount],
        (__gm__ int32_t*)body.values[kGroupStart],
        (__gm__ int32_t*)body.values[kSourceRow],
        (__gm__ int32_t*)body.values[kSourceSlot],
        (__gm__ float*)body.values[kRowWeight],
        (__gm__ int4b_t*)body.values[kStacked],
        (__gm__ int32_t*)body.values[kGateProduct],
        (__gm__ int4b_t*)body.values[kMiddle],
        (__gm__ float*)body.values[kMiddleScale],
        (__gm__ int32_t*)body.values[kDownProduct],
        (__gm__ bfloat16_t*)body.values[kRouteOutput],
        (__gm__ bfloat16_t*)body.values[kOutput],
        body.values[kGateTiling], body.values[kDownTiling],
        body.values[kMatmulWorkspace], ffn_bridge::kExperts, rows, blocks,
        dispatch, nullptr, shared);
    // NoTrace::FinishFinal is empty. This is the required publication seam,
    // after every body-local TBuf and matrix object has left scope.
    ffn_bridge::device::EndPhase(pipe);
    ffn_bridge::device::Epilogue(pipe, body.values[kOutput],
        body.values[pipeline::kSharedOutput], bridge.values[ffn_bridge::kScaledRouted],
        bridge.values[ffn_bridge::kCombined], rows, blocks);
    ffn_bridge::device::EndPhase(pipe);
}
}  // namespace ffn_composed_device

extern "C" void ffn_composed_launch(void* stream, uint64_t sync,
    void* input, void* residual, void* gamma, void* gate, void* bias,
    void** routed_weights, void** bridge, void** body,
    routed_int4::pipeline::Input shared, int64_t rows, int64_t blocks,
    int64_t weight_layout, float epsilon)
{
    ffn_composed_device::WeightAddresses weights{};
    ffn_bridge::Addresses bridge_addresses{};
    ffn_composed_device::BodyAddresses body_addresses{};
    for (int64_t i = 0; i < ffn_composed::kRoutedWeights; ++i) {
        weights.values[i] = (GM_ADDR)routed_weights[i];
    }
    for (int64_t i = 0; i < ffn_composed::kBridgeBuffers; ++i) {
        bridge_addresses.values[i] = (GM_ADDR)bridge[i];
    }
    for (int64_t i = 0; i < ffn_composed::kBodyBuffers; ++i) {
        body_addresses.values[i] = (GM_ADDR)body[i];
    }
    if (weight_layout == routed_int4::kWeightNZ) {
        ffn_composed_device::Kernel<CubeFormat::NZ><<<blocks, nullptr, stream>>>(
            sync, (GM_ADDR)input, (GM_ADDR)residual, (GM_ADDR)gamma, (GM_ADDR)gate,
            (GM_ADDR)bias, weights, bridge_addresses, body_addresses,
            shared, rows, blocks, epsilon);
    } else {
        ffn_composed_device::Kernel<CubeFormat::ND><<<blocks, nullptr, stream>>>(
            sync, (GM_ADDR)input, (GM_ADDR)residual, (GM_ADDR)gamma, (GM_ADDR)gate,
            (GM_ADDR)bias, weights, bridge_addresses, body_addresses,
            shared, rows, blocks, epsilon);
    }
}
