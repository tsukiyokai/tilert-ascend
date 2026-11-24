#include "native.h"

namespace ffn_callable_device {
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
    Forward<format>(pipe, input, residual, gamma, gate, bias,
        w, bridge, body, shared, rows, blocks, epsilon);

}
}  // namespace ffn_callable_device

extern "C" void ffn_callable_launch(void* stream, uint64_t sync,
    void* input, void* residual, void* gamma, void* gate, void* bias,
    void** routed_weights, void** bridge, void** body,
    routed_int4::pipeline::Input shared, int64_t rows, int64_t blocks,
    int64_t weight_layout, float epsilon)
{
    ffn_callable_device::WeightAddresses weights{};
    ffn_bridge::Addresses bridge_addresses{};
    ffn_callable_device::BodyAddresses body_addresses{};
    for (int64_t i = 0; i < ffn_callable::kRoutedWeights; ++i) {
        weights.values[i] = (GM_ADDR)routed_weights[i];
    }
    for (int64_t i = 0; i < ffn_callable::kBridgeBuffers; ++i) {
        bridge_addresses.values[i] = (GM_ADDR)bridge[i];
    }
    for (int64_t i = 0; i < ffn_callable::kBodyBuffers; ++i) {
        body_addresses.values[i] = (GM_ADDR)body[i];
    }
    if (weight_layout == routed_int4::kWeightNZ) {
        ffn_callable_device::Kernel<CubeFormat::NZ><<<blocks, nullptr, stream>>>(
            sync, (GM_ADDR)input, (GM_ADDR)residual, (GM_ADDR)gamma, (GM_ADDR)gate,
            (GM_ADDR)bias, weights, bridge_addresses, body_addresses,
            shared, rows, blocks, epsilon);
    } else {
        ffn_callable_device::Kernel<CubeFormat::ND><<<blocks, nullptr, stream>>>(
            sync, (GM_ADDR)input, (GM_ADDR)residual, (GM_ADDR)gamma, (GM_ADDR)gate,
            (GM_ADDR)bias, weights, bridge_addresses, body_addresses,
            shared, rows, blocks, epsilon);
    }
}
