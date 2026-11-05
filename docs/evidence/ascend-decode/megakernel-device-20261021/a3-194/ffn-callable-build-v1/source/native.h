#ifndef TILERT_FFN_CALLABLE_NATIVE_H
#define TILERT_FFN_CALLABLE_NATIVE_H

#ifndef ASCENDC_CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif

#include "contract.h"
#include "../bridge/source/ffn_bridge/body_native.h"
#include "../body/source/routed_int4/routed_body_native.h"

namespace ffn_callable_device {
using namespace AscendC;
using namespace routed_int4;

struct BodyAddresses {
    GM_ADDR values[ffn_callable::kBodyBuffers];
};

struct WeightAddresses {
    GM_ADDR values[ffn_callable::kRoutedWeights];
};

template <CubeFormat format>
__aicore__ inline
void Forward(TPipe& pipe, GM_ADDR input, GM_ADDR residual, GM_ADDR gamma,
    GM_ADDR gate, GM_ADDR bias, WeightAddresses w, ffn_bridge::Addresses bridge,
    BodyAddresses body, pipeline::Input shared, int64_t rows, int64_t blocks,
    float epsilon)
{
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
}  // namespace ffn_callable_device

#endif
