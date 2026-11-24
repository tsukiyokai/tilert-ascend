#ifndef ASCENDC_CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif

#include "source/contract.h"
#include "source/dual_quant_native.h"
#include "bridge/source/ffn_bridge/native.h"
#include "body/source/routed_int4/routed_int4_front.h"
#include "body/source/routed_int4/routed_int4_trace.h"

namespace ffn_dual_quant_probe_device {
using namespace AscendC;

struct Addresses {
    GM_ADDR bridge[ffn_dual_quant::kBridgeBuffers];
    GM_ADDR body[ffn_dual_quant::kBodyBuffers];
};

template <bool dual>
__global__ __mix__(1, 2) __schedmode__(1) __aicore__
void Kernel(uint64_t sync, Addresses addresses, int64_t rows, int64_t blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync);
    SetSysWorkspace(addresses.bridge[ffn_bridge::kSystem]);
    TPipe pipe;
    auto* input = (__gm__ bfloat16_t*)addresses.bridge[ffn_bridge::kNormalized];
    auto* shared_codes = (__gm__ int8_t*)addresses.bridge[ffn_bridge::kSharedCodes];
    auto* shared_scales = (__gm__ float*)addresses.bridge[ffn_bridge::kSharedScales];
    auto* routed_codes = (__gm__ int8_t*)addresses.body[routed_int4::kRestoredRows];
    auto* routed_scales = (__gm__ float*)addresses.body[routed_int4::kRestoredScale];
    if constexpr (dual) {
        ffn_dual_quant::device::Quantize(pipe, input, shared_codes, shared_scales,
                                       routed_codes, routed_scales, rows);
    } else {
        ffn_bridge::device::Quantize(pipe, input, shared_codes, shared_scales, rows);
    }
    ffn_bridge::device::EndPhase(pipe);
    routed_int4::DispatchInput dispatch{};
    dispatch.bf16_rows = reinterpret_cast<uint64_t>(input);
    dispatch.expert_ids = reinterpret_cast<uint64_t>(addresses.bridge[ffn_bridge::kIds]);
    dispatch.probabilities = reinterpret_cast<uint64_t>(
        addresses.bridge[ffn_bridge::kProbabilities]);
    dispatch.active_list = reinterpret_cast<uint64_t>(
        addresses.body[routed_int4::pipeline::kActiveList]);
    routed_int4::NoTrace timing;
    routed_int4::front::Prepare<dual>(pipe, dispatch, routed_codes, routed_scales,
        (__gm__ int32_t*)addresses.body[routed_int4::kGroupCount],
        (__gm__ int32_t*)addresses.body[routed_int4::kGroupStart],
        (__gm__ int32_t*)addresses.body[routed_int4::kSourceRow],
        (__gm__ int32_t*)addresses.body[routed_int4::kSourceSlot],
        (__gm__ float*)addresses.body[routed_int4::kRowWeight], rows, blocks, timing);
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
}
}

extern "C" void ffn_dual_quant_probe_launch(void* stream, uint64_t sync,
    void** bridge, void** body, int64_t rows, int64_t blocks, bool dual)
{
    ffn_dual_quant_probe_device::Addresses addresses{};
    for (int64_t i = 0; i < ffn_dual_quant::kBridgeBuffers; ++i) {
        addresses.bridge[i] = (GM_ADDR)bridge[i];
    }
    for (int64_t i = 0; i < ffn_dual_quant::kBodyBuffers; ++i) {
        addresses.body[i] = (GM_ADDR)body[i];
    }
    if (dual) {
        ffn_dual_quant_probe_device::Kernel<true><<<blocks, nullptr, stream>>>(
            sync, addresses, rows, blocks);
    } else {
        ffn_dual_quant_probe_device::Kernel<false><<<blocks, nullptr, stream>>>(
            sync, addresses, rows, blocks);
    }
}
