#ifndef ASCENDC_CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif

#include "contract.h"
#include "../communication/communication_device.h"
#include "../local/attention/source/tests/glm52/whole_layer/attention_indexed_native.h"
#include "../local/attention/source/tests/glm52/whole_layer/attention_full_native.h"
#include "../local/ffn/bridge/source/ffn_bridge/body_native.h"
#include "../local/ffn/body/source/routed_int4/routed_body_native.h"

namespace layer_tp_chain {
using namespace AscendC;

__aicore__ inline whole_layer_indexed::DeviceArgs IndexedArguments(
    const AttentionAddresses& addresses)
{
    whole_layer_indexed::DeviceArgs result{};
    result.hidden = (GM_ADDR)addresses.hidden;
    result.token_x = (GM_ADDR)addresses.token_x;
    result.token_scale = (GM_ADDR)addresses.token_scale;
    for (int i = 0; i < 9; ++i) {
        result.main_weights[i] = (GM_ADDR)addresses.main_weights[i];
    }
    for (int i = 0; i < 6; ++i) {
        result.front_weights[i] = (GM_ADDR)addresses.front_weights[i];
        result.main_metadata[i] = (GM_ADDR)addresses.main_metadata[i];
        result.indexer_metadata[i] = (GM_ADDR)addresses.indexer_metadata[i];
    }
    for (int i = 0; i < 3; ++i) {
        result.tail_weights[i] = (GM_ADDR)addresses.tail_weights[i];
    }
    for (int i = 0; i < 4; ++i) {
        result.caches[i] = (GM_ADDR)addresses.caches[i];
    }
    result.bias = (GM_ADDR)addresses.bias;
    result.scratch = (GM_ADDR)addresses.scratch;
    result.tiling = (GM_ADDR)addresses.tiling;
    result.output = (GM_ADDR)addresses.output;
    return result;
}

__aicore__ inline void ReducePhase(TPipe& pipe,
    const Communication& communication, uint64_t partial,
    uint64_t output, int64_t rows, int64_t blocks)
{
    whole_layer::CommunicationBuffers buffers;
    if ASCEND_IS_AIV {
        buffers.Init(pipe);
    }
    whole_layer::AllReduceBf16(buffers,
        (__gm__ uint8_t*)communication.window,
        (__gm__ bfloat16_t*)partial, (__gm__ bfloat16_t*)output,
        rows, communication.capacity, communication.rank,
        communication.ranks, GetBlockIdx(), 2 * blocks);
}

template <bool indexed>
__global__ __mix__(1, 2) __schedmode__(1) __aicore__
void Kernel(uint64_t sync, Config config, Addresses addresses,
            Communication communication)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync);
    SetSysWorkspace((GM_ADDR)addresses.bridge[ffn_bridge::kSystem]);
    TPipe pipe;
    const auto rows = addresses.body.rows;
    const auto blocks = addresses.body.blocks;
    const auto& norm = addresses.norm;
    whole_layer_norm::device::Forward(pipe,
        (__gm__ bfloat16_t*)norm.input, (__gm__ bfloat16_t*)norm.residual,
        (__gm__ bfloat16_t*)norm.gamma, nullptr,
        (__gm__ bfloat16_t*)norm.normalized, (__gm__ bfloat16_t*)norm.retained,
        (__gm__ float*)norm.rstd, (__gm__ int8_t*)norm.codes,
        (__gm__ float*)norm.scales, rows, blocks, config.input_epsilon);
    whole_layer_attention::EndPhase(pipe);

    const auto attention = IndexedArguments(addresses.attention);
    if constexpr (indexed) {
        whole_layer_indexed::Forward(pipe, config.indexed, attention,
                                    config.bias_kind);
    } else {
        const whole_layer_attention::DeviceArgs shared{
            attention.token_x, attention.token_scale,
            attention.main_weights[0], attention.main_weights[1],
            attention.main_weights[2], attention.main_weights[3],
            attention.main_weights[4], attention.main_weights[5],
            attention.main_weights[6], attention.main_weights[7],
            attention.main_weights[8], attention.main_metadata[0],
            attention.main_metadata[1], attention.main_metadata[2],
            attention.caches[0], attention.caches[1],
            (GM_ADDR)addresses.shared_topk, attention.main_metadata[3],
            attention.main_metadata[4], attention.main_metadata[5],
            nullptr, nullptr};
        whole_layer_attention::FullForward(pipe, config.shared, shared,
            attention.tail_weights[0], attention.tail_weights[1],
            attention.tail_weights[2], attention.bias, config.bias_kind,
            attention.scratch, attention.tiling, attention.output);
    }

    ReducePhase(pipe, communication, addresses.attention.output,
                communication.attention_sum, rows, blocks);
    whole_layer_attention::EndPhase(pipe);

    // The norm residual must remain separate from the attention partial.
    ffn_bridge::Addresses bridge{};
    for (int i = 0; i < ffn_bridge::kBuffers; ++i) {
        bridge.values[i] = (GM_ADDR)addresses.bridge[i];
    }
    SetSysWorkspace(bridge.values[ffn_bridge::kSystem]);
    ffn_bridge::Forward(pipe, (GM_ADDR)communication.attention_sum, (GM_ADDR)norm.retained,
        (GM_ADDR)addresses.post_gamma, (GM_ADDR)addresses.gate,
        (GM_ADDR)addresses.correction_bias, bridge, rows, blocks,
        config.ffn_epsilon);
    SetSysWorkspace((GM_ADDR)addresses.body.scratch[
        routed_int4::kMatmulWorkspace]);
    routed_body::Forward<CubeFormat::NZ, true>(pipe, addresses.body,
        (__gm__ bfloat16_t*)bridge.values[ffn_bridge::kScaledRouted],
        (__gm__ bfloat16_t*)bridge.values[ffn_bridge::kCombined]);
    // FFN now has an in-kernel consumer: publish every producer and
    // reset the pipe before allocating the communication buffers.
    whole_layer_attention::EndPhase(pipe);
    ReducePhase(pipe, communication, addresses.bridge[ffn_bridge::kCombined],
                communication.output, rows, blocks);
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
}

}  // namespace layer_tp_chain

extern "C" int layer_tp_indexed_launch(void* stream, uint64_t sync,
    const layer_tp_chain::Config* config,
    const layer_tp_chain::Addresses* addresses,
    const layer_tp_chain::Communication* communication)
{
    if (config == nullptr || addresses == nullptr || communication == nullptr ||
        !layer_tp_chain::Valid(*config, *addresses, *communication)) {
        return -1;
    }
    layer_tp_chain::Kernel<true><<<addresses->body.blocks, nullptr, stream>>>(
        sync, *config, *addresses, *communication);
    return 0;
}

extern "C" int layer_tp_shared_launch(void* stream, uint64_t sync,
    const layer_tp_chain::Config* config,
    const layer_tp_chain::Addresses* addresses,
    const layer_tp_chain::Communication* communication)
{
    if (config == nullptr || addresses == nullptr || communication == nullptr ||
        addresses->shared_topk == 0 ||
        !layer_tp_chain::Valid(*config, *addresses, *communication)) {
        return -1;
    }
    layer_tp_chain::Kernel<false><<<addresses->body.blocks, nullptr, stream>>>(
        sync, *config, *addresses, *communication);
    return 0;
}
