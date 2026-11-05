#ifndef TESTS_GLM52_WHOLE_LAYER_ATTENTION_INDEXED_NATIVE_H
#define TESTS_GLM52_WHOLE_LAYER_ATTENTION_INDEXED_NATIVE_H

#ifndef ASCENDC_CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif
#include "attention_indexed_sfa.h"
#include "attention_native.h"
#include "indexer_front_native.h"
#include "indexer_native.h"
#include "attention_indexed_contract.h"

namespace whole_layer_indexed {

struct DeviceArgs {
    using Address = GM_ADDR;
    Address hidden, token_x, token_scale;
    Address main_weights[9], front_weights[6], tail_weights[3], bias;
    Address main_metadata[6], indexer_metadata[6], caches[4];
    Address scratch, tiling, output;
};

__aicore__ inline void Forward(AscendC::TPipe& pipe, const Config& c,
                               const DeviceArgs& a, int bias_kind,
                               GM_ADDR retained_topk = nullptr)
{
    GM_ADDR memory = a.scratch;
    GM_ADDR topk = retained_topk != nullptr
        ? retained_topk : memory + c.topk;
    const whole_layer_attention::DeviceArgs attention{
        a.token_x, a.token_scale, a.main_weights[0], a.main_weights[1],
        a.main_weights[2], a.main_weights[3], a.main_weights[4],
        a.main_weights[5], a.main_weights[6], a.main_weights[7],
        a.main_weights[8], a.main_metadata[0], a.main_metadata[1],
        a.main_metadata[2], a.caches[0], a.caches[1], topk,
        a.main_metadata[3], a.main_metadata[4], a.main_metadata[5],
        memory + c.attention_offset, memory + c.latent};
    whole_layer_attention::PrologPhase(pipe, c.attention, attention,
        memory + c.query_norm, memory + c.query_norm_scale);

    const indexer_front::device::DeviceArgs front{
        a.hidden, memory + c.query_norm, memory + c.query_norm_scale,
        a.front_weights[0], a.front_weights[1], a.front_weights[2],
        a.front_weights[3], a.front_weights[4], a.indexer_metadata[1],
        a.indexer_metadata[0], a.front_weights[5], memory + c.front_offset,
        a.tiling, memory + c.query, memory + c.key, memory + c.weights,
        memory + c.query_scale, memory + c.key_scale};
    indexer_front::device::Forward(pipe, c.front, front);

    const whole_layer_indexer::DeviceArgs indexer{
        memory + c.query, memory + c.key, memory + c.weights,
        memory + c.query_scale, memory + c.key_scale, a.indexer_metadata[2],
        a.caches[2], a.caches[3], a.indexer_metadata[3], a.indexer_metadata[4],
        a.indexer_metadata[5], memory + c.indexer_offset, topk};
    whole_layer_indexer::Forward(pipe, c.indexer, indexer);
    whole_layer_attention::SfaPhase(pipe, c.attention, attention);

    mla_tail::device::Forward(pipe, (__gm__ bfloat16_t*)(memory + c.latent),
        (__gm__ bfloat16_t*)a.tail_weights[0], (__gm__ int8_t*)a.tail_weights[1],
        (__gm__ float*)a.tail_weights[2], a.bias, bias_kind,
        memory + c.tail_offset, a.tiling + indexer_front::kTilingBytes,
        (__gm__ bfloat16_t*)a.output, c.attention.rows, c.front.capacity,
        c.attention.blocks, c.tail);
    whole_layer_attention::EndPhase(pipe);
}

}  // namespace whole_layer_indexed

#endif
