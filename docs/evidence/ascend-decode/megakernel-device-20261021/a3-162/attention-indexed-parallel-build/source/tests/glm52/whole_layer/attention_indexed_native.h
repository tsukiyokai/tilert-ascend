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

struct PhaseTrace {
#ifdef ATTENTION_INDEXED_PROFILE
    int64_t stamps[kTraceMarks];

    __aicore__ inline void Mark(int index)
    {
        stamps[index] = AscendC::GetSystemCycle();
    }

    __aicore__ inline void Store(const Config& c, GM_ADDR memory)
    {
        int64_t kind = 0;
        if ASCEND_IS_AIV {
            kind = 1;
        }
        const int64_t index = AscendC::GetBlockIdx();
        const int64_t slot = kind == 0 ? index : c.attention.blocks + index;
        AscendC::GlobalTensor<int64_t> record;
        record.SetGlobalBuffer((__gm__ int64_t*)(memory + c.trace) +
            slot * kTraceWords, kTraceWords);
        for (uint64_t word = 0; word < kTraceWords; ++word) {
            record.SetValue(word, 0);
        }
        record.SetValue(0, 1);
        record.SetValue(1, kind);
        record.SetValue(2, index);
        record.SetValue(3, c.attention.rows);
        record.SetValue(4, c.attention.blocks);
        for (uint64_t mark = 0; mark < kTraceMarks; ++mark) {
            record.SetValue(kTraceStampOffset + mark, stamps[mark]);
        }
        for (uint64_t word = 0; word < kTraceWords; word += 8) {
            AscendC::DataCacheCleanAndInvalid<int64_t,
                AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(record[word]);
        }
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
#else
    __aicore__ inline void Mark(int) {}
    __aicore__ inline void Store(const Config&, GM_ADDR) {}
#endif
};

__aicore__ inline void Forward(AscendC::TPipe& pipe, const Config& c,
                               const DeviceArgs& a, int bias_kind)
{
    GM_ADDR memory = a.scratch;
    PhaseTrace trace;
    trace.Mark(0);
    const whole_layer_attention::DeviceArgs attention{
        a.token_x, a.token_scale, a.main_weights[0], a.main_weights[1],
        a.main_weights[2], a.main_weights[3], a.main_weights[4],
        a.main_weights[5], a.main_weights[6], a.main_weights[7],
        a.main_weights[8], a.main_metadata[0], a.main_metadata[1],
        a.main_metadata[2], a.caches[0], a.caches[1], memory + c.topk,
        a.main_metadata[3], a.main_metadata[4], a.main_metadata[5],
        memory + c.attention_offset, memory + c.latent};
    whole_layer_attention::PrologPhase(pipe, c.attention, attention,
        memory + c.query_norm, memory + c.query_norm_scale);
    trace.Mark(1);

    const indexer_front::device::DeviceArgs front{
        a.hidden, memory + c.query_norm, memory + c.query_norm_scale,
        a.front_weights[0], a.front_weights[1], a.front_weights[2],
        a.front_weights[3], a.front_weights[4], a.indexer_metadata[1],
        a.indexer_metadata[0], a.front_weights[5], memory + c.front_offset,
        a.tiling, memory + c.query, memory + c.key, memory + c.weights,
        memory + c.query_scale, memory + c.key_scale};
    indexer_front::device::Forward(pipe, c.front, front);
    trace.Mark(2);

    const whole_layer_indexer::DeviceArgs indexer{
        memory + c.query, memory + c.key, memory + c.weights,
        memory + c.query_scale, memory + c.key_scale, a.indexer_metadata[2],
        a.caches[2], a.caches[3], a.indexer_metadata[3], a.indexer_metadata[4],
        a.indexer_metadata[5], memory + c.indexer_offset, memory + c.topk};
    whole_layer_indexer::Forward(pipe, c.indexer, indexer);
    trace.Mark(3);
    whole_layer_attention::SfaPhase(pipe, c.attention, attention);
    trace.Mark(4);

    mla_tail::device::Forward(pipe, (__gm__ bfloat16_t*)(memory + c.latent),
        (__gm__ bfloat16_t*)a.tail_weights[0], (__gm__ int8_t*)a.tail_weights[1],
        (__gm__ float*)a.tail_weights[2], a.bias, bias_kind,
        memory + c.tail_offset, a.tiling + indexer_front::kTilingBytes,
        (__gm__ bfloat16_t*)a.output, c.attention.rows, c.front.capacity,
        c.attention.blocks, c.tail);
    whole_layer_attention::EndPhase(pipe);
    trace.Mark(5);
    trace.Store(c, memory);
}

}  // namespace whole_layer_indexed

#endif
