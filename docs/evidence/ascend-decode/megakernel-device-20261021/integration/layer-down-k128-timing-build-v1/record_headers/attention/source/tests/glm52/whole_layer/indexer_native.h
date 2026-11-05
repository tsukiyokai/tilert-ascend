#ifndef TESTS_GLM52_WHOLE_LAYER_INDEXER_NATIVE_H
#define TESTS_GLM52_WHOLE_LAYER_INDEXER_NATIVE_H

#include "kernel_operator.h"
#include "indexer_contract.h"
#include "indexer_tiling.h"
// Preserve vendor hardware event slots omitted from the CANN 9.2 aliases.
#define EVENT_ID4 4U
#define EVENT_ID5 5U
#include "attention/quant_lightning_indexer/op_kernel/arch22/quant_lightning_indexer_kernel_arch22.h"
#include "attention/lightning_indexer/op_kernel/arch22/lightning_indexer_kernel_arch22.h"
#undef EVENT_ID4
#undef EVENT_ID5

namespace whole_layer_indexer {

struct DeviceArgs {
    using Address = GM_ADDR;
    Address query, new_keys, weights, query_scale, new_key_scale, slots;
    Address key_cache, scale_cache, query_lengths, key_lengths, page_table;
    Address scratch, output;
};

__aicore__ inline void EndPhase(AscendC::TPipe& pipe)
{
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    AscendC::SyncAll<false>();
    pipe.Reset();
}

template <typename Key, bool Quantized>
__aicore__ inline void WriteCache(AscendC::TPipe& pipe, const Config& config,
                                  const DeviceArgs& args)
{
    if ASCEND_IS_AIV {
        AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
        pipe.InitBuffer(buffer, 768);
        auto key = buffer.Get<Key>();
        auto scale = buffer.Get<half>()[128];
        auto slot_values = buffer.Get<int64_t>()[64];
        AscendC::GlobalTensor<Key> source, cache;
        AscendC::GlobalTensor<half> source_scale, cache_scale;
        AscendC::GlobalTensor<int64_t> slots;
        source.SetGlobalBuffer((__gm__ Key*)args.new_keys);
        cache.SetGlobalBuffer((__gm__ Key*)args.key_cache);
        slots.SetGlobalBuffer((__gm__ int64_t*)args.slots);
        // Metadata may change between graph replays. Load through MTE2
        // instead of retaining scalar-GM cache lines across launches.
        AscendC::DataCopyPad(slot_values, slots,
            {1, config.rows * 8U, 0, 0, 0}, {false, 0, 0, 0});
        const auto slot_event = pipe.FetchEventID(AscendC::HardEvent::MTE2_S);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(slot_event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(slot_event);
        if constexpr (Quantized) {
            source_scale.SetGlobalBuffer((__gm__ half*)args.new_key_scale);
            cache_scale.SetGlobalBuffer((__gm__ half*)args.scale_cache);
        }
        const auto load_event = pipe.FetchEventID(AscendC::HardEvent::MTE2_MTE3);
        const auto store_event = pipe.FetchEventID(AscendC::HardEvent::MTE3_MTE2);
        for (uint32_t row = AscendC::GetBlockIdx(); row < config.rows;
             row += 2 * config.blocks) {
            const int64_t slot = slot_values.GetValue(row);
            if (slot < 0 || slot >= static_cast<int64_t>(config.pages) * kPage) {
                continue;
            }
            AscendC::DataCopy(key, source[row * kHeadDim], kHeadDim);
            if constexpr (Quantized) {
                AscendC::DataCopyPad(scale, source_scale[row],
                    {1, sizeof(half), 0, 0, 0}, {false, 0, 0, 0});
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(load_event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(load_event);
            AscendC::DataCopy(cache[slot * kHeadDim], key, kHeadDim);
            if constexpr (Quantized) {
                AscendC::DataCopyPad(cache_scale[slot], scale,
                    {1, sizeof(half), 0, 0, 0});
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(store_event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(store_event);
        }
    }
}

__aicore__ inline QLITilingData QuantTiling(const Config& config)
{
    QLITilingData tiling{};
    tiling.bSize = 1;
    tiling.tSize = tiling.s1Size = config.rows;
    tiling.n2Size = 1;
    tiling.gSize = kHeads;
    tiling.s2Size = config.table_pages * kPage;
    tiling.sparseCount = kTopK;
    tiling.keyStride0 = kPage * kHeadDim;
    tiling.keyDequantScaleStride0 = kPage;
    tiling.usedCoreNum = config.blocks;
    tiling.blockSize = kPage;
    tiling.maxBlockNumPerBatch = config.table_pages;
    tiling.sparseMode = 3;
    return tiling;
}

__aicore__ inline LITilingData Bf16Tiling(const Config& config)
{
    LITilingData tiling{};
    tiling.bSize = 1;
    tiling.n2Size = 1;
    tiling.gSize = kHeads;
    tiling.s1Size = config.rows;
    tiling.s2Size = config.table_pages * kPage;
    tiling.sparseCount = kTopK;
    tiling.usedCoreNum = config.blocks;
    tiling.blockSize = kPage;
    tiling.maxBlockNumPerBatch = config.table_pages;
    tiling.sparseMode = 3;
    tiling.preTokens = tiling.nextTokens = INT64_MAX;
    tiling.returnValue = 0;
    tiling.keyStride0 = kPage * kHeadDim;
    return tiling;
}

// Every resident AIC and AIV enters both outer barriers. Vendor ProcessDecode
// uses its own AIV-only SyncAll before merging top-k lists; its hardware flag
// differs from the mixed-core barrier and its scratch remains live until the
// final mixed barrier completes.
__aicore__ inline void Forward(AscendC::TPipe& pipe, const Config& config,
                               const DeviceArgs& args)
{
    if (config.quantized) {
        WriteCache<int8_t, true>(pipe, config, args);
    } else {
        WriteCache<bfloat16_t, false>(pipe, config, args);
    }
    EndPhase(pipe);
    if (config.quantized) {
        const auto tiling = QuantTiling(config);
        using Type = QLICommon::QLIType<int8_t, int8_t, int32_t, true,
            QLICommon::LI_LAYOUT::TND, QLICommon::LI_LAYOUT::PA_BSND>;
        QLIKernel::QLIPreload<Type> selector;
        selector.Init(args.query, args.key_cache, args.weights, args.query_scale,
            args.scale_cache, args.query_lengths, args.key_lengths,
            args.page_table, args.output, args.scratch, &tiling, &pipe);
        selector.Process();
    } else {
        const auto tiling = Bf16Tiling(config);
        using Type = LICommon::LIType<bfloat16_t, bfloat16_t, int32_t, true,
            LICommon::LI_LAYOUT::TND, LICommon::LI_LAYOUT::PA_BSND, false>;
        LIKernel::LightningIndexerKernel<Type> selector;
        selector.Init(args.query, args.key_cache, args.weights,
            args.query_lengths, args.key_lengths, args.page_table, args.output,
            nullptr, args.scratch, &tiling, &pipe);
        selector.Process();
    }
    EndPhase(pipe);
}

}  // namespace whole_layer_indexer

#endif
