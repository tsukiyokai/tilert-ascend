#ifndef TESTS_GLM52_WHOLE_LAYER_ATTENTION_NATIVE_H
#define TESTS_GLM52_WHOLE_LAYER_ATTENTION_NATIVE_H

#define MLA_PROLOG_VERSION 3
// Explicit instantiation needs no host tiling-key registry. The two vendor
// registries define the same translation-unit global and cannot be combined.
#define MLA_PROLOG_TEMPLATE_TILING_KEY_H
#define C_TEMPLATE 0
#define V_TEMPLATE 1
#include "kernel_operator.h"
#include "attention_contract.h"
#include "attention_sfa_tiling.h"
#include "attention/mla_prolog/op_kernel/arch22/kernel_mla_prolog_split_n_arch22.h"
// The arch22 source names hardware event slots 4..7; CANN 9.2 no longer
// exports those aliases. Preserve the original slot numbers without editing
// vendor code or allocating extra events through TPipe.
#define EVENT_ID4 4U
#define EVENT_ID5 5U
#define EVENT_ID6 6U
#define EVENT_ID7 7U
#include "attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_kernel_mla.h"
#undef EVENT_ID4
#undef EVENT_ID5
#undef EVENT_ID6
#undef EVENT_ID7

namespace whole_layer_attention {

struct DeviceArgs {
    using Address = GM_ADDR;
    Address token_x, token_scale, weight_dq, weight_uq_qr, weight_uk;
    Address weight_dkv_kr, scale_dq, scale_uq_qr, scale_dkv_kr;
    Address gamma_cq, gamma_ckv, sin, cos, slots;
    Address kv_cache, rope_cache, topk, query_lengths, kv_lengths, page_table;
    Address scratch, output;
};

__aicore__ inline uint32_t Ceil(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1) / divisor;
}

__aicore__ inline optiling::MlaPrologBaseParams PrologTiling(const Config& config)
{
    optiling::MlaPrologBaseParams tiling{};
    tiling.batchSize = 1;
    tiling.stepBatchSize = config.rows;
    tiling.stepNumHeadDequant = kHeads;
    tiling.tokenSize = config.rows;
    tiling.seq1Size = config.rows;
    tiling.seq2Size = config.table_pages * kPage;
    tiling.headSizeX = kHidden;
    tiling.headSizeCq = kQueryRank;
    tiling.headSizeCkv = kLatent;
    tiling.headSizeQc = kHeads * kNope;
    tiling.headSizeQr = kHeads * kRope;
    tiling.headSizeKr = kRope;
    tiling.numHeadSize = kHeads;
    tiling.numHeadKvSize = 1;
    tiling.dimHeadSizeQc = kNope;
    tiling.dimHeadRope = kRope;
    tiling.blockNum = config.pages;
    tiling.blockSize = kPage;
    tiling.mm1SingleCoreN = Ceil(Ceil(kQueryRank, config.blocks), 32) * 32;
    if (tiling.mm1SingleCoreN < 64) {
        tiling.mm1SingleCoreN = 64;
    }
    tiling.mm1BlockNum = Ceil(kQueryRank, tiling.mm1SingleCoreN);
    tiling.mm2SingleCoreN = config.blocks >= 9 ? 64 :
        Ceil(Ceil(kLatent + kRope, config.blocks), 32) * 32;
    tiling.mm2BlockNum = Ceil(kLatent + kRope, tiling.mm2SingleCoreN);
    tiling.mm3SingleCoreN = Ceil(Ceil(kHeads * (kNope + kRope), config.blocks), 32) * 32;
    tiling.mm3BlockNum = Ceil(kHeads * (kNope + kRope), tiling.mm3SingleCoreN);
    tiling.mm4SingleCoreBatch = Ceil(kHeads, config.blocks);
    tiling.mm4BlockNum = Ceil(kHeads, tiling.mm4SingleCoreBatch);
    tiling.vectorBlockNum = config.rows < 2 * config.blocks ? config.rows : 2 * config.blocks;
    tiling.dtileSize = kLatent;
    tiling.tileSize = kPage;
    tiling.reciprocalCq = 1.0F / kQueryRank;
    tiling.epsilonCq = config.epsilon_cq;
    tiling.reciprocalCkv = 1.0F / kLatent;
    tiling.epsilonCkv = config.epsilon_ckv;
    tiling.qcQrScale = 1.0F;
    tiling.kcScale = 1.0F;
    tiling.mm1NGroup = tiling.mm1KGroup = 1;
    tiling.mm2NGroup = tiling.mm2KGroup = 1;
    tiling.kvCacheStride0 = kPage * kLatent;
    tiling.krCacheStride0 = kPage * kRope;
    return tiling;
}

__aicore__ inline SparseFlashAttentionTilingDataMla SfaTiling(const Config& config)
{
    SparseFlashAttentionTilingDataMla tiling{};
    auto& base = tiling.baseParams;
    base.batchSize = 1;
    base.seqSize = config.table_pages * kPage;
    base.qSeqSize = config.rows;
    base.blockSize = kPage;
    base.maxBlockNumPerBatch = config.table_pages;
    base.scaleValue = config.scale;
    base.nNumOfQInOneGroup = kHeads;
    base.actualLenDimsQ = 1;
    base.actualLenDimsKV = 1;
    base.outputLayout = 1;
    base.sparseMode = 3;
    base.preTokens = INT64_MAX;
    base.nextTokens = INT64_MAX;
    base.attentionMode = 2;
    base.sparseBlockSize = 1;
    base.sparseBlockCount = kTopK;
    base.keyStride0 = kPage * kLatent;
    tiling.singleCoreParams.usedCoreNum = config.blocks;
    tiling.singleCoreTensorSize.mmResUbSize = config.mm_elements;
    tiling.singleCoreTensorSize.bmm2ResUbSize = config.bmm_elements;
    tiling.innerSplitParams.mBaseSize = kHeads;
    tiling.innerSplitParams.s2BaseSize = config.s2_base;
    return tiling;
}

__aicore__ inline void EndPhase(AscendC::TPipe& pipe)
{
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    AscendC::SyncAll<false>();
    pipe.Reset();
}

// Every launched AIC and AIV calls this function. The caller owns the sync base
// and launches no more mixed groups than can be resident on the device.
__aicore__ inline void Forward(AscendC::TPipe& pipe, const Config& config,
                                const DeviceArgs& args)
{
    GM_ADDR query = args.scratch + config.query_offset;
    GM_ADDR rope = args.scratch + config.rope_offset;
    GM_ADDR workspace = args.scratch + config.work_offset;
    {
        const auto tiling = PrologTiling(config);
        using PrologType = MlaProlog::MLAPType<int8_t, int8_t, bfloat16_t, float,
            MlaProlog::CACHE_MODE::PA_BSND, false, false,
            MlaProlog::EMPTY_TENSOR_MODE::NON_EMPTY,
            MlaProlog::ACTUAL_SEQ_MODE::DISABLED, false, 2, true>;
        MlaProlog::MlaPrologVecS1CubS2<PrologType> prolog(&pipe, nullptr, &tiling);
        prolog.Init(args.token_x, args.weight_dq, args.weight_uq_qr,
            args.weight_uk, args.weight_dkv_kr, args.gamma_cq, args.gamma_ckv,
            args.sin, args.cos, args.slots, args.kv_cache, args.rope_cache,
            args.token_scale, args.scale_dq, args.scale_uq_qr, args.scale_dkv_kr,
            nullptr, nullptr, nullptr, nullptr, nullptr,
            query, rope, nullptr, nullptr, nullptr, workspace);
        prolog.Process();
    }
    EndPhase(pipe);
    {
        const auto tiling = SfaTiling(config);
        using AttentionType = SFAType<bfloat16_t, bfloat16_t, bfloat16_t,
            false, SFA_LAYOUT::TND, SFA_LAYOUT::PA_BSND, V_TEMPLATE, true>;
        SparseFlashAttentionMla<AttentionType> attention;
        attention.Init(query, args.kv_cache, args.kv_cache, args.topk,
            args.query_lengths, args.kv_lengths, args.page_table,
            rope, args.rope_cache, args.output, nullptr, nullptr, workspace,
            &tiling, nullptr, &pipe);
        attention.Process();
    }
    EndPhase(pipe);
}

}  // namespace whole_layer_attention

#endif
