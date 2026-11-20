#ifndef TESTS_GLM52_WHOLE_LAYER_ATTENTION_NATIVE_H
#define TESTS_GLM52_WHOLE_LAYER_ATTENTION_NATIVE_H

#define MLA_PROLOG_VERSION 3


#define MLA_PROLOG_TEMPLATE_TILING_KEY_H
#define C_TEMPLATE 0
#define V_TEMPLATE 1
#include "kernel_operator.h"
#include "attention_contract.h"
#include "attention_sfa_tiling.h"
#include "attention/mla_prolog/op_kernel/arch22/kernel_mla_prolog_split_n_arch22.h"


#define EVENT_ID4 4U
#define EVENT_ID5 5U
#define EVENT_ID6 6U
#define EVENT_ID7 7U
#include "attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_kernel_mla.h"
#undef EVENT_ID4
#undef EVENT_ID5
#undef EVENT_ID6
#undef EVENT_ID7
#ifdef K112840_UNION_SFA
#include "union_sfa_native.h"
#endif
#ifndef DRAFT_SFA_TOPK
#define DRAFT_SFA_TOPK kTopK
#endif

#ifndef K11240_TOP_NOINLINE
#ifdef K11240_OUTLINE_TOP
#define K11240_TOP_NOINLINE __attribute__((noinline))
#else
#define K11240_TOP_NOINLINE inline
#endif
#endif

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
#ifdef K112194_PROLOG_CUBE_BAL
    tiling.mm2SingleCoreN = config.blocks >= 18 ? 32 :
        (config.blocks >= 9 ? 64 : Ceil(Ceil(kLatent + kRope, config.blocks), 32) * 32);
#else
    tiling.mm2SingleCoreN = config.blocks >= 9 ? 64 :
        Ceil(Ceil(kLatent + kRope, config.blocks), 32) * 32;
#endif
#ifdef K112854_TP16_PROLOG_PAR
    if (kHeads == 4U && config.blocks == 24U) {
        tiling.mm1SingleCoreN = 128;
        tiling.mm1BlockNum = 16;
        tiling.mm2SingleCoreN = 96;
        tiling.mm2BlockNum = 6;
    } else {
        tiling.mm2BlockNum = Ceil(kLatent + kRope, tiling.mm2SingleCoreN);
    }
#else
    tiling.mm2BlockNum = Ceil(kLatent + kRope, tiling.mm2SingleCoreN);
#endif
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
    base.sparseBlockCount = DRAFT_SFA_TOPK;
    base.keyStride0 = kPage * kLatent;
    tiling.singleCoreParams.usedCoreNum = config.blocks;
#ifdef K112120_SFA_COMPACT_WS
    const uint32_t compact_mm = 16U * ((config.s2_base + 31U) & ~31U);
    const uint32_t compact_bmm = 16U * kLatent;
    tiling.singleCoreTensorSize.mmResUbSize =
        (compact_mm < config.mm_elements) ? compact_mm : config.mm_elements;
    tiling.singleCoreTensorSize.bmm2ResUbSize =
        (compact_bmm < config.bmm_elements) ? compact_bmm : config.bmm_elements;
#else
    tiling.singleCoreTensorSize.mmResUbSize = config.mm_elements;
    tiling.singleCoreTensorSize.bmm2ResUbSize = config.bmm_elements;
#endif
    tiling.innerSplitParams.mBaseSize = kHeads;
    tiling.innerSplitParams.s2BaseSize = config.s2_base;
    return tiling;
}

__aicore__ inline void EndPhase(AscendC::TPipe& pipe)
{
    AscendC::PipeBarrier<PIPE_ALL>();
#ifndef K112702_NO_DDR_INTRADIE
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
#endif
    AscendC::SyncAll<false>();
    pipe.Reset();
}


__aicore__ inline void PrologPhase(
    AscendC::TPipe& pipe, const Config& config, const DeviceArgs& args,
    GM_ADDR query_norm, GM_ADDR query_norm_scale,
    GM_ADDR scratch_override = nullptr)
{
    GM_ADDR eff_scratch = (scratch_override != nullptr) ? scratch_override : args.scratch;
    GM_ADDR query = eff_scratch + config.query_offset;
    GM_ADDR rope = eff_scratch + config.rope_offset;
    GM_ADDR workspace = eff_scratch + config.work_offset;
    auto tiling = PrologTiling(config);
    tiling.queryNormFlag = query_norm != nullptr ? 1U : 0U;
#ifdef K112706_PROLOG_DEQ_OPT
    using PrologType = MlaProlog::MLAPType<int8_t, int8_t, bfloat16_t, float,
        MlaProlog::CACHE_MODE::PA_BSND, true, false,
        MlaProlog::EMPTY_TENSOR_MODE::NON_EMPTY,
        MlaProlog::ACTUAL_SEQ_MODE::DISABLED, false, 2, true>;
#elif defined(K112708_PROLOG_GRP_OPT)
    using PrologType = MlaProlog::MLAPType<int8_t, int8_t, bfloat16_t, float,
        MlaProlog::CACHE_MODE::PA_BSND, false, true,
        MlaProlog::EMPTY_TENSOR_MODE::NON_EMPTY,
        MlaProlog::ACTUAL_SEQ_MODE::DISABLED, false, 2, true>;
#else
    using PrologType = MlaProlog::MLAPType<int8_t, int8_t, bfloat16_t, float,
        MlaProlog::CACHE_MODE::PA_BSND, false, false,
        MlaProlog::EMPTY_TENSOR_MODE::NON_EMPTY,
        MlaProlog::ACTUAL_SEQ_MODE::DISABLED, false, 2, true>;
#endif
    MlaProlog::MlaPrologVecS1CubS2<PrologType> prolog(&pipe, nullptr, &tiling);
    prolog.Init(args.token_x, args.weight_dq, args.weight_uq_qr,
        args.weight_uk, args.weight_dkv_kr, args.gamma_cq, args.gamma_ckv,
        args.sin, args.cos, args.slots, args.kv_cache, args.rope_cache,
        args.token_scale, args.scale_dq, args.scale_uq_qr, args.scale_dkv_kr,
        nullptr, nullptr, nullptr, nullptr, nullptr,
        query, rope, nullptr, query_norm, query_norm_scale, workspace);
    prolog.Process();
#ifdef K112512_PROLOG_48AIV_V2
    AscendC::PipeBarrier<PIPE_ALL>();
#ifndef K112702_NO_DDR_INTRADIE
    if ASCEND_IS_AIV {
        if (static_cast<uint32_t>(AscendC::GetBlockIdx()) < config.rows) {
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        }
    } else {
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
#endif
    AscendC::SyncAll<false>();
    pipe.Reset();
#else
    EndPhase(pipe);
#endif
}

#if defined(K112130_VEC_RESOLVED_TOPK) && defined(K11090_FAST_SFA_VEC)
__aicore__ __attribute__((noinline)) bool ResolveTopk2048(
    AscendC::TPipe& pipe, GM_ADDR raw_topk_gm, GM_ADDR resolved_topk_gm,
    GM_ADDR page_table_gm, GM_ADDR kv_lengths_gm,
    uint32_t sfa_rows, uint32_t table_pages, uint64_t tree_mask_gm,
    int32_t known_kv_len = -1)
{
    if (raw_topk_gm == nullptr || resolved_topk_gm == nullptr ||
        sfa_rows < 8U || sfa_rows > 32U || table_pages == 0U) {
        return false;
    }
    int32_t kv_len = known_kv_len;
    if (kv_len <= 0) {
        AscendC::GlobalTensor<int32_t> kvl_gm;
        kvl_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(kv_lengths_gm), 8);
        kv_len = kvl_gm.GetValue(0);
    }
    if (kv_len <= static_cast<int32_t>(sfa_rows) || kv_len > 65536) {
        return false;
    }
    const uint32_t active_blocks = (static_cast<uint32_t>(kv_len) + 127U) >> 7U;
    if (active_blocks == 0U || active_blocks > 512U || active_blocks > table_pages) {
        return false;
    }
    if ASCEND_IS_AIV {
        pipe.Reset();
        AscendC::TBuf<AscendC::TPosition::VECCALC> ub_buf;
#ifdef K112140_FAST_RTK
        const uint32_t row_shift = (sfa_rows >= 25U) ? 0U : ((sfa_rows >= 13U) ? 1U : 2U);
        const uint32_t chunk_elems = 2048U >> row_shift;
        const uint32_t total_chunks = sfa_rows << row_shift;
        pipe.InitBuffer(ub_buf, 61440);
        auto ub_i32 = ub_buf.Get<int32_t>();
        auto ub_u32 = ub_buf.Get<uint32_t>();
        auto raw_ub = ub_i32[0];
        auto out_ub = ub_i32[2048];
        auto out_u32 = ub_u32[2048];
        auto r_c = ub_i32[4096];
        auto r_c_u32 = ub_u32[4096];
        auto sign_i32 = ub_i32[6144];
        auto sign_u32 = ub_u32[6144];
        auto tmp_a = ub_i32[8192];
        auto tmp_a_u32 = ub_u32[8192];
        auto tmp_b = ub_i32[10240];
        auto tmp_b_u32 = ub_u32[10240];
        auto idx_u32 = ub_u32[12288];
        auto blk_ub = ub_i32[14336];
        auto mask_ub = ub_u32[14848];
        auto row_lut_ub = ub_i32[14912];
#else
        const uint32_t row_shift = 2U;
        const uint32_t chunk_elems = 512U;
        const uint32_t total_chunks = sfa_rows * 4U;
        pipe.InitBuffer(ub_buf, 18432);
        auto ub_i32 = ub_buf.Get<int32_t>();
        auto ub_u32 = ub_buf.Get<uint32_t>();
        auto raw_ub = ub_i32[0];
        auto out_ub = ub_i32[512];
        auto out_u32 = ub_u32[512];
        auto r_c = ub_i32[1024];
        auto r_c_u32 = ub_u32[1024];
        auto sign_i32 = ub_i32[1536];
        auto sign_u32 = ub_u32[1536];
        auto tmp_a = ub_i32[2048];
        auto tmp_a_u32 = ub_u32[2048];
        auto tmp_b = ub_i32[2560];
        auto tmp_b_u32 = ub_u32[2560];
        auto idx_u32 = ub_u32[3072];
        auto blk_ub = ub_i32[3584];
        auto mask_ub = ub_u32[4096];
        auto row_lut_ub = ub_i32[4160];
#endif

        const uint32_t vid = static_cast<uint32_t>(AscendC::GetBlockIdx());
#ifdef K112140_FAST_RTK
        if (vid < total_chunks)
#endif
        {
            AscendC::GlobalTensor<int32_t> pt_gm, src_gm, dst_gm;
            const uint32_t blk_align = (active_blocks + 7U) & ~7U;
            pt_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(page_table_gm), table_pages);
            src_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(raw_topk_gm), sfa_rows * 2048U);
            dst_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(resolved_topk_gm), sfa_rows * 2048U);

            AscendC::DataCopy(blk_ub, pt_gm, blk_align);
            if (tree_mask_gm != 0ULL) {
                AscendC::GlobalTensor<uint32_t> m_gm;
                m_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(tree_mask_gm), 32);
                AscendC::DataCopy(mask_ub, m_gm, 32);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

            const int32_t prefix_len = kv_len - static_cast<int32_t>(sfa_rows);
            const bool mask_enabled = (tree_mask_gm != 0ULL) && (mask_ub.GetValue(0) != 0U);
            uint32_t cur_lut_qrow = 0xFFFFFFFFU;
            bool lut_used = false;

            for (uint32_t t = vid; t < total_chunks; t += 48U) {
                const uint32_t q_row = t >> row_shift;
                const uint32_t off = t * chunk_elems;
                AscendC::DataCopy(raw_ub, src_gm[off], chunk_elems);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                const bool has_row_mask = mask_enabled && (q_row < 32U);
                const uint32_t row_mask = has_row_mask ? mask_ub.GetValue(q_row) : 0xFFFFFFFFU;
                const uint32_t full_q_bits = (q_row >= 31U) ? 0xFFFFFFFFU : ((1U << (q_row + 1U)) - 1U);
                const bool need_lut = has_row_mask && ((row_mask & full_q_bits) != full_q_bits);
                const int32_t s2_limit = prefix_len + static_cast<int32_t>(q_row) + 1;

                AscendC::Maxs(r_c, raw_ub, -1, chunk_elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mins(r_c, r_c, 16777216, chunk_elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ShiftRight(sign_u32, r_c_u32, 31U, chunk_elems);
                AscendC::PipeBarrier<PIPE_V>();

                if (!need_lut) {
                    AscendC::Adds(tmp_a, r_c, -s2_limit, chunk_elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::ShiftRight(tmp_a_u32, tmp_a_u32, 31U, chunk_elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Sub(tmp_a, tmp_a, sign_i32, chunk_elems);
                    AscendC::PipeBarrier<PIPE_V>();
                } else {
                    if (q_row != cur_lut_qrow) {
                        if (lut_used) {
                            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                        }
                        row_lut_ub.SetValue(0, 1);
                        for (uint32_t d = 0U; d < 32U; ++d) {
                            const int32_t bit_ok = (d <= q_row && ((row_mask >> d) & 1U) != 0U) ? 1 : 0;
                            row_lut_ub.SetValue(1U + d, bit_ok);
                        }
                        for (uint32_t d = 33U; d < 64U; ++d) {
                            row_lut_ub.SetValue(d, 0);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
                        cur_lut_qrow = q_row;
                        lut_used = true;
                    }
                    AscendC::Adds(tmp_b, r_c, 1 - prefix_len, chunk_elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Maxs(tmp_b, tmp_b, 0, chunk_elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mins(tmp_b, tmp_b, 33, chunk_elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::ShiftLeft(idx_u32, tmp_b_u32, 2U, chunk_elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Gather(tmp_a, row_lut_ub, idx_u32, 0U, chunk_elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Sub(tmp_a, tmp_a, sign_i32, chunk_elems);
                    AscendC::PipeBarrier<PIPE_V>();
                }

                AscendC::Maxs(r_c, r_c, 0, chunk_elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mins(r_c, r_c, kv_len - 1, chunk_elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ShiftRight(tmp_b_u32, r_c_u32, 7U, chunk_elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ShiftLeft(idx_u32, tmp_b_u32, 2U, chunk_elems);
                AscendC::ShiftLeft(tmp_b_u32, tmp_b_u32, 7U, chunk_elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Gather(out_ub, blk_ub, idx_u32, 0U, chunk_elems);
                AscendC::Sub(r_c, r_c, tmp_b, chunk_elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ShiftLeft(out_u32, out_u32, 7U, chunk_elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(out_ub, out_ub, r_c, chunk_elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(out_ub, out_ub, 2, chunk_elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(out_ub, tmp_a, out_ub, chunk_elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(out_ub, out_ub, sign_i32, chunk_elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(out_ub, out_ub, -2, chunk_elems);

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopy(dst_gm[off], out_ub, chunk_elems);
                if (t + 48U < total_chunks) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
#ifdef K112512_PROLOG_48AIV_V2
        if (vid < total_chunks) {
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        }
#else
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
#endif
        AscendC::SyncAll<true>();
        pipe.Reset();
    }
    return true;
}
#endif

__aicore__ inline void SfaPhase(AscendC::TPipe& pipe, const Config& config,
                               const DeviceArgs& args, GM_ADDR key = nullptr,
                               uint64_t tree_mask_gm = 0, bool is_shared_only = false,
                               GM_ADDR pre_resolved_topk = nullptr,
                               GM_ADDR scratch_override = nullptr,
                               GM_ADDR output_override = nullptr)
{
    GM_ADDR eff_scratch = (scratch_override != nullptr) ? scratch_override : args.scratch;
    GM_ADDR eff_output = (output_override != nullptr) ? output_override : args.output;
    GM_ADDR query = eff_scratch + config.query_offset;
    GM_ADDR rope = eff_scratch + config.rope_offset;
    GM_ADDR workspace = eff_scratch + config.work_offset;
#ifndef K112120_CLEAN_DCCI
    if (tree_mask_gm != 0) {
        AscendC::GlobalTensor<uint32_t> mask_gm;
        mask_gm.SetGlobalBuffer((__gm__ uint32_t*)tree_mask_gm, 32);
        AscendC::DataCacheCleanAndInvalid<uint32_t,
            AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(mask_gm[0]);
        AscendC::DataCacheCleanAndInvalid<uint32_t,
            AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(mask_gm[16]);
    }
#endif
    {
        auto tiling = SfaTiling(config);
        GM_ADDR q_lens = args.query_lengths;
#ifdef K11085_L78_CAUSAL
        if (is_shared_only && tree_mask_gm == 0) {
            tiling.baseParams.sparseMode = 0;
#ifdef K11089_L78_SFA_TOPK
            tiling.baseParams.sparseBlockCount = K11089_L78_SFA_TOPK;
            if (tiling.innerSplitParams.s2BaseSize > K11089_L78_SFA_TOPK) {
                tiling.innerSplitParams.s2BaseSize = K11089_L78_SFA_TOPK;
                tiling.singleCoreTensorSize.mmResUbSize =
                    tiling.innerSplitParams.mBaseSize * ((K11089_L78_SFA_TOPK + 31U) & ~31U);
            }
#endif
        }
#endif
#ifdef K11050_L78_SHARED_SFA
        if (tree_mask_gm == 0 && (config.rows == 4U || config.rows == 6U || config.rows == 8U)) {
            const uint32_t g_sh = (K11050_L78_SHARED_SFA == 32 && (config.rows & 3U) == 0U) ? 32U : 16U;
            const uint32_t s2_eff = tiling.innerSplitParams.s2BaseSize;
            tiling.baseParams.qSeqSize = (config.rows * kHeads) / g_sh;
            tiling.baseParams.nNumOfQInOneGroup = g_sh;
            tiling.baseParams.actualLenDimsQ = 0;
            tiling.innerSplitParams.mBaseSize = g_sh;
            tiling.singleCoreTensorSize.mmResUbSize = g_sh * ((s2_eff + 31U) & ~31U);
            tiling.singleCoreTensorSize.bmm2ResUbSize = g_sh * kLatent;
            q_lens = nullptr;
        }
#endif
        using AttentionType = SFAType<bfloat16_t, bfloat16_t, bfloat16_t,
            false, SFA_LAYOUT::TND, SFA_LAYOUT::PA_BSND, V_TEMPLATE, true>;
        const bool use_pre_resolved =
            (pre_resolved_topk != nullptr) &&
            (tiling.baseParams.sparseMode == 3) &&
            (tiling.innerSplitParams.mBaseSize == kHeads) &&
            (key == nullptr);
#ifdef K112840_UNION_SFA
        bool union_ok = false;
        if (use_pre_resolved) {
            union_ok = union_sfa::RunUnionSfa(
                pipe, config, query, rope,
                (key == nullptr) ? args.kv_cache : key, args.kv_cache, args.rope_cache,
                pre_resolved_topk, args.kv_lengths, workspace, eff_output,
                config.sfa_bytes, is_shared_only, tree_mask_gm);
        }
        if (!union_ok) {
#endif
        GM_ADDR eff_topk = use_pre_resolved ? pre_resolved_topk : args.topk;
        const uint64_t eff_mask_gm = use_pre_resolved ? (tree_mask_gm | 1ULL) : tree_mask_gm;
        SparseFlashAttentionMla<AttentionType> attention;
        attention.Init(query, key == nullptr ? args.kv_cache : key,
            args.kv_cache, eff_topk,
            q_lens, args.kv_lengths, args.page_table,
            rope, args.rope_cache, eff_output, nullptr, nullptr, workspace,
            &tiling, nullptr, &pipe);
        attention.SetTreeMaskGm(eff_mask_gm);
        attention.Process();
#ifdef K112840_UNION_SFA
        }
#endif
    }
#ifdef K112176_PHASE_DSB_OPT
    if ASCEND_IS_AIV {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    } else {
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    AscendC::SyncAll<false>();
    pipe.Reset();
#else
    EndPhase(pipe);
#endif
}

__aicore__ inline void Forward(AscendC::TPipe& pipe, const Config& config,
                               const DeviceArgs& args)
{
    PrologPhase(pipe, config, args, nullptr, nullptr);
    SfaPhase(pipe, config, args);
}

}

#endif
