#ifndef TESTS_GLM52_WHOLE_LAYER_UNION_SFA_NATIVE_H
#define TESTS_GLM52_WHOLE_LAYER_UNION_SFA_NATIVE_H

#include "kernel_operator.h"
#include "attention_contract.h"

namespace whole_layer_attention::union_sfa {

using namespace AscendC;

constexpr uint32_t kMaxUniq = 24576U;
constexpr uint32_t kMaxPages = 4096U;
constexpr uint32_t kHeadDim = 512U;
constexpr uint32_t kRopeDim = 64U;
constexpr uint32_t kHeadsLocal = 8U;
constexpr uint32_t kTopKSlots = 2048U;
constexpr uint32_t kMm2WaveCap = 24U;
constexpr float kLn2 = 0.6931471805599453094172F;
constexpr float kRecipLn2 = 1.0F / kLn2;
constexpr float kFloatEScalar = 8388608.0F;
constexpr float kSoftmaxMin = -2e38F;
constexpr int32_t kAmlaInitInt32 = 394264576;
constexpr IsResetLoad3dConfig kLoad3dCfg = {true, true};
constexpr SoftmaxConfig kSmCfg = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

__aicore__ inline uint32_t AlignUp(uint32_t v, uint32_t a)
{
    return (v + a - 1U) / a * a;
}

__aicore__ inline bool RunUnionSfa(
    TPipe& pipe, const Config& config, GM_ADDR query_gm, GM_ADDR rope_gm,
    GM_ADDR k_gm, GM_ADDR v_gm, GM_ADDR k_rope_gm, GM_ADDR pre_resolved_topk,
    GM_ADDR kv_lengths_gm, GM_ADDR workspace, GM_ADDR output_gm,
    uint64_t sfa_bytes, bool is_shared_only, uint64_t tree_mask_gm)
{
    (void)is_shared_only;
    (void)tree_mask_gm;
    if (pre_resolved_topk == nullptr || config.blocks != 24U || config.s2_base != 512U ||
        config.rows < 4U || config.rows > 32U || config.pages == 0U || config.pages > kMaxPages) {
        return false;
    }

    int32_t kv_len = 0;
    {
        GlobalTensor<int32_t> kvl_gm;
        kvl_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(kv_lengths_gm), 8);
        kv_len = kvl_gm.GetValue(0);
    }
    if (kv_len < 2048) {
        return false;
    }

    const uint32_t rows = config.rows;
    const uint32_t max_slots = config.pages * 128U;
    const uint32_t max_local_pages = (config.pages + 47U) / 48U;

    const uint64_t off_hdr = 0ULL;
    const uint64_t off_tok_u = 4096ULL;
    const uint64_t off_valid_half = off_tok_u + AlignUp(rows * kTopKSlots * 2U, 512U);
    const uint64_t off_uniq_slots = off_valid_half + AlignUp(rows * 8U * 2U, 512U);
    const uint64_t off_slot_u = off_uniq_slots + AlignUp(kMaxUniq * 4U, 512U);
    const uint64_t off_dyn_base = off_slot_u + AlignUp(max_slots * 2U, 512U);

    auto* hdr_ptr = reinterpret_cast<__gm__ int32_t*>(workspace + off_hdr);
    auto* tok_u_ptr = reinterpret_cast<__gm__ uint16_t*>(workspace + off_tok_u);
    auto* valid_half_ptr = reinterpret_cast<__gm__ uint16_t*>(workspace + off_valid_half);
    auto* uniq_slots_ptr = reinterpret_cast<__gm__ int32_t*>(workspace + off_uniq_slots);
    auto* slot_u_ptr = reinterpret_cast<__gm__ uint16_t*>(workspace + off_slot_u);

    uint32_t total_u = 0U;
    uint32_t overflow_flag = 0U;

    if ASCEND_IS_AIV {
        pipe.Reset();
        const uint32_t vid = static_cast<uint32_t>(GetBlockIdx());
        const uint32_t bm_words = (max_local_pages * 128U + 31U) >> 5U;
        const uint32_t bm_alloc = AlignUp(bm_words * 4U, 256U);
        const uint32_t slot_loc_elems = max_local_pages * 128U;
        const uint32_t touched_words = (max_local_pages + 31U) >> 5U;
        constexpr uint32_t kMaxLocalBucket = 4096U;

        TBuf<TPosition::VECCALC> bm_buf, loc_u_buf, loc_slots_buf, topk_buf, hdr_buf, touched_buf;
        pipe.InitBuffer(bm_buf, bm_alloc);
        pipe.InitBuffer(loc_u_buf, AlignUp(slot_loc_elems * 2U, 256U));
        pipe.InitBuffer(loc_slots_buf, kMaxLocalBucket * 4U);
        pipe.InitBuffer(topk_buf, kTopKSlots * 4U);
        pipe.InitBuffer(hdr_buf, 48U * 16U * 4U);
        pipe.InitBuffer(touched_buf, AlignUp(touched_words * 4U, 64U));

        auto bm_ub = bm_buf.Get<uint32_t>();
        auto loc_u_ub = loc_u_buf.Get<uint16_t>();
        auto loc_slots_ub = loc_slots_buf.Get<int32_t>();
        auto topk_ub = topk_buf.Get<int32_t>();
        auto hdr_ub = hdr_buf.Get<int32_t>();
        auto touched_ub = touched_buf.Get<uint32_t>();

        Duplicate(bm_buf.Get<int32_t>(), 0, static_cast<int32_t>(bm_alloc >> 2U));
        Duplicate(touched_buf.Get<int32_t>(), 0, static_cast<int32_t>(AlignUp(touched_words * 4U, 64U) >> 2U));
        PipeBarrier<PIPE_V>();

        GlobalTensor<int32_t> topk_gm;
        topk_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pre_resolved_topk), rows * kTopKSlots);

        uint32_t local_cnt = 0U;
        uint32_t local_ovf = 0U;
        for (uint32_t t = 0U; t < rows; ++t) {
            DataCopy(topk_ub, topk_gm[t * kTopKSlots], kTopKSlots);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            for (uint32_t j = 0U; j < kTopKSlots; ++j) {
                const int32_t s = topk_ub.GetValue(j);
                if (s == -1) {
                    break;
                }
                if (s < 0) {
                    continue;
                }
                const uint32_t us = static_cast<uint32_t>(s);
                const uint32_t page = us >> 7U;
                if (page >= config.pages) {
                    continue;
                }
                if ((page % 48U) == vid) {
                    const uint32_t lp = page / 48U;
                    const uint32_t lidx = (lp << 7U) | (us & 127U);
                    const uint32_t widx = lidx >> 5U;
                    const uint32_t bit = 1U << (lidx & 31U);
                    const uint32_t wv = bm_ub.GetValue(widx);
                    if ((wv & bit) == 0U) {
                        bm_ub.SetValue(widx, wv | bit);
                        const uint32_t tw = lp >> 5U;
                        touched_ub.SetValue(tw, touched_ub.GetValue(tw) | (1U << (lp & 31U)));
                        if (local_cnt < kMaxLocalBucket) {
                            loc_u_ub.SetValue(lidx, static_cast<uint16_t>(local_cnt));
                            loc_slots_ub.SetValue(local_cnt, s);
                            local_cnt++;
                        } else {
                            local_ovf = 1U;
                        }
                    }
                }
            }
        }

        GlobalTensor<int32_t> hdr_gm;
        hdr_gm.SetGlobalBuffer(hdr_ptr, 48U * 16U);
        Duplicate(hdr_ub, 0, 16);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        hdr_ub.SetValue(0, static_cast<int32_t>(local_cnt));
        hdr_ub.SetValue(1, static_cast<int32_t>(local_ovf));
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(hdr_gm[vid * 16U], hdr_ub, 16);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
        SyncAll<true>();

        DataCopy(hdr_ub, hdr_gm, 48U * 16U);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

        uint32_t my_base = 0U;
        for (uint32_t c = 0U; c < 48U; ++c) {
            const uint32_t cnt_c = static_cast<uint32_t>(hdr_ub.GetValue(c * 16U));
            const uint32_t ovf_c = static_cast<uint32_t>(hdr_ub.GetValue(c * 16U + 1U));
            if (ovf_c != 0U) {
                overflow_flag = 1U;
            }
            if (c < vid) {
                my_base += cnt_c;
            }
            total_u += cnt_c;
        }
        if (total_u > 0U && total_u <= kMaxUniq && overflow_flag == 0U) {
            if (local_cnt > 0U) {
                GlobalTensor<int32_t> uniq_gm;
                uniq_gm.SetGlobalBuffer(uniq_slots_ptr + my_base, AlignUp(local_cnt, 8U));
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopy(uniq_gm, loc_slots_ub, AlignUp(local_cnt, 8U));
            }
            GlobalTensor<uint16_t> slot_u_gm;
            slot_u_gm.SetGlobalBuffer(slot_u_ptr, max_slots);
            for (uint32_t lp = 0U; lp < max_local_pages; ++lp) {
                if ((touched_ub.GetValue(lp >> 5U) & (1U << (lp & 31U))) == 0U) {
                    continue;
                }
                const uint32_t p = vid + lp * 48U;
                if (p >= config.pages) {
                    break;
                }
                const uint32_t base_l = lp << 7U;
                for (uint32_t o = 0U; o < 128U; ++o) {
                    const uint32_t lidx = base_l | o;
                    if ((bm_ub.GetValue(lidx >> 5U) & (1U << (lidx & 31U))) != 0U) {
                        loc_u_ub.SetValue(lidx, static_cast<uint16_t>(my_base + loc_u_ub.GetValue(lidx)));
                    }
                }
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopy(slot_u_gm[p * 128U], loc_u_ub[base_l], 128U);
            }
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::DDR>();
        }
        if (vid == 0U) {
            Duplicate(hdr_ub, 0, 16);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            hdr_ub.SetValue(0, (overflow_flag != 0U) ? 0 : static_cast<int32_t>(total_u));
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(hdr_gm[48U * 16U], hdr_ub, 16);
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::DDR>();
        }
        SyncAll<true>();
        pipe.Reset();
    } else {
        SyncAll<true>();
        {
            TBuf<TPosition::A1> dummy;
        }
        GlobalTensor<int32_t> hdr_gm;
        hdr_gm.SetGlobalBuffer(hdr_ptr, 49U * 16U);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(hdr_gm[48U * 16U]);
        total_u = static_cast<uint32_t>(hdr_gm.GetValue(48U * 16U));
        if (total_u == 0U || total_u > kMaxUniq) {
            SyncAll<false>();
            return false;
        }
    }

    if ASCEND_IS_AIV {
        if (total_u == 0U || total_u > kMaxUniq || overflow_flag != 0U) {
            SyncAll<false>();
            return false;
        }
    }

    const uint32_t u_pad = AlignUp(total_u, 128U);
    const uint32_t m_total = rows * kHeadsLocal;
    const uint32_t m_pad = AlignUp(m_total, 16U);

    const uint64_t off_u_kn = off_dyn_base;
    const uint64_t off_u_kr = off_u_kn + static_cast<uint64_t>(u_pad) * kHeadDim * 2ULL;
    const uint64_t off_su = off_u_kr + static_cast<uint64_t>(u_pad) * kRopeDim * 2ULL;
    const uint64_t off_p = off_su + static_cast<uint64_t>(m_pad) * u_pad * 4ULL;
    const uint64_t off_nup = off_p + static_cast<uint64_t>(rows) * 4ULL * 16ULL * 512ULL * 2ULL;
    const uint64_t off_sum = off_nup + AlignUp(rows * 4U * 8U * 4U, 512U);
    const uint64_t off_mm2c = off_sum + AlignUp(rows * 8U * 4U, 512U);
    const uint64_t off_vpool = off_mm2c + static_cast<uint64_t>(rows) * 4ULL * 16ULL * kHeadDim * 4ULL;
    const uint64_t vpool_bytes = static_cast<uint64_t>(kMm2WaveCap) * 512ULL * kHeadDim * 2ULL;
    const bool dual_vpool = (off_vpool + 2ULL * vpool_bytes <= sfa_bytes);
    const uint64_t total_ws_need = off_vpool + (dual_vpool ? 2ULL * vpool_bytes : vpool_bytes);
    if (total_ws_need > sfa_bytes) {
        SyncAll<false>();
        return false;
    }

    auto* u_kn_ptr = reinterpret_cast<__gm__ bfloat16_t*>(workspace + off_u_kn);
    auto* u_kr_ptr = reinterpret_cast<__gm__ bfloat16_t*>(workspace + off_u_kr);
    auto* su_ptr = reinterpret_cast<__gm__ float*>(workspace + off_su);
    auto* p_ptr = reinterpret_cast<__gm__ bfloat16_t*>(workspace + off_p);
    auto* nup_ptr = reinterpret_cast<__gm__ int32_t*>(workspace + off_nup);
    auto* sum_ptr = reinterpret_cast<__gm__ float*>(workspace + off_sum);
    auto* mm2c_ptr = reinterpret_cast<__gm__ float*>(workspace + off_mm2c);
    auto* vpool_ptr = reinterpret_cast<__gm__ bfloat16_t*>(workspace + off_vpool);

    if ASCEND_IS_AIV {
        const uint32_t vid = static_cast<uint32_t>(GetBlockIdx());
        const bool fit_slot_ub = (max_slots <= 65536U);
        TBuf<TPosition::VECCALC> slot_all_buf, topk_buf, tok_u_buf, vh_buf, uniq_buf, kv_ub_buf[2], kr_ub_buf[2];
        if (fit_slot_ub) {
            pipe.InitBuffer(slot_all_buf, AlignUp(max_slots * 2U, 256U));
        }
        pipe.InitBuffer(topk_buf, kTopKSlots * 4U);
        pipe.InitBuffer(tok_u_buf, kTopKSlots * 2U);
        pipe.InitBuffer(vh_buf, 16U * 2U);
        pipe.InitBuffer(uniq_buf, AlignUp(total_u * 4U, 256U));
        pipe.InitBuffer(kv_ub_buf[0], 16U * kHeadDim * 2U);
        pipe.InitBuffer(kv_ub_buf[1], 16U * kHeadDim * 2U);
        pipe.InitBuffer(kr_ub_buf[0], 16U * kRopeDim * 2U);
        pipe.InitBuffer(kr_ub_buf[1], 16U * kRopeDim * 2U);

        GlobalTensor<int32_t> uniq_gm;
        uniq_gm.SetGlobalBuffer(uniq_slots_ptr, u_pad);
        auto uniq_ub = uniq_buf.Get<int32_t>();
        DataCopy(uniq_ub, uniq_gm, AlignUp(total_u, 8U));

        if (vid < rows) {
            GlobalTensor<int32_t> topk_gm;
            GlobalTensor<uint16_t> tok_u_gm, vh_gm, slot_u_gm;
            topk_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pre_resolved_topk), rows * kTopKSlots);
            tok_u_gm.SetGlobalBuffer(tok_u_ptr, rows * kTopKSlots);
            vh_gm.SetGlobalBuffer(valid_half_ptr, rows * 8U);
            slot_u_gm.SetGlobalBuffer(slot_u_ptr, max_slots);

            auto topk_ub = topk_buf.Get<int32_t>();
            auto tok_u_ub = tok_u_buf.Get<uint16_t>();
            auto vh_ub = vh_buf.Get<uint16_t>();

            DataCopy(topk_ub, topk_gm[vid * kTopKSlots], kTopKSlots);
            LocalTensor<uint16_t> slot_all_ub;
            if (fit_slot_ub) {
                slot_all_ub = slot_all_buf.Get<uint16_t>();
                DataCopy(slot_all_ub, slot_u_gm, AlignUp(max_slots, 16U));
            }
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

            bool stopped = false;
            for (uint32_t h = 0U; h < 8U; ++h) {
                uint32_t out_c = 0U;
                const uint32_t base_j = h * 256U;
                if (!stopped) {
                    for (uint32_t k = 0U; k < 256U; ++k) {
                        const int32_t raw = topk_ub.GetValue(base_j + k);
                        if (raw == -1) {
                            stopped = true;
                            break;
                        }
                        if (raw >= 0) {
                            const uint32_t us = static_cast<uint32_t>(raw);
                            const uint16_t u_id = fit_slot_ub
                                ? slot_all_ub.GetValue(us)
                                : slot_u_gm.GetValue(us);
                            tok_u_ub.SetValue(base_j + out_c, u_id);
                            out_c++;
                        }
                    }
                }
                vh_ub.SetValue(h, static_cast<uint16_t>(out_c));
                for (uint32_t k = out_c; k < 256U; ++k) {
                    tok_u_ub.SetValue(base_j + k, static_cast<uint16_t>(0xFFFFU));
                }
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(tok_u_gm[vid * kTopKSlots], tok_u_ub, kTopKSlots);
            DataCopy(vh_gm[vid * 8U], vh_ub, 16U);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }

        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);

        GlobalTensor<bfloat16_t> k_src_gm, kr_src_gm, u_kn_gm, u_kr_gm;
        k_src_gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(k_gm), max_slots * kHeadDim);
        kr_src_gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(k_rope_gm), max_slots * kRopeDim);
        u_kn_gm.SetGlobalBuffer(u_kn_ptr, static_cast<uint64_t>(u_pad) * kHeadDim);
        u_kr_gm.SetGlobalBuffer(u_kr_ptr, static_cast<uint64_t>(u_pad) * kRopeDim);

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        uint32_t ping = 0U;
        for (uint32_t u0 = vid * 16U; u0 < u_pad; u0 += 48U * 16U) {
            const uint32_t cnt = (u0 + 16U <= total_u) ? 16U : ((u0 < total_u) ? (total_u - u0) : 0U);
            const uint32_t p = ping & 1U;
            const event_t ev = p ? EVENT_ID2 : EVENT_ID1;
            WaitFlag<HardEvent::MTE3_MTE2>(ev);
            auto kv_ub = kv_ub_buf[p].Get<bfloat16_t>();
            auto kr_ub = kr_ub_buf[p].Get<bfloat16_t>();
            if (cnt == 0U) {
                Duplicate(kv_ub_buf[p].Get<int16_t>(), static_cast<int16_t>(0), 16U * kHeadDim);
                Duplicate(kr_ub_buf[p].Get<int16_t>(), static_cast<int16_t>(0), 16U * kRopeDim);
                SetFlag<HardEvent::V_MTE3>(ev);
                WaitFlag<HardEvent::V_MTE3>(ev);
            } else {
                for (uint32_t j = 0U; j < cnt; ++j) {
                    const uint64_t slot = static_cast<uint64_t>(uniq_ub.GetValue(u0 + j));
                    DataCopy(kv_ub[j * kHeadDim], k_src_gm[slot * kHeadDim], kHeadDim);
                    DataCopy(kr_ub[j * kRopeDim], kr_src_gm[slot * kRopeDim], kRopeDim);
                }
                if (cnt < 16U) {
                    Duplicate(kv_ub_buf[p].Get<int16_t>()[cnt * kHeadDim], static_cast<int16_t>(0), (16U - cnt) * kHeadDim);
                    Duplicate(kr_ub_buf[p].Get<int16_t>()[cnt * kRopeDim], static_cast<int16_t>(0), (16U - cnt) * kRopeDim);
                    SetFlag<HardEvent::V_MTE3>(ev);
                    WaitFlag<HardEvent::V_MTE3>(ev);
                }
                SetFlag<HardEvent::MTE2_MTE3>(ev);
                WaitFlag<HardEvent::MTE2_MTE3>(ev);
            }
            const uint32_t rows_out = (u0 + 16U <= u_pad) ? 16U : (u_pad - u0);
            DataCopy(u_kn_gm[u0 * kHeadDim], kv_ub, rows_out * kHeadDim);
            DataCopy(u_kr_gm[u0 * kRopeDim], kr_ub, rows_out * kRopeDim);
            SetFlag<HardEvent::MTE3_MTE2>(ev);
            ping++;
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
        SyncAll<false>();
        pipe.Reset();
    } else {
        SyncAll<false>();
        pipe.Reset();
        const uint32_t cid = static_cast<uint32_t>(GetBlockIdx());
        const uint32_t m_tile = (m_pad >= 64U) ? 64U : m_pad;
        const uint32_t m_blocks = (m_pad + m_tile - 1U) / m_tile;
        const uint32_t u_tiles = u_pad / 128U;

        TBuf<TPosition::A1> a1_buf0, a1_buf1;
        TBuf<TPosition::B1> b1_buf0, b1_buf1;
        TBuf<TPosition::A2> l0a_buf0, l0a_buf1;
        TBuf<TPosition::B2> l0b_buf0, l0b_buf1;
        TBuf<TPosition::CO1> l0c_buf0, l0c_buf1;
        pipe.InitBuffer(a1_buf0, m_tile * 288U * 2U);
        pipe.InitBuffer(a1_buf1, m_tile * 288U * 2U);
        pipe.InitBuffer(b1_buf0, 128U * 288U * 2U);
        pipe.InitBuffer(b1_buf1, 128U * 288U * 2U);
        pipe.InitBuffer(l0a_buf0, m_tile * 96U * 2U);
        pipe.InitBuffer(l0a_buf1, m_tile * 96U * 2U);
        pipe.InitBuffer(l0b_buf0, 96U * 128U * 2U);
        pipe.InitBuffer(l0b_buf1, 96U * 128U * 2U);
        pipe.InitBuffer(l0c_buf0, m_tile * 128U * 4U);
        pipe.InitBuffer(l0c_buf1, m_tile * 128U * 4U);

        LocalTensor<bfloat16_t> a1_k[2] = {a1_buf0.Get<bfloat16_t>(), a1_buf1.Get<bfloat16_t>()};
        LocalTensor<bfloat16_t> b1[2] = {b1_buf0.Get<bfloat16_t>(), b1_buf1.Get<bfloat16_t>()};
        LocalTensor<bfloat16_t> l0a[2] = {l0a_buf0.Get<bfloat16_t>(), l0a_buf1.Get<bfloat16_t>()};
        LocalTensor<bfloat16_t> l0b[2] = {l0b_buf0.Get<bfloat16_t>(), l0b_buf1.Get<bfloat16_t>()};
        LocalTensor<float> l0c[2] = {l0c_buf0.Get<float>(), l0c_buf1.Get<float>()};

        GlobalTensor<bfloat16_t> q_gt, qr_gt, kn_gt, kr_gt;
        GlobalTensor<float> su_gt;
        q_gt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(query_gm), m_total * kHeadDim);
        qr_gt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(rope_gm), m_total * kRopeDim);
        kn_gt.SetGlobalBuffer(u_kn_ptr, static_cast<uint64_t>(u_pad) * kHeadDim);
        kr_gt.SetGlobalBuffer(u_kr_ptr, static_cast<uint64_t>(u_pad) * kRopeDim);
        su_gt.SetGlobalBuffer(su_ptr, static_cast<uint64_t>(m_pad) * u_pad);

        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::FIX_M>(EVENT_ID1);

        auto issue_b = [&](uint32_t bslot, uint32_t ub_idx, uint32_t kl1) __aicore__ {
            const event_t bev = bslot ? EVENT_ID1 : EVENT_ID0;
            WaitFlag<HardEvent::MTE1_MTE2>(bev);
            Nd2NzParams cp{};
            cp.ndNum = 1;
            cp.nValue = 128;
            cp.dstNzC0Stride = 128;
            cp.dstNzNStride = 1;
            const uint64_t uoff = static_cast<uint64_t>(ub_idx) * 128ULL;
            if (kl1 == 0U) {
                cp.dValue = 256;
                cp.srcDValue = kHeadDim;
                DataCopy(b1[bslot], kn_gt[uoff * kHeadDim], cp);
                cp.dValue = 32;
                cp.srcDValue = kRopeDim;
                DataCopy(b1[bslot][256U * 128U], kr_gt[uoff * kRopeDim], cp);
            } else {
                cp.dValue = 32;
                cp.srcDValue = kRopeDim;
                DataCopy(b1[bslot], kr_gt[uoff * kRopeDim + 32U], cp);
                cp.dValue = 256;
                cp.srcDValue = kHeadDim;
                DataCopy(b1[bslot][32U * 128U], kn_gt[uoff * kHeadDim + 256U], cp);
            }
            SetFlag<HardEvent::MTE2_MTE1>(bev);
        };

        uint32_t ab_i = 0U;
        uint32_t c_i = 0U;
        uint32_t b_i = 0U;

        for (uint32_t mb = 0U; mb < m_blocks; ++mb) {
            const uint32_t m_start = mb * m_tile;
            const uint32_t act_m = (m_start + m_tile <= m_total) ? m_tile : (m_total - m_start);
            if (cid < u_tiles) {
                Nd2NzParams cqa{};
                cqa.ndNum = 1;
                cqa.nValue = act_m;
                cqa.dValue = 256;
                cqa.srcDValue = kHeadDim;
                cqa.dstNzC0Stride = m_tile;
                cqa.dstNzNStride = 1;
                DataCopy(a1_k[0], q_gt[m_start * kHeadDim], cqa);
                cqa.dValue = 32;
                cqa.srcDValue = kRopeDim;
                DataCopy(a1_k[0][256U * m_tile], qr_gt[m_start * kRopeDim], cqa);
                DataCopy(a1_k[1], qr_gt[m_start * kRopeDim + 32U], cqa);
                cqa.dValue = 256;
                cqa.srcDValue = kHeadDim;
                DataCopy(a1_k[1][32U * m_tile], q_gt[m_start * kHeadDim + 256U], cqa);
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);

                uint32_t cur_bslot = b_i++ & 1U;
                issue_b(cur_bslot, cid, 0U);
                for (uint32_t ub_idx = cid; ub_idx < u_tiles; ub_idx += 24U) {
                    const uint32_t cs = c_i++ & 1U;
                    const event_t cev = cs ? EVENT_ID1 : EVENT_ID0;
                    WaitFlag<HardEvent::FIX_M>(cev);

                    for (uint32_t kl1 = 0U; kl1 < 2U; ++kl1) {
                        uint32_t next_bslot = 0U;
                        const bool has_next_b = (kl1 == 0U) || (ub_idx + 24U < u_tiles);
                        if (has_next_b) {
                            next_bslot = b_i++ & 1U;
                            issue_b(next_bslot, (kl1 == 0U) ? ub_idx : (ub_idx + 24U), kl1 ^ 1U);
                        }
                        const event_t cur_bev = cur_bslot ? EVENT_ID1 : EVENT_ID0;
                        WaitFlag<HardEvent::MTE2_MTE1>(cur_bev);

                        for (uint32_t kl0 = 0U; kl0 < 3U; ++kl0) {
                            const uint32_t abs_i = ab_i++ & 1U;
                            const event_t abev = abs_i ? EVENT_ID3 : EVENT_ID2;
                            WaitFlag<HardEvent::M_MTE1>(abev);

                            LoadData3DParamsV2<bfloat16_t> la{};
                            la.padList[0] = 0; la.padList[1] = 0; la.padList[2] = 0; la.padList[3] = 255;
                            la.l1H = m_tile / 16U;
                            la.l1W = 16;
                            la.mExtension = m_tile;
                            la.kExtension = 96;
                            la.strideW = 1; la.strideH = 1;
                            la.filterW = 1; la.filterSizeW = 0;
                            la.filterH = 1; la.filterSizeH = 0;
                            la.dilationFilterW = 1; la.dilationFilterH = 1;
                            LoadData<bfloat16_t, kLoad3dCfg>(l0a[abs_i], a1_k[kl1][m_tile * 96U * kl0], la);

                            LoadData2DParams lb{};
                            lb.repeatTimes = (128U / 16U) * (96U / 16U);
                            lb.srcStride = 1;
                            LoadData(l0b[abs_i], b1[cur_bslot][128U * 96U * kl0], lb);

                            SetFlag<HardEvent::MTE1_M>(abev);
                            WaitFlag<HardEvent::MTE1_M>(abev);

                            MmadParams mm{};
                            mm.m = m_tile;
                            mm.n = 128;
                            mm.k = 96;
                            mm.cmatrixInitVal = (kl1 == 0U && kl0 == 0U);
                            mm.cmatrixSource = false;
                            mm.unitFlag = (kl1 == 1U && kl0 == 2U) ? 0b11 : 0b10;
                            Mmad(l0c[cs], l0a[abs_i], l0b[abs_i], mm);
                            SetFlag<HardEvent::M_MTE1>(abev);
                        }
                        SetFlag<HardEvent::MTE1_MTE2>(cur_bev);
                        cur_bslot = next_bslot;
                    }

                    FixpipeParamsV220 fx{};
                    fx.srcStride = m_tile;
                    fx.nSize = 128;
                    fx.mSize = m_tile;
                    fx.dstStride = u_pad;
                    fx.ndNum = 1;
                    fx.unitFlag = 0b11;
                    Fixpipe(su_gt[static_cast<uint64_t>(m_start) * u_pad + static_cast<uint64_t>(ub_idx) * 128ULL], l0c[cs], fx);
                    SetFlag<HardEvent::FIX_M>(cev);
                }
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
            }
        }
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
        pipe.Reset();
    }

    SyncAll<false>();

    const uint32_t total_mm2_chunks = rows * 4U;

    if ASCEND_IS_AIV {
        const uint32_t vid = static_cast<uint32_t>(GetBlockIdx());
        const uint32_t soft_tasks = rows * 2U;

        const uint32_t su_load_cap = (u_pad <= 8192U) ? u_pad : 8192U;
        TBuf<TPosition::VECCALC> tok_u_buf, idx_u32_buf, vh_buf, su_buf, mm_buf, sm_tmp_buf;
        TBuf<TPosition::VECCALC> sm_sum_buf[2], sm_max_buf[2], sm_exp_buf[2], nval_buf[2], cof_buf[2], amla_sum_buf;
        TBuf<TPosition::VECCALC> def_max_buf, def_sum_buf, p_out_buf, nup_out_buf, sum_out_buf, ntmp3_buf;

        pipe.InitBuffer(tok_u_buf, kTopKSlots * 2U);
        pipe.InitBuffer(idx_u32_buf, 512U * 4U);
        pipe.InitBuffer(vh_buf, 16U * 2U);
        pipe.InitBuffer(su_buf, su_load_cap * 4U);
        pipe.InitBuffer(mm_buf, 4U * 512U * 4U);
        pipe.InitBuffer(sm_tmp_buf, 32768U);
        pipe.InitBuffer(sm_sum_buf[0], 256U);
        pipe.InitBuffer(sm_sum_buf[1], 256U);
        pipe.InitBuffer(sm_max_buf[0], 256U);
        pipe.InitBuffer(sm_max_buf[1], 256U);
        pipe.InitBuffer(sm_exp_buf[0], 256U);
        pipe.InitBuffer(sm_exp_buf[1], 256U);
        pipe.InitBuffer(nval_buf[0], 256U);
        pipe.InitBuffer(nval_buf[1], 256U);
        pipe.InitBuffer(cof_buf[0], 256U);
        pipe.InitBuffer(cof_buf[1], 256U);
        pipe.InitBuffer(amla_sum_buf, 256U);
        pipe.InitBuffer(def_max_buf, 256U);
        pipe.InitBuffer(def_sum_buf, 256U);
        pipe.InitBuffer(p_out_buf, 4U * 512U * 2U);
        pipe.InitBuffer(nup_out_buf, 16U * 4U);
        pipe.InitBuffer(sum_out_buf, 16U * 4U);
        pipe.InitBuffer(ntmp3_buf, 128U * 4U);

        auto tok_u_ub = tok_u_buf.Get<uint16_t>();
        auto idx_u32_ub = idx_u32_buf.Get<uint32_t>();
        auto idx_i32_ub = idx_u32_buf.Get<int32_t>();
        auto vh_ub = vh_buf.Get<uint16_t>();
        auto su_ub = su_buf.Get<float>();
        auto mm_ub = mm_buf.Get<float>();
        auto sm_tmp_ub = sm_tmp_buf.Get<uint8_t>();
        auto sm_tmp_f32 = sm_tmp_buf.Get<float>();
        auto amla_sum_ub = amla_sum_buf.Get<float>();
        auto def_max_ub = def_max_buf.Get<float>();
        auto def_sum_ub = def_sum_buf.Get<float>();
        auto p_out_ub = p_out_buf.Get<bfloat16_t>();
        auto nup_out_ub = nup_out_buf.Get<int32_t>();
        auto sum_out_ub = sum_out_buf.Get<float>();
        auto ntmp3_ub = ntmp3_buf.Get<float>();

        Duplicate(def_max_ub, kSoftmaxMin, 64);
        Duplicate(def_sum_ub, 0.0F, 64);
        PipeBarrier<PIPE_V>();

        GlobalTensor<uint16_t> tok_u_gm, vh_gm;
        GlobalTensor<float> su_gm, sum_gm;
        GlobalTensor<bfloat16_t> p_gm;
        GlobalTensor<int32_t> nup_gm;
        tok_u_gm.SetGlobalBuffer(tok_u_ptr, rows * kTopKSlots);
        vh_gm.SetGlobalBuffer(valid_half_ptr, rows * 8U);
        su_gm.SetGlobalBuffer(su_ptr, static_cast<uint64_t>(m_pad) * u_pad);
        p_gm.SetGlobalBuffer(p_ptr, static_cast<uint64_t>(rows) * 4ULL * 16ULL * 512ULL);
        nup_gm.SetGlobalBuffer(nup_ptr, rows * 4U * 8U);
        sum_gm.SetGlobalBuffer(sum_ptr, rows * 8U);

        auto mask_range = [&](LocalTensor<float>& ub, uint32_t s_id, uint32_t e_id) __aicore__ {
            if (s_id >= e_id) {
                return;
            }
            const uint32_t ceil_s = AlignUp(s_id, 8U);
            const uint32_t floor_e = (e_id / 8U) * 8U;
            const uint32_t head_end = (ceil_s >= e_id) ? e_id : ceil_s;
            if (s_id < head_end) {
                SetFlag<HardEvent::V_S>(EVENT_ID1);
                WaitFlag<HardEvent::V_S>(EVENT_ID1);
                for (uint32_t r = 0U; r < 4U; ++r) {
                    for (uint32_t c0 = s_id; c0 < head_end; ++c0) {
                        ub.SetValue(r * 512U + c0, kSoftmaxMin);
                    }
                }
                SetFlag<HardEvent::S_V>(EVENT_ID1);
                WaitFlag<HardEvent::S_V>(EVENT_ID1);
            }
            if (ceil_s < floor_e) {
                for (uint32_t r = 0U; r < 4U; ++r) {
                    Duplicate(ub[r * 512U + ceil_s], kSoftmaxMin, floor_e - ceil_s);
                }
            }
            const uint32_t tail_s = (ceil_s <= floor_e) ? floor_e : ceil_s;
            if (tail_s < e_id) {
                SetFlag<HardEvent::V_S>(EVENT_ID1);
                WaitFlag<HardEvent::V_S>(EVENT_ID1);
                for (uint32_t r = 0U; r < 4U; ++r) {
                    for (uint32_t c0 = tail_s; c0 < e_id; ++c0) {
                        ub.SetValue(r * 512U + c0, kSoftmaxMin);
                    }
                }
                SetFlag<HardEvent::S_V>(EVENT_ID1);
                WaitFlag<HardEvent::S_V>(EVENT_ID1);
            }
        };

        for (uint32_t task = vid; task < soft_tasks; task += 48U) {
            const uint32_t t = task >> 1U;
            const uint32_t h_grp = (task & 1U) * 4U;

            DataCopy(tok_u_ub, tok_u_gm[t * kTopKSlots], kTopKSlots);
            DataCopy(vh_ub, vh_gm[t * 8U], 16U);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

            for (uint32_t c = 0U; c < 4U; ++c) {
                for (uint32_t j = 0U; j < 512U; ++j) {
                    const uint16_t uv = tok_u_ub.GetValue(c * 512U + j);
                    idx_u32_ub.SetValue(j, (uv == static_cast<uint16_t>(0xFFFFU)) ? 0U : (static_cast<uint32_t>(uv) << 2U));
                }
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);

                for (uint32_t h = 0U; h < 4U; ++h) {
                    const uint32_t row_abs = t * kHeadsLocal + h_grp + h;
                    if (u_pad <= 8192U) {
                        DataCopy(su_ub, su_gm[static_cast<uint64_t>(row_abs) * u_pad], u_pad);
                        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                        Gather(mm_ub[h * 512U], su_ub, idx_u32_ub, 0U, 512U);
                        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    } else {
                        for (uint32_t j = 0U; j < 512U; ++j) {
                            const uint16_t uv = tok_u_ub.GetValue(c * 512U + j);
                            const float val = (uv == static_cast<uint16_t>(0xFFFFU)) ? 0.0F : su_gm.GetValue(static_cast<uint64_t>(row_abs) * u_pad + uv);
                            mm_ub.SetValue(h * 512U + j, val);
                        }
                    }
                }
                PipeBarrier<PIPE_V>();
                Muls(mm_ub, mm_ub, config.scale, 4 * 512);
                PipeBarrier<PIPE_V>();

                const uint32_t v0 = static_cast<uint32_t>(vh_ub.GetValue(c * 2U));
                const uint32_t v1 = static_cast<uint32_t>(vh_ub.GetValue(c * 2U + 1U));
                if (v0 < 256U) {
                    mask_range(mm_ub, v0, 256U);
                    PipeBarrier<PIPE_V>();
                }
                if (v1 < 256U) {
                    mask_range(mm_ub, 256U + v1, 512U);
                    PipeBarrier<PIPE_V>();
                }

                const uint32_t cur_pp = c & 1U;
                const uint32_t prev_pp = cur_pp ^ 1U;
                auto in_max = (c == 0U) ? def_max_ub : sm_max_buf[prev_pp].Get<float>();
                auto in_sum = (c == 0U) ? def_sum_ub : sm_sum_buf[prev_pp].Get<float>();
                auto out_max = sm_max_buf[cur_pp].Get<float>();
                auto out_sum = sm_sum_buf[cur_pp].Get<float>();
                auto out_exp = sm_exp_buf[cur_pp].Get<float>();

                SoftMaxShapeInfo src_shape{4U, 512U, 4U, 512U};
                SoftMaxTiling sm_tiling = SoftMaxFlashV2TilingFunc(
                    src_shape, sizeof(float), sizeof(float), sm_tmp_ub.GetSize(), true, false);
                SoftmaxFlashV2<float, true, true, false, false, kSmCfg>(
                    mm_ub, out_sum, out_max, mm_ub, out_exp, in_sum, in_max, sm_tmp_ub, sm_tiling, src_shape);
                PipeBarrier<PIPE_V>();

                auto n_tmp = sm_tmp_f32;
                auto n_upd = sm_tmp_f32[256];
                auto n_upd2 = sm_tmp_f32[512];
                auto n_kvt = sm_tmp_f32[768].template ReinterpretCast<bfloat16_t>();
                auto tmp_cof = sm_tmp_f32[1024];
                auto eps_ub = sm_tmp_f32[1280];

                auto nval_cur = nval_buf[cur_pp].Get<float>();
                auto nval_prev = nval_buf[prev_pp].Get<float>();
                auto cof_cur = cof_buf[cur_pp].Get<float>();
                auto cof_prev = cof_buf[prev_pp].Get<float>();

                Muls(n_tmp, out_max, -kRecipLn2, 4);
                PipeBarrier<PIPE_V>();
                Cast(n_tmp, n_tmp, RoundMode::CAST_ROUND, 4);
                PipeBarrier<PIPE_V>();
                if (c == 0U) {
                    Duplicate(n_upd, 0.0F, 4);
                } else {
                    Sub(n_upd, n_tmp, nval_prev, 4);
                }
                PipeBarrier<PIPE_V>();
                Adds(nval_cur, n_tmp, 0.0F, 4);
                PipeBarrier<PIPE_V>();

                Muls(n_upd2, out_max, kRecipLn2, 4);
                PipeBarrier<PIPE_V>();
                Add(n_tmp, n_upd2, n_tmp, 4);
                PipeBarrier<PIPE_V>();
                Muls(n_tmp, n_tmp, kLn2, 4);
                PipeBarrier<PIPE_V>();
                Exp(n_tmp, n_tmp, 4);
                PipeBarrier<PIPE_V>();
                Cast(n_kvt, n_tmp, RoundMode::CAST_ROUND, 4);
                PipeBarrier<PIPE_V>();
                Cast(n_upd2, n_kvt, RoundMode::CAST_NONE, 4);
                PipeBarrier<PIPE_V>();
                if (c == 3U) {
                    Mul(amla_sum_ub, out_sum, n_upd2, 4);
                }
                Brcb(ntmp3_ub, n_upd2, 1, {1, 8});
                PipeBarrier<PIPE_V>();
                {
                    BinaryRepeatParams rp{};
                    rp.dstBlkStride = 1; rp.src0BlkStride = 1; rp.src1BlkStride = 0;
                    rp.dstRepStride = 64; rp.src0RepStride = 64; rp.src1RepStride = 1;
                    for (uint32_t d = 0U; d < 8U; ++d) {
                        Mul(mm_ub[d * 64U], mm_ub[d * 64U], ntmp3_ub, 64, 4, rp);
                    }
                }
                PipeBarrier<PIPE_V>();
                Cast(p_out_ub, mm_ub, RoundMode::CAST_ROUND, 4 * 512);

                Div(tmp_cof, n_tmp, n_upd2, 4);
                if (c == 0U) {
                    Duplicate(cof_cur, 1.0F, 4);
                    PipeBarrier<PIPE_V>();
                    Div(eps_ub, cof_cur, tmp_cof, 4);
                } else {
                    PipeBarrier<PIPE_V>();
                    Div(eps_ub, cof_prev, tmp_cof, 4);
                }
                PipeBarrier<PIPE_V>();
                Adds(cof_cur, tmp_cof, 0.0F, 4);
                Adds(eps_ub, eps_ub, -1.0F, 4);
                PipeBarrier<PIPE_V>();
                Muls(eps_ub, eps_ub, 1.5F, 4);
                Maxs(n_upd, n_upd, -30.0F, 4);
                PipeBarrier<PIPE_V>();
                Adds(eps_ub, eps_ub, 0.000001F, 4);
                PipeBarrier<PIPE_V>();
                Add(n_upd, n_upd, eps_ub, 4);
                PipeBarrier<PIPE_V>();
                Muls(n_upd, n_upd, kFloatEScalar, 4);
                PipeBarrier<PIPE_V>();
                Cast(nup_out_ub, n_upd, RoundMode::CAST_ROUND, 4);

                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                const uint64_t p_off = static_cast<uint64_t>(t * 4U + c) * 16ULL * 512ULL + h_grp * 512ULL;
                DataCopy(p_gm[p_off], p_out_ub, 4U * 512U);
                for (uint32_t h = 0U; h < 4U; ++h) {
                    nup_gm.SetValue(static_cast<uint64_t>(t * 4U + c) * 8ULL + h_grp + h, nup_out_ub.GetValue(h));
                }
                if (c == 3U) {
                    SetFlag<HardEvent::V_S>(EVENT_ID1);
                    WaitFlag<HardEvent::V_S>(EVENT_ID1);
                    for (uint32_t h = 0U; h < 4U; ++h) {
                        sum_gm.SetValue(static_cast<uint64_t>(t) * 8ULL + h_grp + h, amla_sum_ub.GetValue(h));
                    }
                }
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
        SyncAll<false>();
        pipe.Reset();
    } else {
        SyncAll<false>();
        pipe.Reset();
    }

    for (uint32_t w0 = 0U; w0 < total_mm2_chunks; w0 += kMm2WaveCap) {
        const uint32_t wave_cnt = (w0 + kMm2WaveCap <= total_mm2_chunks) ? kMm2WaveCap : (total_mm2_chunks - w0);
        const uint64_t v_wave_off = dual_vpool ? (static_cast<uint64_t>((w0 / kMm2WaveCap) & 1U) * vpool_bytes) : 0ULL;
        auto* v_wave_ptr = vpool_ptr + (v_wave_off / 2ULL);

        if ASCEND_IS_AIV {
            const uint32_t vid = static_cast<uint32_t>(GetBlockIdx());
            if (vid < wave_cnt * 2U) {
                const uint32_t slot_c = vid >> 1U;
                const uint32_t half = vid & 1U;
                const uint32_t chunk = w0 + slot_c;
                const uint32_t t = chunk >> 2U;
                const uint32_t c = chunk & 3U;

                TBuf<TPosition::VECCALC> tu_buf, v_ub_buf[2];
                pipe.InitBuffer(tu_buf, 256U * 2U);
                pipe.InitBuffer(v_ub_buf[0], 64U * kHeadDim * 2U);
                pipe.InitBuffer(v_ub_buf[1], 64U * kHeadDim * 2U);

                auto tu_ub = tu_buf.Get<uint16_t>();
                GlobalTensor<uint16_t> tok_u_gm;
                GlobalTensor<bfloat16_t> u_kn_gm, vw_gm;
                tok_u_gm.SetGlobalBuffer(tok_u_ptr, rows * kTopKSlots);
                u_kn_gm.SetGlobalBuffer(u_kn_ptr, static_cast<uint64_t>(u_pad) * kHeadDim);
                vw_gm.SetGlobalBuffer(v_wave_ptr + static_cast<uint64_t>(slot_c) * 512ULL * kHeadDim, 512ULL * kHeadDim);

                DataCopy(tu_ub, tok_u_gm[static_cast<uint64_t>(t) * kTopKSlots + c * 512U + half * 256U], 256U);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                for (uint32_t b = 0U; b < 4U; ++b) {
                    const uint32_t bi = b & 1U;
                    const event_t bev = bi ? EVENT_ID1 : EVENT_ID0;
                    WaitFlag<HardEvent::MTE3_MTE2>(bev);
                    auto v_ub = v_ub_buf[bi].Get<bfloat16_t>();
                    auto v_i16 = v_ub_buf[bi].Get<int16_t>();
                    const uint32_t rbase = b * 64U;
                    for (uint32_t r = 0U; r < 64U; ++r) {
                        const uint16_t uv = tu_ub.GetValue(rbase + r);
                        if (uv == static_cast<uint16_t>(0xFFFFU)) {
                            Duplicate(v_i16[r * kHeadDim], static_cast<int16_t>(0), kHeadDim);
                        } else {
                            DataCopy(v_ub[r * kHeadDim], u_kn_gm[static_cast<uint64_t>(uv) * kHeadDim], kHeadDim);
                        }
                    }
                    SetFlag<HardEvent::MTE2_MTE3>(bev);
                    WaitFlag<HardEvent::MTE2_MTE3>(bev);
                    SetFlag<HardEvent::V_MTE3>(bev);
                    WaitFlag<HardEvent::V_MTE3>(bev);
                    DataCopy(vw_gm[static_cast<uint64_t>(half * 256U + rbase) * kHeadDim], v_ub, 64U * kHeadDim);
                    SetFlag<HardEvent::MTE3_MTE2>(bev);
                }
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            }
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::DDR>();
            SyncAll<false>();
            pipe.Reset();
        } else {
            SyncAll<false>();
            const uint32_t cid = static_cast<uint32_t>(GetBlockIdx());
            if (cid < wave_cnt) {
                const uint32_t chunk = w0 + cid;
                TBuf<TPosition::A1> a1_buf[2];
                TBuf<TPosition::B1> b1_buf[2];
                TBuf<TPosition::A2> l0a_buf[2];
                TBuf<TPosition::B2> l0b_buf[2];
                TBuf<TPosition::CO1> l0c_buf[2];

                pipe.InitBuffer(a1_buf[0], 16U * 256U * 2U);
                pipe.InitBuffer(a1_buf[1], 16U * 256U * 2U);
                pipe.InitBuffer(b1_buf[0], 256U * 128U * 2U);
                pipe.InitBuffer(b1_buf[1], 256U * 128U * 2U);
                pipe.InitBuffer(l0a_buf[0], 16U * 128U * 2U);
                pipe.InitBuffer(l0a_buf[1], 16U * 128U * 2U);
                pipe.InitBuffer(l0b_buf[0], 128U * 128U * 2U);
                pipe.InitBuffer(l0b_buf[1], 128U * 128U * 2U);
                pipe.InitBuffer(l0c_buf[0], 16U * 128U * 4U);
                pipe.InitBuffer(l0c_buf[1], 16U * 128U * 4U);

                LocalTensor<bfloat16_t> a1[2] = {a1_buf[0].Get<bfloat16_t>(), a1_buf[1].Get<bfloat16_t>()};
                LocalTensor<bfloat16_t> b1[2] = {b1_buf[0].Get<bfloat16_t>(), b1_buf[1].Get<bfloat16_t>()};
                LocalTensor<bfloat16_t> l0a[2] = {l0a_buf[0].Get<bfloat16_t>(), l0a_buf[1].Get<bfloat16_t>()};
                LocalTensor<bfloat16_t> l0b[2] = {l0b_buf[0].Get<bfloat16_t>(), l0b_buf[1].Get<bfloat16_t>()};
                LocalTensor<float> l0c[2] = {l0c_buf[0].Get<float>(), l0c_buf[1].Get<float>()};

                GlobalTensor<bfloat16_t> p_gt, v_gt;
                GlobalTensor<float> mm2_gt;
                p_gt.SetGlobalBuffer(p_ptr + static_cast<uint64_t>(chunk) * 16ULL * 512ULL, 16ULL * 512ULL);
                v_gt.SetGlobalBuffer(v_wave_ptr + static_cast<uint64_t>(cid) * 512ULL * kHeadDim, 512ULL * kHeadDim);
                mm2_gt.SetGlobalBuffer(mm2c_ptr + static_cast<uint64_t>(chunk) * 16ULL * kHeadDim, 16ULL * kHeadDim);

                for (uint32_t k1 = 0U; k1 < 2U; ++k1) {
                    Nd2NzParams pa{};
                    pa.ndNum = 1; pa.nValue = 8; pa.dValue = 256; pa.srcDValue = 512;
                    pa.dstNzC0Stride = 16; pa.dstNzNStride = 1;
                    DataCopy(a1[k1], p_gt[k1 * 256U], pa);
                }
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);

                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
                SetFlag<HardEvent::FIX_M>(EVENT_ID0);
                SetFlag<HardEvent::FIX_M>(EVENT_ID1);

                uint32_t b_i = 0U;
                uint32_t ab_i = 0U;
                uint32_t c_i = 0U;

                auto issue_v = [&](uint32_t bslot, uint32_t sn, uint32_t k1) __aicore__ {
                    const event_t bev = bslot ? EVENT_ID1 : EVENT_ID0;
                    WaitFlag<HardEvent::MTE1_MTE2>(bev);
                    GlobalTensor<uint32_t> source;
                    source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(v_wave_ptr + static_cast<uint64_t>(cid) * 512ULL * kHeadDim + k1 * 256ULL * kHeadDim + sn * 128U), 65344);
                    Nd2NzParams pair{2, 128, 64, 32768, 256, 128, 1, 8192};
                    DataCopy(b1[bslot].template ReinterpretCast<uint32_t>(), source, pair);
                    SetFlag<HardEvent::MTE2_MTE1>(bev);
                };

                uint32_t cur_bslot = b_i++ & 1U;
                issue_v(cur_bslot, 0U, 0U);
                for (uint32_t sn = 0U; sn < 4U; ++sn) {
                    const uint32_t cs = c_i++ & 1U;
                    const event_t cev = cs ? EVENT_ID1 : EVENT_ID0;
                    WaitFlag<HardEvent::FIX_M>(cev);

                    for (uint32_t k1 = 0U; k1 < 2U; ++k1) {
                        uint32_t next_bslot = 0U;
                        const bool has_next = (k1 == 0U) || (sn + 1U < 4U);
                        if (has_next) {
                            next_bslot = b_i++ & 1U;
                            issue_v(next_bslot, (k1 == 0U) ? sn : (sn + 1U), k1 ^ 1U);
                        }
                        const event_t cur_bev = cur_bslot ? EVENT_ID1 : EVENT_ID0;
                        WaitFlag<HardEvent::MTE2_MTE1>(cur_bev);

                        for (uint32_t kl0 = 0U; kl0 < 2U; ++kl0) {
                            const uint32_t abs_i = ab_i++ & 1U;
                            const event_t abev = abs_i ? EVENT_ID3 : EVENT_ID2;
                            WaitFlag<HardEvent::M_MTE1>(abev);

                            LoadData3DParamsV2<bfloat16_t> lb{};
                            lb.l1H = 128U / 16U; lb.l1W = 16;
                            lb.padList[0] = 0; lb.padList[1] = 0; lb.padList[2] = 0; lb.padList[3] = 255;
                            lb.mExtension = 128; lb.kExtension = 128;
                            lb.strideW = 1; lb.strideH = 1;
                            lb.filterW = 1; lb.filterSizeW = 0;
                            lb.filterH = 1; lb.filterSizeH = 0;
                            lb.dilationFilterW = 1; lb.dilationFilterH = 1;
                            lb.enTranspose = 1; lb.fMatrixCtrl = 0; lb.channelSize = 128;
                            LoadData<bfloat16_t, kLoad3dCfg>(l0b[abs_i], b1[cur_bslot][kl0 * 16384U], lb);

                            LoadData3DParamsV2<bfloat16_t> la{};
                            la.padList[0] = 0; la.padList[1] = 0; la.padList[2] = 0; la.padList[3] = 255;
                            la.l1H = 1; la.l1W = 16;
                            la.mExtension = 16; la.kExtension = 128;
                            la.strideW = 1; la.strideH = 1;
                            la.filterW = 1; la.filterSizeW = 0;
                            la.filterH = 1; la.filterSizeH = 0;
                            la.dilationFilterW = 1; la.dilationFilterH = 1;
                            la.enTranspose = 0; la.fMatrixCtrl = 0; la.channelSize = 128;
                            LoadData<bfloat16_t, kLoad3dCfg>(l0a[abs_i], a1[k1][kl0 * 128U * 16U], la);

                            SetFlag<HardEvent::MTE1_M>(abev);
                            WaitFlag<HardEvent::MTE1_M>(abev);

                            MmadParams mm{};
                            mm.m = 16; mm.n = 128; mm.k = 128;
                            mm.cmatrixInitVal = (k1 == 0U && kl0 == 0U);
                            mm.cmatrixSource = false;
                            mm.unitFlag = (k1 == 1U && kl0 == 1U) ? 0b11 : 0b10;
                            Mmad(l0c[cs], l0a[abs_i], l0b[abs_i], mm);
                            PipeBarrier<PIPE_M>();
                            SetFlag<HardEvent::M_MTE1>(abev);
                        }
                        SetFlag<HardEvent::MTE1_MTE2>(cur_bev);
                        cur_bslot = next_bslot;
                    }

                    FixpipeParamsV220 fx{};
                    fx.nSize = 128; fx.mSize = 16; fx.srcStride = 16; fx.dstStride = kHeadDim;
                    fx.ndNum = 1; fx.unitFlag = 0b11;
                    Fixpipe(mm2_gt[sn * 128U], l0c[cs], fx);
                    SetFlag<HardEvent::FIX_M>(cev);
                }

                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
                WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
                WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
                WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
                WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
            }
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::DDR>();
            pipe.Reset();
        }
    }

    if ASCEND_IS_AIV {
        const uint32_t vid = static_cast<uint32_t>(GetBlockIdx());
        if (vid < rows) {
            TBuf<TPosition::VECCALC> acc_buf, chk_buf, abs_buf, mask_buf, brc_buf, out_buf;
            pipe.InitBuffer(acc_buf, 8U * kHeadDim * 4U);
            pipe.InitBuffer(chk_buf, 8U * kHeadDim * 4U);
            pipe.InitBuffer(abs_buf, 8U * kHeadDim * 4U);
            pipe.InitBuffer(mask_buf, 8U * kHeadDim);
            pipe.InitBuffer(brc_buf, 64U * 4U);
            pipe.InitBuffer(out_buf, 8U * kHeadDim * 2U);

            auto acc_f = acc_buf.Get<float>();
            auto acc_i = acc_buf.Get<int32_t>();
            auto chk_f = chk_buf.Get<float>();
            auto abs_f = abs_buf.Get<float>();
            auto cmp_m = mask_buf.Get<uint8_t>();
            auto brc_f = brc_buf.Get<float>();
            auto out_bf = out_buf.Get<bfloat16_t>();

            GlobalTensor<float> mm2_gm, sum_gm;
            GlobalTensor<int32_t> nup_gm;
            GlobalTensor<bfloat16_t> out_gm;
            mm2_gm.SetGlobalBuffer(mm2c_ptr, static_cast<uint64_t>(rows) * 4ULL * 16ULL * kHeadDim);
            nup_gm.SetGlobalBuffer(nup_ptr, rows * 4U * 8U);
            sum_gm.SetGlobalBuffer(sum_ptr, rows * 8U);
            out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output_gm), m_total * kHeadDim);

            Duplicate(acc_i, kAmlaInitInt32, 8 * kHeadDim);
            PipeBarrier<PIPE_V>();

            for (uint32_t c = 0U; c < 4U; ++c) {
                const uint32_t chunk = vid * 4U + c;
                DataCopy(chk_f, mm2_gm[static_cast<uint64_t>(chunk) * 16ULL * kHeadDim], 8U * kHeadDim);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

                if (c > 0U) {
                    for (uint32_t h = 0U; h < 8U; ++h) {
                        const int32_t nu = nup_gm.GetValue(static_cast<uint64_t>(chunk) * 8ULL + h);
                        if (nu != 0) {
                            Adds(acc_i[h * kHeadDim], acc_i[h * kHeadDim], nu, kHeadDim);
                        }
                    }
                    PipeBarrier<PIPE_V>();
                }
                Add(acc_f, acc_f, chk_f, 8 * kHeadDim);
                PipeBarrier<PIPE_V>();
            }

            Abs(abs_f, acc_f, 8 * kHeadDim);
            PipeBarrier<PIPE_V>();
            CompareScalar(cmp_m, abs_f, 1e10F, CMPMODE::LE, 8 * kHeadDim);
            PipeBarrier<PIPE_V>();
            Select(acc_f, cmp_m, acc_f, 0.0F, SELMODE::VSEL_TENSOR_SCALAR_MODE, 8 * kHeadDim);
            PipeBarrier<PIPE_V>();

            for (uint32_t h = 0U; h < 8U; ++h) {
                abs_f.SetValue(h, sum_gm.GetValue(static_cast<uint64_t>(vid) * 8ULL + h));
            }
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Brcb(brc_f, abs_f, 1, {1, 8});
            PipeBarrier<PIPE_V>();
            {
                BinaryRepeatParams dp{};
                dp.dstBlkStride = 1; dp.src0BlkStride = 1; dp.src1BlkStride = 0;
                dp.dstRepStride = 64; dp.src0RepStride = 64; dp.src1RepStride = 1;
                for (uint32_t d = 0U; d < 8U; ++d) {
                    Div(acc_f[d * 64U], acc_f[d * 64U], brc_f, 64, 8, dp);
                }
            }
            PipeBarrier<PIPE_V>();
            Cast(out_bf, acc_f, RoundMode::CAST_RINT, 8 * kHeadDim);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(out_gm[static_cast<uint64_t>(vid) * 8ULL * kHeadDim], out_bf, 8U * kHeadDim);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
    }
    SyncAll<false>();
    pipe.Reset();
    return true;
}

}

#endif
