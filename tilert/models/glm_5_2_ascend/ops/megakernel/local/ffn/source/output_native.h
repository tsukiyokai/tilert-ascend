#ifndef TILERT_FFN_ORIGINAL_SHARED_GU_STATIC_OUTPUT_NATIVE_H
#define TILERT_FFN_ORIGINAL_SHARED_GU_STATIC_OUTPUT_NATIVE_H


__aicore__ inline void TokenOutputs(
    TPipe& pipe, Workspace& held, __gm__ RoutedProduct* product,
    __gm__ float* correction, __gm__ ScaleStorage* scales,
    __gm__ float* middle_scales, __gm__ int32_t* original_experts,
    __gm__ bfloat16_t* route_output, __gm__ bfloat16_t* routed_output,
    __gm__ bfloat16_t* shared_output, __gm__ bfloat16_t* scaled_output,
    __gm__ bfloat16_t* combined_output, int64_t tokens, int64_t blocks, int32_t expert_offset = 0, bool combine_shared = true, uint32_t saved_local_mask = 0xFFFFFFFFU, __gm__ uint8_t* comm_window = nullptr, int64_t comm_capacity = 0, int32_t comm_rank = 0, int32_t comm_ranks = 0, uint32_t comm_base_epoch = 0, int32_t exp_begin = 0, int32_t exp_end = kLocalExperts, const uint8_t* my_gexp = nullptr, const uint64_t* mw_s = nullptr, const uint32_t* pre_local_origs = nullptr, int32_t pre_local_cnt = -1)
{
    using namespace ffn_original_shared_gu_static;
    static_assert(kHidden == kOutputHidden && kTopK == kOutputTopK);
    static_assert(kVectorTile >= kOutputTile);
    static_assert(kOutputMetadataBytes <= (kVectorTile + kSlack) * sizeof(int32_t));

    const int64_t routes = tokens * kTopK;
    const uint32_t local_mask = saved_local_mask;
    auto expert_ids = held.slot;
    auto inverse = held.weight.ReinterpretCast<int32_t>();
    auto activation = held.maximum;
#ifdef K112600_PIPE_C2V
    if (!combine_shared && pre_local_origs != nullptr && pre_local_cnt == 0 && exp_begin > 0) {
        return;
    }
    if (local_mask != 0U && !(pre_local_origs != nullptr && exp_begin > 0 && !combine_shared)) {
#else
    if (local_mask != 0U) {
#endif
        GlobalTensor<float> activation_gm;
        activation_gm.SetGlobalBuffer(middle_scales, routes);
        Fence<HardEvent::V_MTE2>();
        DataCopy(activation, activation_gm, routes);
        Fence<HardEvent::MTE2_S>();
        DataSyncBarrier<MemDsbT::UB>();
    }

    GlobalTensor<bfloat16_t> shared, combined, route_out_gm;
    GlobalTensor<float> combined_fp32;
    combined_fp32.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(route_output), tokens * kHidden);
    route_out_gm.SetGlobalBuffer(route_output, routes * kHidden);
    shared.SetGlobalBuffer(shared_output, tokens * kHidden);
    combined.SetGlobalBuffer(combined_output, tokens * kHidden);
    auto narrow = held.half_buf.Get<bfloat16_t>();
    constexpr int64_t span = kOutputTile;
    if (!combine_shared) {
#ifdef TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H
        __gm__ bfloat16_t* rt_plane = nullptr;
        bool need_r1_wait = false;
        if (comm_window != nullptr) {
            const bool quad4 = whole_layer_tree::IsQuad4Active(comm_capacity, tokens);
#ifdef K112110_SIP_RED1
#ifdef K112140_SIP_RED1_MIN_ROWS
            need_r1_wait = (!quad4 && tokens >= static_cast<int64_t>(K112140_SIP_RED1_MIN_ROWS) && exp_begin == 0);
#else
            need_r1_wait = (!quad4 && exp_begin == 0);
#endif
#endif
            const auto layout = quad4
                ? whole_layer_tree::MakeQuad4Layout(comm_capacity, comm_ranks, tokens)
                : whole_layer_tree::MakeCommunicationLayout(comm_capacity, comm_ranks);
            const uint32_t p_mask = quad4 ? 3U : 1U;
            const uint32_t rt_epoch = comm_base_epoch + 2U;
            rt_plane = reinterpret_cast<__gm__ bfloat16_t*>(
                comm_window + layout.payload + (rt_epoch & p_mask) * layout.plane_bytes);
            if (GetBlockIdx() == 0 && exp_begin == 0) {
                GlobalTensor<uint32_t> counter;
                counter.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(comm_window), 1);
                counter.SetValue(0, rt_epoch);
                DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(counter);
            }
        }
#endif
        if (local_mask == 0U) {
            return;
        }
        uint32_t local_origs_buf[256];
        const uint32_t* local_origs = local_origs_buf;
        int32_t local_cnt = 0;
#ifdef K11220_FAST_TOK_OUT
        if (pre_local_origs != nullptr && pre_local_cnt >= 0) {
            local_origs = pre_local_origs;
            local_cnt = pre_local_cnt;
        } else
#endif
        {
            (void)pre_local_origs;
            (void)pre_local_cnt;
            for (int32_t exp = exp_begin; exp < exp_end; ++exp) {
                const int32_t cnt = held.count.GetValue(exp);
                if (cnt == 0) continue;
                const int32_t st = held.start.GetValue(exp);
                for (int32_t r = 0; r < cnt; ++r) {
                    const int32_t orig = held.source.GetValue(kMaxRoutes + st + r);
                    if (held.original_weight.GetValue(orig) != 0.0F) {
                        local_origs_buf[local_cnt++] = static_cast<uint32_t>(orig)
                            | (static_cast<uint32_t>(exp) << 8)
                            | (static_cast<uint32_t>(st) << 14)
                            | (static_cast<uint32_t>(r) << 22)
                            | (static_cast<uint32_t>(cnt - 1) << 27);
                    }
                }
            }
        }
        if (local_cnt <= 0) {
            return;
        }
        int64_t last_expert = -1;
        int64_t span = kOutputTile;
        int32_t idx = static_cast<int32_t>(GetBlockIdx()) >> 2;
        int32_t idx_step = 12;
        int64_t column = (GetBlockIdx() & 3) * 1536;
        const uint32_t tid_u = static_cast<uint32_t>(GetBlockIdx());
        auto select_exact48_tok = [&]() __aicore__ {
            const uint32_t ulc = static_cast<uint32_t>(local_cnt);
            if (ulc <= 1U) {
                span = 128;
                idx = 0;
                idx_step = 1;
                column = static_cast<int64_t>(tid_u) * 128;
            } else {
                const uint32_t c2 = ((ulc + 1U) >> 1U) * 352U;
                const uint32_t c3 = (((ulc + 2U) * 683U) >> 11U) * 480U;
                const uint32_t c4 = ((ulc + 3U) >> 2U) * 608U;
                const uint32_t c6 = (((ulc + 5U) * 683U) >> 12U) * 864U;
                const uint32_t c8 = ((ulc + 7U) >> 3U) * 1120U;
                const uint32_t c12 = (((ulc + 11U) * 683U) >> 13U) * 1632U;
                uint32_t best_c = c4;
                uint32_t best_g = 4U;
                if (c6 < best_c) { best_c = c6; best_g = 6U; }
                if (c8 < best_c) { best_c = c8; best_g = 8U; }
                if (c3 < best_c) { best_c = c3; best_g = 3U; }
                if (c2 < best_c) { best_c = c2; best_g = 2U; }
                if (c12 < best_c) { best_c = c12; best_g = 12U; }
                idx_step = static_cast<int32_t>(best_g);
                if (best_g == 2U) {
                    span = 256;
                    idx = static_cast<int32_t>((tid_u * 683U) >> 14U);
                    column = (static_cast<int64_t>(tid_u) - idx * 24) * 256;
                } else if (best_g == 3U) {
                    span = 384;
                    idx = static_cast<int32_t>(tid_u >> 4U);
                    column = static_cast<int64_t>(tid_u & 15U) * 384;
                } else if (best_g == 4U) {
                    span = 512;
                    idx = static_cast<int32_t>((tid_u * 683U) >> 13U);
                    column = (static_cast<int64_t>(tid_u) - idx * 12) * 512;
                } else if (best_g == 6U) {
                    span = 768;
                    idx = static_cast<int32_t>(tid_u >> 3U);
                    column = static_cast<int64_t>(tid_u & 7U) * 768;
                } else if (best_g == 8U) {
                    span = 1024;
                    idx = static_cast<int32_t>((tid_u * 683U) >> 12U);
                    column = (static_cast<int64_t>(tid_u) - idx * 6) * 1024;
                } else {
                    span = 1536;
                    idx = static_cast<int32_t>(tid_u >> 2U);
                    column = static_cast<int64_t>(tid_u & 3U) * 1536;
                }
            }
        };
        (void)select_exact48_tok;
#if defined(K11220_RS768_PUSH) || defined(K112600_RS768_HIW)
        if (tid_u >= 48U) {
            return;
        }
#ifdef K11220_RS768_PUSH
        const bool use_rs768_tok = true;
#else
        const bool use_rs768_tok = (tokens > 16);
#endif
        uint32_t owner_peer = 0U;
        int64_t sub_col = 0;
        if (use_rs768_tok) {
            const uint32_t ulc = static_cast<uint32_t>(local_cnt);
            uint32_t tps = 1U;
            if (ulc == 1U) {
                idx_step = 1;
                span = 128;
                tps = 6U;
                idx = 0;
            } else {
                const uint32_t c2 = ((ulc + 1U) >> 1U) * 512U;
                const uint32_t c3 = (((ulc + 2U) * 683U) >> 11U) * 640U;
                const uint32_t c6 = (((ulc + 5U) * 683U) >> 12U) * 1024U;
                const uint32_t c12 = (((ulc + 11U) * 683U) >> 13U) * 1792U;
                if (ulc >= 7U && c12 <= c6 && c12 <= c3 && c12 <= c2) {
                    idx_step = 12;
                    span = 1536;
                    tps = 0U;
                    idx = static_cast<int32_t>(tid_u >> 2U);
                } else if (c6 <= c3 && c6 <= c2) {
                    idx_step = 6;
                    span = 768;
                    tps = 1U;
                    idx = static_cast<int32_t>(tid_u >> 3U);
                } else if (c3 <= c2) {
                    idx_step = 3;
                    span = 384;
                    tps = 2U;
                    idx = static_cast<int32_t>(tid_u >> 4U);
                } else {
                    idx_step = 2;
                    span = 256;
                    tps = 3U;
                    idx = static_cast<int32_t>((tid_u * 683U) >> 14U);
                }
            }
            if (tps == 0U) {
                owner_peer = (tid_u & 3U) << 1U;
                sub_col = 0;
            } else {
                const uint32_t rem_t = tid_u - static_cast<uint32_t>(idx) * (8U * tps);
                if (tps == 6U) {
                    owner_peer = (rem_t * 683U) >> 12U;
                    sub_col = static_cast<int64_t>(rem_t - owner_peer * 6U) * 128;
                } else if (tps == 3U) {
                    owner_peer = (rem_t * 683U) >> 11U;
                    sub_col = static_cast<int64_t>(rem_t - owner_peer * 3U) * 256;
                } else if (tps == 2U) {
                    owner_peer = rem_t >> 1U;
                    sub_col = static_cast<int64_t>(rem_t & 1U) * 384;
                } else {
                    owner_peer = rem_t;
                    sub_col = 0;
                }
            }
            column = static_cast<int64_t>(owner_peer) * 768 + sub_col;
        } else {
            select_exact48_tok();
        }
#elif defined(K112612_CLEAN_TOK_OUT)
        if (local_cnt <= 1) {
            span = 128;
            idx = 0;
            idx_step = 1;
            column = static_cast<int64_t>(tid_u) * 128;
        } else if (local_cnt == 2) {
            span = 256;
            idx = static_cast<int32_t>((tid_u * 683U) >> 14U);
            idx_step = 2;
            column = (static_cast<int64_t>(tid_u) - idx * 24) * 256;
        } else if (local_cnt == 3 || local_cnt == 9 || local_cnt == 15 || local_cnt == 21) {
            span = 384;
            idx = static_cast<int32_t>(tid_u >> 4U);
            idx_step = 3;
            column = static_cast<int64_t>(tid_u & 15U) * 384;
        } else if (local_cnt == 4 || local_cnt == 19 || local_cnt == 20) {
            span = 512;
            idx = static_cast<int32_t>((tid_u * 683U) >> 13U);
            idx_step = 4;
            column = (static_cast<int64_t>(tid_u) - idx * 12) * 512;
        } else if (local_cnt <= 6 || (local_cnt >= 17 && local_cnt <= 18) || local_cnt == 30 || local_cnt == 42) {
            span = 768;
            idx = static_cast<int32_t>(tid_u >> 3U);
            idx_step = 6;
            column = static_cast<int64_t>(tid_u & 7U) * 768;
        } else if (local_cnt <= 8 || (local_cnt >= 13 && local_cnt <= 16) || (local_cnt >= 25 && local_cnt <= 32) || (local_cnt >= 37 && local_cnt <= 40) || (local_cnt >= 49 && local_cnt <= 56) || (local_cnt >= 61 && local_cnt <= 64)) {
            span = 1024;
            idx = static_cast<int32_t>((tid_u * 683U) >> 12U);
            idx_step = 8;
            column = (static_cast<int64_t>(tid_u) - idx * 6) * 1024;
        }
#elif defined(K112600_EXACT48_RED2_OPT)
        if (tid_u >= 48U) {
            return;
        }
        select_exact48_tok();
#elif defined(K11180_TOK_OUT_BAL)
        if (local_cnt <= 1) {
            span = 128;
            idx = 0;
            idx_step = 1;
            column = static_cast<int64_t>(tid_u) * 128;
        } else if (local_cnt == 2) {
            span = 256;
            idx = static_cast<int32_t>((tid_u * 683U) >> 14U);
            idx_step = 2;
            column = (static_cast<int64_t>(tid_u) - idx * 24) * 256;
        } else if (local_cnt == 3 || local_cnt == 9 || local_cnt == 15 || local_cnt == 21) {
            span = 384;
            idx = static_cast<int32_t>(tid_u >> 4U);
            idx_step = 3;
            column = static_cast<int64_t>(tid_u & 15U) * 384;
        } else if (local_cnt == 4 || local_cnt == 19 || local_cnt == 20) {
            span = 512;
            idx = static_cast<int32_t>((tid_u * 683U) >> 13U);
            idx_step = 4;
            column = (static_cast<int64_t>(tid_u) - idx * 12) * 512;
        } else if (local_cnt <= 6 || (local_cnt >= 17 && local_cnt <= 18)) {
            span = 768;
            idx = static_cast<int32_t>(tid_u >> 3U);
            idx_step = 6;
            column = static_cast<int64_t>(tid_u & 7U) * 768;
        } else if (local_cnt <= 8 || (local_cnt >= 13 && local_cnt <= 16) || (local_cnt >= 25 && local_cnt <= 32)) {
            span = 1024;
            idx = static_cast<int32_t>((tid_u * 683U) >> 12U);
            idx_step = 8;
            column = (static_cast<int64_t>(tid_u) - idx * 6) * 1024;
        }
#else
        if (local_cnt <= 4) {
            span = 512;
            idx = static_cast<int32_t>((tid_u * 683U) >> 13U);
            idx_step = 4;
            column = (static_cast<int64_t>(tid_u) - idx * 12) * 512;
        } else if (local_cnt <= 6 || (local_cnt >= 17 && local_cnt <= 18)) {
            span = 768;
            idx = static_cast<int32_t>(tid_u >> 3U);
            idx_step = 6;
            column = static_cast<int64_t>(tid_u & 7U) * 768;
        } else if (local_cnt <= 8 || (local_cnt >= 13 && local_cnt <= 16) || (local_cnt >= 25 && local_cnt <= 32)) {
            span = 1024;
            idx = static_cast<int32_t>((tid_u * 683U) >> 12U);
            idx_step = 8;
            column = (static_cast<int64_t>(tid_u) - idx * 6) * 1024;
        }
#endif
#ifdef K11220_FAST_TOK_OUT
        const uint32_t u_cnt = static_cast<uint32_t>(local_cnt);
        const uint32_t u_ng = static_cast<uint32_t>(idx_step);
        const uint32_t u_grp = static_cast<uint32_t>(idx);
        const uint32_t q_cnt = u_cnt / u_ng;
        const uint32_t r_cnt = u_cnt - q_cnt * u_ng;
        const int32_t idx_begin = static_cast<int32_t>(u_grp * q_cnt + ((u_grp < r_cnt) ? u_grp : r_cnt));
        const int32_t idx_end = idx_begin + static_cast<int32_t>(q_cnt + ((u_grp < r_cnt) ? 1U : 0U));
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
        for (int32_t cur_i = idx_begin; cur_i < idx_end; ++cur_i) {
            const uint32_t packed = local_origs[cur_i];
            const int64_t original = packed & 0xFFU;
            const int64_t expert = (packed >> 8) & 0x3FU;
            const int64_t begin = (packed >> 14) & 0xFFU;
            const int64_t row = (packed >> 22) & 0x1FU;
            const int64_t count = ((packed >> 27) & 0x1FU) + 1;
            const int64_t route = begin + row;
            GlobalTensor<RoutedProduct> dots;
            GlobalTensor<float> bias, weight_scales;
            dots.SetGlobalBuffer(product + 2 * begin * kHidden,
                                 2 * count * kHidden);
            const bool need_scale = (expert != last_expert);
            last_expert = expert;
            if (need_scale) {
                if (mw_s != nullptr) {
                    const uint32_t lslot = my_gexp[expert];
                    if (lslot < 32U) {
                        bias.SetGlobalBuffer(correction + static_cast<int64_t>(lslot) * kHidden, kHidden);
                        weight_scales.SetGlobalBuffer(scales + static_cast<int64_t>(lslot) * kHidden, kHidden);
                    } else {
                        const int64_t ms = static_cast<int64_t>(lslot - 32U);
                        bias.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(mw_s[4U]) + ms * kHidden, kHidden);
                        weight_scales.SetGlobalBuffer(reinterpret_cast<__gm__ ScaleStorage*>(mw_s[5U]) + ms * kHidden, kHidden);
                    }
                } else {
                    bias.SetGlobalBuffer(correction + expert * kHidden, kHidden);
                    weight_scales.SetGlobalBuffer(scales + expert * kHidden, kHidden);
                }
            }
            RecombineCached(held, dots, bias, row, count, kHidden, column, span, need_scale ? &weight_scales : nullptr);
            Muls(held.value, held.value, activation.GetValue(route), span);
            PipeBarrier<PIPE_V>();
            const uint32_t ping = static_cast<uint32_t>(cur_i - idx_begin) & 1U;
            const event_t out_evt = ping ? EVENT_ID3 : EVENT_ID2;
            LocalTensor<bfloat16_t> dst_ub = held.tok_out[ping * 2048U];
            WaitFlag<HardEvent::MTE3_V>(out_evt);
            Cast(dst_ub, held.value, RoundMode::CAST_RINT, span);
            SetFlag<HardEvent::V_MTE3>(out_evt);
            WaitFlag<HardEvent::V_MTE3>(out_evt);
#if defined(K112110_SIP_RED1) && defined(TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H)
            if (rt_plane != nullptr && need_r1_wait) {
                const auto r1_layout = whole_layer_tree::MakeCommunicationLayout(comm_capacity, comm_ranks);
                uint32_t pend = 0xFFU & ~(1U << static_cast<uint32_t>(comm_rank));
                while (pend != 0U) {
                    for (uint32_t sp = 0U; sp < 7U; ++sp) {
                        const int32_t p = (comm_rank + static_cast<int32_t>(((tid_u + sp) % 7U) + 1U)) & 7;
                        if ((pend & (1U << p)) == 0U) {
                            continue;
                        }
                        __gm__ volatile int32_t* ack_addr =
                            whole_layer_tree::CommunicationR1Done(comm_window, r1_layout, p);
                        dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(
                            const_cast<__gm__ int32_t*>(ack_addr)));
                        if (static_cast<int32_t>(static_cast<uint32_t>(*ack_addr) - comm_base_epoch) >= 0) {
                            pend &= ~(1U << p);
                        }
                    }
                }
                need_r1_wait = false;
            }
#endif
#if (defined(K11220_RS768_PUSH) || defined(K112600_RS768_HIW)) && defined(TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H)
            if (use_rs768_tok) {
                if (rt_plane != nullptr) {
                    __gm__ bfloat16_t* dst_addr = rt_plane + original * 768 + sub_col;
                    if (span == 1536) {
                        for (uint32_t hk = 0U; hk < 2U; ++hk) {
                            const int32_t p_dst = static_cast<int32_t>(owner_peer + hk);
                            LocalTensor<bfloat16_t> sub_ub = dst_ub[hk * 768U];
                            if (p_dst == comm_rank) {
                                GlobalTensor<bfloat16_t> self_rt;
                                self_rt.SetGlobalBuffer(dst_addr, 768);
                                DataCopy(self_rt, sub_ub, 768);
                            } else {
                                GlobalTensor<bfloat16_t> peer_dst;
                                peer_dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
                                    aclshmem_ptr(dst_addr, p_dst)), 768);
                                DataCopy(peer_dst, sub_ub, 768);
                            }
                        }
                    } else if (static_cast<int32_t>(owner_peer) == comm_rank) {
                        GlobalTensor<bfloat16_t> self_rt;
                        self_rt.SetGlobalBuffer(dst_addr, span);
                        DataCopy(self_rt, dst_ub, span);
                    } else {
                        GlobalTensor<bfloat16_t> peer_dst;
                        peer_dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
                            aclshmem_ptr(dst_addr, static_cast<int32_t>(owner_peer))), span);
                        DataCopy(peer_dst, dst_ub, span);
                    }
                } else {
                    DataCopy(route_out_gm[original * kHidden + column], dst_ub, span);
                }
            } else if (rt_plane != nullptr) {
                __gm__ bfloat16_t* dst_addr = rt_plane + original * kHidden + column;
                GlobalTensor<bfloat16_t> self_rt;
                self_rt.SetGlobalBuffer(dst_addr, span);
                DataCopy(self_rt, dst_ub, span);
#ifndef K112808_PULL_RED2
                for (uint32_t p = 0U; p < 3U; ++p) {
                    const uint32_t step2 = (((tid_u + p) % 3U) + 1U) << 1U;
                    const int32_t peer = (comm_rank + static_cast<int32_t>(step2)) & 7;
                    GlobalTensor<bfloat16_t> peer_dst;
                    peer_dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
                        aclshmem_ptr(dst_addr, peer)), span);
                    DataCopy(peer_dst, dst_ub, span);
                }
#endif
            } else {
                DataCopy(route_out_gm[original * kHidden + column], dst_ub, span);
            }
#elif defined(K11150_SIP_RELAY_RED2) && defined(TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H)
            if (rt_plane != nullptr) {
                __gm__ bfloat16_t* dst_addr = rt_plane + original * kHidden + column;
                GlobalTensor<bfloat16_t> self_rt;
                self_rt.SetGlobalBuffer(dst_addr, span);
                DataCopy(self_rt, dst_ub, span);
#ifndef K112808_PULL_RED2
                for (uint32_t p = 0U; p < 3U; ++p) {
                    const uint32_t step2 = (((tid_u + p) % 3U) + 1U) << 1U;
                    const int32_t peer = (comm_rank + static_cast<int32_t>(step2)) & 7;
                    GlobalTensor<bfloat16_t> peer_dst;
                    peer_dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
                        aclshmem_ptr(dst_addr, peer)), span);
                    DataCopy(peer_dst, dst_ub, span);
                }
#endif
            } else {
                DataCopy(route_out_gm[original * kHidden + column], dst_ub, span);
            }
#else
            DataCopy(route_out_gm[original * kHidden + column], dst_ub, span);
#ifdef TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H
            if (rt_plane != nullptr) {
                __gm__ bfloat16_t* dst_addr = rt_plane + original * kHidden + column;
                for (uint32_t p = 0U; p < 7U; ++p) {
                    const uint32_t off = ((tid_u + p) % 7U) + 1U;
                    const int32_t peer = (comm_rank + static_cast<int32_t>(off)) & 7;
                    GlobalTensor<bfloat16_t> peer_dst;
                    peer_dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
                        aclshmem_ptr(dst_addr, peer)), span);
                    DataCopy(peer_dst, dst_ub, span);
                }
            }
#endif
#endif
            SetFlag<HardEvent::MTE3_V>(out_evt);
        }
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
#else
        for (; idx < local_cnt; idx += idx_step) {
            const uint32_t packed = local_origs[idx];
            const int64_t original = packed & 0xFFU;
            const int64_t expert = (packed >> 8) & 0x3FU;
            const int64_t begin = (packed >> 14) & 0xFFU;
            const int64_t row = (packed >> 22) & 0x1FU;
            const int64_t count = ((packed >> 27) & 0x1FU) + 1;
            const int64_t route = begin + row;
            GlobalTensor<RoutedProduct> dots;
            GlobalTensor<float> bias, weight_scales;
            dots.SetGlobalBuffer(product + 2 * begin * kHidden,
                                 2 * count * kHidden);
            const bool need_scale = (expert != last_expert);
            last_expert = expert;
            if (mw_s != nullptr) {
                const uint32_t lslot = my_gexp[expert];
                if (lslot < 32U) {
                    bias.SetGlobalBuffer(correction + static_cast<int64_t>(lslot) * kHidden, kHidden);
                    if (need_scale) {
                        weight_scales.SetGlobalBuffer(scales + static_cast<int64_t>(lslot) * kHidden, kHidden);
                    }
                } else {
                    const int64_t ms = static_cast<int64_t>(lslot - 32U);
                    bias.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(mw_s[4U]) + ms * kHidden, kHidden);
                    if (need_scale) {
                        weight_scales.SetGlobalBuffer(reinterpret_cast<__gm__ ScaleStorage*>(mw_s[5U]) + ms * kHidden, kHidden);
                    }
                }
            } else {
                bias.SetGlobalBuffer(correction + expert * kHidden, kHidden);
                if (need_scale) {
                    weight_scales.SetGlobalBuffer(scales + expert * kHidden, kHidden);
                }
            }
            Recombine(held, dots, bias, row, count, kHidden, column, span, need_scale ? &weight_scales : nullptr);
            Muls(held.value, held.value, activation.GetValue(route), span);
            PipeBarrier<PIPE_V>();
            Cast(narrow, held.value, RoundMode::CAST_RINT, span);
            Fence<HardEvent::V_MTE3>();
#if (defined(K11220_RS768_PUSH) || defined(K112600_RS768_HIW)) && defined(TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H)
            if (use_rs768_tok) {
                if (rt_plane != nullptr) {
                    __gm__ bfloat16_t* dst_addr = rt_plane + original * 768 + sub_col;
                    if (span == 1536) {
                        for (uint32_t hk = 0U; hk < 2U; ++hk) {
                            const int32_t p_dst = static_cast<int32_t>(owner_peer + hk);
                            LocalTensor<bfloat16_t> sub_ub = narrow[hk * 768U];
                            if (p_dst == comm_rank) {
                                GlobalTensor<bfloat16_t> self_rt;
                                self_rt.SetGlobalBuffer(dst_addr, 768);
                                DataCopy(self_rt, sub_ub, 768);
                            } else {
                                GlobalTensor<bfloat16_t> peer_dst;
                                peer_dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
                                    aclshmem_ptr(dst_addr, p_dst)), 768);
                                DataCopy(peer_dst, sub_ub, 768);
                            }
                        }
                    } else if (static_cast<int32_t>(owner_peer) == comm_rank) {
                        GlobalTensor<bfloat16_t> self_rt;
                        self_rt.SetGlobalBuffer(dst_addr, span);
                        DataCopy(self_rt, narrow, span);
                    } else {
                        GlobalTensor<bfloat16_t> peer_dst;
                        peer_dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
                            aclshmem_ptr(dst_addr, static_cast<int32_t>(owner_peer))), span);
                        DataCopy(peer_dst, narrow, span);
                    }
                } else {
                    DataCopy(route_out_gm[original * kHidden + column], narrow, span);
                }
            } else if (rt_plane != nullptr) {
                __gm__ bfloat16_t* dst_addr = rt_plane + original * kHidden + column;
                GlobalTensor<bfloat16_t> self_rt;
                self_rt.SetGlobalBuffer(dst_addr, span);
                DataCopy(self_rt, narrow, span);
                for (uint32_t p = 0U; p < 3U; ++p) {
                    const uint32_t step2 = (((tid_u + p) % 3U) + 1U) << 1U;
                    const int32_t peer = (comm_rank + static_cast<int32_t>(step2)) & 7;
                    GlobalTensor<bfloat16_t> peer_dst;
                    peer_dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
                        aclshmem_ptr(dst_addr, peer)), span);
                    DataCopy(peer_dst, narrow, span);
                }
            } else {
                DataCopy(route_out_gm[original * kHidden + column], narrow, span);
            }
#elif defined(K11150_SIP_RELAY_RED2) && defined(TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H)
            if (rt_plane != nullptr) {
                __gm__ bfloat16_t* dst_addr = rt_plane + original * kHidden + column;
                GlobalTensor<bfloat16_t> self_rt;
                self_rt.SetGlobalBuffer(dst_addr, span);
                DataCopy(self_rt, narrow, span);
                for (uint32_t p = 0U; p < 3U; ++p) {
                    const uint32_t step2 = (((tid_u + p) % 3U) + 1U) << 1U;
                    const int32_t peer = (comm_rank + static_cast<int32_t>(step2)) & 7;
                    GlobalTensor<bfloat16_t> peer_dst;
                    peer_dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
                        aclshmem_ptr(dst_addr, peer)), span);
                    DataCopy(peer_dst, narrow, span);
                }
            } else {
                DataCopy(route_out_gm[original * kHidden + column], narrow, span);
            }
#else
            DataCopy(route_out_gm[original * kHidden + column], narrow, span);
#ifdef TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H
            if (rt_plane != nullptr) {
                __gm__ bfloat16_t* dst_addr = rt_plane + original * kHidden + column;
                for (uint32_t p = 0U; p < 7U; ++p) {
                    const uint32_t off = ((tid_u + p) % 7U) + 1U;
                    const int32_t peer = (comm_rank + static_cast<int32_t>(off)) & 7;
                    GlobalTensor<bfloat16_t> peer_dst;
                    peer_dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
                        aclshmem_ptr(dst_addr, peer)), span);
                    DataCopy(peer_dst, narrow, span);
                }
            }
#endif
#endif
            Fence<HardEvent::MTE3_V>();
        }
#endif
        return;
    }
    for (int64_t task = GetBlockIdx(); task < tokens * kOutputTiles;
         task += 2 * blocks) {
        const int64_t token = OutputToken(task);
        const int64_t column = OutputColumn(task);
        const int64_t output_offset = token * kHidden + column;
        if ((local_mask & (1U << uint32_t(token))) == 0U) {
            if (combine_shared) {
                Fence<HardEvent::V_MTE2>();
                DataCopy(narrow, shared[output_offset], span);
                Fence<HardEvent::MTE2_MTE3>();
                DataCopy(combined[output_offset], narrow, span);
                Fence<HardEvent::MTE3_V>();
            }
            continue;
        }
        Duplicate(held.other, 0.0F, span);
        PipeBarrier<PIPE_V>();
        bool has_local = false;
        for (int64_t slot = 0; slot < kTopK; ++slot) {
            const int64_t original = token * kTopK + slot;
            const int64_t expert = expert_ids.GetValue(original) - expert_offset;
            if (expert < 0 || expert >= kLocalExperts) {
                continue;
            }
            const float probability = held.original_weight.GetValue(original);
            if (probability == 0.0F) {
                continue;
            }
            const int64_t route = inverse.GetValue(original);
            const int64_t begin = held.start.GetValue(expert);
            const int64_t count = held.count.GetValue(expert);
            const int64_t row = ExpertRow(route, begin);
            GlobalTensor<RoutedProduct> dots;
            GlobalTensor<float> bias, weight_scales;
            dots.SetGlobalBuffer(product + 2 * begin * kHidden,
                                 2 * count * kHidden);
            bias.SetGlobalBuffer(correction + expert * kHidden, kHidden);
            weight_scales.SetGlobalBuffer(scales + expert * kHidden, kHidden);
            LoadScale(held, weight_scales, column, span);
            Recombine(held, dots, bias, row, count, kHidden, column, span);
            Muls(held.value, held.value, activation.GetValue(route), span);
            PipeBarrier<PIPE_V>();
            Cast(narrow, held.value, RoundMode::CAST_RINT, span);
            PipeBarrier<PIPE_V>();
            Cast(held.value, narrow, RoundMode::CAST_NONE, span);
            PipeBarrier<PIPE_V>();
            if (!combine_shared) {
                Fence<HardEvent::V_MTE3>();
                DataCopy(route_out_gm[original * kHidden + column], narrow, span);
                Fence<HardEvent::MTE3_V>();
                continue;
            }
            Muls(held.value, held.value, probability, span);
            PipeBarrier<PIPE_V>();
            if (!has_local) {
                DataCopy(held.other, held.value, span);
                has_local = true;
            } else {
                Add(held.other, held.other, held.value, span);
            }
            PipeBarrier<PIPE_V>();
        }

        if (!combine_shared) {
            continue;
        }
        if (has_local) {
            Cast(narrow, held.other, RoundMode::CAST_RINT, span);
            PipeBarrier<PIPE_V>();
            Cast(held.value, narrow, RoundMode::CAST_NONE, span);
            PipeBarrier<PIPE_V>();
            Muls(held.value, held.value, 2.5F, span);
            PipeBarrier<PIPE_V>();
            Cast(narrow, held.value, RoundMode::CAST_RINT, span);
            PipeBarrier<PIPE_V>();
            Cast(held.other, narrow, RoundMode::CAST_NONE, span);
            PipeBarrier<PIPE_V>();
        }
        Fence<HardEvent::V_MTE2>();
        DataCopy(narrow, shared[output_offset], span);
        Fence<HardEvent::MTE2_V>();
        Cast(held.work, narrow, RoundMode::CAST_NONE, span);
        PipeBarrier<PIPE_V>();
        Add(held.value, held.other, held.work, span);
        PipeBarrier<PIPE_V>();
        Cast(narrow, held.value, RoundMode::CAST_RINT, span);
        Fence<HardEvent::V_MTE3>();
        DataCopy(combined[output_offset], narrow, span);
        Fence<HardEvent::MTE3_V>();
    }
}

#endif
