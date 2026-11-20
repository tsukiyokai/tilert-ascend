__aicore__ inline void ReduceCommunicationPayloadTiled(
    CommunicationBuffers& buffers, __gm__ uint8_t* window,
    const CommunicationLayout& layout, __gm__ bfloat16_t* partial,
    __gm__ bfloat16_t* output, int64_t rows, int32_t rank, int32_t ranks,
    uint32_t epoch, int64_t vector_index, int64_t vectors,
    __gm__ bfloat16_t* shared_sum = nullptr, bool quad4 = false)
{
    (void)shared_sum;
    constexpr int64_t kOutTile = 256;
    constexpr int64_t kOutTiles = 24;
    constexpr int64_t kMaxRows = 16;
    const int64_t batch_rows = rows <= 1 ? 1
        : (rows <= 2 * kMaxRows ? ((rows + 1) >> 1) : kMaxRows);
    const int64_t batches = (rows <= batch_rows) ? 1 : 2;
    const int64_t total_tasks = batches * kOutTiles;
    if (vector_index >= total_tasks) {
        return;
    }
    const int32_t signal = static_cast<int32_t>(epoch);
    const TreePlan plan = GetTreePlan(ranks);
    const uint32_t p_mask = quad4 ? 3U : 1U;
    __gm__ bfloat16_t* att_plane = reinterpret_cast<__gm__ bfloat16_t*>(
        window + layout.payload + (epoch & p_mask) * layout.plane_bytes);
#if defined(K112110_SIP_RED1)
#if defined(K112170_DEC_SIP_ATT) && defined(K112140_SIP_RED1_MIN_ROWS)
    const bool sip_att = (rows >= static_cast<int64_t>(K112140_SIP_RED1_MIN_ROWS));
#elif defined(K112140_SIP_RED1_MIN_ROWS)
    const bool sip_att = quad4 || (rows >= static_cast<int64_t>(K112140_SIP_RED1_MIN_ROWS));
#else
    const bool sip_att = true;
#endif
    __gm__ bfloat16_t* att_twin_plane = sip_att
        ? reinterpret_cast<__gm__ bfloat16_t*>(aclshmem_ptr(att_plane, rank ^ 1))
        : nullptr;
#elif defined(K11180_QUAD4_SIP_ALL)
    const bool sip_att = quad4;
    __gm__ bfloat16_t* att_twin_plane = quad4
        ? reinterpret_cast<__gm__ bfloat16_t*>(aclshmem_ptr(att_plane, rank ^ 1))
        : nullptr;
#else
    const bool sip_att = false;
    __gm__ bfloat16_t* att_twin_plane = nullptr;
#endif
    const int64_t rank_stride = layout.rank_bytes / sizeof(bfloat16_t);
    uint32_t unready = (ranks >= 32) ? 0xFFFFFFFFU : ((1U << static_cast<uint32_t>(ranks)) - 1U);
    for (int64_t task = vector_index; task < total_tasks; task += vectors) {
        if (task != vector_index) {
            CommunicationFence<AscendC::HardEvent::MTE3_MTE2>();
        }
        const uint32_t utask = static_cast<uint32_t>(task);
        const uint32_t q = (utask * 683U) >> 14U;
        const int64_t tile = static_cast<int64_t>(utask - q * 24U);
        const int64_t begin = static_cast<int64_t>(q) * batch_rows;
        const int64_t count = rows <= begin ? 0
            : (rows - begin < batch_rows ? rows - begin : batch_rows);
        if (count <= 0) {
            continue;
        }
        const int64_t chunk = count * kOutTile;
        const int64_t pkt_off = begin * kCommunicationHidden + tile * chunk;
        uint32_t pending = unready;
        for (int32_t leaf = 0; leaf < ranks; ++leaf) {
            const TreeStep step = GetTreeStep(plan, leaf);
            if (static_cast<int32_t>(step.rank) == rank) {
                AscendC::GlobalTensor<bfloat16_t> source;
                if (sip_att) {
                    source.SetGlobalBuffer(att_plane + static_cast<int64_t>(step.rank) * rank_stride + pkt_off, chunk);
                } else {
                    source.SetGlobalBuffer(partial + pkt_off, chunk);
                }
                AscendC::DataCopy(buffers.narrow[leaf * chunk], source, chunk);
                pending &= ~(1U << leaf);
            } else if ((pending & (1U << leaf)) == 0U) {
                AscendC::GlobalTensor<bfloat16_t> source;
                __gm__ bfloat16_t* src_base = (sip_att && (((static_cast<int32_t>(step.rank) ^ rank) & 1) != 0))
                    ? att_twin_plane : att_plane;
                source.SetGlobalBuffer(src_base + static_cast<int64_t>(step.rank) * rank_stride + pkt_off, chunk);
                AscendC::DataCopy(buffers.narrow[leaf * chunk], source, chunk);
            }
        }
#ifdef K112830_FAST_POLL_RED2
        if (pending != 0U) {
            constexpr uint32_t kReadyIntsPerRank =
                static_cast<uint32_t>(kCommunicationCacheLine / sizeof(int32_t));
            const int32_t kReadyPollInts = ranks * static_cast<int32_t>(kReadyIntsPerRank);
            AscendC::GlobalTensor<int32_t> ready_gm;
            ready_gm.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t*>(window + layout.ready), kReadyPollInts);
            auto sig_ub = buffers.accumulator[31 * 1024].ReinterpretCast<int32_t>();
            const uint32_t ep_slot = epoch & 3U;
            while (pending != 0U) {
                AscendC::DataCopy(sig_ub, ready_gm, kReadyPollInts);
                CommunicationFence<AscendC::HardEvent::MTE2_S>();
                for (int32_t l = 0; l < ranks; ++l) {
                    const int32_t leaf = (l + static_cast<int32_t>(vector_index)) & (ranks - 1);
                    if ((pending & (1U << leaf)) == 0U) {
                        continue;
                    }
                    const TreeStep step = GetTreeStep(plan, leaf);
                    const uint32_t s_idx =
                        static_cast<uint32_t>(step.rank) * kReadyIntsPerRank + ep_slot;
                    if (sig_ub.GetValue(s_idx) == signal) {
                        AscendC::GlobalTensor<bfloat16_t> source;
                        const bool use_twin =
                            sip_att && (((static_cast<int32_t>(step.rank) ^ rank) & 1) != 0);
                        __gm__ bfloat16_t* src_base = use_twin ? att_twin_plane : att_plane;
                        const int64_t r_off =
                            static_cast<int64_t>(step.rank) * rank_stride + pkt_off;
                        source.SetGlobalBuffer(src_base + r_off, chunk);
                        AscendC::DataCopy(buffers.narrow[leaf * chunk], source, chunk);
                        pending &= ~(1U << leaf);
                    }
                }
            }
        }
#else
        while (pending != 0U) {
            for (int32_t l = 0; l < ranks; ++l) {
                const int32_t leaf = (l + static_cast<int32_t>(vector_index)) & (ranks - 1);
                if ((pending & (1U << leaf)) == 0U) {
                    continue;
                }
                const TreeStep step = GetTreeStep(plan, leaf);
                __gm__ volatile int32_t* sig_addr =
                    CommunicationReady(window, layout, ranks, epoch, step.rank);
                dcci_cacheline((__gm__ uint8_t*)sig_addr);
                if (*sig_addr == signal) {
                    AscendC::GlobalTensor<bfloat16_t> source;
                    __gm__ bfloat16_t* src_base = (sip_att && (((static_cast<int32_t>(step.rank) ^ rank) & 1) != 0))
                        ? att_twin_plane : att_plane;
                    source.SetGlobalBuffer(src_base + static_cast<int64_t>(step.rank) * rank_stride + pkt_off, chunk);
                    AscendC::DataCopy(buffers.narrow[leaf * chunk], source, chunk);
                    pending &= ~(1U << leaf);
                }
            }
        }
#endif
        unready = 0U;
        CommunicationFence<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(buffers.accumulator[0], buffers.narrow[0],
                      AscendC::RoundMode::CAST_NONE, ranks * chunk);
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t stride = 1; stride < ranks; stride <<= 1) {
            for (int32_t i = 0; i + stride < ranks; i += (stride << 1)) {
                AscendC::Add(buffers.accumulator[i * chunk],
                             buffers.accumulator[i * chunk],
                             buffers.accumulator[(i + stride) * chunk], chunk);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Cast(buffers.narrow, buffers.accumulator,
                      AscendC::RoundMode::CAST_RINT, chunk);
        CommunicationFence<AscendC::HardEvent::V_MTE3>();
        AscendC::GlobalTensor<bfloat16_t> destination;
        destination.SetGlobalBuffer(output, rows * kCommunicationHidden);
        AscendC::DataCopyParams copy;
        copy.blockCount = static_cast<uint16_t>(count);
        copy.blockLen = static_cast<uint16_t>(kOutTile * sizeof(bfloat16_t) / 32);
        copy.srcStride = 0;
        copy.dstStride = static_cast<uint16_t>((kCommunicationHidden - kOutTile) * sizeof(bfloat16_t) / 32);
        AscendC::DataCopy(destination[begin * kCommunicationHidden + tile * kOutTile],
                          buffers.narrow, copy);
    }
}

__aicore__ inline void ReduceCommunicationPayloadSlots(
    CommunicationBuffers& buffers, __gm__ uint8_t* window,
    const CommunicationLayout& layout, __gm__ bfloat16_t* partial,
    __gm__ bfloat16_t* output, int64_t rows,
    int32_t rank, int32_t ranks, uint32_t sh_epoch, uint32_t rt_epoch,
    int64_t vector_index, int64_t vectors, __gm__ bfloat16_t* shared_partial,
    __gm__ const int32_t* expert_ids, __gm__ const float* probabilities,
    const __gm__ uint8_t* owner_gm = nullptr, bool quad4 = false,
    __gm__ const bfloat16_t* pre_shared_sum = nullptr, int64_t* prof_acc = nullptr,
    __gm__ const bfloat16_t* next_residual = nullptr,
    __gm__ const bfloat16_t* next_gamma = nullptr,
    bool fused_next_norm = false)
{
    (void)pre_shared_sum;
    (void)prof_acc;
#ifdef ENABLE_KERNEL_PROF
    const int64_t t_r2_enter = (prof_acc != nullptr && vector_index == 0) ? AscendC::GetSystemCycle() : 0;
    int64_t t_r2_arr = t_r2_enter;
    int64_t t_sh_done = t_r2_enter;
#endif
    (void)vectors;
    const int32_t sh_signal = static_cast<int32_t>(sh_epoch);
    const int32_t signal = static_cast<int32_t>(rt_epoch);
    const uint32_t tid = static_cast<uint32_t>(vector_index);
    int64_t chunk = 2048;
    int64_t local_tok = static_cast<int64_t>((tid * 171U) >> 9U);
    int64_t tok_step = 16;
    int64_t col = static_cast<int64_t>(tid - static_cast<uint32_t>(local_tok) * 3U) * 2048;
    const int64_t wave_cap = (ranks > 8) ? 8 : 16;
    int64_t wave_base = 0;
#ifdef K112600_EXACT48_RED2_OPT
    auto map_red2_wave = [&](int64_t w_rows, int64_t& r_tok, int64_t& r_col, int64_t& r_chunk) __aicore__ {
        if (w_rows <= 1) {
            r_chunk = 128; r_tok = 0; r_col = static_cast<int64_t>(tid) * 128;
        } else if (w_rows == 2) {
            r_chunk = 256; const uint32_t q = (tid * 683U) >> 14U;
            r_tok = static_cast<int64_t>(q); r_col = static_cast<int64_t>(tid - q * 24U) * 256;
        } else if (w_rows == 3) {
            r_chunk = 384; r_tok = static_cast<int64_t>(tid >> 4U); r_col = static_cast<int64_t>(tid & 15U) * 384;
        } else if (w_rows == 4) {
            r_chunk = 512; const uint32_t q = (tid * 683U) >> 13U;
            r_tok = static_cast<int64_t>(q); r_col = static_cast<int64_t>(tid - q * 12U) * 512;
        } else if (w_rows <= 6) {
            r_chunk = 768; r_tok = static_cast<int64_t>(tid >> 3U); r_col = static_cast<int64_t>(tid & 7U) * 768;
        } else if (w_rows <= 8) {
            r_chunk = 1024; const uint32_t q = (tid * 683U) >> 12U;
            r_tok = static_cast<int64_t>(q); r_col = static_cast<int64_t>(tid - q * 6U) * 1024;
        } else if (w_rows <= 12) {
            r_chunk = 1536; r_tok = static_cast<int64_t>(tid >> 2U); r_col = static_cast<int64_t>(tid & 3U) * 1536;
        } else {
            r_chunk = 2048; const uint32_t q = (tid * 171U) >> 9U;
            r_tok = static_cast<int64_t>(q); r_col = static_cast<int64_t>(tid - q * 3U) * 2048;
        }
    };
    const int64_t w0_rows = (rows > wave_cap) ? wave_cap : rows;
    map_red2_wave(w0_rows, local_tok, col, chunk);
    if (local_tok >= w0_rows) {
        return;
    }
#else
    if (rows <= 4) {
        chunk = 512;
        local_tok = static_cast<int64_t>((tid * 683U) >> 13U);
        tok_step = 4;
        col = static_cast<int64_t>(tid - static_cast<uint32_t>(local_tok) * 12U) * 512;
    } else if ((rows >= 5 && rows <= 8) || (ranks > 8 && rows > 8)) {
        chunk = 1024;
        local_tok = static_cast<int64_t>((tid * 683U) >> 12U);
        tok_step = 8;
        col = static_cast<int64_t>(tid - static_cast<uint32_t>(local_tok) * 6U) * 1024;
    } else if (rows >= 9 && rows <= 12) {
        chunk = 1536;
        local_tok = static_cast<int64_t>(tid >> 2);
        tok_step = 12;
        col = static_cast<int64_t>(tid & 3U) * 1536;
    }
#endif
    if (local_tok >= rows) {
        return;
    }
    AscendC::GlobalTensor<int32_t> eids_gm;
    AscendC::GlobalTensor<float> probs_gm;
    eids_gm.SetGlobalBuffer(const_cast<__gm__ int32_t*>(expert_ids), rows * 8);
    probs_gm.SetGlobalBuffer(const_cast<__gm__ float*>(probabilities), rows * 8);
    auto eids_ub = buffers.accumulator[28 * 1024].ReinterpretCast<int32_t>();
    auto probs_ub = buffers.accumulator[29 * 1024];
    auto owner_ub = buffers.accumulator[30 * 1024].ReinterpretCast<uint32_t>();
    AscendC::DataCopy(eids_ub, eids_gm, rows * 8);
    AscendC::DataCopy(probs_ub, probs_gm, rows * 8);
    if (owner_gm != nullptr) {
        AscendC::GlobalTensor<uint32_t> own_src;
#ifdef K11180_LITE_MIRROR
        own_src.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(const_cast<__gm__ uint8_t*>(owner_gm)), 16);
        AscendC::DataCopy(owner_ub, own_src, 16);
#else
        own_src.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(const_cast<__gm__ uint8_t*>(owner_gm)), 64);
        AscendC::DataCopy(owner_ub, own_src, 64);
#endif
    }
    const TreePlan plan = GetTreePlan(ranks);
    const uint32_t p_mask = quad4 ? 3U : 1U;
    __gm__ bfloat16_t* sh_plane = reinterpret_cast<__gm__ bfloat16_t*>(
        window + layout.payload + (sh_epoch & p_mask) * layout.plane_bytes);
    __gm__ bfloat16_t* rt_plane = reinterpret_cast<__gm__ bfloat16_t*>(
        window + layout.payload + (rt_epoch & p_mask) * layout.plane_bytes);
#ifdef K11180_QUAD4_SIP_ALL
    __gm__ bfloat16_t* sh_twin_plane = quad4
        ? reinterpret_cast<__gm__ bfloat16_t*>(aclshmem_ptr(sh_plane, rank ^ 1))
        : nullptr;
#endif
#ifdef K11150_SIP_RELAY_RED2
    __gm__ bfloat16_t* rt_twin_plane = reinterpret_cast<__gm__ bfloat16_t*>(
        aclshmem_ptr(rt_plane, rank ^ 1));
#endif

    const int64_t rank_stride = layout.rank_bytes / sizeof(bfloat16_t);
    auto rt_narrow = buffers.narrow[16384];
    uint32_t unready = (ranks >= 32) ? 0xFFFFFFFFU : ((1U << static_cast<uint32_t>(ranks)) - 1U);
    uint32_t rt_unready = unready & ~(1U << static_cast<uint32_t>(rank));
    uint8_t owner_of_exp[256];
#ifdef K112830_FAST_POLL_RED2
    constexpr uint32_t kReadyIntsPerRank =
        static_cast<uint32_t>(kCommunicationCacheLine / sizeof(int32_t));
    const int32_t kReadyPollInts = ranks * static_cast<int32_t>(kReadyIntsPerRank);
    AscendC::GlobalTensor<int32_t> ready_gm;
    ready_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(window + layout.ready), kReadyPollInts);
    auto sig_ub = buffers.accumulator[31 * 1024].ReinterpretCast<int32_t>();
    auto refresh_sigs = [&](uint32_t& mask, uint32_t ep, int32_t sig_val) __aicore__ {
        AscendC::DataCopy(sig_ub, ready_gm, kReadyPollInts);
        CommunicationFence<AscendC::HardEvent::MTE2_S>();
        const uint32_t ep_slot = ep & 3U;
        for (uint32_t r = 0U; r < static_cast<uint32_t>(ranks); ++r) {
            if ((mask & (1U << r)) != 0U) {
                if (sig_ub.GetValue(r * kReadyIntsPerRank + ep_slot) == sig_val) {
                    mask &= ~(1U << r);
                }
            }
        }
    };
#endif

    for (bool first_tile = true; ; first_tile = false) {
        if (!first_tile) {
#ifdef K112600_EXACT48_RED2_OPT
            wave_base += wave_cap;
            if (wave_base >= rows) {
                break;
            }
            const int64_t rem_rows = rows - wave_base;
            const int64_t w_rows = (rem_rows > wave_cap) ? wave_cap : rem_rows;
            int64_t rel1 = 0;
            map_red2_wave(w_rows, rel1, col, chunk);
            if (rel1 >= w_rows) {
                break;
            }
            local_tok = wave_base + rel1;
#else
            if (ranks > 8 && rows > 8) {
                local_tok += tok_step;
            } else if (rows == 24) {
                if (chunk == 1024) {
                    break;
                }
                chunk = 1024;
                local_tok = 16 + static_cast<int64_t>((tid * 683U) >> 12U);
                col = static_cast<int64_t>(tid - static_cast<uint32_t>(local_tok - 16) * 6U) * 1024;
            } else if (rows == 18) {
                if (chunk == 256) {
                    break;
                }
                chunk = 256;
                local_tok = 16 + static_cast<int64_t>((tid * 683U) >> 14U);
                col = static_cast<int64_t>(tid - static_cast<uint32_t>(local_tok - 16) * 24U) * 256;
            } else {
                local_tok += tok_step;
            }
            if (local_tok >= rows) {
                break;
            }
#endif
            CommunicationFence<AscendC::HardEvent::MTE3_MTE2>();
            CommunicationFence<AscendC::HardEvent::V_MTE2>();
        }
        const int64_t sh_off = local_tok * kCommunicationHidden + col;
#ifdef K112612_OVERLAP_SH_P4
        if (pre_shared_sum != nullptr) {
            AscendC::GlobalTensor<bfloat16_t> pre_sh_gm;
            pre_sh_gm.SetGlobalBuffer(const_cast<__gm__ bfloat16_t*>(pre_shared_sum) + sh_off, chunk);
            AscendC::DataCopy(buffers.narrow[0], pre_sh_gm, chunk);
            unready = 0U;
        } else
#endif
        {
#ifdef K112830_FAST_POLL_RED2
        while (unready != 0U) {
            uint32_t rmask = unready;
            refresh_sigs(rmask, sh_epoch, sh_signal);
            const uint32_t sh_s_ep = sh_epoch & 3U;
            const uint32_t rt_s_ep = rt_epoch & 3U;
            for (uint32_t r = 0U; r < static_cast<uint32_t>(ranks); ++r) {
                const uint32_t rt_s_idx = r * kReadyIntsPerRank + rt_s_ep;
                if ((rt_unready & (1U << r)) != 0U && sig_ub.GetValue(rt_s_idx) == signal) {
                    rt_unready &= ~(1U << r);
                }
            }
            if (static_cast<int32_t>(rmask) & (1U << rank)) {
                rmask &= ~(1U << rank);
            }
            unready = rmask;
        }
        for (int32_t leaf = 0; leaf < ranks; ++leaf) {
            const TreeStep step = GetTreeStep(plan, leaf);
            AscendC::GlobalTensor<bfloat16_t> sh_src;
            if (static_cast<int32_t>(step.rank) == rank) {
                if (quad4) {
                    sh_src.SetGlobalBuffer(sh_plane + static_cast<int64_t>(step.rank) * rank_stride + sh_off, chunk);
                } else {
                    sh_src.SetGlobalBuffer(shared_partial + sh_off, chunk);
                }
            } else {
#if defined(K11180_QUAD4_SIP_ALL) && defined(K11220_LOCAL_SH_RED2)
                __gm__ bfloat16_t* sh_base = (rows > 16 && quad4 && (((static_cast<int32_t>(step.rank) ^ rank) & 1) != 0))
                    ? sh_twin_plane : sh_plane;
#elif defined(K11180_QUAD4_SIP_ALL)
                __gm__ bfloat16_t* sh_base = (quad4 && (((static_cast<int32_t>(step.rank) ^ rank) & 1) != 0))
                    ? sh_twin_plane : sh_plane;
#else
                __gm__ bfloat16_t* sh_base = sh_plane;
#endif
                sh_src.SetGlobalBuffer(sh_base + static_cast<int64_t>(step.rank) * rank_stride + sh_off, chunk);
            }
            AscendC::DataCopy(buffers.narrow[leaf * chunk], sh_src, chunk);
        }
#else
        uint32_t pending = unready;
        for (int32_t leaf = 0; leaf < ranks; ++leaf) {
            const TreeStep step = GetTreeStep(plan, leaf);
            if (static_cast<int32_t>(step.rank) == rank) {
                AscendC::GlobalTensor<bfloat16_t> sh_src;
#ifdef K11180_QUAD4_SIP_ALL
                if (quad4) {
                    sh_src.SetGlobalBuffer(sh_plane + static_cast<int64_t>(step.rank) * rank_stride + sh_off, chunk);
                } else {
                    sh_src.SetGlobalBuffer(shared_partial + sh_off, chunk);
                }
#else
                sh_src.SetGlobalBuffer(shared_partial + sh_off, chunk);
#endif
                AscendC::DataCopy(buffers.narrow[leaf * chunk], sh_src, chunk);
                pending &= ~(1U << leaf);
            } else if ((pending & (1U << leaf)) == 0U) {
                AscendC::GlobalTensor<bfloat16_t> sh_src;
#if defined(K11180_QUAD4_SIP_ALL) && defined(K11220_LOCAL_SH_RED2)
                __gm__ bfloat16_t* sh_base = (rows > 16 && quad4 && (((static_cast<int32_t>(step.rank) ^ rank) & 1) != 0))
                    ? sh_twin_plane : sh_plane;
#elif defined(K11180_QUAD4_SIP_ALL)
                __gm__ bfloat16_t* sh_base = (quad4 && (((static_cast<int32_t>(step.rank) ^ rank) & 1) != 0))
                    ? sh_twin_plane : sh_plane;
#else
                __gm__ bfloat16_t* sh_base = sh_plane;
#endif
                sh_src.SetGlobalBuffer(sh_base + static_cast<int64_t>(step.rank) * rank_stride + sh_off, chunk);
                AscendC::DataCopy(buffers.narrow[leaf * chunk], sh_src, chunk);
            }
        }
        while (pending != 0U) {
            for (int32_t l = 0; l < ranks; ++l) {
                const int32_t leaf = (l + static_cast<int32_t>(vector_index)) & (ranks - 1);
                if ((pending & (1U << leaf)) == 0U) {
                    continue;
                }
                const TreeStep step = GetTreeStep(plan, leaf);
                __gm__ volatile int32_t* sig_addr =
                    CommunicationReady(window, layout, ranks, sh_epoch, step.rank);
                dcci_cacheline((__gm__ uint8_t*)sig_addr);
                if (*sig_addr == sh_signal) {
                    AscendC::GlobalTensor<bfloat16_t> sh_src;
#if defined(K11180_QUAD4_SIP_ALL) && defined(K11220_LOCAL_SH_RED2)
                    __gm__ bfloat16_t* sh_base = (rows > 16 && quad4 && (((static_cast<int32_t>(step.rank) ^ rank) & 1) != 0))
                        ? sh_twin_plane : sh_plane;
#elif defined(K11180_QUAD4_SIP_ALL)
                    __gm__ bfloat16_t* sh_base = (quad4 && (((static_cast<int32_t>(step.rank) ^ rank) & 1) != 0))
                        ? sh_twin_plane : sh_plane;
#else
                    __gm__ bfloat16_t* sh_base = sh_plane;
#endif
                    sh_src.SetGlobalBuffer(sh_base + static_cast<int64_t>(step.rank) * rank_stride + sh_off, chunk);
                    AscendC::DataCopy(buffers.narrow[leaf * chunk], sh_src, chunk);
                    pending &= ~(1U << leaf);
                }
            }
        }
        unready = 0U;
#endif
        }
#if defined(ENABLE_KERNEL_PROF) && defined(K112824_SPLIT_PROF)
        if (first_tile && prof_acc != nullptr && vector_index == 0) { t_sh_done = AscendC::GetSystemCycle(); }
#endif
        CommunicationFence<AscendC::HardEvent::MTE2_V>();
#if defined(K112600_EXACT48_RED2_OPT) || defined(K112612_RED2_DEDUP)
        (void)owner_of_exp;
        if (first_tile) {
            CommunicationFence<AscendC::HardEvent::MTE2_S>();
        }
        int8_t slot_owner[8];
        uint32_t slot_pending = 0U;
        uint32_t wait_owners = 0U;
        for (int64_t slot = 0; slot < 8; ++slot) {
            const int64_t orig = local_tok * 8 + slot;
            const int32_t signed_eid = eids_ub.GetValue(orig);
            if (signed_eid < 0) {
                slot_owner[slot] = -1;
                continue;
            }
            const uint32_t ge = static_cast<uint32_t>(signed_eid) & 255U;
            int32_t owner;
            if (owner_gm != nullptr) {
                const uint32_t ow32 = owner_ub.GetValue(ge >> 2U);
                owner = static_cast<int32_t>((ow32 >> ((ge & 3U) << 3U)) & 0xFFU);
            } else {
                owner = static_cast<int32_t>(ge >> kExpertShift);
            }
            slot_owner[slot] = static_cast<int8_t>(owner);
            const uint32_t obit = 1U << static_cast<uint32_t>(owner);
            if ((rt_unready & obit) == 0U) {
                const int64_t elem_off = orig * kCommunicationHidden + col;
#ifdef K112808_PULL_RED2
                __gm__ bfloat16_t* src_base = (owner == rank) ? rt_plane
                    : reinterpret_cast<__gm__ bfloat16_t*>(aclshmem_ptr(rt_plane, owner));
                AscendC::GlobalTensor<bfloat16_t> source;
                source.SetGlobalBuffer(src_base + elem_off, chunk);
                AscendC::DataCopy(rt_narrow[slot * chunk], source, chunk);
#elif defined(K11150_SIP_RELAY_RED2)
                __gm__ bfloat16_t* src_base = (((owner ^ rank) & 1) == 0) ? rt_plane : rt_twin_plane;
                AscendC::GlobalTensor<bfloat16_t> source;
                source.SetGlobalBuffer(src_base + elem_off, chunk);
                AscendC::DataCopy(rt_narrow[slot * chunk], source, chunk);
#else
                AscendC::GlobalTensor<bfloat16_t> source;
                source.SetGlobalBuffer(((owner == rank) ? partial : rt_plane) + elem_off, chunk);
                AscendC::DataCopy(rt_narrow[slot * chunk], source, chunk);
#endif
            } else {
                slot_pending |= (1U << slot);
                wait_owners |= obit;
            }
        }
#ifdef K112612_OVERLAP_SH_P4
        if (pre_shared_sum != nullptr) {
            AscendC::Cast(buffers.accumulator[16384], buffers.narrow[0],
                          AscendC::RoundMode::CAST_NONE, chunk);
        } else
#endif
        {
            AscendC::Cast(buffers.accumulator[0], buffers.narrow[0],
                          AscendC::RoundMode::CAST_NONE, ranks * chunk);
            AscendC::PipeBarrier<PIPE_V>();
            for (int32_t stride = 1; stride < ranks; stride <<= 1) {
                for (int32_t i = 0; i + stride < ranks; i += (stride << 1)) {
                    AscendC::Add(buffers.accumulator[i * chunk],
                                 buffers.accumulator[i * chunk],
                                 buffers.accumulator[(i + stride) * chunk], chunk);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Cast(buffers.narrow[0], buffers.accumulator[0],
                          AscendC::RoundMode::CAST_RINT, chunk);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(buffers.accumulator[16384], buffers.narrow[0],
                          AscendC::RoundMode::CAST_NONE, chunk);
        }
        if (first_tile && fused_next_norm && static_cast<int64_t>(tid) < rows) {
            AscendC::GlobalTensor<bfloat16_t> res_gm, gam_gm;
            res_gm.SetGlobalBuffer(
                const_cast<__gm__ bfloat16_t*>(next_residual) + static_cast<int64_t>(tid) * 6144,
                6144);
            gam_gm.SetGlobalBuffer(const_cast<__gm__ bfloat16_t*>(next_gamma), 6144);
            AscendC::DataCopy(
                buffers.accumulator[21504].ReinterpretCast<bfloat16_t>(), res_gm, 6144);
            AscendC::DataCopy(
                buffers.accumulator[24576].ReinterpretCast<bfloat16_t>(), gam_gm, 6144);
        }
#ifdef K112830_FAST_POLL_RED2
        while (wait_owners != 0U) {
            if ((wait_owners & rt_unready) != 0U) {
                uint32_t rt_chk = rt_unready;
                refresh_sigs(rt_chk, rt_epoch, signal);
                rt_unready = rt_chk;
            }
            const uint32_t issue_owners = wait_owners & ~rt_unready;
            if (issue_owners == 0U) {
                continue;
            }
            for (int64_t slot = 0; slot < 8; ++slot) {
                if ((slot_pending & (1U << slot)) != 0U) {
                    const int32_t owner = static_cast<int32_t>(slot_owner[slot]);
                    if ((issue_owners & (1U << static_cast<uint32_t>(owner))) != 0U) {
#ifdef K112808_PULL_RED2
                        __gm__ bfloat16_t* src_base = (owner == rank) ? rt_plane : reinterpret_cast<__gm__ bfloat16_t*>(aclshmem_ptr(rt_plane, owner));
#elif defined(K11150_SIP_RELAY_RED2)
                        __gm__ bfloat16_t* src_base = (((owner ^ rank) & 1) == 0) ? rt_plane : rt_twin_plane;
#else
                        __gm__ bfloat16_t* src_base = (owner == rank) ? partial : rt_plane;
#endif
                        const int64_t elem_off = (local_tok * 8 + slot) * kCommunicationHidden + col;
                        AscendC::GlobalTensor<bfloat16_t> source;
                        source.SetGlobalBuffer(src_base + elem_off, chunk);
                        AscendC::DataCopy(rt_narrow[slot * chunk], source, chunk);
                        slot_pending &= ~(1U << slot);
                    }
                }
            }
            wait_owners &= ~issue_owners;
        }
#else
        while (wait_owners != 0U) {
            for (int32_t o = 0; o < ranks; ++o) {
                const int32_t owner = (o + static_cast<int32_t>(tid)) & (ranks - 1);
                const uint32_t obit = 1U << static_cast<uint32_t>(owner);
                if ((wait_owners & obit) == 0U) {
                    continue;
                }
                if ((rt_unready & obit) != 0U) {
                    __gm__ volatile int32_t* sig_addr =
                        CommunicationReady(window, layout, ranks, rt_epoch, owner);
                    dcci_cacheline((__gm__ uint8_t*)sig_addr);
                    if (*sig_addr != signal) {
                        continue;
                    }
                    rt_unready &= ~obit;
                }
                wait_owners &= ~obit;
#ifdef K112808_PULL_RED2
                __gm__ bfloat16_t* src_base = (static_cast<int32_t>(owner) == rank) ? rt_plane : reinterpret_cast<__gm__ bfloat16_t*>(aclshmem_ptr(rt_plane, static_cast<int32_t>(owner)));
#elif defined(K11150_SIP_RELAY_RED2)
                __gm__ bfloat16_t* src_base = (((owner ^ rank) & 1) == 0) ? rt_plane : rt_twin_plane;
#else
                __gm__ bfloat16_t* src_base = rt_plane;
#endif
                for (int64_t slot = 0; slot < 8; ++slot) {
                    if (((slot_pending & (1U << slot)) != 0U) && slot_owner[slot] == static_cast<int8_t>(owner)) {
                        const int64_t elem_off = (local_tok * 8 + slot) * kCommunicationHidden + col;
                        AscendC::GlobalTensor<bfloat16_t> source;
                        source.SetGlobalBuffer(src_base + elem_off, chunk);
                        AscendC::DataCopy(rt_narrow[slot * chunk], source, chunk);
                        slot_pending &= ~(1U << slot);
                    }
                }
            }
        }
#endif
#if defined(ENABLE_KERNEL_PROF) && defined(K112824_SPLIT_PROF)
        (void)t_r2_arr;
        if (first_tile && prof_acc != nullptr && vector_index == 0) {
            t_r2_arr = AscendC::GetSystemCycle();
        }
#elif defined(ENABLE_KERNEL_PROF)
        if (first_tile && prof_acc != nullptr && vector_index == 0) {
            t_r2_arr = AscendC::GetSystemCycle();
        }
#endif
#else
        if (first_tile) {
            CommunicationFence<AscendC::HardEvent::MTE2_S>();
            if (owner_gm != nullptr) {
#ifdef K11180_LITE_MIRROR
                const uint64_t packed_mge = owner_ub.template ReinterpretCast<uint64_t>().GetValue(6);
                FillLiteOwners(eids_ub, static_cast<int32_t>(rows * 8), packed_mge, owner_of_exp);
#else
                uint32_t* own_u32 = reinterpret_cast<uint32_t*>(owner_of_exp);
                for (uint32_t w = 0U; w < 64U; ++w) {
                    own_u32[w] = owner_ub.GetValue(w);
                }
#endif
            }
        }
        uint32_t slot_pending = 0U;
        for (int64_t slot = 0; slot < 8; ++slot) {
            const int64_t orig = local_tok * 8 + slot;
            const int32_t signed_eid = eids_ub.GetValue(orig);
            if (signed_eid < 0) {
                continue;
            }
            const uint32_t ge = static_cast<uint32_t>(signed_eid) & 255U;
            const int32_t owner = (owner_gm != nullptr)
                ? static_cast<int32_t>(owner_of_exp[ge])
                : static_cast<int32_t>(ge >> kExpertShift);
            const int64_t elem_off = orig * kCommunicationHidden + col;
#ifdef K112808_PULL_RED2
            if ((rt_unready & (1U << owner)) == 0U) {
                __gm__ bfloat16_t* src_ptr = (owner == rank) ? (rt_plane + elem_off) : reinterpret_cast<__gm__ bfloat16_t*>(aclshmem_ptr(rt_plane + elem_off, owner));
                AscendC::GlobalTensor<bfloat16_t> source;
                source.SetGlobalBuffer(src_ptr, chunk);
                AscendC::DataCopy(rt_narrow[slot * chunk], source, chunk);
            } else {
                slot_pending |= (1U << slot);
            }
#elif defined(K11150_SIP_RELAY_RED2)
            if ((rt_unready & (1U << owner)) == 0U) {
                __gm__ bfloat16_t* src_base = (((owner ^ rank) & 1) == 0) ? rt_plane : rt_twin_plane;
                AscendC::GlobalTensor<bfloat16_t> source;
                source.SetGlobalBuffer(src_base + elem_off, chunk);
                AscendC::DataCopy(rt_narrow[slot * chunk], source, chunk);
            } else {
                slot_pending |= (1U << slot);
            }
#else
            if (owner == rank) {
                AscendC::GlobalTensor<bfloat16_t> source;
                source.SetGlobalBuffer(partial + elem_off, chunk);
                AscendC::DataCopy(rt_narrow[slot * chunk], source, chunk);
            } else if ((rt_unready & (1U << owner)) == 0U) {
                AscendC::GlobalTensor<bfloat16_t> source;
                source.SetGlobalBuffer(rt_plane + elem_off, chunk);
                AscendC::DataCopy(rt_narrow[slot * chunk], source, chunk);
            } else {
                slot_pending |= (1U << slot);
            }
#endif
        }
        AscendC::Cast(buffers.accumulator[0], buffers.narrow[0],
                      AscendC::RoundMode::CAST_NONE, ranks * chunk);
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t stride = 1; stride < ranks; stride <<= 1) {
            for (int32_t i = 0; i + stride < ranks; i += (stride << 1)) {
                AscendC::Add(buffers.accumulator[i * chunk],
                             buffers.accumulator[i * chunk],
                             buffers.accumulator[(i + stride) * chunk], chunk);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Cast(buffers.narrow[0], buffers.accumulator[0],
                      AscendC::RoundMode::CAST_RINT, chunk);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(buffers.accumulator[16384], buffers.narrow[0],
                      AscendC::RoundMode::CAST_NONE, chunk);
        while (slot_pending != 0U) {
            for (int64_t s = 0; s < 8; ++s) {
                const int64_t slot = (s + vector_index) & 7;
                if ((slot_pending & (1U << slot)) == 0U) {
                    continue;
                }
                const int64_t orig = local_tok * 8 + slot;
                const uint32_t ge = static_cast<uint32_t>(eids_ub.GetValue(orig)) & 255U;
                const int32_t owner = (owner_gm != nullptr)
                    ? static_cast<int32_t>(owner_of_exp[ge])
                    : static_cast<int32_t>(ge >> kExpertShift);
                if ((rt_unready & (1U << owner)) != 0U) {
                    __gm__ volatile int32_t* sig_addr =
                        CommunicationReady(window, layout, ranks, rt_epoch, owner);
                    dcci_cacheline((__gm__ uint8_t*)sig_addr);
                    if (*sig_addr != signal) {
                        continue;
                    }
                    rt_unready &= ~(1U << owner);
                }
                const int64_t elem_off = orig * kCommunicationHidden + col;
                AscendC::GlobalTensor<bfloat16_t> source;
#ifdef K112808_PULL_RED2
                __gm__ bfloat16_t* src_ptr = (owner == rank) ? (rt_plane + elem_off) : reinterpret_cast<__gm__ bfloat16_t*>(aclshmem_ptr(rt_plane + elem_off, owner));
                source.SetGlobalBuffer(src_ptr, chunk);
#elif defined(K11150_SIP_RELAY_RED2)
                __gm__ bfloat16_t* src_base = (((owner ^ rank) & 1) == 0) ? rt_plane : rt_twin_plane;
                source.SetGlobalBuffer(src_base + elem_off, chunk);
#else
                source.SetGlobalBuffer(rt_plane + elem_off, chunk);
#endif
                AscendC::DataCopy(rt_narrow[slot * chunk], source, chunk);
                slot_pending &= ~(1U << slot);
            }
        }
#endif
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
#if defined(ENABLE_KERNEL_PROF) && defined(K112824_SPLIT_PROF)
        (void)t_r2_arr;
        if (first_tile && prof_acc != nullptr && vector_index == 0) {
            t_r2_arr = AscendC::GetSystemCycle();
        }
#elif defined(ENABLE_KERNEL_PROF)
        if (first_tile && prof_acc != nullptr && vector_index == 0) {
            t_r2_arr = AscendC::GetSystemCycle();
        }
#endif
        AscendC::PipeBarrier<PIPE_V>();
        const int64_t active_slots = (eids_ub.GetValue(local_tok * 8 + 4) < 0) ? 4 : 8;
        AscendC::Cast(buffers.accumulator[0], rt_narrow[0],
                      AscendC::RoundMode::CAST_NONE, active_slots * chunk);
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t slot = 0; slot < active_slots; ++slot) {
            const float prob = probs_ub.GetValue(local_tok * 8 + slot);
            AscendC::Muls(buffers.accumulator[slot * chunk],
                          buffers.accumulator[slot * chunk], prob, chunk);
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t slot = 1; slot < active_slots; ++slot) {
            AscendC::Add(buffers.accumulator[0], buffers.accumulator[0],
                         buffers.accumulator[slot * chunk], chunk);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Cast(rt_narrow[0], buffers.accumulator[0],
                      AscendC::RoundMode::CAST_RINT, chunk);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(buffers.accumulator[0], rt_narrow[0],
                      AscendC::RoundMode::CAST_NONE, chunk);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(buffers.accumulator[0], buffers.accumulator[0],
                      2.5F, chunk);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(rt_narrow[0], buffers.accumulator[0],
                      AscendC::RoundMode::CAST_RINT, chunk);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(buffers.accumulator[0], rt_narrow[0],
                      AscendC::RoundMode::CAST_NONE, chunk);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(buffers.accumulator[0], buffers.accumulator[0],
                     buffers.accumulator[16384], chunk);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(buffers.narrow[0], buffers.accumulator[0],
                      AscendC::RoundMode::CAST_RINT, chunk);
        CommunicationFence<AscendC::HardEvent::V_MTE3>();
        AscendC::GlobalTensor<bfloat16_t> destination;
        destination.SetGlobalBuffer(output + sh_off, chunk);
        AscendC::DataCopy(destination, buffers.narrow[0], chunk);
    }
#if defined(ENABLE_KERNEL_PROF) && defined(K112824_SPLIT_PROF)
    if (prof_acc != nullptr && vector_index == 0) {
        const int64_t t_r2_end = AscendC::GetSystemCycle();
        const int64_t c_sh = t_sh_done - t_r2_enter;
        const int64_t c_rt = t_r2_arr - t_sh_done;
        prof_acc[22] += (c_sh << 32) + c_rt;
        prof_acc[23] += (t_r2_end - t_r2_arr);
    }
#elif defined(ENABLE_KERNEL_PROF)
    if (prof_acc != nullptr && vector_index == 0) {
        const int64_t t_r2_end = AscendC::GetSystemCycle();
        prof_acc[22] += (t_r2_arr - t_r2_enter);
        prof_acc[23] += (t_r2_end - t_r2_arr);
    }
#endif
}

#if defined(K11220_RS768_PUSH) || defined(K112600_RS768_HIW)
__aicore__ inline void MapRs768Tile(
    uint32_t tid, int64_t wave_rows, int64_t& rel_tok, int64_t& sub_col, int64_t& chunk)
{
    if (wave_rows <= 2) {
        chunk = 32;
        const uint32_t q = (tid * 683U) >> 14U;
        rel_tok = static_cast<int64_t>(q);
        sub_col = static_cast<int64_t>(tid - q * 24U) * 32;
    } else if (wave_rows <= 4) {
        chunk = 64;
        const uint32_t q = (tid * 683U) >> 13U;
        rel_tok = static_cast<int64_t>(q);
        sub_col = static_cast<int64_t>(tid - q * 12U) * 64;
    } else if (wave_rows <= 8) {
        chunk = 128;
        const uint32_t q = (tid * 683U) >> 12U;
        rel_tok = static_cast<int64_t>(q);
        sub_col = static_cast<int64_t>(tid - q * 6U) * 128;
    } else if (wave_rows <= 12) {
        chunk = 192;
        rel_tok = static_cast<int64_t>(tid >> 2U);
        sub_col = static_cast<int64_t>(tid & 3U) * 192;
    } else {
        chunk = 256;
        const uint32_t q = (tid * 171U) >> 9U;
        rel_tok = static_cast<int64_t>(q);
        sub_col = static_cast<int64_t>(tid - q * 3U) * 256;
    }
}

__aicore__ inline void ReduceScatterSlots768_Push(
    CommunicationBuffers& buffers, __gm__ uint8_t* window,
    const CommunicationLayout& layout, int64_t rs_qcap,
    __gm__ bfloat16_t* output, int64_t rows,
    int32_t rank, int32_t ranks, uint32_t sh_epoch, uint32_t rt_epoch,
    int64_t vector_index, __gm__ const int32_t* expert_ids,
    __gm__ const float* probabilities, const __gm__ uint8_t* owner_gm, bool quad4)
{
    if (rows <= 0) {
        return;
    }
    const uint32_t tid = static_cast<uint32_t>(vector_index);
    const int64_t wave0_rows = (rows > 16) ? 16 : rows;
    int64_t rel_tok = 0;
    int64_t sub_col = 0;
    int64_t chunk = 256;
    MapRs768Tile(tid, wave0_rows, rel_tok, sub_col, chunk);
    int64_t local_tok = rel_tok;
    if (local_tok >= wave0_rows) {
        return;
    }

    const int32_t sh_signal = static_cast<int32_t>(sh_epoch);
    const int32_t signal = static_cast<int32_t>(rt_epoch);
    AscendC::GlobalTensor<int32_t> eids_gm;
    AscendC::GlobalTensor<float> probs_gm;
    eids_gm.SetGlobalBuffer(const_cast<__gm__ int32_t*>(expert_ids), rows * 8);
    probs_gm.SetGlobalBuffer(const_cast<__gm__ float*>(probabilities), rows * 8);
    auto eids_ub = buffers.accumulator[28 * 1024].ReinterpretCast<int32_t>();
    auto probs_ub = buffers.accumulator[29 * 1024];
    auto owner_ub = buffers.accumulator[30 * 1024].ReinterpretCast<uint32_t>();
    AscendC::DataCopy(eids_ub, eids_gm, rows * 8);
    AscendC::DataCopy(probs_ub, probs_gm, rows * 8);
    if (owner_gm != nullptr) {
        AscendC::GlobalTensor<uint32_t> own_src;
        own_src.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(const_cast<__gm__ uint8_t*>(owner_gm)), 64);
        AscendC::DataCopy(owner_ub, own_src, 64);
    }

    const TreePlan plan = GetTreePlan(ranks);
    const uint32_t p_mask = quad4 ? 3U : 1U;
    __gm__ bfloat16_t* sh_plane = reinterpret_cast<__gm__ bfloat16_t*>(
        window + layout.payload + (sh_epoch & p_mask) * layout.plane_bytes);
    __gm__ bfloat16_t* rt_plane = reinterpret_cast<__gm__ bfloat16_t*>(
        window + layout.payload + (rt_epoch & p_mask) * layout.plane_bytes);
    const int64_t rs_stride = rs_qcap * 768;
    __gm__ bfloat16_t* ag_self_base = rt_plane + (8 + static_cast<int64_t>(rank)) * rs_stride;

    auto rt_narrow = buffers.narrow[16384];
    uint32_t sh_unready = 0xFFU & ~(1U << static_cast<uint32_t>(rank));
    uint32_t rt_unready = 0xFFU & ~(1U << static_cast<uint32_t>(rank));
    uint64_t own_u64[32];
    uint8_t* owner_of_exp = reinterpret_cast<uint8_t*>(own_u64);

    for (uint32_t wave = 0U; wave < 2U; ++wave) {
        if (wave == 1U) {
            if (rows <= 16) {
                break;
            }
            const int64_t wave1_rows = rows - 16;
            MapRs768Tile(tid, wave1_rows, rel_tok, sub_col, chunk);
            if (rel_tok >= wave1_rows) {
                break;
            }
            local_tok = 16 + rel_tok;
            CommunicationFence<AscendC::HardEvent::MTE3_MTE2>();
            CommunicationFence<AscendC::HardEvent::V_MTE2>();
        }

        const int64_t tok_off = local_tok * 768 + sub_col;
        uint32_t pending = 0U;
        for (int32_t leaf = 0; leaf < 8; ++leaf) {
            const TreeStep step = GetTreeStep(plan, leaf);
            const uint32_t srank = static_cast<uint32_t>(step.rank);
            if ((sh_unready & (1U << srank)) == 0U) {
                AscendC::GlobalTensor<bfloat16_t> sh_src;
                sh_src.SetGlobalBuffer(sh_plane + static_cast<int64_t>(srank) * rs_stride + tok_off, chunk);
                AscendC::DataCopy(buffers.narrow[leaf * chunk], sh_src, chunk);
            } else {
                pending |= (1U << leaf);
            }
        }
        while (pending != 0U) {
            for (int32_t l = 0; l < 8; ++l) {
                const int32_t leaf = (l + static_cast<int32_t>(tid)) & 7;
                if ((pending & (1U << leaf)) == 0U) {
                    continue;
                }
                const TreeStep step = GetTreeStep(plan, leaf);
                const uint32_t srank = static_cast<uint32_t>(step.rank);
                if ((sh_unready & (1U << srank)) != 0U) {
                    __gm__ volatile int32_t* sig_addr =
                        CommunicationReady(window, layout, ranks, sh_epoch, step.rank);
                    dcci_cacheline((__gm__ uint8_t*)sig_addr);
                    if (*sig_addr != sh_signal) {
                        continue;
                    }
                    sh_unready &= ~(1U << srank);
                }
                AscendC::GlobalTensor<bfloat16_t> sh_src;
                sh_src.SetGlobalBuffer(sh_plane + static_cast<int64_t>(srank) * rs_stride + tok_off, chunk);
                AscendC::DataCopy(buffers.narrow[leaf * chunk], sh_src, chunk);
                pending &= ~(1U << leaf);
            }
        }
        CommunicationFence<AscendC::HardEvent::MTE2_V>();
        (void)own_u64;
        (void)owner_of_exp;
        if (wave == 0U) {
            CommunicationFence<AscendC::HardEvent::MTE2_S>();
        }

        int8_t slot_owner[8];
        uint32_t slot_pending = 0U;
        uint32_t wait_owners = 0U;
        for (int64_t slot = 0; slot < 8; ++slot) {
            const int64_t orig = local_tok * 8 + slot;
            const int32_t signed_eid = eids_ub.GetValue(orig);
            if (signed_eid < 0) {
                slot_owner[slot] = -1;
                continue;
            }
            const uint32_t ge = static_cast<uint32_t>(signed_eid) & 255U;
            uint32_t owner;
            if (owner_gm != nullptr) {
                const uint32_t ow32 = owner_ub.GetValue(ge >> 2U);
                owner = (ow32 >> ((ge & 3U) << 3U)) & 0xFFU;
            } else {
                owner = ge >> kExpertShift;
            }
            slot_owner[slot] = static_cast<int8_t>(owner);
            const uint32_t obit = 1U << owner;
            if ((rt_unready & obit) == 0U) {
                AscendC::GlobalTensor<bfloat16_t> source;
                source.SetGlobalBuffer(rt_plane + orig * 768 + sub_col, chunk);
                AscendC::DataCopy(rt_narrow[slot * chunk], source, chunk);
            } else {
                slot_pending |= (1U << slot);
                wait_owners |= obit;
            }
        }

        AscendC::Cast(buffers.accumulator[0], buffers.narrow[0],
                      AscendC::RoundMode::CAST_NONE, ranks * chunk);
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t stride = 1; stride < ranks; stride <<= 1) {
            for (int32_t i = 0; i + stride < ranks; i += (stride << 1)) {
                AscendC::Add(buffers.accumulator[i * chunk],
                             buffers.accumulator[i * chunk],
                             buffers.accumulator[(i + stride) * chunk], chunk);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Cast(buffers.narrow[0], buffers.accumulator[0],
                      AscendC::RoundMode::CAST_RINT, chunk);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(buffers.accumulator[16384], buffers.narrow[0],
                      AscendC::RoundMode::CAST_NONE, chunk);

        while (wait_owners != 0U) {
            for (uint32_t o = 0U; o < 8U; ++o) {
                const uint32_t owner = (o + tid) & 7U;
                const uint32_t obit = 1U << owner;
                if ((wait_owners & obit) == 0U) {
                    continue;
                }
                if ((rt_unready & obit) != 0U) {
                    __gm__ volatile int32_t* sig_addr =
                        CommunicationReady(window, layout, ranks, rt_epoch, static_cast<int32_t>(owner));
                    dcci_cacheline((__gm__ uint8_t*)sig_addr);
                    if (*sig_addr != signal) {
                        continue;
                    }
                    rt_unready &= ~obit;
                }
                wait_owners &= ~obit;
                for (int64_t slot = 0; slot < 8; ++slot) {
                    if (((slot_pending & (1U << slot)) != 0U) && slot_owner[slot] == static_cast<int8_t>(owner)) {
                        AscendC::GlobalTensor<bfloat16_t> source;
                        source.SetGlobalBuffer(rt_plane + (local_tok * 8 + slot) * 768 + sub_col, chunk);
                        AscendC::DataCopy(rt_narrow[slot * chunk], source, chunk);
                        slot_pending &= ~(1U << slot);
                    }
                }
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::PipeBarrier<PIPE_V>();

        const int64_t active_slots = (eids_ub.GetValue(local_tok * 8 + 4) < 0) ? 4 : 8;
        AscendC::Cast(buffers.accumulator[0], rt_narrow[0],
                      AscendC::RoundMode::CAST_NONE, active_slots * chunk);
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t slot = 0; slot < active_slots; ++slot) {
            const float prob = probs_ub.GetValue(local_tok * 8 + slot);
            AscendC::Muls(buffers.accumulator[slot * chunk],
                          buffers.accumulator[slot * chunk], prob, chunk);
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t slot = 1; slot < active_slots; ++slot) {
            AscendC::Add(buffers.accumulator[0], buffers.accumulator[0],
                         buffers.accumulator[slot * chunk], chunk);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Cast(rt_narrow[0], buffers.accumulator[0],
                      AscendC::RoundMode::CAST_RINT, chunk);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(buffers.accumulator[0], rt_narrow[0],
                      AscendC::RoundMode::CAST_NONE, chunk);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(buffers.accumulator[0], buffers.accumulator[0],
                      2.5F, chunk);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(rt_narrow[0], buffers.accumulator[0],
                      AscendC::RoundMode::CAST_RINT, chunk);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(buffers.accumulator[0], rt_narrow[0],
                      AscendC::RoundMode::CAST_NONE, chunk);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(buffers.accumulator[0], buffers.accumulator[0],
                     buffers.accumulator[16384], chunk);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(buffers.narrow[0], buffers.accumulator[0],
                      AscendC::RoundMode::CAST_RINT, chunk);
        CommunicationFence<AscendC::HardEvent::V_MTE3>();

        AscendC::GlobalTensor<bfloat16_t> out_self_dst;
        out_self_dst.SetGlobalBuffer(output + local_tok * kCommunicationHidden + static_cast<int64_t>(rank) * 768 + sub_col, chunk);
        AscendC::DataCopy(out_self_dst, buffers.narrow[0], chunk);

        __gm__ bfloat16_t* my_ag_slot = ag_self_base + tok_off;
#ifdef K112600_RS768_HIW
        AscendC::GlobalTensor<bfloat16_t> self_ag_dst;
        self_ag_dst.SetGlobalBuffer(my_ag_slot, chunk);
        AscendC::DataCopy(self_ag_dst, buffers.narrow[0], chunk);
        for (uint32_t p = 0U; p < 3U; ++p) {
            const int32_t peer = (rank + static_cast<int32_t>((((tid + p) % 3U) + 1U) << 1U)) & 7;
            AscendC::GlobalTensor<bfloat16_t> peer_dst;
            peer_dst.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t*>(aclshmem_ptr(my_ag_slot, peer)), chunk);
            AscendC::DataCopy(peer_dst, buffers.narrow[0], chunk);
        }
#else
        for (uint32_t p = 0U; p < 7U; ++p) {
            const uint32_t off = ((tid + p) % 7U) + 1U;
            const int32_t peer = (rank + static_cast<int32_t>(off)) & 7;
            AscendC::GlobalTensor<bfloat16_t> peer_dst;
            peer_dst.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t*>(aclshmem_ptr(my_ag_slot, peer)), chunk);
            AscendC::DataCopy(peer_dst, buffers.narrow[0], chunk);
        }
#endif
    }
}

__aicore__ inline void AllGatherRecv768(
    CommunicationBuffers& buffers, __gm__ uint8_t* window,
    const CommunicationLayout& layout, int64_t rs_qcap,
    __gm__ bfloat16_t* output, int64_t rows,
    int32_t rank, int32_t ranks, uint32_t rt_epoch,
    int64_t vector_index, bool quad4)
{
    if (rows <= 0) {
        return;
    }
    (void)ranks;
    const uint32_t tid = static_cast<uint32_t>(vector_index);
    if (tid > 0U && tid < 8U) {
        const int32_t sig_peer = (rank + static_cast<int32_t>(tid)) & 7;
        SignalAgReady768(window, layout, rt_epoch, rank, sig_peer);
    }
    if (tid >= 42U) {
        return;
    }
    const uint32_t p_idx = (tid * 171U) >> 10U;
    const uint32_t seg = tid - p_idx * 6U;
    const int32_t peer = (rank + 1 + static_cast<int32_t>(p_idx)) & 7;
    const uint32_t u_rows = static_cast<uint32_t>(rows);
    const uint32_t q_r = (u_rows * 683U) >> 12U;
    const uint32_t rem_r = u_rows - q_r * 6U;
    const int64_t r_beg = static_cast<int64_t>(seg * q_r + ((seg < rem_r) ? seg : rem_r));
    const int64_t r_cnt = static_cast<int64_t>(q_r + ((seg < rem_r) ? 1U : 0U));
    if (r_cnt <= 0) {
        return;
    }

    const uint32_t p_mask = quad4 ? 3U : 1U;
    __gm__ bfloat16_t* rt_plane = reinterpret_cast<__gm__ bfloat16_t*>(
        window + layout.payload + (rt_epoch & p_mask) * layout.plane_bytes);
    const int64_t ag_stride = rs_qcap * 768;
    __gm__ bfloat16_t* ag_base = rt_plane + 8 * ag_stride;

    const int32_t signal = static_cast<int32_t>(rt_epoch);
    __gm__ volatile int32_t* sig_addr = AgReady768(window, layout, rt_epoch, peer);
    for (;;) {
        dcci_cacheline((__gm__ uint8_t*)sig_addr);
        if (*sig_addr == signal) {
            break;
        }
    }
    const int64_t elems = r_cnt * 768;
    AscendC::GlobalTensor<bfloat16_t> peer_local_src;
#ifdef K112600_RS768_HIW
    __gm__ bfloat16_t* ag_src_base = (((peer ^ rank) & 1) == 0)
        ? ag_base
        : reinterpret_cast<__gm__ bfloat16_t*>(aclshmem_ptr(ag_base, rank ^ 1));
    peer_local_src.SetGlobalBuffer(ag_src_base + static_cast<int64_t>(peer) * ag_stride + r_beg * 768, elems);
#else
    peer_local_src.SetGlobalBuffer(ag_base + static_cast<int64_t>(peer) * ag_stride + r_beg * 768, elems);
#endif
    AscendC::DataCopy(buffers.narrow[0], peer_local_src, elems);
    CommunicationFence<AscendC::HardEvent::MTE2_MTE3>();
    AscendC::GlobalTensor<bfloat16_t> out_dst;
    out_dst.SetGlobalBuffer(
        output + r_beg * kCommunicationHidden + static_cast<int64_t>(peer) * 768,
        r_cnt * kCommunicationHidden);
    AscendC::DataCopy(
        out_dst, buffers.narrow[0],
        AscendC::DataCopyParams{static_cast<uint16_t>(r_cnt), 48, 0, 336});
}
#endif
