#ifndef TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H
#define TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H

#include "communication.h"
#include "kernel_operator.h"
#include "shmem.h"
#include "tree_plan.h"

namespace whole_layer_tree {

class CommunicationBuffers {
 public:
    __aicore__ inline void Init(AscendC::TPipe& pipe)
    {
        pipe.InitBuffer(narrow_buffer_, 4096 * 2 * 8);
        pipe.InitBuffer(accumulator_buffer_, 4096 * 4 * 8);
        narrow = narrow_buffer_.Get<bfloat16_t>();
        accumulator = accumulator_buffer_.Get<float>();
        term = accumulator[kCommunicationChunk];
        push_scratch = accumulator_buffer_.Get<bfloat16_t>();
    }

    AscendC::LocalTensor<bfloat16_t> narrow;
    AscendC::LocalTensor<float> accumulator;
    AscendC::LocalTensor<float> term;
    AscendC::LocalTensor<bfloat16_t> push_scratch;

 private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> narrow_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumulator_buffer_;
};

template <AscendC::HardEvent event>
__aicore__ inline void CommunicationFence()
{
    AscendC::SetFlag<event>(EVENT_ID0);
    AscendC::WaitFlag<event>(EVENT_ID0);
}

__aicore__ inline void CommunicationPublish()
{
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
}

__aicore__ inline uint32_t ReadCommunicationEpoch(__gm__ uint8_t* window)
{
    AscendC::GlobalTensor<uint32_t> counter;
    counter.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(window), 1);
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                     AscendC::DcciDst::CACHELINE_OUT>(counter);
    return counter.GetValue(0);
}

__aicore__ inline __gm__ int32_t* CommunicationReady(
    __gm__ uint8_t* window, const CommunicationLayout& layout,
    int32_t ranks, uint32_t epoch, int32_t sender)
{
#if defined(K11180_QUAD4_SIP_ALL) || defined(K112110_SIP_RED1)
    (void)ranks;
    return reinterpret_cast<__gm__ int32_t*>(
        window + layout.ready + static_cast<uint64_t>(sender) * kCommunicationCacheLine)
        + (epoch & 3U);
#else
    return reinterpret_cast<__gm__ int32_t*>(
        window + layout.ready
        + ((epoch & 1U) * ranks + sender) * kCommunicationCacheLine);
#endif
}

__aicore__ inline __gm__ int32_t* CommunicationR1Done(
    __gm__ uint8_t* window, const CommunicationLayout& layout, int32_t ranks, int32_t sender)
{
    return reinterpret_cast<__gm__ int32_t*>(
        window + layout.ready + static_cast<uint64_t>(ranks + sender) * kCommunicationCacheLine);
}

__aicore__ inline void SignalPeerReady(
    __gm__ uint8_t* window, const CommunicationLayout& layout,
    int32_t ranks, uint32_t epoch, int32_t rank, int32_t peer)
{
    __gm__ int32_t* remote_sig = reinterpret_cast<__gm__ int32_t*>(
        aclshmem_ptr(CommunicationReady(window, layout, ranks, epoch, rank), peer));
    *remote_sig = static_cast<int32_t>(epoch);
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(remote_sig));
}

__aicore__ inline constexpr int64_t RsQCap(int64_t capacity, int64_t rows, bool quad4)
{
    if (!quad4) {
        return capacity;
    }
    const int64_t half_cap = capacity >> 1;
    return (rows <= 4 && half_cap >= 4) ? 4
        : ((rows <= 8 && half_cap >= 8) ? 8 : half_cap);
}

#if defined(K11220_RS768_PUSH) || defined(K112600_RS768_HIW)
__aicore__ inline __gm__ int32_t* AgReady768(
    __gm__ uint8_t* window, const CommunicationLayout& layout,
    uint32_t epoch, int32_t sender)
{
    return reinterpret_cast<__gm__ int32_t*>(
        window + layout.ready + static_cast<uint64_t>(8 + sender) * kCommunicationCacheLine)
        + (epoch & 15U);
}

__aicore__ inline void SignalAgReady768(
    __gm__ uint8_t* window, const CommunicationLayout& layout,
    uint32_t epoch, int32_t rank, int32_t peer)
{
    __gm__ int32_t* remote_sig = reinterpret_cast<__gm__ int32_t*>(
        aclshmem_ptr(AgReady768(window, layout, epoch, rank), peer));
    *remote_sig = static_cast<int32_t>(epoch);
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(remote_sig));
}
#endif

struct LayerMirrorMeta {
    uint64_t cand_ge_pack[4];
    uint64_t cand_mask_pack[4];
    uint64_t mirror_weights[6];
    uint64_t my_ge_pack;
    uint64_t header_u64;
    uint64_t pad0[8];
    uint64_t cand_ge_ext[4];
    uint64_t cand_mask_ext[4];
    uint64_t pad[48];
};
static_assert(sizeof(LayerMirrorMeta) == 640);

#ifdef K11220_FAST_MIRROR_V4
__aicore__ inline uint32_t RankWaveCost(uint32_t blk, uint32_t extra)
{
#ifdef K11220_EXACT_COST
    const uint32_t gu_steps = (blk == 0U) ? 0U : (((2U * blk + 2U) * 683U) >> 11U);
    return gu_steps * 192U + blk * 64U + extra;
#elif defined(K11220_WAVE6_COST)
    const uint32_t w6 = (blk == 0U) ? 0U : (((blk + 5U) * 683U) >> 12U);
    return w6 * 384U + blk * 256U + extra;
#else
    return blk * 256U + extra;
#endif
}

__aicore__ inline uint32_t AssignExpertOwnersFastV4(
    const int32_t* raw_r, int32_t routes, uint32_t num_cands,
    const uint64_t* cand_ge_u64, const uint64_t* cand_mask_u64,
    uint8_t* exp_cnt, uint8_t* act_ge, uint8_t* act_own)
{
    uint64_t* cnt_u64 = reinterpret_cast<uint64_t*>(exp_cnt);
    for (uint32_t w = 0U; w < 32U; ++w) {
        cnt_u64[w] = 0ULL;
    }
    constexpr uint32_t kRanks = static_cast<uint32_t>(kDefaultWorldSize);
    constexpr uint32_t kMaxLocalExp = 256U / kRanks;
    uint32_t r_ecnt[kRanks] = {};
    uint32_t r_rcnt[kRanks] = {};
    uint32_t r_blk[kRanks] = {};
    uint32_t r_extra[kRanks] = {};
    for (int32_t i = 0; i < routes; ++i) {
        const int32_t r_val = raw_r[i];
        if (r_val >= 0) {
            const uint32_t ge = static_cast<uint32_t>(r_val) & 255U;
            const uint32_t hr = ge >> kExpertShift;
            const uint32_t old = exp_cnt[ge];
            exp_cnt[ge] = static_cast<uint8_t>(old + 1U);
            r_rcnt[hr]++;
#ifdef K11220_MULTI_M16
            if ((old & 15U) == 0U) {
                r_blk[hr]++;
                if (old == 0U) {
                    r_ecnt[hr]++;
                }
            } else if ((old & 15U) == 8U) {
                r_extra[hr] += 16U;
            }
#else
            if ((old & 7U) == 0U) {
                r_blk[hr]++;
                if (old == 0U) {
                    r_ecnt[hr]++;
                }
            }
#endif
        }
    }
    uint8_t act_mk[32];
    uint8_t act_c[32];
    uint8_t act_b[32];
    uint8_t act_dm[32];
    uint32_t r_mov[kRanks] = {};
    uint32_t n_act = 0U;
    for (uint32_t idx = 0U; idx < num_cands; ++idx) {
        const uint32_t sh = (idx & 7U) << 3U;
        const uint32_t ge = static_cast<uint32_t>((cand_ge_u64[idx >> 3U] >> sh) & 0xFFULL);
        const uint32_t c = exp_cnt[ge];
        if (c > 0U) {
            const uint32_t hr = ge >> kExpertShift;
            const uint32_t mk = static_cast<uint32_t>((cand_mask_u64[idx >> 3U] >> sh) & 0xFFULL);
            act_ge[n_act] = static_cast<uint8_t>(ge);
            act_mk[n_act] = static_cast<uint8_t>(mk);
            act_own[n_act] = static_cast<uint8_t>(hr);
            act_c[n_act] = static_cast<uint8_t>(c);
            if ((mk & ~(1U << hr)) != 0U) {
                r_mov[hr] |= (1U << n_act);
            }
#ifdef K11220_MULTI_M16
            act_b[n_act] = static_cast<uint8_t>((c + 15U) >> 4U);
            act_dm[n_act] = static_cast<uint8_t>(16U * ((c + 7U) >> 4U));
#else
            act_b[n_act] = static_cast<uint8_t>((c + 7U) >> 3U);
            act_dm[n_act] = 0U;
#endif
            ++n_act;
        }
    }
    if (n_act == 0U) {
        return 0U;
    }
    uint32_t r_cost[kRanks];
    for (uint32_t r = 0U; r < kRanks; ++r) {
        r_cost[r] = RankWaveCost(r_blk[r], r_extra[r]);
    }
    for (uint32_t step = 0U; step < 12U; ++step) {
        uint32_t max_b = r_blk[0];
        uint32_t max_cost = r_cost[0];
        uint32_t min_cost = max_cost;
        for (uint32_t r = 1U; r < kRanks; ++r) {
            const uint32_t rb = r_blk[r];
            const uint32_t rc = r_cost[r];
            if (rb > max_b) {
                max_b = rb;
            }
            if (rc > max_cost) {
                max_cost = rc;
            }
            if (rc < min_cost) {
                min_cost = rc;
            }
        }
        if (max_b <= 1U || max_cost <= min_cost + 32U) {
            break;
        }
        uint32_t cand_set = 0U;
        for (uint32_t r = 0U; r < kRanks; ++r) {
            if (r_blk[r] == max_b) {
                cand_set |= r_mov[r];
            }
        }
        if (cand_set == 0U) {
            break;
        }
        uint32_t best_key = 0xFFFFFFFFU;
        int32_t best_idx = -1;
        uint32_t best_hi = 0U;
        uint32_t best_lo = 0U;
        while (cand_set != 0U) {
            const uint32_t idx = static_cast<uint32_t>(__builtin_ctz(cand_set));
            cand_set &= (cand_set - 1U);
            const uint32_t hi = act_own[idx];
            const uint32_t ge = act_ge[idx];
            const uint32_t c = act_c[idx];
            const uint32_t b = act_b[idx];
            const uint32_t dm = act_dm[idx];
            const uint32_t home = ge >> kExpertShift;
            uint32_t m = static_cast<uint32_t>(act_mk[idx]) & ~(1U << hi);
            while (m != 0U) {
                const uint32_t lo = static_cast<uint32_t>(__builtin_ctz(m));
                m &= (m - 1U);
                if (r_ecnt[lo] >= kMaxLocalExp || r_rcnt[lo] + c > 256U) {
                    continue;
                }
                const uint32_t new_lo_blk = r_blk[lo] + b;
                const uint32_t new_lo_ext = r_extra[lo] + dm + ((lo == home) ? 0U : b);
                const uint32_t new_lo_cost = RankWaveCost(new_lo_blk, new_lo_ext);
                if (new_lo_cost + 16U >= r_cost[hi]) {
                    continue;
                }
                const uint32_t b_inv = 4U - ((b > 4U) ? 4U : b);
                const uint32_t key = (new_lo_cost << 14U) | (b_inv << 11U) | (lo << 8U) | ge;
                if (key < best_key) {
                    best_key = key;
                    best_idx = static_cast<int32_t>(idx);
                    best_hi = hi;
                    best_lo = lo;
                }
            }
        }
        if (best_idx < 0) {
            break;
        }
        const uint32_t uidx = static_cast<uint32_t>(best_idx);
        const uint32_t move_ge = act_ge[uidx];
        const uint32_t home = move_ge >> kExpertShift;
        const uint32_t c = act_c[uidx];
        const uint32_t b = act_b[uidx];
        const uint32_t dm = act_dm[uidx];
        act_own[uidx] = static_cast<uint8_t>(best_lo);
        r_mov[best_hi] &= ~(1U << uidx);
        if ((static_cast<uint32_t>(act_mk[uidx]) & ~(1U << best_lo)) != 0U) {
            r_mov[best_lo] |= (1U << uidx);
        }
        r_ecnt[best_hi]--;
        r_ecnt[best_lo]++;
        r_rcnt[best_hi] -= c;
        r_rcnt[best_lo] += c;
        r_blk[best_hi] -= b;
        r_blk[best_lo] += b;
        r_extra[best_hi] -= dm + ((best_hi == home) ? 0U : b);
        r_extra[best_lo] += dm + ((best_lo == home) ? 0U : b);
        r_cost[best_hi] = RankWaveCost(r_blk[best_hi], r_extra[best_hi]);
        r_cost[best_lo] = RankWaveCost(r_blk[best_lo], r_extra[best_lo]);
    }
    return n_act;
}

#ifdef K11220_FAST_P1_V3
__aicore__ inline uint32_t AssignExpertOwnersFastV5(
    const int32_t* raw_r, int32_t routes, uint32_t num_cands,
    const uint64_t* cand_ge_u64, const uint64_t* cand_mask_u64,
    uint8_t* exp_cnt, uint8_t* act_ge, uint8_t* act_own)
{
    uint64_t* cnt_u64 = reinterpret_cast<uint64_t*>(exp_cnt);
    for (uint32_t w = 0U; w < 32U; ++w) {
        cnt_u64[w] = 0ULL;
    }
    constexpr uint32_t kRanks = static_cast<uint32_t>(kDefaultWorldSize);
    constexpr uint32_t kMaxLocalExp = 256U / kRanks;
    uint32_t r_ecnt[kRanks] = {};
    uint32_t r_rcnt[kRanks] = {};
    uint32_t r_blk[kRanks] = {};
    uint32_t r_extra[kRanks] = {};
    for (int32_t i = 0; i < routes; ++i) {
        const int32_t r_val = raw_r[i];
        if (r_val >= 0) {
            const uint32_t ge = static_cast<uint32_t>(r_val) & 255U;
            const uint32_t hr = ge >> kExpertShift;
            const uint32_t old = exp_cnt[ge];
            exp_cnt[ge] = static_cast<uint8_t>(old + 1U);
            r_rcnt[hr]++;
#ifdef K11220_MULTI_M32
            if ((old & 31U) == 0U) {
                r_blk[hr]++;
                if (old == 0U) {
                    r_ecnt[hr]++;
                }
            } else if ((old & 7U) == 0U) {
                r_extra[hr] += 8U;
            }
#elif defined(K11220_MULTI_M16)
            if ((old & 15U) == 0U) {
                r_blk[hr]++;
                if (old == 0U) {
                    r_ecnt[hr]++;
                }
            } else if ((old & 15U) == 8U) {
                r_extra[hr] += 16U;
            }
#else
            if ((old & 7U) == 0U) {
                r_blk[hr]++;
                if (old == 0U) {
                    r_ecnt[hr]++;
                }
            }
#endif
        }
    }
    uint8_t act_mk[48];
    uint8_t act_c[48];
    uint8_t act_b[48];
    uint8_t act_dm[48];
    uint64_t r_mov[kRanks] = {};
    uint32_t n_act = 0U;
    for (uint32_t idx = 0U; idx < num_cands; ++idx) {
        const uint32_t sh = (idx & 7U) << 3U;
        const uint32_t ge = static_cast<uint32_t>((cand_ge_u64[idx >> 3U] >> sh) & 0xFFULL);
        const uint32_t c = exp_cnt[ge];
        if (c > 0U) {
            const uint32_t hr = ge >> kExpertShift;
            const uint32_t mk = static_cast<uint32_t>((cand_mask_u64[idx >> 3U] >> sh) & 0xFFULL);
            act_ge[n_act] = static_cast<uint8_t>(ge);
            act_mk[n_act] = static_cast<uint8_t>(mk);
            act_own[n_act] = static_cast<uint8_t>(hr);
            act_c[n_act] = static_cast<uint8_t>(c);
            if ((mk & ~(1U << hr)) != 0U) {
                r_mov[hr] |= (1ULL << n_act);
            }
#ifdef K11220_MULTI_M32
            act_b[n_act] = static_cast<uint8_t>((c + 31U) >> 5U);
            act_dm[n_act] = static_cast<uint8_t>(8U * ((c - 1U) >> 3U));
#elif defined(K11220_MULTI_M16)
            act_b[n_act] = static_cast<uint8_t>((c + 15U) >> 4U);
            act_dm[n_act] = static_cast<uint8_t>(16U * ((c + 7U) >> 4U));
#else
            act_b[n_act] = static_cast<uint8_t>((c + 7U) >> 3U);
            act_dm[n_act] = 0U;
#endif
            ++n_act;
        }
    }
    if (n_act == 0U) {
        return 0U;
    }
    uint32_t r_cost[kRanks];
    for (uint32_t r = 0U; r < kRanks; ++r) {
        r_cost[r] = RankWaveCost(r_blk[r], r_extra[r]);
    }
    for (uint32_t step = 0U; step < 16U; ++step) {
        uint32_t max_b = r_blk[0];
        uint32_t max_cost = r_cost[0];
        uint32_t min_cost = max_cost;
        for (uint32_t r = 1U; r < kRanks; ++r) {
            const uint32_t rb = r_blk[r];
            const uint32_t rc = r_cost[r];
            if (rb > max_b) {
                max_b = rb;
            }
            if (rc > max_cost) {
                max_cost = rc;
            }
            if (rc < min_cost) {
                min_cost = rc;
            }
        }
        if (max_b <= 1U || max_cost <= min_cost + 32U) {
            break;
        }
        uint64_t cand_set = 0ULL;
        for (uint32_t r = 0U; r < kRanks; ++r) {
            if (r_cost[r] == max_cost) {
                cand_set |= r_mov[r];
            }
        }
        if (cand_set == 0ULL) {
            for (uint32_t r = 0U; r < kRanks; ++r) {
                if (r_blk[r] == max_b) {
                    cand_set |= r_mov[r];
                }
            }
        }
        if (cand_set == 0ULL) {
            break;
        }
        uint32_t best_key = 0xFFFFFFFFU;
        int32_t best_idx = -1;
        uint32_t best_hi = 0U;
        uint32_t best_lo = 0U;
        while (cand_set != 0ULL) {
            const uint32_t idx = static_cast<uint32_t>(__builtin_ctzll(cand_set));
            cand_set &= (cand_set - 1ULL);
            const uint32_t hi = act_own[idx];
            const uint32_t ge = act_ge[idx];
            const uint32_t c = act_c[idx];
            const uint32_t b = act_b[idx];
            const uint32_t dm = act_dm[idx];
            const uint32_t home = ge >> kExpertShift;
            uint32_t m = static_cast<uint32_t>(act_mk[idx]) & ~(1U << hi);
            while (m != 0U) {
                const uint32_t lo = static_cast<uint32_t>(__builtin_ctz(m));
                m &= (m - 1U);
                if (r_ecnt[lo] >= kMaxLocalExp || r_rcnt[lo] + c > 256U) {
                    continue;
                }
                const uint32_t new_lo_blk = r_blk[lo] + b;
                const uint32_t new_lo_ext = r_extra[lo] + dm + ((lo == home) ? 0U : b);
                const uint32_t new_lo_cost = RankWaveCost(new_lo_blk, new_lo_ext);
                if (new_lo_cost + 16U >= r_cost[hi]) {
                    continue;
                }
                const uint32_t b_inv = 4U - ((b > 4U) ? 4U : b);
                const uint32_t key = (new_lo_cost << 14U) | (b_inv << 11U) | (lo << 8U) | ge;
                if (key < best_key) {
                    best_key = key;
                    best_idx = static_cast<int32_t>(idx);
                    best_hi = hi;
                    best_lo = lo;
                }
            }
        }
        if (best_idx < 0) {
            break;
        }
        const uint32_t uidx = static_cast<uint32_t>(best_idx);
        const uint32_t move_ge = act_ge[uidx];
        const uint32_t home = move_ge >> kExpertShift;
        const uint32_t c = act_c[uidx];
        const uint32_t b = act_b[uidx];
        const uint32_t dm = act_dm[uidx];
        act_own[uidx] = static_cast<uint8_t>(best_lo);
        r_mov[best_hi] &= ~(1ULL << uidx);
        if ((static_cast<uint32_t>(act_mk[uidx]) & ~(1U << best_lo)) != 0U) {
            r_mov[best_lo] |= (1ULL << uidx);
        }
        r_ecnt[best_hi]--;
        r_ecnt[best_lo]++;
        r_rcnt[best_hi] -= c;
        r_rcnt[best_lo] += c;
        r_blk[best_hi] -= b;
        r_blk[best_lo] += b;
        r_extra[best_hi] -= dm + ((best_hi == home) ? 0U : b);
        r_extra[best_lo] += dm + ((best_lo == home) ? 0U : b);
        r_cost[best_hi] = RankWaveCost(r_blk[best_hi], r_extra[best_hi]);
        r_cost[best_lo] = RankWaveCost(r_blk[best_lo], r_extra[best_lo]);
    }
    return n_act;
}
#endif

#ifdef K112600_MIRROR_64
__aicore__ inline uint32_t AssignExpertOwnersFastV6(
    const int32_t* raw_r, int32_t routes, uint32_t num_cands,
    const uint64_t* cand_ge_u64, const uint64_t* cand_mask_u64,
    uint8_t* exp_cnt, uint8_t* act_ge, uint8_t* act_own)
{
    uint64_t* cnt_u64 = reinterpret_cast<uint64_t*>(exp_cnt);
    for (uint32_t w = 0U; w < 32U; ++w) {
        cnt_u64[w] = 0ULL;
    }
    constexpr uint32_t kRanks = static_cast<uint32_t>(kDefaultWorldSize);
    constexpr uint32_t kMaxLocalExp = 256U / kRanks;
    uint32_t r_ecnt[kRanks] = {};
    uint32_t r_rcnt[kRanks] = {};
    uint32_t r_blk[kRanks] = {};
    uint32_t r_extra[kRanks] = {};
    for (int32_t i = 0; i < routes; ++i) {
        const int32_t r_val = raw_r[i];
        if (r_val >= 0) {
            const uint32_t ge = static_cast<uint32_t>(r_val) & 255U;
            const uint32_t hr = ge >> kExpertShift;
            const uint32_t old = exp_cnt[ge];
            exp_cnt[ge] = static_cast<uint8_t>(old + 1U);
            r_rcnt[hr]++;
#ifdef K11220_MULTI_M32
            if ((old & 31U) == 0U) {
                r_blk[hr]++;
                if (old == 0U) {
                    r_ecnt[hr]++;
                }
            } else if ((old & 7U) == 0U) {
                r_extra[hr] += 8U;
            }
#elif defined(K11220_MULTI_M16)
            if ((old & 15U) == 0U) {
                r_blk[hr]++;
                if (old == 0U) {
                    r_ecnt[hr]++;
                }
            } else if ((old & 15U) == 8U) {
                r_extra[hr] += 16U;
            }
#else
            if ((old & 7U) == 0U) {
                r_blk[hr]++;
                if (old == 0U) {
                    r_ecnt[hr]++;
                }
            }
#endif
        }
    }
    uint8_t act_mk[64];
    uint8_t act_c[64];
    uint8_t act_b[64];
    uint8_t act_dm[64];
    uint64_t r_mov[kRanks] = {};
    uint32_t n_act = 0U;
    const uint32_t lim_cands = (num_cands > 64U) ? 64U : num_cands;
    for (uint32_t idx = 0U; idx < lim_cands; ++idx) {
        const uint32_t sh = (idx & 7U) << 3U;
        const uint32_t ge = static_cast<uint32_t>((cand_ge_u64[idx >> 3U] >> sh) & 0xFFULL);
        const uint32_t c = exp_cnt[ge];
        if (c > 0U) {
            const uint32_t hr = ge >> kExpertShift;
            const uint32_t mk = static_cast<uint32_t>((cand_mask_u64[idx >> 3U] >> sh) & 0xFFULL);
            act_ge[n_act] = static_cast<uint8_t>(ge);
            act_mk[n_act] = static_cast<uint8_t>(mk);
            act_own[n_act] = static_cast<uint8_t>(hr);
            act_c[n_act] = static_cast<uint8_t>(c);
            if ((mk & ~(1U << hr)) != 0U) {
                r_mov[hr] |= (1ULL << n_act);
            }
#ifdef K11220_MULTI_M32
            act_b[n_act] = static_cast<uint8_t>((c + 31U) >> 5U);
            act_dm[n_act] = static_cast<uint8_t>(8U * ((c - 1U) >> 3U));
#elif defined(K11220_MULTI_M16)
            act_b[n_act] = static_cast<uint8_t>((c + 15U) >> 4U);
            act_dm[n_act] = static_cast<uint8_t>(16U * ((c + 7U) >> 4U));
#else
            act_b[n_act] = static_cast<uint8_t>((c + 7U) >> 3U);
            act_dm[n_act] = 0U;
#endif
            ++n_act;
        }
    }
    if (n_act == 0U) {
        return 0U;
    }
    uint32_t r_cost[kRanks];
    for (uint32_t r = 0U; r < kRanks; ++r) {
        r_cost[r] = RankWaveCost(r_blk[r], r_extra[r]);
    }
    for (uint32_t step = 0U; step < 20U; ++step) {
        uint32_t max_b = r_blk[0];
        uint32_t max_cost = r_cost[0];
        uint32_t min_cost = max_cost;
        for (uint32_t r = 1U; r < kRanks; ++r) {
            const uint32_t rb = r_blk[r];
            const uint32_t rc = r_cost[r];
            if (rb > max_b) {
                max_b = rb;
            }
            if (rc > max_cost) {
                max_cost = rc;
            }
            if (rc < min_cost) {
                min_cost = rc;
            }
        }
        if (max_b <= 1U || max_cost <= min_cost + 32U) {
            break;
        }
        uint64_t cand_set = 0ULL;
        for (uint32_t r = 0U; r < kRanks; ++r) {
            if (r_cost[r] == max_cost) {
                cand_set |= r_mov[r];
            }
        }
        if (cand_set == 0ULL) {
            for (uint32_t r = 0U; r < kRanks; ++r) {
                if (r_blk[r] == max_b) {
                    cand_set |= r_mov[r];
                }
            }
        }
        if (cand_set == 0ULL) {
            break;
        }
        uint32_t best_key = 0xFFFFFFFFU;
        int32_t best_idx = -1;
        uint32_t best_hi = 0U;
        uint32_t best_lo = 0U;
        while (cand_set != 0ULL) {
            const uint32_t idx = static_cast<uint32_t>(__builtin_ctzll(cand_set));
            cand_set &= (cand_set - 1ULL);
            const uint32_t hi = act_own[idx];
            const uint32_t ge = act_ge[idx];
            const uint32_t c = act_c[idx];
            const uint32_t b = act_b[idx];
            const uint32_t dm = act_dm[idx];
            const uint32_t home = ge >> kExpertShift;
            uint32_t m = static_cast<uint32_t>(act_mk[idx]) & ~(1U << hi);
            while (m != 0U) {
                const uint32_t lo = static_cast<uint32_t>(__builtin_ctz(m));
                m &= (m - 1U);
                if (r_ecnt[lo] >= kMaxLocalExp || r_rcnt[lo] + c > 256U) {
                    continue;
                }
                const uint32_t new_lo_blk = r_blk[lo] + b;
                const uint32_t new_lo_ext = r_extra[lo] + dm + ((lo == home) ? 0U : b);
                const uint32_t new_lo_cost = RankWaveCost(new_lo_blk, new_lo_ext);
                if (new_lo_cost + 16U >= r_cost[hi]) {
                    continue;
                }
                const uint32_t b_inv = 4U - ((b > 4U) ? 4U : b);
                const uint32_t key = (new_lo_cost << 14U) | (b_inv << 11U) | (lo << 8U) | ge;
                if (key < best_key) {
                    best_key = key;
                    best_idx = static_cast<int32_t>(idx);
                    best_hi = hi;
                    best_lo = lo;
                }
            }
        }
#ifdef K112612_AUG2_OWN
        if (best_idx < 0) {
            uint32_t best2_key = 0xFFFFFFFFU;
            int32_t b_i1 = -1;
            int32_t b_i2 = -1;
            uint32_t b_hi1 = 0U;
            uint32_t b_mid = 0U;
            uint32_t b_lo2 = 0U;
            uint64_t cs1 = 0ULL;
            for (uint32_t r = 0U; r < kRanks; ++r) {
                if (r_cost[r] == max_cost) {
                    cs1 |= r_mov[r];
                }
            }
            while (cs1 != 0ULL) {
                const uint32_t i1 = static_cast<uint32_t>(__builtin_ctzll(cs1));
                cs1 &= (cs1 - 1ULL);
                const uint32_t hi1 = act_own[i1];
                const uint32_t ge1 = act_ge[i1];
                const uint32_t c1 = act_c[i1];
                const uint32_t b1 = act_b[i1];
                const uint32_t dm1 = act_dm[i1];
                const uint32_t hm1 = ge1 >> kExpertShift;
                uint32_t m1_mask = static_cast<uint32_t>(act_mk[i1]) & ~(1U << hi1);
                while (m1_mask != 0U) {
                    const uint32_t mid = static_cast<uint32_t>(__builtin_ctz(m1_mask));
                    m1_mask &= (m1_mask - 1U);
                    uint64_t cs2 = r_mov[mid] & ~(1ULL << i1);
                    while (cs2 != 0ULL) {
                        const uint32_t i2 = static_cast<uint32_t>(__builtin_ctzll(cs2));
                        cs2 &= (cs2 - 1ULL);
                        const uint32_t ge2 = act_ge[i2];
                        const uint32_t c2 = act_c[i2];
                        const uint32_t b2 = act_b[i2];
                        const uint32_t dm2 = act_dm[i2];
                        const uint32_t hm2 = ge2 >> kExpertShift;
                        if (r_rcnt[mid] + c1 > 256U + c2) {
                            continue;
                        }
                        const uint32_t mid_blk = r_blk[mid] - b2 + b1;
                        const uint32_t mid_ext = r_extra[mid]
                            - (dm2 + ((mid == hm2) ? 0U : b2))
                            + (dm1 + ((mid == hm1) ? 0U : b1));
                        const uint32_t mid_cost = RankWaveCost(mid_blk, mid_ext);
                        if (mid_cost + 16U >= r_cost[hi1]) {
                            continue;
                        }
                        uint32_t m2_mask = static_cast<uint32_t>(act_mk[i2]) & ~((1U << mid) | (1U << hi1));
                        while (m2_mask != 0U) {
                            const uint32_t lo2 = static_cast<uint32_t>(__builtin_ctz(m2_mask));
                            m2_mask &= (m2_mask - 1U);
                            if (r_ecnt[lo2] >= kMaxLocalExp || r_rcnt[lo2] + c2 > 256U) {
                                continue;
                            }
                            const uint32_t lo2_blk = r_blk[lo2] + b2;
                            const uint32_t lo2_ext = r_extra[lo2] + dm2 + ((lo2 == hm2) ? 0U : b2);
                            const uint32_t lo2_cost = RankWaveCost(lo2_blk, lo2_ext);
                            if (lo2_cost + 16U >= r_cost[hi1]) {
                                continue;
                            }
                            const uint32_t peak2 = (mid_cost >= lo2_cost) ? mid_cost : lo2_cost;
                            const uint32_t k2 = (peak2 << 18U) | (lo2_cost << 8U) | (lo2 << 5U) | (i1 & 31U);
                            if (k2 < best2_key) {
                                best2_key = k2;
                                b_i1 = static_cast<int32_t>(i1);
                                b_i2 = static_cast<int32_t>(i2);
                                b_hi1 = hi1;
                                b_mid = mid;
                                b_lo2 = lo2;
                            }
                        }
                    }
                }
            }
            if (b_i1 >= 0) {
                const uint32_t u2 = static_cast<uint32_t>(b_i2);
                const uint32_t hm2 = act_ge[u2] >> kExpertShift;
                const uint32_t c2 = act_c[u2];
                const uint32_t b2 = act_b[u2];
                const uint32_t dm2 = act_dm[u2];
                act_own[u2] = static_cast<uint8_t>(b_lo2);
                r_mov[b_mid] &= ~(1ULL << u2);
                if ((static_cast<uint32_t>(act_mk[u2]) & ~(1U << b_lo2)) != 0U) {
                    r_mov[b_lo2] |= (1ULL << u2);
                }
                r_ecnt[b_mid]--;
                r_ecnt[b_lo2]++;
                r_rcnt[b_mid] -= c2;
                r_rcnt[b_lo2] += c2;
                r_blk[b_mid] -= b2;
                r_blk[b_lo2] += b2;
                r_extra[b_mid] -= dm2 + ((b_mid == hm2) ? 0U : b2);
                r_extra[b_lo2] += dm2 + ((b_lo2 == hm2) ? 0U : b2);
                r_cost[b_lo2] = RankWaveCost(r_blk[b_lo2], r_extra[b_lo2]);

                const uint32_t u1 = static_cast<uint32_t>(b_i1);
                const uint32_t hm1 = act_ge[u1] >> kExpertShift;
                const uint32_t c1 = act_c[u1];
                const uint32_t b1 = act_b[u1];
                const uint32_t dm1 = act_dm[u1];
                act_own[u1] = static_cast<uint8_t>(b_mid);
                r_mov[b_hi1] &= ~(1ULL << u1);
                if ((static_cast<uint32_t>(act_mk[u1]) & ~(1U << b_mid)) != 0U) {
                    r_mov[b_mid] |= (1ULL << u1);
                }
                r_ecnt[b_hi1]--;
                r_ecnt[b_mid]++;
                r_rcnt[b_hi1] -= c1;
                r_rcnt[b_mid] += c1;
                r_blk[b_hi1] -= b1;
                r_blk[b_mid] += b1;
                r_extra[b_hi1] -= dm1 + ((b_hi1 == hm1) ? 0U : b1);
                r_extra[b_mid] += dm1 + ((b_mid == hm1) ? 0U : b1);
                r_cost[b_hi1] = RankWaveCost(r_blk[b_hi1], r_extra[b_hi1]);
                r_cost[b_mid] = RankWaveCost(r_blk[b_mid], r_extra[b_mid]);
                continue;
            }
            uint64_t sec_set = 0ULL;
            for (uint32_t r = 0U; r < kRanks; ++r) {
                if (r_cost[r] > min_cost + 32U && r_cost[r] < max_cost) {
                    sec_set |= r_mov[r];
                }
            }
            while (sec_set != 0ULL) {
                const uint32_t idx = static_cast<uint32_t>(__builtin_ctzll(sec_set));
                sec_set &= (sec_set - 1ULL);
                const uint32_t hi = act_own[idx];
                const uint32_t ge = act_ge[idx];
                const uint32_t c = act_c[idx];
                const uint32_t b = act_b[idx];
                const uint32_t dm = act_dm[idx];
                const uint32_t home = ge >> kExpertShift;
                uint32_t m = static_cast<uint32_t>(act_mk[idx]) & ~(1U << hi);
                while (m != 0U) {
                    const uint32_t lo = static_cast<uint32_t>(__builtin_ctz(m));
                    m &= (m - 1U);
                    if (r_ecnt[lo] >= kMaxLocalExp || r_rcnt[lo] + c > 256U) {
                        continue;
                    }
                    const uint32_t new_lo_blk = r_blk[lo] + b;
                    const uint32_t new_lo_ext = r_extra[lo] + dm + ((lo == home) ? 0U : b);
                    const uint32_t new_lo_cost = RankWaveCost(new_lo_blk, new_lo_ext);
                    if (new_lo_cost + 16U >= r_cost[hi]) {
                        continue;
                    }
                    const uint32_t b_inv = 4U - ((b > 4U) ? 4U : b);
                    const uint32_t key = (new_lo_cost << 14U) | (b_inv << 11U) | (lo << 8U) | ge;
                    if (key < best_key) {
                        best_key = key;
                        best_idx = static_cast<int32_t>(idx);
                        best_hi = hi;
                        best_lo = lo;
                    }
                }
            }
            if (best_idx < 0) {
                break;
            }
        }
#else
        if (best_idx < 0) {
            break;
        }
#endif
        const uint32_t uidx = static_cast<uint32_t>(best_idx);
        const uint32_t move_ge = act_ge[uidx];
        const uint32_t home = move_ge >> kExpertShift;
        const uint32_t c = act_c[uidx];
        const uint32_t b = act_b[uidx];
        const uint32_t dm = act_dm[uidx];
        act_own[uidx] = static_cast<uint8_t>(best_lo);
        r_mov[best_hi] &= ~(1ULL << uidx);
        if ((static_cast<uint32_t>(act_mk[uidx]) & ~(1U << best_lo)) != 0U) {
            r_mov[best_lo] |= (1ULL << uidx);
        }
        r_ecnt[best_hi]--;
        r_ecnt[best_lo]++;
        r_rcnt[best_hi] -= c;
        r_rcnt[best_lo] += c;
        r_blk[best_hi] -= b;
        r_blk[best_lo] += b;
        r_extra[best_hi] -= dm + ((best_hi == home) ? 0U : b);
        r_extra[best_lo] += dm + ((best_lo == home) ? 0U : b);
        r_cost[best_hi] = RankWaveCost(r_blk[best_hi], r_extra[best_hi]);
        r_cost[best_lo] = RankWaveCost(r_blk[best_lo], r_extra[best_lo]);
    }
    return n_act;
}
#endif
#endif

struct LiteMirrorMeta {
    uint64_t mirror_weights[6];
    uint64_t packed_mirror_ge;
    uint64_t pad;
};
static_assert(sizeof(LiteMirrorMeta) == 64);

#ifdef K11180_LITE_MIRROR
__aicore__ inline void FillLiteOwners(
    const AscendC::LocalTensor<int32_t>& eids_ub, int32_t routes,
    uint64_t packed_mge, uint8_t* owner_of_exp)
{
    uint32_t* own_u32 = reinterpret_cast<uint32_t*>(owner_of_exp);
    for (uint32_t r = 0U; r < 8U; ++r) {
        const uint32_t v = 0x01010101U * r;
        for (uint32_t k = 0U; k < 8U; ++k) {
            own_u32[r * 8U + k] = v;
        }
    }
    uint8_t exp_cnt[256];
    uint64_t* cnt_u64 = reinterpret_cast<uint64_t*>(exp_cnt);
    for (uint32_t w = 0U; w < 32U; ++w) {
        cnt_u64[w] = 0ULL;
    }
    uint8_t r_waves[8] = {0, 0, 0, 0, 0, 0, 0};
    uint8_t r_ecnt[8] = {0, 0, 0, 0, 0, 0, 0};
    for (int32_t i = 0; i < routes; ++i) {
        const int32_t raw = eids_ub.GetValue(i);
        if (raw < 0) {
            continue;
        }
        const uint32_t ge = static_cast<uint32_t>(raw) & 255U;
        const uint8_t old = exp_cnt[ge];
        exp_cnt[ge] = static_cast<uint8_t>(old + 1U);
        if ((old & 7U) == 0U) {
            const uint32_t hr = ge >> 5U;
            r_waves[hr]++;
            if (old == 0U) {
                r_ecnt[hr]++;
            }
        }
    }
    uint32_t steal_mask = 0U;
    for (uint32_t step = 0U; step < 8U; ++step) {
        uint32_t max_w = r_waves[0];
        for (uint32_t r = 1U; r < 8U; ++r) {
            if (r_waves[r] > max_w) {
                max_w = r_waves[r];
            }
        }
        if (max_w <= 1U) {
            break;
        }
        int32_t best_mr = -1;
        uint32_t best_dst_w = 255U;
        for (uint32_t mr = 0U; mr < 8U; ++mr) {
            if ((steal_mask & (1U << mr)) != 0U || r_ecnt[mr] >= 32U) {
                continue;
            }
            const uint32_t ge = static_cast<uint32_t>((packed_mge >> (mr * 8U)) & 255ULL);
            const uint32_t hr = ge >> 5U;
            const uint32_t c = exp_cnt[ge];
            if (c == 0U || mr == hr || r_waves[hr] != max_w) {
                continue;
            }
            const uint32_t w = (c + 7U) >> 3U;
            const uint32_t dst_w = r_waves[mr] + w;
            if (dst_w < max_w && dst_w < best_dst_w) {
                best_dst_w = dst_w;
                best_mr = static_cast<int32_t>(mr);
            }
        }
        if (best_mr < 0) {
            break;
        }
        const uint32_t umr = static_cast<uint32_t>(best_mr);
        const uint32_t ge = static_cast<uint32_t>((packed_mge >> (umr * 8U)) & 255ULL);
        const uint32_t hr = ge >> 5U;
        const uint32_t w = (static_cast<uint32_t>(exp_cnt[ge]) + 7U) >> 3U;
        exp_cnt[ge] = 0U;
        r_waves[hr] = static_cast<uint8_t>(r_waves[hr] - w);
        r_waves[umr] = static_cast<uint8_t>(r_waves[umr] + w);
        r_ecnt[hr]--;
        r_ecnt[umr]++;
        steal_mask |= (1U << umr);
        owner_of_exp[ge] = static_cast<uint8_t>(umr);
    }
}
#endif

#include "reduce.inc"

__aicore__ inline uint32_t AllReduceBf16(
    CommunicationBuffers& buffers, __gm__ uint8_t* window,
    __gm__ bfloat16_t* partial, __gm__ bfloat16_t* output,
    int64_t rows, int64_t capacity, int32_t rank, int32_t ranks,
    int64_t vector_index, int64_t vectors,
    __gm__ bfloat16_t* shared_sum = nullptr, bool sync_end = true)
{
    const bool quad4 = IsQuad4Active(capacity, rows);
    const CommunicationLayout layout = quad4
        ? MakeQuad4Layout(capacity, ranks, rows)
        : MakeCommunicationLayout(capacity, ranks);
    uint32_t epoch = 0;
    if ASCEND_IS_AIV {
        epoch = ReadCommunicationEpoch(window);
        ReduceCommunicationPayloadTiled(buffers, window, layout, partial, output,
                                        rows, rank, ranks, epoch,
                                        vector_index, vectors, shared_sum, quad4);
        if (sync_end) {
#ifdef K112824_NO_R2_END_DSB
            AscendC::PipeBarrier<PIPE_ALL>();
#else
            CommunicationPublish();
#endif
            AscendC::SyncAll<true>();
#ifdef K112110_SIP_RED1
#ifdef K112140_SIP_RED1_MIN_ROWS
            const bool sip_r1_ack = (!quad4 && rows >= static_cast<int64_t>(K112140_SIP_RED1_MIN_ROWS));
#else
            const bool sip_r1_ack = !quad4;
#endif
            if (sip_r1_ack && vector_index > 0 && vector_index < ranks) {
                const int32_t peer = (rank + static_cast<int32_t>(vector_index)) & (ranks - 1);
                __gm__ int32_t* remote_ack = reinterpret_cast<__gm__ int32_t*>(
                    aclshmem_ptr(CommunicationR1Done(window, layout, ranks, rank), peer));
                *remote_ack = static_cast<int32_t>(epoch);
                dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(remote_ack));
            }
#endif
        }
    }
    return epoch;
}

__aicore__ inline void AllReduceSlotsToBf16(
    CommunicationBuffers& buffers, __gm__ uint8_t* window,
    __gm__ bfloat16_t* partial, __gm__ bfloat16_t* output,
    int64_t rows, int64_t capacity, int32_t rank, int32_t ranks,
    int64_t vector_index, int64_t vectors,
    __gm__ bfloat16_t* shared_partial,
    __gm__ const int32_t* expert_ids,
    __gm__ const float* probabilities,
    const __gm__ uint8_t* owner_gm = nullptr,
    __gm__ const bfloat16_t* pre_shared_sum = nullptr,
    int64_t* prof_acc = nullptr)
{
    const bool quad4 = IsQuad4Active(capacity, rows);
    const CommunicationLayout layout = quad4
        ? MakeQuad4Layout(capacity, ranks, rows)
        : MakeCommunicationLayout(capacity, ranks);
    if ASCEND_IS_AIV {
        const uint32_t rt_epoch = ReadCommunicationEpoch(window);
        const uint32_t sh_epoch = rt_epoch - 1U;
        for (int64_t off = vector_index; off < ranks; off += vectors) {
            if (off == 0) {
                continue;
            }
            const int32_t peer = (rank + static_cast<int32_t>(off)) & (ranks - 1);
            SignalPeerReady(window, layout, ranks, rt_epoch, rank, peer);
        }
#if defined(K11220_RS768_PUSH) || defined(K112600_RS768_HIW)
#ifdef K11220_RS768_PUSH
        const bool use_rs768 = (ranks == 8);
#else
        const bool use_rs768 = (ranks == 8) && (rows > 16);
#endif
        if (use_rs768) {
            const int64_t rs_qcap = RsQCap(capacity, rows, quad4);
            ReduceScatterSlots768_Push(buffers, window, layout, rs_qcap,
                                       output, rows, rank, ranks,
                                       sh_epoch, rt_epoch, vector_index,
                                       expert_ids, probabilities, owner_gm, quad4);
            CommunicationPublish();
            AscendC::SyncAll<true>();
            AllGatherRecv768(buffers, window, layout, rs_qcap,
                             output, rows, rank, ranks, rt_epoch, vector_index, quad4);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<true>();
        } else {
            ReduceCommunicationPayloadSlots(buffers, window, layout,
                                             partial, output, rows, rank, ranks,
                                             sh_epoch, rt_epoch, vector_index, vectors,
                                             shared_partial, expert_ids, probabilities,
                                             owner_gm, quad4, pre_shared_sum, prof_acc);
#ifdef K112824_NO_R2_END_DSB
            AscendC::PipeBarrier<PIPE_ALL>();
#else
            CommunicationPublish();
#endif
            AscendC::SyncAll<true>();
        }
#else
        ReduceCommunicationPayloadSlots(buffers, window, layout,
                                         partial, output, rows, rank, ranks,
                                         sh_epoch, rt_epoch, vector_index, vectors,
                                         shared_partial, expert_ids, probabilities,
                                         owner_gm, quad4, pre_shared_sum, prof_acc);
#ifdef K112824_NO_R2_END_DSB
        AscendC::PipeBarrier<PIPE_ALL>();
#else
        CommunicationPublish();
#endif
        AscendC::SyncAll<true>();
#endif
    }
}

}
#endif
