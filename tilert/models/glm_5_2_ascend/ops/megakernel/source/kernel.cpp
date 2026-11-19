#if defined(__CCE_AICORE__) && !defined(__NPU_DEVICE__)
#define __NPU_DEVICE__ 1
#endif
#ifndef ASCENDC_CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif

#include "contract.h"
#include "../communication/communication_device.h"
#include "../local/attention/source/tests/glm52/whole_layer/attention_indexed_native.h"
#include "../local/attention/source/tests/glm52/whole_layer/attention_full_native.h"
#include "../local/ffn/bridge/source/ffn_bridge/body_native.h"
#include "../local/ffn/body/source/routed_int4/routed_body_native.h"

#ifndef KERNEL_VER
#define KERNEL_VER 1
#endif
namespace layer_tp_tree_chain {
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

struct PrefetchSpan {
    uint64_t addr;
    uint32_t bytes;
};

__aicore__ __attribute__((noinline)) void PrefetchSpansAic(
    TPipe& pipe, const PrefetchSpan* spans, int32_t count,
    const Communication* comm = nullptr, uint32_t rt_epoch_pf = 0U, bool poll_ag768 = false,
    __gm__ volatile int32_t* red2_done_gm = nullptr)
{
    if ASCEND_IS_AIC {
        constexpr uint32_t kChunkBytes = 65536U;
        uint32_t total_chunks = 0U;
        for (int32_t s = 0; s < count; ++s) {
            if (spans[s].addr != 0ULL) {
                total_chunks += spans[s].bytes >> 16U;
            }
        }
        const uint32_t cid = static_cast<uint32_t>(GetBlockIdx());
        if (cid < 24U && total_chunks > cid) {
            pipe.Reset();
            TBuf<TPosition::A1> pf_buf;
            pipe.InitBuffer(pf_buf, 4 * kChunkBytes);
            auto l1 = pf_buf.Get<int8_t>();
            uint32_t slot = 0U;
            uint32_t cur_base = 0U;
            int32_t s_idx = 0;
#ifdef K11100_OPP_PREFETCH
            uint32_t pending_peers = (comm != nullptr && rt_epoch_pf != 0U)
                ? (0xFFU & ~(1U << static_cast<uint32_t>(comm->rank)))
                : 0U;
            const auto layout = (comm != nullptr)
                ? whole_layer_tree::MakeCommunicationLayout(comm->capacity, comm->ranks)
                : whole_layer_tree::CommunicationLayout{};
#endif
            for (uint32_t g = cid; g < total_chunks; g += 24U) {
#ifdef K11100_OPP_PREFETCH
#ifdef K11220_PF_RED2_FULL
                if (red2_done_gm != nullptr) {
                    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ int32_t*>(red2_done_gm)));
                    if (*red2_done_gm != 0) {
                        break;
                    }
                } else
#endif
                if (comm != nullptr && rt_epoch_pf != 0U) {
                    while (pending_peers != 0U) {
                        const int32_t p = __builtin_ctz(pending_peers);
#if defined(K11220_RS768_PUSH) || defined(K112600_RS768_HIW)
                        __gm__ volatile int32_t* sig_addr = poll_ag768
                            ? whole_layer_tree::AgReady768((__gm__ uint8_t*)comm->window, layout, rt_epoch_pf, p)
                            : whole_layer_tree::CommunicationReady((__gm__ uint8_t*)comm->window, layout, 8, rt_epoch_pf, p);
#else
                        (void)poll_ag768;
                        __gm__ volatile int32_t* sig_addr = whole_layer_tree::CommunicationReady(
                            (__gm__ uint8_t*)comm->window, layout, 8, rt_epoch_pf, p);
#endif
                        dcci_cacheline((__gm__ uint8_t*)sig_addr);
                        if (*sig_addr != static_cast<int32_t>(rt_epoch_pf)) {
                            break;
                        }
                        pending_peers &= ~(1U << static_cast<uint32_t>(p));
                    }
                    if (pending_peers == 0U) {
                        break;
                    }
                }
#else
                (void)poll_ag768;
                (void)red2_done_gm;
#endif
                while (s_idx < count) {
                    const uint32_t sc = (spans[s_idx].addr != 0ULL) ? (spans[s_idx].bytes >> 16U) : 0U;
                    if (g < cur_base + sc) {
                        break;
                    }
                    cur_base += sc;
                    ++s_idx;
                }
                const uint64_t off = static_cast<uint64_t>(g - cur_base) << 16U;
                GlobalTensor<int8_t> gm;
                gm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(spans[s_idx].addr + off), kChunkBytes);
                DataCopy(l1[(slot & 3U) * kChunkBytes], gm, kChunkBytes);
                ++slot;
#ifdef K11100_OPP_PREFETCH
#ifdef K11220_PF_RED2_FULL
                if (red2_done_gm != nullptr) {
                    if ((slot & 1U) == 0U) {
                        PipeBarrier<PIPE_MTE2>();
                    }
                } else
#endif
                if (comm != nullptr && rt_epoch_pf != 0U) {
                    PipeBarrier<PIPE_MTE2>();
                }
#endif
            }
            PipeBarrier<PIPE_MTE2>();
            PipeBarrier<PIPE_ALL>();
            pipe.Reset();
        }
    }
}

__aicore__ __attribute__((noinline)) void ReducePhase(TPipe& pipe,
    const Communication& communication, uint64_t partial,
    uint64_t output, int64_t rows, int64_t blocks,
    uint64_t topk_src = 0, uint64_t topk_dst = 0,
    uint64_t shared_sum = 0, uint64_t expert_ids = 0, uint64_t probabilities = 0,
    uint64_t owner_gm = 0, int64_t* prof_acc = nullptr)
{
    whole_layer_tree::CommunicationBuffers buffers;
    if ASCEND_IS_AIV {
        buffers.Init(pipe);
        if (shared_sum != 0) {
#ifdef K112612_OVERLAP_SH_P4
            const __gm__ bfloat16_t* pre_sh_ptr = (communication.window != 0ULL)
                ? reinterpret_cast<const __gm__ bfloat16_t*>(shared_sum)
                : nullptr;
#else
            const __gm__ bfloat16_t* pre_sh_ptr = nullptr;
#endif
            whole_layer_tree::AllReduceSlotsToBf16(buffers,
                (__gm__ uint8_t*)communication.window,
                (__gm__ bfloat16_t*)partial, (__gm__ bfloat16_t*)output,
                rows, communication.capacity, communication.rank,
                8, GetBlockIdx(), 48,
                (__gm__ bfloat16_t*)topk_src,
                (__gm__ const int32_t*)expert_ids,
                (__gm__ const float*)probabilities,
                reinterpret_cast<const __gm__ uint8_t*>(owner_gm),
                pre_sh_ptr, prof_acc);
        } else {
            whole_layer_tree::AllReduceBf16(buffers,
                (__gm__ uint8_t*)communication.window,
                (__gm__ bfloat16_t*)partial, (__gm__ bfloat16_t*)output,
                rows, communication.capacity, communication.rank,
                8, GetBlockIdx(), 48,
                nullptr);
        }
        pipe.Reset();
    }
}

__aicore__ inline void RunLayer(
    TPipe& pipe, bool indexed, const Config& config, const Addresses& addresses,
    const Communication& communication, bool record_prof = false,
    uint64_t tree_mask_gm = 0, const ChainStep* next_step = nullptr,
    uint64_t layer_pw = 0, int64_t* prof_acc = nullptr,
    GM_ADDR resolved_topk_buf = nullptr, bool need_resolve_shared = false,
    bool* inout_resolved_ok = nullptr, int32_t known_kv_len = -1,
    int32_t known_q_len = -1, uint64_t fused_rtk_hdr = 0ULL)
{
    int32_t q_len = known_q_len;
    if (q_len < 0) {
        AscendC::GlobalTensor<int32_t> q_len_gm;
        q_len_gm.SetGlobalBuffer((__gm__ int32_t*)addresses.attention.main_metadata[3], 1);
        AscendC::DataCacheCleanAndInvalid<int32_t,
            AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(q_len_gm);
        q_len = q_len_gm.GetValue(0);
    }
    if (q_len <= 0) {
        return;
    }
    SetSysWorkspace((GM_ADDR)addresses.bridge[ffn_bridge::kSystem]);
    const auto rows = addresses.body.rows;
    const auto blocks = addresses.body.blocks;
    const int64_t active_rows = (q_len <= rows) ? q_len : rows;
    const int64_t norm_rows = active_rows;
    const auto& norm = addresses.norm;
    int64_t ts0 = GetSystemCycle();
    whole_layer_norm::device::Forward(pipe,
        (__gm__ bfloat16_t*)norm.input, (__gm__ bfloat16_t*)norm.residual,
        (__gm__ bfloat16_t*)norm.gamma, nullptr,
#ifdef K11040_F0_NZ
        indexed ? (__gm__ bfloat16_t*)norm.normalized : nullptr, (__gm__ bfloat16_t*)norm.retained,
        nullptr, (__gm__ int8_t*)norm.codes,
#else
        (__gm__ bfloat16_t*)norm.normalized, (__gm__ bfloat16_t*)norm.retained,
        (__gm__ float*)norm.rstd, (__gm__ int8_t*)norm.codes,
#endif
        (__gm__ float*)norm.scales, norm_rows, blocks, config.input_epsilon);
    int64_t ts1 = GetSystemCycle();

    __gm__ int64_t* attn_prof_ptr = (record_prof || prof_acc != nullptr)
        ? ((__gm__ int64_t*)addresses.body.scratch[routed_int4::kGateTiling] + 32)
        : nullptr;
    if (indexed) {
        const auto attention = IndexedArguments(addresses.attention);
        whole_layer_indexed::Forward(pipe, config.indexed, attention,
                                    config.bias_kind,
                                    (GM_ADDR)addresses.shared_topk,
                                    (__gm__ uint8_t*)communication.window, communication.capacity,
                                    communication.rank, communication.ranks, tree_mask_gm, attn_prof_ptr,
                                    resolved_topk_buf, inout_resolved_ok, known_kv_len, q_len, fused_rtk_hdr);
    } else {
        const auto& a = addresses.attention;
        const whole_layer_attention::DeviceArgs shared{
            (GM_ADDR)a.token_x, (GM_ADDR)a.token_scale,
            (GM_ADDR)a.main_weights[0], (GM_ADDR)a.main_weights[1],
            (GM_ADDR)a.main_weights[2], (GM_ADDR)a.main_weights[3],
            (GM_ADDR)a.main_weights[4], (GM_ADDR)a.main_weights[5],
            (GM_ADDR)a.main_weights[6], (GM_ADDR)a.main_weights[7],
            (GM_ADDR)a.main_weights[8], (GM_ADDR)a.main_metadata[0],
            (GM_ADDR)a.main_metadata[1], (GM_ADDR)a.main_metadata[2],
            (GM_ADDR)a.caches[0], (GM_ADDR)a.caches[1],
            (GM_ADDR)addresses.shared_topk, (GM_ADDR)a.main_metadata[3],
            (GM_ADDR)a.main_metadata[4], (GM_ADDR)a.main_metadata[5],
            nullptr, nullptr};
        whole_layer_attention::FullForward(pipe, config.shared, shared,
            (GM_ADDR)a.tail_weights[0], (GM_ADDR)a.tail_weights[1],
            (GM_ADDR)a.tail_weights[2], (GM_ADDR)a.bias, config.bias_kind,
            (GM_ADDR)a.scratch, (GM_ADDR)a.tiling, (GM_ADDR)a.output,
            (__gm__ uint8_t*)communication.window, communication.capacity,
            communication.rank, communication.ranks, tree_mask_gm, attn_prof_ptr,
            resolved_topk_buf, need_resolve_shared, inout_resolved_ok, known_kv_len, q_len);
    }
    int64_t ts2 = GetSystemCycle();
#ifdef ENABLE_KERNEL_PROF
    if (prof_acc != nullptr) {
        if ASCEND_IS_AIV {
            if (GetBlockIdx() == 0) {
                if (indexed) {
                    prof_acc[8]  += (attn_prof_ptr[1] - attn_prof_ptr[0]);
                    prof_acc[9]  += (attn_prof_ptr[2] - attn_prof_ptr[1]);
                    prof_acc[10] += (attn_prof_ptr[3] - attn_prof_ptr[2]);
                    prof_acc[11] += (attn_prof_ptr[4] - attn_prof_ptr[3]);
                    prof_acc[12] += (attn_prof_ptr[5] - attn_prof_ptr[4]);
                } else {
                    prof_acc[13] += (attn_prof_ptr[1] - attn_prof_ptr[0]);
                    prof_acc[14] += (attn_prof_ptr[4] - attn_prof_ptr[3]);
                    prof_acc[15] += (attn_prof_ptr[5] - attn_prof_ptr[4]);
                }
            }
        }
    }
#endif

#ifdef K11050_AIC_PREFETCH
    if ASCEND_IS_AIC {
#ifdef K112728_DROP_R1_GATE_PF
        const PrefetchSpan r1_spans[2] = {
            {addresses.body.shared.gate_weight, 3145728U},
            {addresses.body.shared.down_weight, 1572864U},
        };
        PrefetchSpansAic(pipe, r1_spans, 2);
#else
        const PrefetchSpan r1_spans[3] = {
            {addresses.gate, 3145728U},
            {addresses.body.shared.gate_weight, 3145728U},
            {addresses.body.shared.down_weight, 1572864U},
        };
        PrefetchSpansAic(pipe, r1_spans, 3);
#endif
    }
#endif
    ReducePhase(pipe, communication, addresses.attention.output,
                communication.attention_sum, active_rows, blocks, 0, 0);
    int64_t ts3 = GetSystemCycle();


    ffn_bridge::Addresses bridge{};
    for (int i = 0; i < ffn_bridge::kBuffers; ++i) {
        bridge.values[i] = (GM_ADDR)addresses.bridge[i];
    }
    SetSysWorkspace(bridge.values[ffn_bridge::kSystem]);
    const int64_t eff_shared_cap =
        (active_rows <= 16) ? 16 : addresses.body.shared.capacity;
#ifdef K11089_MTP_TOPK4
    const int64_t eff_topk = indexed ? 8 : 4;
#else
    const int64_t eff_topk = 8;
#endif
#ifdef K11220_PF_RED2_FULL
    __gm__ volatile int32_t* red2_done_gm =
        reinterpret_cast<__gm__ volatile int32_t*>(addresses.body.scratch[routed_int4::kGroupCount]) + 128;
    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0) {
            *red2_done_gm = 0;
            dcci_cacheline(
                reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ int32_t*>(red2_done_gm)));
        }
    }
#endif
    ffn_bridge::Forward(pipe, (GM_ADDR)communication.attention_sum, (GM_ADDR)norm.retained,
        (GM_ADDR)addresses.post_gamma, (GM_ADDR)addresses.gate,
        (GM_ADDR)addresses.correction_bias, bridge, active_rows, blocks,
        config.ffn_epsilon,
        (GM_ADDR)addresses.body.scratch[routed_int4::kRestoredRows],
        (GM_ADDR)addresses.body.scratch[routed_int4::kRestoredScale],
        addresses.body.shared.payload, eff_shared_cap, eff_topk);
    int64_t ts4 = GetSystemCycle();
    uint32_t rt_epoch_pf = 0U;
#ifdef K11100_OPP_PREFETCH
    if ASCEND_IS_AIC {
        if (communication.window != 0ULL) {
            rt_epoch_pf = whole_layer_tree::ReadCommunicationEpoch((__gm__ uint8_t*)communication.window) + 2U;
        }
    }
#endif
#if defined(K11080_DIRECT_W4A8) && defined(K11150_SHARED_W8_NZ)
    if (active_rows > 32 || blocks < 24) {
        SetSysWorkspace((GM_ADDR)addresses.body.scratch[
            routed_int4::kMatmulWorkspace]);
    }
#else
    SetSysWorkspace((GM_ADDR)addresses.body.scratch[
        routed_int4::kMatmulWorkspace]);
#endif
    auto body_args = addresses.body;
    body_args.rows = active_rows;
    body_args.shared.capacity = eff_shared_cap;
    routed_body::Forward<CubeFormat::NZ, true>(pipe, body_args,
        (__gm__ bfloat16_t*)bridge.values[ffn_bridge::kScaledRouted],
        (__gm__ bfloat16_t*)bridge.values[ffn_bridge::kCombined],
        communication.rank * routed_int4::kLocalExperts, false, false,
        nullptr, nullptr, nullptr,
        (__gm__ uint8_t*)communication.window, communication.capacity,
        communication.rank, communication.ranks, layer_pw,
        (prof_acc != nullptr) ? (prof_acc + 16) : nullptr);


    if ASCEND_IS_AIC {
        pipe.Reset();
    }
    if ASCEND_IS_AIV {
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
        SyncAll<true>();
        pipe.Reset();
    }
    int64_t ts5 = GetSystemCycle();
    int64_t ts5b = ts5;
#if defined(K11050_AIC_PREFETCH) && !defined(K112800_NO_R2_PF)
    if ASCEND_IS_AIC {
        if (next_step != nullptr) {
            const auto& na = next_step->addresses.attention;
            const bool n_idx = ((next_step->indexed & 1ULL) != 0ULL);
#ifdef K112802_MAIN_R2_PF
            const PrefetchSpan r2_spans[4] = {
                {na.main_weights[0], 12582912U},
                {na.main_weights[1], 4194304U},
                {na.main_weights[2], 1572864U},
                {na.main_weights[3], 3538944U},
            };
            PrefetchSpansAic(pipe, r2_spans, 4, &communication, rt_epoch_pf, false, red2_done_gm);
#else
            const PrefetchSpan r2_spans[8] = {
                {na.main_weights[0], 12582912U},
                {na.main_weights[1], 4194304U},
                {na.main_weights[2], 1572864U},
                {na.main_weights[3], 3538944U},
                {na.tail_weights[0], 2097152U},
                {na.tail_weights[1], 12582912U},
                {n_idx ? na.front_weights[0] : 0ULL, 8388608U},
                {n_idx ? na.front_weights[2] : 0ULL, 1966080U},
            };
#ifdef K11220_PF_RED2_FULL
            PrefetchSpansAic(pipe, r2_spans, 8, &communication, rt_epoch_pf, false, red2_done_gm);
#elif defined(K11220_RS768_PUSH)
            PrefetchSpansAic(pipe, r2_spans, 8, &communication, rt_epoch_pf, true);
#else
            PrefetchSpansAic(pipe, r2_spans, 8, &communication, rt_epoch_pf, false);
#endif
#endif
        }
    }
#endif
#ifdef K11180_LITE_MIRROR
    ReducePhase(pipe, communication, addresses.body.scratch[routed_int4::kRouteOutput],
                communication.output, active_rows, blocks,
                addresses.body.shared.output, 0,
                addresses.bridge[ffn_bridge::kScaledRouted],
                addresses.bridge[ffn_bridge::kIds],
                addresses.bridge[ffn_bridge::kProbabilities],
                layer_pw);
#else
    ReducePhase(pipe, communication, addresses.body.scratch[routed_int4::kRouteOutput],
                communication.output, active_rows, blocks,
                addresses.body.shared.output, 0,
                addresses.bridge[ffn_bridge::kScaledRouted],
                addresses.bridge[ffn_bridge::kIds],
                addresses.bridge[ffn_bridge::kProbabilities],
                (layer_pw != 0ULL) ? addresses.body.scratch[routed_int4::kDownTiling] : 0ULL,
                prof_acc);
#endif
#ifdef K11220_PF_RED2_FULL
    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0) {
            *red2_done_gm = 1;
            dcci_cacheline(
                reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ int32_t*>(red2_done_gm)));
        }
    }
#endif
    if (prof_acc != nullptr) {
        const int64_t cyc6 = GetSystemCycle();
        prof_acc[0] += (ts1 - ts0);
        prof_acc[indexed ? 1 : 2] += (ts2 - ts1);
        prof_acc[3] += (ts3 - ts2);
        prof_acc[4] += (ts4 - ts3);
        prof_acc[5] += (ts5 - ts4);
        prof_acc[6] += (cyc6 - ts5);
    }
    int64_t ts6 = GetSystemCycle();
    if (record_prof) { if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0) {
            GlobalTensor<int64_t> prof;
            prof.SetGlobalBuffer((__gm__ int64_t*)addresses.body.scratch[routed_int4::kGateTiling] + 40, 16);
            prof.SetValue(0, ts0);
            prof.SetValue(1, ts1);
            prof.SetValue(2, ts2);
            prof.SetValue(3, ts3);
            prof.SetValue(4, ts4);
            prof.SetValue(5, ts5);
            prof.SetValue(7, ts5b);
            prof.SetValue(6, ts6);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(prof[0]);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(prof[8]);
        }
    } }
}

#ifndef SKIP_SINGLE_KERNELS
template <bool indexed, int Ver = KERNEL_VER>
__global__ __mix__(1, 2) __schedmode__(1) __aicore__
void Kernel(uint64_t sync, Config config, Addresses addresses,
            Communication communication)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync);
    TPipe pipe;
#ifdef ENABLE_KERNEL_PROF
    RunLayer(pipe, indexed, config, addresses, communication, true, 0);
#else
    RunLayer(pipe, indexed, config, addresses, communication, false, 0);
#endif
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
}
#endif

#ifdef K11220_DENSE78_COLD
__aicore__ __attribute__((noinline)) void RunDenseLayer(
    TPipe& pipe, bool indexed, bool pre_normed, const Config& config,
    const Addresses& addresses, const Communication& communication,
    uint64_t tree_mask_gm, const ChainStep* next_step, uint64_t layer_pw,
    int64_t* prof_acc);
#endif

#ifndef SKIP_CHAIN_KERNEL
template <int Ver = KERNEL_VER>
__global__ __mix__(1, 2) __schedmode__(1) __aicore__
void ChainKernel(uint64_t steps_gm, int32_t count, uint64_t tree_mask_gm, uint64_t peer_weights_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    if (count <= 0) {
        return;
    }
    const uint64_t top_ql_ptr =
        reinterpret_cast<const ChainStep*>(steps_gm)[0].addresses.attention.main_metadata[3];
    if (top_ql_ptr != 0ULL) {
        AscendC::GlobalTensor<int32_t> top_q_gm;
        top_q_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(top_ql_ptr), 1);
        AscendC::DataCacheCleanAndInvalid<int32_t,
            AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(top_q_gm);
        if (top_q_gm.GetValue(0) <= 0) {
            return;
        }
    }
    TPipe pipe;
    uint64_t current_sync = 0;
#ifdef ENABLE_KERNEL_PROF
    int64_t prof_acc[24];
    for (int i = 0; i < 24; ++i) { prof_acc[i] = 0; }
    const int64_t t_chain_start = GetSystemCycle();
#endif
    int32_t i = 0;
#ifdef K11220_DENSE78_COLD
    for (; i < count; ++i) {
        const ChainStep& step = reinterpret_cast<const ChainStep*>(steps_gm)[i];
        const uint64_t sflags = step.indexed;
        if ((sflags & 2ULL) == 0ULL) {
            break;
        }
        if (i == 0 || step.sync != current_sync) {
            SetSyncBaseAddr(step.sync);
            current_sync = step.sync;
        }
        const bool last = (i == count - 1);
        const ChainStep* next_ptr = last ? nullptr : (reinterpret_cast<const ChainStep*>(steps_gm) + (i + 1));
        const uint64_t layer_pw = (peer_weights_gm != 0ULL)
            ? (peer_weights_gm + static_cast<uint64_t>(i) * sizeof(whole_layer_tree::LayerMirrorMeta))
            : 0ULL;
#ifdef ENABLE_KERNEL_PROF
        RunDenseLayer(pipe, (sflags & 1ULL) != 0ULL, (sflags & 4ULL) != 0ULL, step.config, step.addresses, step.communication, tree_mask_gm, next_ptr, layer_pw, prof_acc);
#else
        RunDenseLayer(pipe, (sflags & 1ULL) != 0ULL, (sflags & 4ULL) != 0ULL, step.config, step.addresses, step.communication, tree_mask_gm, next_ptr, layer_pw, nullptr);
#endif
    }
#endif

    GM_ADDR resolved_topk_buf = nullptr;
    uint64_t cached_topk_src = 0ULL;
    uint64_t cached_pt_src = 0ULL;
    uint64_t cached_kv_src = 0ULL;
    uint64_t init_ql_ptr = 0ULL;
    int32_t init_ql_val = -1;
    uint64_t init_kvl_ptr = 0ULL;
    int32_t init_kvl_val = -1;
    uint64_t init_pt_ptr = 0ULL;
    uint32_t init_tp = 0U;
    uint64_t my_fused_hdr = 0ULL;
    bool resolved_topk_ok = false;
#if defined(K112130_VEC_RESOLVED_TOPK) && defined(K11080_DIRECT_W4A8) && defined(K11150_SHARED_W8_NZ)
    if (i < count) {
        const ChainStep& first_step = reinterpret_cast<const ChainStep*>(steps_gm)[i];
        init_ql_ptr = first_step.addresses.attention.main_metadata[3];
        if (init_ql_ptr != 0ULL) {
            AscendC::GlobalTensor<int32_t> q_gm;
            q_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(init_ql_ptr), 1);
            AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(q_gm);
            init_ql_val = q_gm.GetValue(0);
        }
        init_kvl_ptr = first_step.addresses.attention.main_metadata[4];
        if (init_kvl_ptr != 0ULL) {
            AscendC::GlobalTensor<int32_t> kv_gm;
            kv_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(init_kvl_ptr), 1);
            AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(kv_gm);
            init_kvl_val = kv_gm.GetValue(0);
        }
        init_pt_ptr = first_step.addresses.attention.main_metadata[5];
        init_tp = first_step.config.indexed.attention.table_pages;
        if (first_step.addresses.body.rows >= 1 && first_step.addresses.body.rows <= 32) {
            const uint64_t rtk_addr = first_step.addresses.body.scratch[routed_int4::kMatmulWorkspace] + (15ULL << 20);
            resolved_topk_buf = reinterpret_cast<GM_ADDR>(rtk_addr);
#if defined(K112162_FUSED_IDX_RTK) && defined(K112120_CLEAN_DCCI)
            const int64_t cap_rows = first_step.addresses.body.rows;
            const uint32_t eff_rows = static_cast<uint32_t>((init_ql_val > 0 && init_ql_val <= cap_rows) ? init_ql_val : cap_rows);
            if ASCEND_IS_AIV {
                const uint32_t vid = static_cast<uint32_t>(GetBlockIdx());
                if (vid < 48U) {
                    const uint64_t hdr_addr = rtk_addr - 4096ULL + static_cast<uint64_t>(vid) * 64ULL;
                    __gm__ uint64_t* hdr_ptr = reinterpret_cast<__gm__ uint64_t*>(hdr_addr);
                    hdr_ptr[0] = rtk_addr; hdr_ptr[1] = init_pt_ptr; hdr_ptr[2] = tree_mask_gm;
                    hdr_ptr[3] = static_cast<uint64_t>(static_cast<uint32_t>(init_kvl_val));
                    hdr_ptr[4] = static_cast<uint64_t>(eff_rows); hdr_ptr[5] = static_cast<uint64_t>(init_tp);
                    my_fused_hdr = hdr_addr;
                }
            } else {
                my_fused_hdr = rtk_addr - 4096ULL;
            }
#endif
        }
    }
#endif

    for (; i < count; ++i) {
        const ChainStep& step = reinterpret_cast<const ChainStep*>(steps_gm)[i];
        if (i == 0 || step.sync != current_sync) {
            SetSyncBaseAddr(step.sync);
            current_sync = step.sync;
        }
        const bool last = (i == count - 1);
        const ChainStep* next_ptr = last ? nullptr : (reinterpret_cast<const ChainStep*>(steps_gm) + (i + 1));
#ifdef K11180_LITE_MIRROR
        const uint64_t layer_pw = (peer_weights_gm != 0ULL)
            ? (peer_weights_gm + static_cast<uint64_t>(i) * sizeof(whole_layer_tree::LiteMirrorMeta))
            : 0ULL;
#else
        const uint64_t layer_pw = (peer_weights_gm != 0ULL)
            ? (peer_weights_gm + static_cast<uint64_t>(i) * sizeof(whole_layer_tree::LayerMirrorMeta))
            : 0ULL;
#endif
        const bool step_is_idx = ((step.indexed & 1ULL) != 0ULL);
        bool need_resolve_shared = false;
        const int32_t step_q_len = (step.addresses.attention.main_metadata[3] == init_ql_ptr) ? init_ql_val : -1;
        const int32_t step_kv_len = (step.addresses.attention.main_metadata[4] == init_kvl_ptr) ? init_kvl_val : -1;
        uint64_t step_fused_hdr = 0ULL;
#if defined(K112130_VEC_RESOLVED_TOPK) && defined(K11080_DIRECT_W4A8) && defined(K11150_SHARED_W8_NZ)
        if (resolved_topk_buf != nullptr) {
            const uint64_t cur_pt = step.addresses.attention.main_metadata[5];
            const uint64_t cur_kv = step.addresses.attention.main_metadata[4];
            if (step_is_idx) {
                const uint64_t cur_src = (step.addresses.shared_topk != 0ULL) ? step.addresses.shared_topk : (step.addresses.attention.scratch + step.config.indexed.topk);
                cached_topk_src = (step.addresses.shared_topk != 0ULL) ? cur_src : 0ULL;
                cached_pt_src = cur_pt; cached_kv_src = cur_kv;
#ifdef K112162_FUSED_IDX_RTK
                if (my_fused_hdr != 0ULL && cur_pt == init_pt_ptr && cur_kv == init_kvl_ptr && step.addresses.attention.main_metadata[3] == init_ql_ptr && step.config.indexed.attention.table_pages == init_tp) {
                    step_fused_hdr = my_fused_hdr;
                }
#endif
            } else {
                const uint64_t cur_src = step.addresses.shared_topk;
                if (cur_src != cached_topk_src || cur_pt != cached_pt_src || cur_kv != cached_kv_src) {
                    need_resolve_shared = true;
                    cached_topk_src = cur_src; cached_pt_src = cur_pt; cached_kv_src = cur_kv;
                }
            }
        }
#endif
#ifdef ENABLE_KERNEL_PROF
        RunLayer(pipe, step_is_idx, step.config, step.addresses, step.communication, false, tree_mask_gm, next_ptr, layer_pw, prof_acc, resolved_topk_buf, need_resolve_shared, &resolved_topk_ok, step_kv_len, step_q_len, step_fused_hdr);
#else
        RunLayer(pipe, step_is_idx, step.config, step.addresses, step.communication, false, tree_mask_gm, next_ptr, layer_pw, nullptr, resolved_topk_buf, need_resolve_shared, &resolved_topk_ok, step_kv_len, step_q_len, step_fused_hdr);
#endif
    }
#ifdef ENABLE_KERNEL_PROF
    prof_acc[7] = GetSystemCycle() - t_chain_start;
    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0 && count > 0) {
            const ChainStep& last_step = reinterpret_cast<const ChainStep*>(steps_gm)[count - 1];
            GlobalTensor<int64_t> prof_out;
            prof_out.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(last_step.addresses.body.scratch[routed_int4::kGateTiling]) + 40, 24);
            for (int32_t k = 0; k < 24; ++k) {
                prof_out.SetValue(k, prof_acc[k]);
            }
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(prof_out[0]);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(prof_out[8]);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(prof_out[16]);
        }
    }
#endif
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
}
#endif

#ifdef K11220_DENSE78_COLD
__aicore__ inline void DirectDenseGateUp24Aic(
    TPipe& pipe, __gm__ int8_t* input, __gm__ int8_t* weights,
    __gm__ int32_t* output, int64_t proj_rows)
{
    using namespace AscendC;
    const uint32_t cid = static_cast<uint32_t>(GetBlockIdx());
    if (cid >= 24U) {
        return;
    }
    const bool dual_m = (proj_rows > 16);
    const uint16_t m0 = dual_m ? 16U : static_cast<uint16_t>(proj_rows);
    const uint16_t m1 = dual_m ? static_cast<uint16_t>(proj_rows - 16) : 0U;
    const uint32_t m_cap = dual_m ? 32U : 16U;

    TBuf<TPosition::B1> b1_buf0, b1_buf1;
    TBuf<TPosition::A1> a1_buf;
    TBuf<TPosition::A2> l0a_buf;
    TBuf<TPosition::B2> l0b_buf0, l0b_buf1;
    TBuf<TPosition::CO1> l0c_buf;
    pipe.InitBuffer(b1_buf0, 65536);
    pipe.InitBuffer(b1_buf1, 65536);
    pipe.InitBuffer(a1_buf, m_cap * 1024U);
    pipe.InitBuffer(l0a_buf, m_cap * 1024U);
    pipe.InitBuffer(l0b_buf0, 256 * 128);
    pipe.InitBuffer(l0b_buf1, 256 * 128);
    pipe.InitBuffer(l0c_buf, m_cap * 128U * sizeof(int32_t));

    LocalTensor<int8_t> a1 = a1_buf.Get<int8_t>();
    LocalTensor<int8_t> b1[2] = {b1_buf0.Get<int8_t>(), b1_buf1.Get<int8_t>()};
    LocalTensor<int8_t> l0a = l0a_buf.Get<int8_t>();
    LocalTensor<int8_t> l0b[2] = {l0b_buf0.Get<int8_t>(), l0b_buf1.Get<int8_t>()};
    LocalTensor<int32_t> l0c = l0c_buf.Get<int32_t>();

    GlobalTensor<int8_t> a_gm, b_gm;
    GlobalTensor<int32_t> c_gm;
    a_gm.SetGlobalBuffer(input, static_cast<uint64_t>(proj_rows) * 6144ULL);
    b_gm.SetGlobalBuffer(weights + static_cast<uint64_t>(cid) * 786432ULL, 786432ULL);
    b_gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    c_gm.SetGlobalBuffer(output + static_cast<uint64_t>(cid) * 128ULL, static_cast<uint64_t>(proj_rows) * 3072ULL);

    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID5);

    LoadData2DParams load_a{};
    load_a.repeatTimes = dual_m ? 64 : 32;
    load_a.srcStride = 1;
    load_a.ifTranspose = false;
    LoadData2dTransposeParams trans(0, 4, 16, 1, 0, inc);

    for (uint32_t kp = 0U; kp < 6U; ++kp) {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        Nd2NzParams copy_a{};
        copy_a.ndNum = 1;
        copy_a.nValue = m0;
        copy_a.dValue = 1024;
        copy_a.srcDValue = 6144;
        copy_a.dstNzC0Stride = 16;
        copy_a.dstNzNStride = 1;
        DataCopy(a1, a_gm[kp * 1024ULL], copy_a);
        if (dual_m) {
            copy_a.nValue = m1;
            DataCopy(a1[16384U], a_gm[16ULL * 6144ULL + kp * 1024ULL], copy_a);
        }
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);

        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        DataCopy(b1[0], b_gm[kp * 131072ULL], 65536);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        DataCopy(b1[1], b_gm[kp * 131072ULL + 65536ULL], 65536);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);

        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID5);
        LoadData(l0a, a1, load_a);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID5);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID5);

        for (uint32_t step = 0U; step < 4U; ++step) {
            const uint32_t bi = step >> 1U;
            const uint32_t k_sub_off = (step & 1U) * 8192U;
            const uint32_t lbi = step & 1U;
            const event_t lb_evt = lbi ? EVENT_ID4 : EVENT_ID3;
            if (step == 0U) {
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
            } else if (step == 2U) {
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
            }
            WaitFlag<HardEvent::M_MTE1>(lb_evt);
            #pragma unroll
            for (uint32_t i = 0U; i < 8U; ++i) {
                LoadDataWithTranspose(l0b[lbi][i * 4096U], b1[bi][k_sub_off + i * 1024U], trans);
            }
            if (step == 1U) {
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            } else if (step == 3U) {
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
            }
            SetFlag<HardEvent::MTE1_M>(lb_evt);
            WaitFlag<HardEvent::MTE1_M>(lb_evt);
            MmadParams mad{};
            mad.m = 16;
            mad.n = 128;
            mad.k = 256;
            mad.cmatrixSource = false;
            mad.cmatrixInitVal = (kp == 0U && step == 0U);
            Mmad(l0c, l0a[step * 4096U], l0b[lbi], mad);
            if (dual_m) {
                Mmad(l0c[2048U], l0a[16384U + step * 4096U], l0b[lbi], mad);
            }
            PipeBarrier<PIPE_M>();
            SetFlag<HardEvent::M_MTE1>(lb_evt);
        }
        SetFlag<HardEvent::M_MTE1>(EVENT_ID5);
    }

    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID5);
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    FixpipeParamsV220 fix{};
    fix.nSize = 128;
    fix.mSize = m0;
    fix.srcStride = 16;
    fix.dstStride = 3072;
    fix.ndNum = 1;
    fix.quantPre = QuantMode_t::NoQuant;
    Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(c_gm, l0c, fix);
    if (dual_m) {
        fix.mSize = m1;
        Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(c_gm[16ULL * 3072ULL], l0c[2048U], fix);
    }
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    pipe.Reset();
}

__aicore__ inline void ActivateDenseSwigluQuant(
    TPipe& pipe, __gm__ int32_t* gu_accum, __gm__ float* gu_scale,
    __gm__ float* tok_scales, __gm__ int8_t* mid_i8, __gm__ float* mid_scale,
    int64_t active_rows, int64_t capacity)
{
    using namespace AscendC;
    const int64_t row = GetBlockIdx();
    if (row >= active_rows) {
        return;
    }
    TBuf<TPosition::VECCALC> acc_buf, wsc_buf, val_buf, wrk_buf, oth_buf, bf16_buf, sc_buf, cod_buf;
    pipe.InitBuffer(acc_buf, 3072 * sizeof(int32_t));
    pipe.InitBuffer(wsc_buf, (3072 + 8) * sizeof(float));
    pipe.InitBuffer(val_buf, 3072 * sizeof(float));
    pipe.InitBuffer(wrk_buf, 3072 * sizeof(float));
    pipe.InitBuffer(oth_buf, 1536 * sizeof(float));
    pipe.InitBuffer(bf16_buf, 3072 * sizeof(bfloat16_t));
    pipe.InitBuffer(sc_buf, 8 * sizeof(float));
    pipe.InitBuffer(cod_buf, 1536);

    auto acc = acc_buf.Get<int32_t>();
    auto wsc = wsc_buf.Get<float>();
    auto val = val_buf.Get<float>();
    auto wrk = wrk_buf.Get<float>();
    auto oth = oth_buf.Get<float>();
    auto bf16 = bf16_buf.Get<bfloat16_t>();
    auto scalar = sc_buf.Get<float>();
    auto codes = cod_buf.Get<int8_t>();

    GlobalTensor<int32_t> acc_gm;
    GlobalTensor<float> wsc_gm, tsc_gm, msc_gm;
    GlobalTensor<int8_t> mid_gm;
    acc_gm.SetGlobalBuffer(gu_accum + row * 3072, 3072);
    wsc_gm.SetGlobalBuffer(gu_scale, 3072);
    tsc_gm.SetGlobalBuffer(tok_scales + row, 1);
    mid_gm.SetGlobalBuffer(mid_i8, capacity * 1536);
    msc_gm.SetGlobalBuffer(mid_scale, capacity * 8);

    DataCopy(acc, acc_gm, 3072);
    DataCopy(wsc, wsc_gm, 3072);
    DataCopyPad(wsc[3072], tsc_gm, DataCopyExtParams{1, sizeof(float), 0, 0, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

    const float tok_s = wsc.GetValue(3072);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);

    Cast(val, acc, RoundMode::CAST_RINT, 3072);
    Muls(wrk, wsc, tok_s, 3072);
    PipeBarrier<PIPE_V>();
    Mul(val, val, wrk, 3072);
    PipeBarrier<PIPE_V>();
    Cast(bf16, val, RoundMode::CAST_RINT, 3072);
    PipeBarrier<PIPE_V>();

    Cast(val, bf16, RoundMode::CAST_NONE, 3072);
    PipeBarrier<PIPE_V>();
    Muls(wrk, val, -1.0F, 1536);
    PipeBarrier<PIPE_V>();
    Exp(wrk, wrk, 1536);
    PipeBarrier<PIPE_V>();
    Adds(wrk, wrk, 1.0F, 1536);
    PipeBarrier<PIPE_V>();
    Div(val, val, wrk, 1536);
    PipeBarrier<PIPE_V>();
    Mul(val, val, val[1536], 1536);
    PipeBarrier<PIPE_V>();
    Cast(bf16, val, RoundMode::CAST_RINT, 1536);
    PipeBarrier<PIPE_V>();

    Cast(val, bf16, RoundMode::CAST_NONE, 1536);
    PipeBarrier<PIPE_V>();
    Abs(oth, val, 1536);
    Duplicate(scalar, 0.0F, 8);
    PipeBarrier<PIPE_V>();
    ReduceMax(scalar, oth, wrk, 1536, false);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const float maximum = scalar.GetValue(0);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(wrk, maximum == 0.0F ? 1.0F : maximum, 64);
    Duplicate(oth, 127.0F, 64);
    PipeBarrier<PIPE_V>();
    Div(oth, oth, wrk, 64);
    PipeBarrier<PIPE_V>();
    Mul(val, val, oth, 64, 24, BinaryRepeatParams{1, 1, 0, 8, 8, 0});
    Muls(scalar, scalar, 1.0F / 127.0F, 8);
    PipeBarrier<PIPE_V>();
    Mins(val, val, 127.0F, 1536);
    PipeBarrier<PIPE_V>();
    Maxs(val, val, -127.0F, 1536);
    PipeBarrier<PIPE_V>();
    Cast(acc, val, RoundMode::CAST_RINT, 1536);
    PipeBarrier<PIPE_V>();
    SetDeqScale(static_cast<half>(1.0F));
    PipeBarrier<PIPE_V>();
    auto fp16 = bf16.ReinterpretCast<half>();
    Cast(fp16, acc, RoundMode::CAST_ROUND, 1536);
    PipeBarrier<PIPE_V>();
    Cast(codes, fp16, RoundMode::CAST_TRUNC, 1536);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(mid_gm[row * 1536], codes, 1536);
    DataCopy(msc_gm[row * 8], scalar, 8);
    PipeBarrier<PIPE_ALL>();
    pipe.Reset();
}

__aicore__ inline void DirectDenseDown24Aic(
    TPipe& pipe, __gm__ int8_t* input, __gm__ int8_t* weights,
    __gm__ int32_t* output, int64_t capacity, int64_t proj_rows)
{
    using namespace AscendC;
    const uint32_t cid = static_cast<uint32_t>(GetBlockIdx());
    if (cid >= 24U) {
        return;
    }
    const bool dual_m = (proj_rows > 16);
    const uint16_t m0 = dual_m ? 16U : static_cast<uint16_t>(proj_rows);
    const uint16_t m1 = dual_m ? static_cast<uint16_t>(proj_rows - 16) : 0U;
    const uint32_t m_cap = dual_m ? 32U : 16U;

    TBuf<TPosition::B1> b1_buf0, b1_buf1;
    TBuf<TPosition::A1> a1_buf;
    TBuf<TPosition::A2> l0a_buf;
    TBuf<TPosition::B2> l0b_buf0, l0b_buf1;
    TBuf<TPosition::CO1> l0c_buf;
    pipe.InitBuffer(b1_buf0, 32768);
    pipe.InitBuffer(b1_buf1, 32768);
    pipe.InitBuffer(a1_buf, m_cap * 1536U);
    pipe.InitBuffer(l0a_buf, m_cap * 1536U);
    pipe.InitBuffer(l0b_buf0, 128 * 256);
    pipe.InitBuffer(l0b_buf1, 128 * 256);
    pipe.InitBuffer(l0c_buf, m_cap * 256U * sizeof(int32_t));

    LocalTensor<int8_t> a1 = a1_buf.Get<int8_t>();
    LocalTensor<int8_t> b1[2] = {b1_buf0.Get<int8_t>(), b1_buf1.Get<int8_t>()};
    LocalTensor<int8_t> l0a = l0a_buf.Get<int8_t>();
    LocalTensor<int8_t> l0b[2] = {l0b_buf0.Get<int8_t>(), l0b_buf1.Get<int8_t>()};
    LocalTensor<int32_t> l0c = l0c_buf.Get<int32_t>();

    GlobalTensor<int8_t> a_gm, b_gm;
    GlobalTensor<int32_t> c_gm;
    a_gm.SetGlobalBuffer(input, static_cast<uint64_t>(proj_rows) * 1536ULL);
    b_gm.SetGlobalBuffer(weights + static_cast<uint64_t>(cid) * 393216ULL, 393216ULL);
    b_gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    c_gm.SetGlobalBuffer(output + static_cast<uint64_t>(cid) * static_cast<uint64_t>(capacity) * 256ULL,
                         static_cast<uint64_t>(capacity) * 256ULL);

    Nd2NzParams copy_a{};
    copy_a.ndNum = 1;
    copy_a.nValue = m0;
    copy_a.dValue = 1536;
    copy_a.srcDValue = 1536;
    copy_a.dstNzC0Stride = 16;
    copy_a.dstNzNStride = 1;
    DataCopy(a1, a_gm, copy_a);
    if (dual_m) {
        copy_a.nValue = m1;
        DataCopy(a1[24576U], a_gm[24576U], copy_a);
    }
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);

    LoadData2DParams load_a{};
    load_a.repeatTimes = dual_m ? 96 : 48;
    load_a.srcStride = 1;
    load_a.ifTranspose = false;
    LoadData(l0a, a1, load_a);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);

    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
    LoadData2dTransposeParams trans(0, 8, 4, 1, 0, inc);

    for (uint32_t kc = 0U; kc < 6U; ++kc) {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        DataCopy(b1[0], b_gm[kc * 65536ULL], 32768);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        DataCopy(b1[1], b_gm[kc * 65536ULL + 32768ULL], 32768);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);

        for (uint32_t step = 0U; step < 2U; ++step) {
            const event_t b_evt = step ? EVENT_ID1 : EVENT_ID0;
            const event_t lb_evt = step ? EVENT_ID4 : EVENT_ID3;
            WaitFlag<HardEvent::MTE2_MTE1>(b_evt);
            WaitFlag<HardEvent::M_MTE1>(lb_evt);
            #pragma unroll
            for (uint32_t i = 0U; i < 4U; ++i) {
                LoadDataWithTranspose(l0b[step][i * 8192U], b1[step][i * 1024U], trans);
            }
            SetFlag<HardEvent::MTE1_MTE2>(b_evt);
            SetFlag<HardEvent::MTE1_M>(lb_evt);
            WaitFlag<HardEvent::MTE1_M>(lb_evt);
            MmadParams mad{};
            mad.m = 16;
            mad.n = 256;
            mad.k = 128;
            mad.cmatrixSource = false;
            mad.cmatrixInitVal = (kc == 0U && step == 0U);
            const uint32_t a_off = (kc * 2U + step) * 2048U;
            Mmad(l0c, l0a[a_off], l0b[step], mad);
            if (dual_m) {
                Mmad(l0c[4096U], l0a[24576U + a_off], l0b[step], mad);
            }
            PipeBarrier<PIPE_M>();
            SetFlag<HardEvent::M_MTE1>(lb_evt);
        }
    }

    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    FixpipeParamsV220 fix{};
    fix.nSize = 256;
    fix.mSize = m0;
    fix.srcStride = 16;
    fix.dstStride = 256;
    fix.ndNum = 1;
    fix.quantPre = QuantMode_t::NoQuant;
    Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(c_gm, l0c, fix);
    if (dual_m) {
        fix.mSize = m1;
        Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(c_gm[4096U], l0c[4096U], fix);
    }
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    pipe.Reset();
}

__aicore__ __attribute__((noinline)) void RunDenseLayer(
    TPipe& pipe, bool indexed, bool pre_normed, const Config& config,
    const Addresses& addresses, const Communication& communication,
    uint64_t tree_mask_gm, const ChainStep* next_step, uint64_t layer_pw,
    int64_t* prof_acc)
{
    using namespace AscendC;
    GlobalTensor<int32_t> q_len_gm;
    q_len_gm.SetGlobalBuffer((__gm__ int32_t*)addresses.attention.main_metadata[3], 1);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(q_len_gm);
    const int32_t q_len = q_len_gm.GetValue(0);
    if (q_len <= 0) {
        return;
    }
    SetSysWorkspace((GM_ADDR)addresses.bridge[ffn_bridge::kSystem]);
    const auto rows = addresses.body.rows;
    const auto blocks = addresses.body.blocks;
    const int64_t active_rows = (q_len <= rows) ? q_len : rows;
    const auto& norm = addresses.norm;

    GlobalTensor<uint64_t> dw_gm;
    dw_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(layer_pw + 128ULL), 8);
    DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(dw_gm);
    const uint64_t gu_w_ptr = dw_gm.GetValue(0);
    const uint64_t gu_s_ptr = dw_gm.GetValue(1);
    const uint64_t dn_w_ptr = dw_gm.GetValue(2);
    const uint64_t dn_s_ptr = dw_gm.GetValue(3);

    int64_t ts0 = GetSystemCycle();
    if (pre_normed) {
        if ASCEND_IS_AIV {
            const int64_t vector = GetBlockIdx();
            for (int64_t row = vector; row < active_rows; row += 2 * blocks) {
                whole_layer_norm::device::DynamicQuantRow quant;
                quant.Init(pipe);
                quant.Process((__gm__ bfloat16_t*)norm.input + row * 6144,
                              (__gm__ int8_t*)norm.codes + row * 6144,
                              (__gm__ float*)norm.scales + row);
                TBuf<TPosition::VECCALC> cp_buf;
                pipe.InitBuffer(cp_buf, 6144 * sizeof(bfloat16_t));
                auto cp_ub = cp_buf.Get<bfloat16_t>();
                GlobalTensor<bfloat16_t> res_src, ret_dst;
                res_src.SetGlobalBuffer((__gm__ bfloat16_t*)norm.residual + row * 6144, 6144);
                ret_dst.SetGlobalBuffer((__gm__ bfloat16_t*)norm.retained + row * 6144, 6144);
                DataCopy(cp_ub, res_src, 6144);
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                DataCopy(ret_dst, cp_ub, 6144);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                if (indexed) {
                    GlobalTensor<bfloat16_t> in_src, nrm_dst;
                    in_src.SetGlobalBuffer((__gm__ bfloat16_t*)norm.input + row * 6144, 6144);
                    nrm_dst.SetGlobalBuffer((__gm__ bfloat16_t*)norm.normalized + row * 6144, 6144);
                    DataCopy(cp_ub, in_src, 6144);
                    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                    DataCopy(nrm_dst, cp_ub, 6144);
                }
                PipeBarrier<PIPE_ALL>();
                pipe.Reset();
            }
            DataSyncBarrier<MemDsbT::DDR>();
        }
        SyncAll<false>();
    } else {
        whole_layer_norm::device::Forward(pipe,
            (__gm__ bfloat16_t*)norm.input, (__gm__ bfloat16_t*)norm.residual,
            (__gm__ bfloat16_t*)norm.gamma, nullptr,
            indexed ? (__gm__ bfloat16_t*)norm.normalized : nullptr,
            (__gm__ bfloat16_t*)norm.retained,
            nullptr, (__gm__ int8_t*)norm.codes,
            (__gm__ float*)norm.scales, active_rows, blocks, config.input_epsilon);
    }
    int64_t ts1 = GetSystemCycle();

    __gm__ int64_t* attn_prof_ptr = (prof_acc != nullptr)
        ? ((__gm__ int64_t*)addresses.body.scratch[routed_int4::kGateTiling] + 32)
        : nullptr;
    if (indexed) {
        const auto attention = IndexedArguments(addresses.attention);
        whole_layer_indexed::Forward(pipe, config.indexed, attention,
                                    config.bias_kind,
                                    (GM_ADDR)addresses.shared_topk,
                                    (__gm__ uint8_t*)communication.window, communication.capacity,
                                    communication.rank, communication.ranks, tree_mask_gm, attn_prof_ptr);
    } else {
        const auto& a = addresses.attention;
        const whole_layer_attention::DeviceArgs shared{
            (GM_ADDR)a.token_x, (GM_ADDR)a.token_scale,
            (GM_ADDR)a.main_weights[0], (GM_ADDR)a.main_weights[1],
            (GM_ADDR)a.main_weights[2], (GM_ADDR)a.main_weights[3],
            (GM_ADDR)a.main_weights[4], (GM_ADDR)a.main_weights[5],
            (GM_ADDR)a.main_weights[6], (GM_ADDR)a.main_weights[7],
            (GM_ADDR)a.main_weights[8], (GM_ADDR)a.main_metadata[0],
            (GM_ADDR)a.main_metadata[1], (GM_ADDR)a.main_metadata[2],
            (GM_ADDR)a.caches[0], (GM_ADDR)a.caches[1],
            (GM_ADDR)addresses.shared_topk, (GM_ADDR)a.main_metadata[3],
            (GM_ADDR)a.main_metadata[4], (GM_ADDR)a.main_metadata[5],
            nullptr, nullptr};
        whole_layer_attention::FullForward(pipe, config.shared, shared,
            (GM_ADDR)a.tail_weights[0], (GM_ADDR)a.tail_weights[1],
            (GM_ADDR)a.tail_weights[2], (GM_ADDR)a.bias, config.bias_kind,
            (GM_ADDR)a.scratch, (GM_ADDR)a.tiling, (GM_ADDR)a.output,
            (__gm__ uint8_t*)communication.window, communication.capacity,
            communication.rank, communication.ranks, tree_mask_gm, attn_prof_ptr);
    }
    int64_t ts2 = GetSystemCycle();
#ifdef ENABLE_KERNEL_PROF
    if (prof_acc != nullptr) {
        if ASCEND_IS_AIV {
            if (GetBlockIdx() == 0) {
                if (indexed) {
                    prof_acc[8]  += (attn_prof_ptr[1] - attn_prof_ptr[0]);
                    prof_acc[9]  += (attn_prof_ptr[2] - attn_prof_ptr[1]);
                    prof_acc[10] += (attn_prof_ptr[3] - attn_prof_ptr[2]);
                    prof_acc[11] += (attn_prof_ptr[4] - attn_prof_ptr[3]);
                    prof_acc[12] += (attn_prof_ptr[5] - attn_prof_ptr[4]);
                } else {
                    prof_acc[13] += (attn_prof_ptr[1] - attn_prof_ptr[0]);
                    prof_acc[14] += (attn_prof_ptr[4] - attn_prof_ptr[3]);
                    prof_acc[15] += (attn_prof_ptr[5] - attn_prof_ptr[4]);
                }
            }
        }
    }
#endif

#ifdef K11050_AIC_PREFETCH
    if ASCEND_IS_AIC {
        const PrefetchSpan r1_dense[2] = {
            {gu_w_ptr, 18874368U},
            {dn_w_ptr, 9437184U},
        };
        PrefetchSpansAic(pipe, r1_dense, 2);
    }
#endif
    ReducePhase(pipe, communication, addresses.attention.output,
                communication.attention_sum, active_rows, blocks, 0, 0);
    int64_t ts3 = GetSystemCycle();

    __gm__ int8_t* gu_in_i8 = (__gm__ int8_t*)addresses.bridge[ffn_bridge::kSharedCodes];
    __gm__ float* gu_in_sc = (__gm__ float*)addresses.bridge[ffn_bridge::kSharedScales];
    whole_layer_norm::device::Forward(pipe,
        (__gm__ bfloat16_t*)communication.attention_sum,
        (__gm__ bfloat16_t*)norm.retained,
        (__gm__ bfloat16_t*)addresses.post_gamma, nullptr,
        nullptr,
        (__gm__ bfloat16_t*)addresses.bridge[ffn_bridge::kRetained],
        nullptr, gu_in_i8, gu_in_sc,
        active_rows, blocks, config.ffn_epsilon);
    int64_t ts4 = GetSystemCycle();

    const int64_t dense_cap = (active_rows <= 16) ? 16 : 32;
    __gm__ uint8_t* gp_base = (__gm__ uint8_t*)addresses.body.scratch[routed_int4::kGateProduct];
    __gm__ int32_t* gu_accum = reinterpret_cast<__gm__ int32_t*>(gp_base);
    __gm__ int8_t* mid_i8 = reinterpret_cast<__gm__ int8_t*>(gp_base + 524288ULL);
    __gm__ float* mid_scale = reinterpret_cast<__gm__ float*>(gp_base + 589824ULL);
    __gm__ int32_t* dn_accum = (__gm__ int32_t*)addresses.body.scratch[routed_int4::kDownProduct];
    __gm__ bfloat16_t* dense_partial = (__gm__ bfloat16_t*)addresses.body.scratch[routed_int4::kRouteOutput];

    if ASCEND_IS_AIC {
        DirectDenseGateUp24Aic(pipe, gu_in_i8, (__gm__ int8_t*)gu_w_ptr, gu_accum, active_rows);
#ifndef K112722_NO_DDR_EXTRA
        DataSyncBarrier<MemDsbT::DDR>();
#endif
    }
    SyncAll<false>();

    if ASCEND_IS_AIV {
        ActivateDenseSwigluQuant(pipe, gu_accum, (__gm__ float*)gu_s_ptr, gu_in_sc,
                                 mid_i8, mid_scale, active_rows, dense_cap);
#ifndef K112722_NO_DDR_EXTRA
        DataSyncBarrier<MemDsbT::DDR>();
#endif
    }
    SyncAll<false>();

    uint32_t dense_epoch = 0U;
    if ASCEND_IS_AIC {
        if (communication.window != 0ULL) {
            dense_epoch = whole_layer_tree::ReadCommunicationEpoch((__gm__ uint8_t*)communication.window) + 1U;
        }
        DirectDenseDown24Aic(pipe, mid_i8, (__gm__ int8_t*)dn_w_ptr, dn_accum, dense_cap, active_rows);
#ifndef K112722_NO_DDR_EXTRA
        DataSyncBarrier<MemDsbT::DDR>();
#endif
    }
#ifdef K11220_PF_RED2_FULL
    __gm__ volatile int32_t* dense_red2_done_gm =
        reinterpret_cast<__gm__ volatile int32_t*>(addresses.body.scratch[routed_int4::kGroupCount]) + 128;
#endif
    if ASCEND_IS_AIV {
        dense_epoch = whole_layer_tree::ReadCommunicationEpoch((__gm__ uint8_t*)communication.window) + 1U;
#ifdef K11220_PF_RED2_FULL
        if (GetBlockIdx() == 0) {
            *dense_red2_done_gm = 0;
            dcci_cacheline(
                reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ int32_t*>(dense_red2_done_gm)));
        }
#endif
    }
    SyncAll<false>();

    if ASCEND_IS_AIV {
        const bool dense_quad4 = whole_layer_tree::IsQuad4Active(communication.capacity, active_rows);
        const auto comm_layout = dense_quad4
            ? whole_layer_tree::MakeQuad4Layout(communication.capacity, communication.ranks, active_rows)
            : whole_layer_tree::MakeCommunicationLayout(communication.capacity, communication.ranks);
        const uint32_t dense_pmask = dense_quad4 ? 3U : 1U;
        if (GetBlockIdx() == 0) {
            GlobalTensor<uint32_t> counter;
            counter.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(communication.window), 1);
            counter.SetValue(0, dense_epoch);
            DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(counter);
        }
        __gm__ bfloat16_t* dense_dst_base = reinterpret_cast<__gm__ bfloat16_t*>(
            (__gm__ uint8_t*)communication.window + comm_layout.payload
            + (dense_epoch & dense_pmask) * comm_layout.plane_bytes
            + communication.rank * comm_layout.rank_bytes);
        const int64_t deq_cores = (active_rows <= 1) ? 24 : 48;
        if (GetBlockIdx() < deq_cores) {
            mla_tail::device::OutputWorkspace deq_held;
            TBuf<TPosition::VECCALC> comb_buf;
            deq_held.Init(pipe);
            pipe.InitBuffer(comb_buf, 16 * 256 * sizeof(float));
            LocalTensor<float> comb = comb_buf.Get<float>();
            mla_tail::device::FinishOutputRows<mla_tail::device::ScaleOrder::CombinedScales, 16, true>(
                deq_held, comb, dn_accum, (__gm__ float*)dn_s_ptr, mid_scale, nullptr, 0,
                dense_partial, active_rows, dense_cap, GetBlockIdx(), 2 * blocks,
                dense_dst_base, communication.rank, communication.ranks,
                dense_quad4 && (dense_dst_base != nullptr));
        }
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
        SyncAll<true>();
        const int32_t sig_off = static_cast<int32_t>(GetBlockIdx());
        if (sig_off > 0 && sig_off < communication.ranks) {
            const int32_t peer = (communication.rank + sig_off) & 7;
            whole_layer_tree::SignalPeerReady(
                (__gm__ uint8_t*)communication.window, comm_layout,
                communication.ranks, dense_epoch, communication.rank, peer);
        }
        pipe.Reset();
    }
    int64_t ts5 = GetSystemCycle();

#ifdef K11050_AIC_PREFETCH
    if ASCEND_IS_AIC {
        if (next_step != nullptr) {
            const auto& na = next_step->addresses.attention;
            const bool n_idx = ((next_step->indexed & 1ULL) != 0ULL);
            const PrefetchSpan r2_spans[8] = {
                {na.main_weights[0], 12582912U},
                {na.main_weights[1], 4194304U},
                {na.main_weights[2], 1572864U},
                {na.main_weights[3], 3538944U},
                {na.tail_weights[0], 2097152U},
                {na.tail_weights[1], 12582912U},
                {n_idx ? na.front_weights[0] : 0ULL, 8388608U},
                {n_idx ? na.front_weights[2] : 0ULL, 1966080U},
            };
#ifdef K11220_PF_RED2_FULL
            PrefetchSpansAic(pipe, r2_spans, 8, &communication, dense_epoch, false, dense_red2_done_gm);
#else
            PrefetchSpansAic(pipe, r2_spans, 8, &communication, dense_epoch);
#endif
        }
    }
#endif
    ReducePhase(pipe, communication, (uint64_t)dense_partial,
                communication.output, active_rows, blocks, 0, 0);
#ifdef K11220_PF_RED2_FULL
    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0) {
            *dense_red2_done_gm = 1;
            dcci_cacheline(
                reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ int32_t*>(dense_red2_done_gm)));
        }
    }
#endif
    if (prof_acc != nullptr) {
        const int64_t cyc6 = GetSystemCycle();
        prof_acc[0] += (ts1 - ts0);
        prof_acc[indexed ? 1 : 2] += (ts2 - ts1);
        prof_acc[3] += (ts3 - ts2);
        prof_acc[4] += (ts4 - ts3);
        prof_acc[5] += (ts5 - ts4);
        prof_acc[6] += (cyc6 - ts5);
        if ASCEND_IS_AIV {
            if (GetBlockIdx() == 0) {
                prof_acc[22] += (cyc6 - ts3);
            }
        }
    }
}
#endif


}

extern "C" int layer_tp_tree_indexed_launch(void* stream, uint64_t sync,
    const layer_tp_tree_chain::Config* config,
    const layer_tp_tree_chain::Addresses* addresses,
    const layer_tp_tree_chain::Communication* communication)
{
#ifdef SKIP_SINGLE_KERNELS
    return -1;
#else
    if (config == nullptr || addresses == nullptr || communication == nullptr ||
        !layer_tp_tree_chain::Valid(*config, *addresses, *communication)) {
        return -1;
    }
    layer_tp_tree_chain::Kernel<true><<<addresses->body.blocks, nullptr, stream>>>(
        sync, *config, *addresses, *communication);
    return 0;
#endif
}

extern "C" int layer_tp_tree_shared_launch(void* stream, uint64_t sync,
    const layer_tp_tree_chain::Config* config,
    const layer_tp_tree_chain::Addresses* addresses,
    const layer_tp_tree_chain::Communication* communication)
{
#ifdef SKIP_SINGLE_KERNELS
    return -1;
#else
    if (config == nullptr || addresses == nullptr || communication == nullptr ||
        addresses->shared_topk == 0 ||
        !layer_tp_tree_chain::Valid(*config, *addresses, *communication)) {
        return -1;
    }
    layer_tp_tree_chain::Kernel<false><<<addresses->body.blocks, nullptr, stream>>>(
        sync, *config, *addresses, *communication);
    return 0;
#endif
}

static uint64_t g_chain_tree_mask_gm = 0;
static uint64_t g_chain_peer_weights_gm = 0;

extern "C" void layer_tp_tree_set_mask_ptr(uint64_t ptr)
{
    g_chain_tree_mask_gm = ptr;
}

extern "C" void layer_tp_tree_set_peer_weights_ptr(uint64_t ptr)
{
    g_chain_peer_weights_gm = ptr;
}

extern "C" int layer_tp_tree_chain_launch(void* stream, int64_t blocks,
    const layer_tp_tree_chain::ChainStep* steps_host,
    const void* steps_device, int32_t count)
{
#ifdef SKIP_CHAIN_KERNEL
    return -1;
#else
    if (steps_host == nullptr || steps_device == nullptr || count <= 0 ||
        reinterpret_cast<uintptr_t>(steps_device) % 64 != 0) {
        return -1;
    }
    for (int32_t i = 0; i < count; ++i) {
        const auto& s = steps_host[i];
        if (s.sync == 0 || s.addresses.body.blocks != blocks ||
            (((s.indexed & 1ULL) == 0ULL) && s.addresses.shared_topk == 0) ||
            !layer_tp_tree_chain::Valid(s.config, s.addresses, s.communication)) {
            return -1;
        }
    }
    layer_tp_tree_chain::ChainKernel<<<blocks, nullptr, stream>>>(
        reinterpret_cast<uint64_t>(steps_device), count, g_chain_tree_mask_gm, g_chain_peer_weights_gm);
    return 0;
#endif
}
