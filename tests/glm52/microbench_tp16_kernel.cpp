#include "source/kernel.cpp"
#include <cstring>

namespace bench_tp16 {
using namespace AscendC;
using layer_tp_tree_chain::Config;
using layer_tp_tree_chain::Addresses;
using layer_tp_tree_chain::AttentionAddresses;

__aicore__ inline whole_layer_indexed::DeviceArgs MakeIndexedArgs(
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

__aicore__ inline int32_t ReadScalarI32(uint64_t gm_addr)
{
    if (gm_addr == 0ULL) {
        return -1;
    }
    GlobalTensor<int32_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(gm_addr), 1);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(gm);
    return gm.GetValue(0);
}

__aicore__ inline uint64_t PrepareFusedRtk(
    const Addresses& addresses, int32_t active_rows, int32_t kv_len,
    uint64_t tree_mask_gm, GM_ADDR& out_resolved_topk_buf)
{
    out_resolved_topk_buf = nullptr;
    uint64_t hdr = 0ULL;
    const uint64_t ws_ptr = addresses.body.scratch[routed_int4::kMatmulWorkspace];
    if (ws_ptr == 0ULL || addresses.body.rows < 1 || addresses.body.rows > 32) {
        return 0ULL;
    }
    const uint64_t rtk_addr = ws_ptr + (15ULL << 20);
    out_resolved_topk_buf = reinterpret_cast<GM_ADDR>(rtk_addr);
#if defined(K112162_FUSED_IDX_RTK) && defined(K112120_CLEAN_DCCI)
    const int64_t cap_rows = addresses.body.rows;
    const uint32_t eff_rows = static_cast<uint32_t>(
        (active_rows > 0 && active_rows <= cap_rows) ? active_rows : cap_rows);
    if ASCEND_IS_AIV {
        const uint32_t vid = static_cast<uint32_t>(GetBlockIdx());
        if (vid < 48U) {
            const uint64_t hdr_addr = rtk_addr - 4096ULL + static_cast<uint64_t>(vid) * 64ULL;
            __gm__ uint64_t* hdr_ptr = reinterpret_cast<__gm__ uint64_t*>(hdr_addr);
            hdr_ptr[0] = rtk_addr;
            hdr_ptr[1] = addresses.attention.main_metadata[5];
            hdr_ptr[2] = tree_mask_gm;
            hdr_ptr[3] = static_cast<uint64_t>(static_cast<uint32_t>(kv_len));
            hdr_ptr[4] = static_cast<uint64_t>(eff_rows);
            hdr_ptr[5] = static_cast<uint64_t>(addresses.body.blocks);
            hdr = hdr_addr;
        }
    } else {
        hdr = rtk_addr - 4096ULL;
    }
#endif
    return hdr;
}

__aicore__ inline void FlushProfAcc(const int64_t* prof_acc, uint64_t prof_acc_gm)
{
    if (prof_acc_gm == 0ULL) {
        return;
    }
    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0) {
            GlobalTensor<int64_t> prof_out;
            prof_out.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(prof_acc_gm), 24);
            for (int32_t k = 0; k < 24; ++k) {
                prof_out.SetValue(k, prof_acc[k]);
            }
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                prof_out[0]);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                prof_out[8]);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                prof_out[16]);
        }
    }
}

__global__ __mix__(1, 2) __schedmode__(1) __aicore__
void BenchIndexedAttnKernel(
    uint64_t sync, Config config, Addresses addresses,
    int32_t active_rows, uint64_t tree_mask_gm, uint64_t prof_acc_gm, int32_t include_norm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync);
    TPipe pipe;
    SetSysWorkspace((GM_ADDR)addresses.bridge[ffn_bridge::kSystem]);

    int64_t prof_acc[24];
    for (int32_t i = 0; i < 24; ++i) {
        prof_acc[i] = 0;
    }
    const int32_t kv_len = ReadScalarI32(addresses.attention.main_metadata[4]);
    GM_ADDR resolved_topk_buf = nullptr;
    const uint64_t fused_hdr = PrepareFusedRtk(
        addresses, active_rows, kv_len, tree_mask_gm, resolved_topk_buf);
    if (fused_hdr != 0ULL) {
        if ASCEND_IS_AIV {
            const uint32_t vid = static_cast<uint32_t>(GetBlockIdx());
            if (vid < 48U) {
                reinterpret_cast<__gm__ uint64_t*>(fused_hdr)[5] =
                    static_cast<uint64_t>(config.indexed.attention.table_pages);
            }
        }
    }
    bool resolved_ok = false;

    const int64_t ts0 = GetSystemCycle();
    if (include_norm != 0) {
        const auto& norm = addresses.norm;
        whole_layer_norm::device::Forward(pipe,
            (__gm__ bfloat16_t*)norm.input, (__gm__ bfloat16_t*)norm.residual,
            (__gm__ bfloat16_t*)norm.gamma, nullptr,
#ifdef K11040_F0_NZ
            (__gm__ bfloat16_t*)norm.normalized, (__gm__ bfloat16_t*)norm.retained,
            nullptr, (__gm__ int8_t*)norm.codes,
#else
            (__gm__ bfloat16_t*)norm.normalized, (__gm__ bfloat16_t*)norm.retained,
            (__gm__ float*)norm.rstd, (__gm__ int8_t*)norm.codes,
#endif
            (__gm__ float*)norm.scales, active_rows, config.indexed.attention.blocks,
            config.input_epsilon);
    }
    const int64_t ts1 = GetSystemCycle();

    __gm__ int64_t* attn_prof_ptr = (prof_acc_gm != 0ULL)
        ? ((__gm__ int64_t*)addresses.body.scratch[routed_int4::kGateTiling] + 32)
        : nullptr;
    whole_layer_indexed::Forward(
        pipe, config.indexed, MakeIndexedArgs(addresses.attention), config.bias_kind,
        (GM_ADDR)addresses.shared_topk, nullptr, 0, 0, 16, tree_mask_gm,
        attn_prof_ptr, resolved_topk_buf, &resolved_ok, kv_len, active_rows,
        fused_hdr, 0U);
    const int64_t ts2 = GetSystemCycle();

    if (prof_acc_gm != 0ULL) {
        prof_acc[0] = ts1 - ts0;
        prof_acc[1] = ts2 - ts1;
#ifdef ENABLE_KERNEL_PROF
        if ASCEND_IS_AIV {
            if (GetBlockIdx() == 0 && attn_prof_ptr != nullptr) {
                prof_acc[8] = attn_prof_ptr[1] - attn_prof_ptr[0];
                prof_acc[9] = attn_prof_ptr[2] - attn_prof_ptr[1];
                prof_acc[10] = attn_prof_ptr[3] - attn_prof_ptr[2];
                prof_acc[11] = attn_prof_ptr[4] - attn_prof_ptr[3];
                prof_acc[12] = attn_prof_ptr[5] - attn_prof_ptr[4];
            }
        }
#endif
        FlushProfAcc(prof_acc, prof_acc_gm);
    }
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
}

__global__ __mix__(1, 2) __schedmode__(1) __aicore__
void BenchSharedAttnKernel(
    uint64_t sync, Config config, Addresses addresses,
    int32_t active_rows, uint64_t tree_mask_gm, uint64_t prof_acc_gm,
    int32_t include_norm, int32_t need_resolve)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync);
    TPipe pipe;
    SetSysWorkspace((GM_ADDR)addresses.bridge[ffn_bridge::kSystem]);

    int64_t prof_acc[24];
    for (int32_t i = 0; i < 24; ++i) {
        prof_acc[i] = 0;
    }
    const int32_t kv_len = ReadScalarI32(addresses.attention.main_metadata[4]);
    const uint64_t ws_ptr = addresses.body.scratch[routed_int4::kMatmulWorkspace];
    GM_ADDR resolved_topk_buf = (ws_ptr != 0ULL && addresses.body.rows >= 1 &&
        addresses.body.rows <= 32)
        ? reinterpret_cast<GM_ADDR>(ws_ptr + (15ULL << 20))
        : nullptr;
    bool resolved_ok = (need_resolve == 2);

    const int64_t ts0 = GetSystemCycle();
    if (include_norm != 0) {
        const auto& norm = addresses.norm;
        whole_layer_norm::device::Forward(pipe,
            (__gm__ bfloat16_t*)norm.input, (__gm__ bfloat16_t*)norm.residual,
            (__gm__ bfloat16_t*)norm.gamma, nullptr,
#ifdef K11040_F0_NZ
            nullptr, (__gm__ bfloat16_t*)norm.retained,
            nullptr, (__gm__ int8_t*)norm.codes,
#else
            (__gm__ bfloat16_t*)norm.normalized, (__gm__ bfloat16_t*)norm.retained,
            (__gm__ float*)norm.rstd, (__gm__ int8_t*)norm.codes,
#endif
            (__gm__ float*)norm.scales, active_rows, config.shared.attention.blocks,
            config.input_epsilon);
    }
    const int64_t ts1 = GetSystemCycle();

    __gm__ int64_t* attn_prof_ptr = (prof_acc_gm != 0ULL)
        ? ((__gm__ int64_t*)addresses.body.scratch[routed_int4::kGateTiling] + 32)
        : nullptr;
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
    whole_layer_attention::FullForward(
        pipe, config.shared, shared,
        (GM_ADDR)a.tail_weights[0], (GM_ADDR)a.tail_weights[1],
        (GM_ADDR)a.tail_weights[2], (GM_ADDR)a.bias, config.bias_kind,
        (GM_ADDR)a.scratch, (GM_ADDR)a.tiling, (GM_ADDR)a.output,
        nullptr, 0, 0, 16, tree_mask_gm, attn_prof_ptr,
        resolved_topk_buf, need_resolve == 1, &resolved_ok, kv_len, active_rows, 0U);
    const int64_t ts2 = GetSystemCycle();

    if (prof_acc_gm != 0ULL) {
        prof_acc[0] = ts1 - ts0;
        prof_acc[2] = ts2 - ts1;
#ifdef ENABLE_KERNEL_PROF
        if ASCEND_IS_AIV {
            if (GetBlockIdx() == 0 && attn_prof_ptr != nullptr) {
                prof_acc[13] = attn_prof_ptr[1] - attn_prof_ptr[0];
                prof_acc[14] = attn_prof_ptr[4] - attn_prof_ptr[3];
                prof_acc[15] = attn_prof_ptr[5] - attn_prof_ptr[4];
            }
        }
#endif
        FlushProfAcc(prof_acc, prof_acc_gm);
    }
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
}

__global__ __mix__(1, 2) __schedmode__(1) __aicore__
void BenchRoutedMoeKernel(
    uint64_t sync, Config config, Addresses addresses,
    int32_t active_rows, int32_t expert_offset, int32_t run_bridge,
    uint64_t override_ids_gm, uint64_t override_probs_gm, uint64_t prof_acc_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    SetSyncBaseAddr(sync);
    TPipe pipe;
    ffn_bridge::Addresses bridge{};
    for (int i = 0; i < ffn_bridge::kBuffers; ++i) {
        bridge.values[i] = (GM_ADDR)addresses.bridge[i];
    }
    SetSysWorkspace(bridge.values[ffn_bridge::kSystem]);

    int64_t prof_acc[24];
    for (int32_t i = 0; i < 24; ++i) {
        prof_acc[i] = 0;
    }
    const int64_t blocks = addresses.body.blocks;
    const int64_t eff_shared_cap =
        (active_rows <= 16) ? 16 : addresses.body.shared.capacity;
    const int64_t ts0 = GetSystemCycle();
    if (run_bridge != 0) {
        ffn_bridge::Forward(
            pipe, (GM_ADDR)addresses.attention.output, (GM_ADDR)addresses.norm.retained,
            (GM_ADDR)addresses.post_gamma, (GM_ADDR)addresses.gate,
            (GM_ADDR)addresses.correction_bias, bridge, active_rows, blocks,
            config.ffn_epsilon,
            (GM_ADDR)addresses.body.scratch[routed_int4::kRestoredRows],
            (GM_ADDR)addresses.body.scratch[routed_int4::kRestoredScale],
            addresses.body.shared.payload, eff_shared_cap, 8);
    }
    if (override_ids_gm != 0ULL) {
        if ASCEND_IS_AIV {
            if (GetBlockIdx() == 0) {
                __gm__ int32_t* dst_ids =
                    reinterpret_cast<__gm__ int32_t*>(bridge.values[ffn_bridge::kIds]);
                __gm__ const int32_t* src_ids =
                    reinterpret_cast<__gm__ const int32_t*>(override_ids_gm);
                const int64_t n = static_cast<int64_t>(active_rows) * 8;
                for (int64_t idx = 0; idx < n; ++idx) {
                    dst_ids[idx] = src_ids[idx];
                }
                for (int64_t off = 0; off < n; off += 16) {
                    GlobalTensor<int32_t> id_cl;
                    id_cl.SetGlobalBuffer(dst_ids + off, 16);
                    DataCacheCleanAndInvalid<
                        int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(id_cl);
                }
                if (override_probs_gm != 0ULL) {
                    __gm__ float* dst_p =
                        reinterpret_cast<__gm__ float*>(bridge.values[ffn_bridge::kProbabilities]);
                    __gm__ const float* src_p =
                        reinterpret_cast<__gm__ const float*>(override_probs_gm);
                    for (int64_t idx = 0; idx < n; ++idx) {
                        dst_p[idx] = src_p[idx];
                    }
                    for (int64_t off = 0; off < n; off += 16) {
                        GlobalTensor<float> pr_cl;
                        pr_cl.SetGlobalBuffer(dst_p + off, 16);
                        DataCacheCleanAndInvalid<
                            float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(pr_cl);
                    }
                }
            }
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::DDR>();
            SyncAll<true>();
        }
    }
    const int64_t ts1 = GetSystemCycle();

#if defined(K11080_DIRECT_W4A8) && defined(K11150_SHARED_W8_NZ)
    if (active_rows > 32 || blocks < 24) {
        SetSysWorkspace((GM_ADDR)addresses.body.scratch[routed_int4::kMatmulWorkspace]);
    }
#else
    SetSysWorkspace((GM_ADDR)addresses.body.scratch[routed_int4::kMatmulWorkspace]);
#endif
    auto body_args = addresses.body;
    body_args.rows = active_rows;
    body_args.shared.capacity = eff_shared_cap;
    routed_body::Forward<CubeFormat::NZ, true>(
        pipe, body_args,
        (__gm__ bfloat16_t*)bridge.values[ffn_bridge::kScaledRouted],
        (__gm__ bfloat16_t*)bridge.values[ffn_bridge::kCombined],
        expert_offset, false, false, nullptr, nullptr, nullptr,
        nullptr, 0, 0, 16, 0ULL, (prof_acc_gm != 0ULL) ? (prof_acc + 16) : nullptr);

    if ASCEND_IS_AIC {
        pipe.Reset();
    }
    if ASCEND_IS_AIV {
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
        SyncAll<true>();
        pipe.Reset();
    }
    const int64_t ts2 = GetSystemCycle();

    if (prof_acc_gm != 0ULL) {
        prof_acc[4] = ts1 - ts0;
        prof_acc[5] = ts2 - ts1;
        FlushProfAcc(prof_acc, prof_acc_gm);
    }
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
}

}

extern "C" int bench_tp16_indexed_attn_launch(
    void* stream, uint64_t sync, const void* config_record, size_t config_bytes,
    const void* address_record, size_t address_bytes, int32_t active_rows,
    uint64_t tree_mask_gm, uint64_t prof_acc_gm, int32_t include_norm)
{
    using namespace layer_local_chain;
    if (sync == 0 || config_record == nullptr || address_record == nullptr ||
        config_bytes != sizeof(Config) || address_bytes != sizeof(Addresses)) {
        return -1;
    }
    Config config;
    Addresses addresses;
    std::memcpy(&config, config_record, sizeof(config));
    std::memcpy(&addresses, address_record, sizeof(addresses));
    if (!Valid(config, addresses) || active_rows < 1 ||
        active_rows > static_cast<int32_t>(config.indexed.attention.rows)) {
        return -1;
    }
    bench_tp16::BenchIndexedAttnKernel<<<addresses.body.blocks, nullptr, stream>>>(
        sync, config, addresses, active_rows, tree_mask_gm, prof_acc_gm, include_norm);
    return 0;
}

extern "C" int bench_tp16_shared_attn_launch(
    void* stream, uint64_t sync, const void* config_record, size_t config_bytes,
    const void* address_record, size_t address_bytes, int32_t active_rows,
    uint64_t tree_mask_gm, uint64_t prof_acc_gm, int32_t include_norm, int32_t need_resolve)
{
    using namespace layer_local_chain;
    if (sync == 0 || config_record == nullptr || address_record == nullptr ||
        config_bytes != sizeof(Config) || address_bytes != sizeof(Addresses) ||
        static_cast<const Addresses*>(address_record)->shared_topk == 0) {
        return -1;
    }
    Config config;
    Addresses addresses;
    std::memcpy(&config, config_record, sizeof(config));
    std::memcpy(&addresses, address_record, sizeof(addresses));
    if (!Valid(config, addresses) || active_rows < 1 ||
        active_rows > static_cast<int32_t>(config.shared.attention.rows)) {
        return -1;
    }
    bench_tp16::BenchSharedAttnKernel<<<addresses.body.blocks, nullptr, stream>>>(
        sync, config, addresses, active_rows, tree_mask_gm, prof_acc_gm,
        include_norm, need_resolve);
    return 0;
}

extern "C" int bench_tp16_moe_launch(
    void* stream, uint64_t sync, const void* config_record, size_t config_bytes,
    const void* address_record, size_t address_bytes, int32_t active_rows,
    int32_t expert_offset, int32_t run_bridge, uint64_t override_ids_gm,
    uint64_t override_probs_gm, uint64_t prof_acc_gm)
{
    using namespace layer_local_chain;
    if (sync == 0 || config_record == nullptr || address_record == nullptr ||
        config_bytes != sizeof(Config) || address_bytes != sizeof(Addresses)) {
        return -1;
    }
    Config config;
    Addresses addresses;
    std::memcpy(&config, config_record, sizeof(config));
    std::memcpy(&addresses, address_record, sizeof(addresses));
    if (!Valid(config, addresses) || active_rows < 1 ||
        active_rows > static_cast<int32_t>(addresses.body.rows)) {
        return -1;
    }
    bench_tp16::BenchRoutedMoeKernel<<<addresses.body.blocks, nullptr, stream>>>(
        sync, config, addresses, active_rows, expert_offset, run_bridge,
        override_ids_gm, override_probs_gm, prof_acc_gm);
    return 0;
}
