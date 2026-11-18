#ifndef TESTS_GLM52_ROUTED_INT4_PIPELINE_NATIVE_H
#define TESTS_GLM52_ROUTED_INT4_PIPELINE_NATIVE_H

#include "routed_int4_pipeline.h"
#include "../shared_w8/shared_w8_native.h"

namespace routed_int4::pipeline {

static_assert(shared_w8::kDynamicPerformanceQuant,
              "the shared pipeline requires CANN dynamic-performance quantization");

struct Stages {
    Input input;
    int64_t rows;
    int64_t blocks;
    shared_w8::device::Buffers held;

    __aicore__ inline Stages(Input value, int64_t tokens, int64_t groups)
        : input(value), rows(tokens), blocks(groups) {}

    template <typename T>
    __aicore__ inline __gm__ T* Payload(uint64_t offset = 0)
    {
        return reinterpret_cast<__gm__ T*>(input.payload + offset);
    }

    __aicore__ inline void StageInput(AscendC::TPipe& pipe)
    {
        if ASCEND_IS_AIV {


            shared_w8::device::Buffers staging;
            staging.Init(pipe);
            shared_w8::device::StageInput(staging,
                reinterpret_cast<__gm__ int8_t*>(input.input), Payload<int8_t>(),
                rows, input.capacity, 2 * blocks);
            shared_w8::device::Publish();
        }
    }

    __aicore__ inline void InitResident(AscendC::TPipe& pipe)
    {
        if ASCEND_IS_AIV {
            held.Init(pipe);
        }
    }

    __aicore__ inline void Gate(AscendC::TPipe& pipe)
    {
        if ASCEND_IS_AIC {
            shared_w8::device::Project(pipe, Payload<int8_t>(),
                reinterpret_cast<__gm__ int8_t*>(input.gate_weight),
                Payload<int32_t>(input.partial_offset),
                reinterpret_cast<__gm__ uint8_t*>(input.tiling),
                input.capacity, blocks, true, rows);
#ifdef K112612_TRIM_BARRIERS
            AscendC::PipeBarrier<PIPE_ALL>();
#else
            shared_w8::device::Publish();
#endif
        }
    }

    __aicore__ inline void Activate()
    {
        if ASCEND_IS_AIV {
            shared_w8::device::Activate(held,
                Payload<int32_t>(input.partial_offset),
                reinterpret_cast<__gm__ float*>(input.gate_scale),
                reinterpret_cast<__gm__ float*>(input.input_scale),
                Payload<int8_t>(input.middle_offset),
                Payload<float>(input.scale_offset),
                rows, input.capacity, 2 * blocks);
#ifdef K112612_TRIM_BARRIERS
            AscendC::PipeBarrier<PIPE_ALL>();
#else
            shared_w8::device::Publish();
#endif
        }
    }

    __aicore__ inline void Down(AscendC::TPipe& pipe)
    {
        if ASCEND_IS_AIC {
            pipe.Reset();
            shared_w8::device::Project(pipe, Payload<int8_t>(input.middle_offset),
                reinterpret_cast<__gm__ int8_t*>(input.down_weight),
                Payload<int32_t>(input.down_offset),
                reinterpret_cast<__gm__ uint8_t*>(
                    input.tiling + shared_w8::kTilingStride),
                input.capacity, blocks, false, rows);
#ifdef K112612_TRIM_BARRIERS
            AscendC::PipeBarrier<PIPE_ALL>();
#else
            shared_w8::device::Publish();
#endif
        }
    }

    template <typename WorkspaceT>
    __aicore__ inline void Store(WorkspaceT& ws, __gm__ bfloat16_t* shmem_dst_base = nullptr, int32_t comm_rank = 0, int32_t comm_ranks = 0, bool sip_relay = false)
    {
        if ASCEND_IS_AIV {
#if defined(K11220_RS768_PUSH) || defined(K112600_RS768_HIW)
#ifdef K11220_RS768_PUSH
            const bool store_rs768 = true;
#else
            const bool store_rs768 = (rows > 16);
#endif
            if (store_rs768 && rows > 0) {
                using namespace AscendC;
                const uint32_t tid = static_cast<uint32_t>(GetBlockIdx());
                if (tid < 48U) {
                    const uint32_t d = tid & 7U;
                    const uint32_t g = tid >> 3U;
                    const uint32_t u_rows = static_cast<uint32_t>(rows);
                    const uint32_t q_r = (u_rows * 683U) >> 12U;
                    const uint32_t rem_r = u_rows - q_r * 6U;
                    const int64_t r_beg = static_cast<int64_t>(g * q_r + ((g < rem_r) ? g : rem_r));
                    const int64_t r_end = r_beg + static_cast<int64_t>(q_r + ((g < rem_r) ? 1U : 0U));
                    if (r_beg < r_end) {
                        const int64_t cap = input.capacity;
                        const int64_t t0 = static_cast<int64_t>(d) * 3;
                        const int64_t col = static_cast<int64_t>(d) * 768;
                        GlobalTensor<int32_t> source;
                        GlobalTensor<float> weight_scale, token_scale;
                        GlobalTensor<bfloat16_t> destination;
                        source.SetGlobalBuffer(Payload<int32_t>(input.down_offset), cap * 6144);
                        weight_scale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input.down_scale), 6144);
                        token_scale.SetGlobalBuffer(Payload<float>(input.scale_offset), cap * 8);
                        destination.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(input.output), rows * 6144);
                        auto out_bf16 = ws.other.template ReinterpretCast<bfloat16_t>();
                        mla_tail::device::Fence<HardEvent::V_MTE2>();
                        mla_tail::device::Fence<HardEvent::MTE3_V>();
                        DataCopy(ws.scale, weight_scale[col], 768);
                        for (int64_t b_beg = r_beg; b_beg < r_end; b_beg += 4) {
                            const int64_t b_cnt = (r_end - b_beg < 4) ? (r_end - b_beg) : 4;
                            for (int64_t rr = 0; rr < b_cnt; ++rr) {
                                const int64_t r = b_beg + rr;
                                DataCopy(ws.staging, source[(t0 * cap + r) * 256],
                                         DataCopyParams{3, 32, static_cast<uint16_t>((cap - 1) * 32), 0});
                                DataCopy(ws.maximum, token_scale[r * 8], 8);
                                mla_tail::device::Fence<HardEvent::MTE2_V>();
                                mla_tail::device::Fence<HardEvent::MTE2_S>();
                                Cast(ws.value, ws.staging, RoundMode::CAST_RINT, 768);
                                PipeBarrier<PIPE_V>();
                                Muls(ws.work, ws.scale, ws.maximum.GetValue(0), 768);
                                PipeBarrier<PIPE_V>();
                                Mul(ws.value, ws.value, ws.work, 768);
                                PipeBarrier<PIPE_V>();
                                Cast(out_bf16[rr * 768], ws.value, RoundMode::CAST_RINT, 768);
                                if (rr + 1 < b_cnt) {
                                    mla_tail::device::Fence<HardEvent::V_MTE2>();
                                }
                            }
                            mla_tail::device::Fence<HardEvent::V_MTE3>();
                            const int64_t elems = b_cnt * 768;
#ifdef TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H
                            (void)comm_ranks;
                            (void)sip_relay;
                            if (shmem_dst_base != nullptr) {
                                if (static_cast<int32_t>(d) == comm_rank) {
                                    GlobalTensor<bfloat16_t> self_sh;
                                    self_sh.SetGlobalBuffer(shmem_dst_base + b_beg * 768, elems);
                                    DataCopy(self_sh, out_bf16, elems);
                                } else {
                                    GlobalTensor<bfloat16_t> peer_dst;
                                    peer_dst.SetGlobalBuffer(
                                        reinterpret_cast<__gm__ bfloat16_t*>(aclshmem_ptr(shmem_dst_base + b_beg * 768, static_cast<int32_t>(d))),
                                        elems);
                                    DataCopy(peer_dst, out_bf16, elems);
                                }
                            } else {
                                DataCopy(destination[b_beg * 6144 + col], out_bf16,
                                         DataCopyParams{static_cast<uint16_t>(b_cnt), 48, 0, 336});
                            }
#else
                            (void)comm_rank;
                            (void)comm_ranks;
                            (void)sip_relay;
                            DataCopy(destination[b_beg * 6144 + col], out_bf16,
                                     DataCopyParams{static_cast<uint16_t>(b_cnt), 48, 0, 336});
#endif
                            if (b_beg + 4 < r_end) {
                                mla_tail::device::Fence<HardEvent::MTE3_MTE2>();
                                mla_tail::device::Fence<HardEvent::MTE3_V>();
                            }
                        }
                    }
                }
            }
            if (store_rs768) {
                shared_w8::device::Publish();
                return;
            }
#endif
#if defined(K11220_SH_ROW_1D)
            if (rows > 0) {
                using namespace AscendC;
                const uint32_t tid = static_cast<uint32_t>(GetBlockIdx());
                const int64_t vectors = 2 * blocks;
                const int64_t cap = input.capacity;
                GlobalTensor<int32_t> source;
                GlobalTensor<float> weight_scale, token_scale;
                GlobalTensor<bfloat16_t> destination;
                source.SetGlobalBuffer(Payload<int32_t>(input.down_offset), cap * 6144);
                weight_scale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input.down_scale), 6144);
                token_scale.SetGlobalBuffer(Payload<float>(input.scale_offset), cap * 8);
                destination.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(input.output), rows * 6144);
                auto out_bf16 = ws.other.template ReinterpretCast<bfloat16_t>();
                mla_tail::device::Fence<HardEvent::V_MTE2>();
                mla_tail::device::Fence<HardEvent::MTE3_V>();
                auto run_one = [&](int64_t r, int64_t col, int64_t span, bool has_more) __aicore__ {
                    const int64_t t0 = col >> 8;
                    const uint16_t ntiles = static_cast<uint16_t>(span >> 8);
                    if (ntiles == 1U) {
                        DataCopy(ws.staging, source[(t0 * cap + r) * 256], 256);
                    } else {
                        DataCopy(ws.staging, source[(t0 * cap + r) * 256],
                                 DataCopyParams{ntiles, 32, static_cast<uint16_t>((cap - 1) * 32), 0});
                    }
                    DataCopy(ws.scale, weight_scale[col], span);
                    DataCopy(ws.maximum, token_scale[r * 8], 8);
                    mla_tail::device::Fence<HardEvent::MTE2_V>();
                    mla_tail::device::Fence<HardEvent::MTE2_S>();
                    Cast(ws.value, ws.staging, RoundMode::CAST_RINT, span);
                    PipeBarrier<PIPE_V>();
                    Muls(ws.work, ws.scale, ws.maximum.GetValue(0), span);
                    PipeBarrier<PIPE_V>();
                    Mul(ws.value, ws.value, ws.work, span);
                    PipeBarrier<PIPE_V>();
                    Cast(out_bf16, ws.value, RoundMode::CAST_RINT, span);
                    mla_tail::device::Fence<HardEvent::V_MTE3>();
                    const int64_t dst_off = r * 6144 + col;
#ifdef TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H
#ifdef K11220_LOCAL_SH_RED2
                    if (shmem_dst_base != nullptr && rows > 16 && sip_relay) {
                        GlobalTensor<bfloat16_t> self_sh;
                        self_sh.SetGlobalBuffer(shmem_dst_base + dst_off, span);
                        DataCopy(self_sh, out_bf16, span);
                        for (uint32_t p = 0U; p < 3U; ++p) {
                            const int32_t peer = (comm_rank + static_cast<int32_t>((((tid + p) % 3U) + 1U) << 1U)) & 7;
                            GlobalTensor<bfloat16_t> peer_dst;
                            peer_dst.SetGlobalBuffer(
                                reinterpret_cast<__gm__ bfloat16_t*>(aclshmem_ptr(shmem_dst_base + dst_off, peer)), span);
                            DataCopy(peer_dst, out_bf16, span);
                        }
                    } else if (shmem_dst_base != nullptr) {
                        if (sip_relay) {
                            GlobalTensor<bfloat16_t> self_sh;
                            self_sh.SetGlobalBuffer(shmem_dst_base + dst_off, span);
                            DataCopy(self_sh, out_bf16, span);
                        } else {
                            DataCopy(destination[dst_off], out_bf16, span);
                        }
                        for (uint32_t p = 0U; p < 7U; ++p) {
                            const int32_t peer = (comm_rank + static_cast<int32_t>(((tid + p) % 7U) + 1U)) & 7;
                            GlobalTensor<bfloat16_t> peer_dst;
                            peer_dst.SetGlobalBuffer(
                                reinterpret_cast<__gm__ bfloat16_t*>(aclshmem_ptr(shmem_dst_base + dst_off, peer)), span);
                            DataCopy(peer_dst, out_bf16, span);
                        }
                    } else {
                        DataCopy(destination[dst_off], out_bf16, span);
                    }
#else
                    if (shmem_dst_base != nullptr && sip_relay) {
                        GlobalTensor<bfloat16_t> self_sh;
                        self_sh.SetGlobalBuffer(shmem_dst_base + dst_off, span);
                        DataCopy(self_sh, out_bf16, span);
                        for (uint32_t p = 0U; p < 3U; ++p) {
                            const int32_t peer = (comm_rank + static_cast<int32_t>((((tid + p) % 3U) + 1U) << 1U)) & 7;
                            GlobalTensor<bfloat16_t> peer_dst;
                            peer_dst.SetGlobalBuffer(
                                reinterpret_cast<__gm__ bfloat16_t*>(aclshmem_ptr(shmem_dst_base + dst_off, peer)), span);
                            DataCopy(peer_dst, out_bf16, span);
                        }
                    } else {
                        DataCopy(destination[dst_off], out_bf16, span);
                        if (shmem_dst_base != nullptr) {
                            for (uint32_t p = 0U; p < 7U; ++p) {
                                const int32_t peer = (comm_rank + static_cast<int32_t>(((tid + p) % 7U) + 1U)) & 7;
                                GlobalTensor<bfloat16_t> peer_dst;
                                peer_dst.SetGlobalBuffer(
                                    reinterpret_cast<__gm__ bfloat16_t*>(aclshmem_ptr(shmem_dst_base + dst_off, peer)), span);
                                DataCopy(peer_dst, out_bf16, span);
                            }
                        }
                    }
#endif
#else
                    (void)comm_rank;
                    (void)comm_ranks;
                    (void)sip_relay;
                    DataCopy(destination[dst_off], out_bf16, span);
#endif
                    if (has_more) {
                        mla_tail::device::Fence<HardEvent::MTE3_MTE2>();
                        mla_tail::device::Fence<HardEvent::MTE3_V>();
                    }
                };
                if (rows == 24 && vectors == 48) {
                    if (tid < 48U) {
                        const uint32_t r0 = (tid * 171U) >> 9U;
                        run_one(static_cast<int64_t>(r0), static_cast<int64_t>(tid - r0 * 3U) * 2048, 2048, true);
                        const uint32_t r1 = (tid * 683U) >> 12U;
                        run_one(16 + static_cast<int64_t>(r1), static_cast<int64_t>(tid - r1 * 6U) * 1024, 1024, false);
                    }
                } else {
                    int64_t tpr = 3;
                    int64_t span = 2048;
                    if (rows <= 2) {
                        tpr = 24;
                        span = 256;
                    } else if (rows <= 4) {
                        tpr = 12;
                        span = 512;
                    } else if (rows <= 8) {
                        tpr = 6;
                        span = 1024;
                    } else if (rows <= 12) {
                        tpr = 4;
                        span = 1536;
                    }
                    const int64_t total_tasks = rows * tpr;
                    for (int64_t task = static_cast<int64_t>(tid); task < total_tasks; task += vectors) {
                        const uint32_t ut = static_cast<uint32_t>(task);
                        uint32_t qr = 0U;
                        if (tpr == 24) {
                            qr = (ut * 683U) >> 14U;
                        } else if (tpr == 12) {
                            qr = (ut * 683U) >> 13U;
                        } else if (tpr == 6) {
                            qr = (ut * 683U) >> 12U;
                        } else if (tpr == 4) {
                            qr = ut >> 2U;
                        } else {
                            qr = (ut * 171U) >> 9U;
                        }
                        const int64_t col = static_cast<int64_t>(ut - qr * static_cast<uint32_t>(tpr)) * span;
                        run_one(static_cast<int64_t>(qr), col, span, (task + vectors < total_tasks));
                    }
                }
            }
            shared_w8::device::Publish();
#else
            mla_tail::device::OutputWorkspace out_ws;
            out_ws.accumulators = ws.staging;
            out_ws.values = ws.value;
            out_ws.output = ws.other.template ReinterpretCast<bfloat16_t>();
            out_ws.weights = ws.scale;
            out_ws.bias = ws.mask;
            out_ws.scales = ws.maximum;
            mla_tail::device::FinishOutputRows<
                mla_tail::device::ScaleOrder::CombinedScales, 8>(
                out_ws, ws.work, Payload<int32_t>(input.down_offset),
                reinterpret_cast<__gm__ float*>(input.down_scale),
                Payload<float>(input.scale_offset), nullptr, 0,
                reinterpret_cast<__gm__ bfloat16_t*>(input.output),
                rows, input.capacity, AscendC::GetBlockIdx(), 2 * blocks,
                shmem_dst_base, comm_rank, comm_ranks, sip_relay);
            shared_w8::device::Publish();
#endif
        }
    }
};

}

#endif
