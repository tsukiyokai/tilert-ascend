#ifndef TESTS_GLM52_INDEXER_FRONT_NATIVE_H
#define TESTS_GLM52_INDEXER_FRONT_NATIVE_H

#include "../mla_tail/mla_tail_native.h"
#include "indexer_front_contract.h"

namespace indexer_front::device {
using namespace AscendC;
using mla_tail::device::Fence;
using mla_tail::device::LoadTiling;

struct DeviceArgs {
    using Address = GM_ADDR;
    Address hidden, query_norm, query_norm_scale, query_weight;
    Address query_weight_scale, key_weight, gamma, beta, cos, sin, hadamard;
    Address scratch, tiling, query, key, weights, query_scale, key_scale;
};

__aicore__ inline void EndPhase(TPipe& pipe)
{
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
    SyncAll<false>();
    pipe.Reset();
}

template <class T, class C>
__aicore__ inline void Project(TPipe& pipe, const Config& c,
    GM_ADDR input, GM_ADDR weights, GM_ADDR output, GM_ADDR tiling,
    int64_t columns, int64_t inner, int64_t tiles)
{
    using A = MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using B = MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using Result = MatmulType<TPosition::GM, CubeFormat::ND, C>;
    TCubeTiling cube;
    LoadTiling(cube, tiling);
    Matmul<A, B, Result, Result> mm;
    mm.Init(&cube, &pipe);
    GlobalTensor<T> x, weight;
    GlobalTensor<C> destination;
    x.SetGlobalBuffer((__gm__ T*)input, c.rows * inner);
    for (int64_t tile = GetBlockIdx(); tile < tiles; tile += c.blocks) {
        weight.SetGlobalBuffer((__gm__ T*)weights + tile * inner * columns,
                               inner * columns);
        destination.SetGlobalBuffer(
            (__gm__ C*)output + tile * c.capacity * columns,
            c.capacity * columns);
        mm.SetOrgShape(c.rows, columns, inner);
        mm.SetSingleShape(c.rows, columns, inner);
        mm.SetTensorA(x);
        mm.SetTensorB(weight, false);
        mm.IterateAll(destination);
        mm.End();
    }
}

struct VectorWorkspace {
    TBuf<TPosition::VECCALC> narrow_b, half_b, codes_b;
    TBuf<TPosition::VECCALC> value_b, work_b, other_b, weight_b, beta_b;
    TBuf<TPosition::VECCALC> cos_b, sin_b, cos_float_b, sin_float_b;
    TBuf<TPosition::VECCALC> left_b, right_b, pairs_b, temp_b, offsets_b, scalar_b;
    LocalTensor<bfloat16_t> narrow, cosine_bf16, sine_bf16;
    LocalTensor<half> halves;
    LocalTensor<int8_t> codes;
    LocalTensor<float> value, work, other, weight, beta, cosine, sine;
    LocalTensor<float> left, right, pairs, temp, scalar;
    LocalTensor<uint32_t> offsets;

    __aicore__ inline void Init(TPipe& pipe)
    {
        pipe.InitBuffer(narrow_b, 256 * 2);
        pipe.InitBuffer(half_b, 128 * 2);
        pipe.InitBuffer(codes_b, 128);
        pipe.InitBuffer(value_b, 256 * 4);
        pipe.InitBuffer(work_b, 256 * 4);
        pipe.InitBuffer(other_b, 256 * 4);
        pipe.InitBuffer(weight_b, 256 * 4);
        pipe.InitBuffer(beta_b, 128 * 4);
        pipe.InitBuffer(cos_b, 64 * 2);
        pipe.InitBuffer(sin_b, 64 * 2);
        pipe.InitBuffer(cos_float_b, 64 * 4);
        pipe.InitBuffer(sin_float_b, 64 * 4);
        pipe.InitBuffer(left_b, 32 * 4);
        pipe.InitBuffer(right_b, 32 * 4);
        pipe.InitBuffer(pairs_b, 64 * 4);
        pipe.InitBuffer(temp_b, 64 * 4);
        pipe.InitBuffer(offsets_b, 128 * 4);
        pipe.InitBuffer(scalar_b, 16 * 4);
        narrow = narrow_b.Get<bfloat16_t>();
        halves = half_b.Get<half>();
        codes = codes_b.Get<int8_t>();
        value = value_b.Get<float>();
        work = work_b.Get<float>();
        other = other_b.Get<float>();
        weight = weight_b.Get<float>();
        beta = beta_b.Get<float>();
        cosine_bf16 = cos_b.Get<bfloat16_t>();
        sine_bf16 = sin_b.Get<bfloat16_t>();
        cosine = cos_float_b.Get<float>();
        sine = sin_float_b.Get<float>();
        left = left_b.Get<float>();
        right = right_b.Get<float>();
        pairs = pairs_b.Get<float>();
        temp = temp_b.Get<float>();
        offsets = offsets_b.Get<uint32_t>();
        scalar = scalar_b.Get<float>();
        for (uint32_t i = 0; i < 32; ++i) {
            offsets.SetValue(i, i * 8);
            offsets.SetValue(32 + i, i * 8 + 4);
        }
        for (uint32_t i = 0; i < 64; ++i) {
            offsets.SetValue(64 + i, ((i % 2) * 32 + i / 2) * 4);
        }
        Fence<HardEvent::S_V>();
    }
};

// Input and output in held.narrow preserve the BF16 boundary on both sides.
__aicore__ inline void Rope(VectorWorkspace& held, const DeviceArgs& args,
                            int64_t row)
{
    GlobalTensor<bfloat16_t> cos_gm, sin_gm;
    cos_gm.SetGlobalBuffer((__gm__ bfloat16_t*)args.cos + row * kRope, kRope);
    sin_gm.SetGlobalBuffer((__gm__ bfloat16_t*)args.sin + row * kRope, kRope);
    DataCopy(held.cosine_bf16, cos_gm, kRope);
    DataCopy(held.sine_bf16, sin_gm, kRope);
    Fence<HardEvent::MTE2_V>();
    Cast(held.value, held.narrow, RoundMode::CAST_NONE, kDim);
    Cast(held.cosine, held.cosine_bf16, RoundMode::CAST_NONE, kRope);
    Cast(held.sine, held.sine_bf16, RoundMode::CAST_NONE, kRope);
    PipeBarrier<PIPE_V>();
    Gather(held.left, held.value, held.offsets, 0, 32);
    Gather(held.right, held.value, held.offsets[32], 0, 32);
    PipeBarrier<PIPE_V>();
    Mul(held.pairs, held.left, held.cosine, 32);
    Mul(held.temp, held.right, held.sine, 32);
    Mul(held.pairs[32], held.right, held.cosine, 32);
    Mul(held.temp[32], held.left, held.sine, 32);
    PipeBarrier<PIPE_V>();
    Sub(held.pairs, held.pairs, held.temp, 32);
    Add(held.pairs[32], held.pairs[32], held.temp[32], 32);
    PipeBarrier<PIPE_V>();
    Gather(held.value, held.pairs, held.offsets[64], 0, kRope);
    PipeBarrier<PIPE_V>();
    Cast(held.narrow, held.value, RoundMode::CAST_RINT, kDim);
}

__aicore__ inline void QueryEpilogue(VectorWorkspace& held, const Config& c,
                                    const DeviceArgs& args)
{
    GlobalTensor<int32_t> product;
    GlobalTensor<float> channel_scale, token_scale;
    GlobalTensor<bfloat16_t> channel_scale_bf16, projection, rope, output;
    product.SetGlobalBuffer((__gm__ int32_t*)(args.scratch + c.query_acc));
    channel_scale.SetGlobalBuffer((__gm__ float*)args.query_weight_scale);
    channel_scale_bf16.SetGlobalBuffer(
        (__gm__ bfloat16_t*)args.query_weight_scale);
    token_scale.SetGlobalBuffer((__gm__ float*)args.query_norm_scale);
    projection.SetGlobalBuffer((__gm__ bfloat16_t*)(args.scratch + c.query_projection));
    rope.SetGlobalBuffer((__gm__ bfloat16_t*)(args.scratch + c.query_rope));
    output.SetGlobalBuffer((__gm__ bfloat16_t*)args.query);
    for (int64_t task = GetBlockIdx(); task < c.rows * kHeads;
         task += 2 * c.blocks) {
        const int64_t row = task / kHeads;
        const int64_t head = task % kHeads;
        const int64_t offset = ((head / 2) * c.capacity + row) * kQueryTile
                               + (head % 2) * kDim;
        DataCopy(held.work.ReinterpretCast<int32_t>(), product[offset], kDim);
        if (c.query_scale_bf16) {
            DataCopy(held.narrow, channel_scale_bf16[head * kDim], kDim);
        } else {
            DataCopy(held.weight, channel_scale[head * kDim], kDim);
        }
        DataCopyPad(held.scalar, token_scale[row],
                    DataCopyExtParams{1, sizeof(float), 0, 0, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        Fence<HardEvent::MTE2_V>();
        Fence<HardEvent::MTE2_S>();
        const float scale = held.scalar.GetValue(0);
        if (c.query_scale_bf16) {
            Cast(held.weight, held.narrow, RoundMode::CAST_NONE, kDim);
        }
        Cast(held.value, held.work.ReinterpretCast<int32_t>(),
             RoundMode::CAST_RINT, kDim);
        PipeBarrier<PIPE_V>();
        // WQ K2048/N4096 uses distinct CANN dequantization orders for the
        // original BF16 and FP32 scale descriptors.
        if (c.query_scale_bf16) {
            Muls(held.weight, held.weight, scale, kDim);
            PipeBarrier<PIPE_V>();
            Mul(held.value, held.weight, held.value, kDim);
        } else {
            Mul(held.value, held.value, held.weight, kDim);
            PipeBarrier<PIPE_V>();
            Muls(held.value, held.value, scale, kDim);
        }
        PipeBarrier<PIPE_V>();
        Cast(held.narrow, held.value, RoundMode::CAST_RINT, kDim);
        Fence<HardEvent::V_MTE3>();
        DataCopy(projection[task * kDim], held.narrow, kDim);
        Fence<HardEvent::MTE3_V>();
        Rope(held, args, row);
        Fence<HardEvent::V_MTE3>();
        DataCopy(rope[task * kDim], held.narrow, kDim);
        if (!c.quantized) {
            DataCopy(output[task * kDim], held.narrow, kDim);
        }
        Fence<HardEvent::MTE3_MTE2>();
    }
}

__aicore__ inline void NormalizeKey(VectorWorkspace& held, float& mean,
                                    float& rstd)
{
    Cast(held.value, held.narrow, RoundMode::CAST_NONE, kDim);
    PipeBarrier<PIPE_V>();
    Muls(held.work, held.value, 1.0F / kDim, kDim);
    PipeBarrier<PIPE_V>();
    ReduceSum(held.scalar, held.work, held.other, kDim);
    Fence<HardEvent::V_S>();
    mean = held.scalar.GetValue(0);
    Adds(held.value, held.value, -mean, kDim);
    PipeBarrier<PIPE_V>();
    Mul(held.work, held.value, held.value, kDim);
    PipeBarrier<PIPE_V>();
    Muls(held.work, held.work, 1.0F / kDim, kDim);
    PipeBarrier<PIPE_V>();
    ReduceSum(held.scalar, held.work, held.other, kDim);
    PipeBarrier<PIPE_V>();
    Adds(held.scalar, held.scalar, 1e-6F, 1);
    PipeBarrier<PIPE_V>();
    Sqrt(held.scalar, held.scalar, 1);
    Duplicate(held.other, 1.0F, 8);
    PipeBarrier<PIPE_V>();
    Div(held.scalar, held.other, held.scalar, 1);
    Fence<HardEvent::V_S>();
    rstd = held.scalar.GetValue(0);
    Muls(held.value, held.value, rstd, kDim);
    PipeBarrier<PIPE_V>();
    Mul(held.value, held.value, held.weight, kDim);
    PipeBarrier<PIPE_V>();
    Add(held.value, held.value, held.beta, kDim);
    PipeBarrier<PIPE_V>();
    Cast(held.narrow, held.value, RoundMode::CAST_RINT, kDim);
}

__aicore__ inline void KeyEpilogue(VectorWorkspace& held, const Config& c,
                                  const DeviceArgs& args)
{
    GlobalTensor<float> product, gamma, beta, stats;
    GlobalTensor<bfloat16_t> projection, normalized, rope, output, weights_bf16;
    GlobalTensor<half> weights_half;
    product.SetGlobalBuffer((__gm__ float*)(args.scratch + c.key_acc));
    gamma.SetGlobalBuffer((__gm__ float*)args.gamma, kDim);
    beta.SetGlobalBuffer((__gm__ float*)args.beta, kDim);
    stats.SetGlobalBuffer((__gm__ float*)(args.scratch + c.norm_stats));
    projection.SetGlobalBuffer((__gm__ bfloat16_t*)(args.scratch + c.key_projection));
    normalized.SetGlobalBuffer((__gm__ bfloat16_t*)(args.scratch + c.key_normalized));
    rope.SetGlobalBuffer((__gm__ bfloat16_t*)(args.scratch + c.key_rope));
    output.SetGlobalBuffer((__gm__ bfloat16_t*)args.key);
    weights_bf16.SetGlobalBuffer((__gm__ bfloat16_t*)args.weights);
    weights_half.SetGlobalBuffer((__gm__ half*)args.weights);
    DataCopy(held.weight, gamma, kDim);
    DataCopy(held.beta, beta, kDim);
    Fence<HardEvent::MTE2_V>();
    for (int64_t row = GetBlockIdx(); row < c.rows; row += 2 * c.blocks) {
        for (int64_t tile = 0; tile < kKeyColumns / kKeyTile; ++tile) {
            DataCopy(held.value[tile * kKeyTile],
                     product[(tile * c.capacity + row) * kKeyTile], kKeyTile);
        }
        Fence<HardEvent::MTE2_V>();
        Cast(held.narrow, held.value, RoundMode::CAST_RINT, kKeyColumns);
        Fence<HardEvent::V_MTE3>();
        DataCopy(projection[row * kKeyColumns], held.narrow, kKeyColumns);
        if (c.quantized) {
            PipeBarrier<PIPE_V>();
            Cast(held.other, held.narrow[kDim], RoundMode::CAST_NONE, kHeads);
            PipeBarrier<PIPE_V>();
            Cast(held.halves, held.other, RoundMode::CAST_RINT, kHeads);
            Fence<HardEvent::V_MTE3>();
            DataCopy(weights_half[row * kHeads], held.halves, kHeads);
        } else {
            DataCopy(weights_bf16[row * kHeads], held.narrow[kDim], kHeads);
        }
        Fence<HardEvent::MTE3_V>();
        float mean, rstd;
        NormalizeKey(held, mean, rstd);
        Fence<HardEvent::V_MTE3>();
        DataCopy(normalized[row * kDim], held.narrow, kDim);
        Fence<HardEvent::MTE3_V>();
        held.scalar.SetValue(0, mean);
        held.scalar.SetValue(1, rstd);
        Fence<HardEvent::S_MTE3>();
        DataCopyPad(stats[row * 2], held.scalar,
                    DataCopyExtParams{1, 2 * sizeof(float), 0, 0, 0});
        Fence<HardEvent::MTE3_V>();
        Rope(held, args, row);
        Fence<HardEvent::V_MTE3>();
        DataCopy(rope[row * kDim], held.narrow, kDim);
        if (!c.quantized) {
            DataCopy(output[row * kDim], held.narrow, kDim);
        }
        Fence<HardEvent::MTE3_MTE2>();
    }
}

__aicore__ inline void Hadamard(TPipe& pipe, const Config& c,
                               const DeviceArgs& args)
{
    using A = mla_tail::device::ValueA;
    using B = mla_tail::device::ValueB;
    using C = mla_tail::device::ValueC;
    TCubeTiling cube;
    LoadTiling(cube, args.tiling + 2 * kTilingStride);
    Matmul<A, B, C, C> mm;
    mm.Init(&cube, &pipe);
    GlobalTensor<bfloat16_t> input, weight;
    GlobalTensor<float> output;
    weight.SetGlobalBuffer((__gm__ bfloat16_t*)args.hadamard, kDim * kDim);
    for (int64_t task = GetBlockIdx(); task <= c.rows; task += c.blocks) {
        const bool key = task == c.rows;
        const int64_t count = key ? c.rows : kHadamardRows;
        const uint64_t input_offset = key ? c.key_rope :
            c.query_rope + task * kHadamardRows * kDim * 2;
        const uint64_t output_offset = key ? c.key_hadamard :
            c.query_hadamard + task * kHadamardRows * kDim * 4;
        input.SetGlobalBuffer((__gm__ bfloat16_t*)(args.scratch + input_offset),
                              count * kDim);
        output.SetGlobalBuffer((__gm__ float*)(args.scratch + output_offset),
                               count * kDim);
        mm.SetOrgShape(count, kDim, kDim);
        mm.SetSingleShape(count, kDim, kDim);
        mm.SetTensorA(input);
        mm.SetTensorB(weight, false);
        mm.IterateAll(output);
        mm.End();
    }
}

__aicore__ inline void Quantize(VectorWorkspace& held, const Config& c,
                                const DeviceArgs& args)
{
    GlobalTensor<float> query, key;
    GlobalTensor<int8_t> query_out, key_out;
    GlobalTensor<half> query_scale, key_scale;
    query.SetGlobalBuffer((__gm__ float*)(args.scratch + c.query_hadamard));
    key.SetGlobalBuffer((__gm__ float*)(args.scratch + c.key_hadamard));
    query_out.SetGlobalBuffer((__gm__ int8_t*)args.query);
    key_out.SetGlobalBuffer((__gm__ int8_t*)args.key);
    query_scale.SetGlobalBuffer((__gm__ half*)args.query_scale);
    key_scale.SetGlobalBuffer((__gm__ half*)args.key_scale);
    for (int64_t task = GetBlockIdx(); task < c.rows * (kHeads + 1);
         task += 2 * c.blocks) {
        const bool is_key = task >= c.rows * kHeads;
        const int64_t row = is_key ? task - c.rows * kHeads : task;
        DataCopy(held.value, is_key ? key[row * kDim] : query[row * kDim], kDim);
        Fence<HardEvent::MTE2_V>();
        Cast(held.narrow, held.value, RoundMode::CAST_RINT, kDim);
        PipeBarrier<PIPE_V>();
        Cast(held.value, held.narrow, RoundMode::CAST_NONE, kDim);
        PipeBarrier<PIPE_V>();
        Abs(held.other, held.value, kDim);
        Duplicate(held.scalar, 0.0F, 8);
        PipeBarrier<PIPE_V>();
        ReduceMax(held.scalar, held.other, held.work, kDim, false);
        Fence<HardEvent::V_S>();
        const float maximum = held.scalar.GetValue(0);
        Duplicate(held.work, maximum == 0.0F ? 1.0F : maximum, kDim);
        Duplicate(held.other, 127.0F, kDim);
        PipeBarrier<PIPE_V>();
        Div(held.other, held.other, held.work, kDim);
        PipeBarrier<PIPE_V>();
        Mul(held.value, held.value, held.other, kDim);
        Muls(held.scalar, held.scalar, 1.0F / 127.0F, 8);
        PipeBarrier<PIPE_V>();
        Mins(held.value, held.value, 127.0F, kDim);
        PipeBarrier<PIPE_V>();
        Maxs(held.value, held.value, -127.0F, kDim);
        PipeBarrier<PIPE_V>();
        Cast(held.work.ReinterpretCast<int32_t>(), held.value,
             RoundMode::CAST_RINT, kDim);
        PipeBarrier<PIPE_V>();
        SetDeqScale(static_cast<half>(1.0F));
        PipeBarrier<PIPE_V>();
        Cast(held.halves, held.work.ReinterpretCast<int32_t>(),
             RoundMode::CAST_ROUND, kDim);
        PipeBarrier<PIPE_V>();
        Cast(held.codes, held.halves, RoundMode::CAST_TRUNC, kDim);
        PipeBarrier<PIPE_V>();
        Cast(held.halves, held.scalar, RoundMode::CAST_RINT, 8);
        Fence<HardEvent::V_MTE3>();
        DataCopy(is_key ? key_out[row * kDim] : query_out[row * kDim],
                  held.codes, kDim);
        DataCopyPad(is_key ? key_scale[row] : query_scale[row], held.halves,
                    DataCopyExtParams{1, sizeof(half), 0, 0, 0});
        Fence<HardEvent::MTE3_MTE2>();
    }
}

__aicore__ inline void Forward(TPipe& pipe, const Config& c,
                               const DeviceArgs& args)
{
    SetSysWorkspace(args.scratch + c.system);
    if ASCEND_IS_AIC {
        Project<int8_t, int32_t>(pipe, c, args.query_norm, args.query_weight,
            args.scratch + c.query_acc, args.tiling,
            kQueryTile, kQueryRank, kQueryColumns / kQueryTile);
        pipe.Reset();
        Project<bfloat16_t, float>(pipe, c, args.hidden, args.key_weight,
            args.scratch + c.key_acc, args.tiling + kTilingStride,
            kKeyTile, kHidden, kKeyColumns / kKeyTile);
    }
    EndPhase(pipe);
    if ASCEND_IS_AIV {
        VectorWorkspace held;
        held.Init(pipe);
        QueryEpilogue(held, c, args);
        KeyEpilogue(held, c, args);
    }
    EndPhase(pipe);
    if (c.quantized) {
        if ASCEND_IS_AIC {
            Hadamard(pipe, c, args);
        }
        EndPhase(pipe);
        if ASCEND_IS_AIV {
            VectorWorkspace held;
            held.Init(pipe);
            Quantize(held, c, args);
        }
        EndPhase(pipe);
    }
}

}  // namespace indexer_front::device

#endif
