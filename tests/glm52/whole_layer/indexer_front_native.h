#ifndef TESTS_GLM52_INDEXER_FRONT_NATIVE_H
#define TESTS_GLM52_INDEXER_FRONT_NATIVE_H

#include "../mla_tail/mla_tail_native.h"
#include "indexer_front_contract.h"
#ifndef INDEXER_FRONT_PARALLEL_PROJECTIONS
#define INDEXER_FRONT_PARALLEL_PROJECTIONS
#endif
#ifdef INDEXER_FRONT_PARALLEL_PROJECTIONS
#include "indexer_front_schedule.h"
#endif

namespace indexer_front::device {
using namespace AscendC;
using mla_tail::device::Fence;
using mla_tail::device::LoadTiling;

struct DeviceArgs {
    using Address = GM_ADDR;
    Address hidden, query_norm, query_norm_scale, query_weight;
    Address query_weight_scale, key_weight, gamma, beta, cos, sin, hadamard;
    Address scratch, tiling, query, key, weights, query_scale, key_scale;
    Address slots = nullptr, key_cache = nullptr, scale_cache = nullptr;
    int64_t cache_tokens = 0;
    int64_t cache_rows = 0, cache_active_rows = 0;
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
    int64_t columns, int64_t inner, int64_t tiles,
    int64_t first_tile = GetBlockIdx(), int64_t out_stride = 0)
{
#ifdef INDEXER_FRONT_PARALLEL_PROJECTIONS
    if (first_tile < 0 || first_tile >= tiles) {
        return;
    }
#endif
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
    for (int64_t tile = first_tile; tile < tiles; tile += c.blocks) {
        weight.SetGlobalBuffer((__gm__ T*)weights + tile * inner * columns,
                               inner * columns);
        if (out_stride > 0) {
            destination.SetGlobalBuffer((__gm__ C*)output + tile * columns,
                                        c.capacity * out_stride);
            mm.SetOrgShape(c.rows, columns, inner, inner, out_stride);
        } else {
            destination.SetGlobalBuffer(
                (__gm__ C*)output + tile * c.capacity * columns,
                c.capacity * columns);
            mm.SetOrgShape(c.rows, columns, inner);
        }
        mm.SetSingleShape(c.rows, columns, inner);
        mm.SetTensorA(x);
        mm.SetTensorB(weight, false);
        mm.IterateAll(destination);
        mm.End();
    }
}

__aicore__ inline void ProjectKeyKShift(
    TPipe& pipe, const Config& c, GM_ADDR hidden, GM_ADDR weight,
    GM_ADDR result, int32_t col_tile, int32_t first_row_tile, bool dual_row)
{
    constexpr int32_t kTile = 16;
    constexpr int32_t kPackedColumns = 32;
    constexpr int32_t kMmadInner = 128;
    constexpr int32_t kTotalSteps = 48;
    constexpr int32_t kStageElems = kTile * kMmadInner;
    constexpr int32_t kAccElems = kTile * kTile;
    const bool single_row = c.rows == 1;
    const int32_t rows_m = single_row ? 1 : kTile;
    const int32_t steps_per_segment = single_row ? 4 : 2;
    const int32_t segments = kTotalSteps / steps_per_segment;
    const int32_t weight_tile = col_tile / 2;
    const int32_t column_offset = (col_tile & 1) * kTile;
    TBuf<TPosition::A1> a_stage;
    TBuf<TPosition::B1> b_stage;
    TBuf<TPosition::A2> a_operand;
    TBuf<TPosition::B2> b_operand;
    TBuf<TPosition::CO1> c_accumulator;
    pipe.InitBuffer(a_stage, (dual_row ? 4 : 2) * kStageElems * sizeof(bfloat16_t));
    pipe.InitBuffer(b_stage, 2 * kStageElems * sizeof(bfloat16_t));
    pipe.InitBuffer(a_operand, (dual_row ? 4 : 2) * kStageElems * sizeof(bfloat16_t));
    pipe.InitBuffer(b_operand, 2 * kStageElems * sizeof(bfloat16_t));
    pipe.InitBuffer(c_accumulator, (dual_row ? 2 : 1) * kAccElems * sizeof(float));
    auto a_l1 = a_stage.Get<bfloat16_t>();
    auto b_l1 = b_stage.Get<bfloat16_t>();
    auto a_l0 = a_operand.Get<bfloat16_t>();
    auto b_l0 = b_operand.Get<bfloat16_t>();
    auto c_l0 = c_accumulator.Get<float>();
    GlobalTensor<bfloat16_t> a_gm, b_gm;
    GlobalTensor<float> c_gm;
    a_gm.SetGlobalBuffer((__gm__ bfloat16_t*)hidden, c.capacity * kHidden);
    b_gm.SetGlobalBuffer((__gm__ bfloat16_t*)weight, kKeyColumns * kHidden);
    c_gm.SetGlobalBuffer((__gm__ float*)result, c.capacity * kKeyColumns);

    Nd2NzParams copy_a{};
    copy_a.ndNum = 1;
    copy_a.nValue = rows_m;
    copy_a.dValue = kMmadInner;
    copy_a.srcDValue = kHidden;
    copy_a.dstNzC0Stride = (rows_m == 1) ? 1 : kTile;
    copy_a.dstNzNStride = 1;
    Nd2NzParams copy_b{};
    copy_b.ndNum = 1;
    copy_b.nValue = kMmadInner;
    copy_b.dValue = kTile;
    copy_b.srcDValue = kPackedColumns;
    copy_b.dstNzC0Stride = kMmadInner;
    copy_b.dstNzNStride = 1;
    LoadData2DParams load_a{};
    load_a.repeatTimes = (rows_m == 1) ? 1 : (kMmadInner / kTile);
    load_a.srcStride = 1;
    load_a.ifTranspose = false;
    LoadData2DParams load_b{};
    load_b.repeatTimes = kMmadInner / kTile;
    load_b.srcStride = 1;
    load_b.ifTranspose = true;

    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
    for (int32_t s = 0; s < kTotalSteps; ++s) {
        const int32_t buf = s & 1;
        const event_t ev = buf ? EVENT_ID1 : EVENT_ID0;
        const int32_t outer = s / steps_per_segment;
        const int32_t direction = (col_tile & 1) ? segments - 1 - outer : outer;
        const int32_t segment =
            (direction + weight_tile * (single_row ? 2 : 4)) % segments;
        const int32_t k_start =
            (segment * steps_per_segment + s % steps_per_segment) * kMmadInner;
        const int64_t a0_offset =
            static_cast<int64_t>(first_row_tile) * kTile * kHidden + k_start;
        const int64_t b_offset =
            static_cast<int64_t>(weight_tile) * kHidden * kPackedColumns
            + static_cast<int64_t>(k_start) * kPackedColumns + column_offset;
        WaitFlag<HardEvent::MTE1_MTE2>(ev);
        DataCopy(a_l1[buf * kStageElems], a_gm[a0_offset], copy_a);
        if (dual_row) {
            DataCopy(a_l1[(2 + buf) * kStageElems],
                     a_gm[kTile * kHidden + k_start], copy_a);
        }
        DataCopy(b_l1[buf * kStageElems], b_gm[b_offset], copy_b);
        SetFlag<HardEvent::MTE2_MTE1>(ev);
        WaitFlag<HardEvent::MTE2_MTE1>(ev);
        WaitFlag<HardEvent::M_MTE1>(ev);
        LoadData(a_l0[buf * kStageElems], a_l1[buf * kStageElems], load_a);
        if (dual_row) {
            LoadData(a_l0[(2 + buf) * kStageElems],
                     a_l1[(2 + buf) * kStageElems], load_a);
        }
        LoadData(b_l0[buf * kStageElems], b_l1[buf * kStageElems], load_b);
        SetFlag<HardEvent::MTE1_MTE2>(ev);
        SetFlag<HardEvent::MTE1_M>(ev);
        WaitFlag<HardEvent::MTE1_M>(ev);
        MmadParams mad{};
        mad.m = rows_m;
        mad.n = kTile;
        mad.k = kMmadInner;
        mad.cmatrixInitVal = (s == 0);
        mad.cmatrixSource = false;
        Mmad(c_l0, a_l0[buf * kStageElems], b_l0[buf * kStageElems], mad);
        if (dual_row) {
            Mmad(c_l0[kAccElems], a_l0[(2 + buf) * kStageElems],
                 b_l0[buf * kStageElems], mad);
        }
        SetFlag<HardEvent::M_MTE1>(ev);
    }
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    FixpipeParamsV220 store{};
    store.nSize = kTile;
    store.mSize = rows_m;
    store.srcStride = kTile;
    store.dstStride = kTile;
    store.ndNum = 1;
    store.quantPre = QuantMode_t::NoQuant;
    const int64_t base_out =
        static_cast<int64_t>(col_tile) * c.capacity * kTile;
    Fixpipe(c_gm[base_out + static_cast<int64_t>(first_row_tile) * kTile * kTile],
            c_l0, store);
    if (dual_row) {
        Fixpipe(c_gm[base_out + kTile * kTile], c_l0[kAccElems], store);
    }
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
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
        pipe.InitBuffer(narrow_b, 4096 * 2);
        pipe.InitBuffer(half_b, 4096 * 2);
        pipe.InitBuffer(codes_b, 4096);
        pipe.InitBuffer(value_b, 4096 * 4);
        pipe.InitBuffer(work_b, 4096 * 4);
        pipe.InitBuffer(other_b, 4096 * 4);
        pipe.InitBuffer(weight_b, 256 * 4);
        pipe.InitBuffer(beta_b, 128 * 4);
        pipe.InitBuffer(cos_b, 32 * 64 * 2);
        pipe.InitBuffer(sin_b, 32 * 64 * 2);
        pipe.InitBuffer(cos_float_b, 32 * 64 * 4);
        pipe.InitBuffer(sin_float_b, 32 * 64 * 4);
        pipe.InitBuffer(left_b, 32 * 4);
        pipe.InitBuffer(right_b, 32 * 4);
        pipe.InitBuffer(pairs_b, 64 * 4);
        pipe.InitBuffer(temp_b, 128 * 4);
        pipe.InitBuffer(offsets_b, 128 * 4);
        pipe.InitBuffer(scalar_b, 256 * 4);
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
        SetDeqScale(static_cast<half>(1.0F));
        Fence<HardEvent::S_V>();
    }
};

// Input and output in held.narrow preserve the BF16 boundary on both sides.
__aicore__ inline void Rope(VectorWorkspace& held, const DeviceArgs& args,
                            int64_t row, bool loaded_trig = false)
{
    if (!loaded_trig) {
        GlobalTensor<bfloat16_t> cos_gm, sin_gm;
        cos_gm.SetGlobalBuffer((__gm__ bfloat16_t*)args.cos + row * kRope, kRope);
        sin_gm.SetGlobalBuffer((__gm__ bfloat16_t*)args.sin + row * kRope, kRope);
        DataCopy(held.cosine_bf16, cos_gm, kRope);
        DataCopy(held.sine_bf16, sin_gm, kRope);
        Fence<HardEvent::MTE2_V>();
    }
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
    GlobalTensor<bfloat16_t> channel_scale_bf16, rope, output, cos_gm, sin_gm;
    product.SetGlobalBuffer((__gm__ int32_t*)(args.scratch + c.query_acc));
    channel_scale.SetGlobalBuffer((__gm__ float*)args.query_weight_scale);
    channel_scale_bf16.SetGlobalBuffer(
        (__gm__ bfloat16_t*)args.query_weight_scale);
    token_scale.SetGlobalBuffer((__gm__ float*)args.query_norm_scale);
    rope.SetGlobalBuffer((__gm__ bfloat16_t*)(args.scratch + c.query_rope));
    output.SetGlobalBuffer((__gm__ bfloat16_t*)args.query);
    cos_gm.SetGlobalBuffer((__gm__ bfloat16_t*)args.cos, c.rows * kRope);
    sin_gm.SetGlobalBuffer((__gm__ bfloat16_t*)args.sin, c.rows * kRope);
    const int64_t total = c.rows * kDim;
    const DataCopyParams copy_in{
        static_cast<uint16_t>(c.rows),
        static_cast<uint16_t>(kDim * sizeof(int32_t) / 32),
        static_cast<uint16_t>((kQueryTile - kDim) * sizeof(int32_t) / 32),
        0,
    };
    const DataCopyParams copy_out{
        static_cast<uint16_t>(c.rows),
        static_cast<uint16_t>(kDim * sizeof(bfloat16_t) / 32),
        0,
        static_cast<uint16_t>((kHeads - 1) * kDim * sizeof(bfloat16_t) / 32),
    };
    for (int64_t head = GetBlockIdx(); head < kHeads; head += 2 * c.blocks) {
        const int64_t base_offset = (head / 2) * c.capacity * kQueryTile
                                    + (head % 2) * kDim;
        DataCopy(held.work.ReinterpretCast<int32_t>(), product[base_offset],
                 copy_in);
        if (c.query_scale_bf16) {
            DataCopy(held.halves.ReinterpretCast<bfloat16_t>(),
                     channel_scale_bf16[head * kDim], kDim);
        } else {
            DataCopy(held.weight, channel_scale[head * kDim], kDim);
        }
        DataCopy(held.cosine_bf16, cos_gm, c.rows * kRope);
        DataCopy(held.sine_bf16, sin_gm, c.rows * kRope);
        DataCopyPad(held.scalar, token_scale,
                    DataCopyExtParams{1, static_cast<uint32_t>(c.rows * sizeof(float)), 0, 0, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        Fence<HardEvent::MTE2_V>();
        Fence<HardEvent::MTE2_S>();
        if (c.query_scale_bf16) {
            Cast(held.weight, held.halves.ReinterpretCast<bfloat16_t>(),
                 RoundMode::CAST_NONE, kDim);
        }
        Cast(held.value, held.work.ReinterpretCast<int32_t>(),
             RoundMode::CAST_RINT, total);
        Cast(held.cosine, held.cosine_bf16, RoundMode::CAST_NONE,
             c.rows * kRope);
        Cast(held.sine, held.sine_bf16, RoundMode::CAST_NONE, c.rows * kRope);
        PipeBarrier<PIPE_V>();
        for (int64_t row = 0; row < c.rows; ++row) {
            Muls(held.other[row * kDim], held.weight,
                 held.scalar.GetValue(row), kDim);
        }
        PipeBarrier<PIPE_V>();
        Mul(held.value, held.other, held.value, total);
        PipeBarrier<PIPE_V>();
        Cast(held.narrow, held.value, RoundMode::CAST_RINT, total);
        PipeBarrier<PIPE_V>();
        Cast(held.value, held.narrow, RoundMode::CAST_NONE, total);
        PipeBarrier<PIPE_V>();
        for (int64_t row = 0; row < c.rows; ++row) {
            Gather(held.work[row * kRope], held.value[row * kDim],
                   held.offsets, 0, 32);
            Gather(held.work[row * kRope + 32], held.value[row * kDim],
                   held.offsets[32], 0, 32);
        }
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams rope_rep{1, 1, 1, 8, 8, 8};
        Mul(held.other[0], held.work[0], held.cosine[0], 32, c.rows, rope_rep);
        Mul(held.other[2048], held.work[32], held.sine[0], 32, c.rows, rope_rep);
        Mul(held.other[32], held.work[32], held.cosine[0], 32, c.rows, rope_rep);
        Mul(held.other[2048 + 32], held.work[0], held.sine[0], 32, c.rows, rope_rep);
        PipeBarrier<PIPE_V>();
        Sub(held.other[0], held.other[0], held.other[2048], 32, c.rows, rope_rep);
        Add(held.other[32], held.other[32], held.other[2048 + 32], 32, c.rows, rope_rep);
        PipeBarrier<PIPE_V>();
        for (int64_t row = 0; row < c.rows; ++row) {
            Gather(held.value[row * kDim], held.other[row * kRope],
                   held.offsets[64], 0, kRope);
        }
        PipeBarrier<PIPE_V>();
        Cast(held.narrow, held.value, RoundMode::CAST_RINT, total);
        Fence<HardEvent::V_MTE3>();
        DataCopy(rope[head * kDim], held.narrow, copy_out);
        if (!c.quantized) {
            DataCopy(output[head * kDim], held.narrow, copy_out);
        }
        if (head + 2 * c.blocks < kHeads) {
            Fence<HardEvent::MTE3_MTE2>();
        }
    }
}

__aicore__ inline void NormalizeKey(VectorWorkspace& held, float& mean,
                                    float& rstd)
{
    Cast(held.value, held.narrow, RoundMode::CAST_NONE, kDim);
    PipeBarrier<PIPE_V>();
    Add(held.work, held.value, held.value[64], 64);
    PipeBarrier<PIPE_V>();
    ReduceSum(held.scalar, held.work, held.other, 64);
    Fence<HardEvent::V_S>();
    mean = held.scalar.GetValue(0) * (1.0F / kDim);
    Fence<HardEvent::S_V>();
    Duplicate(held.other, mean, kDim);
    PipeBarrier<PIPE_V>();
    Sub(held.value, held.value, held.other, kDim);
    PipeBarrier<PIPE_V>();
    Mul(held.work, held.value, held.value, kDim);
    PipeBarrier<PIPE_V>();
    Add(held.work, held.work, held.work[64], 64);
    PipeBarrier<PIPE_V>();
    ReduceSum(held.scalar, held.work, held.other, 64);
    Fence<HardEvent::V_S>();
    float var = held.scalar.GetValue(0) * (1.0F / kDim);
    Fence<HardEvent::S_V>();
    Duplicate(held.other, var + 1e-6F, 8);
    PipeBarrier<PIPE_V>();
    Sqrt(held.other, held.other, 8);
    Duplicate(held.work, 1.0F, 8);
    PipeBarrier<PIPE_V>();
    Div(held.other, held.work, held.other, 8);
    Fence<HardEvent::V_S>();
    rstd = held.other.GetValue(0);
    Fence<HardEvent::S_V>();
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
    const int64_t vectors = 2 * c.blocks;
    if (vectors > kHeads && GetBlockIdx() < kHeads) {
        return;
    }
    const int64_t key_vectors = vectors > kHeads ? (vectors - kHeads) : vectors;
    const int64_t key_worker = vectors > kHeads
        ? (GetBlockIdx() - kHeads)
        : GetBlockIdx();
    if (key_worker >= c.rows) {
        return;
    }
    GlobalTensor<float> product, gamma, beta;
    GlobalTensor<bfloat16_t> rope, output, weights_bf16, cos_gm, sin_gm;
    GlobalTensor<half> weights_half;
    product.SetGlobalBuffer((__gm__ float*)(args.scratch + c.key_acc));
    gamma.SetGlobalBuffer((__gm__ float*)args.gamma, kDim);
    beta.SetGlobalBuffer((__gm__ float*)args.beta, kDim);
    rope.SetGlobalBuffer((__gm__ bfloat16_t*)(args.scratch + c.key_rope));
    output.SetGlobalBuffer((__gm__ bfloat16_t*)args.key);
    weights_bf16.SetGlobalBuffer((__gm__ bfloat16_t*)args.weights);
    weights_half.SetGlobalBuffer((__gm__ half*)args.weights);
    DataCopy(held.weight, gamma, kDim);
    DataCopy(held.beta, beta, kDim);
    Fence<HardEvent::MTE2_V>();
    const int64_t key_tile = (c.capacity == 16 || c.capacity == 32) ? 16 : kKeyTile;
    const DataCopyParams copy_key{
        static_cast<uint16_t>(kKeyColumns / key_tile),
        static_cast<uint16_t>(key_tile * sizeof(float) / 32),
        static_cast<uint16_t>((c.capacity - 1) * key_tile * sizeof(float) / 32),
        0,
    };
    for (int64_t row = key_worker; row < c.rows; row += key_vectors) {
        DataCopy(held.value, product[row * key_tile], copy_key);
        cos_gm.SetGlobalBuffer((__gm__ bfloat16_t*)args.cos + row * kRope, kRope);
        sin_gm.SetGlobalBuffer((__gm__ bfloat16_t*)args.sin + row * kRope, kRope);
        DataCopy(held.cosine_bf16, cos_gm, kRope);
        DataCopy(held.sine_bf16, sin_gm, kRope);
        Fence<HardEvent::MTE2_V>();
        Cast(held.narrow, held.value, RoundMode::CAST_RINT, kKeyColumns);
        if (c.quantized) {
            PipeBarrier<PIPE_V>();
            Cast(held.other, held.narrow[kDim], RoundMode::CAST_NONE, kHeads);
            PipeBarrier<PIPE_V>();
            Cast(held.halves, held.other, RoundMode::CAST_RINT, kHeads);
            Fence<HardEvent::V_MTE3>();
            DataCopy(weights_half[row * kHeads], held.halves, kHeads);
        } else {
            Fence<HardEvent::V_MTE3>();
            DataCopy(weights_bf16[row * kHeads], held.narrow[kDim], kHeads);
        }
        Fence<HardEvent::MTE3_V>();
        float mean, rstd;
        NormalizeKey(held, mean, rstd);
        PipeBarrier<PIPE_V>();
        Rope(held, args, row, true);
        Fence<HardEvent::V_MTE3>();
        DataCopy(rope[row * kDim], held.narrow, kDim);
        if (!c.quantized) {
            DataCopy(output[row * kDim], held.narrow, kDim);
        }
        if (row + key_vectors < c.rows) {
            Fence<HardEvent::MTE3_MTE2>();
        }
    }
}

__aicore__ inline void Hadamard(TPipe& pipe, const Config& c,
                               const DeviceArgs& args)
{
    using A = mla_tail::device::ValueA;
    using B = mla_tail::device::ValueB;
    using C = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
    const int64_t idx = GetBlockIdx();
    if (c.rows >= c.blocks && c.rows <= 2 * (c.blocks - 1)) {
        if (idx >= c.blocks) {
            return;
        }
        const int64_t query_cores = c.blocks - 1;
        const int64_t extra = c.rows - query_cores;
        const bool key = (idx == query_cores);
        const bool pair = (!key && idx < extra);
        const int64_t task = key ? c.rows : (pair ? 2 * idx : extra + idx);
        const int64_t count = key ? c.rows : (pair ? 2 * kHadamardRows : kHadamardRows);
        TCubeTiling cube;
        LoadTiling(cube, args.tiling + 2 * kTilingStride + (pair ? 256 : 0));
        Matmul<A, B, C, C> mm;
        mm.Init(&cube, &pipe);
        GlobalTensor<bfloat16_t> input, weight;
        GlobalTensor<bfloat16_t> output;
        weight.SetGlobalBuffer((__gm__ bfloat16_t*)args.hadamard, kDim * kDim);
        const uint64_t input_offset = key ? c.key_rope :
            c.query_rope + task * kHadamardRows * kDim * 2;
        const uint64_t output_offset = key ? c.key_hadamard :
            c.query_hadamard + task * kHadamardRows * kDim * 2;
        input.SetGlobalBuffer((__gm__ bfloat16_t*)(args.scratch + input_offset),
                              count * kDim);
        output.SetGlobalBuffer((__gm__ bfloat16_t*)(args.scratch + output_offset),
                               count * kDim);
        mm.SetOrgShape(count, kDim, kDim);
        mm.SetSingleShape(count, kDim, kDim);
        mm.SetTensorA(input);
        mm.SetTensorB(weight, false);
        mm.IterateAll(output);
        mm.End();
        return;
    }
    if (idx > c.rows) {
        return;
    }
    TCubeTiling cube;
    LoadTiling(cube, args.tiling + 2 * kTilingStride);
    Matmul<A, B, C, C> mm;
    mm.Init(&cube, &pipe);
    GlobalTensor<bfloat16_t> input, weight;
    GlobalTensor<bfloat16_t> output;
    weight.SetGlobalBuffer((__gm__ bfloat16_t*)args.hadamard, kDim * kDim);
    for (int64_t task = idx; task <= c.rows; task += c.blocks) {
        const bool key = task == c.rows;
        const int64_t count = key ? c.rows : kHadamardRows;
        const uint64_t input_offset = key ? c.key_rope :
            c.query_rope + task * kHadamardRows * kDim * 2;
        const uint64_t output_offset = key ? c.key_hadamard :
            c.query_hadamard + task * kHadamardRows * kDim * 2;
        input.SetGlobalBuffer((__gm__ bfloat16_t*)(args.scratch + input_offset),
                              count * kDim);
        output.SetGlobalBuffer((__gm__ bfloat16_t*)(args.scratch + output_offset),
                               count * kDim);
        mm.SetOrgShape(count, kDim, kDim);
        mm.SetSingleShape(count, kDim, kDim);
        mm.SetTensorA(input);
        mm.SetTensorB(weight, false);
        mm.IterateAll(output);
        mm.End();
    }
}

template <bool is_key = false>
__aicore__ inline void QuantizeBatch(
    VectorWorkspace& held, GlobalTensor<bfloat16_t>& src,
    GlobalTensor<int8_t>& dst_out, GlobalTensor<half>& dst_scale,
    int64_t begin, int64_t count, bool write_gm = true,
    bool fence_after = false)
{
    if (count <= 0) {
        return;
    }
    const int64_t total = count * kDim;
    auto temp_half = held.other.ReinterpretCast<half>();
    auto integers = held.work.ReinterpretCast<int32_t>();
    for (int64_t r = 0; r < count; ++r) {
        DataCopy(held.narrow, src[(begin + r) * kDim], kDim);
        Fence<HardEvent::MTE2_V>();
        Cast(held.value, held.narrow, RoundMode::CAST_NONE, kDim);
        PipeBarrier<PIPE_V>();
        Abs(held.other, held.value, kDim);
        PipeBarrier<PIPE_V>();
        Max(held.other, held.other[64], held.other, 64, 1,
            BinaryRepeatParams{1, 1, 1, 0, 0, 0});
        PipeBarrier<PIPE_V>();
        Duplicate(held.scalar, 0.0F, 64);
        PipeBarrier<PIPE_V>();
        WholeReduceMax(held.scalar, held.other, 64, 1, 8, 1, 8);
        PipeBarrier<PIPE_V>();
        Duplicate(held.temp, 127.0F, 64);
        PipeBarrier<PIPE_V>();
        Div(held.temp, held.temp, held.scalar, 64);
        Fence<HardEvent::V_S>();
        const float amax = held.scalar.GetValue(0);
        const float scale_factor = held.temp.GetValue(0);
        const half scale_val = static_cast<half>(amax * (1.0F / 127.0F));
        Fence<HardEvent::S_V>();
        Muls(held.value, held.value, scale_factor, kDim);
        PipeBarrier<PIPE_V>();
        Cast(integers, held.value, RoundMode::CAST_RINT, kDim);
        PipeBarrier<PIPE_V>();
        SetDeqScale(static_cast<half>(1.0F));
        PipeBarrier<PIPE_V>();
        Cast(temp_half, integers, RoundMode::CAST_ROUND, kDim);
        PipeBarrier<PIPE_V>();
        Cast(held.codes[r * kDim], temp_half, RoundMode::CAST_TRUNC, kDim);
        held.halves.SetValue(r, scale_val);
    }
    Fence<HardEvent::V_MTE3>();
    Fence<HardEvent::S_MTE3>();
    if (write_gm) {
        DataCopy(dst_out[begin * kDim], held.codes, total);
        DataCopyPad(dst_scale[begin], held.halves,
                    DataCopyExtParams{1, static_cast<uint32_t>(count * sizeof(half)), 0, 0, 0});
        if (fence_after) {
            Fence<HardEvent::MTE3_MTE2>();
        }
    }
}

__aicore__ inline void WriteQuantizedKeyCache(
    VectorWorkspace& held, const DeviceArgs& args, int64_t begin,
    int64_t count, int64_t rows)
{
    if (rows <= 0 || args.key_cache == nullptr) {
        return;
    }
    GlobalTensor<int64_t> slots;
    GlobalTensor<int8_t> cache;
    GlobalTensor<half> cache_scale;
    slots.SetGlobalBuffer((__gm__ int64_t*)args.slots);
    cache.SetGlobalBuffer((__gm__ int8_t*)args.key_cache);
    cache_scale.SetGlobalBuffer((__gm__ half*)args.scale_cache);
    auto slot_values = held.work.ReinterpretCast<int64_t>();
    DataCopyPad(slot_values, slots,
        DataCopyExtParams{1, static_cast<uint32_t>((begin + rows) * sizeof(int64_t)), 0, 0, 0},
        DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
    Fence<HardEvent::MTE2_S>();
    const int64_t slot0 = slot_values.GetValue(begin);
    int64_t contig = 0;
    if (slot0 >= 0 && slot0 + rows <= args.cache_tokens) {
        contig = 1;
        for (int64_t r = 1; r < rows; ++r) {
            if (slot_values.GetValue(begin + r) != slot0 + r) {
                break;
            }
            ++contig;
        }
    }
    Fence<HardEvent::V_S>();
    for (int64_t r = count; r < rows; ++r) {
        held.halves.SetValue(r, static_cast<half>(0));
    }
    Fence<HardEvent::S_MTE3>();
    if (contig == rows) {
        if (count > 0) {
            DataCopy(cache[slot0 * kDim], held.codes, count * kDim);
        }
        DataCopyPad(cache_scale[slot0], held.halves,
            DataCopyExtParams{1, static_cast<uint32_t>(rows * sizeof(half)), 0, 0, 0});
        return;
    }
    auto scale_aligned = held.other.ReinterpretCast<half>();
    Fence<HardEvent::V_S>();
    for (int64_t r = 0; r < rows; ++r) {
        scale_aligned.SetValue(r * 16, held.halves.GetValue(r));
    }
    Fence<HardEvent::S_MTE3>();
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t slot = slot_values.GetValue(begin + r);
        if (slot < 0 || slot >= args.cache_tokens) {
            continue;
        }
        if (r < count) {
            DataCopy(cache[slot * kDim], held.codes[r * kDim], kDim);
        }
        DataCopyPad(cache_scale[slot], scale_aligned[r * 16],
            DataCopyExtParams{1, sizeof(half), 0, 0, 0});
    }
}

__aicore__ inline void Quantize(VectorWorkspace& held, const Config& c,
                                const DeviceArgs& args)
{
    GlobalTensor<bfloat16_t> query, key;
    GlobalTensor<int8_t> query_out, key_out;
    GlobalTensor<half> query_scale, key_scale;
    query.SetGlobalBuffer((__gm__ bfloat16_t*)(args.scratch + c.query_hadamard));
    key.SetGlobalBuffer((__gm__ bfloat16_t*)(args.scratch + c.key_hadamard));
    query_out.SetGlobalBuffer((__gm__ int8_t*)args.query);
    key_out.SetGlobalBuffer((__gm__ int8_t*)args.key);
    query_scale.SetGlobalBuffer((__gm__ half*)args.query_scale);
    key_scale.SetGlobalBuffer((__gm__ half*)args.key_scale);
    const int64_t vectors = 2 * c.blocks;
    for (int64_t group = GetBlockIdx(); group < kHeads; group += vectors) {
        QuantizeBatch<false>(held, query, query_out, query_scale,
                      group * c.rows, c.rows, true,
                      group + vectors < kHeads || vectors <= kHeads);
    }
    const bool write_key_gm = (args.key_cache == nullptr);
    const int64_t cache_rows = args.cache_rows > 0 ? args.cache_rows : c.rows;
    const int64_t active_rows = args.cache_rows > 0 ? args.cache_active_rows : c.rows;
    if (vectors > kHeads) {
        if (GetBlockIdx() >= kHeads) {
            const int64_t key_workers = vectors - kHeads;
            const int64_t kw = GetBlockIdx() - kHeads;
            const int64_t per_worker = (cache_rows + key_workers - 1) / key_workers;
            const int64_t begin = kw * per_worker;
            const int64_t rows = cache_rows <= begin ? 0
                : (cache_rows - begin < per_worker ? cache_rows - begin : per_worker);
            const int64_t count = active_rows <= begin ? 0
                : (active_rows - begin < rows ? active_rows - begin : rows);
            QuantizeBatch<true>(held, key, key_out, key_scale, begin, count, write_key_gm);
            WriteQuantizedKeyCache(held, args, begin, count, rows);
        }
    } else if (GetBlockIdx() == 0) {
        QuantizeBatch<true>(held, key, key_out, key_scale, 0, active_rows, write_key_gm);
        WriteQuantizedKeyCache(held, args, 0, active_rows, cache_rows);
    }
}

struct NoInputStage {
    __aicore__ inline void Run(TPipe&, const Config&) const {}
};

template <typename InputStage = NoInputStage>
__aicore__ inline void Forward(TPipe& pipe, const Config& c,
                               const DeviceArgs& args,
                               InputStage input_stage = {})
{
    using mla_tail::device::Publish;
    SetSysWorkspace(args.scratch + c.system);
    VectorWorkspace held;
    if ASCEND_IS_AIC {
#ifdef INDEXER_FRONT_PARALLEL_PROJECTIONS
        const auto owner = ProjectionOwnerFor(GetBlockIdx(), c.blocks);
        Project<int8_t, int32_t>(pipe, c, args.query_norm, args.query_weight,
            args.scratch + c.query_acc, args.tiling,
            kQueryTile, kQueryRank, kQueryColumns / kQueryTile,
            owner.query_tile);
        const bool kshift = (c.capacity == 16 || c.capacity == 32);
        if (kshift && c.blocks >= 24) {
            const int32_t idx = GetBlockIdx();
            if (idx >= 16 && idx < 24) {
                ProjectKeyKShift(pipe, c, args.hidden, args.key_weight,
                    args.scratch + c.key_acc, idx - 16, 0, c.capacity == 32);
            } else if (idx < (c.capacity == 32 ? 4 : 2)) {
                pipe.Reset();
                ProjectKeyKShift(pipe, c, args.hidden, args.key_weight,
                    args.scratch + c.key_acc, 8 + (idx & 1), idx >> 1, false);
            }
        } else if (kshift) {
            if (owner.query_tile >= 0) {
                pipe.Reset();
            }
            for (int32_t ct = GetBlockIdx(); ct < 10; ct += c.blocks) {
                ProjectKeyKShift(pipe, c, args.hidden, args.key_weight,
                    args.scratch + c.key_acc, ct, 0, c.capacity == 32);
            }
        } else {
            if (owner.query_tile >= 0 && owner.key_tile >= 0) {
                pipe.Reset();
            }
            Project<bfloat16_t, float>(pipe, c, args.hidden, args.key_weight,
                args.scratch + c.key_acc, args.tiling + kTilingStride,
                kKeyTile, kHidden, kKeyColumns / kKeyTile, owner.key_tile);
        }
#else
        Project<int8_t, int32_t>(pipe, c, args.query_norm, args.query_weight,
            args.scratch + c.query_acc, args.tiling,
            kQueryTile, kQueryRank, kQueryColumns / kQueryTile);
        pipe.Reset();
        Project<bfloat16_t, float>(pipe, c, args.hidden, args.key_weight,
            args.scratch + c.key_acc, args.tiling + kTilingStride,
            kKeyTile, kHidden, kKeyColumns / kKeyTile);
#endif
        Publish();
        pipe.Reset();
    }
    if ASCEND_IS_AIV {
        input_stage.Run(pipe, c);
        held.Init(pipe);
    }
    SyncAll<false>();
    if ASCEND_IS_AIV {
        QueryEpilogue(held, c, args);
        KeyEpilogue(held, c, args);
        Publish();
    }
    if (c.quantized) {
        SyncAll<false>();
        if ASCEND_IS_AIC {
            Hadamard(pipe, c, args);
            Publish();
        }
        SyncAll<false>();
        if ASCEND_IS_AIV {
            Quantize(held, c, args);
        }
    }
    EndPhase(pipe);
}

}  // namespace indexer_front::device

#endif
