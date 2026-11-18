
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
#ifdef K112722_NO_DDR_EXTRA
    PipeBarrier<PIPE_ALL>();
#elif defined(K112160_AIV_DSB_OPT)
    if ASCEND_IS_AIV {
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::DDR>();
    } else {
        PipeBarrier<PIPE_ALL>();
    }
#else
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
#endif
    SyncAll<false>();
    pipe.Reset();
}

template <class T, class C>
__aicore__ inline void Project(TPipe& pipe, const Config& c,
    GM_ADDR input, GM_ADDR weights, GM_ADDR output, GM_ADDR tiling,
    int64_t columns, int64_t inner, int64_t tiles,
    int64_t first_tile = GetBlockIdx(), int64_t out_stride = 0,
    int64_t tile_step = 0)
{
#ifdef INDEXER_FRONT_PARALLEL_PROJECTIONS
    if (first_tile < 0 || first_tile >= tiles) {
        return;
    }
#endif
    using A = MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using B = MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using Result = MatmulType<TPosition::GM, CubeFormat::ND, C>;
    const int64_t proj_rows = c.rows;
    const int64_t step = tile_step > 0 ? tile_step : c.blocks;
    GlobalTensor<T> x, weight;
    GlobalTensor<C> destination;
    x.SetGlobalBuffer((__gm__ T*)input, proj_rows * inner);
    TCubeTiling cube;
    LoadTiling(cube, tiling);
    Matmul<A, B, Result, Result> mm;
    mm.Init(&cube, &pipe);
    for (int64_t tile = first_tile; tile < tiles; tile += step) {
        weight.SetGlobalBuffer((__gm__ T*)weights + tile * inner * columns,
                               inner * columns);
        if (out_stride > 0) {
            destination.SetGlobalBuffer((__gm__ C*)output + tile * columns,
                                        c.capacity * out_stride);
            mm.SetOrgShape(proj_rows, columns, inner, inner, out_stride);
        } else {
            destination.SetGlobalBuffer(
                (__gm__ C*)output + tile * c.capacity * columns,
                c.capacity * columns);
            mm.SetOrgShape(proj_rows, columns, inner);
        }
        mm.SetSingleShape(proj_rows, columns, inner);
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
#ifdef L78_EXACT_INDEXER
    const bool single_row = (c.rows == 1);
    const bool use_k512 = true;
#else
    const bool single_row = c.capacity == 1;
    const bool use_k512 = single_row;
#endif
    const int32_t rows_m = single_row
        ? 1
        : (c.capacity < kTile ? static_cast<int32_t>(c.capacity) : kTile);
    const int32_t steps_per_segment = use_k512 ? 4 : 2;
    const int32_t segments = kTotalSteps / steps_per_segment;
    const int32_t weight_tile = col_tile / 2;
    const int32_t column_offset = (col_tile & 1) * kTile;
    TBuf<TPosition::A1> a_stage;
    TBuf<TPosition::B1> b_stage;
    TBuf<TPosition::A2> a_operand;
    TBuf<TPosition::B2> b_operand;
    TBuf<TPosition::CO1> c_accumulator;
#ifdef K112514_IDX_PIPE_V2
    pipe.InitBuffer(a_stage, (dual_row ? 8 : 4) * kStageElems * sizeof(bfloat16_t));
    pipe.InitBuffer(b_stage, 4 * kStageElems * sizeof(bfloat16_t));
#else
    pipe.InitBuffer(a_stage, (dual_row ? 4 : 2) * kStageElems * sizeof(bfloat16_t));
    pipe.InitBuffer(b_stage, 2 * kStageElems * sizeof(bfloat16_t));
#endif
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

    const int32_t r0_rem = static_cast<int32_t>(c.rows) - first_row_tile * kTile;
    const int32_t r0_valid = single_row
        ? 1
        : ((r0_rem <= 0) ? 1 : ((r0_rem < rows_m) ? r0_rem : rows_m));
    Nd2NzParams copy_a{};
    copy_a.ndNum = 1;
    copy_a.nValue = r0_valid;
    copy_a.dValue = kMmadInner;
    copy_a.srcDValue = kHidden;
    copy_a.dstNzC0Stride = (rows_m == 1) ? 1 : kTile;
    copy_a.dstNzNStride = 1;
    Nd2NzParams copy_a1 = copy_a;
    if (dual_row) {
        const int32_t r1_rem = static_cast<int32_t>(c.rows) - kTile;
        copy_a1.nValue = (r1_rem <= 0) ? 1 : ((r1_rem < rows_m) ? r1_rem : rows_m);
    }
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

    const int64_t row_base_in = static_cast<int64_t>(first_row_tile) * kTile * kHidden;
    const int64_t row_base_out = static_cast<int64_t>(first_row_tile) * kTile * kTile;
#if defined(K112514_IDX_PIPE_V2) && defined(K112140_PIPE_KEY_ONLY)
    auto slot_mte_ev = [](int32_t b) __aicore__ -> event_t {
        if (b == 0) {
            return EVENT_ID0;
        }
        if (b == 1) {
            return EVENT_ID1;
        }
        return (b == 2) ? EVENT_ID2 : static_cast<event_t>(5);
    };
    auto issue_step4 = [&](int32_t st) __aicore__ {
        const int32_t b = st & 3;
        const event_t mte_ev = slot_mte_ev(b);
#ifdef L78_EXACT_INDEXER
        const int32_t k_start = st * kMmadInner;
#else
        const int32_t outer = st / steps_per_segment;
        const int32_t direction = (col_tile & 1) ? segments - 1 - outer : outer;
        const int32_t segment =
            (direction + weight_tile * (use_k512 ? 2 : 4)) % segments;
        const int32_t k_start =
            (segment * steps_per_segment + st % steps_per_segment) * kMmadInner;
#endif
        const int64_t a0_offset = row_base_in + k_start;
        const int64_t b_offset =
            static_cast<int64_t>(weight_tile) * kHidden * kPackedColumns
            + static_cast<int64_t>(k_start) * kPackedColumns + column_offset;
        WaitFlag<HardEvent::MTE1_MTE2>(mte_ev);
        DataCopy(a_l1[b * kStageElems], a_gm[a0_offset], copy_a);
        if (dual_row) {
            DataCopy(a_l1[(4 + b) * kStageElems],
                     a_gm[kTile * kHidden + k_start], copy_a1);
        }
        DataCopy(b_l1[b * kStageElems], b_gm[b_offset], copy_b);
        SetFlag<HardEvent::MTE2_MTE1>(mte_ev);
    };
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(5));
    SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
    issue_step4(0);
    issue_step4(1);
    issue_step4(2);
    for (int32_t s = 0; s < kTotalSteps; ++s) {
        if (s + 3 < kTotalSteps) {
            issue_step4(s + 3);
        }
        const int32_t b4 = s & 3;
        const int32_t buf = s & 1;
        const event_t mte_ev = slot_mte_ev(b4);
        const event_t m_ev = buf ? EVENT_ID4 : EVENT_ID3;
        WaitFlag<HardEvent::MTE2_MTE1>(mte_ev);
        WaitFlag<HardEvent::M_MTE1>(m_ev);
        LoadData(a_l0[buf * kStageElems], a_l1[b4 * kStageElems], load_a);
        if (dual_row) {
            LoadData(a_l0[(2 + buf) * kStageElems],
                     a_l1[(4 + b4) * kStageElems], load_a);
        }
        LoadData(b_l0[buf * kStageElems], b_l1[b4 * kStageElems], load_b);
        SetFlag<HardEvent::MTE1_MTE2>(mte_ev);
        SetFlag<HardEvent::MTE1_M>(m_ev);
        WaitFlag<HardEvent::MTE1_M>(m_ev);
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
        PipeBarrier<PIPE_M>();
        SetFlag<HardEvent::M_MTE1>(m_ev);
    }
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(5));
#elif defined(K112140_PIPE_KEY_ONLY)
    auto issue_step = [&](int32_t st) __aicore__ {
        const int32_t b = st & 1;
        const event_t mte_ev = b ? EVENT_ID1 : EVENT_ID0;
#ifdef L78_EXACT_INDEXER
        const int32_t k_start = st * kMmadInner;
#else
        const int32_t outer = st / steps_per_segment;
        const int32_t direction = (col_tile & 1) ? segments - 1 - outer : outer;
        const int32_t segment =
            (direction + weight_tile * (use_k512 ? 2 : 4)) % segments;
        const int32_t k_start =
            (segment * steps_per_segment + st % steps_per_segment) * kMmadInner;
#endif
        const int64_t a0_offset = row_base_in + k_start;
        const int64_t b_offset =
            static_cast<int64_t>(weight_tile) * kHidden * kPackedColumns
            + static_cast<int64_t>(k_start) * kPackedColumns + column_offset;
        WaitFlag<HardEvent::MTE1_MTE2>(mte_ev);
        DataCopy(a_l1[b * kStageElems], a_gm[a0_offset], copy_a);
        if (dual_row) {
            DataCopy(a_l1[(2 + b) * kStageElems],
                     a_gm[kTile * kHidden + k_start], copy_a1);
        }
        DataCopy(b_l1[b * kStageElems], b_gm[b_offset], copy_b);
        SetFlag<HardEvent::MTE2_MTE1>(mte_ev);
    };
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
    issue_step(0);
    for (int32_t s = 0; s < kTotalSteps; ++s) {
        if (s + 1 < kTotalSteps) {
            issue_step(s + 1);
        }
        const int32_t buf = s & 1;
        const event_t mte_ev = buf ? EVENT_ID1 : EVENT_ID0;
        const event_t m_ev = buf ? EVENT_ID4 : EVENT_ID3;
        WaitFlag<HardEvent::MTE2_MTE1>(mte_ev);
        WaitFlag<HardEvent::M_MTE1>(m_ev);
        LoadData(a_l0[buf * kStageElems], a_l1[buf * kStageElems], load_a);
        if (dual_row) {
            LoadData(a_l0[(2 + buf) * kStageElems],
                     a_l1[(2 + buf) * kStageElems], load_a);
        }
        LoadData(b_l0[buf * kStageElems], b_l1[buf * kStageElems], load_b);
        SetFlag<HardEvent::MTE1_MTE2>(mte_ev);
        SetFlag<HardEvent::MTE1_M>(m_ev);
        WaitFlag<HardEvent::MTE1_M>(m_ev);
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
        PipeBarrier<PIPE_M>();
        SetFlag<HardEvent::M_MTE1>(m_ev);
    }
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
#else
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
    for (int32_t s = 0; s < kTotalSteps; ++s) {
        const int32_t buf = s & 1;
        const event_t ev = buf ? EVENT_ID1 : EVENT_ID0;
#ifdef L78_EXACT_INDEXER
        const int32_t k_start = s * kMmadInner;
#else
        const int32_t outer = s / steps_per_segment;
        const int32_t direction = (col_tile & 1) ? segments - 1 - outer : outer;
        const int32_t segment =
            (direction + weight_tile * (use_k512 ? 2 : 4)) % segments;
        const int32_t k_start =
            (segment * steps_per_segment + s % steps_per_segment) * kMmadInner;
#endif
        const int64_t a0_offset = row_base_in + k_start;
        const int64_t b_offset =
            static_cast<int64_t>(weight_tile) * kHidden * kPackedColumns
            + static_cast<int64_t>(k_start) * kPackedColumns + column_offset;
        WaitFlag<HardEvent::MTE1_MTE2>(ev);
        DataCopy(a_l1[buf * kStageElems], a_gm[a0_offset], copy_a);
        if (dual_row) {
            DataCopy(a_l1[(2 + buf) * kStageElems],
                     a_gm[kTile * kHidden + k_start], copy_a1);
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
        PipeBarrier<PIPE_M>();
        SetFlag<HardEvent::M_MTE1>(ev);
    }
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
#endif
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
    Fixpipe(c_gm[base_out + row_base_out], c_l0, store);
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
#ifdef K112514_IDX_PIPE_V2
        pipe.InitBuffer(offsets_b, (128 + 2048) * 4);
#else
        pipe.InitBuffer(offsets_b, 128 * 4);
#endif
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
#ifdef K112514_IDX_PIPE_V2
        auto off_i32 = offsets.ReinterpretCast<int32_t>();
        Adds(off_i32[128], off_i32[0], 0, 64);
        for (int32_t r = 1; r < 32; ++r) {
            Adds(off_i32[128 + r * 64], off_i32[0], r * 512, 64);
        }
        PipeBarrier<PIPE_V>();
#endif
    }
};


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
#ifdef K112514_IDX_PIPE_V2
        Gather(held.work, held.value, held.offsets[128], 0, c.rows * kRope);
#else
        for (int64_t row = 0; row < c.rows; ++row) {
            Gather(held.work[row * kRope], held.value[row * kDim],
                   held.offsets, 0, 32);
            Gather(held.work[row * kRope + 32], held.value[row * kDim],
                   held.offsets[32], 0, 32);
        }
#endif
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
        if (c.quantized) {
            DataCopy(rope[head * kDim], held.narrow, copy_out);
        } else {
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
#ifdef L78_EXACT_INDEXER
    Cast(held.value, held.narrow, RoundMode::CAST_NONE, kDim);
    PipeBarrier<PIPE_V>();
    Muls(held.work, held.value, 1.0F / kDim, kDim);
    PipeBarrier<PIPE_V>();
    ReduceSum(held.scalar, held.work, held.other, kDim);
    Fence<HardEvent::V_S>();
    mean = held.scalar.GetValue(0);
    Fence<HardEvent::S_V>();
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
    Fence<HardEvent::S_V>();
#else
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
#endif
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
    GlobalTensor<bfloat16_t> rope, output, weights_bf16, cos_gm, sin_gm, key_cache_gm;
    GlobalTensor<half> weights_half;
    GlobalTensor<int64_t> slots_gm;
    auto slot_values = held.codes.ReinterpretCast<int64_t>();
    product.SetGlobalBuffer((__gm__ float*)(args.scratch + c.key_acc));
    gamma.SetGlobalBuffer((__gm__ float*)args.gamma, kDim);
    beta.SetGlobalBuffer((__gm__ float*)args.beta, kDim);
    rope.SetGlobalBuffer((__gm__ bfloat16_t*)(args.scratch + c.key_rope));
    output.SetGlobalBuffer((__gm__ bfloat16_t*)args.key);
    weights_bf16.SetGlobalBuffer((__gm__ bfloat16_t*)args.weights);
    weights_half.SetGlobalBuffer((__gm__ half*)args.weights);
    DataCopy(held.weight, gamma, kDim);
    DataCopy(held.beta, beta, kDim);
    if (!c.quantized && args.key_cache != nullptr) {
        slots_gm.SetGlobalBuffer((__gm__ int64_t*)args.slots);
        key_cache_gm.SetGlobalBuffer((__gm__ bfloat16_t*)args.key_cache);
        DataCopyPad(slot_values, slots_gm,
            DataCopyExtParams{1, static_cast<uint32_t>(c.rows * sizeof(int64_t)), 0, 0, 0},
            DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
        Fence<HardEvent::MTE2_S>();
    }
    Fence<HardEvent::MTE2_V>();
    const int64_t key_tile =
        (c.capacity == 12 || c.capacity == 16 || c.capacity == 32) ? 16 : kKeyTile;
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
        if (c.quantized) {
            DataCopy(rope[row * kDim], held.narrow, kDim);
        } else if (args.key_cache != nullptr) {
            const int64_t slot = slot_values.GetValue(row);
            if (slot >= 0 && slot < args.cache_tokens) {
                DataCopy(key_cache_gm[slot * kDim], held.narrow, kDim);
            }
        } else {
            DataCopy(output[row * kDim], held.narrow, kDim);
        }
        if (row + key_vectors < c.rows) {
            Fence<HardEvent::MTE3_MTE2>();
        }
    }
}

#ifdef K112184_IDX_FRONT_OPT
__aicore__ inline void DirectHadamard24AicM32(
    TPipe& pipe, const Config& c, const DeviceArgs& args)
{
    const uint32_t cid = static_cast<uint32_t>(GetBlockIdx());
    if (cid >= 24U) {
        return;
    }
    bool is_key = false;
    uint32_t q_task = 0U;
    uint16_t m_valid = 0U;
    if (c.rows <= 23) {
        if (cid > static_cast<uint32_t>(c.rows)) {
#ifdef K112514_IDX_PIPE_V2
            CrossCoreSetFlag<0x2, PIPE_FIX>(8);
#endif
            return;
        }
        is_key = (cid == static_cast<uint32_t>(c.rows));
        q_task = cid;
        m_valid = is_key ? static_cast<uint16_t>(c.rows) : 32U;
    } else {
        const uint32_t extra = static_cast<uint32_t>(c.rows - 23);
        is_key = (cid == 23U);
        const bool pair = (!is_key && cid < extra);
        q_task = is_key ? 0U : (pair ? (2U * cid) : (extra + cid));
        m_valid = is_key ? static_cast<uint16_t>(c.rows) : (pair ? 64U : 32U);
    }
    const uint16_t n_mtiles = static_cast<uint16_t>((m_valid + 15U) >> 4U);
    const uint16_t m_mad = static_cast<uint16_t>(n_mtiles * 16U);

    pipe.Reset();
    TBuf<TPosition::B1> b1_buf;
    TBuf<TPosition::B2> l0b_buf;
    TBuf<TPosition::A1> a1_buf;
    TBuf<TPosition::A2> l0a_buf;
    TBuf<TPosition::CO1> l0c_buf;
    pipe.InitBuffer(b1_buf, 128 * 128 * sizeof(bfloat16_t));
    pipe.InitBuffer(l0b_buf, 128 * 128 * sizeof(bfloat16_t));
    pipe.InitBuffer(a1_buf, 64 * 128 * sizeof(bfloat16_t));
    pipe.InitBuffer(l0a_buf, 64 * 128 * sizeof(bfloat16_t));
    pipe.InitBuffer(l0c_buf, 64 * 128 * sizeof(float));

    LocalTensor<bfloat16_t> b1 = b1_buf.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> l0b = l0b_buf.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> a1 = a1_buf.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> l0a = l0a_buf.Get<bfloat16_t>();
    LocalTensor<float> l0c = l0c_buf.Get<float>();

    __gm__ bfloat16_t* src_ptr = is_key
        ? reinterpret_cast<__gm__ bfloat16_t*>(args.scratch + c.key_rope)
        : (reinterpret_cast<__gm__ bfloat16_t*>(args.scratch + c.query_rope)
           + static_cast<uint64_t>(q_task) * 4096ULL);
    __gm__ bfloat16_t* dst_ptr = is_key
        ? reinterpret_cast<__gm__ bfloat16_t*>(args.scratch + c.key_hadamard)
        : (reinterpret_cast<__gm__ bfloat16_t*>(args.scratch + c.query_hadamard)
           + static_cast<uint64_t>(q_task) * 4096ULL);

    GlobalTensor<bfloat16_t> b_gm, a_gm, c_gm;
    b_gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(args.hadamard), 128 * 128);
    a_gm.SetGlobalBuffer(src_ptr, static_cast<uint64_t>(m_valid) * 128ULL);
    c_gm.SetGlobalBuffer(dst_ptr, static_cast<uint64_t>(m_valid) * 128ULL);

    Nd2NzParams copy_b{};
    copy_b.ndNum = 1;
    copy_b.nValue = 128;
    copy_b.dValue = 128;
    copy_b.srcDValue = 128;
    copy_b.dstNzC0Stride = 128;
    copy_b.dstNzNStride = 1;
    DataCopy(b1, b_gm, copy_b);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);

    Nd2NzParams copy_a{};
    copy_a.ndNum = 1;
    copy_a.dValue = 128;
    copy_a.srcDValue = 128;
    copy_a.dstNzC0Stride = 16;
    copy_a.dstNzNStride = 1;
    for (uint16_t mi = 0U; mi < n_mtiles; ++mi) {
        const uint16_t rem = static_cast<uint16_t>(m_valid - mi * 16U);
        copy_a.nValue = (rem > 16U) ? 16U : rem;
        const uint32_t m_off = static_cast<uint32_t>(mi) * 2048U;
        DataCopy(a1[m_off], a_gm[m_off], copy_a);
    }
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
    LoadData2DParams load_b{};
    load_b.repeatTimes = 8;
    load_b.srcStride = 1;
    load_b.dstGap = 7;
    load_b.ifTranspose = true;
    for (uint32_t ni = 0U; ni < 8U; ++ni) {
        LoadData(l0b[ni * 256U], b1[ni * 2048U], load_b);
    }

    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    LoadData2DParams load_a{};
    load_a.repeatTimes = static_cast<uint16_t>(8U * n_mtiles);
    load_a.srcStride = 1;
    load_a.ifTranspose = false;
    LoadData(l0a, a1, load_a);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);

    MmadParams mad{};
    mad.m = m_mad;
    mad.n = 128;
    mad.k = 128;
    mad.cmatrixSource = false;
    mad.cmatrixInitVal = true;
    Mmad(l0c, l0a, l0b, mad);
    PipeBarrier<PIPE_M>();

    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    FixpipeParamsV220 fix{};
    fix.nSize = 128;
    fix.mSize = m_valid;
    fix.srcStride = m_mad;
    fix.dstStride = 128;
    fix.ndNum = 1;
    fix.quantPre = QuantMode_t::F322BF16;
    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(c_gm, l0c, fix);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
#ifdef K112514_IDX_PIPE_V2
    mla_tail::device::Publish();
    CrossCoreSetFlag<0x2, PIPE_FIX>(8);
#endif
    pipe.Reset();
}
#endif

__aicore__ inline void Hadamard(TPipe& pipe, const Config& c,
                               const DeviceArgs& args)
{
#ifdef K112192_DIRECT_IDX_Q14
    DirectHadamard24AicM32(pipe, c, args);
    return;
#elif defined(K112184_IDX_FRONT_OPT)
    if (c.rows >= 1 && c.rows <= 32 && c.blocks == 24) {
        DirectHadamard24AicM32(pipe, c, args);
        return;
    }
#endif
    using A = mla_tail::device::ValueA;
    using B = mla_tail::device::ValueB;
    using C = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
    const int64_t idx = GetBlockIdx();
#ifndef K112192_DIRECT_IDX_Q14
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
#endif
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
    auto i16 = held.work.ReinterpretCast<int16_t>();
    auto temp_half = held.other.ReinterpretCast<half>();
    auto brcb = held.pairs;
    DataCopy(held.narrow, src[begin * kDim], total);
    Fence<HardEvent::MTE2_V>();
    Cast(held.value, held.narrow, RoundMode::CAST_NONE, total);
    PipeBarrier<PIPE_V>();
    Abs(held.other, held.value, total);
    PipeBarrier<PIPE_V>();
    Max(held.other, held.other[64], held.other, 64, count,
        BinaryRepeatParams{1, 1, 1, 16, 16, 16});
    Duplicate(held.scalar, 0.0F, count * 8);
    Duplicate(held.weight, 127.0F, count * 8);
    PipeBarrier<PIPE_V>();
    WholeReduceMax(held.scalar, held.other, 64, count, 4, 1, 16);
    PipeBarrier<PIPE_V>();
    Div(held.weight, held.weight, held.scalar, count * 8);
    Muls(held.scalar, held.scalar, 1.0F / 127.0F, count * 8);
    PipeBarrier<PIPE_V>();
    Cast(held.halves[128], held.scalar, RoundMode::CAST_RINT, count * 8);
    for (int64_t r = 0; r < count; ++r) {
        Brcb(held.other[r * 64], held.weight[r * 8], 1, BrcbRepeatParams{1, 8});
    }
    PipeBarrier<PIPE_V>();
    const BinaryRepeatParams quant_mul_rep{1, 1, 0, 16, 16, 8};
    Mul(held.value[0], held.value[0], held.other[0], 64, count, quant_mul_rep);
    Mul(held.value[64], held.value[64], held.other[0], 64, count, quant_mul_rep);
    PipeBarrier<PIPE_V>();
    Cast(i16, held.value, RoundMode::CAST_RINT, total);
    PipeBarrier<PIPE_V>();
    Cast(temp_half, i16, RoundMode::CAST_ROUND, total);
    PipeBarrier<PIPE_V>();
    Cast(held.codes, temp_half, RoundMode::CAST_TRUNC, total);
    Fence<HardEvent::V_S>();
    for (int64_t r = 0; r < count; ++r) {
        held.halves.SetValue(r, held.halves.GetValue(128 + r * 8));
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
#ifndef K112192_DIRECT_IDX_Q14
    SetSysWorkspace(args.scratch + c.system);
#endif
    VectorWorkspace held;
    if ASCEND_IS_AIC {
#ifdef INDEXER_FRONT_PARALLEL_PROJECTIONS
        const bool kshift = (c.capacity == 12 || c.capacity == 16 || c.capacity == 32);
        if (kshift && c.blocks >= 24) {
            const int32_t idx = GetBlockIdx();
#ifdef K112848_IDX_Q21_K3
            constexpr int32_t kQWorkers = 21;
#elif defined(K11070_IDX16_8)
            constexpr int32_t kQWorkers = 16;
#else
            constexpr int32_t kQWorkers = 14;
#endif
            if (idx < kQWorkers) {
#if defined(K112848_IDX_Q21_K3) && defined(K112192_DIRECT_IDX_Q14) && defined(K11040_F0_NZ)
                const bool dual_m = (c.rows > 16);
                const uint16_t m0 = dual_m ? 16U : static_cast<uint16_t>(c.rows);
                const uint16_t m1 = dual_m ? static_cast<uint16_t>(c.rows - 16) : 0U;
                const uint32_t m_cap = dual_m ? 32U : 16U;
                TBuf<TPosition::B1> b1_buf0, b1_buf1;
                TBuf<TPosition::A1> a1_buf;
                TBuf<TPosition::A2> l0a_buf;
                TBuf<TPosition::B2> l0b_buf0, l0b_buf1;
                TBuf<TPosition::CO1> l0c_buf;
                pipe.InitBuffer(b1_buf0, 512 * 128);
                pipe.InitBuffer(b1_buf1, 512 * 128);
                pipe.InitBuffer(a1_buf, m_cap * 2048U);
                pipe.InitBuffer(l0a_buf, m_cap * 2048U);
                pipe.InitBuffer(l0b_buf0, 128 * 128);
                pipe.InitBuffer(l0b_buf1, 128 * 128);
                pipe.InitBuffer(l0c_buf, m_cap * 128U * sizeof(int32_t));
                LocalTensor<int8_t> a1 = a1_buf.Get<int8_t>();
                LocalTensor<int8_t> b1[2] = {b1_buf0.Get<int8_t>(), b1_buf1.Get<int8_t>()};
                LocalTensor<int8_t> l0a = l0a_buf.Get<int8_t>();
                LocalTensor<int8_t> l0b[2] = {l0b_buf0.Get<int8_t>(), l0b_buf1.Get<int8_t>()};
                LocalTensor<int32_t> l0c = l0c_buf.Get<int32_t>();
                GlobalTensor<int8_t> a_gm, b_gm;
                GlobalTensor<int32_t> c_gm;
                a_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(args.query_norm),
                                     static_cast<uint64_t>(c.capacity) * 2048ULL);
                Nd2NzParams copy_a{};
                copy_a.ndNum = 1;
                copy_a.nValue = m0;
                copy_a.dValue = 2048;
                copy_a.srcDValue = 2048;
                copy_a.dstNzC0Stride = 16;
                copy_a.dstNzNStride = 1;
                DataCopy(a1, a_gm, copy_a);
                if (dual_m) {
                    copy_a.nValue = m1;
                    DataCopy(a1[32768U], a_gm[32768U], copy_a);
                }
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
                LoadData2DParams load_a{};
                load_a.repeatTimes = dual_m ? 128 : 64;
                load_a.srcStride = 1;
                load_a.ifTranspose = false;
                LoadData(l0a, a1, load_a);
                SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
                LoadData2dTransposeParams trans(0, 4, 16, 1, 0, inc);
                for (int32_t qh = idx; qh < 32; qh += kQWorkers) {
                    const int32_t qt = qh >> 1;
                    const int32_t half = qh & 1;
                    b_gm.SetGlobalBuffer(
                        reinterpret_cast<__gm__ int8_t*>(args.query_weight) +
                            static_cast<uint64_t>(qt) * 524288ULL +
                            static_cast<uint64_t>(half) * 262144ULL,
                        262144ULL);
                    c_gm.SetGlobalBuffer(
                        reinterpret_cast<__gm__ int32_t*>(args.scratch + c.query_acc) +
                            static_cast<uint64_t>(qt) * static_cast<uint64_t>(c.capacity) * 256ULL +
                            static_cast<uint64_t>(half) * 128ULL,
                        static_cast<uint64_t>(c.capacity) * 256ULL);
                    DataCopy(b1[0], b_gm[0], DataCopyParams{4, 512, 1536, 0});
                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
                    SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
                    SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
                    for (uint32_t kc = 0U; kc < 4U; ++kc) {
                        const uint32_t bi = kc & 1U;
                        const event_t b_evt = bi ? EVENT_ID1 : EVENT_ID0;
                        if (kc + 1U < 4U) {
                            const uint32_t nbi = (kc + 1U) & 1U;
                            const event_t nb_evt = nbi ? EVENT_ID1 : EVENT_ID0;
                            if (kc + 1U >= 2U) {
                                WaitFlag<HardEvent::MTE1_MTE2>(nb_evt);
                            }
                            DataCopy(
                                b1[nbi],
                                b_gm[(kc + 1U) * 16384U],
                                DataCopyParams{4, 512, 1536, 0});
                            SetFlag<HardEvent::MTE2_MTE1>(nb_evt);
                        }
                        WaitFlag<HardEvent::MTE2_MTE1>(b_evt);
                        for (uint32_t sub = 0U; sub < 4U; ++sub) {
                            const uint32_t step = kc * 4U + sub;
                            const uint32_t lbi = step & 1U;
                            const event_t lb_evt = lbi ? EVENT_ID4 : EVENT_ID3;
                            WaitFlag<HardEvent::M_MTE1>(lb_evt);
                            const uint32_t k_sub_off = sub * 4096U;
                            for (uint32_t i = 0U; i < 4U; ++i) {
                                LoadDataWithTranspose(
                                    l0b[lbi][i * 4096U],
                                    b1[bi][k_sub_off + i * 1024U],
                                    trans);
                            }
                            if (sub == 3U) {
                                SetFlag<HardEvent::MTE1_MTE2>(b_evt);
                            }
                            SetFlag<HardEvent::MTE1_M>(lb_evt);
                            WaitFlag<HardEvent::MTE1_M>(lb_evt);
                            MmadParams mad{};
                            mad.m = 16;
                            mad.n = 128;
                            mad.k = 128;
                            mad.cmatrixSource = false;
                            mad.cmatrixInitVal = (step == 0U);
                            Mmad(l0c, l0a[step * 2048U], l0b[lbi], mad);
                            if (dual_m) {
                                Mmad(l0c[2048U], l0a[32768U + step * 2048U], l0b[lbi], mad);
                            }
                            if (step == 15U) {
                                PipeBarrier<PIPE_M>();
                            }
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
                    fix.nSize = 128;
                    fix.mSize = m0;
                    fix.srcStride = 16;
                    fix.dstStride = 256;
                    fix.ndNum = 1;
                    fix.quantPre = QuantMode_t::NoQuant;
                    Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(c_gm, l0c, fix);
                    if (dual_m) {
                        fix.mSize = m1;
                        Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(c_gm[4096U], l0c[2048U], fix);
                    }
                    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
                    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
                }
#elif defined(K112192_DIRECT_IDX_Q14) && defined(K11040_F0_NZ)
                const bool dual_m = (c.rows > 16);
                const uint16_t m0 = dual_m ? 16U : static_cast<uint16_t>(c.rows);
                const uint16_t m1 = dual_m ? static_cast<uint16_t>(c.rows - 16) : 0U;
                const uint32_t m_cap = dual_m ? 32U : 16U;
                TBuf<TPosition::B1> b1_buf0, b1_buf1;
                TBuf<TPosition::A1> a1_buf;
                TBuf<TPosition::A2> l0a_buf;
                TBuf<TPosition::B2> l0b_buf0, l0b_buf1;
                TBuf<TPosition::CO1> l0c_buf;
                pipe.InitBuffer(b1_buf0, 512 * 256);
                pipe.InitBuffer(b1_buf1, 512 * 256);
                pipe.InitBuffer(a1_buf, m_cap * 2048U);
                pipe.InitBuffer(l0a_buf, m_cap * 2048U);
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
                a_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(args.query_norm),
                                     static_cast<uint64_t>(c.capacity) * 2048ULL);
                Nd2NzParams copy_a{};
                copy_a.ndNum = 1;
                copy_a.nValue = m0;
                copy_a.dValue = 2048;
                copy_a.srcDValue = 2048;
                copy_a.dstNzC0Stride = 16;
                copy_a.dstNzNStride = 1;
                DataCopy(a1, a_gm, copy_a);
                if (dual_m) {
                    copy_a.nValue = m1;
                    DataCopy(a1[32768U], a_gm[32768U], copy_a);
                }
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
                LoadData2DParams load_a{};
                load_a.repeatTimes = dual_m ? 128 : 64;
                load_a.srcStride = 1;
                load_a.ifTranspose = false;
                LoadData(l0a, a1, load_a);
                SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
                LoadData2dTransposeParams trans(0, 8, 16, 1, 0, inc);
                for (int32_t qt = idx; qt < 16; qt += kQWorkers) {
                    b_gm.SetGlobalBuffer(
                        reinterpret_cast<__gm__ int8_t*>(args.query_weight)
                            + static_cast<uint64_t>(qt) * 524288ULL,
                        524288ULL);
                    c_gm.SetGlobalBuffer(
                        reinterpret_cast<__gm__ int32_t*>(args.scratch + c.query_acc) +
                            static_cast<uint64_t>(qt) * static_cast<uint64_t>(c.capacity) * 256ULL,
                        static_cast<uint64_t>(c.capacity) * 256ULL);
                    DataCopy(b1[0], b_gm[0], DataCopyParams{8, 512, 1536, 0});
                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
                    SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
                    SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
                    for (uint32_t kc = 0U; kc < 4U; ++kc) {
                        const uint32_t bi = kc & 1U;
                        const event_t b_evt = bi ? EVENT_ID1 : EVENT_ID0;
                        if (kc + 1U < 4U) {
                            const uint32_t nbi = (kc + 1U) & 1U;
                            const event_t nb_evt = nbi ? EVENT_ID1 : EVENT_ID0;
                            if (kc + 1U >= 2U) {
                                WaitFlag<HardEvent::MTE1_MTE2>(nb_evt);
                            }
                            DataCopy(
                                b1[nbi],
                                b_gm[(kc + 1U) * 16384U],
                                DataCopyParams{8, 512, 1536, 0});
                            SetFlag<HardEvent::MTE2_MTE1>(nb_evt);
                        }
                        WaitFlag<HardEvent::MTE2_MTE1>(b_evt);
                        for (uint32_t sub = 0U; sub < 4U; ++sub) {
                            const uint32_t step = kc * 4U + sub;
                            const uint32_t lbi = step & 1U;
                            const event_t lb_evt = lbi ? EVENT_ID4 : EVENT_ID3;
                            WaitFlag<HardEvent::M_MTE1>(lb_evt);
                            const uint32_t k_sub_off = sub * 4096U;
                            for (uint32_t i = 0U; i < 4U; ++i) {
                                LoadDataWithTranspose(
                                    l0b[lbi][i * 8192U],
                                    b1[bi][k_sub_off + i * 1024U],
                                    trans);
                            }
                            if (sub == 3U) {
                                SetFlag<HardEvent::MTE1_MTE2>(b_evt);
                            }
                            SetFlag<HardEvent::MTE1_M>(lb_evt);
                            WaitFlag<HardEvent::MTE1_M>(lb_evt);
                            MmadParams mad{};
                            mad.m = 16;
                            mad.n = 256;
                            mad.k = 128;
                            mad.cmatrixSource = false;
                            mad.cmatrixInitVal = (step == 0U);
                            Mmad(l0c, l0a[step * 2048U], l0b[lbi], mad);
                            if (dual_m) {
                                Mmad(l0c[4096U], l0a[32768U + step * 2048U], l0b[lbi], mad);
                            }
                            if (step == 15U) {
                                PipeBarrier<PIPE_M>();
                            }
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
                }
#elif defined(K11040_F0_NZ)
                GlobalTensor<int8_t> q_x, q_w;
                GlobalTensor<int32_t> q_dst;
                q_x.SetGlobalBuffer((__gm__ int8_t*)args.query_norm, c.rows * kQueryRank);
                TCubeTiling q_cube{};
                q_cube.M = c.capacity;
                q_cube.singleCoreM = c.rows;
                if (c.rows <= 16) {
                    Matmul<mla_tail::device::OutputA, mla_tail::device::OutputBNZ,
                           mla_tail::device::OutputC, mla_tail::device::OutputBias,
                           mla_tail::device::kQueryTiling16> q_mm;
                    q_mm.Init(&q_cube, &pipe);
                    for (int32_t qt = idx; qt < 16; qt += kQWorkers) {
                        q_w.SetGlobalBuffer(
                            (__gm__ int8_t*)args.query_weight
                                + static_cast<int64_t>(qt) * kQueryRank * kQueryTile,
                            kQueryRank * kQueryTile);
                        q_dst.SetGlobalBuffer(
                            (__gm__ int32_t*)(args.scratch + c.query_acc)
                                + static_cast<int64_t>(qt) * c.capacity * kQueryTile,
                            c.capacity * kQueryTile);
                        q_mm.SetOrgShape(c.rows, kQueryTile, kQueryRank);
                        q_mm.SetSingleShape(c.rows, kQueryTile, kQueryRank);
                        q_mm.SetTensorA(q_x);
                        q_mm.SetTensorB(q_w, false);
                        q_mm.IterateAll(q_dst);
                    }
                    q_mm.End();
                } else {
                    Matmul<mla_tail::device::OutputA, mla_tail::device::OutputBNZ,
                           mla_tail::device::OutputC, mla_tail::device::OutputBias,
                           mla_tail::device::kQueryTiling32> q_mm;
                    q_mm.Init(&q_cube, &pipe);
                    for (int32_t qt = idx; qt < 16; qt += kQWorkers) {
                        q_w.SetGlobalBuffer(
                            (__gm__ int8_t*)args.query_weight
                                + static_cast<int64_t>(qt) * kQueryRank * kQueryTile,
                            kQueryRank * kQueryTile);
                        q_dst.SetGlobalBuffer(
                            (__gm__ int32_t*)(args.scratch + c.query_acc)
                                + static_cast<int64_t>(qt) * c.capacity * kQueryTile,
                            c.capacity * kQueryTile);
                        q_mm.SetOrgShape(c.rows, kQueryTile, kQueryRank);
                        q_mm.SetSingleShape(c.rows, kQueryTile, kQueryRank);
                        q_mm.SetTensorA(q_x);
                        q_mm.SetTensorB(q_w, false);
                        q_mm.IterateAll(q_dst);
                    }
                    q_mm.End();
                }
#else
                Project<int8_t, int32_t>(pipe, c, args.query_norm, args.query_weight,
                    args.scratch + c.query_acc, args.tiling,
                    kQueryTile, kQueryRank, kQueryColumns / kQueryTile,
                    idx, 0, kQWorkers);
#endif
            } else if (idx < 24) {
#ifdef K112848_IDX_Q21_K3
                for (int32_t kt = idx - kQWorkers; kt < 10; kt += 3) {
                    if (kt > idx - kQWorkers) {
                        pipe.Reset();
                    }
                    ProjectKeyKShift(pipe, c, args.hidden, args.key_weight,
                        args.scratch + c.key_acc, kt, 0, c.rows > 16);
                }
#else
                ProjectKeyKShift(pipe, c, args.hidden, args.key_weight,
                    args.scratch + c.key_acc, idx - kQWorkers, 0, c.rows > 16);
#ifdef K11070_IDX16_8
                if (idx < 18) {
                    pipe.Reset();
                    ProjectKeyKShift(pipe, c, args.hidden, args.key_weight,
                        args.scratch + c.key_acc, idx - 8, 0, c.rows > 16);
                }
#endif
#endif
            }
        } else if (kshift) {
            const auto owner = ProjectionOwnerFor(GetBlockIdx(), c.blocks);
            Project<int8_t, int32_t>(pipe, c, args.query_norm, args.query_weight,
                args.scratch + c.query_acc, args.tiling,
                kQueryTile, kQueryRank, kQueryColumns / kQueryTile,
                owner.query_tile);
            if (owner.query_tile >= 0) {
                pipe.Reset();
            }
            for (int32_t ct = GetBlockIdx(); ct < 10; ct += c.blocks) {
                ProjectKeyKShift(pipe, c, args.hidden, args.key_weight,
                    args.scratch + c.key_acc, ct, 0, c.rows > 16);
            }
        } else {
            const auto owner = ProjectionOwnerFor(GetBlockIdx(), c.blocks);
            Project<int8_t, int32_t>(pipe, c, args.query_norm, args.query_weight,
                args.scratch + c.query_acc, args.tiling,
                kQueryTile, kQueryRank, kQueryColumns / kQueryTile,
                owner.query_tile);
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
#if defined(K112514_IDX_PIPE_V2) && defined(K112184_IDX_FRONT_OPT)
        if (c.rows >= 1 && c.rows <= 32 && c.blocks == 24) {
            if ASCEND_IS_AIC {
                DirectHadamard24AicM32(pipe, c, args);
            }
            if ASCEND_IS_AIV {
                const uint32_t vid = static_cast<uint32_t>(GetBlockIdx());
                if (vid < 48U) {
                    CrossCoreWaitFlag(8);
                    const uint32_t cid = vid >> 1U;
                    const uint32_t sub = vid & 1U;
                    if (!(c.rows <= 23 && cid > static_cast<uint32_t>(c.rows))) {
                        bool is_key = false;
                        uint32_t q_task = 0U;
                        uint32_t m_valid = 0U;
                        if (c.rows <= 23) {
                            is_key = (cid == static_cast<uint32_t>(c.rows));
                            q_task = cid;
                            m_valid = is_key ? static_cast<uint32_t>(c.rows) : 32U;
                        } else {
                            const uint32_t extra = static_cast<uint32_t>(c.rows - 23);
                            is_key = (cid == 23U);
                            const bool pair = (!is_key && cid < extra);
                            q_task = is_key ? 0U : (pair ? (2U * cid) : (extra + cid));
                            m_valid = is_key ? static_cast<uint32_t>(c.rows) : (pair ? 64U : 32U);
                        }
                        GlobalTensor<bfloat16_t> query, key;
                        GlobalTensor<int8_t> query_out, key_out;
                        GlobalTensor<half> query_scale, key_scale;
                        query.SetGlobalBuffer(
                            (__gm__ bfloat16_t*)(args.scratch + c.query_hadamard));
                        key.SetGlobalBuffer((__gm__ bfloat16_t*)(args.scratch + c.key_hadamard));
                        query_out.SetGlobalBuffer((__gm__ int8_t*)args.query);
                        key_out.SetGlobalBuffer((__gm__ int8_t*)args.key);
                        query_scale.SetGlobalBuffer((__gm__ half*)args.query_scale);
                        key_scale.SetGlobalBuffer((__gm__ half*)args.key_scale);
                        if (!is_key) {
                            const int64_t half_v = static_cast<int64_t>(m_valid >> 1U);
                            const int64_t beg = static_cast<int64_t>(q_task * 32U)
                                + static_cast<int64_t>(sub) * half_v;
                            QuantizeBatch<false>(
                                held, query, query_out, query_scale, beg, half_v, true, false);
                        } else if (sub == 0U) {
                            const bool write_key_gm = (args.key_cache == nullptr);
                            const int64_t cache_rows =
                                args.cache_rows > 0 ? args.cache_rows : c.rows;
                            const int64_t active_rows =
                                args.cache_rows > 0 ? args.cache_active_rows : c.rows;
                            QuantizeBatch<true>(
                                held, key, key_out, key_scale, 0, active_rows, write_key_gm);
                            WriteQuantizedKeyCache(held, args, 0, active_rows, cache_rows);
                        }
                    }
                }
            }
        } else
#endif
        {
            if ASCEND_IS_AIC {
                Hadamard(pipe, c, args);
                Publish();
            }
            SyncAll<false>();
            if ASCEND_IS_AIV {
                Quantize(held, c, args);
            }
        }
    }
    EndPhase(pipe);
}

}

#endif
