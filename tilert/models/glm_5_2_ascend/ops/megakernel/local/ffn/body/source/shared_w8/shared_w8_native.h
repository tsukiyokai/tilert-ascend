#ifndef TESTS_GLM52_SHARED_W8_NATIVE_H
#define TESTS_GLM52_SHARED_W8_NATIVE_H

#include "shared_w8.h"
#include "../mla_tail/mla_tail_dequant.h"

namespace shared_w8::device {
using namespace AscendC;
using mla_tail::device::Fence;
using mla_tail::device::LoadTiling;
using mla_tail::device::Publish;

struct Buffers {
    TBuf<TPosition::VECCALC> input_buf, sum_buf, term_buf, scale_buf;
    TBuf<TPosition::VECCALC> value_buf, work_buf, maximum_buf, half_buf, code_buf;
    LocalTensor<int8_t> input, codes;
    LocalTensor<int32_t> sum, term;
    LocalTensor<float> scales, values, work, maximum;
    LocalTensor<half> halves;

    __aicore__ inline void Init(TPipe& pipe)
    {
        pipe.InitBuffer(input_buf, kInnerTile);
        pipe.InitBuffer(sum_buf, kGateColumns * 4);
        pipe.InitBuffer(term_buf, kGateColumns * 4);
        pipe.InitBuffer(scale_buf, (kGateColumns + kScaleStride) * 4);
        pipe.InitBuffer(value_buf, kGateColumns * 4);
        pipe.InitBuffer(work_buf, kGateColumns * 4);
        pipe.InitBuffer(maximum_buf, kScaleStride * 4);
        pipe.InitBuffer(half_buf, kIntermediate * 2);
        pipe.InitBuffer(code_buf, kIntermediate);
        input = input_buf.Get<int8_t>();
        sum = sum_buf.Get<int32_t>();
        term = term_buf.Get<int32_t>();
        scales = scale_buf.Get<float>();
        values = value_buf.Get<float>();
        work = work_buf.Get<float>();
        maximum = maximum_buf.Get<float>();
        halves = half_buf.Get<half>();
        codes = code_buf.Get<int8_t>();
    }
};

__aicore__ inline void StageInput(Buffers& held, __gm__ int8_t* input,
    __gm__ int8_t* staged, int64_t rows, int64_t capacity, int64_t vectors)
{
    GlobalTensor<int8_t> source, destination;
    source.SetGlobalBuffer(input, rows * kHidden);
    destination.SetGlobalBuffer(staged, capacity * kHidden);
    for (int64_t task = GetBlockIdx(); task < kInnerParts * capacity;
         task += vectors) {
        const int64_t part = task / capacity;
        const int64_t row = task % capacity;
        if (row < rows) {
            DataCopy(held.input, source[row * kHidden + part * kInnerTile],
                     kInnerTile);
            Fence<HardEvent::MTE2_MTE3>();
        } else {
            Duplicate(held.input.ReinterpretCast<int16_t>(), int16_t(0),
                      kInnerTile / 2);
            Fence<HardEvent::V_MTE3>();
        }
        DataCopy(destination[task * kInnerTile], held.input, kInnerTile);
        Fence<HardEvent::MTE3_V>();
        Fence<HardEvent::V_MTE2>();
    }
}

using Operand = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
using Product = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

__aicore__ constexpr ::MatmulApiStaticTiling MakeSharedGateTiling16()
{
    ::MatmulApiStaticTiling v{};
    v.M = 16; v.N = 128; v.Ka = 1024; v.Kb = 1024;
    v.singleCoreM = 16; v.singleCoreN = 128; v.singleCoreK = 1024;
    v.baseM = 16; v.baseN = 128; v.baseK = 256;
    v.depthA1 = 4; v.depthB1 = 4; v.stepM = 1; v.stepN = 1;
    v.stepKa = 4; v.stepKb = 4; v.iterateOrder = 0;
    v.shareL1Size = 147456; v.shareL0CSize = 8192; v.shareUbSize = 0;
    v.dbL0A = 2; v.dbL0B = 2; v.dbL0C = 1;
    v.cfg = ::CFG_NORM;
    return v;
}

__aicore__ constexpr ::MatmulApiStaticTiling MakeSharedDownTiling16()
{
    ::MatmulApiStaticTiling v{};
    v.M = 16; v.N = 256; v.Ka = 256; v.Kb = 256;
    v.singleCoreM = 16; v.singleCoreN = 256; v.singleCoreK = 256;
    v.baseM = 16; v.baseN = 256; v.baseK = 128;
    v.depthA1 = 2; v.depthB1 = 2; v.stepM = 1; v.stepN = 1;
    v.stepKa = 2; v.stepKb = 2; v.iterateOrder = 0;
    v.shareL1Size = 69632; v.shareL0CSize = 16384; v.shareUbSize = 0;
    v.dbL0A = 2; v.dbL0B = 2; v.dbL0C = 1;
    v.cfg = ::CFG_NORM;
    return v;
}

__aicore__ constexpr ::MatmulApiStaticTiling MakeSharedGateTiling32()
{
    ::MatmulApiStaticTiling v = MakeSharedGateTiling16();
    v.M = 32; v.singleCoreM = 32; v.baseM = 32;
    v.shareL1Size = 163840; v.shareL0CSize = 16384;
    return v;
}

__aicore__ constexpr ::MatmulApiStaticTiling MakeSharedDownTiling32()
{
    ::MatmulApiStaticTiling v = MakeSharedDownTiling16();
    v.M = 32; v.singleCoreM = 32; v.baseM = 32;
    v.shareL1Size = 73728; v.shareL0CSize = 32768;
    return v;
}

inline constexpr auto kSharedGateTiling16 = MakeSharedGateTiling16();
inline constexpr auto kSharedDownTiling16 = MakeSharedDownTiling16();
inline constexpr auto kSharedGateTiling32 = MakeSharedGateTiling32();
inline constexpr auto kSharedDownTiling32 = MakeSharedDownTiling32();

#ifdef K11150_SHARED_W8_NZ
__aicore__ inline void DirectSharedW8Nz24Aic(
    TPipe& pipe, __gm__ int8_t* input, __gm__ int8_t* weights,
    __gm__ int32_t* output, int64_t capacity, int64_t proj_rows, bool gate)
{
    const uint32_t cid = static_cast<uint32_t>(GetBlockIdx());
    if (cid >= 24U) {
        return;
    }
    const bool dual_m = (proj_rows > 16);
    const uint16_t m0 = dual_m ? 16U : static_cast<uint16_t>(proj_rows);
    const uint16_t m1 = dual_m ? static_cast<uint16_t>(proj_rows - 16) : 0U;
    const uint32_t m_cap = dual_m ? 32U : 16U;
    GlobalTensor<int8_t> a, b;
    GlobalTensor<int32_t> c;
    if (gate) {
        constexpr uint32_t kGateHalfBytes = static_cast<uint32_t>(kGateTile) * 512U;
        constexpr uint32_t kGateBBytes = kGateHalfBytes * 2U;
        constexpr uint32_t kGateRepeats = static_cast<uint32_t>(kGateTile) / 32U;
        constexpr uint32_t kL0bStep = kGateRepeats * 1024U;
        TBuf<TPosition::B1> b1_buf0, b1_buf1;
        TBuf<TPosition::A1> a1_buf;
        TBuf<TPosition::A2> l0a_buf;
        TBuf<TPosition::B2> l0b_buf0, l0b_buf1;
        TBuf<TPosition::CO1> l0c_buf;
        pipe.InitBuffer(b1_buf0, kGateHalfBytes);
        pipe.InitBuffer(b1_buf1, kGateHalfBytes);
        pipe.InitBuffer(a1_buf, m_cap * 1024U);
        pipe.InitBuffer(l0a_buf, m_cap * 1024U);
        pipe.InitBuffer(l0b_buf0, 256U * static_cast<uint32_t>(kGateTile));
        pipe.InitBuffer(l0b_buf1, 256U * static_cast<uint32_t>(kGateTile));
        pipe.InitBuffer(l0c_buf, m_cap * static_cast<uint32_t>(kGateTile) * sizeof(int32_t));

        LocalTensor<int8_t> a1 = a1_buf.Get<int8_t>();
        LocalTensor<int8_t> b1[2] = {b1_buf0.Get<int8_t>(), b1_buf1.Get<int8_t>()};
        LocalTensor<int8_t> l0a = l0a_buf.Get<int8_t>();
        LocalTensor<int8_t> l0b[2] = {l0b_buf0.Get<int8_t>(), l0b_buf1.Get<int8_t>()};
        LocalTensor<int32_t> l0c = l0c_buf.Get<int32_t>();

        a.SetGlobalBuffer(input + static_cast<uint64_t>(cid >> 2U) * static_cast<uint64_t>(capacity) * 1024ULL,
                          static_cast<uint64_t>(capacity) * 1024ULL);
        b.SetGlobalBuffer(weights + static_cast<uint64_t>(cid) * kGateBBytes, kGateBBytes);
        c.SetGlobalBuffer(output + static_cast<uint64_t>(cid) * static_cast<uint64_t>(capacity) * kGateTile,
                          static_cast<uint64_t>(capacity) * kGateTile);

        Nd2NzParams copy_a{};
        copy_a.ndNum = 1;
        copy_a.nValue = m0;
        copy_a.dValue = 1024;
        copy_a.srcDValue = 1024;
        copy_a.dstNzC0Stride = 16;
        copy_a.dstNzNStride = 1;
        DataCopy(a1, a, copy_a);
        if (dual_m) {
            copy_a.nValue = m1;
            DataCopy(a1[16384U], a[16384U], copy_a);
        }
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);

        DataCopy(b1[0], b[0], kGateHalfBytes);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        DataCopy(b1[1], b[kGateHalfBytes], kGateHalfBytes);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);

        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
        LoadData2DParams load_a{};
        load_a.repeatTimes = dual_m ? 64 : 32;
        load_a.srcStride = 1;
        load_a.ifTranspose = false;
        LoadData(l0a, a1, load_a);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);

        LoadData2dTransposeParams trans(0, static_cast<uint8_t>(kGateRepeats), 16, 1, 0, inc);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
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
                LoadDataWithTranspose(l0b[lbi][i * kL0bStep], b1[bi][k_sub_off + i * 1024U], trans);
            }
            SetFlag<HardEvent::MTE1_M>(lb_evt);
            WaitFlag<HardEvent::MTE1_M>(lb_evt);
            MmadParams mad{};
            mad.m = 16;
            mad.n = static_cast<uint16_t>(kGateTile);
            mad.k = 256;
            mad.cmatrixSource = false;
            mad.cmatrixInitVal = (step == 0U);
            Mmad(l0c, l0a[step * 4096U], l0b[lbi], mad);
            if (dual_m) {
                Mmad(l0c[kGateTile * 16U], l0a[16384U + step * 4096U], l0b[lbi], mad);
            }
            PipeBarrier<PIPE_M>();
            SetFlag<HardEvent::M_MTE1>(lb_evt);
        }
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
        FixpipeParamsV220 fix{};
        fix.nSize = static_cast<uint16_t>(kGateTile);
        fix.mSize = m0;
        fix.srcStride = 16;
        fix.dstStride = static_cast<uint16_t>(kGateTile);
        fix.ndNum = 1;
        fix.quantPre = QuantMode_t::NoQuant;
        Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(c, l0c, fix);
        if (dual_m) {
            fix.mSize = m1;
            Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(c[kGateTile * 16U], l0c[kGateTile * 16U], fix);
        }
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        pipe.Reset();
    } else {
        constexpr uint32_t kDnK = static_cast<uint32_t>(kIntermediate);
        constexpr uint32_t kDnSteps = kDnK / 128U;
        constexpr uint64_t kDnBBytes = static_cast<uint64_t>(kDnK) * 256ULL;
        TBuf<TPosition::B1> b1_buf0, b1_buf1;
        TBuf<TPosition::A1> a1_buf;
        TBuf<TPosition::A2> l0a_buf;
        TBuf<TPosition::B2> l0b_buf0, l0b_buf1;
        TBuf<TPosition::CO1> l0c_buf;
        pipe.InitBuffer(b1_buf0, 32768);
        pipe.InitBuffer(b1_buf1, 32768);
        pipe.InitBuffer(a1_buf, m_cap * kDnK);
        pipe.InitBuffer(l0a_buf, m_cap * kDnK);
        pipe.InitBuffer(l0b_buf0, 128 * 256);
        pipe.InitBuffer(l0b_buf1, 128 * 256);
        pipe.InitBuffer(l0c_buf, m_cap * 256U * sizeof(int32_t));

        LocalTensor<int8_t> a1 = a1_buf.Get<int8_t>();
        LocalTensor<int8_t> b1[2] = {b1_buf0.Get<int8_t>(), b1_buf1.Get<int8_t>()};
        LocalTensor<int8_t> l0a = l0a_buf.Get<int8_t>();
        LocalTensor<int8_t> l0b[2] = {l0b_buf0.Get<int8_t>(), l0b_buf1.Get<int8_t>()};
        LocalTensor<int32_t> l0c = l0c_buf.Get<int32_t>();

        a.SetGlobalBuffer(input, static_cast<uint64_t>(capacity) * kDnK);
        b.SetGlobalBuffer(weights + static_cast<uint64_t>(cid) * kDnBBytes, kDnBBytes);
        c.SetGlobalBuffer(output + static_cast<uint64_t>(cid) * static_cast<uint64_t>(capacity) * 256ULL,
                          static_cast<uint64_t>(capacity) * 256ULL);

        Nd2NzParams copy_a{};
        copy_a.ndNum = 1;
        copy_a.nValue = m0;
        copy_a.dValue = static_cast<uint16_t>(kDnK);
        copy_a.srcDValue = static_cast<uint16_t>(kDnK);
        copy_a.dstNzC0Stride = 16;
        copy_a.dstNzNStride = 1;
        DataCopy(a1, a, copy_a);
        if (dual_m) {
            copy_a.nValue = m1;
            DataCopy(a1[16U * kDnK], a[16U * kDnK], copy_a);
        }
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);

        DataCopy(b1[0], b[0], 32768);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        if constexpr (kDnSteps > 1U) {
            DataCopy(b1[1], b[32768], 32768);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
        }

        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
        LoadData2DParams load_a{};
        load_a.repeatTimes = dual_m ? (kDnSteps * 8U) : (kDnSteps * 4U);
        load_a.srcStride = 1;
        load_a.ifTranspose = false;
        LoadData(l0a, a1, load_a);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);

        LoadData2dTransposeParams trans(0, 8, 4, 1, 0, inc);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
        for (uint32_t step = 0U; step < kDnSteps; ++step) {
            const event_t b_evt = step ? EVENT_ID1 : EVENT_ID0;
            const event_t lb_evt = step ? EVENT_ID4 : EVENT_ID3;
            WaitFlag<HardEvent::MTE2_MTE1>(b_evt);
            WaitFlag<HardEvent::M_MTE1>(lb_evt);
            #pragma unroll
            for (uint32_t i = 0U; i < 4U; ++i) {
                LoadDataWithTranspose(l0b[step][i * 8192U], b1[step][i * 1024U], trans);
            }
            SetFlag<HardEvent::MTE1_M>(lb_evt);
            WaitFlag<HardEvent::MTE1_M>(lb_evt);
            MmadParams mad{};
            mad.m = 16;
            mad.n = 256;
            mad.k = 128;
            mad.cmatrixSource = false;
            mad.cmatrixInitVal = (step == 0U);
            Mmad(l0c, l0a[step * 2048U], l0b[step], mad);
            if (dual_m) {
                Mmad(l0c[4096U], l0a[16U * kDnK + step * 2048U], l0b[step], mad);
            }
            PipeBarrier<PIPE_M>();
            SetFlag<HardEvent::M_MTE1>(lb_evt);
        }
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
        Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(c, l0c, fix);
        if (dual_m) {
            fix.mSize = m1;
            Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(c[4096U], l0c[4096U], fix);
        }
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        pipe.Reset();
    }
}
#endif

__aicore__ inline void Project(TPipe& pipe, __gm__ int8_t* input,
    __gm__ int8_t* weights, __gm__ int32_t* output, __gm__ uint8_t* tiling,
    int64_t capacity, int64_t blocks, bool gate, int64_t active_rows = 0)
{
    const int64_t tasks = gate ? kInnerParts * kGateParts : kDownParts;
    if (GetBlockIdx() >= tasks) {
        return;
    }
#ifdef K11150_SHARED_W8_NZ
    const int64_t proj_rows = (active_rows > 0 && active_rows <= capacity) ? active_rows : capacity;
    if (proj_rows <= 32 && capacity <= 32 && blocks >= 24) {
        DirectSharedW8Nz24Aic(pipe, input, weights, output, capacity, proj_rows, gate);
        return;
    }
#endif
    const int64_t columns = gate ? kGateTile : kDownTile;
    const int64_t inner = gate ? kInnerTile : kIntermediate;
    GlobalTensor<int8_t> a, b;
    GlobalTensor<int32_t> c;
    if (capacity == 16) {
        if (gate) {
            TCubeTiling cube{}; cube.M = 16; cube.singleCoreM = 16;
            Matmul<Operand, Operand, Product, Product, kSharedGateTiling16> mm;
            mm.Init(&cube, &pipe);
            for (int64_t task = GetBlockIdx(); task < tasks; task += blocks) {
                const int64_t input_part = task / kGateParts;
                a.SetGlobalBuffer(input + input_part * 16 * inner, 16 * inner);
                b.SetGlobalBuffer(weights + task * inner * columns, inner * columns);
                c.SetGlobalBuffer(output + task * 16 * columns, 16 * columns);
                mm.SetOrgShape(16, columns, inner);
                mm.SetTensorA(a);
                mm.SetTensorB(b, false);
                mm.IterateAll(c);
            }
            mm.End();
        } else {
            TCubeTiling cube{}; cube.M = 16; cube.singleCoreM = 16;
            Matmul<Operand, Operand, Product, Product, kSharedDownTiling16> mm;
            mm.Init(&cube, &pipe);
            for (int64_t task = GetBlockIdx(); task < tasks; task += blocks) {
                a.SetGlobalBuffer(input, 16 * inner);
                b.SetGlobalBuffer(weights + task * inner * columns, inner * columns);
                c.SetGlobalBuffer(output + task * 16 * columns, 16 * columns);
                mm.SetOrgShape(16, columns, inner);
                mm.SetTensorA(a);
                mm.SetTensorB(b, false);
                mm.IterateAll(c);
            }
            mm.End();
        }
        return;
    } else if (capacity == 32) {
        if (gate) {
            TCubeTiling cube{}; cube.M = 32; cube.singleCoreM = 32;
            Matmul<Operand, Operand, Product, Product, kSharedGateTiling32> mm;
            mm.Init(&cube, &pipe);
            for (int64_t task = GetBlockIdx(); task < tasks; task += blocks) {
                const int64_t input_part = task / kGateParts;
                a.SetGlobalBuffer(input + input_part * 32 * inner, 32 * inner);
                b.SetGlobalBuffer(weights + task * inner * columns, inner * columns);
                c.SetGlobalBuffer(output + task * 32 * columns, 32 * columns);
                mm.SetOrgShape(32, columns, inner);
                mm.SetTensorA(a);
                mm.SetTensorB(b, false);
                mm.IterateAll(c);
            }
            mm.End();
        } else {
            TCubeTiling cube{}; cube.M = 32; cube.singleCoreM = 32;
            Matmul<Operand, Operand, Product, Product, kSharedDownTiling32> mm;
            mm.Init(&cube, &pipe);
            for (int64_t task = GetBlockIdx(); task < tasks; task += blocks) {
                a.SetGlobalBuffer(input, 32 * inner);
                b.SetGlobalBuffer(weights + task * inner * columns, inner * columns);
                c.SetGlobalBuffer(output + task * 32 * columns, 32 * columns);
                mm.SetOrgShape(32, columns, inner);
                mm.SetTensorA(a);
                mm.SetTensorB(b, false);
                mm.IterateAll(c);
            }
            mm.End();
        }
        return;
    }
    TCubeTiling cube;
    LoadTiling(cube, tiling);
    Matmul<Operand, Operand, Product, Product> mm;
    mm.Init(&cube, &pipe);
    for (int64_t task = GetBlockIdx(); task < tasks; task += blocks) {
        const int64_t input_part = gate ? task / kGateParts : 0;
        a.SetGlobalBuffer(input + input_part * capacity * inner,
                          capacity * inner);
        b.SetGlobalBuffer(weights + task * inner * columns, inner * columns);
        c.SetGlobalBuffer(output + task * capacity * columns, capacity * columns);
        mm.SetOrgShape(capacity, columns, inner);
        mm.SetTensorA(a);
        mm.SetTensorB(b, false);
        mm.IterateAll(c);
        mm.End();
    }
}

__aicore__ inline void Activate(Buffers& held, __gm__ int32_t* partial,
    __gm__ float* channel_scale, __gm__ float* input_scale,
    __gm__ int8_t* middle, __gm__ float* middle_scale,
    int64_t rows, int64_t capacity, int64_t vectors)
{
    GlobalTensor<int32_t> source;
    GlobalTensor<float> channels, tokens, out_scale;
    GlobalTensor<int8_t> output;
    source.SetGlobalBuffer(partial, kInnerParts * kGateColumns * capacity);
    channels.SetGlobalBuffer(channel_scale, kGateColumns);
    tokens.SetGlobalBuffer(input_scale, rows);
    output.SetGlobalBuffer(middle, capacity * kIntermediate);
    out_scale.SetGlobalBuffer(middle_scale, capacity * kScaleStride);
    DataCopyParams load_parts;
    load_parts.blockCount = kGateParts;
    load_parts.blockLen = kGateTile * sizeof(int32_t) / 32;
    load_parts.srcStride = (capacity - 1) * kGateTile * sizeof(int32_t) / 32;
    load_parts.dstStride = 0;
    for (int64_t row = GetBlockIdx(); row < rows; row += vectors) {
        {
            DataCopy(held.sum, source[row * kGateTile], load_parts);
            Fence<HardEvent::MTE2_V>();
            for (int64_t part = 1; part < kInnerParts; ++part) {
                DataCopy(held.term,
                    source[(part * kGateParts * capacity + row) * kGateTile],
                    load_parts);
                Fence<HardEvent::MTE2_V>();
                Add(held.sum, held.sum, held.term, kGateColumns);
                Fence<HardEvent::V_MTE2>();
            }
            DataCopy(held.scales, channels, kGateColumns);
            DataCopyPad(held.scales[kGateColumns], tokens[row],
                DataCopyExtParams{1, sizeof(float), 0, 0, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
            Fence<HardEvent::MTE2_V>();
            Fence<HardEvent::MTE2_S>();


            Cast(held.values, held.sum, RoundMode::CAST_NONE, kGateColumns);
            PipeBarrier<PIPE_V>();
            Mul(held.values, held.values, held.scales, kGateColumns);
            PipeBarrier<PIPE_V>();
            Muls(held.values, held.values,
                 held.scales.GetValue(kGateColumns), kGateColumns);
            PipeBarrier<PIPE_V>();
            Muls(held.work, held.values, -1.0F, kIntermediate);
            PipeBarrier<PIPE_V>();
            Exp(held.work, held.work, kIntermediate);
            PipeBarrier<PIPE_V>();
            Adds(held.work, held.work, 1.0F, kIntermediate);
            PipeBarrier<PIPE_V>();
            Div(held.values, held.values, held.work, kIntermediate);
            PipeBarrier<PIPE_V>();
            Mul(held.values, held.values, held.values[kIntermediate],
                kIntermediate);
            PipeBarrier<PIPE_V>();
            Abs(held.work, held.values, kIntermediate);
            Duplicate(held.maximum, 0.0F, kScaleStride);
            PipeBarrier<PIPE_V>();
            ReduceMax(held.maximum, held.work, held.work[kIntermediate],
                      kIntermediate, false);
            PipeBarrier<PIPE_V>();
            Muls(held.maximum, held.maximum, 1.0F / 127.0F, kScaleStride);
            Fence<HardEvent::V_S>();
            const float scale = held.maximum.GetValue(0);
            Fence<HardEvent::S_V>();
            Duplicate(held.work, scale == 0.0F ? 1.0F : scale, kIntermediate);
            PipeBarrier<PIPE_V>();
            Div(held.values, held.values, held.work, kIntermediate);
            PipeBarrier<PIPE_V>();
            Cast(held.sum, held.values, RoundMode::CAST_RINT, kIntermediate);
            PipeBarrier<PIPE_V>();
            SetDeqScale(static_cast<half>(1.0F));
            PipeBarrier<PIPE_V>();
            Cast(held.halves, held.sum, RoundMode::CAST_ROUND, kIntermediate);
            PipeBarrier<PIPE_V>();
            Cast(held.codes, held.halves, RoundMode::CAST_TRUNC, kIntermediate);
        }
        Fence<HardEvent::V_MTE3>();
        DataCopy(output[row * kIntermediate], held.codes, kIntermediate);
        DataCopy(out_scale[row * kScaleStride], held.maximum, kScaleStride);
        Fence<HardEvent::MTE3_V>();
        Fence<HardEvent::V_MTE2>();
    }
}
}

#endif
