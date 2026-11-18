#ifndef TESTS_GLM52_MLA_TAIL_NATIVE_H
#define TESTS_GLM52_MLA_TAIL_NATIVE_H


#ifndef ASCENDC_CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "mla_tail.h"

namespace mla_tail::device {
using namespace AscendC;

template <HardEvent event>
__aicore__ inline void Fence()
{
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
}

__aicore__ inline void Publish()
{
    PipeBarrier<PIPE_ALL>();
#ifndef K112702_NO_DDR_INTRADIE
    DataSyncBarrier<MemDsbT::DDR>();
#endif
}

__aicore__ inline void LoadTiling(TCubeTiling& cube, __gm__ uint8_t* source)
{
    auto* destination = reinterpret_cast<uint32_t*>(&cube);
    auto* words = reinterpret_cast<__gm__ uint32_t*>(source);
    for (uint32_t index = 0; index < sizeof(TCubeTiling) / 4; ++index) {
        destination[index] = words[index];
    }
}

template <int64_t kHeads = mla_tail::kHeads>
struct VectorWorkspace {
    static constexpr int64_t kProjectionInput = kHeads * kValue;
    TBuf<TPosition::VECCALC> bf16_buffer;
    TBuf<TPosition::VECCALC> half_buffer;
    TBuf<TPosition::VECCALC> code_buffer;
    TBuf<TPosition::VECCALC> value_buffer;
    TBuf<TPosition::VECCALC> other_buffer;
    TBuf<TPosition::VECCALC> work_buffer;
    TBuf<TPosition::VECCALC> accumulator_buffer;
    TBuf<TPosition::VECCALC> bias_buffer;
    TBuf<TPosition::VECCALC> scalar_buffer;
    LocalTensor<bfloat16_t> bf16;
    LocalTensor<half> fp16;
    LocalTensor<int8_t> codes;
    LocalTensor<float> value;
    LocalTensor<float> other;
    LocalTensor<float> work;
    LocalTensor<int32_t> accumulator;
    LocalTensor<int32_t> bias;
    LocalTensor<float> scalar;

    __aicore__ inline void Init(TPipe& pipe)
    {
        pipe.InitBuffer(bf16_buffer, kProjectionInput * 2);
        pipe.InitBuffer(half_buffer, kProjectionInput * 2);
        pipe.InitBuffer(code_buffer, kProjectionInput);
        pipe.InitBuffer(value_buffer, kProjectionInput * 4);
        pipe.InitBuffer(other_buffer, kProjectionInput * 4);
        pipe.InitBuffer(work_buffer, kProjectionInput * 4);
        pipe.InitBuffer(accumulator_buffer, kOutputTile * 4);
        pipe.InitBuffer(bias_buffer, kOutputTile * 4);
        pipe.InitBuffer(scalar_buffer, kScaleStride * 4);
        bf16 = bf16_buffer.Get<bfloat16_t>();
        fp16 = half_buffer.Get<half>();
        codes = code_buffer.Get<int8_t>();
        value = value_buffer.Get<float>();
        other = other_buffer.Get<float>();
        work = work_buffer.Get<float>();
        accumulator = accumulator_buffer.Get<int32_t>();
        bias = bias_buffer.Get<int32_t>();
        scalar = scalar_buffer.Get<float>();
    }
};

constexpr int64_t kStageRows = 8;

template <int64_t kHeads = mla_tail::kHeads>
__aicore__ inline void StageHeads(VectorWorkspace<kHeads>& held,
                                 __gm__ bfloat16_t* latent,
                                 __gm__ bfloat16_t* heads,
                                 int64_t rows, int64_t capacity,
                                 int64_t vector_index, int64_t vectors)
{
    GlobalTensor<bfloat16_t> source;
    GlobalTensor<bfloat16_t> destination;
    source.SetGlobalBuffer(latent, rows * kHeads * kLatent);
    destination.SetGlobalBuffer(heads, capacity * kHeads * kLatent);
    const int64_t stage_rows = rows <= 6 ? 1 : ((rows + 5) / 6);
    const int64_t batches = (rows + stage_rows - 1) / stage_rows;
    auto stage = held.value.template ReinterpretCast<bfloat16_t>();
    for (int64_t task = vector_index; task < kHeads * batches;
         task += vectors) {
        const int64_t head = task / batches;
        const int64_t begin = (task % batches) * stage_rows;
        const int64_t valid = rows <= begin
            ? 0 : (rows - begin < stage_rows ? rows - begin : stage_rows);
        if (valid <= 0) {
            continue;
        }
        DataCopyParams copy{
            static_cast<uint16_t>(valid),
            static_cast<uint16_t>(kLatent * sizeof(bfloat16_t) / 32),
            static_cast<uint16_t>((kHeads - 1) * kLatent * sizeof(bfloat16_t) / 32),
            0,
        };
        DataCopy(stage, source[(begin * kHeads + head) * kLatent], copy);
        Fence<HardEvent::MTE2_MTE3>();
        DataCopy(destination[(head * capacity + begin) * kLatent], stage,
                 valid * kLatent);
        if (task + vectors < kHeads * batches) {
            Fence<HardEvent::MTE3_MTE2>();
        }
    }
}

using ValueA = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using ValueB = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using ValueC = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using ValueBias = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using ValueBNZ = MatmulType<TPosition::GM, CubeFormat::NZ, bfloat16_t, false>;
using OutputA = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
using OutputB = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
using OutputBNZ = MatmulType<TPosition::GM, CubeFormat::NZ, int8_t, false>;
using OutputC = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using OutputBias = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

#ifdef K11070_TAIL_K256
template <typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class TailFastCubeInBuffer {
    using TransT = typename INPUT_TYPE::TRANS_T;
public:
    __aicore__ inline TailFastCubeInBuffer() {}
    __aicore__ inline ~TailFastCubeInBuffer() {}
    __aicore__ inline void Init(int32_t baseBlockSize, int32_t cacheNum)
    {
        baseBlockSize_ = baseBlockSize;
        int32_t matrixByteSize = baseBlockSize_ * AscendC::GetBitSize<TransT>() / AscendC::ONE_BYTE_BIT_SIZE;
        GetTPipePtr()->InitBuffer(qidCache_, 1, cacheNum * matrixByteSize);
    }
    __aicore__ inline void Destroy()
    {
        if (validMask_ != 0ULL) {
            qidCache_.FreeTensor(cacheHead_);
            validMask_ = 0ULL;
        }
        qidCache_.FreeAllEvent();
    }
    __aicore__ inline AscendC::LocalTensor<TransT> AllocTensor(int32_t bufferPos = -1)
    {
        if (validMask_ == 0ULL) {
            cacheHead_ = qidCache_.template AllocTensor<TransT>();
        }
        validMask_ |= (1ULL << static_cast<uint32_t>(bufferPos));
        return cacheHead_[bufferPos * baseBlockSize_];
    }
    __aicore__ inline void FreeTensor(int32_t = -1, const AscendC::LocalTensor<TransT>& = AscendC::LocalTensor<TransT>{}) {}
    __aicore__ inline void Reset()
    {
        if (validMask_ != 0ULL) {
            qidCache_.FreeTensor(cacheHead_);
            validMask_ = 0ULL;
        }
    }
    __aicore__ inline bool Hit(int32_t iterIndex, int32_t = -1)
    {
        return ((validMask_ >> static_cast<uint32_t>(iterIndex)) & 1ULL) != 0ULL;
    }
    __aicore__ inline AscendC::LocalTensor<TransT> GetBuffer(int32_t iterIndex, int32_t = -1)
    {
        return cacheHead_[iterIndex * baseBlockSize_];
    }
    __aicore__ inline void EnQue(AscendC::LocalTensor<TransT>&)
    {
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(static_cast<event_t>(1 + (evtIdx_ & 1)));
        }
    }
    __aicore__ inline void DeQue()
    {
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(static_cast<event_t>(1 + (evtIdx_++ & 1)));
        }
    }
private:
    typename AscendC::Impl::Detail::CubeInQueType<INPUT_TYPE>::QUE qidCache_;
    AscendC::LocalTensor<TransT> cacheHead_;
    int32_t baseBlockSize_{0};
    uint64_t validMask_{0};
    uint32_t evtIdx_{0};
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
struct TailFastMatmulPolicy : public AscendC::Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
    using TransT_B = decltype(GetTransBDataType<A_TYPE, B_TYPE, MM_CFG>());
    using CubeInBufferA = TailFastCubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = TailFastCubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename TransT_B::T>, MM_CFG>;
};
#endif


__aicore__ constexpr ::MatmulApiStaticTiling MakeValueTiling(int32_t m_cap) {
    ::MatmulApiStaticTiling v{};
    v.usedCoreNum = 1;
    v.M = m_cap;
    v.N = 128;
    v.Ka = 2048;
    v.Kb = 512;
    v.singleCoreM = m_cap;
    v.singleCoreN = 128;
    v.singleCoreK = 512;
    v.baseM = m_cap;
    v.baseN = 128;
    v.baseK = 128;
    v.depthA1 = 4;
    v.depthB1 = 4;
    v.stepM = 1;
    v.stepN = 1;
    v.stepKa = 4;
    v.stepKb = 4;
    v.iterateOrder = 0;
    v.shareL1Size = (m_cap == 16) ? 147456 : 163840;
    v.shareL0CSize = (m_cap == 16) ? 8192 : 16384;
    v.dbL0A = 2;
    v.dbL0B = 2;
    v.dbL0C = 1;
    v.cfg = ::CFG_NORM;
    return v;
}
__aicore__ constexpr ::MatmulApiStaticTiling MakeOutputTiling(int32_t m_cap) {
    ::MatmulApiStaticTiling v{};
    v.usedCoreNum = 1;
    v.M = m_cap;
    v.N = 256;
    v.Ka = 2048;
    v.Kb = 2048;
    v.singleCoreM = m_cap;
    v.singleCoreN = 256;
    v.singleCoreK = 2048;
    v.baseM = m_cap;
#ifdef K11070_TAIL_K256
    v.baseN = (m_cap == 16) ? 128 : 256;
    v.baseK = (m_cap == 16) ? 256 : 128;
    v.depthA1 = (m_cap == 16) ? 8 : 16;
    v.depthB1 = 8;
    v.stepM = 1;
    v.stepN = 1;
    v.stepKa = (m_cap == 16) ? 8 : 16;
    v.stepKb = 4;
#else
    v.baseN = 256;
    v.baseK = 128;
    v.depthA1 = 16;
    v.depthB1 = 8;
    v.stepM = 1;
    v.stepN = 1;
    v.stepKa = 16;
    v.stepKb = 4;
#endif
    v.iterateOrder = 1;
    v.shareL1Size = (m_cap == 16) ? 294912 : 327680;
    v.shareL0CSize = (m_cap == 16) ? 16384 : 32768;
    v.dbL0A = 2;
    v.dbL0B = 2;
    v.dbL0C = 1;
    v.cfg = ::CFG_NORM;
    return v;
}
inline constexpr auto kValueTiling16 = MakeValueTiling(16);
inline constexpr auto kValueTiling32 = MakeValueTiling(32);
__aicore__ constexpr ::MatmulApiStaticTiling MakeQueryTiling(int32_t m_cap) {
    ::MatmulApiStaticTiling v = MakeOutputTiling(m_cap);
    v.shareL0CSize = (m_cap == 16) ? 32768 : 65536;
    v.dbL0C = 2;
    return v;
}
inline constexpr auto kOutputTiling16 = MakeOutputTiling(16);
inline constexpr auto kOutputTiling32 = MakeOutputTiling(32);
inline constexpr auto kQueryTiling16 = MakeQueryTiling(16);
inline constexpr auto kQueryTiling32 = MakeQueryTiling(32);

template <int64_t kHeads = mla_tail::kHeads, const auto& kValTiling>
__aicore__ inline void RunProjectValues(TPipe& pipe, __gm__ bfloat16_t* latent, __gm__ bfloat16_t* weights, __gm__ float* values, int64_t capacity, int64_t blocks, int64_t proj_rows)
{
    constexpr int64_t kProjectionInput = kHeads * kValue;
    constexpr int64_t kValueTile = kProjectionInput / kValueTiles;
    TCubeTiling cube{};
    cube.M = capacity;
    cube.singleCoreM = proj_rows;
#if defined(K11040_TAIL_NZ) && defined(K11070_TAIL_K256)
    Matmul<ValueA, ValueBNZ, ValueC, ValueBias, kValTiling, MatmulCallBackFunc<nullptr, nullptr, nullptr>, TailFastMatmulPolicy> mm;
#elif defined(K11040_TAIL_NZ)
    Matmul<ValueA, ValueBNZ, ValueC, ValueBias, kValTiling> mm;
#else
    Matmul<ValueA, ValueB, ValueC, ValueBias, kValTiling> mm;
#endif
    mm.Init(&cube, &pipe);
    GlobalTensor<bfloat16_t> a, b;
    GlobalTensor<float> c;
    for (int64_t tile = GetBlockIdx(); tile < kValueTiles; tile += blocks) {
        const int64_t head = tile / (kValue / kValueTile);
        a.SetGlobalBuffer(latent + head * kLatent, capacity * kHeads * kLatent);
        b.SetGlobalBuffer(weights + tile * kLatent * kValueTile, kLatent * kValueTile);
        c.SetGlobalBuffer(values + tile * kValueTile, capacity * kProjectionInput);
        mm.SetOrgShape(proj_rows, kValueTile, kHeads * kLatent, kLatent, kProjectionInput);
        mm.SetSingleShape(proj_rows, kValueTile, kLatent);
        mm.SetTensorA(a);
        mm.SetTensorB(b, false);
        mm.IterateAll(c);
    }
    mm.End();
}

#if defined(K11140_DIRECT_MLA_TAIL) && defined(K11040_TAIL_NZ)
__aicore__ inline void DirectProjectValues24Aic(
    TPipe& pipe, __gm__ bfloat16_t* latent, __gm__ bfloat16_t* weights,
    __gm__ float* values, int64_t capacity, int64_t proj_rows)
{
    const uint32_t cid = static_cast<uint32_t>(GetBlockIdx());
    if (cid >= 24U) {
        return;
    }
    const bool dual_m = (proj_rows > 16);
    const uint16_t m0 = dual_m ? 16U : static_cast<uint16_t>(proj_rows);
    const uint16_t m1 = dual_m ? static_cast<uint16_t>(proj_rows - 16) : 0U;
    const uint32_t m_cap = dual_m ? 32U : 16U;
    const uint32_t head = (cid * 683U) >> 11U;
    const uint32_t sub = cid - head * 3U;
    const uint32_t strip_off = sub * 6U;
    const uint32_t n1 = (sub < 2U) ? 6U : 4U;
    const uint16_t n_cols = static_cast<uint16_t>(n1 * 16U);
    const uint32_t col_off = strip_off * 16U;

    TBuf<TPosition::B1> b1_buf0, b1_buf1;
    TBuf<TPosition::A1> a1_buf;
    TBuf<TPosition::A2> l0a_buf;
    TBuf<TPosition::B2> l0b_buf0, l0b_buf1;
    TBuf<TPosition::CO1> l0c_buf;
    pipe.InitBuffer(b1_buf0, 65536);
    pipe.InitBuffer(b1_buf1, 65536);
    pipe.InitBuffer(a1_buf, m_cap * 512U * sizeof(bfloat16_t));
    pipe.InitBuffer(l0a_buf, m_cap * 512U * sizeof(bfloat16_t));
    pipe.InitBuffer(l0b_buf0, 32768);
    pipe.InitBuffer(l0b_buf1, 32768);
    pipe.InitBuffer(l0c_buf, m_cap * 96U * sizeof(float));

    LocalTensor<bfloat16_t> a1 = a1_buf.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> b1[2] = {b1_buf0.Get<bfloat16_t>(), b1_buf1.Get<bfloat16_t>()};
    LocalTensor<bfloat16_t> l0a = l0a_buf.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> l0b[2] = {l0b_buf0.Get<bfloat16_t>(), l0b_buf1.Get<bfloat16_t>()};
    LocalTensor<float> l0c = l0c_buf.Get<float>();

    GlobalTensor<bfloat16_t> a_gm, b_gm;
    GlobalTensor<float> c_gm;
    a_gm.SetGlobalBuffer(latent + static_cast<uint64_t>(head) * 512ULL, static_cast<uint64_t>(capacity) * 4096ULL);
    b_gm.SetGlobalBuffer(weights + static_cast<uint64_t>(head * 16U + strip_off) * 8192ULL, static_cast<uint64_t>(n1) * 8192ULL);
    c_gm.SetGlobalBuffer(values + static_cast<uint64_t>(head * 256U + col_off), static_cast<uint64_t>(capacity) * 2048ULL);

    Nd2NzParams copy_a{};
    copy_a.ndNum = 1;
    copy_a.nValue = m0;
    copy_a.dValue = 512;
    copy_a.srcDValue = 4096;
    copy_a.dstNzC0Stride = 16;
    copy_a.dstNzNStride = 1;
    DataCopy(a1, a_gm, copy_a);
    if (dual_m) {
        copy_a.nValue = m1;
        DataCopy(a1[8192U], a_gm[16U * 4096U], copy_a);
    }
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);

    DataCopy(b1[0], b_gm[0], DataCopyParams{static_cast<uint16_t>(n1), 256, 256, 0});
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    DataCopy(b1[1], b_gm[256 * 16], DataCopyParams{static_cast<uint16_t>(n1), 256, 256, 0});
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);

    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
    LoadData2DParams load_a{};
    load_a.repeatTimes = dual_m ? 64 : 32;
    load_a.srcStride = 1;
    load_a.ifTranspose = false;
    LoadData(l0a, a1, load_a);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);

    LoadData2DParams load_b{};
    load_b.repeatTimes = 8;
    load_b.srcStride = 1;
    load_b.dstGap = static_cast<uint16_t>(n1 - 1U);
    load_b.ifTranspose = true;

    SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
    for (uint32_t step = 0U; step < 4U; ++step) {
        const uint32_t bi = step >> 1U;
        const uint32_t k_sub = (step & 1U) * 128U;
        const uint32_t lbi = step & 1U;
        const event_t lb_evt = lbi ? EVENT_ID4 : EVENT_ID3;
        if (step == 0U) {
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        } else if (step == 2U) {
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
        }
        WaitFlag<HardEvent::M_MTE1>(lb_evt);
        for (uint32_t ni = 0U; ni < n1; ++ni) {
            LoadData(l0b[lbi][ni * 256U], b1[bi][ni * 4096U + k_sub * 16U], load_b);
        }
        SetFlag<HardEvent::MTE1_M>(lb_evt);
        WaitFlag<HardEvent::MTE1_M>(lb_evt);
        MmadParams mad{};
        mad.m = 16;
        mad.n = n_cols;
        mad.k = 128;
        mad.cmatrixSource = false;
        mad.cmatrixInitVal = (step == 0U);
        Mmad(l0c, l0a[step * 2048U], l0b[lbi], mad);
        if (dual_m) {
            Mmad(l0c[1536U], l0a[8192U + step * 2048U], l0b[lbi], mad);
        }
#ifdef K112704_TAIL_PIPE
        if (step == 3U) {
            PipeBarrier<PIPE_M>();
        }
#elif defined(K112186_MLA_TAIL_FAST)
        if (!dual_m || n1 < 5U || step == 3U) {
            PipeBarrier<PIPE_M>();
        }
#else
        PipeBarrier<PIPE_M>();
#endif
        SetFlag<HardEvent::M_MTE1>(lb_evt);
    }
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    FixpipeParamsV220 fix{};
    fix.nSize = n_cols;
    fix.mSize = m0;
    fix.srcStride = 16;
    fix.dstStride = 2048;
    fix.ndNum = 1;
#ifdef K112186_MLA_TAIL_FAST
    GlobalTensor<bfloat16_t> c_bf16_gm;
    c_bf16_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t*>(values) + static_cast<uint64_t>(head * 256U + col_off),
        static_cast<uint64_t>(capacity) * 2048ULL);
    fix.quantPre = QuantMode_t::F322BF16;
    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(c_bf16_gm, l0c, fix);
    if (dual_m) {
        fix.mSize = m1;
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(c_bf16_gm[16U * 2048U], l0c[1536U], fix);
    }
#else
    fix.quantPre = QuantMode_t::NoQuant;
    Fixpipe<float, float, CFG_ROW_MAJOR>(c_gm, l0c, fix);
    if (dual_m) {
        fix.mSize = m1;
        Fixpipe<float, float, CFG_ROW_MAJOR>(c_gm[16U * 2048U], l0c[1536U], fix);
    }
#endif
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
}
#endif

template <int64_t kHeads = mla_tail::kHeads>
__aicore__ inline void ProjectValues(TPipe& pipe,
                                    __gm__ bfloat16_t* latent,
                                    __gm__ bfloat16_t* weights,
                                    __gm__ float* values,
                                    __gm__ uint8_t* tiling,
                                    int64_t capacity, int64_t blocks,
                                    int64_t rows = 0)
{
    const int64_t active_rows = rows > 0 ? rows : capacity;
#if defined(K11140_DIRECT_MLA_TAIL) && defined(K11040_TAIL_NZ)
    if constexpr (kHeads == 8) {
#ifdef K11150_MLA_TAIL_M32
        if (active_rows <= 32 && blocks >= 24) {
#else
        if (active_rows <= 16 && blocks >= 24) {
#endif
            DirectProjectValues24Aic(pipe, latent, weights, values, capacity, active_rows);
            return;
        }
    }
#endif
    if (GetBlockIdx() >= kValueTiles) {
        return;
    }
    if (active_rows <= 16) {
        RunProjectValues<kHeads, kValueTiling16>(pipe, latent, weights, values, capacity, blocks, active_rows);
    } else {
        RunProjectValues<kHeads, kValueTiling32>(pipe, latent, weights, values, capacity, blocks, active_rows);
    }
}

template <int64_t kHeads = mla_tail::kHeads>
__aicore__ inline void QuantizeValues(VectorWorkspace<kHeads>& held,
                                     __gm__ float* values,
                                     __gm__ int8_t* quantized,
                                     __gm__ float* token_scales,
                                     int64_t capacity, int64_t vector_index,
                                     int64_t vectors, int64_t rows = 0)
{
    const int64_t active_rows = rows > 0 ? rows : capacity;
    if (vector_index >= active_rows) {
        return;
    }
    constexpr int64_t kProjectionInput = kHeads * kValue;
    GlobalTensor<float> source;
    GlobalTensor<int8_t> destination;
    GlobalTensor<float> scales;
    source.SetGlobalBuffer(values, capacity * kProjectionInput);
    destination.SetGlobalBuffer(quantized, capacity * kProjectionInput);
    scales.SetGlobalBuffer(token_scales, capacity * kScaleStride);
    SetDeqScale(static_cast<half>(1.0F));
    for (int64_t row = vector_index; row < active_rows; row += vectors) {
#if defined(K112186_MLA_TAIL_FAST) && defined(K11140_DIRECT_MLA_TAIL) && defined(K11040_TAIL_NZ) && defined(K11150_MLA_TAIL_M32)
        if (kHeads == 8 && active_rows <= 32 && vectors >= 48) {
            GlobalTensor<bfloat16_t> src_bf16;
            src_bf16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(values), capacity * kProjectionInput);
            DataCopy(held.bf16, src_bf16[row * kProjectionInput], kProjectionInput);
            Fence<HardEvent::MTE2_V>();
            Cast(held.value, held.bf16, RoundMode::CAST_NONE, kProjectionInput);
        } else
#endif
        {
            DataCopy(held.value, source[row * kProjectionInput], kProjectionInput);
            Fence<HardEvent::MTE2_V>();
            Cast(held.bf16, held.value, RoundMode::CAST_RINT, kProjectionInput);
            PipeBarrier<PIPE_V>();
            Cast(held.value, held.bf16, RoundMode::CAST_NONE, kProjectionInput);
        }
        PipeBarrier<PIPE_V>();
        Abs(held.other, held.value, kProjectionInput);
        Duplicate(held.scalar, 0.0F, kScaleStride);
        PipeBarrier<PIPE_V>();
        ReduceMax(held.scalar, held.other, held.work, kProjectionInput, false);
        Fence<HardEvent::V_S>();
        const float maximum = held.scalar.GetValue(0);


        Duplicate(held.work, maximum == 0.0F ? 1.0F : maximum, 64);
        Duplicate(held.other, 127.0F, 64);
        PipeBarrier<PIPE_V>();
        Div(held.other, held.other, held.work, 64);
        PipeBarrier<PIPE_V>();
        Mul(held.value, held.value, held.other, 64, kProjectionInput / 64,
            BinaryRepeatParams{1, 1, 0, 8, 8, 0});
        Muls(held.scalar, held.scalar, 1.0F / 127.0F, kScaleStride);
        PipeBarrier<PIPE_V>();
        Mins(held.value, held.value, 127.0F, kProjectionInput);
        PipeBarrier<PIPE_V>();
        Maxs(held.value, held.value, -127.0F, kProjectionInput);
        PipeBarrier<PIPE_V>();


        auto integers = held.work.template ReinterpretCast<int32_t>();
        Cast(integers, held.value, RoundMode::CAST_RINT, kProjectionInput);
        PipeBarrier<PIPE_V>();
        Cast(held.fp16, integers, RoundMode::CAST_ROUND, kProjectionInput);
        PipeBarrier<PIPE_V>();
        Cast(held.codes, held.fp16, RoundMode::CAST_TRUNC, kProjectionInput);
        Fence<HardEvent::V_MTE3>();
        DataCopy(destination[row * kProjectionInput], held.codes,
                 kProjectionInput);
        DataCopy(scales[row * kScaleStride], held.scalar, kScaleStride);
        if (row + vectors < active_rows) {
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::V_MTE2>();
        }
    }
}


template <int64_t kHeads = mla_tail::kHeads, const auto& kOutTiling>
__aicore__ inline void RunProjectOutput(TPipe& pipe, __gm__ int8_t* quantized, __gm__ int8_t* weights, __gm__ int32_t* accumulators, int64_t capacity, int64_t blocks, int64_t proj_rows)
{
    constexpr int64_t kProjectionInput = kHeads * kValue;
    TCubeTiling cube{};
    cube.M = capacity;
    cube.singleCoreM = proj_rows;
#ifdef K11040_TAIL_NZ
    Matmul<OutputA, OutputBNZ, OutputC, OutputBias, kOutTiling> mm;
#else
    Matmul<OutputA, OutputB, OutputC, OutputBias, kOutTiling> mm;
#endif
    mm.Init(&cube, &pipe);
    GlobalTensor<int8_t> a, b;
    GlobalTensor<int32_t> c;
    a.SetGlobalBuffer(quantized, capacity * kProjectionInput);
    for (int64_t tile = GetBlockIdx(); tile < kOutputTiles; tile += blocks) {
        b.SetGlobalBuffer(weights + tile * kProjectionInput * kOutputTile, kProjectionInput * kOutputTile);
        c.SetGlobalBuffer(accumulators + tile * capacity * kOutputTile, capacity * kOutputTile);
        mm.SetOrgShape(proj_rows, kOutputTile, kProjectionInput);
        mm.SetSingleShape(proj_rows, kOutputTile, kProjectionInput);
        mm.SetTensorA(a);
        mm.SetTensorB(b, false);
        mm.IterateAll(c);
    }
    mm.End();
}

#if defined(K11140_DIRECT_MLA_TAIL) && defined(K11040_TAIL_NZ)
#ifdef K112168_MLA_TAIL_OVERLAP
struct DirectOutPrefetchState {
    TBuf<TPosition::B1> b1_buf0, b1_buf1;
#if defined(K112186_MLA_TAIL_FAST) || defined(K112724_TAIL_OUT_3BUF)
    TBuf<TPosition::B1> b1_buf2;
#endif
    TBuf<TPosition::A1> a1_buf;
    TBuf<TPosition::A2> l0a_buf;
    TBuf<TPosition::B2> l0b_buf0, l0b_buf1;
    TBuf<TPosition::CO1> l0c_buf;
    LocalTensor<int8_t> a1;
    LocalTensor<int8_t> b1[3];
    LocalTensor<int8_t> l0a;
    LocalTensor<int8_t> l0b[2];
    LocalTensor<int32_t> l0c;
    GlobalTensor<int8_t> b_gm;
    bool active = false;
};

__aicore__ inline void PrefetchDirectProjectOutput24Aic(
    TPipe& pipe, DirectOutPrefetchState& st, __gm__ int8_t* weights, int64_t proj_rows)
{
    const uint32_t cid = static_cast<uint32_t>(GetBlockIdx());
    if (cid >= 24U) {
        return;
    }
    const bool dual_m = (proj_rows > 16);
    const uint32_t m_cap = dual_m ? 32U : 16U;
    pipe.InitBuffer(st.b1_buf0, 512 * 256);
    pipe.InitBuffer(st.b1_buf1, 512 * 256);
#if defined(K112186_MLA_TAIL_FAST) || defined(K112724_TAIL_OUT_3BUF)
    pipe.InitBuffer(st.b1_buf2, 512 * 256);
#endif
    pipe.InitBuffer(st.a1_buf, m_cap * 2048U);
    pipe.InitBuffer(st.l0a_buf, m_cap * 2048U);
    pipe.InitBuffer(st.l0b_buf0, 128 * 256);
    pipe.InitBuffer(st.l0b_buf1, 128 * 256);
    pipe.InitBuffer(st.l0c_buf, m_cap * 256U * sizeof(int32_t));
    st.a1 = st.a1_buf.Get<int8_t>();
    st.b1[0] = st.b1_buf0.Get<int8_t>();
    st.b1[1] = st.b1_buf1.Get<int8_t>();
#if defined(K112186_MLA_TAIL_FAST) || defined(K112724_TAIL_OUT_3BUF)
    st.b1[2] = st.b1_buf2.Get<int8_t>();
#endif
    st.l0a = st.l0a_buf.Get<int8_t>();
    st.l0b[0] = st.l0b_buf0.Get<int8_t>();
    st.l0b[1] = st.l0b_buf1.Get<int8_t>();
    st.l0c = st.l0c_buf.Get<int32_t>();
    st.b_gm.SetGlobalBuffer(weights + static_cast<uint64_t>(cid) * 524288ULL, 524288ULL);
    DataCopy(st.b1[0], st.b_gm[0], DataCopyParams{8, 512, 1536, 0});
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    DataCopy(st.b1[1], st.b_gm[16384U], DataCopyParams{8, 512, 1536, 0});
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
#ifdef K112724_TAIL_OUT_3BUF
    DataCopy(st.b1[2], st.b_gm[2U * 16384U], DataCopyParams{8, 512, 1536, 0});
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID5);
#endif
    st.active = true;
}

__aicore__ inline void RunPrefetchedDirectProjectOutput24Aic(
    DirectOutPrefetchState& st, __gm__ int8_t* quantized,
    __gm__ int32_t* accumulators, int64_t capacity, int64_t proj_rows)
{
    if (!st.active) {
        return;
    }
    const uint32_t cid = static_cast<uint32_t>(GetBlockIdx());
    const bool dual_m = (proj_rows > 16);
    const uint16_t m0 = dual_m ? 16U : static_cast<uint16_t>(proj_rows);
    const uint16_t m1 = dual_m ? static_cast<uint16_t>(proj_rows - 16) : 0U;
    GlobalTensor<int8_t> a_gm;
    GlobalTensor<int32_t> c_gm;
    a_gm.SetGlobalBuffer(quantized, static_cast<uint64_t>(capacity) * 2048ULL);
    c_gm.SetGlobalBuffer(accumulators + static_cast<uint64_t>(cid) * static_cast<uint64_t>(capacity) * 256ULL,
                         static_cast<uint64_t>(capacity) * 256ULL);
    Nd2NzParams copy_a{};
    copy_a.ndNum = 1;
    copy_a.nValue = m0;
    copy_a.dValue = 2048;
    copy_a.srcDValue = 2048;
    copy_a.dstNzC0Stride = 16;
    copy_a.dstNzNStride = 1;
#ifdef K112510_MLA_TAIL_XCORE_V2
    CrossCoreWaitFlag(9);
#endif
    DataCopy(st.a1, a_gm, copy_a);
    if (dual_m) {
        copy_a.nValue = m1;
        DataCopy(st.a1[32768U], a_gm[32768U], copy_a);
    }
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
#ifdef K112186_MLA_TAIL_FAST
    DataCopy(st.b1[2], st.b_gm[2U * 16384U], DataCopyParams{8, 512, 1536, 0});
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID5);
#endif
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
    LoadData2DParams load_a{};
    load_a.repeatTimes = dual_m ? 128 : 64;
    load_a.srcStride = 1;
    load_a.ifTranspose = false;
    LoadData(st.l0a, st.a1, load_a);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
    LoadData2dTransposeParams trans(0, 8, 16, 1, 0, inc);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
    for (uint32_t kc = 0U; kc < 4U; ++kc) {
#if defined(K112186_MLA_TAIL_FAST) || defined(K112724_TAIL_OUT_3BUF)
        const uint32_t bi = (kc == 2U) ? 2U : ((kc == 1U) ? 1U : 0U);
        const event_t b_evt = (kc == 2U) ? EVENT_ID5 : ((kc == 1U) ? EVENT_ID1 : EVENT_ID0);
#else
        const uint32_t bi = kc & 1U;
        const event_t b_evt = bi ? EVENT_ID1 : EVENT_ID0;
        if (kc >= 1U && kc + 1U < 4U) {
            const uint32_t nbi = (kc + 1U) & 1U;
            const event_t nb_evt = nbi ? EVENT_ID1 : EVENT_ID0;
            WaitFlag<HardEvent::MTE1_MTE2>(nb_evt);
            DataCopy(st.b1[nbi], st.b_gm[(kc + 1U) * 16384U], DataCopyParams{8, 512, 1536, 0});
            SetFlag<HardEvent::MTE2_MTE1>(nb_evt);
        }
#endif
        WaitFlag<HardEvent::MTE2_MTE1>(b_evt);
        for (uint32_t sub = 0U; sub < 4U; ++sub) {
            const uint32_t step = kc * 4U + sub;
            const uint32_t lbi = step & 1U;
            const event_t lb_evt = lbi ? EVENT_ID4 : EVENT_ID3;
            WaitFlag<HardEvent::M_MTE1>(lb_evt);
            const uint32_t k_sub_off = sub * 4096U;
            for (uint32_t i = 0U; i < 4U; ++i) {
                LoadDataWithTranspose(st.l0b[lbi][i * 8192U], st.b1[bi][k_sub_off + i * 1024U], trans);
            }
#if defined(K112186_MLA_TAIL_FAST) || defined(K112724_TAIL_OUT_3BUF)
            if (sub == 3U && kc == 0U) {
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
                DataCopy(st.b1[0], st.b_gm[3U * 16384U], DataCopyParams{8, 512, 1536, 0});
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
            }
#else
            if (sub == 3U && kc < 2U) {
                SetFlag<HardEvent::MTE1_MTE2>(b_evt);
            }
#endif
            SetFlag<HardEvent::MTE1_M>(lb_evt);
            WaitFlag<HardEvent::MTE1_M>(lb_evt);
            MmadParams mad{};
            mad.m = 16;
            mad.n = 256;
            mad.k = 128;
            mad.cmatrixSource = false;
            mad.cmatrixInitVal = (step == 0U);
            Mmad(st.l0c, st.l0a[step * 2048U], st.l0b[lbi], mad);
            if (dual_m) {
                Mmad(st.l0c[4096U], st.l0a[32768U + step * 2048U], st.l0b[lbi], mad);
            }
            if (step == 15U) {
                PipeBarrier<PIPE_M>();
            }
            SetFlag<HardEvent::M_MTE1>(lb_evt);
        }
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
    Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(c_gm, st.l0c, fix);
    if (dual_m) {
        fix.mSize = m1;
        Fixpipe<int32_t, int32_t, CFG_ROW_MAJOR>(c_gm[4096U], st.l0c[4096U], fix);
    }
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
#ifdef K112510_MLA_TAIL_XCORE_V2
    Publish();
    CrossCoreSetFlag<0x2, PIPE_FIX>(8);
#endif
}
#endif
__aicore__ inline void DirectProjectOutput24Aic(
    TPipe& pipe, __gm__ int8_t* quantized, __gm__ int8_t* weights,
    __gm__ int32_t* accumulators, int64_t capacity, int64_t proj_rows)
{
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
    a_gm.SetGlobalBuffer(quantized, static_cast<uint64_t>(capacity) * 2048ULL);
    b_gm.SetGlobalBuffer(weights + static_cast<uint64_t>(cid) * 524288ULL, 524288ULL);
    c_gm.SetGlobalBuffer(accumulators + static_cast<uint64_t>(cid) * static_cast<uint64_t>(capacity) * 256ULL,
                         static_cast<uint64_t>(capacity) * 256ULL);

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

    DataCopy(b1[0], b_gm[0], DataCopyParams{8, 512, 1536, 0});
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
    LoadData2DParams load_a{};
    load_a.repeatTimes = dual_m ? 128 : 64;
    load_a.srcStride = 1;
    load_a.ifTranspose = false;
    LoadData(l0a, a1, load_a);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);

    LoadData2dTransposeParams trans(0, 8, 16, 1, 0, inc);
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
            DataCopy(b1[nbi], b_gm[(kc + 1U) * 16384U], DataCopyParams{8, 512, 1536, 0});
            SetFlag<HardEvent::MTE2_MTE1>(nb_evt);
        }
        WaitFlag<HardEvent::MTE2_MTE1>(b_evt);
        for (uint32_t sub = 0U; sub < 4U; ++sub) {
            const uint32_t step = kc * 4U + sub;
            const uint32_t lbi = step & 1U;
            const event_t lb_evt = lbi ? EVENT_ID4 : EVENT_ID3;
            WaitFlag<HardEvent::M_MTE1>(lb_evt);
            const uint32_t k_sub_off = sub * 4096U;
            #pragma unroll
            for (uint32_t i = 0U; i < 4U; ++i) {
                LoadDataWithTranspose(l0b[lbi][i * 8192U], b1[bi][k_sub_off + i * 1024U], trans);
            }
            if (sub == 3U && kc < 2U) {
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
#endif

template <int64_t kHeads = mla_tail::kHeads>
__aicore__ inline void ProjectOutput(TPipe& pipe,
                                    __gm__ int8_t* quantized,
                                    __gm__ int8_t* weights,
                                    __gm__ int32_t* accumulators,
                                    __gm__ uint8_t* tiling,
                                    int64_t capacity, int64_t blocks,
                                    int64_t rows = 0)
{
    const int64_t active_rows = rows > 0 ? rows : capacity;
#if defined(K11140_DIRECT_MLA_TAIL) && defined(K11040_TAIL_NZ)
    if constexpr (kHeads == 8) {
#ifdef K11150_MLA_TAIL_M32
        if (active_rows <= 32 && blocks >= 24) {
#else
        if (active_rows <= 16 && blocks >= 24) {
#endif
            DirectProjectOutput24Aic(pipe, quantized, weights, accumulators, capacity, active_rows);
            return;
        }
    }
#endif
    if (active_rows <= 16) {
        RunProjectOutput<kHeads, kOutputTiling16>(pipe, quantized, weights, accumulators, capacity, blocks, active_rows);
    } else {
        RunProjectOutput<kHeads, kOutputTiling32>(pipe, quantized, weights, accumulators, capacity, blocks, active_rows);
    }
}

template <int64_t kHeads = mla_tail::kHeads>
__aicore__ inline void DequantizeOutput(
    VectorWorkspace<kHeads>& held, __gm__ int32_t* accumulators,
    __gm__ float* weight_scales, __gm__ float* token_scales,
    __gm__ uint8_t* bias, int bias_kind, __gm__ bfloat16_t* output,
    int64_t rows, int64_t capacity, int64_t vector_index, int64_t vectors)
{
    GlobalTensor<int32_t> source;
    GlobalTensor<float> weight_scale;
    GlobalTensor<float> token_scale;
    GlobalTensor<int32_t> integer_bias;
    GlobalTensor<float> floating_bias;
    GlobalTensor<bfloat16_t> destination;
    source.SetGlobalBuffer(accumulators, capacity * kHidden);
    weight_scale.SetGlobalBuffer(weight_scales, kHidden);
    token_scale.SetGlobalBuffer(token_scales, capacity * kScaleStride);
    integer_bias.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(bias));
    floating_bias.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));
    destination.SetGlobalBuffer(output, rows * kHidden);
    for (int64_t task = vector_index; task < rows * kOutputTiles;
         task += vectors) {
        const int64_t row = task / kOutputTiles;
        const int64_t tile = task % kOutputTiles;
        DataCopy(held.accumulator,
                 source[(tile * capacity + row) * kOutputTile], kOutputTile);
        DataCopy(held.other, weight_scale[tile * kOutputTile], kOutputTile);
        DataCopy(held.scalar, token_scale[row * kScaleStride], kScaleStride);
        if (bias_kind == 1) {
            DataCopy(held.bias, integer_bias[tile * kOutputTile], kOutputTile);
        }
        Fence<HardEvent::MTE2_V>();
        Fence<HardEvent::MTE2_S>();
        const float row_scale = held.scalar.GetValue(0);
        if (bias_kind == 1) {
            Add(held.accumulator, held.accumulator, held.bias, kOutputTile);
            PipeBarrier<PIPE_V>();
        }
        Cast(held.value, held.accumulator, RoundMode::CAST_RINT, kOutputTile);
        PipeBarrier<PIPE_V>();

        Mul(held.value, held.value, held.other, kOutputTile);
        PipeBarrier<PIPE_V>();
        Muls(held.value, held.value, row_scale, kOutputTile);
        PipeBarrier<PIPE_V>();
        if (bias_kind == 2) {
            Fence<HardEvent::V_MTE2>();
            DataCopy(held.other, floating_bias[tile * kOutputTile], kOutputTile);
            Fence<HardEvent::MTE2_V>();
            Add(held.value, held.value, held.other, kOutputTile);
            PipeBarrier<PIPE_V>();
        }
        Cast(held.bf16, held.value, RoundMode::CAST_RINT, kOutputTile);
        Fence<HardEvent::V_MTE3>();
        DataCopy(destination[row * kHidden + tile * kOutputTile], held.bf16,
                 kOutputTile);
        Fence<HardEvent::MTE3_V>();
        Fence<HardEvent::V_MTE2>();
    }
}

constexpr int64_t kDequantRows = 16;

struct OutputWorkspace {
    TBuf<TPosition::VECCALC> accumulator_buffer, value_buffer, output_buffer;
    TBuf<TPosition::VECCALC> weight_buffer, bias_buffer, scale_buffer;
    LocalTensor<int32_t> accumulators, bias;
    LocalTensor<float> values, weights, scales;
    LocalTensor<bfloat16_t> output;

    __aicore__ inline void Init(TPipe& pipe)
    {
        pipe.InitBuffer(accumulator_buffer, kDequantRows * kOutputTile * 4);
        pipe.InitBuffer(value_buffer, kDequantRows * kOutputTile * 4);
        pipe.InitBuffer(output_buffer, kDequantRows * kOutputTile * 2);
        pipe.InitBuffer(weight_buffer, kOutputTile * 4);
        pipe.InitBuffer(bias_buffer, kOutputTile * 4);
        pipe.InitBuffer(scale_buffer, kDequantRows * kScaleStride * 4);
        accumulators = accumulator_buffer.Get<int32_t>();
        values = value_buffer.Get<float>();
        output = output_buffer.Get<bfloat16_t>();
        weights = weight_buffer.Get<float>();
        bias = bias_buffer.Get<int32_t>();
        scales = scale_buffer.Get<float>();
    }
};

enum class ScaleOrder { ChannelThenToken, CombinedScales };

template <ScaleOrder order = ScaleOrder::CombinedScales, int64_t MaxRows = kDequantRows, bool ContiguousComm = false>
__aicore__ inline void FinishOutputRows(
    OutputWorkspace& held, LocalTensor<float>& combined,
    __gm__ int32_t* accumulators, __gm__ float* weight_scales,
    __gm__ float* token_scales, __gm__ uint8_t* bias, int bias_kind,
    __gm__ bfloat16_t* output, int64_t rows, int64_t capacity,
    int64_t vector_index, int64_t vectors,
    __gm__ bfloat16_t* shmem_dst_base = nullptr, int32_t comm_rank = 0, int32_t comm_ranks = 0,
    bool sip_relay = false)
{
    const int64_t batch_rows = rows <= 1 ? 1
        : (rows <= 2 * MaxRows ? ((rows + 1) >> 1) : MaxRows);
    const int64_t batches = (rows <= batch_rows) ? 1
        : ((rows <= 2 * batch_rows) ? 2 : ((rows + 7) >> 3));
    if (vector_index >= batches * kOutputTiles) {
        return;
    }
    GlobalTensor<int32_t> source, integer_bias;
    GlobalTensor<float> weight_scale, token_scale, floating_bias;
    GlobalTensor<bfloat16_t> destination;
    source.SetGlobalBuffer(accumulators, capacity * kHidden);
    weight_scale.SetGlobalBuffer(weight_scales, kHidden);
    token_scale.SetGlobalBuffer(token_scales, capacity * kScaleStride);
    integer_bias.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(bias));
    floating_bias.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));
    destination.SetGlobalBuffer(output, rows * kHidden);

    for (int64_t task = vector_index; task < batches * kOutputTiles;
         task += vectors) {
        const uint32_t utask = static_cast<uint32_t>(task);
        const uint32_t q = (utask * 683U) >> 14U;
        const int64_t tile = static_cast<int64_t>(utask - q * 24U);
        const int64_t begin = static_cast<int64_t>(q) * batch_rows;
        const int64_t count = rows <= begin ? 0
            : (rows - begin < batch_rows ? rows - begin : batch_rows);
        if (count <= 0) {
            continue;
        }
        const int64_t total = count * kOutputTile;
        DataCopy(held.accumulators, source[(tile * capacity + begin) * kOutputTile],
                  total);
        DataCopy(held.weights, weight_scale[tile * kOutputTile], kOutputTile);
        DataCopy(held.scales, token_scale[begin * kScaleStride],
                  count * kScaleStride);
        if (bias_kind == 1) {
            DataCopy(held.bias, integer_bias[tile * kOutputTile], kOutputTile);
        } else if (bias_kind == 2) {
            DataCopy(held.bias.ReinterpretCast<float>(),
                      floating_bias[tile * kOutputTile], kOutputTile);
        }
        Fence<HardEvent::MTE2_V>();
        Fence<HardEvent::MTE2_S>();

        if (bias_kind == 1) {
            for (int64_t row = 0; row < count; ++row) {
                Add(held.accumulators[row * kOutputTile],
                    held.accumulators[row * kOutputTile], held.bias, kOutputTile);
            }
            PipeBarrier<PIPE_V>();
        }
        Cast(held.values, held.accumulators, RoundMode::CAST_RINT, total);
        PipeBarrier<PIPE_V>();
        if constexpr (order == ScaleOrder::CombinedScales) {
            for (int64_t row = 0; row < count; ++row) {
                Muls(combined[row * kOutputTile], held.weights,
                     held.scales.GetValue(row * kScaleStride), kOutputTile);
            }
            PipeBarrier<PIPE_V>();
            Mul(held.values, held.values, combined, total);
            PipeBarrier<PIPE_V>();
        } else {
            for (int64_t row = 0; row < count; ++row) {
                Mul(held.values[row * kOutputTile],
                    held.values[row * kOutputTile], held.weights, kOutputTile);
            }
            PipeBarrier<PIPE_V>();
            for (int64_t row = 0; row < count; ++row) {
                Muls(held.values[row * kOutputTile],
                     held.values[row * kOutputTile],
                     held.scales.GetValue(row * kScaleStride), kOutputTile);
            }
            PipeBarrier<PIPE_V>();
        }
        if (bias_kind == 2) {
            for (int64_t row = 0; row < count; ++row) {
                Add(held.values[row * kOutputTile],
                    held.values[row * kOutputTile],
                    held.bias.ReinterpretCast<float>(), kOutputTile);
            }
            PipeBarrier<PIPE_V>();
        }
        Cast(held.output, held.values, RoundMode::CAST_RINT, total);
        Fence<HardEvent::V_MTE3>();
        if constexpr (ContiguousComm) {
            if (shmem_dst_base != nullptr) {
                const int64_t pkt_off = begin * kHidden + tile * total;
#ifdef TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H
                if (sip_relay) {
                    GlobalTensor<bfloat16_t> self_sh;
                    self_sh.SetGlobalBuffer(shmem_dst_base + pkt_off, total);
                    DataCopy(self_sh, held.output, total);
                    for (uint32_t p = 0U; p < 3U; ++p) {
                        const uint32_t step2 = (((static_cast<uint32_t>(vector_index) + p) % 3U) + 1U) << 1U;
                        const int32_t peer = (comm_rank + static_cast<int32_t>(step2)) & 7;
                        __gm__ bfloat16_t* peer_ptr = reinterpret_cast<__gm__ bfloat16_t*>(
                            aclshmem_ptr(shmem_dst_base + pkt_off, peer));
                        GlobalTensor<bfloat16_t> peer_dst;
                        peer_dst.SetGlobalBuffer(peer_ptr, total);
                        DataCopy(peer_dst, held.output, total);
                    }
                } else {
                    DataCopy(destination[pkt_off], held.output, total);
                    for (uint32_t p = 0U; p < 7U; ++p) {
                        const uint32_t off = ((static_cast<uint32_t>(vector_index) + p) % 7U) + 1U;
                        const int32_t peer = (comm_rank + static_cast<int32_t>(off)) & 7;
                        __gm__ bfloat16_t* peer_ptr = reinterpret_cast<__gm__ bfloat16_t*>(
                            aclshmem_ptr(shmem_dst_base + pkt_off, peer));
                        GlobalTensor<bfloat16_t> peer_dst;
                        peer_dst.SetGlobalBuffer(peer_ptr, total);
                        DataCopy(peer_dst, held.output, total);
                    }
                }
#else
                DataCopy(destination[pkt_off], held.output, total);
#endif
            } else {
                DataCopyParams copy;
                copy.blockCount = static_cast<uint16_t>(count);
                copy.blockLen = kOutputTile * sizeof(bfloat16_t) / 32;
                copy.srcStride = 0;
                copy.dstStride = (kHidden - kOutputTile) * sizeof(bfloat16_t) / 32;
                DataCopy(destination[begin * kHidden + tile * kOutputTile],
                          held.output, copy);
            }
        } else {
            DataCopyParams copy;
            copy.blockCount = static_cast<uint16_t>(count);
            copy.blockLen = kOutputTile * sizeof(bfloat16_t) / 32;
            copy.srcStride = 0;
            copy.dstStride = (kHidden - kOutputTile) * sizeof(bfloat16_t) / 32;
#ifdef TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H
            if (shmem_dst_base != nullptr && sip_relay) {
                const int64_t dst_off = begin * kHidden + tile * kOutputTile;
                GlobalTensor<bfloat16_t> self_sh;
                self_sh.SetGlobalBuffer(shmem_dst_base + dst_off, rows * kHidden);
                DataCopy(self_sh, held.output, copy);
                for (uint32_t p = 0U; p < 3U; ++p) {
                    const uint32_t step2 = (((static_cast<uint32_t>(vector_index) + p) % 3U) + 1U) << 1U;
                    const int32_t peer = (comm_rank + static_cast<int32_t>(step2)) & 7;
                    __gm__ bfloat16_t* peer_ptr = reinterpret_cast<__gm__ bfloat16_t*>(
                        aclshmem_ptr(shmem_dst_base + dst_off, peer));
                    GlobalTensor<bfloat16_t> peer_dst;
                    peer_dst.SetGlobalBuffer(peer_ptr, rows * kHidden);
                    DataCopy(peer_dst, held.output, copy);
                }
            } else {
                DataCopy(destination[begin * kHidden + tile * kOutputTile],
                          held.output, copy);
                if (shmem_dst_base != nullptr) {
                    const int64_t dst_off = begin * kHidden + tile * kOutputTile;
                    for (uint32_t p = 0U; p < 7U; ++p) {
                        const uint32_t off = ((static_cast<uint32_t>(vector_index) + p) % 7U) + 1U;
                        const int32_t peer = (comm_rank + static_cast<int32_t>(off)) & 7;
                        __gm__ bfloat16_t* peer_ptr = reinterpret_cast<__gm__ bfloat16_t*>(
                            aclshmem_ptr(shmem_dst_base + dst_off, peer));
                        GlobalTensor<bfloat16_t> peer_dst;
                        peer_dst.SetGlobalBuffer(peer_ptr, rows * kHidden);
                        DataCopy(peer_dst, held.output, copy);
                    }
                }
            }
#else
            DataCopy(destination[begin * kHidden + tile * kOutputTile],
                      held.output, copy);
#endif
        }
        if (task + vectors < batches * kOutputTiles) {
            Fence<HardEvent::MTE3_MTE2>();
            Fence<HardEvent::MTE3_V>();
        }
    }
}

template <ScaleOrder order = ScaleOrder::CombinedScales>
__aicore__ inline void DequantizeOutputRows(
    TPipe& pipe, __gm__ int32_t* accumulators, __gm__ float* weight_scales,
    __gm__ float* token_scales, __gm__ uint8_t* bias, int bias_kind,
    __gm__ bfloat16_t* output, int64_t rows, int64_t capacity,
    int64_t vector_index, int64_t vectors,
    __gm__ bfloat16_t* shmem_dst_base = nullptr, int32_t comm_rank = 0, int32_t comm_ranks = 0)
{
    OutputWorkspace held;
    TBuf<TPosition::VECCALC> combined_buffer;
    LocalTensor<float> combined;
    held.Init(pipe);
    if constexpr (order == ScaleOrder::CombinedScales) {
        pipe.InitBuffer(combined_buffer,
                        kDequantRows * kOutputTile * sizeof(float));
        combined = combined_buffer.Get<float>();
    }
    FinishOutputRows<order>(held, combined, accumulators, weight_scales,
                            token_scales, bias, bias_kind, output, rows,
                            capacity, vector_index, vectors,
                            shmem_dst_base, comm_rank, comm_ranks);
}


#ifdef K112168_MLA_TAIL_OVERLAP
struct PreloadedDeqTask {
    int64_t tile = 0;
    int64_t begin = 0;
    int64_t count = 0;
    int64_t total = 0;
    bool valid = false;
};

__aicore__ inline PreloadedDeqTask PreloadOutputRowScales(
    OutputWorkspace& held, LocalTensor<float>& combined,
    __gm__ float* weight_scales, __gm__ float* token_scales,
    __gm__ uint8_t* bias, int bias_kind, int64_t rows, int64_t capacity,
    int64_t vector_index)
{
    PreloadedDeqTask t{};
    const int64_t batch_rows = (rows <= 1) ? 1 : ((rows + 1) >> 1);
    const int64_t batches = (rows <= batch_rows) ? 1 : 2;
#ifdef K112510_MLA_TAIL_XCORE_V2
    if (vector_index >= 2 * kOutputTiles) {
        return t;
    }
    const uint32_t utask = static_cast<uint32_t>(vector_index);
    const uint32_t q = utask & 1U;
    if (static_cast<int64_t>(q) >= batches) {
        return t;
    }
    t.tile = static_cast<int64_t>(utask >> 1U);
#else
    if (vector_index >= batches * kOutputTiles) {
        return t;
    }
    const uint32_t utask = static_cast<uint32_t>(vector_index);
    const uint32_t q = (utask * 683U) >> 14U;
    t.tile = static_cast<int64_t>(utask - q * 24U);
#endif
    t.begin = static_cast<int64_t>(q) * batch_rows;
    t.count = (rows <= t.begin) ? 0 : ((rows - t.begin < batch_rows) ? (rows - t.begin) : batch_rows);
    if (t.count <= 0) {
        return t;
    }
    t.total = t.count * kOutputTile;
    t.valid = true;
    GlobalTensor<float> weight_scale, token_scale, floating_bias;
    GlobalTensor<int32_t> integer_bias;
    weight_scale.SetGlobalBuffer(weight_scales, kHidden);
    token_scale.SetGlobalBuffer(token_scales, capacity * kScaleStride);
    integer_bias.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(bias));
    floating_bias.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));
    DataCopy(held.weights, weight_scale[t.tile * kOutputTile], kOutputTile);
    DataCopy(held.scales, token_scale[t.begin * kScaleStride], t.count * kScaleStride);
    if (bias_kind == 1) {
        DataCopy(held.bias, integer_bias[t.tile * kOutputTile], kOutputTile);
    } else if (bias_kind == 2) {
        DataCopy(held.bias.ReinterpretCast<float>(), floating_bias[t.tile * kOutputTile], kOutputTile);
    }
    Fence<HardEvent::MTE2_V>();
    Fence<HardEvent::MTE2_S>();
    for (int64_t row = 0; row < t.count; ++row) {
        Muls(combined[row * kOutputTile], held.weights,
             held.scales.GetValue(row * kScaleStride), kOutputTile);
    }
    PipeBarrier<PIPE_V>();
    return t;
}

__aicore__ inline void FinishPreloadedOutputRow(
    const PreloadedDeqTask& t, OutputWorkspace& held, LocalTensor<float>& combined,
    __gm__ int32_t* accumulators, int bias_kind, __gm__ bfloat16_t* output,
    int64_t rows, int64_t capacity, int64_t vector_index,
    __gm__ bfloat16_t* shmem_dst_base, int32_t comm_rank, bool sip_relay)
{
    if (!t.valid) {
        return;
    }
    GlobalTensor<int32_t> source;
    GlobalTensor<bfloat16_t> destination;
    source.SetGlobalBuffer(accumulators, capacity * kHidden);
    destination.SetGlobalBuffer(output, rows * kHidden);
    DataCopy(held.accumulators, source[(t.tile * capacity + t.begin) * kOutputTile], t.total);
    Fence<HardEvent::MTE2_V>();
    if (bias_kind == 1) {
        for (int64_t row = 0; row < t.count; ++row) {
            Add(held.accumulators[row * kOutputTile],
                held.accumulators[row * kOutputTile], held.bias, kOutputTile);
        }
        PipeBarrier<PIPE_V>();
    }
    Cast(held.values, held.accumulators, RoundMode::CAST_RINT, t.total);
    PipeBarrier<PIPE_V>();
    Mul(held.values, held.values, combined, t.total);
    PipeBarrier<PIPE_V>();
    if (bias_kind == 2) {
        for (int64_t row = 0; row < t.count; ++row) {
            Add(held.values[row * kOutputTile],
                held.values[row * kOutputTile],
                held.bias.ReinterpretCast<float>(), kOutputTile);
        }
        PipeBarrier<PIPE_V>();
    }
    Cast(held.output, held.values, RoundMode::CAST_RINT, t.total);
    Fence<HardEvent::V_MTE3>();
    if (shmem_dst_base != nullptr) {
        const int64_t pkt_off = t.begin * kHidden + t.tile * t.total;
#ifdef TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H
        if (sip_relay) {
            GlobalTensor<bfloat16_t> self_sh;
            self_sh.SetGlobalBuffer(shmem_dst_base + pkt_off, t.total);
            DataCopy(self_sh, held.output, t.total);
            for (uint32_t p = 0U; p < 3U; ++p) {
                const uint32_t step2 = (((static_cast<uint32_t>(vector_index) + p) % 3U) + 1U) << 1U;
                const int32_t peer = (comm_rank + static_cast<int32_t>(step2)) & 7;
                __gm__ bfloat16_t* peer_ptr = reinterpret_cast<__gm__ bfloat16_t*>(
                    aclshmem_ptr(shmem_dst_base + pkt_off, peer));
                GlobalTensor<bfloat16_t> peer_dst;
                peer_dst.SetGlobalBuffer(peer_ptr, t.total);
                DataCopy(peer_dst, held.output, t.total);
            }
        } else {
            DataCopy(destination[pkt_off], held.output, t.total);
            for (uint32_t p = 0U; p < 7U; ++p) {
                const uint32_t off = ((static_cast<uint32_t>(vector_index) + p) % 7U) + 1U;
                const int32_t peer = (comm_rank + static_cast<int32_t>(off)) & 7;
                __gm__ bfloat16_t* peer_ptr = reinterpret_cast<__gm__ bfloat16_t*>(
                    aclshmem_ptr(shmem_dst_base + pkt_off, peer));
                GlobalTensor<bfloat16_t> peer_dst;
                peer_dst.SetGlobalBuffer(peer_ptr, t.total);
                DataCopy(peer_dst, held.output, t.total);
            }
        }
#else
        DataCopy(destination[pkt_off], held.output, t.total);
#endif
    } else {
        DataCopyParams copy;
        copy.blockCount = static_cast<uint16_t>(t.count);
        copy.blockLen = kOutputTile * sizeof(bfloat16_t) / 32;
        copy.srcStride = 0;
        copy.dstStride = (kHidden - kOutputTile) * sizeof(bfloat16_t) / 32;
        DataCopy(destination[t.begin * kHidden + t.tile * kOutputTile], held.output, copy);
    }
}
#endif

template <int64_t kHeads = mla_tail::kHeads>
__aicore__ inline uint32_t Forward(
    TPipe& pipe, __gm__ bfloat16_t* latent,
    __gm__ bfloat16_t* value_weight, __gm__ int8_t* output_weight,
    __gm__ float* output_scale, __gm__ uint8_t* bias, int bias_kind,
    __gm__ uint8_t* scratch, __gm__ uint8_t* tiling,
    __gm__ bfloat16_t* output, int64_t rows, int64_t capacity,
    int64_t blocks, const WorkspaceLayout& layout, bool publish_end = true,
    __gm__ uint8_t* comm_window = nullptr, int64_t comm_capacity = 0, int32_t comm_rank = 0, int32_t comm_ranks = 0)
{
#if defined(K11140_DIRECT_MLA_TAIL) && defined(K11040_TAIL_NZ)
#ifdef K11150_MLA_TAIL_M32
    if (!(kHeads == 8 && (rows > 0 ? rows : capacity) <= 32 && blocks >= 24)) {
#else
    if (!(kHeads == 8 && (rows > 0 ? rows : capacity) <= 16 && blocks >= 24)) {
#endif
        SetSysWorkspace(scratch + layout.system);
    }
#else
    SetSysWorkspace(scratch + layout.system);
#endif
    const uint64_t values_offset = layout.values;
    const uint64_t quantized_offset = layout.quantized;
    const uint64_t scales_offset = layout.token_scales;
    const uint64_t accumulators_offset = layout.accumulators;
    auto* values = reinterpret_cast<__gm__ float*>(scratch + values_offset);
    auto* quantized = reinterpret_cast<__gm__ int8_t*>(scratch + quantized_offset);
    auto* scales = reinterpret_cast<__gm__ float*>(scratch + scales_offset);
    auto* accumulators =
        reinterpret_cast<__gm__ int32_t*>(scratch + accumulators_offset);
    const int64_t active_rows = rows > 0 ? rows : capacity;
    VectorWorkspace<kHeads> held;
    if ASCEND_IS_AIC {
        ProjectValues<kHeads>(pipe, latent, value_weight, values, tiling, capacity, blocks, rows);
        Publish();
    }
    uint32_t red1_epoch = 0U;
    if ASCEND_IS_AIV {
#ifdef TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H
        if (comm_window != nullptr) {
            red1_epoch = whole_layer_tree::ReadCommunicationEpoch(comm_window) + 1U;
        }
#endif
        if (GetBlockIdx() < active_rows) {
            held.Init(pipe);
        }
    }
    SyncAll<false>();
#if defined(K112168_MLA_TAIL_OVERLAP) && defined(K11140_DIRECT_MLA_TAIL) && defined(K11040_TAIL_NZ) && defined(K11150_MLA_TAIL_M32)
    DirectOutPrefetchState out_pf;
    const bool use_pf_out = (kHeads == 8) && (active_rows >= 1) && (active_rows <= 32) && (blocks == 24);
#endif
#if defined(K112510_MLA_TAIL_XCORE_V2) && defined(K112168_MLA_TAIL_OVERLAP) && defined(K11140_DIRECT_MLA_TAIL) && defined(K11040_TAIL_NZ) && defined(K11150_MLA_TAIL_M32)
    if (use_pf_out) {
        if ASCEND_IS_AIC {
            pipe.Reset();
            PrefetchDirectProjectOutput24Aic(pipe, out_pf, output_weight, active_rows);
            RunPrefetchedDirectProjectOutput24Aic(out_pf, quantized, accumulators, capacity, active_rows);
            pipe.Reset();
        }
        if ASCEND_IS_AIV {
            if (GetBlockIdx() < active_rows) {
                QuantizeValues<kHeads>(held, values, quantized, scales, capacity,
                                       GetBlockIdx(), 2 * blocks, rows);
                Publish();
                pipe.Reset();
            }
            SyncAll<true>();
            if (GetBlockIdx() < 48) {
                CrossCoreSetFlag<0x2, PIPE_MTE3>(9);
            }
            __gm__ bfloat16_t* xc_dst_base = nullptr;
            bool xc_quad4 = false;
#ifdef TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H
            if (comm_window != nullptr) {
                if (GetBlockIdx() == 0) {
                    GlobalTensor<uint32_t> counter;
                    counter.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(comm_window), 1);
                    counter.SetValue(0, red1_epoch);
                    DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE,
                                             DcciDst::CACHELINE_OUT>(counter);
                }
                xc_quad4 = whole_layer_tree::IsQuad4Active(comm_capacity, active_rows);
                const auto comm_layout = xc_quad4
                    ? whole_layer_tree::MakeQuad4Layout(comm_capacity, comm_ranks, active_rows)
                    : whole_layer_tree::MakeCommunicationLayout(comm_capacity, comm_ranks);
                const uint32_t p_mask = xc_quad4 ? 3U : 1U;
                xc_dst_base = reinterpret_cast<__gm__ bfloat16_t*>(
                    comm_window + comm_layout.payload + (red1_epoch & p_mask) * comm_layout.plane_bytes
                    + comm_rank * comm_layout.rank_bytes);
            }
#endif
            const bool sip_r1_xc =
#ifdef K112110_SIP_RED1
#if defined(K112170_DEC_SIP_ATT) && defined(K112140_SIP_RED1_MIN_ROWS)
                (xc_dst_base != nullptr) && (active_rows >= static_cast<int64_t>(K112140_SIP_RED1_MIN_ROWS));
#elif defined(K112140_SIP_RED1_MIN_ROWS)
                (xc_dst_base != nullptr) && (xc_quad4 || active_rows >= static_cast<int64_t>(K112140_SIP_RED1_MIN_ROWS));
#else
                (xc_dst_base != nullptr);
#endif
#else
                xc_quad4;
#endif
            OutputWorkspace deq_held_xc;
            TBuf<TPosition::VECCALC> combined_buf_xc;
            LocalTensor<float> combined_xc;
            PreloadedDeqTask pre_xc{};
            if (GetBlockIdx() < 48) {
                deq_held_xc.Init(pipe);
                pipe.InitBuffer(combined_buf_xc, kDequantRows * kOutputTile * sizeof(float));
                combined_xc = combined_buf_xc.Get<float>();
                pre_xc = PreloadOutputRowScales(
                    deq_held_xc, combined_xc, output_scale, scales, bias, bias_kind,
                    active_rows, capacity, GetBlockIdx());
                CrossCoreWaitFlag(8);
                FinishPreloadedOutputRow(
                    pre_xc, deq_held_xc, combined_xc, accumulators, bias_kind,
                    output, active_rows, capacity, GetBlockIdx(),
                    xc_dst_base, comm_rank, sip_r1_xc);
            }
            if (publish_end) {
                Publish();
            }
        }
        return red1_epoch;
    }
#endif
    if ASCEND_IS_AIC {
        pipe.Reset();
#if defined(K112168_MLA_TAIL_OVERLAP) && defined(K11140_DIRECT_MLA_TAIL) && defined(K11040_TAIL_NZ) && defined(K11150_MLA_TAIL_M32)
        if (use_pf_out) {
            PrefetchDirectProjectOutput24Aic(pipe, out_pf, output_weight, active_rows);
        }
#endif
    }
    if ASCEND_IS_AIV {
        if (GetBlockIdx() < active_rows) {
            QuantizeValues<kHeads>(held, values, quantized, scales, capacity,
                                   GetBlockIdx(), 2 * blocks, rows);
            Publish();
            pipe.Reset();
        }
    }
    SyncAll<false>();
    if ASCEND_IS_AIC {
#if defined(K112168_MLA_TAIL_OVERLAP) && defined(K11140_DIRECT_MLA_TAIL) && defined(K11040_TAIL_NZ) && defined(K11150_MLA_TAIL_M32)
        if (use_pf_out) {
            RunPrefetchedDirectProjectOutput24Aic(out_pf, quantized, accumulators, capacity, active_rows);
        } else
#endif
        {
            ProjectOutput<kHeads>(pipe, quantized, output_weight, accumulators,
                                  tiling + kCubeTilingBytes, capacity, blocks, rows);
        }
        Publish();
    }
    __gm__ bfloat16_t* red1_dst_base = nullptr;
    bool red1_quad4 = false;
    OutputWorkspace deq_held;
    TBuf<TPosition::VECCALC> combined_buffer;
    LocalTensor<float> combined;
    const int64_t deq_active_cores = (active_rows <= 1) ? kOutputTiles : (2 * kOutputTiles);
#ifdef K112168_MLA_TAIL_OVERLAP
    PreloadedDeqTask pre_deq{};
    const bool use_pre_deq = (active_rows >= 1) && (active_rows <= 32) && (blocks == 24);
#endif
    if ASCEND_IS_AIV {
#ifdef TESTS_GLM52_COMMUNICATION_TREE_COMMUNICATION_DEVICE_H
        if (comm_window != nullptr) {
            if (GetBlockIdx() == 0) {
                GlobalTensor<uint32_t> counter;
                counter.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(comm_window), 1);
                counter.SetValue(0, red1_epoch);
                DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE,
                                         DcciDst::CACHELINE_OUT>(counter);
            }
            red1_quad4 = whole_layer_tree::IsQuad4Active(comm_capacity, active_rows);
            const auto comm_layout = red1_quad4
                ? whole_layer_tree::MakeQuad4Layout(comm_capacity, comm_ranks, active_rows)
                : whole_layer_tree::MakeCommunicationLayout(comm_capacity, comm_ranks);
            const uint32_t p_mask = red1_quad4 ? 3U : 1U;
            red1_dst_base = reinterpret_cast<__gm__ bfloat16_t*>(
                comm_window + comm_layout.payload + (red1_epoch & p_mask) * comm_layout.plane_bytes
                + comm_rank * comm_layout.rank_bytes);
        }
#endif
        if (GetBlockIdx() < deq_active_cores) {
            deq_held.Init(pipe);
            pipe.InitBuffer(combined_buffer, kDequantRows * kOutputTile * sizeof(float));
            combined = combined_buffer.Get<float>();
#ifdef K112168_MLA_TAIL_OVERLAP
            if (use_pre_deq) {
                pre_deq = PreloadOutputRowScales(
                    deq_held, combined, output_scale, scales, bias, bias_kind,
                    active_rows, capacity, GetBlockIdx());
            }
#endif
        }
    }
    SyncAll<false>();
    if ASCEND_IS_AIC {
        pipe.Reset();
    }
    if ASCEND_IS_AIV {
        const bool sip_r1 =
#ifdef K112110_SIP_RED1
#if defined(K112170_DEC_SIP_ATT) && defined(K112140_SIP_RED1_MIN_ROWS)
            (red1_dst_base != nullptr) && (active_rows >= static_cast<int64_t>(K112140_SIP_RED1_MIN_ROWS));
#elif defined(K112140_SIP_RED1_MIN_ROWS)
            (red1_dst_base != nullptr) && (red1_quad4 || active_rows >= static_cast<int64_t>(K112140_SIP_RED1_MIN_ROWS));
#else
            (red1_dst_base != nullptr);
#endif
#else
            red1_quad4;
#endif
        if (GetBlockIdx() < deq_active_cores) {
#ifdef K112168_MLA_TAIL_OVERLAP
            if (use_pre_deq) {
                FinishPreloadedOutputRow(
                    pre_deq, deq_held, combined, accumulators, bias_kind,
                    output, active_rows, capacity, GetBlockIdx(),
                    red1_dst_base, comm_rank, sip_r1);
            } else
#endif
            {
                FinishOutputRows<ScaleOrder::CombinedScales, kDequantRows, true>(
                    deq_held, combined, accumulators, output_scale, scales, bias, bias_kind,
                    output, active_rows, capacity, GetBlockIdx(), 2 * blocks,
                    red1_dst_base, comm_rank, comm_ranks, sip_r1);
            }
        }
        if (publish_end) {
            Publish();
        }
    }
    return red1_epoch;
}

}

#endif
