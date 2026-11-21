#ifndef TILERT_FFN_BRIDGE_NATIVE_H
#define TILERT_FFN_BRIDGE_NATIVE_H

#ifndef ASCENDC_CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "contract.h"
#include "../norm_native.h"


struct BridgeTopKTiling {
    int64_t perCoreRowCount = 1;
    int64_t lastCoreRowCount = 1;
    int64_t expertCount = ffn_bridge::kExperts;
    int64_t addBias = 1;
    int64_t outFlag = 0;
    int64_t k = ffn_bridge::kTopK;
    int64_t renorm = 1;
    int64_t normType = 1;
    float routedScalingFactor = 1.0F;
    float eps = 1.0e-20F;
};

#define MoeGatingTopKTilingData BridgeTopKTiling
#include "moe_gating_top_k_without_group.h"
#undef MoeGatingTopKTilingData

namespace ffn_bridge::device {
using namespace AscendC;
using namespace matmul;

template <HardEvent event>
__aicore__ inline void Fence()
{
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
}

__aicore__ inline void EndPhase(TPipe& pipe)
{
    PipeBarrier<PIPE_ALL>();
#ifndef K112722_NO_DDR_EXTRA
    DataSyncBarrier<MemDsbT::DDR>();
#endif
    SyncAll<false>();
    pipe.Reset();
}

using BF16 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using BF16_NZ = MatmulType<TPosition::GM, CubeFormat::NZ, bfloat16_t, false>;
using FP32 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using GateMatmul = Matmul<BF16, BF16, FP32, FP32>;

#ifdef K11070_TAIL_K256
template <typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class BridgeFastCubeInBuffer {
    using TransT = typename INPUT_TYPE::TRANS_T;
public:
    __aicore__ inline BridgeFastCubeInBuffer() {}
    __aicore__ inline ~BridgeFastCubeInBuffer() {}
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
struct BridgeFastMatmulPolicy : public AscendC::Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
    using TransT_B = decltype(GetTransBDataType<A_TYPE, B_TYPE, MM_CFG>());
    using CubeInBufferA = BridgeFastCubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = BridgeFastCubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename TransT_B::T>, MM_CFG>;
};
#endif

__aicore__ constexpr ::MatmulApiStaticTiling MakeBridgeGateTiling16()
{
    ::MatmulApiStaticTiling v{};
    v.N = kGateTile; v.Ka = kHidden; v.Kb = kHidden;
    v.singleCoreN = kGateTile; v.singleCoreK = kHidden;
    v.baseM = 16; v.baseN = 16; v.baseK = 1024;
#ifdef K11070_TAIL_K256
    v.depthA1 = 6; v.depthB1 = 6; v.stepM = 1; v.stepN = 1;
    v.stepKa = 6; v.stepKb = 6; v.iterateOrder = 0;
#else
    v.depthA1 = 6; v.depthB1 = 2; v.stepM = 1; v.stepN = 1;
    v.stepKa = 6; v.stepKb = 1; v.iterateOrder = 0;
#endif
    v.shareL1Size = 393216; v.shareL0CSize = 1024; v.shareUbSize = 0;
    v.dbL0A = 2; v.dbL0B = 2; v.dbL0C = 1;
    v.cfg = ::CFG_NORM;
    return v;
}

__aicore__ constexpr ::MatmulApiStaticTiling MakeBridgeGateTiling32()
{
    ::MatmulApiStaticTiling v = MakeBridgeGateTiling16();
    return v;
}

inline constexpr auto kBridgeGateTiling16 = MakeBridgeGateTiling16();
inline constexpr auto kBridgeGateTiling32 = MakeBridgeGateTiling32();
#ifdef K11070_TAIL_K256
using GateMatmul16 = Matmul<BF16, BF16_NZ, FP32, FP32, kBridgeGateTiling16, MatmulCallBackFunc<nullptr, nullptr, nullptr>, BridgeFastMatmulPolicy>;
#else
using GateMatmul16 = Matmul<BF16, BF16_NZ, FP32, FP32, kBridgeGateTiling16>;
#endif
using GateMatmul32 = Matmul<BF16, BF16_NZ, FP32, FP32, kBridgeGateTiling32>;

__aicore__ inline void DirectBridgeGate16Aic(
    TPipe& pipe, __gm__ bfloat16_t* input, __gm__ bfloat16_t* weight,
    __gm__ float* output, int64_t rows)
{
    if ASCEND_IS_AIC {
        const uint32_t core = static_cast<uint32_t>(GetBlockIdx());
        if (core >= 16U || rows <= 0) {
            return;
        }
        const bool dual_m = (rows > 16);
        const uint16_t m0 = dual_m ? 16U : static_cast<uint16_t>(rows);
        const uint16_t m1 = dual_m ? static_cast<uint16_t>(rows - 16) : 0U;

        TBuf<TPosition::A1> a1_buf;
        TBuf<TPosition::B1> b1_buf[2];
        TBuf<TPosition::A2> l0a_buf;
        TBuf<TPosition::B2> l0b_buf[2];
        TBuf<TPosition::CO1> l0c_buf;
        pipe.InitBuffer(a1_buf, 32 * 1024 * sizeof(bfloat16_t));
        pipe.InitBuffer(b1_buf[0], 16 * 1024 * sizeof(bfloat16_t));
        pipe.InitBuffer(b1_buf[1], 16 * 1024 * sizeof(bfloat16_t));
        pipe.InitBuffer(l0a_buf, 32 * 1024 * sizeof(bfloat16_t));
        pipe.InitBuffer(l0b_buf[0], 16 * 512 * sizeof(bfloat16_t));
        pipe.InitBuffer(l0b_buf[1], 16 * 512 * sizeof(bfloat16_t));
        pipe.InitBuffer(l0c_buf, 32 * 16 * sizeof(float));

        auto a1 = a1_buf.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> b1[2] = {b1_buf[0].Get<bfloat16_t>(), b1_buf[1].Get<bfloat16_t>()};
        auto l0a = l0a_buf.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> l0b[2] = {l0b_buf[0].Get<bfloat16_t>(), l0b_buf[1].Get<bfloat16_t>()};
        auto l0c = l0c_buf.Get<float>();

        if (m0 < 16U || (dual_m && m1 < 16U)) {
            InitConstValue(a1, InitConstValueParams<bfloat16_t>{1U, 4096U, 0U, static_cast<bfloat16_t>(0)});
            PipeBarrier<PIPE_MTE2>();
        }

        GlobalTensor<bfloat16_t> a_gm, b_gm;
        GlobalTensor<float> c_gm;
        a_gm.SetGlobalBuffer(input, static_cast<uint64_t>(rows) * 6144ULL);
        b_gm.SetGlobalBuffer(weight + static_cast<uint64_t>(core) * 98304ULL, 98304ULL);
        c_gm.SetGlobalBuffer(output + static_cast<uint64_t>(core) * 16ULL, static_cast<uint64_t>(rows) * 256ULL);

        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID5);

        LoadData2DParams load_a{};
        load_a.repeatTimes = dual_m ? 128 : 64;
        load_a.srcStride = 1;
        load_a.ifTranspose = false;
        LoadData2DParams load_b{};
        load_b.repeatTimes = 32;
        load_b.srcStride = 1;
        load_b.ifTranspose = true;

        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        DataCopy(b1[0], b_gm[0], 16384);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        for (uint32_t kp = 0U; kp < 6U; ++kp) {
            const uint32_t bi = kp & 1U;
            const event_t b_evt = bi ? EVENT_ID1 : EVENT_ID0;
            if (kp + 1U < 6U) {
                const uint32_t nbi = (kp + 1U) & 1U;
                const event_t nb_evt = nbi ? EVENT_ID1 : EVENT_ID0;
                WaitFlag<HardEvent::MTE1_MTE2>(nb_evt);
                DataCopy(b1[nbi], b_gm[(kp + 1U) * 16384ULL], 16384);
                SetFlag<HardEvent::MTE2_MTE1>(nb_evt);
            }

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

            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID5);
            LoadData(l0a, a1, load_a);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID5);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID5);

            WaitFlag<HardEvent::MTE2_MTE1>(b_evt);
            for (uint32_t step = 0U; step < 2U; ++step) {
                const event_t lb_evt = step ? EVENT_ID4 : EVENT_ID3;
                WaitFlag<HardEvent::M_MTE1>(lb_evt);
                LoadData(l0b[step], b1[bi][step * 8192U], load_b);
                if (step == 1U) {
                    SetFlag<HardEvent::MTE1_MTE2>(b_evt);
                }
                SetFlag<HardEvent::MTE1_M>(lb_evt);
                WaitFlag<HardEvent::MTE1_M>(lb_evt);

                MmadParams mad{};
                mad.m = 16;
                mad.n = 16;
                mad.k = 512;
                mad.cmatrixSource = false;
                mad.cmatrixInitVal = (kp == 0U && step == 0U);
                Mmad(l0c, l0a[step * 8192U], l0b[step], mad);
                if (dual_m) {
                    Mmad(l0c[256U], l0a[16384U + step * 8192U], l0b[step], mad);
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
        fix.nSize = 16;
        fix.mSize = m0;
        fix.srcStride = 16;
        fix.dstStride = 256;
        fix.ndNum = 1;
        fix.quantPre = QuantMode_t::NoQuant;
        Fixpipe(c_gm, l0c, fix);
        if (dual_m) {
            fix.mSize = m1;
            Fixpipe(c_gm[16ULL * 256ULL], l0c[256U], fix);
        }
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    }
}

#ifdef K112726_BRIDGE_PRELOAD_GATE
struct BridgeGatePreload {
    TBuf<TPosition::B1> b1_buf;
    LocalTensor<bfloat16_t> b1;
    bool active = false;
};

__aicore__ inline BridgeGatePreload PreloadDirectBridgeGate16Aic(
    TPipe& pipe, __gm__ bfloat16_t* weight, int64_t rows)
{
    BridgeGatePreload st{};
    if ASCEND_IS_AIC {
        const uint32_t core = static_cast<uint32_t>(GetBlockIdx());
        if (core < 16U && rows > 0) {
            pipe.InitBuffer(st.b1_buf, 6U * 16384U * sizeof(bfloat16_t));
            st.b1 = st.b1_buf.Get<bfloat16_t>();
            GlobalTensor<bfloat16_t> b_gm;
            b_gm.SetGlobalBuffer(weight + static_cast<uint64_t>(core) * 98304ULL, 98304ULL);
            DataCopy(st.b1, b_gm[0], 6U * 16384U);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
            st.active = true;
        }
    }
    return st;
}

__aicore__ inline void RunDirectBridgeGate16Aic(
    TPipe& pipe, BridgeGatePreload& st, __gm__ bfloat16_t* input, __gm__ float* output, int64_t rows)
{
    if ASCEND_IS_AIC {
        const uint32_t core = static_cast<uint32_t>(GetBlockIdx());
        if (!st.active || core >= 16U || rows <= 0) {
            return;
        }
        const bool dual_m = (rows > 16);
        const uint16_t m0 = dual_m ? 16U : static_cast<uint16_t>(rows);
        const uint16_t m1 = dual_m ? static_cast<uint16_t>(rows - 16) : 0U;

        TBuf<TPosition::A1> a1_buf;
        TBuf<TPosition::A2> l0a_buf;
        TBuf<TPosition::B2> l0b_buf[2];
        TBuf<TPosition::CO1> l0c_buf;
        pipe.InitBuffer(a1_buf, 32 * 1024 * sizeof(bfloat16_t));
        pipe.InitBuffer(l0a_buf, 32 * 1024 * sizeof(bfloat16_t));
        pipe.InitBuffer(l0b_buf[0], 16 * 512 * sizeof(bfloat16_t));
        pipe.InitBuffer(l0b_buf[1], 16 * 512 * sizeof(bfloat16_t));
        pipe.InitBuffer(l0c_buf, 32 * 16 * sizeof(float));

        auto a1 = a1_buf.Get<bfloat16_t>();
        auto l0a = l0a_buf.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> l0b[2] = {l0b_buf[0].Get<bfloat16_t>(), l0b_buf[1].Get<bfloat16_t>()};
        auto l0c = l0c_buf.Get<float>();

        if (m0 < 16U || (dual_m && m1 < 16U)) {
            InitConstValue(a1, InitConstValueParams<bfloat16_t>{1U, 4096U, 0U, static_cast<bfloat16_t>(0)});
            PipeBarrier<PIPE_MTE2>();
        }

        GlobalTensor<bfloat16_t> a_gm;
        GlobalTensor<float> c_gm;
        a_gm.SetGlobalBuffer(input, static_cast<uint64_t>(rows) * 6144ULL);
        c_gm.SetGlobalBuffer(output + static_cast<uint64_t>(core) * 16ULL, static_cast<uint64_t>(rows) * 256ULL);

        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID5);

        LoadData2DParams load_a{};
        load_a.repeatTimes = dual_m ? 128 : 64;
        load_a.srcStride = 1;
        load_a.ifTranspose = false;
        LoadData2DParams load_b{};
        load_b.repeatTimes = 32;
        load_b.srcStride = 1;
        load_b.ifTranspose = true;

        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
#ifdef K113000_BRIDGE_GATE_OVERLAP
        CrossCoreWaitFlag(10);
#endif
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
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID5);
            LoadData(l0a, a1, load_a);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID5);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID5);

            for (uint32_t step = 0U; step < 2U; ++step) {
                const event_t lb_evt = step ? EVENT_ID4 : EVENT_ID3;
                WaitFlag<HardEvent::M_MTE1>(lb_evt);
                LoadData(l0b[step], st.b1[kp * 16384U + step * 8192U], load_b);
                SetFlag<HardEvent::MTE1_M>(lb_evt);
                WaitFlag<HardEvent::MTE1_M>(lb_evt);

                MmadParams mad{};
                mad.m = 16;
                mad.n = 16;
                mad.k = 512;
                mad.cmatrixSource = false;
                mad.cmatrixInitVal = (kp == 0U && step == 0U);
                Mmad(l0c, l0a[step * 8192U], l0b[step], mad);
                if (dual_m) {
                    Mmad(l0c[256U], l0a[16384U + step * 8192U], l0b[step], mad);
                }
                PipeBarrier<PIPE_M>();
                SetFlag<HardEvent::M_MTE1>(lb_evt);
            }
            SetFlag<HardEvent::M_MTE1>(EVENT_ID5);
        }
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID5);
        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
        FixpipeParamsV220 fix{};
        fix.nSize = 16;
        fix.mSize = m0;
        fix.srcStride = 16;
        fix.dstStride = 256;
        fix.ndNum = 1;
        fix.quantPre = QuantMode_t::NoQuant;
        Fixpipe(c_gm, l0c, fix);
        if (dual_m) {
            fix.mSize = m1;
            Fixpipe(c_gm[16ULL * 256ULL], l0c[256U], fix);
        }
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    }
}
#endif

__aicore__ inline void Gate(TPipe& pipe, __gm__ bfloat16_t* input,
    __gm__ bfloat16_t* weight, __gm__ float* output, GM_ADDR tiling,
    int64_t rows, int64_t blocks)
{
#ifdef K112612_BRIDGE_FAST
    (void)tiling;
    (void)blocks;
    DirectBridgeGate16Aic(pipe, input, weight, output, rows);
#else
    if ASCEND_IS_AIC {
        const int64_t core = GetBlockIdx();
        if (core >= kExperts / kGateTile) {
            return;
        }
        GlobalTensor<bfloat16_t> a, b;
        GlobalTensor<float> c;
        a.SetGlobalBuffer(input, rows * kHidden);
        b.SetGlobalBuffer(weight, kHidden * kExperts);
        c.SetGlobalBuffer(output, rows * kExperts);
        const int64_t cap = (rows <= 16) ? 16 : 32;
        TCubeTiling cube{};
        cube.M = cap;
        cube.singleCoreM = cap;
        const int64_t column = core * kGateTile;
        GateMatmul16 mm;
        mm.Init(&cube, &pipe);
        mm.SetOrgShape(cap, kGateTile, kHidden, kHidden, kExperts);
        mm.SetSingleShape(rows, kGateTile, kHidden);
        mm.SetTensorA(a, false);
        mm.SetTensorB(b[column * kHidden], false);
        mm.IterateAll(c[column]);
        mm.End();
    }
#endif
}

__aicore__ inline void TopK(TPipe& pipe, GM_ADDR logits, GM_ADDR bias,
    GM_ADDR probabilities, GM_ADDR ids, GM_ADDR system, int64_t rows)
{
    if ASCEND_IS_AIV {


        if (GetBlockIdx() < rows) {
            BridgeTopKTiling tiling;
            MoeGatingTopK::MoeGatingTopKWithoutGroup<float> router;
            router.Init(logits, bias, probabilities, ids, logits, system,
                        &tiling, &pipe);
            router.Process();
        }
    }
}

__aicore__ inline void Quantize(TPipe& pipe, __gm__ bfloat16_t* input,
    __gm__ int8_t* output, __gm__ float* scales, int64_t rows,
    __gm__ int8_t* staged = nullptr, int64_t capacity = 0,
    __gm__ int8_t* routed_rows = nullptr, __gm__ float* routed_scales = nullptr)
{
    if ASCEND_IS_AIV {
        const int64_t row = GetBlockIdx();
        if (row < rows) {
            whole_layer_norm::device::DynamicQuantRow quant;
            quant.Init(pipe);
            quant.Process(input + row * kHidden, output + row * kHidden,
                          scales + row, staged, row, capacity,
                          routed_rows != nullptr ? (routed_rows + row * kHidden) : nullptr,
                          routed_scales != nullptr ? (routed_scales + row) : nullptr);
        }
    }
}

__aicore__ inline void Epilogue(TPipe& pipe, GM_ADDR routed, GM_ADDR shared,
    GM_ADDR scaled, GM_ADDR combined, int64_t rows, int64_t blocks)
{
    if ASCEND_IS_AIV {
        constexpr int64_t tile = 1024;
        TBuf<TPosition::VECCALC> buffer;
        pipe.InitBuffer(buffer, tile * 12);
        auto bytes = buffer.Get<uint8_t>();
        auto x = bytes.ReinterpretCast<bfloat16_t>();
        auto y = bytes[tile * 2].ReinterpretCast<bfloat16_t>();
        auto value = bytes[tile * 4].ReinterpretCast<float>();
        auto term = bytes[tile * 8].ReinterpretCast<float>();
        GlobalTensor<bfloat16_t> input, other, intermediate, output;
        input.SetGlobalBuffer((__gm__ bfloat16_t*)routed, rows * kHidden);
        other.SetGlobalBuffer((__gm__ bfloat16_t*)shared, rows * kHidden);
        intermediate.SetGlobalBuffer((__gm__ bfloat16_t*)scaled, rows * kHidden);
        output.SetGlobalBuffer((__gm__ bfloat16_t*)combined, rows * kHidden);
        for (int64_t index = GetBlockIdx(); index < rows * kHidden / tile;
             index += 2 * blocks) {
            const int64_t begin = index * tile;
            DataCopy(x, input[begin], tile);
            DataCopy(y, other[begin], tile);
            Fence<HardEvent::MTE2_V>();
            Cast(value, x, RoundMode::CAST_NONE, tile);
            Cast(term, y, RoundMode::CAST_NONE, tile);
            PipeBarrier<PIPE_V>();
            Muls(value, value, 2.5F, tile);
            PipeBarrier<PIPE_V>();
            Cast(x, value, RoundMode::CAST_RINT, tile);
            Fence<HardEvent::V_MTE3>();
            DataCopy(intermediate[begin], x, tile);
            PipeBarrier<PIPE_V>();
            Cast(value, x, RoundMode::CAST_NONE, tile);
            PipeBarrier<PIPE_V>();
            Add(value, value, term, tile);
            PipeBarrier<PIPE_V>();
            Cast(y, value, RoundMode::CAST_RINT, tile);
            Fence<HardEvent::V_MTE3>();
            DataCopy(output[begin], y, tile);
            Fence<HardEvent::MTE3_MTE2>();
        }
    }
}


__aicore__ inline void QuantizeRouted(TPipe& pipe, __gm__ bfloat16_t* input,
    __gm__ int8_t* output_rows, __gm__ float* output_scales, int64_t rows, int64_t blocks)
{
    if ASCEND_IS_AIV {
        const int64_t idx = GetBlockIdx();
        const int64_t first = (rows <= 16) ? rows : 0;
        if (idx >= first && idx < first + rows) {
            TBuf<TPosition::VECCALC> input_buf, other_buf, work_buf, output_buf, scale_buf, input_narrow_buf;
            pipe.InitBuffer(input_buf, kHidden * sizeof(float));
            pipe.InitBuffer(other_buf, kHidden * sizeof(float));
            pipe.InitBuffer(input_narrow_buf, kHidden * sizeof(bfloat16_t));
            pipe.InitBuffer(work_buf, kHidden * sizeof(float));
            pipe.InitBuffer(output_buf, kHidden);
            pipe.InitBuffer(scale_buf, 32);
            auto values = input_buf.Get<float>();
            auto other = other_buf.Get<float>();
            auto work = work_buf.Get<float>();
            auto output = output_buf.Get<int8_t>();
            auto scale = scale_buf.Get<float>();
            auto bf16_input = input_narrow_buf.Get<bfloat16_t>();
            auto half_output = input_narrow_buf.Get<half>();
            GlobalTensor<bfloat16_t> hidden;
            GlobalTensor<int8_t> output_gm;
            GlobalTensor<float> scale_gm;
            hidden.SetGlobalBuffer(input, rows * kHidden);
            output_gm.SetGlobalBuffer(output_rows, rows * kHidden);
            scale_gm.SetGlobalBuffer(output_scales, rows);
            for (int64_t token = idx - first; token < rows; token += 2 * blocks) {
                DataCopy(bf16_input, hidden[token * kHidden], kHidden);
                Fence<HardEvent::MTE2_V>();
                Cast(values, bf16_input, RoundMode::CAST_NONE, kHidden);
                PipeBarrier<PIPE_V>();
                Abs(other, values, kHidden);
                PipeBarrier<PIPE_V>();
                ReduceMax(scale, other, work, kHidden, false);
                Fence<HardEvent::V_S>();
                const float quantum = scale.GetValue(0) / 127.0F;
                Fence<HardEvent::S_V>();
                Duplicate(scale, quantum, 8);
                Duplicate(work, quantum, 64);
                Duplicate(other, 1.0F, 64);
                PipeBarrier<PIPE_V>();
                Div(other, other, work, 64);
                PipeBarrier<PIPE_V>();
                Mul(work, values, other, 64, kHidden / 64,
                    BinaryRepeatParams{1, 1, 0, 8, 8, 0});
                PipeBarrier<PIPE_V>();
                auto integer = work.ReinterpretCast<int32_t>();
                Cast(integer, work, RoundMode::CAST_RINT, kHidden);
                PipeBarrier<PIPE_V>();
                SetDeqScale(half(1.0F));
                Cast(half_output, integer, RoundMode::CAST_ROUND, kHidden);
                PipeBarrier<PIPE_V>();
                Cast(output, half_output, RoundMode::CAST_TRUNC, kHidden);
                Fence<HardEvent::V_MTE3>();
                DataCopy(output_gm[token * kHidden], output, kHidden);
                DataCopyPad(scale_gm[token], scale,
                            DataCopyExtParams{1, sizeof(float), 0, 0, 0});
                Fence<HardEvent::MTE3_V>();
                Fence<HardEvent::MTE3_MTE2>();
            }
        }
    }
}

__aicore__ inline void StageSharedInput(TPipe& pipe, __gm__ int8_t* input,
    __gm__ int8_t* staged, int64_t rows, int64_t capacity, int64_t blocks)
{
    if ASCEND_IS_AIV {
        const int64_t vectors = 2 * blocks;
        const int64_t first = (2 * rows + capacity <= vectors) ? 2 * rows : 0;
        const int64_t row = GetBlockIdx() - first;
        if (row >= 0 && row < capacity) {
            constexpr int64_t kInnerTile = 1024;
            constexpr int64_t kInnerParts = kHidden / kInnerTile;
            TBuf<TPosition::VECCALC> buf;
            pipe.InitBuffer(buf, kHidden);
            auto held_input = buf.Get<int8_t>();
            GlobalTensor<int8_t> source, destination;
            source.SetGlobalBuffer(input, rows * kHidden);
            destination.SetGlobalBuffer(staged, capacity * kHidden);
            if (row < rows) {
                DataCopy(held_input, source[row * kHidden], kHidden);
                Fence<HardEvent::MTE2_MTE3>();
            } else {
                Duplicate(held_input.ReinterpretCast<int16_t>(), int16_t(0), kHidden / 2);
                Fence<HardEvent::V_MTE3>();
            }
            DataCopyParams store_parts;
            store_parts.blockCount = kInnerParts;
            store_parts.blockLen = kInnerTile / 32;
            store_parts.srcStride = 0;
            store_parts.dstStride = (capacity - 1) * (kInnerTile / 32);
            DataCopy(destination[row * kInnerTile], held_input, store_parts);
        }
    }
}
}
#endif
