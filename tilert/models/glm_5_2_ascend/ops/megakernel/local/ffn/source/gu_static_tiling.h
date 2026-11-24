#ifndef TILERT_FFN_ORIGINAL_SHARED_GU_STATIC_TILING_H
#define TILERT_FFN_ORIGINAL_SHARED_GU_STATIC_TILING_H

#include "lib/matmul_intf.h"
#include "routed_int4_contract.h"

namespace ffn_original_shared_gu_static {

template <typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class FastCubeInBuffer {
    using TransT = typename INPUT_TYPE::TRANS_T;
public:
    __aicore__ inline FastCubeInBuffer() {}
    __aicore__ inline ~FastCubeInBuffer() {}

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

    __aicore__ inline void FreeTensor(int32_t = -1, const AscendC::LocalTensor<TransT>& = AscendC::LocalTensor<TransT>{})
    {}

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
struct FastMatmulPolicy : public AscendC::Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
    using TransT_B = decltype(GetTransBDataType<A_TYPE, B_TYPE, MM_CFG>());
    using CubeInBufferA = FastCubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FastCubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename TransT_B::T>, MM_CFG>;
};

__aicore__ constexpr ::MatmulApiStaticTiling GateTiling()
{
    ::MatmulApiStaticTiling value{};
    value.N = routed_int4::kGateUp;
    value.Ka = routed_int4::kHidden;
    value.Kb = routed_int4::kHidden;
    value.singleCoreN = -1;
    value.singleCoreK = routed_int4::kHidden;
    value.baseM = 64;
    value.baseN = 256;
    value.baseK = 256;
    value.depthA1 = 24;
    value.depthB1 = 8;
    value.stepM = 1;
    value.stepN = 1;
    value.stepKa = 24;
    value.stepKb = 4;
    value.iterateOrder = 1;
    value.shareL1Size = 458752;
    value.shareL0CSize = 65536;
    value.shareUbSize = 0;
    value.dbL0A = 2;
    value.dbL0B = 2;
    value.dbL0C = 1;
    value.cfg = ::CFG_NORM;
    return value;
}

__aicore__ constexpr ::MatmulApiStaticTiling DownTiling()
{
    ::MatmulApiStaticTiling value{};
    value.N = routed_int4::kHidden;
    value.Ka = routed_int4::kIntermediate;
    value.Kb = routed_int4::kIntermediate;
    value.singleCoreN = -1;
    value.iterateOrder = 1;
    value.singleCoreK = routed_int4::kIntermediate;
    value.baseM = 64;
    value.baseN = 256;
    value.baseK = 256;
    value.depthA1 = 8;
    value.depthB1 = 8;
    value.stepM = 1;
    value.stepN = 1;
    value.stepKa = 8;
    value.stepKb = 8;
    value.iterateOrder = 1;
    value.shareL1Size = 327680;
    value.shareL0CSize = 65536;
    value.shareUbSize = 0;
    value.dbL0A = 2;
    value.dbL0B = 2;
    value.dbL0C = 1;
    value.cfg = ::CFG_NORM;
    return value;
}

inline constexpr auto kGateTiling = GateTiling();
inline constexpr auto kDownTiling = DownTiling();
static_assert(kGateTiling.M == -1 && kGateTiling.singleCoreM == -1);
static_assert(kDownTiling.M == -1 && kDownTiling.singleCoreM == -1);


__aicore__ constexpr ::MatmulApiStaticTiling GateTiling16()
{
    ::MatmulApiStaticTiling value{};
    value.N = routed_int4::kGateUp;
    value.Ka = routed_int4::kHidden;
    value.Kb = routed_int4::kHidden;
    value.singleCoreN = -1;
    value.singleCoreK = routed_int4::kHidden;
    value.baseM = 16;
#if defined(K11040_GU_FAST128)
    value.baseN = 128;
    value.baseK = 512;
    value.depthA1 = 12;
    value.depthB1 = 12;
    value.stepKa = 12;
    value.stepKb = 6;
#elif GU_WIDE
    value.baseN = 256;
    value.baseK = 256;
    value.depthA1 = 24;
    value.depthB1 = 4;
    value.stepKa = 24;
    value.stepKb = 2;
#else
    value.baseN = 128;
    value.baseK = 512;
    value.depthA1 = 12;
    value.depthB1 = 12;
    value.stepKa = 12;
    value.stepKb = 1;
#endif
    value.stepM = 1;
    value.stepN = 1;
    value.iterateOrder = 1;
    value.shareL1Size = 458752;
    value.shareL0CSize = 16384;
    value.shareUbSize = 0;
    value.dbL0A = 2;
    value.dbL0B = 2;
    value.dbL0C = 2;
    value.cfg = ::CFG_NORM;
    return value;
}

__aicore__ constexpr ::MatmulApiStaticTiling DownTiling16()
{
    ::MatmulApiStaticTiling value{};
    value.N = routed_int4::kHidden;
    value.Ka = routed_int4::kIntermediate;
    value.Kb = routed_int4::kIntermediate;
    value.singleCoreN = -1;
    value.iterateOrder = 1;
    value.singleCoreK = routed_int4::kIntermediate;
    value.baseM = 16;
#ifdef K11050_DOWN_FAST128
    value.baseN = 128;
    value.baseK = 512;
    value.depthA1 = 4;
    value.depthB1 = 12;
    value.stepM = 1;
    value.stepN = 1;
    value.stepKa = 4;
    value.stepKb = 4;
    value.iterateOrder = 1;
    value.shareL1Size = 458752;
    value.shareL0CSize = 16384;
#else
    value.baseN = 256;
    value.baseK = 256;
    value.depthA1 = 8;
    value.depthB1 = 8;
    value.stepM = 1;
    value.stepN = 1;
    value.stepKa = 8;
    value.stepKb = 2;
    value.iterateOrder = 1;
    value.shareL1Size = 327680;
    value.shareL0CSize = 32768;
#endif
    value.shareUbSize = 0;
    value.dbL0A = 2;
    value.dbL0B = 2;
    value.dbL0C = 2;
    value.cfg = ::CFG_NORM;
    return value;
}

inline constexpr auto kGateTiling16 = GateTiling16();
inline constexpr auto kDownTiling16 = DownTiling16();

__aicore__ constexpr ::MatmulApiStaticTiling GateTiling16_N1024()
{
    ::MatmulApiStaticTiling value = GateTiling16();
    value.singleCoreN = 1024;
    value.iterateOrder = 1;
    return value;
}

__aicore__ constexpr ::MatmulApiStaticTiling DownTiling16_N1536()
{
    ::MatmulApiStaticTiling value = DownTiling16();
    value.singleCoreN = 1536;
    value.iterateOrder = 1;
    return value;
}

inline constexpr auto kGateTiling16_N1024 = GateTiling16_N1024();
inline constexpr auto kDownTiling16_N1536 = DownTiling16_N1536();

}

#endif
