

#ifndef LIGHTNING_INDEXER_SERVICE_VECTOR_ARCH22_H
#define LIGHTNING_INDEXER_SERVICE_VECTOR_ARCH22_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../lightning_indexer_common.h"
#include "lightning_indexer_vector.h"

namespace LIKernel {
using namespace LICommon;
using namespace LIServiceVec;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t SPARSE_COUNT_4K = 4096;
constexpr uint32_t LD_PARAM_NUM = 16;
constexpr uint32_t EVENTID_V_TO_MTE2_PING = 0;
constexpr uint32_t EVENTID_V_TO_MTE2_PONG = 1;
constexpr uint32_t EVENTID_V_TO_MTE2_TMPUB = 2;


template <typename Q_T, typename W_T = void>
struct LightningIndexerTypeTraits {
    using weightsType = Q_T;
};


template <typename Q_T>
struct LightningIndexerTypeTraits<Q_T, float> {
    using weightsType = float;
};

template <typename LIT>
class LightningIndexerServiceVector {
public:


    static constexpr bool DT_W_FLAG = LIT::weightsTypeFlag;
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    static constexpr LI_LAYOUT LAYOUT_T = LIT::layout;
    using W_T =
        typename LightningIndexerTypeTraits<Q_T, typename std::conditional<DT_W_FLAG, float, void>::type>::weightsType;


    using MM1_OUT_T = float;

    __aicore__ inline LightningIndexerServiceVector(){};
    __aicore__ inline void ProcessVec(const LICommon::RunInfo& info);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitBuffers(TPipe* pipe);
    __aicore__ inline void InitParams(const struct LICommon::ConstInfo& constInfo,
                                      const LITilingData* __restrict tilingData);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm,
                                                GlobalTensor<int64_t> vec1ParamGm, GlobalTensor<W_T> weightsGm,
                                                GlobalTensor<int32_t> indiceOutGm, GlobalTensor<K_T> valueOutGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe* pipe);
    uint64_t treeMaskGmAddr_ = 0;

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<int64_t> vec1ParamGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<K_T> valueOutGm;


private:


    TQue<QuePosition::VECOUT, 1> outQueue_;


    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> reduceOutBuf_;
    TBuf<TPosition::VECCALC> brcBuf_;
    TBuf<TPosition::VECCALC> paramBuf_;


    TBuf<> ldToBeMrgBuf_;
    TBuf<> ldTmpBuf_;
    TBuf<> ldOutValueBuf_;
    TBuf<> ldOutIdxBuf_;

    LocalTensor<float> tmpUb_;
    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;
    LocalTensor<float> SortedBasicBlock_;

    int32_t blockId_ = -1;

    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;


    uint32_t mrgListNum_ = 4;
    uint32_t paramNum_ = 16;
    int32_t virTopK = 0;

    constexpr static uint32_t REDUCE_BANK_CONFLICT_OFFSETS = 256;
    constexpr static uint32_t REDUCE_BANK_CONFLICT_NUM = REDUCE_BANK_CONFLICT_OFFSETS / sizeof(float);

    struct LICommon::ConstInfo constInfo_;
};

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::InitBuffers(TPipe* pipe)
{
    uint32_t outNeedBufSize = (BASE_TOPK * 2) * 2 * sizeof(float);
    uint32_t reduceCacheSize = REDUCE_BANK_CONFLICT_OFFSETS + groupInner_ * s2BaseSize_ * sizeof(float);
    outNeedBufSize = reduceCacheSize > outNeedBufSize ? reduceCacheSize : outNeedBufSize;
    virTopK = constInfo_.isSparseCountOver2K ? constInfo_.sparseCount : BASE_TOPK;

    pipe->InitBuffer(outQueue_, 1, outNeedBufSize);

    pipe->InitBuffer(tmpBuf_, (groupInner_ * s2BaseSize_ + s2BaseSize_) * 2 * sizeof(float));
    pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * virTopK * 2 * sizeof(float));
    pipe->InitBuffer(indexBuf_, s2BaseSize_ * sizeof(int32_t));
    pipe->InitBuffer(reduceOutBuf_, s2BaseSize_ * 2 * sizeof(float));
    pipe->InitBuffer(brcBuf_, groupInner_ * 8 * sizeof(float));
    pipe->InitBuffer(paramBuf_, LD_PARAM_NUM * sizeof(int64_t));

    tmpUb_ = tmpBuf_.Get<float>();
    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    SortedBasicBlock_ = globalTopkUb_[virTopK * 2 * 2];
    globalTopkNum_ = 0;


    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);

    InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * virTopK * 2);


    LocalTensor<float> tmpBuff = outQueue_.AllocTensor<float>();
    Duplicate(tmpBuff.template ReinterpretCast<int32_t>(), -1, 2 * (s1BaseSize_ / 2) * paramNum_ * 2);
    outQueue_.EnQue<float>(tmpBuff);
    tmpBuff = outQueue_.DeQue<float>();
    int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ +
                           (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * paramNum_;
    DataCopyPad(vec1ParamGm[wsInfoOffset], tmpBuff.template ReinterpretCast<int64_t>(),
                {1, static_cast<uint16_t>((s1BaseSize_ / 2) * 2 * paramNum_ * sizeof(int64_t)), 0, 0});
    outQueue_.FreeTensor(tmpBuff);
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::InitLDBuffers(TPipe* pipe)
{
    pipe->Reset();
    pipe->InitBuffer(ldToBeMrgBuf_, 2 * BASE_TOPK * mrgListNum_ * sizeof(float));
    pipe->InitBuffer(ldTmpBuf_, 2 * BASE_TOPK * mrgListNum_ * sizeof(float));
    pipe->InitBuffer(ldOutValueBuf_, BASE_TOPK * sizeof(float));
    pipe->InitBuffer(ldOutIdxBuf_, BASE_TOPK * sizeof(int32_t));
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::InitParams(const struct LICommon::ConstInfo& constInfo,
                                                                      const LITilingData* __restrict tilingData)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;

    kHeadNum_ = constInfo.kHeadNum;

    s1BaseSize_ = constInfo.s1BaseSize;
    s2BaseSize_ = constInfo.s2BaseSize;


    groupInner_ = 16;

    blockId_ = GetBlockIdx();
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm, GlobalTensor<int64_t> vec1ParamGm,
    GlobalTensor<W_T> weightsGm, GlobalTensor<int32_t> indiceOutGm, GlobalTensor<K_T> valueOutGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->vec1ParamGm = vec1ParamGm;
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
    this->valueOutGm = valueOutGm;
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PING);
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PONG);
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PING);
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PONG);
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::CleanInvalidOutput(int64_t invalidS1offset)
{

    LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
    LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>();
    Duplicate(idxULocal1, constInfo_.INVALID_IDX, constInfo_.sparseCount);
    outQueue_.EnQue<float>(valueULocal);
    valueULocal = outQueue_.DeQue<float>();
    LIServiceVec::CopyOut(indiceOutGm[invalidS1offset], idxULocal1, constInfo_.sparseCount);
    outQueue_.FreeTensor(valueULocal);

    if (constInfo_.returnValue) {
        uint16_t negInf = 0;
        if constexpr (std::is_same<K_T, float16_t>::value) {
            negInf = 0xFC00;
        } else {
            negInf = 0xFF80;
        }
        LocalTensor<uint16_t> valueULocal = outQueue_.AllocTensor<uint16_t>();
        Duplicate(valueULocal, negInf, constInfo_.sparseCount);
        outQueue_.EnQue<uint16_t>(valueULocal);
        valueULocal = outQueue_.DeQue<uint16_t>();
        GlobalTensor<uint16_t> valueOutGmTmp;
        valueOutGmTmp.SetGlobalBuffer((__gm__ uint16_t*)valueOutGm.GetPhyAddr());
        LIServiceVec::CopyOut(valueOutGmTmp[invalidS1offset], valueULocal, constInfo_.sparseCount);
        outQueue_.FreeTensor(valueULocal);
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::ProcessVec(const LICommon::RunInfo& info)
{
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;


    int64_t mmGmOffset = (info.loop % 2) * (constInfo_.mBaseSizeAlign * s2BaseSize_);

    int64_t weightGmOffset = info.tensorWeightsOffset + cuBaseS1Idx * kHeadNum_ * gSize_;

    PipeBarrier<PIPE_V>();

    int32_t cuS1BeginIdxPerAiv = cuBaseS1Idx;
    int32_t cuS1ProcNum =
        cuS1BeginIdxPerAiv + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;

    int32_t cuS1ProcNumPerAiv = blockId_ % 2 == 0 ? CeilDiv(cuS1ProcNum, 2) : (cuS1ProcNum / 2);
    cuS1BeginIdxPerAiv += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2);


    weightGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * kHeadNum_ * gSize_;
    mmGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * gSize_ * info.actualSingleProcessSInnerSizeAlign;


    int32_t outerG = CeilDiv(gSize_, groupInner_);


    if (info.loop != 0 && info.s2Idx == 0) {

        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * virTopK * 2);
        blockS2StartIdx_ = 0;
    } else if (info.loop == 0) {
        blockS2StartIdx_ = info.s2Idx;
    }

    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {

        cuRealAcSeq = info.actS2Size - (info.actS1Size - cuS1BeginIdxPerAiv);
    }
    LocalTensor<float> reduceOutBuff = reduceOutBuf_.Get<float>();
    LocalTensor<float> brcBuf = brcBuf_.Get<float>();

    uint32_t ldS1Offset = (blockId_ % 2 == 0) ? s1BaseSize_ / 2 - cuS1ProcNumPerAiv : 0;
    for (int innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; innerS1Idx++) {
        if (constInfo_.attenMaskFlag) {
            cuRealAcSeq += 1;
        }
        int32_t cuS2Len = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq ? cuRealAcSeq - cuBaseS2Idx : s2BaseSize_;
        int32_t cuS1Idx = cuS1BeginIdxPerAiv + innerS1Idx;
        if (constInfo_.qSeqSize > 0 && cuS1Idx >= static_cast<int32_t>(constInfo_.qSeqSize)) {
            continue;
        }
        if (cuRealAcSeq > 0 && cuS2Len > 0) {
            int32_t cuS2LenVecAlign = CeilDiv(cuS2Len, s2BaseSize_) * s2BaseSize_;
            int32_t mmUbStride = (cuS2LenVecAlign - info.actualSingleProcessSInnerSizeAlign) / B32_BLOCK_ALIGN_NUM;
            LocalTensor<float> reduceOutInner = reduceOutBuff[s2BaseSize_];
            PipeBarrier<PIPE_V>();
            LocalTensor<float> reduceCacheBuf = outQueue_.AllocTensor<float>();
            if (constInfo_.isSparseCountOver2K) {
                WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
            }
            for (int outerGidx = 0; outerGidx < outerG; outerGidx++) {
                int32_t procGnum = outerGidx != outerG - 1 ? groupInner_ : gSize_ - outerGidx * groupInner_;

                int32_t pingpong = outerGidx % 2;
                LocalTensor<float> dbTmpUb = tmpUb_[pingpong * (groupInner_ * s2BaseSize_ + s2BaseSize_)];
                LocalTensor<float> weightsInUb = dbTmpUb[procGnum * s2BaseSize_];
                WaitFlag<HardEvent::V_MTE2>(pingpong);
                LocalTensor<W_T> weightsInTUb = weightsInUb.template ReinterpretCast<W_T>();
                if constexpr (!IsSameType<W_T, float>::value) {
                    weightsInTUb = weightsInTUb[groupInner_];
                }
                int64_t mmGmAllOffet = mmGmOffset + innerS1Idx * gSize_ * info.actualSingleProcessSInnerSizeAlign +
                                       outerGidx * groupInner_ * info.actualSingleProcessSInnerSizeAlign;
                int64_t weightGmAllOffset = weightGmOffset + innerS1Idx * gSize_ + outerGidx * groupInner_;

                LIServiceVec::CopyIn(dbTmpUb, weightsInTUb, mm1ResGm, weightsGm, mmGmAllOffet, weightGmAllOffset,
                                     procGnum, info.actualSingleProcessSInnerSizeAlign, mmUbStride);

                SetFlag<HardEvent::MTE2_V>(pingpong);
                WaitFlag<HardEvent::MTE2_V>(pingpong);
                LIServiceVec::DoScale(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], dbTmpUb, weightsInUb, weightsInTUb,
                                      brcBuf, procGnum, s2BaseSize_, outerGidx);


                SetFlag<HardEvent::V_MTE2>(pingpong);
            }

            int32_t gRedCnt = groupInner_ > gSize_ ? gSize_ : groupInner_;
            bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq;
            LIServiceVec::DoReduce(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], reduceOutInner, gRedCnt, s2BaseSize_);
            outQueue_.FreeTensor(reduceCacheBuf);

            LocalTensor<float> sortScoreUb = reduceOutBuff;
            LocalTensor<float> sortIndiceUb = reduceOutBuff[cuS2LenVecAlign];
            Duplicate(sortScoreUb.template ReinterpretCast<int32_t>(), LIServiceVec::NEG_INF, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            Adds(sortScoreUb, reduceOutInner, 0.0f, cuS2Len);
            PipeBarrier<PIPE_V>();
            LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();

            if (cuS2LenVecAlign != cuS2Len) {
                Duplicate(sortIndiceUbInt, -1, cuS2LenVecAlign);
            }
            PipeBarrier<PIPE_V>();
            Adds(sortIndiceUbInt, globalTopkIndice_, static_cast<int32_t>(cuBaseS2Idx), cuS2Len);
            PipeBarrier<PIPE_V>();
#ifdef K112162_FUSED_IDX_RTK
            const uint64_t effTreeMaskAddr = ((treeMaskGmAddr_ & 1ULL) != 0ULL)
                ? reinterpret_cast<const __gm__ uint64_t*>(treeMaskGmAddr_ & ~1ULL)[2]
                : treeMaskGmAddr_;
#else
            const uint64_t effTreeMaskAddr = treeMaskGmAddr_;
#endif
            if (effTreeMaskAddr != 0ULL && constInfo_.attenMaskFlag && cuS1Idx >= 0 && cuS1Idx < 32) {
                const int32_t prefixLen = static_cast<int32_t>(info.actS2Size) - static_cast<int32_t>(info.actS1Size);
                if (static_cast<int32_t>(cuBaseS2Idx) + cuS2Len > prefixLen) {
                    const __gm__ uint32_t* maskPtr = reinterpret_cast<const __gm__ uint32_t*>(effTreeMaskAddr);
                    if (maskPtr[0] != 0U) {
                        const uint32_t rowMask = maskPtr[cuS1Idx];
                        AscendC::PipeBarrier<PIPE_ALL>();
                        auto scoreInt = sortScoreUb.template ReinterpretCast<int32_t>();
                        for (int32_t j = 0; j <= cuS1Idx; ++j) {
                            if (((rowMask >> j) & 1U) == 0U) {
                                const int32_t col = prefixLen + j - static_cast<int32_t>(cuBaseS2Idx);
                                if (col >= 0 && col < cuS2Len) {
                                    scoreInt.SetValue(col, LIServiceVec::NEG_INF);
                                    sortIndiceUbInt.SetValue(col, -1);
                                }
                            }
                        }
                        AscendC::PipeBarrier<PIPE_ALL>();
                    }
                }
            }

            LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
            if (info.actS1Size > 4 || constInfo_.isSparseCountOver2K) {

                LIServiceVec::SortAll(reduceOutBuff, tmpSortBuf,
                                      cuS2LenVecAlign);
                PipeBarrier<PIPE_V>();
                LocalTensor<float> UbTmpSort = constInfo_.isSparseCountOver2K ? tmpUb_ : tmpSortBuf;
                LIServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK, reduceOutBuff,
                                        cuS2LenVecAlign, UbTmpSort);
            } else {
                int64_t globalTopkUbCacheIdx = (info.s2Idx - blockS2StartIdx_) % 4;
                Sort<float, true>(
                    SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2 + globalTopkUbCacheIdx * s2BaseSize_ * 2],
                    reduceOutBuff, sortIndiceUbInt.template ReinterpretCast<uint32_t>(), tmpSortBuf,
                    cuS2LenVecAlign / 32);
                AscendC::PipeBarrier<PIPE_V>();

                if (globalTopkUbCacheIdx == 3 || isS2End || info.isAllLoopEnd) {
                    LocalTensor<float> tt = SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2];

                    if (info.s2Idx - blockS2StartIdx_ < 4) {
                        MrgBasicBlock(globalTopkUb_[innerS1Idx * BASE_TOPK * 2], tt,
                                      static_cast<int64_t>(globalTopkUbCacheIdx + 1), s2BaseSize_);
                    } else {
                        if (globalTopkUbCacheIdx > 0) {
                            MrgBasicBlock(tmpSortBuf, tt, static_cast<int64_t>(globalTopkUbCacheIdx + 1), s2BaseSize_);
                            PipeBarrier<PIPE_V>();
                            DataCopy(SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2], tmpSortBuf,
                                     (globalTopkUbCacheIdx + 1) * s2BaseSize_ * 2);
                        }
                        PipeBarrier<PIPE_V>();
                        SparseTopK(globalTopkUb_[innerS1Idx * BASE_TOPK * 2],
                                   SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2], tmpSortBuf, BASE_TOPK,
                                   s2BaseSize_ * (globalTopkUbCacheIdx + 1));
                    }
                }
            }
            if (constInfo_.isSparseCountOver2K) {
                SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
            }

            PipeBarrier<PIPE_V>();
            outQueue_.FreeTensor(tmpSortBuf);

            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End;


            bool needCopyWsGm = info.isAllLoopEnd || isS2End;

            if (needCopyOutGm) {
                int64_t offset = (constInfo_.sparseCount <= SPARSE_COUNT_4K) ? virTopK : constInfo_.sparseCount / 2;
                int64_t copyLen =
                    (constInfo_.sparseCount <= SPARSE_COUNT_4K) ? constInfo_.sparseCount : constInfo_.sparseCount / 2;
                int64_t copyNum = (constInfo_.sparseCount <= SPARSE_COUNT_4K) ? 1 : 2;
                for (int64_t i = 0; i < copyNum; i++) {
                    LocalTensor<float> outValueUb = outQueue_.AllocTensor<float>();
                    LocalTensor<uint32_t> outIdxUb = outValueUb[offset].template ReinterpretCast<uint32_t>();
                    Extract(outValueUb, outIdxUb, globalTopkUb_[innerS1Idx * virTopK * 2 + 2 * i * offset],
                            (offset / 32));

                    LocalTensor<K_T> valueULocal1 = outValueUb.template ReinterpretCast<K_T>();
                    if (constInfo_.returnValue) {
                        PipeBarrier<PIPE_V>();
                        Cast(valueULocal1, outValueUb, RoundMode::CAST_ROUND, copyLen);
                    }

                    LocalTensor<int32_t> idxULocal1 = outValueUb[offset].template ReinterpretCast<int32_t>();
                    outQueue_.EnQue<float>(outValueUb);
                    outValueUb = outQueue_.DeQue<float>();

                    const bool skipIdxOut = ((treeMaskGmAddr_ & 1ULL) != 0ULL &&
                                             constInfo_.sparseCount == 2048U &&
                                             !constInfo_.returnValue);
                    if (!skipIdxOut) {
                        LIServiceVec::CopyOut(
                            indiceOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount + i * offset],
                            idxULocal1, copyLen);
                        if (constInfo_.returnValue) {
                            LIServiceVec::CopyOut(
                                valueOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount + i * offset],
                                valueULocal1, copyLen);
                        }
                    }
#ifdef K112162_FUSED_IDX_RTK
                    if (skipIdxOut) {
                        const uint64_t* rtkHdr = reinterpret_cast<const uint64_t*>(treeMaskGmAddr_ & ~1ULL);
                        const int32_t kvLenRtk = static_cast<int32_t>(rtkHdr[3]);
                        const uint32_t sfaRowsRtk = static_cast<uint32_t>(rtkHdr[4]);
                        const uint32_t tablePagesRtk = static_cast<uint32_t>(rtkHdr[5]);
                        const uint32_t qRowRtk =
                            static_cast<uint32_t>(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount) >> 11U;
                        const uint32_t actBlksRtk = (static_cast<uint32_t>(kvLenRtk) + 127U) >> 7U;
                        LocalTensor<int32_t> workI32 = tmpBuf_.Get<int32_t>();
                        LocalTensor<uint32_t> workU32 = tmpBuf_.Get<uint32_t>();
                        LocalTensor<int32_t> outUb = workI32[0];
                        LocalTensor<uint32_t> outU32 = workU32[0];
                        LocalTensor<int32_t> rC = workI32[512];
                        LocalTensor<uint32_t> rCU32 = workU32[512];
                        LocalTensor<int32_t> signI32 = workI32[1024];
                        LocalTensor<uint32_t> signU32 = workU32[1024];
                        LocalTensor<int32_t> tmpA = workI32[1536];
                        LocalTensor<uint32_t> tmpAU32 = workU32[1536];
                        LocalTensor<int32_t> tmpB = workI32[2048];
                        LocalTensor<uint32_t> tmpBU32 = workU32[2048];
                        LocalTensor<uint32_t> idxU32 = workU32[2560];
                        LocalTensor<int32_t> blkUb = workI32[3072];
                        LocalTensor<uint32_t> maskUb = workU32[3584];
                        LocalTensor<int32_t> rowLutUb = workI32[3648];
                        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
                        AscendC::GlobalTensor<int32_t> ptGm, dstGm;
                        ptGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rtkHdr[1]), tablePagesRtk);
                        dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rtkHdr[0]), sfaRowsRtk * 2048U);
                        AscendC::DataCopy(blkUb, ptGm, (actBlksRtk + 7U) & ~7U);
                        if (rtkHdr[2] != 0ULL) {
                            AscendC::GlobalTensor<uint32_t> mGm;
                            mGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(rtkHdr[2]), 32);
                            AscendC::DataCopy(maskUb, mGm, 32);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID7);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID7);
                        const int32_t prefLenRtk = kvLenRtk - static_cast<int32_t>(sfaRowsRtk);
                        const bool maskEnRtk = (rtkHdr[2] != 0ULL) && (maskUb.GetValue(0) != 0U);
                        const bool hasRowMaskRtk = maskEnRtk && (qRowRtk < 32U);
                        const uint32_t rowMaskRtk = hasRowMaskRtk ? maskUb.GetValue(qRowRtk) : 0xFFFFFFFFU;
                        const uint32_t fullQBitsRtk = (qRowRtk >= 31U) ? 0xFFFFFFFFU : ((1U << (qRowRtk + 1U)) - 1U);
                        const bool needLutRtk = hasRowMaskRtk && ((rowMaskRtk & fullQBitsRtk) != fullQBitsRtk);
                        const int32_t s2LimitRtk = prefLenRtk + static_cast<int32_t>(qRowRtk) + 1;
                        if (needLutRtk) {
                            rowLutUb.SetValue(0, 1);
                            for (uint32_t d = 0U; d < 32U; ++d) {
                                const int32_t bitOk = (d <= qRowRtk && ((rowMaskRtk >> d) & 1U) != 0U) ? 1 : 0;
                                rowLutUb.SetValue(1U + d, bitOk);
                            }
                            for (uint32_t d = 33U; d < 64U; ++d) {
                                rowLutUb.SetValue(d, 0);
                            }
                            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID7);
                            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID7);
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        for (uint32_t cIdx = 0U; cIdx < 4U; ++cIdx) {
                            const uint32_t cOff = cIdx * 512U;
                            AscendC::Maxs(rC, idxULocal1[cOff], -1, 512U);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::Mins(rC, rC, 16777216, 512U);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::ShiftRight(signU32, rCU32, 31U, 512U);
                            AscendC::PipeBarrier<PIPE_V>();
                            if (!needLutRtk) {
                                AscendC::Adds(tmpA, rC, -s2LimitRtk, 512U);
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::ShiftRight(tmpAU32, tmpAU32, 31U, 512U);
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::Sub(tmpA, tmpA, signI32, 512U);
                                AscendC::PipeBarrier<PIPE_V>();
                            } else {
                                AscendC::Adds(tmpB, rC, 1 - prefLenRtk, 512U);
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::Maxs(tmpB, tmpB, 0, 512U);
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::Mins(tmpB, tmpB, 33, 512U);
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::ShiftLeft(idxU32, tmpBU32, 2U, 512U);
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::Gather(tmpA, rowLutUb, idxU32, 0U, 512U);
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::Sub(tmpA, tmpA, signI32, 512U);
                                AscendC::PipeBarrier<PIPE_V>();
                            }
                            AscendC::Maxs(rC, rC, 0, 512U);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::Mins(rC, rC, kvLenRtk - 1, 512U);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::ShiftRight(tmpBU32, rCU32, 7U, 512U);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::ShiftLeft(idxU32, tmpBU32, 2U, 512U);
                            AscendC::ShiftLeft(tmpBU32, tmpBU32, 7U, 512U);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::Gather(outUb, blkUb, idxU32, 0U, 512U);
                            AscendC::Sub(rC, rC, tmpB, 512U);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::ShiftLeft(outU32, outU32, 7U, 512U);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::Add(outUb, outUb, rC, 512U);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::Adds(outUb, outUb, 2, 512U);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::Mul(outUb, tmpA, outUb, 512U);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::Add(outUb, outUb, signI32, 512U);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::Adds(outUb, outUb, -2, 512U);
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
                            AscendC::DataCopy(dstGm[qRowRtk * 2048U + cOff], outUb, 512U);
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);
                    }
#endif
                    outQueue_.FreeTensor(outValueUb);
                }
            } else if (needCopyWsGm) {


                int64_t wsOffset = (blockId_ / 2) * s1BaseSize_ * 2 * 2 * BASE_TOPK +
                                   (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * 2 * BASE_TOPK +
                                   (ldS1Offset + innerS1Idx) * 2 * 2 * BASE_TOPK;
                int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ +
                                       (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * paramNum_ +
                                       (ldS1Offset + innerS1Idx) * 2 * paramNum_;

                LocalTensor<int64_t> tmpiBuff = paramBuf_.Get<int64_t>();
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                tmpiBuff.SetValue(0, static_cast<int64_t>(1));
                tmpiBuff.SetValue(1, static_cast<int64_t>(cuRealAcSeq));
                tmpiBuff.SetValue(2, static_cast<int64_t>(blockS2StartIdx_));
                tmpiBuff.SetValue(3, static_cast<int64_t>(cuBaseS2Idx + cuS2Len));
                tmpiBuff.SetValue(4, static_cast<int64_t>(isS2End));
                tmpiBuff.SetValue(5, static_cast<int64_t>(info.bN2Idx));
                tmpiBuff.SetValue(6, static_cast<int64_t>(cuS1Idx));
                tmpiBuff.SetValue(7, static_cast<int64_t>(cuS1ProcNum));
                tmpiBuff.SetValue(8, static_cast<int64_t>(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount));


                bool isTailReduce = blockS2StartIdx_ == 0;


                if (isTailReduce) {
                    wsInfoOffset += paramNum_;
                    wsOffset += 2 * BASE_TOPK;
                }
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                LIServiceVec::CopyOut(vec1ParamGm[wsInfoOffset], tmpiBuff, 16);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                LIServiceVec::CopyOut(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * BASE_TOPK * 2], 2 * BASE_TOPK);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            }
        } else if (cuRealAcSeq <= 0) {
            CleanInvalidOutput(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount);
        }
    }


    if (LAYOUT_T == LI_LAYOUT::BSND) {

        bool isS1LoopEnd = (cuBaseS1Idx + s1BaseSize_) >= info.actS1Size;
        int32_t invalidS1Num = constInfo_.qSeqSize - info.actS1Size;

        if (invalidS1Num > 0 && isS1LoopEnd && blockS2StartIdx_ == 0) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num, 2) : (invalidS1Num / 2);
            int32_t s1OffsetPerAiv = info.actS1Size + (blockId_ % 2) * CeilDiv(invalidS1Num, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput(info.indiceOutOffset + (s1OffsetPerAiv + innerS1Idx) * constInfo_.sparseCount);
            }
        }

        int32_t invalidS1Num2 = info.actS1Size - info.actS2Size;
        if (invalidS1Num2 > 0 && isS1LoopEnd && blockS2StartIdx_ == 0 && constInfo_.attenMaskFlag) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num2, 2) : (invalidS1Num2 / 2);
            int32_t s1OffsetPerAiv = (blockId_ % 2) * CeilDiv(invalidS1Num2, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput((info.bN2Idx * constInfo_.qSeqSize + s1OffsetPerAiv + innerS1Idx) *
                                   constInfo_.sparseCount);
            }
        }
    }

    if (info.isLastS2InnerLoop) {

        blockS2StartIdx_ = 0;
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::ProcessLD()
{
    int32_t curCubeId = blockId_ / 2;
    int32_t tmpCubeId = curCubeId;

    int64_t s2ActSeq;
    int64_t s2Start;
    int64_t s2End;
    int64_t isS2End;
    int64_t bn2Idx;
    int64_t s1Idx;
    uint32_t acc_list_num = 0;
    int64_t bIdx = 0;
    int64_t needFd;
    int64_t wsOffset;
    int64_t wsInfoOffset = 0;
    int64_t nextneedFd;
    int64_t valueOffset = 0;
    int64_t outOffset = 0;

    LocalTensor<float> curValueIdxUb = ldToBeMrgBuf_.Get<float>();
    LocalTensor<float> tmpUb = ldTmpBuf_.Get<float>();


    uint32_t s1LdStartIdx = 0;
    uint32_t s1ProcNum = 0;
    uint64_t paramGmCoreOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_;
    for (uint32_t innerS1Idx = 0; innerS1Idx < s1BaseSize_; innerS1Idx++) {
        needFd = vec1ParamGm.GetValue(paramGmCoreOffset + innerS1Idx * 2 * paramNum_ + paramNum_);
        if (needFd == 1) {
            s1LdStartIdx = (s1ProcNum == 0) ? innerS1Idx : s1LdStartIdx;
            s1ProcNum++;
        }
    }

    if (s1ProcNum == 0) {
        return;
    }


    uint32_t s1VecNum = CeilDiv(s1ProcNum, 2);
    if (blockId_ % 2 == 1) {
        s1LdStartIdx = s1LdStartIdx + s1VecNum;
        s1VecNum = s1ProcNum - s1VecNum;
    }
    for (uint32_t innerS1Idx = s1LdStartIdx; innerS1Idx < s1LdStartIdx + s1VecNum; innerS1Idx++) {

        tmpCubeId = curCubeId;
        acc_list_num = 0;
        valueOffset = 0;


        wsOffset = tmpCubeId * s1BaseSize_ * 2 * 2 * BASE_TOPK +
                   innerS1Idx * 2 * 2 * BASE_TOPK + 2 * BASE_TOPK;
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        DataCopyPad(curValueIdxUb, vec1ResGm[wsOffset],
                    {1, static_cast<uint16_t>(2 * BASE_TOPK * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
        acc_list_num++;
        valueOffset += 2 * BASE_TOPK;


        tmpCubeId++;
        wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
        needFd = vec1ParamGm.GetValue(wsInfoOffset);
        isS2End = vec1ParamGm.GetValue(wsInfoOffset + 4);
        s1Idx = vec1ParamGm.GetValue(wsInfoOffset + 6);
        outOffset = vec1ParamGm.GetValue(wsInfoOffset + 8);

        while (needFd == 1) {

            wsOffset = tmpCubeId * s1BaseSize_ * 2 * 2 * BASE_TOPK +
                       innerS1Idx * 2 * 2 * BASE_TOPK;
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            DataCopyPad(curValueIdxUb[valueOffset], vec1ResGm[wsOffset],
                        {1, static_cast<uint16_t>(2 * BASE_TOPK * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
            valueOffset += 2 * BASE_TOPK;
            acc_list_num++;


            if (acc_list_num == mrgListNum_) {

                AscendC::MrgSort4Info params;
                params.elementLengths[0] = BASE_TOPK;
                params.elementLengths[1] = BASE_TOPK;
                params.elementLengths[2] = BASE_TOPK;
                params.elementLengths[3] = BASE_TOPK;
                params.ifExhaustedSuspension = true;
                params.validBit = 0b1111;
                params.repeatTimes = 1;

                AscendC::MrgSortSrcList<float> srcList;
                srcList.src1 = curValueIdxUb[0];
                srcList.src2 = curValueIdxUb[2 * BASE_TOPK];
                srcList.src3 = curValueIdxUb[4 * BASE_TOPK];
                srcList.src4 = curValueIdxUb[6 * BASE_TOPK];
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                MrgSort(tmpUb, srcList, params);
                PipeBarrier<PIPE_V>();
                DataCopy(curValueIdxUb, tmpUb, 2 * BASE_TOPK);
                PipeBarrier<PIPE_V>();
                acc_list_num = 1;
                valueOffset = 2 * BASE_TOPK;
            }


            if (isS2End == 1) {
                break;
            }

            tmpCubeId++;
            wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
            needFd = vec1ParamGm.GetValue(wsInfoOffset);
            isS2End = vec1ParamGm.GetValue(wsInfoOffset + 4);
        }


        if (acc_list_num != 1) {
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = BASE_TOPK;
            params.elementLengths[1] = BASE_TOPK;
            params.elementLengths[2] = BASE_TOPK;
            params.elementLengths[3] = BASE_TOPK;
            params.ifExhaustedSuspension = true;
            if (acc_list_num == 2) {
                params.validBit = 0b0011;
            } else if (acc_list_num == 3) {
                params.validBit = 0b0111;
            }
            params.repeatTimes = 1;

            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[2 * BASE_TOPK];
            srcList.src3 = curValueIdxUb[4 * BASE_TOPK];
            srcList.src4 = curValueIdxUb[6 * BASE_TOPK];
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, 2 * BASE_TOPK);
            PipeBarrier<PIPE_V>();
        }


        LocalTensor<float> outValueUb = ldOutValueBuf_.Get<float>();
        LocalTensor<uint32_t> outIdxUb = ldOutIdxBuf_.Get<uint32_t>();
        const bool skipIdxOut = ((treeMaskGmAddr_ & 1ULL) != 0ULL &&
                                 constInfo_.sparseCount == 2048U &&
                                 !constInfo_.returnValue);
        if (!constInfo_.returnValue) {
            Extract(outValueUb, outIdxUb, curValueIdxUb, (BASE_TOPK / 32));
            LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
            if (!skipIdxOut) {
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                DataCopyPad(indiceOutGm[outOffset], idxULocal1,
                            {1, static_cast<uint16_t>(constInfo_.sparseCount * sizeof(int32_t)), 0, 0});
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            }
#ifdef K112162_FUSED_IDX_RTK
            if (skipIdxOut) {
                const uint64_t* rtkHdr = reinterpret_cast<const uint64_t*>(treeMaskGmAddr_ & ~1ULL);
                const int32_t kvLenRtk = static_cast<int32_t>(rtkHdr[3]);
                const uint32_t sfaRowsRtk = static_cast<uint32_t>(rtkHdr[4]);
                const uint32_t tablePagesRtk = static_cast<uint32_t>(rtkHdr[5]);
                const uint32_t qRowRtk = static_cast<uint32_t>(outOffset) >> 11U;
                const uint32_t actBlksRtk = (static_cast<uint32_t>(kvLenRtk) + 127U) >> 7U;
                LocalTensor<int32_t> workI32 = ldToBeMrgBuf_.Get<int32_t>();
                LocalTensor<uint32_t> workU32 = ldToBeMrgBuf_.Get<uint32_t>();
                LocalTensor<int32_t> outUb = workI32[0];
                LocalTensor<uint32_t> outU32 = workU32[0];
                LocalTensor<int32_t> rC = workI32[2048];
                LocalTensor<uint32_t> rCU32 = workU32[2048];
                LocalTensor<int32_t> signI32 = workI32[4096];
                LocalTensor<uint32_t> signU32 = workU32[4096];
                LocalTensor<int32_t> tmpA = workI32[6144];
                LocalTensor<uint32_t> tmpAU32 = workU32[6144];
                LocalTensor<int32_t> tmpB = workI32[8192];
                LocalTensor<uint32_t> tmpBU32 = workU32[8192];
                LocalTensor<uint32_t> idxU32 = workU32[10240];
                LocalTensor<int32_t> blkUb = workI32[12288];
                LocalTensor<uint32_t> maskUb = workU32[12800];
                LocalTensor<int32_t> rowLutUb = workI32[12864];
                SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
                AscendC::GlobalTensor<int32_t> ptGm, dstGm;
                ptGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rtkHdr[1]), tablePagesRtk);
                dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rtkHdr[0]), sfaRowsRtk * 2048U);
                AscendC::DataCopy(blkUb, ptGm, (actBlksRtk + 7U) & ~7U);
                if (rtkHdr[2] != 0ULL) {
                    AscendC::GlobalTensor<uint32_t> mGm;
                    mGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(rtkHdr[2]), 32);
                    AscendC::DataCopy(maskUb, mGm, 32);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID7);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID7);
                const int32_t prefLenRtk = kvLenRtk - static_cast<int32_t>(sfaRowsRtk);
                const bool maskEnRtk = (rtkHdr[2] != 0ULL) && (maskUb.GetValue(0) != 0U);
                const bool hasRowMaskRtk = maskEnRtk && (qRowRtk < 32U);
                const uint32_t rowMaskRtk = hasRowMaskRtk ? maskUb.GetValue(qRowRtk) : 0xFFFFFFFFU;
                const uint32_t fullQBitsRtk = (qRowRtk >= 31U) ? 0xFFFFFFFFU : ((1U << (qRowRtk + 1U)) - 1U);
                const bool needLutRtk = hasRowMaskRtk && ((rowMaskRtk & fullQBitsRtk) != fullQBitsRtk);
                const int32_t s2LimitRtk = prefLenRtk + static_cast<int32_t>(qRowRtk) + 1;
                if (needLutRtk) {
                    rowLutUb.SetValue(0, 1);
                    for (uint32_t d = 0U; d < 32U; ++d) {
                        const int32_t bitOk = (d <= qRowRtk && ((rowMaskRtk >> d) & 1U) != 0U) ? 1 : 0;
                        rowLutUb.SetValue(1U + d, bitOk);
                    }
                    for (uint32_t d = 33U; d < 64U; ++d) {
                        rowLutUb.SetValue(d, 0);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID7);
                    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID7);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Maxs(rC, idxULocal1, -1, 2048U);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mins(rC, rC, 16777216, 2048U);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ShiftRight(signU32, rCU32, 31U, 2048U);
                AscendC::PipeBarrier<PIPE_V>();
                if (!needLutRtk) {
                    AscendC::Adds(tmpA, rC, -s2LimitRtk, 2048U);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::ShiftRight(tmpAU32, tmpAU32, 31U, 2048U);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Sub(tmpA, tmpA, signI32, 2048U);
                    AscendC::PipeBarrier<PIPE_V>();
                } else {
                    AscendC::Adds(tmpB, rC, 1 - prefLenRtk, 2048U);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Maxs(tmpB, tmpB, 0, 2048U);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mins(tmpB, tmpB, 33, 2048U);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::ShiftLeft(idxU32, tmpBU32, 2U, 2048U);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Gather(tmpA, rowLutUb, idxU32, 0U, 2048U);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Sub(tmpA, tmpA, signI32, 2048U);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Maxs(rC, rC, 0, 2048U);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mins(rC, rC, kvLenRtk - 1, 2048U);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ShiftRight(tmpBU32, rCU32, 7U, 2048U);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ShiftLeft(idxU32, tmpBU32, 2U, 2048U);
                AscendC::ShiftLeft(tmpBU32, tmpBU32, 7U, 2048U);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Gather(outUb, blkUb, idxU32, 0U, 2048U);
                AscendC::Sub(rC, rC, tmpB, 2048U);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ShiftLeft(outU32, outU32, 7U, 2048U);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(outUb, outUb, rC, 2048U);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(outUb, outUb, 2, 2048U);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(outUb, tmpA, outUb, 2048U);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(outUb, outUb, signI32, 2048U);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(outUb, outUb, -2, 2048U);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
                AscendC::DataCopy(dstGm[qRowRtk * 2048U], outUb, 2048U);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);
            }
#endif
        } else {
            Extract(outValueUb, outIdxUb, curValueIdxUb, (BASE_TOPK / 32));
            PipeBarrier<PIPE_V>();
            LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
            LocalTensor<K_T> valueULocal1 = outValueUb.template ReinterpretCast<K_T>();
            Cast(valueULocal1, outValueUb, RoundMode::CAST_ROUND, constInfo_.sparseCount);
            PipeBarrier<PIPE_V>();
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(indiceOutGm[outOffset], idxULocal1,
                        {1, static_cast<uint16_t>(constInfo_.sparseCount * sizeof(int32_t)), 0, 0});
            DataCopyPad(valueOutGm[outOffset], valueULocal1,
                        {1, static_cast<uint16_t>(constInfo_.sparseCount * sizeof(K_T)), 0, 0});
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
    }
}
}
#endif
