

#ifndef quant_lightning_indexer_SERVICE_VECTOR_H
#define quant_lightning_indexer_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../quant_lightning_indexer_common.h"
#include "quant_lightning_indexer_vector.h"

namespace QLIKernel {
using namespace QLICommon;
using namespace QLIServiceVec;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t BASE_TOPK_VALUE_IDX_SIZE = 4096;
constexpr uint32_t LD_PARAM_NUM = 16;

template <typename QLIT>
class QLIVector {
public:

    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIT::keyLayout;
    static constexpr bool PAGE_ATTENTION = QLIT::pageAttention;

    using MM1_OUT_T = float;

    __aicore__ inline QLIVector(){};
    __aicore__ inline void ProcessVec0(const QLICommon::RunInfo &info);
    __aicore__ inline void ProcessVec1(const QLICommon::RunInfo &info);
    __aicore__ inline void ProcessLD(int32_t ldCubeId, uint32_t ldInnerS1Idx);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct QLICommon::ConstInfo &constInfo,
                                      const QLITilingData *__restrict tilingData);
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<half> vec0OutGm, GlobalTensor<MM1_OUT_T> mm1ResGm,
                                                  GlobalTensor<float> vec1ResGm, GlobalTensor<int64_t> vec1ParamGm);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<half> weightsGm, GlobalTensor<half> qScaleGm,
                                              GlobalTensor<half> kScaleGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<int32_t> blockTableGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe *pipe);
    uint64_t treeMaskGmAddr_ = 0;

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<int64_t> vec1ParamGm;
    GlobalTensor<half> weightsGm;
    GlobalTensor<half> qScaleGm;
    GlobalTensor<half> kScaleGm;
    GlobalTensor<half> vec0OutGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> blockTableGm;


private:
    __aicore__ inline void GetKeyScale(const QLICommon::RunInfo &runInfo, const LocalTensor<half> &resUb,
                                       int64_t batchId, int64_t startS2, int64_t getLen);


    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;


    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> paramBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> kScaleBuf_;


    TBuf<> ldToBeMrgBuf_;
    TBuf<> ldTmpBuf_;
    TBuf<> ldOutValueBuf_;
    TBuf<> ldOutIdxBuf_;

    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;

    int32_t blockId_ = -1;

    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kSeqSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t qHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    int32_t kCacheBlockSize_ = 0;
    int32_t maxBlockNumPerBatch_ = 0;


    uint32_t mrgListNum_ = 4;
    uint32_t paramNum_ = 16;

    struct QLICommon::ConstInfo constInfo_;
};

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::GetKeyScale(const QLICommon::RunInfo &runInfo, const LocalTensor<half> &resUb,
                                                    int64_t batchId, int64_t startS2, int64_t getLen)
{
    if constexpr (PAGE_ATTENTION) {
        int32_t startBlockTableIdx = startS2 >> 7;
        int32_t blockTableBatchOffset = batchId * maxBlockNumPerBatch_;
        int32_t getLoopNum = CeilDiv(getLen, 128);
        for (int32_t i = 0; i < getLoopNum; i++) {
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx + i);
            uint64_t kScaleOffset = blockId * constInfo_.keyDequantScaleStride0;
            if (i == getLoopNum - 1 && (getLen & 127) != 0) {
                AscendC::DataCopyExtParams copyInParams{1, static_cast<uint32_t>((getLen - i * 128) * sizeof(half)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
                AscendC::DataCopyPad(resUb[i * 128], kScaleGm[kScaleOffset], copyInParams, padParams);
            } else {
                AscendC::DataCopy(resUb[i * 128], kScaleGm[kScaleOffset], 128);
            }
        }
    } else {
        AscendC::DataCopyExtParams copyInParams{1, static_cast<uint32_t>(getLen * sizeof(half)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(resUb, kScaleGm[runInfo.tensorKeyScaleOffset], copyInParams, padParams);
    }
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(paramBuf_, 2 * LD_PARAM_NUM * sizeof(int64_t));
    pipe->InitBuffer(kScaleBuf_, s2BaseSize_ * sizeof(float));
    pipe->InitBuffer(inQueue_, 2, s2BaseSize_ * sizeof(float) * 2);
    pipe->InitBuffer(outQueue_, 1, BASE_TOPK * sizeof(float));
    pipe->InitBuffer(tmpBuf_, 64 * 1024);
    pipe->InitBuffer(indexBuf_, s2BaseSize_ * sizeof(int32_t));
    pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK_VALUE_IDX_SIZE * sizeof(float));

    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    globalTopkNum_ = 0;


    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);


    LocalTensor<float> tmpBuff = outQueue_.AllocTensor<float>();
    Duplicate(tmpBuff.template ReinterpretCast<int32_t>(), -1, 2 * (s1BaseSize_ / 2) * paramNum_ * 2);
    outQueue_.EnQue<float>(tmpBuff);
    tmpBuff = outQueue_.DeQue<float>();
    int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ +
                           (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * paramNum_;
    DataCopyPad(vec1ParamGm[wsInfoOffset], tmpBuff.template ReinterpretCast<int64_t>(),
                {1, static_cast<uint16_t>((s1BaseSize_ / 2) * 2 * paramNum_ * sizeof(int64_t)), 0, 0});
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    outQueue_.FreeTensor(tmpBuff);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitLDBuffers(TPipe *pipe)
{
    pipe->Reset();
    pipe->InitBuffer(ldToBeMrgBuf_, BASE_TOPK_VALUE_IDX_SIZE * mrgListNum_ * sizeof(float));
    pipe->InitBuffer(ldTmpBuf_, BASE_TOPK_VALUE_IDX_SIZE * mrgListNum_ * sizeof(float));
    pipe->InitBuffer(ldOutIdxBuf_, BASE_TOPK * sizeof(int32_t));
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitParams(const struct QLICommon::ConstInfo &constInfo,
                                                   const QLITilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;

    qHeadNum_ = constInfo.qHeadNum;
    kHeadNum_ = constInfo.kHeadNum;

    s1BaseSize_ = constInfo.s1BaseSize;
    s2BaseSize_ = constInfo.s2BaseSize;
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    blockId_ = GetBlockIdx();
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitVecInputTensor(GlobalTensor<half> weightsGm, GlobalTensor<half> qScaleGm,
                                                           GlobalTensor<half> kScaleGm,
                                                           GlobalTensor<int32_t> indiceOutGm,
                                                           GlobalTensor<int32_t> blockTableGm)
{
    this->weightsGm = weightsGm;
    this->qScaleGm = qScaleGm;
    this->kScaleGm = kScaleGm;
    this->indiceOutGm = indiceOutGm;
    this->blockTableGm = blockTableGm;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitVecWorkspaceTensor(GlobalTensor<half> vec0OutGm,
                                                               GlobalTensor<MM1_OUT_T> mm1ResGm,
                                                               GlobalTensor<float> vec1ResGm,
                                                               GlobalTensor<int64_t> vec1ParamGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->vec0OutGm = vec0OutGm;
    this->vec1ParamGm = vec1ParamGm;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::AllocEventID()
{}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::FreeEventID()
{}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::CleanInvalidOutput(int64_t invalidS1offset)
{

    LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
    LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>();
    Duplicate(idxULocal1, constInfo_.INVALID_IDX, constInfo_.sparseCount);
    outQueue_.EnQue<float>(valueULocal);
    valueULocal = outQueue_.DeQue<float>();
    QLIServiceVec::CopyOut(indiceOutGm[invalidS1offset], idxULocal1, constInfo_.sparseCount);
    outQueue_.FreeTensor(valueULocal);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessVec0(const QLICommon::RunInfo &info)
{

    if (blockId_ % 2 != 0) {
        return;
    }
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;

    int64_t vec0OutGmOffset = (info.loop % 2) * ((s1BaseSize_ * gSize_ * BLOCK_CUBE));

    int64_t weightGmOffset = info.tensorWeightsOffset + cuBaseS1Idx * qHeadNum_;

    int32_t cuS1ProcNum = cuBaseS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int32_t cuProcEleNum = CeilAlign(cuS1ProcNum * gSize_, 32);

    LocalTensor<half> inWeightsUb = inQueue_.AllocTensor<half>();
    LocalTensor<half> inQScaleUb = inWeightsUb[cuProcEleNum];
    AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = cuS1ProcNum * gSize_ * sizeof(half);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;
    AscendC::DataCopyPad(inWeightsUb, weightsGm[weightGmOffset], copyInParams, padParams);
    AscendC::DataCopyPad(inQScaleUb, qScaleGm[weightGmOffset], copyInParams, padParams);

    inQueue_.EnQue<half>(inWeightsUb);
    inWeightsUb = inQueue_.DeQue<half>();
    AscendC::Mul(inWeightsUb, inWeightsUb, inQScaleUb, cuProcEleNum);
    PipeBarrier<PIPE_V>();
    LocalTensor<half> resUb = outQueue_.AllocTensor<half>();
    AscendC::Brcb(resUb, inWeightsUb, static_cast<uint8_t>(cuProcEleNum / 8), {1, 8});
    inQueue_.FreeTensor(inWeightsUb);

    outQueue_.EnQue<half>(resUb);
    resUb = outQueue_.DeQue<half>();
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = cuS1ProcNum * gSize_ * BLOCK_CUBE * sizeof(half);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    AscendC::DataCopyPad(vec0OutGm[vec0OutGmOffset], resUb, copyOutParams);
    outQueue_.FreeTensor(resUb);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessVec1(const QLICommon::RunInfo &info)
{
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;


    int64_t mmGmOffset = (info.loop % 2) * (s1BaseSize_ * s2BaseSize_);


    int32_t cuS1BeginIdxPerAiv = cuBaseS1Idx;
    int32_t cuS1ProcNum =
        cuS1BeginIdxPerAiv + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;

    int32_t cuS1ProcNumPerAiv = blockId_ % 2 == 0 ? CeilDiv(cuS1ProcNum, 2) : (cuS1ProcNum / 2);
    cuS1BeginIdxPerAiv += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2);

    mmGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * s2BaseSize_;

    if (info.loop != 0 && info.s2Idx == 0) {


        blockS2StartIdx_ = 0;
    } else if (info.loop == 0) {
        blockS2StartIdx_ = info.s2Idx;
    }

    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {

        cuRealAcSeq = info.actS2Size - (info.actS1Size - cuS1BeginIdxPerAiv);
    }


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
            bool isFirstSegBlock = (info.loop == 0) || (info.s2Idx == 0);
            int32_t cuS2LenVecAlign = (cuS2Len <= 128) ? 128 : ((cuS2Len <= 512) ? 512 : ((cuS2Len <= 1024) ? 1024 : s2BaseSize_));
            LocalTensor<float> mmInUb = inQueue_.AllocTensor<float>();
            LocalTensor<float> kScaleUb = kScaleBuf_.Get<float>();
            LocalTensor<half> kScaleTUb = mmInUb[s2BaseSize_].template ReinterpretCast<half>();
            AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
            AscendC::DataCopyExtParams copyInParams;
            copyInParams.blockCount = 1;
            copyInParams.blockLen = cuS2Len * sizeof(float);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = 0;
            copyInParams.rsv = 0;
            AscendC::DataCopyPad(mmInUb, mm1ResGm[mmGmOffset + innerS1Idx * s2BaseSize_], copyInParams, padParams);
            if (innerS1Idx == 0) {
                int32_t maxRealAcSeq = constInfo_.attenMaskFlag ? (cuRealAcSeq + cuS1ProcNumPerAiv - 1) : cuRealAcSeq;
                int32_t maxS2Len = cuBaseS2Idx + s2BaseSize_ >= maxRealAcSeq ? maxRealAcSeq - cuBaseS2Idx : s2BaseSize_;
                GetKeyScale(info, kScaleTUb, info.bIdx, cuBaseS2Idx, maxS2Len);
            }
            inQueue_.EnQue<float>(mmInUb);
            mmInUb = inQueue_.DeQue<float>();
            if (innerS1Idx == 0) {
                int32_t maxRealAcSeq = constInfo_.attenMaskFlag ? (cuRealAcSeq + cuS1ProcNumPerAiv - 1) : cuRealAcSeq;
                int32_t maxS2Len = cuBaseS2Idx + s2BaseSize_ >= maxRealAcSeq ? maxRealAcSeq - cuBaseS2Idx : s2BaseSize_;
                AscendC::Cast(kScaleUb, kScaleTUb, RoundMode::CAST_NONE, maxS2Len);
                PipeBarrier<PIPE_V>();
            }
            LocalTensor<float> dstTopk = globalTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE];
            LocalTensor<float> sortBuff = isFirstSegBlock ? dstTopk : tmpBuf_.Get<float>();
            LocalTensor<float> sortScoreUb = sortBuff;
            LocalTensor<float> sortIndiceUb = sortBuff[cuS2LenVecAlign];
            LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();
            if (cuS2LenVecAlign != cuS2Len) {
                int32_t padOffset = cuS2Len & ~7;
                int32_t padLen = cuS2LenVecAlign - padOffset;
                Duplicate(sortScoreUb[padOffset].template ReinterpretCast<int32_t>(), QLIServiceVec::NEG_INF, padLen);
                Duplicate(sortIndiceUbInt[padOffset], -1, padLen);
                PipeBarrier<PIPE_V>();
            }
            AscendC::Mul(sortScoreUb, mmInUb, kScaleUb, cuS2Len);
            inQueue_.FreeTensor(mmInUb);
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
                                    scoreInt.SetValue(col, QLIServiceVec::NEG_INF);
                                    sortIndiceUbInt.SetValue(col, -1);
                                }
                            }
                        }
                        AscendC::PipeBarrier<PIPE_ALL>();
                    }
                }
            }
            LocalTensor<float> tmpSortBuf = (isFirstSegBlock && innerS1Idx == 1) ? tmpBuf_.Get<float>() : tmpBuf_.Get<float>()[2 * s2BaseSize_];
            QLIServiceVec::SortAll(sortBuff, tmpSortBuf, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq;
            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End;
            bool needCopyWsGm = info.isAllLoopEnd || isS2End;
            bool isLastInner = (innerS1Idx == cuS1ProcNumPerAiv - 1);
            if (isFirstSegBlock) {
                if (cuS2LenVecAlign != BASE_TOPK) {
                    InitSortOutBuf(dstTopk[cuS2LenVecAlign * 2], BASE_TOPK_VALUE_IDX_SIZE - cuS2LenVecAlign * 2);
                }
            } else {
                if ((needCopyOutGm || needCopyWsGm) && isLastInner && info.isAllLoopEnd) {
                    AscendC::MrgSort4Info mparams;
                    mparams.elementLengths[0] = cuS2LenVecAlign;
                    mparams.elementLengths[1] = BASE_TOPK;
                    mparams.ifExhaustedSuspension = false;
                    mparams.validBit = 0b0011;
                    mparams.repeatTimes = 1;
                    AscendC::MrgSortSrcList<float> msrcList;
                    msrcList.src1 = sortBuff;
                    msrcList.src2 = dstTopk;
                    AscendC::MrgSort<float>(tmpSortBuf, msrcList, mparams);
                    PipeBarrier<PIPE_V>();
                    dstTopk = tmpSortBuf;
                } else {
                    QLIServiceVec::MergeSort(dstTopk, BASE_TOPK, sortBuff, cuS2LenVecAlign, tmpSortBuf);
                    PipeBarrier<PIPE_V>();
                }
            }
            if (needCopyOutGm) {
                LocalTensor<uint32_t> idxULocal = outQueue_.AllocTensor<uint32_t>();
                ExtractIndex(idxULocal,
                             dstTopk.template ReinterpretCast<uint32_t>(),
                             BASE_TOPK);
                PipeBarrier<PIPE_V>();
                outQueue_.EnQue<uint32_t>(idxULocal);
                idxULocal = outQueue_.DeQue<uint32_t>();
                const bool skipIdxOut = ((treeMaskGmAddr_ & 1ULL) != 0ULL &&
                                         constInfo_.sparseCount == 2048U);
                if (!skipIdxOut) {
                    QLIServiceVec::CopyOut(
                        indiceOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount],
                        idxULocal.template ReinterpretCast<int32_t>(),
                        constInfo_.sparseCount);
                }
#ifdef K112162_FUSED_IDX_RTK
                if (skipIdxOut) {
                    const uint64_t* rtkHdr = reinterpret_cast<const uint64_t*>(treeMaskGmAddr_ & ~1ULL);
                    const int32_t kvLenRtk = static_cast<int32_t>(rtkHdr[3]);
                    const uint32_t sfaRowsRtk = static_cast<uint32_t>(rtkHdr[4]);
                    const uint32_t tablePagesRtk = static_cast<uint32_t>(rtkHdr[5]);
                    const uint32_t qRowRtk = static_cast<uint32_t>(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount) >> 11U;
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
                    LocalTensor<int32_t> rawRowUb = idxULocal.template ReinterpretCast<int32_t>();
                    for (uint32_t cIdx = 0U; cIdx < 4U; ++cIdx) {
                        const uint32_t cOff = cIdx * 512U;
                        AscendC::Maxs(rC, rawRowUb[cOff], -1, 512U);
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
                outQueue_.FreeTensor(idxULocal);
            } else if (needCopyWsGm) {
                int64_t wsOffset =
                    (blockId_ / 2) * s1BaseSize_ * 2 * BASE_TOPK_VALUE_IDX_SIZE +
                    (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * BASE_TOPK_VALUE_IDX_SIZE +
                    (ldS1Offset + innerS1Idx) * 2 * BASE_TOPK_VALUE_IDX_SIZE;
                int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ +
                                       (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * paramNum_ +
                                       (ldS1Offset + innerS1Idx) * 2 * paramNum_;

                LocalTensor<int64_t> tmpiBuff = paramBuf_.Get<int64_t>()[innerS1Idx * LD_PARAM_NUM];
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
                    wsOffset += BASE_TOPK_VALUE_IDX_SIZE;
                }
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                QLIServiceVec::CopyOut(vec1ParamGm[wsInfoOffset], tmpiBuff, 16);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                QLIServiceVec::CopyOut(vec1ResGm[wsOffset], dstTopk,
                                       BASE_TOPK_VALUE_IDX_SIZE);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            }
        } else if (cuRealAcSeq <= 0) {
            CleanInvalidOutput(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount);
        }
    }


    if (Q_LAYOUT_T == LI_LAYOUT::BSND) {

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

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessLD(int32_t ldCubeId, uint32_t ldInnerS1Idx)
{
    int32_t curCubeId = ldCubeId;
    int32_t tmpCubeId = curCubeId;

    int64_t isS2End;
    uint32_t acc_list_num = 0;
    int64_t needFd;
    int64_t wsOffset;
    int64_t wsInfoOffset = 0;
    int64_t valueOffset = 0;
    int64_t outOffset = 0;

    uint64_t paramGmCoreOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_;
    uint64_t initParamOffset = paramGmCoreOffset + ldInnerS1Idx * 2 * paramNum_ + paramNum_;
    AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                      AscendC::DcciDst::CACHELINE_OUT>(vec1ParamGm[initParamOffset]);
    needFd = vec1ParamGm.GetValue(initParamOffset);
    if (needFd != 1) {
        return;
    }

    LocalTensor<float> curValueIdxUb = ldToBeMrgBuf_.Get<float>();
    LocalTensor<float> tmpUb = ldTmpBuf_.Get<float>();

    {
        uint32_t innerS1Idx = ldInnerS1Idx;

        tmpCubeId = curCubeId;
        acc_list_num = 0;
        valueOffset = 0;


        wsOffset = tmpCubeId * s1BaseSize_ * 2 * BASE_TOPK_VALUE_IDX_SIZE +
                   innerS1Idx * 2 * BASE_TOPK_VALUE_IDX_SIZE + BASE_TOPK_VALUE_IDX_SIZE;
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        DataCopy(curValueIdxUb, vec1ResGm[wsOffset], BASE_TOPK_VALUE_IDX_SIZE);
        acc_list_num++;
        valueOffset += BASE_TOPK_VALUE_IDX_SIZE;


        tmpCubeId++;
        wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(vec1ParamGm[wsInfoOffset]);
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(vec1ParamGm[wsInfoOffset + 8]);
        needFd = vec1ParamGm.GetValue(wsInfoOffset);
        isS2End = vec1ParamGm.GetValue(wsInfoOffset + 4);
        outOffset = vec1ParamGm.GetValue(wsInfoOffset + 8);

        bool resInTmp = false;
        while (needFd == 1) {

            wsOffset = tmpCubeId * s1BaseSize_ * 2 * BASE_TOPK_VALUE_IDX_SIZE +
                       innerS1Idx * 2 * BASE_TOPK_VALUE_IDX_SIZE;
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            DataCopy(curValueIdxUb[valueOffset], vec1ResGm[wsOffset], BASE_TOPK_VALUE_IDX_SIZE);
            valueOffset += BASE_TOPK_VALUE_IDX_SIZE;
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
                srcList.src1 = curValueIdxUb[0 * BASE_TOPK_VALUE_IDX_SIZE];
                srcList.src2 = curValueIdxUb[1 * BASE_TOPK_VALUE_IDX_SIZE];
                srcList.src3 = curValueIdxUb[2 * BASE_TOPK_VALUE_IDX_SIZE];
                srcList.src4 = curValueIdxUb[3 * BASE_TOPK_VALUE_IDX_SIZE];
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                MrgSort(tmpUb, srcList, params);
                PipeBarrier<PIPE_V>();
                acc_list_num = 1;
                valueOffset = BASE_TOPK_VALUE_IDX_SIZE;
                if (isS2End == 1) {
                    resInTmp = true;
                    break;
                }
                DataCopy(curValueIdxUb, tmpUb, BASE_TOPK_VALUE_IDX_SIZE);
                PipeBarrier<PIPE_V>();
                SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            }


            if (isS2End == 1) {
                break;
            }

            tmpCubeId++;
            wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
            AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(vec1ParamGm[wsInfoOffset]);
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
            srcList.src2 = curValueIdxUb[BASE_TOPK_VALUE_IDX_SIZE];
            srcList.src3 = curValueIdxUb[2 * BASE_TOPK_VALUE_IDX_SIZE];
            srcList.src4 = curValueIdxUb[3 * BASE_TOPK_VALUE_IDX_SIZE];
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            resInTmp = true;
        }


        LocalTensor<uint32_t> outIdxUb = ldOutIdxBuf_.Get<uint32_t>();
        QLIServiceVec::ExtractIndex(outIdxUb, (resInTmp ? tmpUb : curValueIdxUb).template ReinterpretCast<uint32_t>(), BASE_TOPK);
        LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        const bool skipIdxOut = ((treeMaskGmAddr_ & 1ULL) != 0ULL &&
                                 constInfo_.sparseCount == 2048U);
        if (!skipIdxOut) {
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopy(indiceOutGm[outOffset], idxULocal1, constInfo_.sparseCount);
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
#ifdef K112514_IDX_PIPE_V2
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
#else
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
#endif
        }
#endif
    }
}
}
#endif
