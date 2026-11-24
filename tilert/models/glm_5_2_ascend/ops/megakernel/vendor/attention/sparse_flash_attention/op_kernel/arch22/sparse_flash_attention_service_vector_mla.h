

#ifndef SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H
#define SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_flash_attention_common.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename SFAT>
class SFAVectorService {
public:

    using T = float;
    using KV_T = typename SFAT::kvType;
    using OUT_T = typename SFAT::outputType;
    using UPDATE_T = T;
    using MM1_OUT_T = float;
    using MM2_OUT_T = float;

    __aicore__ inline SFAVectorService(){};
    __aicore__ inline void ProcessVec1L(const RunInfo& sfaVecInfo);
    __aicore__ inline void ProcessVec2L(const RunInfo& sfaVecInfo);
    __aicore__ inline void InitBuffers(TPipe* pipe);
    __aicore__ inline void InitParams(const struct ConstInfo& sfaVecConstInfo,
                                      const SparseFlashAttentionTilingDataMla* __restrict tilingData);
    __aicore__ inline void InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm);
    __aicore__ inline void InitVec0GlobalTensor(const GlobalTensor<int32_t>& kvValidSizeGm,
                                                const GlobalTensor<KV_T>& kvMergeGm,
                                                const GlobalTensor<KV_T>& keyRopeGm, const GlobalTensor<KV_T>& keyGm,
                                                const GlobalTensor<int32_t>& blkTableGm);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm,
                                                GlobalTensor<int32_t> actualSeqLengthsQGm,
                                                GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm,
                                                GlobalTensor<T> lseSumFdGm, GlobalTensor<int32_t> topKGm,
                                                GlobalTensor<T> softmaxMaxGm, GlobalTensor<T> softmaxSumGm);
    __aicore__ inline void InitVec2GlobalTensor(GlobalTensor<T> accumOutGm, GlobalTensor<UPDATE_T> vec2ResGm,
                                                GlobalTensor<MM2_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void InitSoftmaxDefaultBuffer();
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    uint64_t treeMaskGmAddr_ = 0;
    __aicore__ inline bool IsTreeKeyValid(int64_t realS2Idx, const RunInfo& sfaVecRunInfo, int64_t s2IdLimit)
    {
        if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
            return false;
        }
        if (treeMaskGmAddr_ != 0) {
            const int64_t prefixLen = static_cast<int64_t>(sfaVecRunInfo.curActualSeqLenOri) - static_cast<int64_t>(sfaVecRunInfo.actS1Size);
            if (unlikely(realS2Idx >= prefixLen)) {
                const __gm__ uint32_t* maskPtr = reinterpret_cast<const __gm__ uint32_t*>(treeMaskGmAddr_);
                if (maskPtr[0] != 0U) {
                    const uint32_t qIdx = static_cast<uint32_t>(sfaVecRunInfo.gS1Idx / sfaVecConstInfo.gSize);
                    const uint32_t bit = static_cast<uint32_t>(realS2Idx - prefixLen);
                    if (qIdx < 32U && bit < 32U && ((maskPtr[qIdx] >> bit) & 1U) == 0U) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    __aicore__ inline void RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                   uint32_t sfaDealRows, uint32_t sfaColumns, uint32_t sfaActualColumns);
    __aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                   uint32_t sfaDealRows, uint32_t sfaColumns, uint32_t sfaActualColumns);

    __aicore__ inline void MergeKv(const RunInfo& sfaVecRunInfo);
    __aicore__ inline void MergeKvCoop(const RunInfo& sfaVecRunInfo, uint32_t dstCore, uint32_t halfIdx);
    __aicore__ inline int64_t GetKeyGmOffset(int64_t realS2Idx, const RunInfo& sfaVecRunInfo, int64_t s2IdLimit);
    __aicore__ inline int64_t GetKeyRopeGmOffset(int64_t realS2Idx, const RunInfo& sfaVecRunInfo, int64_t s2IdLimit);
    __aicore__ inline void GetRealS2Idx(int64_t s2GmOffset, int64_t& realS2Idx, int64_t topkGmBaseOffset,
                                        const RunInfo& sfaVecRunInfo);
    __aicore__ inline void CopyInKv(int64_t& mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx1,
                                    int64_t realS2Idx2, const RunInfo& sfaVecRunInfo);
    __aicore__ inline void CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size, int64_t s2StartGmOffset,
                                             int64_t mergeMte3Idx, const RunInfo& sfaVecRunInfo);
    __aicore__ inline void CopyInSingleKv(int64_t& mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                          int64_t keyBNBOffset, int64_t s2IdLimit, const RunInfo& sfaVecRunInfo);
    __aicore__ inline void SetInfInBlk(const LocalTensor<T>& mmResUb, uint32_t sfaDealRows, uint32_t sfaColumns,
                                       uint64_t startId, uint64_t endId);
    __aicore__ inline void SetMidInf(const LocalTensor<T>& mmResUb, uint32_t sfaDealRows, uint32_t sfaColumns,
                                     uint64_t startId, uint64_t endId);

    __aicore__ inline void ProcessVec1SingleBuf(const RunInfo& sfaVecInfo, const MSplitInfo& sfaVecSplitInfo);
    __aicore__ inline void DealBmm1ResBaseBlock(const RunInfo& sfaVecInfo, const MSplitInfo& sfaVecSplitInfo,
                                                uint32_t startRow, uint32_t sfaDealRows, uint32_t sfaColumns,
                                                uint32_t loopId);
    __aicore__ inline void SoftmaxFlashV2Compute(const RunInfo& sfaVecInfo, const MSplitInfo& sfaVecSplitInfo,
                                                 LocalTensor<T>& mmResUb, LocalTensor<uint8_t>& softmaxTmpUb,
                                                 uint32_t startRow, uint32_t sfaDealRows, uint32_t sfaColumns,
                                                 uint32_t sfaActualColumns);
    __aicore__ inline void AmlaVecCompute(const RunInfo& sfaVecInfo, const MSplitInfo& sfaVecSplitInfo,
                                          LocalTensor<T>& mmResUb, LocalTensor<uint8_t>& softmaxTmpUb,
                                          uint32_t startRow, uint32_t sfaDealRows, uint32_t sfaColumns,
                                          uint32_t sfaActualColumns);
    __aicore__ inline void ElewiseCompute(const RunInfo& sfaVecInfo, const LocalTensor<T>& mmResUb,
                                          uint32_t sfaDealRows, uint32_t sfaColumns);
    __aicore__ inline void ProcessAmlaNupdate(const RunInfo& sfaVecInfo, const MSplitInfo& sfaVecSplitInfo);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfo& sfaVecInfo, const MSplitInfo& sfaVecSplitInfo,
                                                       LocalTensor<T>& softmaxSumUb, LocalTensor<T>& softmaxMaxUb);
    __aicore__ inline void CopyFALseToGm(const RunInfo& sfaVecInfo, const MSplitInfo& sfaVecSplitInfo,
                                         LocalTensor<T>& softmaxSumUb, LocalTensor<T>& softmaxMaxUb);
    __aicore__ inline void SetBmm2FirstSInnerBias(const RunInfo& sfaVecInfo, const MSplitInfo& sfaVecSplitInfo);

    __aicore__ inline void ProcessVec2SingleBuf(const RunInfo& sfaVecInfo, const MSplitInfo& sfaVecSplitInfo);
    __aicore__ inline void DealBmm2ResBaseBlock(const RunInfo& sfaVecInfo, const MSplitInfo& sfaVecSplitInfo,
                                                uint32_t startRow, uint32_t sfaDealRows, uint32_t sfaColumns,
                                                uint32_t sfaActualColumns);
    __aicore__ inline void ProcessVec2Inner(const RunInfo& sfaVecInfo, const MSplitInfo& sfaVecSplitInfo,
                                            uint32_t mStartRow, uint32_t mDealSize);
    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfo& sfaVecInfo, LocalTensor<OUT_T>& attenOutUb,
                                                uint32_t sfaWorkspaceRow, uint32_t sfaDealRows, uint32_t sfaColumns,
                                                uint32_t sfaActualColumns);
    __aicore__ inline void Bmm2ResCopyOut(const RunInfo& sfaVecInfo, LocalTensor<T>& sfaBmm2ResultUb,
                                          uint32_t sfaWorkspaceRow, uint32_t sfaDealRows, uint32_t sfaColumns,
                                          uint32_t sfaActualColumns);
    __aicore__ inline void Bmm2CastAndCopyOut(const RunInfo& sfaVecInfo, LocalTensor<T>& sfaBmm2ResultUb,
                                              uint32_t sfaWorkspaceRow, uint32_t sfaDealRows, uint32_t sfaColumns,
                                              uint32_t sfaActualColumns);
    __aicore__ inline void Bmm2FDDataCopyOut(const RunInfo& sfaVecInfo, LocalTensor<T>& sfaBmm2ResultUb,
                                             uint32_t sfaWorkspaceRow, uint32_t sfaDealRows, uint32_t sfaColumns,
                                             uint32_t sfaActualColumns);
    __aicore__ inline uint64_t CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx);
    __aicore__ inline void GetConfusionTransposeTiling(int64_t numR, int64_t numC, const uint32_t stackBufferSize,
                                                       const uint32_t typeSize, ConfusionTransposeTiling& tiling);


    static constexpr uint64_t BYTE_BLOCK = 32UL;
    static constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;

    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);

    static constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;

private:
    static constexpr bool PAGE_ATTENTION = SFAT::pageAttention;
    static constexpr int TEMPLATE_MODE = SFAT::templateMode;
    static constexpr bool FLASH_DECODE = SFAT::flashDecode;
    static constexpr SFA_LAYOUT LAYOUT_T = SFAT::layout;
    static constexpr SFA_LAYOUT KV_LAYOUT_T = SFAT::kvLayout;

    static constexpr uint64_t MERGE_CACHE_GM_BUF_NUM = 4;
    static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_INPUT_BUF1_PONG_FLAG = 3;
    static constexpr uint64_t SYNC_INPUT_BUF2_FLAG = 4;
    static constexpr uint64_t SYNC_INPUT_BUF2_PONG_FLAG = 5;
    static constexpr uint64_t SYNC_OUTPUT_BUF1_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF2_FLAG = 5;
    static constexpr uint64_t SYNC_INPUT_V0BUF_FLAG = 6;
    static constexpr uint32_t INPUT1_BUFFER_OFFSET = ConstInfo::SFA_BUFFER_BYTES_32K;
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_OFFSET = ConstInfo::SFA_BUFFER_BYTES_1K;
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = ConstInfo::SFA_BUFFER_BYTES_32K / sizeof(T);
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);
    static constexpr T FLOAT_E_SCALAR = 8388608;
    static constexpr T LN2 = 0.6931471805599453094172;
    static constexpr T RECIP_OF_LN2 = 1 / LN2;
    static constexpr T SOFTMAX_MIN_NUM = -2e38;

    const SparseFlashAttentionTilingDataMla* __restrict tilingData;

    uint32_t pingpongFlag = 0U;
    ConstInfo sfaVecConstInfo = {};

    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;
    GlobalTensor<T> softmaxMaxGm;
    GlobalTensor<T> softmaxSumGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<UPDATE_T> vec2ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<int32_t> blkTableGm_;

    GlobalTensor<KV_T> kvMergeGm_;
    GlobalTensor<KV_T> keyRopeGm_;
    GlobalTensor<KV_T> keyGm_;
    GlobalTensor<int32_t> topkGm_;
    GlobalTensor<int32_t> kvValidSizeGm_;
    GlobalTensor<KV_T> kvMergeBaseGm_;
    GlobalTensor<int32_t> kvValidSizeBaseGm_;
    GlobalTensor<MM1_OUT_T> mm1ResBaseGm_;


    TBuf<> inputBuff1;
    TBuf<> inputBuff2;
    TBuf<> outputBuff1;
    TBuf<> outputBuff2;

    TBuf<> tmpBuff1;
    TBuf<> v0ValidSizeBuff;

    TBuf<> nValueBuff;
    TBuf<> cofValueBuff;
    TBuf<> aMlaSumBuff;
    TBuf<> softmaxMaxBuff;
    TBuf<> softmaxExpBuff;
    TBuf<> softmaxSumBuff;
    TBuf<> softmaxMaxDefaultBuff;
    TBuf<> softmaxSumDefaultBuff;

    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    LocalTensor<T> nValueUb;
    LocalTensor<T> cofValueUb;
    LocalTensor<T> aMlaSumUb;
    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;
    LocalTensor<KV_T> kvMergUb_;
    LocalTensor<KV_T> ropeMergUb_;
    LocalTensor<int32_t> v0ValidSizeUb_;
};

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::InitBuffers(TPipe* pipe)
{
    pipe->InitBuffer(inputBuff1, ConstInfo::SFA_BUFFER_BYTES_32K * 2);
    pipe->InitBuffer(inputBuff2, ConstInfo::SFA_BUFFER_BYTES_8K * 2);
    pipe->InitBuffer(outputBuff1, ConstInfo::SFA_BUFFER_BYTES_32K);
    pipe->InitBuffer(outputBuff2, ConstInfo::SFA_BUFFER_BYTES_4K);

    pipe->InitBuffer(tmpBuff1, ConstInfo::SFA_BUFFER_BYTES_32K);
    pipe->InitBuffer(v0ValidSizeBuff, ConstInfo::SFA_BUFFER_BYTES_8K);


    pipe->InitBuffer(nValueBuff, ConstInfo::SFA_BUFFER_BYTES_1K * sfaVecConstInfo.preLoadNum);
    pipe->InitBuffer(cofValueBuff, ConstInfo::SFA_BUFFER_BYTES_1K * sfaVecConstInfo.preLoadNum);
    pipe->InitBuffer(aMlaSumBuff, ConstInfo::SFA_BUFFER_BYTES_1K * sfaVecConstInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxBuff, ConstInfo::SFA_BUFFER_BYTES_1K * sfaVecConstInfo.preLoadNum);
    pipe->InitBuffer(softmaxExpBuff, ConstInfo::SFA_BUFFER_BYTES_1K * sfaVecConstInfo.preLoadNum);
    pipe->InitBuffer(softmaxSumBuff, ConstInfo::SFA_BUFFER_BYTES_1K * sfaVecConstInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxDefaultBuff, ConstInfo::SFA_BUFFER_BYTES_1K);
    pipe->InitBuffer(softmaxSumDefaultBuff, ConstInfo::SFA_BUFFER_BYTES_1K);

    nValueUb = nValueBuff.Get<T>();
    cofValueUb = cofValueBuff.Get<T>();
    aMlaSumUb = aMlaSumBuff.Get<T>();

    softmaxMaxUb = softmaxMaxBuff.Get<T>();
    softmaxSumUb = softmaxSumBuff.Get<T>();
    softmaxExpUb = softmaxExpBuff.Get<T>();

    softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
    softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();

    kvMergUb_ = inputBuff1.Get<KV_T>();
    ropeMergUb_ = inputBuff2.Get<KV_T>();

    v0ValidSizeUb_ = v0ValidSizeBuff.Get<int32_t>();
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::InitParams(
    const struct ConstInfo& sfaVecConstInfo, const SparseFlashAttentionTilingDataMla* __restrict tilingData)
{
    this->sfaVecConstInfo = sfaVecConstInfo;
    this->tilingData = tilingData;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm)
{
    this->mm2ResInt32Gm = mm2ResInt32Gm;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::InitVec0GlobalTensor(const GlobalTensor<int32_t>& kvValidSizeGm,
                                                                    const GlobalTensor<KV_T>& kvMergeGm,
                                                                    const GlobalTensor<KV_T>& keyRopeGm,
                                                                    const GlobalTensor<KV_T>& keyGm,
                                                                    const GlobalTensor<int32_t>& blkTableGm)
{
    this->kvMergeGm_ = kvMergeGm;
    this->keyRopeGm_ = keyRopeGm;
    this->keyGm_ = keyGm;
    this->blkTableGm_ = blkTableGm;
    this->kvValidSizeGm_ = kvValidSizeGm;
    uint64_t coreIdx = GetBlockIdx() / 2;
    this->kvMergeBaseGm_.SetGlobalBuffer(reinterpret_cast<__gm__ KV_T*>(kvMergeGm.GetPhyAddr(0)) -
                                         coreIdx * 512ULL * 576ULL * 4ULL);
    this->kvValidSizeBaseGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(kvValidSizeGm.GetPhyAddr(0)) -
                                             coreIdx * 2ULL * 128ULL * 4ULL);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm, GlobalTensor<int32_t> actualSeqLengthsQGm,
    GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm, GlobalTensor<T> lseSumFdGm,
    GlobalTensor<int32_t> topKGm, GlobalTensor<T> softmaxMaxGm, GlobalTensor<T> softmaxSumGm)
{
    this->mm1ResGm = mm1ResGm;
    uint64_t coreIdx = GetBlockIdx() / 2;
    this->mm1ResBaseGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MM1_OUT_T*>(mm1ResGm.GetPhyAddr(0)) -
                                        coreIdx * 2ULL * sfaVecConstInfo.mmResUbSize);
    this->vec1ResGm = vec1ResGm;
    this->actualSeqLengthsQGm = actualSeqLengthsQGm;
    this->actualSeqLengthsKVGm = actualSeqLengthsKVGm;
    this->lseMaxFdGm = lseMaxFdGm;
    this->lseSumFdGm = lseSumFdGm;
    this->topkGm_ = topKGm;
    this->softmaxMaxGm = softmaxMaxGm;
    this->softmaxSumGm = softmaxSumGm;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::InitVec2GlobalTensor(GlobalTensor<T> accumOutGm,
                                                                    GlobalTensor<UPDATE_T> vec2ResGm,
                                                                    GlobalTensor<MM2_OUT_T> mm2ResGm,
                                                                    GlobalTensor<OUT_T> attentionOutGm)
{
    this->accumOutGm = accumOutGm;
    this->vec2ResGm = vec2ResGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::InitSoftmaxDefaultBuffer()
{
    Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
    Duplicate(softmaxSumDefaultUb, ConstInfo::FLOAT_ZERO, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::CopyFALseToGm(const RunInfo& sfaVecInfo,
                                                             const MSplitInfo& sfaVecSplitInfo,
                                                             LocalTensor<T>& softmaxSumUb, LocalTensor<T>& softmaxMaxUb)

{
    if (sfaVecSplitInfo.vecDealM == 0) {
        return;
    }
    uint64_t baseOffset = sfaVecSplitInfo.nBufferStartM / 2;
    size_t size = sfaVecSplitInfo.vecDealM;

    int64_t offset = 0;
    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        uint64_t actualSeqQTotal =
            (sfaVecInfo.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(sfaVecConstInfo.batchSize - 1);
        uint64_t actualSeqQPrefixSum = (sfaVecInfo.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(sfaVecInfo.bIdx - 1);
        offset += sfaVecInfo.n2Idx * actualSeqQTotal * sfaVecConstInfo.gSize +
                  (actualSeqQPrefixSum + sfaVecInfo.gS1Idx / sfaVecConstInfo.gSize) * sfaVecConstInfo.gSize +
                  sfaVecSplitInfo.nBufferStartM + sfaVecSplitInfo.vecStartM;
    } else {
        offset += sfaVecInfo.bIdx * sfaVecConstInfo.kvHeadNum * sfaVecConstInfo.qSeqSize * sfaVecConstInfo.gSize +
                  sfaVecInfo.n2Idx * sfaVecConstInfo.qSeqSize * sfaVecConstInfo.gSize +
                  sfaVecInfo.gS1Idx / sfaVecConstInfo.gSize * sfaVecConstInfo.gSize + sfaVecSplitInfo.nBufferStartM +
                  sfaVecSplitInfo.vecStartM;
    }

    if (sfaVecInfo.actualSingleProcessSInnerSize != 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = sizeof(T) * size;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        size_t alignedSize = (sizeof(T) * size + 31) / 32 * 32 / sizeof(T);
        LocalTensor<T> tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(tmp, softmaxMaxUb[baseOffset], alignedSize);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopyPad(softmaxMaxGm[offset], tmp, dataCopyParams);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);

        tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(tmp, softmaxSumUb[baseOffset], alignedSize);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopyPad(softmaxSumGm[offset], tmp, dataCopyParams);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    } else {
        matmul::InitOutput<T>(softmaxSumGm[offset], size, ConstInfo::FLOAT_ZERO);
        matmul::InitOutput<T>(softmaxMaxGm[offset], size, SOFTMAX_MIN_NUM);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ComputeLogSumExpAndCopyToGm(const RunInfo& sfaVecInfo,
                                                                           const MSplitInfo& sfaVecSplitInfo,
                                                                           LocalTensor<T>& softmaxSumUb,
                                                                           LocalTensor<T>& softmaxMaxUb)
{
    if (sfaVecSplitInfo.vecDealM == 0) {
        return;
    }
    uint64_t baseOffset = sfaVecSplitInfo.nBufferStartM / 2;
    size_t size = sfaVecSplitInfo.vecDealM * FP32_BLOCK_ELEMENT_NUM;
    uint64_t sfaAccumOutputCount = CalcAccumOffset(sfaVecInfo.bIdx, sfaVecInfo.gS1Idx);
    uint64_t offset =
        (sfaAccumOutputCount * sfaVecConstInfo.kvHeadNum * sfaVecConstInfo.mBaseSize +
         sfaVecInfo.tndCoreStartKVSplitPos * sfaVecConstInfo.kvHeadNum * sfaVecConstInfo.mBaseSize +
         sfaVecSplitInfo.nBufferStartM + sfaVecSplitInfo.vecStartM) *
        FP32_BLOCK_ELEMENT_NUM;
    if (sfaVecInfo.actualSingleProcessSInnerSize != 0) {
        LocalTensor<T> tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(tmp, softmaxSumUb[baseOffset], (sfaVecSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseSumFdGm[offset], tmp, size);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);

        tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(tmp, softmaxMaxUb[baseOffset], (sfaVecSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseMaxFdGm[offset], tmp, size);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    } else {
        matmul::InitOutput<T>(lseSumFdGm[offset], size, ConstInfo::FLOAT_ZERO);
        matmul::InitOutput<T>(lseMaxFdGm[offset], size, SOFTMAX_MIN_NUM);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ElewiseCompute(const RunInfo& sfaVecInfo, const LocalTensor<T>& mmResUb,
                                                              uint32_t sfaDealRows, uint32_t sfaColumns)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), sfaDealRows * sfaColumns);
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {

        uint64_t s2ValidSizeFirstPart = v0ValidSizeUb_.GetValue(128 + sfaVecInfo.loop % MERGE_CACHE_GM_BUF_NUM);
        uint64_t s2ValidSizeSecondPart = v0ValidSizeUb_.GetValue(256 + sfaVecInfo.loop % MERGE_CACHE_GM_BUF_NUM);

        int64_t s2ProcessSize = sfaVecInfo.actualSingleProcessSInnerSize;
        int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * sfaVecConstInfo.sparseBlockSize);
        int64_t s2Mid = CeilDiv(s2Pair, 2L) * 2 * sfaVecConstInfo.sparseBlockSize;
        if (s2Mid > s2ProcessSize) {
            s2Mid = s2ProcessSize;
        }
        if (unlikely(s2ValidSizeFirstPart < s2Mid)) {
            int64_t s2StartCeilAlign = CeilAlign(s2ValidSizeFirstPart, 8);
            int64_t s2MidFloorAlign = s2Mid / 8 * 8;


            SetInfInBlk(mmResUb, sfaDealRows, sfaColumns, s2ValidSizeFirstPart,
                        s2StartCeilAlign >= s2Mid ? s2Mid : s2StartCeilAlign);
            SetMidInf(mmResUb, sfaDealRows, sfaColumns, s2StartCeilAlign, s2MidFloorAlign);
            SetInfInBlk(mmResUb, sfaDealRows, sfaColumns,
                        s2StartCeilAlign <= s2MidFloorAlign ? s2MidFloorAlign : s2StartCeilAlign, s2Mid);
        }
        if (unlikely(s2ValidSizeSecondPart < s2ProcessSize - s2Mid)) {


            int64_t s2StartCeilAlign = CeilAlign(s2Mid + s2ValidSizeSecondPart, 8);
            int64_t s2EndFloorAlign = s2ProcessSize / 8 * 8;
            SetInfInBlk(mmResUb, sfaDealRows, sfaColumns, s2Mid + s2ValidSizeSecondPart,
                        s2StartCeilAlign >= s2ProcessSize ? s2ProcessSize : s2StartCeilAlign);
            SetMidInf(mmResUb, sfaDealRows, sfaColumns, s2StartCeilAlign, s2EndFloorAlign);
            SetInfInBlk(mmResUb, sfaDealRows, sfaColumns,
                        s2StartCeilAlign <= s2EndFloorAlign ? s2EndFloorAlign : s2StartCeilAlign, s2ProcessSize);
        }
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::SetInfInBlk(const LocalTensor<T>& mmResUb, uint32_t sfaDealRows,
                                                           uint32_t sfaColumns, uint64_t startId, uint64_t endId)
{


    if (startId >= endId) {
        return;
    }

    uint64_t startFloorAlignSize = startId / BLOCK_ELEMENT_NUM * BLOCK_ELEMENT_NUM;
    uint64_t notComputePreMaskOneBlk = (1 << (startId - startFloorAlignSize)) - 1;
    uint64_t notComputePostMaskOneBlk = ~((1 << (endId - startFloorAlignSize)) - 1);
    uint64_t notComputeMaskOneBlk = notComputePreMaskOneBlk ^ notComputePostMaskOneBlk;

    uint64_t maskOneBlk = ~notComputeMaskOneBlk;
    uint64_t mask[1] = {maskOneBlk};
    for (int i = 1; i < 8; i++) {
        mask[0] = mask[0] | (maskOneBlk << (i * 8));
    }
    for (uint64_t rowId = 0; rowId < sfaDealRows; rowId += 8) {
        Duplicate(mmResUb[rowId * sfaColumns + startFloorAlignSize], SOFTMAX_MIN_NUM, mask, 1, CeilDiv(sfaColumns, 8),
                  0);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::SetMidInf(const LocalTensor<T>& mmResUb, uint32_t sfaDealRows,
                                                         uint32_t sfaColumns, uint64_t startId, uint64_t endId)
{
    if (startId >= endId) {
        return;
    }


    for (uint64_t rowId = 0; rowId < sfaDealRows; rowId++) {
        Duplicate(mmResUb[rowId * sfaColumns + startId], SOFTMAX_MIN_NUM, endId - startId);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::SoftmaxFlashV2Compute(const RunInfo& sfaVecInfo,
                                                                     const MSplitInfo& sfaVecSplitInfo,
                                                                     LocalTensor<T>& mmResUb,
                                                                     LocalTensor<uint8_t>& softmaxTmpUb,
                                                                     uint32_t startRow, uint32_t sfaDealRows,
                                                                     uint32_t sfaColumns, uint32_t sfaActualColumns)
{
    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t baseOffset = sfaVecSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t outIdx = sfaVecInfo.loop % (sfaVecConstInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    if (sfaVecInfo.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (sfaVecInfo.loop - 1) % (sfaVecConstInfo.preLoadNum);
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
    }
    if (sfaActualColumns != 0) {
        SoftMaxShapeInfo srcShape{sfaDealRows, sfaColumns, sfaDealRows, sfaActualColumns};
        SoftMaxTiling newTiling =
            SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);
        SoftmaxFlashV2<T, true, true, false, false, SFA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
            mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
            softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    } else {
        uint32_t dealRowCountAlign = SFAAlign(sfaDealRows, FP32_BLOCK_ELEMENT_NUM);
        DataCopy(softmaxSumUb[softmaxOutOffset], inSumTensor, dealRowCountAlign);
        PipeBarrier<PIPE_V>();
        DataCopy(softmaxMaxUb[softmaxOutOffset], inMaxTensor, dealRowCountAlign);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::AmlaVecCompute(const RunInfo& sfaVecInfo,
                                                              const MSplitInfo& sfaVecSplitInfo,
                                                              LocalTensor<T>& mmResUb,
                                                              LocalTensor<uint8_t>& softmaxTmpUb, uint32_t startRow,
                                                              uint32_t sfaDealRows, uint32_t sfaColumns,
                                                              uint32_t sfaActualColumns)
{
    uint32_t baseOffset = sfaVecSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t calCount = sfaDealRows;
    uint32_t outIdx = sfaVecInfo.loop % (sfaVecConstInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;

    LocalTensor<T> nTmp = softmaxTmpUb.template ReinterpretCast<T>();
    LocalTensor<T> nUpdateTmp = nTmp[SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Muls(nTmp, softmaxMaxUb[softmaxOutOffset], ((T)(-1.0)) * RECIP_OF_LN2, calCount);

    PipeBarrier<PIPE_V>();
    Cast(nTmp, nTmp, RoundMode::CAST_ROUND, calCount);
    PipeBarrier<PIPE_V>();

    uint32_t prOutIdx = (sfaVecInfo.loop - 1) % (sfaVecConstInfo.preLoadNum);
    uint32_t PreSoftmaxOutOffset = prOutIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;

    if (sfaVecInfo.isFirstSInnerLoop) {
        Duplicate(nUpdateTmp, ConstInfo::FLOAT_ZERO, calCount);
    } else {
        Sub(nUpdateTmp, nTmp, nValueUb[PreSoftmaxOutOffset], calCount);
    }
    PipeBarrier<PIPE_V>();

    Adds(nValueUb[softmaxOutOffset], nTmp, ConstInfo::FLOAT_ZERO, calCount);
    PipeBarrier<PIPE_V>();


    LocalTensor<T> nUpdateTmp2 = nTmp[2 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    LocalTensor<KV_T> nTmp_KvT = nTmp[3 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)].template ReinterpretCast<KV_T>();
    LocalTensor<T> tmpCofUb = nTmp[4 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    LocalTensor<T> epsUb = nTmp[5 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Muls(nUpdateTmp2, softmaxMaxUb[softmaxOutOffset], RECIP_OF_LN2, calCount);
    PipeBarrier<PIPE_V>();
    Add(nTmp, nUpdateTmp2, nTmp, calCount);
    PipeBarrier<PIPE_V>();
    Muls(nTmp, nTmp, LN2, calCount);
    PipeBarrier<PIPE_V>();
    Exp(nTmp, nTmp, calCount);
    PipeBarrier<PIPE_V>();
    Cast(nTmp_KvT, nTmp, RoundMode::CAST_ROUND, calCount);
    PipeBarrier<PIPE_V>();
    Cast(nUpdateTmp2, nTmp_KvT, RoundMode::CAST_NONE, calCount);
    PipeBarrier<PIPE_V>();
    if (sfaVecInfo.s2Idx + 1 == sfaVecInfo.curSInnerLoopTimes) {
        Mul(aMlaSumUb[softmaxOutOffset], softmaxSumUb[softmaxOutOffset], nUpdateTmp2, calCount);
    }
    if (sfaActualColumns == 0) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        return;
    }
    LocalTensor<T> nTmp3 = nTmp[6 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Brcb(nTmp3, nUpdateTmp2, (sfaDealRows + 7) / 8, {1, 8});
    PipeBarrier<PIPE_V>();
    RowMuls(mmResUb, mmResUb, nTmp3, sfaDealRows, sfaColumns, sfaActualColumns);

    Div(tmpCofUb, nTmp, nUpdateTmp2, calCount);
    if (sfaVecInfo.isFirstSInnerLoop) {
        Duplicate(cofValueUb[softmaxOutOffset], (T)1.0, calCount);
        PipeBarrier<PIPE_V>();
        Div(epsUb, cofValueUb[softmaxOutOffset], tmpCofUb, calCount);
    } else {
        PipeBarrier<PIPE_V>();
        Div(epsUb, cofValueUb[PreSoftmaxOutOffset], tmpCofUb, calCount);
    }
    PipeBarrier<PIPE_V>();

    Adds(cofValueUb[softmaxOutOffset], tmpCofUb, ConstInfo::FLOAT_ZERO, calCount);
    Adds(epsUb, epsUb, (T)(-1.0), calCount);
    PipeBarrier<PIPE_V>();
    Muls(epsUb, epsUb, (T)1.5, calCount);

    Maxs(nUpdateTmp, nUpdateTmp, (T)(-30.0), calCount);
    PipeBarrier<PIPE_V>();
    Adds(epsUb, epsUb, (T)(0.000001), calCount);
    PipeBarrier<PIPE_V>();
    Add(nUpdateTmp, nUpdateTmp, epsUb, calCount);
    PipeBarrier<PIPE_V>();
    Muls(nUpdateTmp, nUpdateTmp, FLOAT_E_SCALAR, calCount);
    PipeBarrier<PIPE_V>();


    LocalTensor<int32_t> tmQue = outputBuff2.Get<int32_t>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    LocalTensor<int32_t> nInt32Out = tmQue[startRow];

    Cast(nInt32Out, nUpdateTmp, RoundMode::CAST_ROUND, sfaDealRows);
    PipeBarrier<PIPE_V>();

    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::DealBmm1ResBaseBlock(const RunInfo& sfaVecInfo,
                                                                    const MSplitInfo& sfaVecSplitInfo,
                                                                    uint32_t startRow, uint32_t sfaDealRows,
                                                                    uint32_t sfaColumns, uint32_t loopId)
{
    uint32_t computeSize = sfaDealRows * sfaColumns;
    uint64_t inOutGmOffset = (sfaVecInfo.loop % sfaVecConstInfo.preLoadNum) * sfaVecConstInfo.mmResUbSize +
                             (sfaVecSplitInfo.nBufferStartM + sfaVecSplitInfo.vecStartM + startRow) * sfaColumns;
    LocalTensor<MM1_OUT_T> mmResUb = inputBuff1.Get<MM1_OUT_T>();
    mmResUb = mmResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM1_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    if (sfaVecInfo.useCoopMm1) {
        uint64_t coopMm1Offset = static_cast<uint64_t>(sfaVecInfo.coopMm1Slot) * sfaVecConstInfo.mmResUbSize +
                                 (sfaVecSplitInfo.nBufferStartM + sfaVecSplitInfo.vecStartM + startRow) * sfaColumns;
        DataCopy(mmResUb, mm1ResBaseGm_[coopMm1Offset], computeSize);
    } else {
        DataCopy(mmResUb, mm1ResGm[inOutGmOffset], computeSize);
    }
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        if (loopId == 0) {
            WaitFlag<HardEvent::MTE2_S>(0);
        }
    }
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    ElewiseCompute(sfaVecInfo, mmResUb, sfaDealRows, sfaColumns);

    PipeBarrier<PIPE_V>();
    LocalTensor<T> tmpAFloorUb = tmpBuff1.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();

    SoftmaxFlashV2Compute(sfaVecInfo, sfaVecSplitInfo, mmResUb, softmaxTmpUb, startRow, sfaDealRows, sfaColumns,
                          sfaVecInfo.actualSingleProcessSInnerSize);

    PipeBarrier<PIPE_V>();
    AmlaVecCompute(sfaVecInfo, sfaVecSplitInfo, mmResUb, softmaxTmpUb, startRow, sfaDealRows, sfaColumns,
                   sfaVecInfo.actualSingleProcessSInnerSize);

    PipeBarrier<PIPE_V>();
    LocalTensor<KV_T> tmpMMResCastTensor = outputBuff1.Get<KV_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);

    Cast(tmpMMResCastTensor, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(vec1ResGm[inOutGmOffset], tmpMMResCastTensor, computeSize);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::SetBmm2FirstSInnerBias(const RunInfo& sfaVecInfo,
                                                                      const MSplitInfo& sfaVecSplitInfo)
{
    uint32_t mSplitSize = 16U;
    uint64_t baseoffset = (sfaVecInfo.bn2IdxInCurCore % sfaVecConstInfo.preLoadNum) * sfaVecConstInfo.bmm2ResUbSize +
                          (sfaVecSplitInfo.nBufferStartM + sfaVecSplitInfo.vecStartM) * sfaVecConstInfo.headDim;
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    LocalTensor<int32_t> tmpTensor = outputBuff1.Get<int32_t>();
#ifdef K112130_SPLIT_HEADS_4_4
    if (sfaVecSplitInfo.vecDealM == 4U && sfaVecSplitInfo.nBufferDealM == 8U && sfaVecConstInfo.headDim == 512U) {
        const uint32_t initRows = (sfaVecSplitInfo.vecStartM == 0U) ? 4U : 12U;
        Duplicate(tmpTensor, static_cast<int32_t>(394264576), initRows * sfaVecConstInfo.headDim);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        DataCopy(mm2ResInt32Gm[baseoffset], tmpTensor, initRows * sfaVecConstInfo.headDim);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        return;
    }
#endif
    Duplicate(tmpTensor, static_cast<int32_t>(394264576),
              mSplitSize * sfaVecConstInfo.headDim);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    uint32_t loopCount = (sfaVecSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    for (uint32_t loop = 0; loop < loopCount; loop++) {
        DataCopy(mm2ResInt32Gm[baseoffset + loop * mSplitSize * sfaVecConstInfo.headDim], tmpTensor,
                 mSplitSize * sfaVecConstInfo.headDim);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ProcessAmlaNupdate(const RunInfo& sfaVecInfo,
                                                                  const MSplitInfo& sfaVecSplitInfo)
{
    if (sfaVecSplitInfo.vecDealM == 0) {
        return;
    }
    if (sfaVecInfo.isFirstSInnerLoop) {
        SetBmm2FirstSInnerBias(sfaVecInfo, sfaVecSplitInfo);
        return;
    }

    LocalTensor<int32_t> nUpdateTensor = outputBuff2.Get<int32_t>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);

    constexpr uint32_t dGroupSize = 128U;
    constexpr uint32_t mSplitSize =
        64U;
    constexpr uint32_t ONE_BLOCK_SIZE = 32U;

#ifdef K112130_SPLIT_HEADS_4_4
    uint32_t subMSize = (sfaVecSplitInfo.vecDealM == 4U && sfaVecSplitInfo.nBufferDealM == 8U &&
                         sfaVecConstInfo.headDim == 512U)
                            ? 4U
                            : SFAAlign(sfaVecSplitInfo.vecDealM, 16U);
#else
    uint32_t subMSize = SFAAlign(sfaVecSplitInfo.vecDealM, 16U);
#endif
    uint16_t elementPerBlock = ONE_BLOCK_SIZE / sizeof(int32_t);
    uint32_t loopCount = (subMSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = subMSize - (loopCount - 1) * mSplitSize;

    for (uint32_t loop = 0, processMSize = mSplitSize; loop < loopCount; loop++) {
        if (loop == (loopCount - 1)) {
            processMSize = tailSplitSize;
        }
        LocalTensor<int32_t> tmpQue = outputBuff1.Get<int32_t>();

        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);

        for (uint32_t i = 0; i < dGroupSize / elementPerBlock; i++) {
            Brcb(tmpQue[i * elementPerBlock], nUpdateTensor[loop * mSplitSize],
                 static_cast<uint8_t>((processMSize + elementPerBlock - 1) / elementPerBlock),
                 {static_cast<uint16_t>(
                      dGroupSize / elementPerBlock),
                  static_cast<uint16_t>(dGroupSize)});
        }

        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);

        uint64_t baseoffset =
            (sfaVecInfo.bn2IdxInCurCore % sfaVecConstInfo.preLoadNum) * sfaVecConstInfo.bmm2ResUbSize +
            (sfaVecSplitInfo.nBufferStartM + sfaVecSplitInfo.vecStartM + loop * mSplitSize) * sfaVecConstInfo.headDim;

        SetAtomicAdd<int32_t>();
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = static_cast<uint16_t>(processMSize);
        dataCopyParams.blockLen = dGroupSize * sizeof(int32_t) / ONE_BLOCK_SIZE;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = static_cast<uint16_t>((sfaVecConstInfo.headDim - dGroupSize) * sizeof(int32_t) /
                                                         ONE_BLOCK_SIZE);
        for (uint32_t i = 0; i < sfaVecConstInfo.headDim / dGroupSize; i++) {
            DataCopy(mm2ResInt32Gm[baseoffset + i * dGroupSize], tmpQue, dataCopyParams);
        }
        SetAtomicNone();
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ProcessVec1SingleBuf(const RunInfo& sfaVecInfo,
                                                                    const MSplitInfo& sfaVecSplitInfo)
{
    if (sfaVecSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t mSplitSize = sfaVecInfo.actualSingleProcessSInnerSize == 0 ?
                              16 :
                              BASE_BLOCK_MAX_ELEMENT_NUM / sfaVecInfo.actualSingleProcessSInnerSizeAlign;


    mSplitSize = mSplitSize / 8 * 8;

    if (mSplitSize > sfaVecSplitInfo.vecDealM) {
        mSplitSize = sfaVecSplitInfo.vecDealM;
    }
    uint32_t loopCount = (sfaVecSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = sfaVecSplitInfo.vecDealM - (loopCount - 1) * mSplitSize;

    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        DataCopy(v0ValidSizeUb_[128],
                 (sfaVecInfo.useCoopKv ?
                      kvValidSizeBaseGm_[static_cast<uint64_t>(sfaVecInfo.coopKvSlot) * (128 * 2)] :
                      kvValidSizeGm_[sfaVecInfo.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2)]),
                 256);
        SetFlag<HardEvent::MTE2_S>(0);
        if (unlikely(loopCount == 0)) {

            WaitFlag<HardEvent::MTE2_S>(0);
        }
    }
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm1ResBaseBlock(sfaVecInfo, sfaVecSplitInfo, i * mSplitSize, dealSize,
                             sfaVecInfo.actualSingleProcessSInnerSizeAlign, i);
        pingpongFlag ^= 1;
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::GetRealS2Idx(int64_t s2GmOffset, int64_t& realS2Idx,
                                                            int64_t topkGmBaseOffset, const RunInfo& sfaVecRunInfo)
{
    int64_t topkGmIdx =
        (s2GmOffset + sfaVecRunInfo.s2Idx * sfaVecConstInfo.s2BaseSize) / sfaVecConstInfo.sparseBlockSize;
    if (unlikely(topkGmIdx >= sfaVecConstInfo.sparseBlockCount)) {
        realS2Idx = -1;
        return;
    }
    realS2Idx = topkGm_.GetValue(topkGmBaseOffset + topkGmIdx) * static_cast<int64_t>(sfaVecConstInfo.sparseBlockSize) +
                static_cast<int64_t>((s2GmOffset + sfaVecRunInfo.s2Idx * sfaVecConstInfo.s2BaseSize) %
                                     sfaVecConstInfo.sparseBlockSize);
}

template <typename SFAT>
__aicore__ inline int64_t SFAVectorService<SFAT>::GetKeyGmOffset(int64_t realS2Idx, const RunInfo& sfaVecRunInfo,
                                                                 int64_t s2IdLimit)
{
    if (!IsTreeKeyValid(realS2Idx, sfaVecRunInfo, s2IdLimit)) {
        return -1;
    }
    int64_t realKeyGmOffset = 0;
    if constexpr (PAGE_ATTENTION) {
        int64_t blkTableIdx = realS2Idx / sfaVecConstInfo.kvCacheBlockSize;
        int64_t blkTableOffset = realS2Idx % sfaVecConstInfo.kvCacheBlockSize;
        realKeyGmOffset = blkTableGm_.GetValue(sfaVecRunInfo.bIdx * sfaVecConstInfo.maxBlockNumPerBatch + blkTableIdx) *
                              static_cast<int64_t>(sfaVecConstInfo.kvCacheBlockSize) *
                              static_cast<int64_t>(sfaVecConstInfo.kvHeadNum) +
                          blkTableOffset;
    } else {
        realKeyGmOffset =
            (sfaVecRunInfo.tensorBOffset + realS2Idx * sfaVecConstInfo.kvHeadNum * sfaVecConstInfo.headDim) /
            sfaVecConstInfo.headDim;
    }
    return realKeyGmOffset;
}

template <typename SFAT>
__aicore__ inline int64_t SFAVectorService<SFAT>::GetKeyRopeGmOffset(int64_t realS2Idx, const RunInfo& sfaVecRunInfo,
                                                                     int64_t s2IdLimit)
{
    if (!IsTreeKeyValid(realS2Idx, sfaVecRunInfo, s2IdLimit)) {
        return -1;
    }
    if constexpr (!SFAT::hasRope) {
        return -1;
    }
    int64_t realKeyRopeGmOffset = 0;
    realKeyRopeGmOffset =
        (sfaVecRunInfo.tensorBRopeOffset + realS2Idx * sfaVecConstInfo.kvHeadNum * sfaVecConstInfo.headDimRope) /
        sfaVecConstInfo.headDimRope;
    return realKeyRopeGmOffset;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::CopyInSingleKv(int64_t& mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
                                                              int64_t realS2Idx, int64_t keyBNBOffset,
                                                              int64_t s2IdLimit, const RunInfo& sfaVecRunInfo)
{
    if (keyBNBOffset < 0) {
        return;
    }
    int64_t validS2Count = (realS2Idx + sfaVecConstInfo.sparseBlockSize > s2IdLimit ? s2IdLimit - realS2Idx :
                                                                                      sfaVecConstInfo.sparseBlockSize);
    if (validS2Count == 1 && sfaVecConstInfo.headDim == 512 && sfaVecConstInfo.headDimRope == 64) {
        DataCopy(kvMergUb_[(mergeMte3Idx & 1) * 32 * 512 + (mte2Size - mte3Size) * 512],
                 keyGm_[keyBNBOffset * 512], 512);
        if constexpr (SFAT::hasRope) {
            DataCopy(ropeMergUb_[(mergeMte3Idx & 1) * 32 * 64 + (mte2Size - mte3Size) * 64],
                     keyRopeGm_[keyBNBOffset * 64], 64);
        }
        mte2Size += 1;
        return;
    }
    DataCopyExtParams intriParams;
    intriParams.blockLen = validS2Count * sfaVecConstInfo.headDim * sizeof(KV_T);
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * 32 * 512 + (mte2Size - mte3Size) * sfaVecConstInfo.headDim],
                keyGm_[keyBNBOffset * sfaVecConstInfo.headDim], intriParams, padParams);
    if constexpr (!SFAT::hasRope) {
        mte2Size += validS2Count;
        return;
    }
    intriParams.blockLen = validS2Count * sfaVecConstInfo.headDimRope * sizeof(KV_T);

    DataCopyPad(ropeMergUb_[mergeMte3Idx % 2 * 32 * 64 + (mte2Size - mte3Size) * sfaVecConstInfo.headDimRope],
                keyRopeGm_[keyBNBOffset * sfaVecConstInfo.headDimRope], intriParams, padParams);
    mte2Size += validS2Count;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::CopyInKv(int64_t& mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
                                                        int64_t realS2Idx1, int64_t realS2Idx2,
                                                        const RunInfo& sfaVecRunInfo)
{
    int64_t s2IdLimit = sfaVecRunInfo.curActualSeqLenOri;
    if (sfaVecConstInfo.sparseMode == 3) {
        s2IdLimit = sfaVecRunInfo.curActualSeqLenOri - sfaVecRunInfo.actS1Size +
                    sfaVecRunInfo.gS1Idx / sfaVecConstInfo.gSize + 1;
    }

    int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, sfaVecRunInfo, s2IdLimit);
    int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, sfaVecRunInfo, s2IdLimit);
    if (unlikely(keyOffset1 < 0 && keyOffset2 < 0)) {
        return;
    }

    int64_t keySrcStride = 0;
    int64_t keyRopeSrcStride = 0;
    if constexpr (PAGE_ATTENTION) {
        int64_t blkTableSrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) : (keyOffset2 - keyOffset1)) -
                                     sfaVecConstInfo.sparseBlockSize);
        keySrcStride = blkTableSrcStride * sfaVecConstInfo.headDim * sizeof(KV_T);
        keyRopeSrcStride = blkTableSrcStride * sfaVecConstInfo.headDimRope * sizeof(KV_T);
    } else {
        int64_t keyRopeOffset1 = GetKeyRopeGmOffset(realS2Idx1, sfaVecRunInfo, s2IdLimit);
        int64_t keyRopeOffset2 = GetKeyRopeGmOffset(realS2Idx2, sfaVecRunInfo, s2IdLimit);
        keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) : (keyOffset2 - keyOffset1)) -
                        sfaVecConstInfo.sparseBlockSize) *
                       sfaVecConstInfo.headDim * sizeof(KV_T);
        keyRopeSrcStride =
            ((keyRopeOffset1 > keyRopeOffset2 ? (keyRopeOffset1 - keyRopeOffset2) : (keyRopeOffset2 - keyRopeOffset1)) -
             sfaVecConstInfo.sparseBlockSize) *
            sfaVecConstInfo.headDimRope * sizeof(KV_T);
    }

    if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0 ||
                 (!PAGE_ATTENTION && (keyRopeSrcStride >= INT32_MAX || keyRopeSrcStride < 0)) ||
                 realS2Idx1 + sfaVecConstInfo.sparseBlockSize >= s2IdLimit ||
                 realS2Idx2 + sfaVecConstInfo.sparseBlockSize >= s2IdLimit)) {

        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyOffset1, s2IdLimit, sfaVecRunInfo);
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyOffset2, s2IdLimit, sfaVecRunInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockLen = sfaVecConstInfo.sparseBlockSize * sfaVecConstInfo.headDim * sizeof(KV_T);
        intriParams.blockCount = (keyOffset1 >= 0) + (keyOffset2 >= 0);
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t startGmOffset = keyOffset1 > -1 ? keyOffset1 : keyOffset2;
        if (keyOffset2 > -1 && keyOffset2 < keyOffset1) {
            startGmOffset = keyOffset2;
        }
        DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * 32 * 512 + (mte2Size - mte3Size) * sfaVecConstInfo.headDim],
                    keyGm_[startGmOffset * sfaVecConstInfo.headDim], intriParams, padParams);

        if constexpr (SFAT::hasRope) {
            intriParams.blockLen = sfaVecConstInfo.sparseBlockSize * sfaVecConstInfo.headDimRope * sizeof(KV_T);
            intriParams.dstStride = 0;
            intriParams.srcStride = keyRopeSrcStride;
            DataCopyPad(ropeMergUb_[mergeMte3Idx % 2 * 32 * 64 + (mte2Size - mte3Size) * sfaVecConstInfo.headDimRope],
                        keyRopeGm_[startGmOffset * sfaVecConstInfo.headDimRope], intriParams, padParams);
        }
        mte2Size += ((keyOffset1 > -1) + (keyOffset2 > -1)) * sfaVecConstInfo.sparseBlockSize;
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size,
                                                                 int64_t s2GmStartOffset, int64_t mergeMte3Idx,
                                                                 const RunInfo& sfaVecRunInfo)
{
    if (mte2Size <= mte3Size) {
        return;
    }
#ifdef K112174_FAST_MERGE_KV
    const int32_t m2m3Ev = static_cast<int32_t>(mergeMte3Idx & 1);
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(m2m3Ev);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(m2m3Ev);
#else
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
#endif

    uint32_t copyRows = static_cast<uint32_t>(mte2Size - mte3Size);
    DataCopy(kvMergeGm_[(sfaVecRunInfo.loop & 3) * 512 * 576 + (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDim],
             kvMergUb_[(mergeMte3Idx & 1) * 32 * 512], copyRows * sfaVecConstInfo.headDim);

    if constexpr (SFAT::hasRope) {
        DataCopy(kvMergeGm_[(sfaVecRunInfo.loop & 3) * 512 * 576 + 512 * 512 +
                            (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDimRope],
                 ropeMergUb_[(mergeMte3Idx & 1) * 32 * 64], copyRows * sfaVecConstInfo.headDimRope);
    }
}


template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::MergeKv(const RunInfo& sfaVecRunInfo)
{
    int64_t s2ProcessSize = sfaVecRunInfo.actualSingleProcessSInnerSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * sfaVecConstInfo.sparseBlockSize);
    int64_t topkGmBaseOffset = 0;

#ifdef K11089_L78_SFA_TOPK
    const uint32_t topkRowStride = (sfaVecConstInfo.sparseBlockCount < 2048U) ? 2048U : sfaVecConstInfo.sparseBlockCount;
#else
    const uint32_t topkRowStride = sfaVecConstInfo.sparseBlockCount;
#endif
    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum =
            (sfaVecRunInfo.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(sfaVecRunInfo.bIdx - 1);
        topkGmBaseOffset += (actualSeqQPrefixSum + sfaVecRunInfo.gS1Idx / sfaVecConstInfo.gSize) *
                                sfaVecConstInfo.kvHeadNum * topkRowStride +
                            sfaVecRunInfo.n2Idx * topkRowStride;
    } else {
        topkGmBaseOffset += sfaVecRunInfo.bIdx * sfaVecConstInfo.qSeqSize * topkRowStride +
                            sfaVecRunInfo.gS1Idx / sfaVecConstInfo.gSize * topkRowStride;
    }
    int64_t mergeMte3Idx = 0;
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    int64_t s2IdxArray0 = -1;
    int64_t s2IdxArray1 = -1;
    bool needWaitMte3ToMte2 = true;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    int64_t s2GmStartOffset = GetSubBlockIdx() == 0 ? 0 : CeilDiv(s2Pair, 2L) * 2 * sfaVecConstInfo.sparseBlockSize;
    int64_t s2GmLimit =
        GetSubBlockIdx() == 0 ? CeilDiv(s2Pair, 2L) * 2 * sfaVecConstInfo.sparseBlockSize : s2ProcessSize;
    if (s2GmLimit > s2ProcessSize) {
        s2GmLimit = s2ProcessSize;
    }
    uint32_t activeBlocks = static_cast<uint32_t>((sfaVecRunInfo.curActualSeqLenOri + 127ULL) >> 7U);
    if constexpr (PAGE_ATTENTION) {
        if (sfaVecConstInfo.sparseBlockSize == 1 && sfaVecConstInfo.kvCacheBlockSize == 128 &&
            sfaVecConstInfo.kvHeadNum == 1 && sfaVecConstInfo.headDim == 512 &&
            sfaVecConstInfo.headDimRope == 64 && activeBlocks > 0U && activeBlocks <= 512U) {
#if defined(K112516_SFA_V2) && defined(K112174_FAST_MERGE_KV) && defined(K112130_VEC_RESOLVED_TOPK) && defined(K11120_SFA_GATHER_OPT) && defined(K11090_FAST_SFA_VEC)
            const int64_t asymMid = CeilDiv(s2Pair, 2L) * 2L * sfaVecConstInfo.sparseBlockSize;
            const bool preResAsym = (treeMaskGmAddr_ & 1ULL) != 0ULL;
            if (sizeof(KV_T) == 2 && preResAsym && sfaVecRunInfo.actS1Size > 8U &&
                sfaVecRunInfo.loop >= 1U && asymMid >= 160 && s2ProcessSize > asymMid && s2ProcessSize <= 512) {
                constexpr int64_t kAsymSplit = 128;
                LocalTensor<int32_t> topkUbA = v0ValidSizeUb_[512];
                LocalTensor<uint64_t> topkUbA64 = topkUbA.template ReinterpretCast<uint64_t>();
                const int64_t maxRem0 =
                    static_cast<int64_t>(sfaVecConstInfo.sparseBlockCount) -
                    static_cast<int64_t>(sfaVecRunInfo.s2Idx * sfaVecConstInfo.s2BaseSize);
                WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
                auto runFastSlice = [&](int64_t idxBegin, int64_t idxEnd, int64_t dstBase) __aicore__ -> int64_t {
                    SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                    SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
                    int64_t m2 = 0;
                    int64_t m3 = 0;
                    int64_t mIdx = 0;
                    bool waitM3M2 = true;
                    for (int64_t localIdx = idxBegin; localIdx < idxEnd;) {
                        if (waitM3M2) {
                            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mIdx & 1);
                            waitM3M2 = false;
                        }
                        const uint64_t pairVal = topkUbA64.GetValue(localIdx >> 1);
                        const int32_t ko0 = (localIdx < maxRem0) ? static_cast<int32_t>(pairVal & 0xFFFFFFFFULL) : -1;
                        if (unlikely(ko0 == -1)) {
                            CopyOutMrgeResult(m2, m3, dstBase, mIdx, sfaVecRunInfo);
                            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mIdx & 1);
                            mIdx++;
                            break;
                        }
                        const int32_t ko1 = (localIdx + 1 < idxEnd && localIdx + 1 < maxRem0)
                                                ? static_cast<int32_t>(pairVal >> 32ULL)
                                                : -2;
                        int64_t stepIdx = 2;
                        if (likely(ko0 >= 0 && ko1 >= 0 && ko0 != ko1)) {
                            const int64_t startGmOffset = (ko1 < ko0) ? static_cast<int64_t>(ko1) : static_cast<int64_t>(ko0);
                            const int64_t endGmOffset = (ko1 < ko0) ? static_cast<int64_t>(ko0) : static_cast<int64_t>(ko1);
                            const int64_t blkTableSrcStride = endGmOffset - startGmOffset - 1;
                            const int64_t ubKvOff = (mIdx & 1) * 32 * 512 + (m2 - m3) * 512;
                            const int64_t ubRopeOff = (mIdx & 1) * 32 * 64 + (m2 - m3) * 64;
                            if (blkTableSrcStride == 0) {
                                int64_t runPairs = 1;
                                const int64_t maxPairs = (32 - (m2 - m3)) >> 1;
                                const int64_t remPairs = ((idxEnd < maxRem0 ? idxEnd : maxRem0) - localIdx) >> 1;
                                const int64_t limPairs = (maxPairs < remPairs) ? maxPairs : remPairs;
                                while (runPairs < limPairs) {
                                    const uint64_t nextPair = topkUbA64.GetValue((localIdx >> 1) + runPairs);
                                    const int32_t n0 = static_cast<int32_t>(nextPair & 0xFFFFFFFFULL);
                                    const int32_t n1 = static_cast<int32_t>(nextPair >> 32ULL);
                                    const int32_t exp0 = static_cast<int32_t>(startGmOffset + (runPairs << 1));
                                    if (!((n0 == exp0 && n1 == exp0 + 1) || (n1 == exp0 && n0 == exp0 + 1))) {
                                        break;
                                    }
                                    runPairs++;
                                }
                                DataCopy(kvMergUb_[ubKvOff], keyGm_[startGmOffset * 512], static_cast<uint32_t>(runPairs * 1024));
                                if constexpr (SFAT::hasRope) {
                                    DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64], static_cast<uint32_t>(runPairs * 128));
                                }
                                m2 += (runPairs << 1);
                                stepIdx = (runPairs << 1);
                            } else if (blkTableSrcStride <= 2047) {
                                DataCopy(kvMergUb_[ubKvOff], keyGm_[startGmOffset * 512],
                                         DataCopyParams{2, 32, static_cast<uint16_t>(blkTableSrcStride * 32), 0});
                                if constexpr (SFAT::hasRope) {
                                    DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64],
                                             DataCopyParams{2, 4, static_cast<uint16_t>(blkTableSrcStride * 4), 0});
                                }
                                m2 += 2;
                            } else {
                                DataCopy(kvMergUb_[ubKvOff], keyGm_[startGmOffset * 512], 512);
                                DataCopy(kvMergUb_[ubKvOff + 512], keyGm_[endGmOffset * 512], 512);
                                if constexpr (SFAT::hasRope) {
                                    if (likely(blkTableSrcStride <= 16383)) {
                                        DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64],
                                                 DataCopyParams{2, 4, static_cast<uint16_t>(blkTableSrcStride * 4), 0});
                                    } else {
                                        DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64], 64);
                                        DataCopy(ropeMergUb_[ubRopeOff + 64], keyRopeGm_[endGmOffset * 64], 64);
                                    }
                                }
                                m2 += 2;
                            }
                        } else {
                            if (ko0 >= 0) {
                                const int64_t ubKvOff = (mIdx & 1) * 32 * 512 + (m2 - m3) * 512;
                                const int64_t ubRopeOff = (mIdx & 1) * 32 * 64 + (m2 - m3) * 64;
                                DataCopy(kvMergUb_[ubKvOff], keyGm_[static_cast<int64_t>(ko0) * 512], 512);
                                if constexpr (SFAT::hasRope) {
                                    DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[static_cast<int64_t>(ko0) * 64], 64);
                                }
                                m2 += 1;
                            }
                            if (ko1 >= 0) {
                                const int64_t ubKvOff = (mIdx & 1) * 32 * 512 + (m2 - m3) * 512;
                                const int64_t ubRopeOff = (mIdx & 1) * 32 * 64 + (m2 - m3) * 64;
                                DataCopy(kvMergUb_[ubKvOff], keyGm_[static_cast<int64_t>(ko1) * 512], 512);
                                if constexpr (SFAT::hasRope) {
                                    DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[static_cast<int64_t>(ko1) * 64], 64);
                                }
                                m2 += 1;
                            }
                        }
                        localIdx += stepIdx;
                        if ((m2 - m3 + 2 > 32) || (localIdx >= idxEnd)) {
                            CopyOutMrgeResult(m2, m3, dstBase, mIdx, sfaVecRunInfo);
                            m3 = m2;
                            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mIdx & 1);
                            mIdx++;
                            waitM3M2 = true;
                        }
                    }
                    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
                    return m2;
                };
                if (GetSubBlockIdx() == 0) {
                    DataCopy(topkUbA, topkGm_[topkGmBaseOffset + sfaVecRunInfo.s2Idx * sfaVecConstInfo.s2BaseSize],
                             static_cast<uint32_t>(kAsymSplit));
                    SetFlag<AscendC::HardEvent::MTE2_S>(1);
                    WaitFlag<AscendC::HardEvent::MTE2_S>(1);
                    (void)runFastSlice(0, kAsymSplit, 0);
                    return;
                } else {
                    const uint32_t fullAlign = (static_cast<uint32_t>(s2ProcessSize) + 7U) & ~7U;
                    DataCopy(topkUbA, topkGm_[topkGmBaseOffset + sfaVecRunInfo.s2Idx * sfaVecConstInfo.s2BaseSize],
                             fullAlign);
                    SetFlag<AscendC::HardEvent::MTE2_S>(1);
                    WaitFlag<AscendC::HardEvent::MTE2_S>(1);
                    int64_t aiv0Valid = 0;
                    bool aiv0HitEnd = false;
                    for (int64_t li = 0; li < kAsymSplit; li += 2) {
                        const uint64_t pv = topkUbA64.GetValue(li >> 1);
                        const int32_t k0 = (li < maxRem0) ? static_cast<int32_t>(pv & 0xFFFFFFFFULL) : -1;
                        if (unlikely(k0 == -1)) {
                            aiv0HitEnd = true;
                            break;
                        }
                        const int32_t k1 = (li + 1 < kAsymSplit && li + 1 < maxRem0) ? static_cast<int32_t>(pv >> 32ULL) : -2;
                        if (k0 >= 0) {
                            aiv0Valid++;
                        }
                        if (k1 >= 0) {
                            aiv0Valid++;
                        }
                    }
                    const int64_t aiv1TailValid = aiv0HitEnd ? 0 : runFastSlice(kAsymSplit, asymMid, aiv0Valid);
                    const int64_t totalValid0 = aiv0Valid + aiv1TailValid;
                    const int64_t totalValid1 = aiv0HitEnd ? 0 : runFastSlice(asymMid, s2ProcessSize, asymMid);
                    const uint64_t mBaseZ = static_cast<uint64_t>(sfaVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM) * 512ULL * 576ULL;
                    if (unlikely(totalValid0 < asymMid || asymMid + totalValid1 < s2ProcessSize)) {
                        SetFlag<AscendC::HardEvent::MTE2_V>(0);
                        WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                        SetFlag<AscendC::HardEvent::MTE3_V>(0);
                        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                        Duplicate(kvMergUb_, static_cast<KV_T>(0.0), 64U * sfaVecConstInfo.headDim);
                        SetFlag<AscendC::HardEvent::V_MTE3>(0);
                        WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                        for (int64_t s2Off0 = totalValid0; s2Off0 < asymMid; s2Off0 += 64) {
                            const uint32_t chunkR0 = (asymMid - s2Off0 > 64) ? 64U : static_cast<uint32_t>(asymMid - s2Off0);
                            DataCopy(kvMergeGm_[mBaseZ + static_cast<uint64_t>(s2Off0) * sfaVecConstInfo.headDim],
                                     kvMergUb_, chunkR0 * sfaVecConstInfo.headDim);
                            if constexpr (SFAT::hasRope) {
                                DataCopy(kvMergeGm_[mBaseZ + 512ULL * sfaVecConstInfo.headDim +
                                                    static_cast<uint64_t>(s2Off0) * sfaVecConstInfo.headDimRope],
                                         kvMergUb_, chunkR0 * sfaVecConstInfo.headDimRope);
                            }
                        }
                        for (int64_t s2Off1 = asymMid + totalValid1; s2Off1 < s2ProcessSize; s2Off1 += 64) {
                            const uint32_t chunkR1 = (s2ProcessSize - s2Off1 > 64) ? 64U : static_cast<uint32_t>(s2ProcessSize - s2Off1);
                            DataCopy(kvMergeGm_[mBaseZ + static_cast<uint64_t>(s2Off1) * sfaVecConstInfo.headDim],
                                     kvMergUb_, chunkR1 * sfaVecConstInfo.headDim);
                            if constexpr (SFAT::hasRope) {
                                DataCopy(kvMergeGm_[mBaseZ + 512ULL * sfaVecConstInfo.headDim +
                                                    static_cast<uint64_t>(s2Off1) * sfaVecConstInfo.headDimRope],
                                         kvMergUb_, chunkR1 * sfaVecConstInfo.headDimRope);
                            }
                        }
                        SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                    }
                    const uint32_t slotIdx = sfaVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM;
                    v0ValidSizeUb_.SetValue(slotIdx, totalValid0);
                    v0ValidSizeUb_.SetValue(128U + slotIdx, totalValid1);
                    SetFlag<AscendC::HardEvent::S_MTE3>(1);
                    WaitFlag<AscendC::HardEvent::S_MTE3>(1);
                    DataCopy(kvValidSizeGm_[slotIdx * (128 * 2)], v0ValidSizeUb_, 256);
                    SetFlag<AscendC::HardEvent::MTE3_S>(SYNC_INPUT_V0BUF_FLAG);
                    WaitFlag<AscendC::HardEvent::MTE3_S>(SYNC_INPUT_V0BUF_FLAG);
                    return;
                }
            }
#endif
            int64_t subCount = s2GmLimit - s2GmStartOffset;
            if (subCount > 0) {
                LocalTensor<int32_t> topkUb = v0ValidSizeUb_[512];
                LocalTensor<int32_t> blkUb = v0ValidSizeUb_[1024];
                uint32_t topkAlign = (static_cast<uint32_t>(subCount) + 7U) & ~7U;
                uint32_t blkAlign = (activeBlocks + 7U) & ~7U;
                DataCopy(topkUb,
                         topkGm_[topkGmBaseOffset + sfaVecRunInfo.s2Idx * sfaVecConstInfo.s2BaseSize + s2GmStartOffset],
                         topkAlign);
#ifdef K112130_VEC_RESOLVED_TOPK
                const bool preResolved = (treeMaskGmAddr_ & 1ULL) != 0ULL;
                if (!preResolved && sfaVecRunInfo.isFirstSInnerLoop) {
                    DataCopy(blkUb, blkTableGm_[sfaVecRunInfo.bIdx * sfaVecConstInfo.maxBlockNumPerBatch], blkAlign);
                }
#else
                if (sfaVecRunInfo.isFirstSInnerLoop) {
                    DataCopy(blkUb, blkTableGm_[sfaVecRunInfo.bIdx * sfaVecConstInfo.maxBlockNumPerBatch], blkAlign);
                }
#endif
                SetFlag<AscendC::HardEvent::MTE2_S>(1);
                WaitFlag<AscendC::HardEvent::MTE2_S>(1);
#ifdef K11090_FAST_SFA_VEC
                const uint32_t qRowIdx = static_cast<uint32_t>(sfaVecRunInfo.gS1Idx / sfaVecConstInfo.gSize);
                const int64_t prefixLen =
                    static_cast<int64_t>(sfaVecRunInfo.curActualSeqLenOri) - static_cast<int64_t>(sfaVecRunInfo.actS1Size);
                bool hasRowTreeMask = false;
                uint32_t rowTreeMask = 0xFFFFFFFFU;
                const uint64_t cleanTreeMaskAddr = treeMaskGmAddr_ & ~1ULL;
#ifdef K112130_VEC_RESOLVED_TOPK
                if (!preResolved && cleanTreeMaskAddr != 0ULL) {
#else
                if (cleanTreeMaskAddr != 0ULL) {
#endif
                    const __gm__ uint32_t* maskPtr = reinterpret_cast<const __gm__ uint32_t*>(cleanTreeMaskAddr);
                    if (maskPtr[0] != 0U && qRowIdx < 32U) {
                        hasRowTreeMask = true;
                        rowTreeMask = maskPtr[qRowIdx];
                    }
                }
                int64_t s2IdLimit = sfaVecRunInfo.curActualSeqLenOri;
                if (sfaVecConstInfo.sparseMode == 3) {
                    s2IdLimit = prefixLen + static_cast<int64_t>(qRowIdx) + 1;
                }
#ifdef K112130_VEC_RESOLVED_TOPK
                if (preResolved) {
                    s2IdLimit = 1;
                }
#endif
                int64_t maxTopkRemain =
                    static_cast<int64_t>(sfaVecConstInfo.sparseBlockCount) -
                    static_cast<int64_t>(sfaVecRunInfo.s2Idx * sfaVecConstInfo.s2BaseSize + s2GmStartOffset);
                LocalTensor<uint64_t> topkUb64 = topkUb.template ReinterpretCast<uint64_t>();
#if defined(K112174_FAST_MERGE_KV) && defined(K112130_VEC_RESOLVED_TOPK) && defined(K11120_SFA_GATHER_OPT)
                if (sizeof(KV_T) == 2 && preResolved) {
                    for (int64_t localIdx = 0; localIdx < subCount;) {
                        if (needWaitMte3ToMte2) {
                            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
                            needWaitMte3ToMte2 = false;
                        }
                        const uint64_t pairVal = topkUb64.GetValue(localIdx >> 1);
                        const int32_t ko0 = (localIdx < maxTopkRemain) ? static_cast<int32_t>(pairVal & 0xFFFFFFFFULL) : -1;
                        if (unlikely(ko0 == -1)) {
                            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, sfaVecRunInfo);
                            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
                            mergeMte3Idx++;
                            break;
                        }
                        const int32_t ko1 = (localIdx + 1 < subCount && localIdx + 1 < maxTopkRemain)
                                                ? static_cast<int32_t>(pairVal >> 32ULL)
                                                : -2;
                        int64_t stepIdx = 2;
                        if (likely(ko0 >= 0 && ko1 >= 0 && ko0 != ko1)) {
                            const int64_t startGmOffset = (ko1 < ko0) ? static_cast<int64_t>(ko1) : static_cast<int64_t>(ko0);
                            const int64_t endGmOffset = (ko1 < ko0) ? static_cast<int64_t>(ko0) : static_cast<int64_t>(ko1);
                            const int64_t blkTableSrcStride = endGmOffset - startGmOffset - 1;
                            const int64_t ubKvOff = (mergeMte3Idx & 1) * 32 * 512 + (mte2Size - mte3Size) * 512;
                            const int64_t ubRopeOff = (mergeMte3Idx & 1) * 32 * 64 + (mte2Size - mte3Size) * 64;
                            if (blkTableSrcStride == 0) {
                                int64_t runPairs = 1;
                                const int64_t maxPairs = (32 - (mte2Size - mte3Size)) >> 1;
                                const int64_t remPairs = ((subCount < maxTopkRemain ? subCount : maxTopkRemain) - localIdx) >> 1;
                                const int64_t limPairs = (maxPairs < remPairs) ? maxPairs : remPairs;
                                while (runPairs < limPairs) {
                                    const uint64_t nextPair = topkUb64.GetValue((localIdx >> 1) + runPairs);
                                    const int32_t n0 = static_cast<int32_t>(nextPair & 0xFFFFFFFFULL);
                                    const int32_t n1 = static_cast<int32_t>(nextPair >> 32ULL);
                                    const int32_t exp0 = static_cast<int32_t>(startGmOffset + (runPairs << 1));
                                    if (!((n0 == exp0 && n1 == exp0 + 1) || (n1 == exp0 && n0 == exp0 + 1))) {
                                        break;
                                    }
                                    runPairs++;
                                }
                                DataCopy(kvMergUb_[ubKvOff], keyGm_[startGmOffset * 512], static_cast<uint32_t>(runPairs * 1024));
                                if constexpr (SFAT::hasRope) {
                                    DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64], static_cast<uint32_t>(runPairs * 128));
                                }
                                mte2Size += (runPairs << 1);
                                stepIdx = (runPairs << 1);
                            } else if (blkTableSrcStride <= 2047) {
                                DataCopy(kvMergUb_[ubKvOff], keyGm_[startGmOffset * 512],
                                         DataCopyParams{2, 32, static_cast<uint16_t>(blkTableSrcStride * 32), 0});
                                if constexpr (SFAT::hasRope) {
                                    DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64],
                                             DataCopyParams{2, 4, static_cast<uint16_t>(blkTableSrcStride * 4), 0});
                                }
                                mte2Size += 2;
                            } else {
                                DataCopy(kvMergUb_[ubKvOff], keyGm_[startGmOffset * 512], 512);
                                DataCopy(kvMergUb_[ubKvOff + 512], keyGm_[endGmOffset * 512], 512);
                                if constexpr (SFAT::hasRope) {
                                    if (likely(blkTableSrcStride <= 16383)) {
                                        DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64],
                                                 DataCopyParams{2, 4, static_cast<uint16_t>(blkTableSrcStride * 4), 0});
                                    } else {
                                        DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64], 64);
                                        DataCopy(ropeMergUb_[ubRopeOff + 64], keyRopeGm_[endGmOffset * 64], 64);
                                    }
                                }
                                mte2Size += 2;
                            }
                        } else {
                            if (ko0 >= 0) {
                                const int64_t ubKvOff = (mergeMte3Idx & 1) * 32 * 512 + (mte2Size - mte3Size) * 512;
                                const int64_t ubRopeOff = (mergeMte3Idx & 1) * 32 * 64 + (mte2Size - mte3Size) * 64;
                                DataCopy(kvMergUb_[ubKvOff], keyGm_[static_cast<int64_t>(ko0) * 512], 512);
                                if constexpr (SFAT::hasRope) {
                                    DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[static_cast<int64_t>(ko0) * 64], 64);
                                }
                                mte2Size += 1;
                            }
                            if (ko1 >= 0) {
                                const int64_t ubKvOff = (mergeMte3Idx & 1) * 32 * 512 + (mte2Size - mte3Size) * 512;
                                const int64_t ubRopeOff = (mergeMte3Idx & 1) * 32 * 64 + (mte2Size - mte3Size) * 64;
                                DataCopy(kvMergUb_[ubKvOff], keyGm_[static_cast<int64_t>(ko1) * 512], 512);
                                if constexpr (SFAT::hasRope) {
                                    DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[static_cast<int64_t>(ko1) * 64], 64);
                                }
                                mte2Size += 1;
                            }
                        }
                        localIdx += stepIdx;
                        if ((mte2Size - mte3Size + 2 > 32) || (localIdx >= subCount)) {
                            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, sfaVecRunInfo);
                            mte3Size = mte2Size;
                            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
                            mergeMte3Idx++;
                            needWaitMte3ToMte2 = true;
                        }
                    }
                } else
#endif
                for (int64_t localIdx = 0; localIdx < subCount; localIdx += 2) {
                    if (needWaitMte3ToMte2) {
                        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
                        needWaitMte3ToMte2 = false;
                    }
                    const uint64_t pairVal = topkUb64.GetValue(localIdx >> 1);
                    const int32_t raw0 = static_cast<int32_t>(pairVal & 0xFFFFFFFFULL);
                    const int32_t raw1 = static_cast<int32_t>(pairVal >> 32ULL);
                    int64_t keyOffset1 = -1;
                    int64_t keyOffset2 = -1;
#ifdef K112130_VEC_RESOLVED_TOPK
                    if (preResolved) {
                        const int32_t ko0 = (localIdx < maxTopkRemain) ? raw0 : -1;
                        if (unlikely(ko0 == -1)) {
                            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, sfaVecRunInfo);
                            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
                            mergeMte3Idx++;
                            break;
                        }
                        const int32_t ko1 = (localIdx + 1 < subCount && localIdx + 1 < maxTopkRemain) ? raw1 : -2;
                        keyOffset1 = (ko0 >= 0) ? static_cast<int64_t>(ko0) : -1;
                        keyOffset2 = (ko1 >= 0) ? static_cast<int64_t>(ko1) : -1;
                        s2IdxArray0 = 0;
                        s2IdxArray1 = 0;
                    } else
#endif
                    {
                        s2IdxArray0 = (localIdx < maxTopkRemain) ? static_cast<int64_t>(raw0) : -1;
                        if (unlikely(s2IdxArray0 < 0)) {
                            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, sfaVecRunInfo);
                            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
                            mergeMte3Idx++;
                            break;
                        }
                        s2IdxArray1 = (localIdx + 1 < subCount && localIdx + 1 < maxTopkRemain) ?
                                          static_cast<int64_t>(raw1) :
                                          -1;
                        if (s2IdxArray0 >= 0 && s2IdxArray0 < s2IdLimit &&
                            likely(!hasRowTreeMask || s2IdxArray0 < prefixLen ||
                                   (static_cast<uint32_t>(s2IdxArray0 - prefixLen) < 32U &&
                                    ((rowTreeMask >> static_cast<uint32_t>(s2IdxArray0 - prefixLen)) & 1U) != 0U))) {
                            const uint32_t u0 = static_cast<uint32_t>(s2IdxArray0);
                            keyOffset1 = (static_cast<int64_t>(blkUb.GetValue(u0 >> 7U)) << 7) + (u0 & 127U);
                        }
                        if (s2IdxArray1 >= 0 && s2IdxArray1 < s2IdLimit &&
                            likely(!hasRowTreeMask || s2IdxArray1 < prefixLen ||
                                   (static_cast<uint32_t>(s2IdxArray1 - prefixLen) < 32U &&
                                    ((rowTreeMask >> static_cast<uint32_t>(s2IdxArray1 - prefixLen)) & 1U) != 0U))) {
                            const uint32_t u1 = static_cast<uint32_t>(s2IdxArray1);
                            keyOffset2 = (static_cast<int64_t>(blkUb.GetValue(u1 >> 7U)) << 7) + (u1 & 127U);
                        }
                    }
                    if (likely(keyOffset1 >= 0 || keyOffset2 >= 0)) {
                        int64_t diff =
                            (keyOffset1 > keyOffset2) ? (keyOffset1 - keyOffset2) : (keyOffset2 - keyOffset1);
                        int64_t blkTableSrcStride = diff - 1;
                        int64_t keySrcStride = blkTableSrcStride * 512 * sizeof(KV_T);
                        if (unlikely(keyOffset1 < 0 || keyOffset2 < 0 || keySrcStride >= INT32_MAX ||
                                     keySrcStride < 0)) {
                            CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, keyOffset1, s2IdLimit,
                                           sfaVecRunInfo);
                            CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray1, keyOffset2, s2IdLimit,
                                           sfaVecRunInfo);
                        } else {
                            int64_t startGmOffset = (keyOffset2 < keyOffset1) ? keyOffset2 : keyOffset1;
#ifdef K11120_SFA_GATHER_OPT
                            const int64_t endGmOffset = (keyOffset2 < keyOffset1) ? keyOffset1 : keyOffset2;
                            const int64_t ubKvOff = (mergeMte3Idx & 1) * 32 * 512 + (mte2Size - mte3Size) * 512;
                            const int64_t ubRopeOff = (mergeMte3Idx & 1) * 32 * 64 + (mte2Size - mte3Size) * 64;
                            if constexpr (sizeof(KV_T) == 2) {
#ifdef K112162_MERGE_CONTIG
                                if (blkTableSrcStride == 0) {
                                    DataCopy(kvMergUb_[ubKvOff], keyGm_[startGmOffset * 512], 1024);
                                    if constexpr (SFAT::hasRope) {
                                        DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64], 128);
                                    }
                                } else
#endif
                                if (blkTableSrcStride <= 2047) {
                                    DataCopy(kvMergUb_[ubKvOff], keyGm_[startGmOffset * 512],
                                             DataCopyParams{2, 32, static_cast<uint16_t>(blkTableSrcStride * 32), 0});
                                    if constexpr (SFAT::hasRope) {
                                        DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64],
                                                 DataCopyParams{2, 4, static_cast<uint16_t>(blkTableSrcStride * 4), 0});
                                    }
                                } else {
                                    DataCopy(kvMergUb_[ubKvOff], keyGm_[startGmOffset * 512], 512);
                                    DataCopy(kvMergUb_[ubKvOff + 512], keyGm_[endGmOffset * 512], 512);
                                    if constexpr (SFAT::hasRope) {
#ifdef K112174_FAST_MERGE_KV
                                        if (likely(blkTableSrcStride <= 16383)) {
                                            DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64],
                                                     DataCopyParams{2, 4, static_cast<uint16_t>(blkTableSrcStride * 4), 0});
                                        } else
#endif
                                        {
                                            DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64], 64);
                                            DataCopy(ropeMergUb_[ubRopeOff + 64], keyRopeGm_[endGmOffset * 64], 64);
                                        }
                                    }
                                }
                                mte2Size += 2;
                            } else if constexpr (sizeof(KV_T) == 1) {
                                if (blkTableSrcStride <= 4095) {
                                    DataCopy(kvMergUb_[ubKvOff], keyGm_[startGmOffset * 512],
                                             DataCopyParams{2, 16, static_cast<uint16_t>(blkTableSrcStride * 16), 0});
                                    if constexpr (SFAT::hasRope) {
                                        DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64],
                                                 DataCopyParams{2, 2, static_cast<uint16_t>(blkTableSrcStride * 2), 0});
                                    }
                                } else {
                                    DataCopy(kvMergUb_[ubKvOff], keyGm_[startGmOffset * 512], 512);
                                    DataCopy(kvMergUb_[ubKvOff + 512], keyGm_[endGmOffset * 512], 512);
                                    if constexpr (SFAT::hasRope) {
                                        DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64], 64);
                                        DataCopy(ropeMergUb_[ubRopeOff + 64], keyRopeGm_[endGmOffset * 64], 64);
                                    }
                                }
                                mte2Size += 2;
                            } else {
                                DataCopyExtParams intriParams{2, 512 * sizeof(KV_T), static_cast<uint32_t>(keySrcStride), 0, 0};
                                DataCopyPadExtParams<KV_T> padParams;
                                DataCopyPad(kvMergUb_[ubKvOff], keyGm_[startGmOffset * 512], intriParams, padParams);
                                if constexpr (SFAT::hasRope) {
                                    intriParams.blockLen = 64 * sizeof(KV_T);
                                    intriParams.srcStride = blkTableSrcStride * 64 * sizeof(KV_T);
                                    DataCopyPad(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64], intriParams, padParams);
                                }
                                mte2Size += 2;
                            }
#else
                            if constexpr (sizeof(KV_T) == 2) {
                                if (likely(blkTableSrcStride <= 2047)) {
                                    DataCopy(kvMergUb_[(mergeMte3Idx & 1) * 32 * 512 + (mte2Size - mte3Size) * 512],
                                             keyGm_[startGmOffset * 512],
                                             DataCopyParams{2, 32, static_cast<uint16_t>(blkTableSrcStride * 32), 0});
                                    if constexpr (SFAT::hasRope) {
                                        DataCopy(ropeMergUb_[(mergeMte3Idx & 1) * 32 * 64 + (mte2Size - mte3Size) * 64],
                                                 keyRopeGm_[startGmOffset * 64],
                                                 DataCopyParams{2, 4, static_cast<uint16_t>(blkTableSrcStride * 4), 0});
                                    }
                                    mte2Size += 2;
                                } else {
                                    DataCopyExtParams intriParams{2, 512 * sizeof(KV_T), static_cast<uint32_t>(keySrcStride), 0, 0};
                                    DataCopyPadExtParams<KV_T> padParams;
                                    DataCopyPad(kvMergUb_[(mergeMte3Idx & 1) * 32 * 512 + (mte2Size - mte3Size) * 512],
                                                keyGm_[startGmOffset * 512], intriParams, padParams);
                                    if constexpr (SFAT::hasRope) {
                                        intriParams.blockLen = 64 * sizeof(KV_T);
                                        intriParams.srcStride = blkTableSrcStride * 64 * sizeof(KV_T);
                                        DataCopyPad(ropeMergUb_[(mergeMte3Idx & 1) * 32 * 64 + (mte2Size - mte3Size) * 64],
                                                    keyRopeGm_[startGmOffset * 64], intriParams, padParams);
                                    }
                                    mte2Size += 2;
                                }
                            } else {
                                DataCopyExtParams intriParams{2, 512 * sizeof(KV_T), static_cast<uint32_t>(keySrcStride), 0, 0};
                                DataCopyPadExtParams<KV_T> padParams;
                                DataCopyPad(kvMergUb_[(mergeMte3Idx & 1) * 32 * 512 + (mte2Size - mte3Size) * 512],
                                            keyGm_[startGmOffset * 512], intriParams, padParams);
                                if constexpr (SFAT::hasRope) {
                                    intriParams.blockLen = 64 * sizeof(KV_T);
                                    intriParams.srcStride = blkTableSrcStride * 64 * sizeof(KV_T);
                                    DataCopyPad(ropeMergUb_[(mergeMte3Idx & 1) * 32 * 64 + (mte2Size - mte3Size) * 64],
                                                keyRopeGm_[startGmOffset * 64], intriParams, padParams);
                                }
                                mte2Size += 2;
                            }
#endif
                        }
                    }
#else
                int64_t s2IdLimit = sfaVecRunInfo.curActualSeqLenOri;
                if (sfaVecConstInfo.sparseMode == 3) {
                    s2IdLimit = sfaVecRunInfo.curActualSeqLenOri - sfaVecRunInfo.actS1Size +
                                sfaVecRunInfo.gS1Idx / sfaVecConstInfo.gSize + 1;
                }
                int64_t maxTopkRemain =
                    static_cast<int64_t>(sfaVecConstInfo.sparseBlockCount) -
                    static_cast<int64_t>(sfaVecRunInfo.s2Idx * sfaVecConstInfo.s2BaseSize + s2GmStartOffset);
                for (int64_t localIdx = 0; localIdx < subCount; localIdx += 2) {
                    if (needWaitMte3ToMte2) {
                        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
                        needWaitMte3ToMte2 = false;
                    }
                    s2IdxArray0 = (localIdx < maxTopkRemain) ? static_cast<int64_t>(topkUb.GetValue(localIdx)) : -1;
                    if (unlikely(s2IdxArray0 < 0)) {
                        CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, sfaVecRunInfo);
                        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
                        mergeMte3Idx++;
                        break;
                    }
                    s2IdxArray1 = (localIdx + 1 < subCount && localIdx + 1 < maxTopkRemain) ?
                                      static_cast<int64_t>(topkUb.GetValue(localIdx + 1)) :
                                      -1;
                    int64_t keyOffset1 = -1;
                    int64_t keyOffset2 = -1;
                    if (IsTreeKeyValid(s2IdxArray0, sfaVecRunInfo, s2IdLimit)) {
                        uint32_t u0 = static_cast<uint32_t>(s2IdxArray0);
                        keyOffset1 = (static_cast<int64_t>(blkUb.GetValue(u0 >> 7)) << 7) + (u0 & 127U);
                    }
                    if (IsTreeKeyValid(s2IdxArray1, sfaVecRunInfo, s2IdLimit)) {
                        uint32_t u1 = static_cast<uint32_t>(s2IdxArray1);
                        keyOffset2 = (static_cast<int64_t>(blkUb.GetValue(u1 >> 7)) << 7) + (u1 & 127U);
                    }
                    if (likely(keyOffset1 >= 0 || keyOffset2 >= 0)) {
                        int64_t diff =
                            (keyOffset1 > keyOffset2) ? (keyOffset1 - keyOffset2) : (keyOffset2 - keyOffset1);
                        int64_t blkTableSrcStride = diff - 1;
                        int64_t keySrcStride = blkTableSrcStride * 512 * sizeof(KV_T);
                        if (unlikely(keyOffset1 < 0 || keyOffset2 < 0 || keySrcStride >= INT32_MAX ||
                                     keySrcStride < 0)) {
                            CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, keyOffset1, s2IdLimit,
                                           sfaVecRunInfo);
                            CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray1, keyOffset2, s2IdLimit,
                                           sfaVecRunInfo);
                        } else {
                            DataCopyExtParams intriParams;
                            intriParams.blockLen = 512 * sizeof(KV_T);
                            intriParams.blockCount = 2;
                            intriParams.dstStride = 0;
                            intriParams.srcStride = keySrcStride;
                            DataCopyPadExtParams<KV_T> padParams;
                            int64_t startGmOffset = (keyOffset2 < keyOffset1) ? keyOffset2 : keyOffset1;
                            DataCopyPad(kvMergUb_[(mergeMte3Idx & 1) * 32 * 512 + (mte2Size - mte3Size) * 512],
                                        keyGm_[startGmOffset * 512], intriParams, padParams);
                            if constexpr (SFAT::hasRope) {
                                intriParams.blockLen = 64 * sizeof(KV_T);
                                intriParams.srcStride = blkTableSrcStride * 64 * sizeof(KV_T);
                                DataCopyPad(ropeMergUb_[(mergeMte3Idx & 1) * 32 * 64 + (mte2Size - mte3Size) * 64],
                                            keyRopeGm_[startGmOffset * 64], intriParams, padParams);
                            }
                            mte2Size += 2;
                        }
                    }
#endif
                    if ((mte2Size - mte3Size + 2 > 32) || (localIdx + 2 >= subCount)) {
                        CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, sfaVecRunInfo);
                        mte3Size = mte2Size;
                        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
                        mergeMte3Idx++;
                        needWaitMte3ToMte2 = true;
                    }
                }
            }
        } else {
            for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit;
                 s2GmOffsetArray += 2 * sfaVecConstInfo.sparseBlockSize) {
                if (needWaitMte3ToMte2) {
                    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
                    needWaitMte3ToMte2 = false;
                }
                GetRealS2Idx(s2GmOffsetArray, s2IdxArray0, topkGmBaseOffset, sfaVecRunInfo);
                if (unlikely(s2IdxArray0 < 0)) {
                    CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, sfaVecRunInfo);
                    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
                    mergeMte3Idx++;
                    break;
                }
                GetRealS2Idx(s2GmOffsetArray + sfaVecConstInfo.sparseBlockSize, s2IdxArray1, topkGmBaseOffset,
                             sfaVecRunInfo);
                CopyInKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1, sfaVecRunInfo);
                if ((mte2Size - mte3Size + 2 * sfaVecConstInfo.sparseBlockSize > 32) ||
                    s2GmOffsetArray + 2 * sfaVecConstInfo.sparseBlockSize >= s2GmLimit) {
                    CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, sfaVecRunInfo);
                    mte3Size = mte2Size;
                    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
                    mergeMte3Idx++;
                    needWaitMte3ToMte2 = true;
                }
            }
        }
    } else {
        for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit;
             s2GmOffsetArray += 2 * sfaVecConstInfo.sparseBlockSize) {
            if (needWaitMte3ToMte2) {
                WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
                needWaitMte3ToMte2 = false;
            }
            GetRealS2Idx(s2GmOffsetArray, s2IdxArray0, topkGmBaseOffset, sfaVecRunInfo);
            if (unlikely(s2IdxArray0 < 0)) {
                CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, sfaVecRunInfo);
                SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
                mergeMte3Idx++;
                break;
            }
            GetRealS2Idx(s2GmOffsetArray + sfaVecConstInfo.sparseBlockSize, s2IdxArray1, topkGmBaseOffset,
                         sfaVecRunInfo);
            CopyInKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1, sfaVecRunInfo);
            if ((mte2Size - mte3Size + 2 * sfaVecConstInfo.sparseBlockSize > 32) ||
                s2GmOffsetArray + 2 * sfaVecConstInfo.sparseBlockSize >= s2GmLimit) {
                CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, sfaVecRunInfo);
                mte3Size = mte2Size;
                SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
                mergeMte3Idx++;
                needWaitMte3ToMte2 = true;
            }
        }
    }

#ifdef K11090_FAST_SFA_VEC
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    if (unlikely(s2GmStartOffset + mte2Size < s2GmLimit)) {
        SetFlag<AscendC::HardEvent::MTE2_V>(0);
        WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        const int64_t remRows = s2GmLimit - (s2GmStartOffset + mte2Size);
        const uint32_t zeroRows = (remRows > 64) ? 64U : static_cast<uint32_t>(remRows);
        Duplicate(kvMergUb_, static_cast<KV_T>(0.0), zeroRows * sfaVecConstInfo.headDim);
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        const uint64_t mBase = static_cast<uint64_t>(sfaVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM) * 512ULL * 576ULL;
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset += 64) {
            const uint32_t chunkR = (s2GmLimit - s2GmOffset > 64) ? 64U : static_cast<uint32_t>(s2GmLimit - s2GmOffset);
            DataCopy(kvMergeGm_[mBase + static_cast<uint64_t>(s2GmOffset) * sfaVecConstInfo.headDim],
                     kvMergUb_, chunkR * sfaVecConstInfo.headDim);
            if constexpr (SFAT::hasRope) {
                DataCopy(kvMergeGm_[mBase + 512ULL * sfaVecConstInfo.headDim +
                                    static_cast<uint64_t>(s2GmOffset) * sfaVecConstInfo.headDimRope],
                         kvMergUb_, chunkR * sfaVecConstInfo.headDimRope);
            }
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    }
#else
    if (unlikely(s2GmStartOffset + mte2Size < s2GmLimit)) {
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
        Duplicate(kvMergUb_, static_cast<KV_T>(0.0), sfaVecConstInfo.headDim);
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = sfaVecConstInfo.headDim * sizeof(KV_T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset++) {
            DataCopyPad(kvMergeGm_[sfaVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                                   s2GmOffset * sfaVecConstInfo.headDim],
                        kvMergUb_, dataCopyParams);
        }
        if constexpr (SFAT::hasRope) {
            dataCopyParams.blockLen = sfaVecConstInfo.headDimRope * sizeof(KV_T);
            for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset++) {
                DataCopyPad(kvMergeGm_[sfaVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                                       512 * sfaVecConstInfo.headDim + s2GmOffset * sfaVecConstInfo.headDimRope],
                            kvMergUb_, dataCopyParams);
            }
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
        mergeMte3Idx++;
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
#endif
    v0ValidSizeUb_.SetValue(sfaVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM, mte2Size);
    SetFlag<AscendC::HardEvent::S_MTE3>(1);
    WaitFlag<AscendC::HardEvent::S_MTE3>(1);
    DataCopy(kvValidSizeGm_[sfaVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2) + GetSubBlockIdx() * 128],
             v0ValidSizeUb_, 128);
    SetFlag<AscendC::HardEvent::MTE3_S>(SYNC_INPUT_V0BUF_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_S>(SYNC_INPUT_V0BUF_FLAG);
    return;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ProcessVec1L(const RunInfo& sfaVecInfo)
{
    uint32_t nBufferLoopTimes =
        (sfaVecInfo.actMBaseSize + sfaVecConstInfo.nBufferMBaseSize - 1) / sfaVecConstInfo.nBufferMBaseSize;
    uint32_t nBufferTail = sfaVecInfo.actMBaseSize - (nBufferLoopTimes - 1) * sfaVecConstInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo sfaVecSplitInfo;
        sfaVecSplitInfo.nBufferIdx = i;
        sfaVecSplitInfo.nBufferStartM = i * sfaVecConstInfo.nBufferMBaseSize;
        sfaVecSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? sfaVecConstInfo.nBufferMBaseSize : nBufferTail;

#ifdef K112130_SPLIT_HEADS_4_4
#ifdef K112172_SPLIT_HEADS_ALL
        if (sfaVecSplitInfo.nBufferDealM == 8U && sfaVecConstInfo.headDim == 512U) {
#else
        if (sfaVecSplitInfo.nBufferDealM == 8U && sfaVecConstInfo.headDim == 512U && sfaVecInfo.actS1Size <= 8U) {
#endif
            sfaVecSplitInfo.vecDealM = 4U;
            sfaVecSplitInfo.vecStartM = (GetBlockIdx() % 2 == 1) ? 4U : 0U;
        } else
#endif
        {
            sfaVecSplitInfo.vecDealM = (sfaVecSplitInfo.nBufferDealM <= 16) ?
                                           sfaVecSplitInfo.nBufferDealM :
                                           (((sfaVecSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
            sfaVecSplitInfo.vecStartM = 0;
            if (GetBlockIdx() % 2 == 1) {
                sfaVecSplitInfo.vecStartM = sfaVecSplitInfo.vecDealM;
                sfaVecSplitInfo.vecDealM = sfaVecSplitInfo.nBufferDealM - sfaVecSplitInfo.vecDealM;
            }
        }

        CrossCoreWaitFlag(sfaVecConstInfo.syncC1V1);

        ProcessVec1SingleBuf(sfaVecInfo, sfaVecSplitInfo);
        CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_MTE3>(sfaVecConstInfo.syncV1C2);
        CrossCoreWaitFlag(sfaVecConstInfo.syncC2V1);

        if (sfaVecInfo.actualSingleProcessSInnerSize != 0) {
            ProcessAmlaNupdate(sfaVecInfo, sfaVecSplitInfo);
            CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_MTE3>(sfaVecConstInfo.syncV1NupdateC2);
        }

        if (sfaVecInfo.s2Idx == sfaVecInfo.curSInnerLoopTimes - 1 &&
            (sfaVecConstInfo.returnSoftmaxLse || sfaVecInfo.tndIsS2SplitCore)) {
            uint32_t outIdx = sfaVecInfo.loop % (sfaVecConstInfo.preLoadNum);
            auto sumTensor = softmaxSumUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
            auto maxTensor = softmaxMaxUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
            if (sfaVecConstInfo.returnSoftmaxLse) {
                CopyFALseToGm(sfaVecInfo, sfaVecSplitInfo, sumTensor, maxTensor);
            }
            if (sfaVecInfo.tndIsS2SplitCore) {
                if constexpr (FLASH_DECODE) {
                    ComputeLogSumExpAndCopyToGm(sfaVecInfo, sfaVecSplitInfo, sumTensor, maxTensor);
                }
            }
        }
    }
}

template <typename SFAT>
__aicore__ inline uint64_t SFAVectorService<SFAT>::CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx)
{
    return 0;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ProcessVec2SingleBuf(const RunInfo& sfaVecInfo,
                                                                    const MSplitInfo& sfaVecSplitInfo)
{
    if (sfaVecInfo.s2Idx + 1 != sfaVecInfo.curSInnerLoopTimes) {
        return;
    }
    if (sfaVecSplitInfo.vecDealM == 0) {
        return;
    }

    ProcessVec2Inner(sfaVecInfo, sfaVecSplitInfo, 0, sfaVecSplitInfo.vecDealM);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ProcessVec2L(const RunInfo& sfaVecInfo)
{
    uint32_t nBufferLoopTimes =
        (sfaVecInfo.actMBaseSize + sfaVecConstInfo.nBufferMBaseSize - 1) / sfaVecConstInfo.nBufferMBaseSize;
    uint32_t nBufferTail = sfaVecInfo.actMBaseSize - (nBufferLoopTimes - 1) * sfaVecConstInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo sfaVecSplitInfo;
        sfaVecSplitInfo.nBufferIdx = i;
        sfaVecSplitInfo.nBufferStartM = i * sfaVecConstInfo.nBufferMBaseSize;
        sfaVecSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? sfaVecConstInfo.nBufferMBaseSize : nBufferTail;

#ifdef K112130_SPLIT_HEADS_4_4
#ifdef K112172_SPLIT_HEADS_ALL
        if (sfaVecSplitInfo.nBufferDealM == 8U && sfaVecConstInfo.headDim == 512U) {
#else
        if (sfaVecSplitInfo.nBufferDealM == 8U && sfaVecConstInfo.headDim == 512U && sfaVecInfo.actS1Size <= 8U) {
#endif
            sfaVecSplitInfo.vecDealM = 4U;
            sfaVecSplitInfo.vecStartM = (GetBlockIdx() % 2 == 1) ? 4U : 0U;
        } else
#endif
        {
            sfaVecSplitInfo.vecStartM = 0;
            sfaVecSplitInfo.vecDealM = (sfaVecSplitInfo.nBufferDealM <= 16) ?
                                           sfaVecSplitInfo.nBufferDealM :
                                           (((sfaVecSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
            if (GetBlockIdx() % 2 == 1) {
                sfaVecSplitInfo.vecStartM = sfaVecSplitInfo.vecDealM;
                sfaVecSplitInfo.vecDealM = sfaVecSplitInfo.nBufferDealM - sfaVecSplitInfo.vecDealM;
            }
        }
        CrossCoreWaitFlag(sfaVecConstInfo.syncC2V2);
        ProcessVec2SingleBuf(sfaVecInfo, sfaVecSplitInfo);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ProcessVec2Inner(const RunInfo& sfaVecInfo,
                                                                const MSplitInfo& sfaVecSplitInfo, uint32_t mStartRow,
                                                                uint32_t mDealSize)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / sfaVecConstInfo.headDim;
    if (mSplitSize > mDealSize) {
        mSplitSize = mDealSize;
    }

    uint32_t loopCount = (mDealSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mDealSize - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(sfaVecInfo, sfaVecSplitInfo, i * mSplitSize + mStartRow, dealSize, sfaVecConstInfo.headDim,
                             sfaVecConstInfo.headDim);
        pingpongFlag ^= 1;
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::GetConfusionTransposeTiling(int64_t numR, int64_t numC,
                                                                           const uint32_t stackBufferSize,
                                                                           const uint32_t typeSize,
                                                                           ConfusionTransposeTiling& tiling)
{
    (void)stackBufferSize;
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t height = numC;
    uint32_t width = numR;
    uint32_t highBlock = height / BLOCK_CUBE;
    uint32_t stride = height * blockSize * typeSize / ONE_BLK_SIZE;
    uint32_t repeat = width / blockSize;

    tiling.param0 = blockSize;
    tiling.param1 = height;
    tiling.param2 = width;
    tiling.param3 = highBlock;
    tiling.param4 = stride;
    tiling.param5 = repeat;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::Bmm2FDDataCopyOut(const RunInfo& sfaVecInfo,
                                                                 LocalTensor<T>& sfaBmm2ResultUb,
                                                                 uint32_t sfaWorkspaceRow, uint32_t sfaDealRows,
                                                                 uint32_t sfaColumns, uint32_t sfaActualColumns)
{
    LocalTensor<T> tmp = outputBuff1.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(tmp, sfaBmm2ResultUb, sfaColumns * sfaDealRows);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    uint64_t sfaAccumOutputCount = CalcAccumOffset(sfaVecInfo.bIdx, sfaVecInfo.gS1Idx);
    uint64_t offset = sfaAccumOutputCount * sfaVecConstInfo.kvHeadNum * sfaVecConstInfo.mBaseSize *
                          sfaVecConstInfo.headDim +
                      sfaVecInfo.tndCoreStartKVSplitPos * sfaVecConstInfo.kvHeadNum * sfaVecConstInfo.mBaseSize *
                          sfaVecConstInfo.headDim +
                      sfaWorkspaceRow * sfaActualColumns;
    GlobalTensor<T> dst = accumOutGm[offset];
    if (sfaVecInfo.actualSingleProcessSInnerSize == 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = sfaDealRows;
        dataCopyParams.blockLen = sfaActualColumns * sizeof(T);
        dataCopyParams.dstStride = 0;
        dataCopyParams.srcStride = (sfaColumns - sfaActualColumns) / (BYTE_BLOCK / sizeof(T));
        DataCopyPad(dst, tmp, dataCopyParams);
    } else {
        matmul::InitOutput<T>(dst, sfaDealRows * sfaActualColumns, ConstInfo::FLOAT_ZERO);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::Bmm2DataCopyOutTrans(const RunInfo& sfaVecInfo,
                                                                    LocalTensor<OUT_T>& attenOutUb,
                                                                    uint32_t sfaWorkspaceRow, uint32_t sfaDealRows,
                                                                    uint32_t sfaColumns, uint32_t sfaActualColumns)
{
    if (sfaColumns == sfaActualColumns && ((sfaActualColumns * sizeof(OUT_T)) & 31U) == 0U) {
        DataCopy(attentionOutGm[sfaVecInfo.attenOutOffset + sfaWorkspaceRow * sfaActualColumns],
                 attenOutUb, sfaDealRows * sfaActualColumns);
    } else {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = sfaDealRows;
        dataCopyParams.blockLen = sfaActualColumns * sizeof(OUT_T);
        dataCopyParams.srcStride = (sfaColumns - sfaActualColumns) / (BYTE_BLOCK / sizeof(OUT_T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(attentionOutGm[sfaVecInfo.attenOutOffset + sfaWorkspaceRow * sfaActualColumns], attenOutUb,
                    dataCopyParams);
    }
    return;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::Bmm2CastAndCopyOut(const RunInfo& sfaVecInfo,
                                                                  LocalTensor<T>& sfaBmm2ResultUb,
                                                                  uint32_t sfaWorkspaceRow, uint32_t sfaDealRows,
                                                                  uint32_t sfaColumns, uint32_t sfaActualColumns)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputBuff1.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) {
        Cast(tmpBmm2ResCastTensor, sfaBmm2ResultUb, AscendC::RoundMode::CAST_RINT, sfaDealRows * sfaColumns);
    } else {
        Cast(tmpBmm2ResCastTensor, sfaBmm2ResultUb, AscendC::RoundMode::CAST_ROUND, sfaDealRows * sfaColumns);
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    Bmm2DataCopyOutTrans(sfaVecInfo, tmpBmm2ResCastTensor, sfaWorkspaceRow, sfaDealRows, sfaColumns, sfaActualColumns);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::Bmm2ResCopyOut(const RunInfo& sfaVecInfo,
                                                              LocalTensor<T>& sfaBmm2ResultUb, uint32_t sfaWorkspaceRow,
                                                              uint32_t sfaDealRows, uint32_t sfaColumns,
                                                              uint32_t sfaActualColumns)
{
    if constexpr (FLASH_DECODE) {
        if (sfaVecInfo.tndIsS2SplitCore) {
            Bmm2FDDataCopyOut(sfaVecInfo, sfaBmm2ResultUb, sfaWorkspaceRow, sfaDealRows, sfaColumns, sfaActualColumns);
        } else {
            Bmm2CastAndCopyOut(sfaVecInfo, sfaBmm2ResultUb, sfaWorkspaceRow, sfaDealRows, sfaColumns, sfaActualColumns);
        }
    } else {
        Bmm2CastAndCopyOut(sfaVecInfo, sfaBmm2ResultUb, sfaWorkspaceRow, sfaDealRows, sfaColumns, sfaActualColumns);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::DealBmm2ResBaseBlock(const RunInfo& sfaVecInfo,
                                                                    const MSplitInfo& sfaVecSplitInfo,
                                                                    uint32_t startRow, uint32_t sfaDealRows,
                                                                    uint32_t sfaColumns, uint32_t sfaActualColumns)
{
    uint32_t vec2ComputeSize = sfaDealRows * sfaColumns;
    uint32_t mStart = sfaVecSplitInfo.nBufferStartM + sfaVecSplitInfo.vecStartM + startRow;
    uint64_t srcGmOffset =
        (sfaVecInfo.bn2IdxInCurCore % sfaVecConstInfo.preLoadNum) * sfaVecConstInfo.bmm2ResUbSize + mStart * sfaColumns;
    LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputBuff1.Get<MM2_OUT_T>();
    tmpBmm2ResUb = tmpBmm2ResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset], vec2ComputeSize);

    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);


    LocalTensor<T> sfaBmm2ResultUb = tmpBuff1.Get<T>();
    sfaBmm2ResultUb.SetSize(vec2ComputeSize);
    LocalTensor<T> absBmm2ResUb = sfaBmm2ResultUb.template ReinterpretCast<T>();
    Abs(absBmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> cmpMaskUb = absBmm2ResUb.template ReinterpretCast<uint8_t>();
    CompareScalar(cmpMaskUb, absBmm2ResUb, (T)1e10, CMPMODE::LE, vec2ComputeSize);
    PipeBarrier<PIPE_V>();
    Select(tmpBmm2ResUb, cmpMaskUb, tmpBmm2ResUb, ConstInfo::FLOAT_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE,
           vec2ComputeSize);
    PipeBarrier<PIPE_V>();
    uint32_t baseOffset = sfaVecSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t idx = sfaVecInfo.loop % (sfaVecConstInfo.preLoadNum);
    LocalTensor<T> tmpSumUb = v0ValidSizeBuff.Get<T>()[384];
    Brcb(tmpSumUb, aMlaSumUb[idx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset], (sfaDealRows + 7) / 8, {1, 8});
    PipeBarrier<PIPE_V>();
    RowDivs(sfaBmm2ResultUb, tmpBmm2ResUb, tmpSumUb, sfaDealRows, sfaColumns, sfaActualColumns);
    PipeBarrier<PIPE_V>();
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    Bmm2ResCopyOut(sfaVecInfo, sfaBmm2ResultUb, mStart, sfaDealRows, sfaColumns, sfaActualColumns);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub,
                                                       LocalTensor<float> src1Ub, uint32_t sfaDealRows,
                                                       uint32_t sfaColumns, uint32_t sfaActualColumns)
{


    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = sfaActualColumns / dtypeMask;
    uint32_t dRemain = sfaActualColumns % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 0;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = sfaColumns / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 1;
    repeatParamsDiv.dstRepStride = sfaColumns / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    if (columnRepeatCount <= sfaDealRows) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub, dtypeMask, sfaDealRows, repeatParamsDiv);
            offset += dtypeMask;
        }
    } else {
        BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.src0RepStride = 8;
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < sfaDealRows; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub[i * FP32_BLOCK_ELEMENT_NUM], dtypeMask, columnRepeatCount,
                columnRepeatParams);
            offset += sfaColumns;
        }
    }
    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub, dRemain, sfaDealRows, repeatParamsDiv);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub,
                                                       LocalTensor<T> src1Ub, uint32_t sfaDealRows, uint32_t sfaColumns,
                                                       uint32_t sfaActualColumns)
{


    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {

        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2;
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;
    }


    uint32_t dRemain = sfaActualColumns % repeatElementNum;
    uint32_t dLoop = sfaActualColumns / repeatElementNum;

    if (sfaColumns < REPEATE_STRIDE_UP_BOUND * blockElementNum) {
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = sfaColumns / blockElementNum;
        repeatParams.src0RepStride = sfaColumns / blockElementNum;
        repeatParams.src1RepStride = 1;


        if (dLoop <= sfaDealRows) {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, sfaDealRows, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            BinaryRepeatParams columnRepeatParams;
            columnRepeatParams.dstBlkStride = 1;
            columnRepeatParams.src0BlkStride = 1;
            columnRepeatParams.src1BlkStride = 0;
            columnRepeatParams.dstRepStride = 8;
            columnRepeatParams.src0RepStride = 8;
            columnRepeatParams.src1RepStride = 0;
            for (uint32_t i = 0; i < sfaDealRows; i++) {
                Mul(dstUb[i * sfaColumns], src0Ub[i * sfaColumns], src1Ub[i * blockElementNum], repeatElementNum, dLoop,
                    columnRepeatParams);
            }
        }


        if (dRemain > 0) {
            Mul(dstUb[dLoop * repeatElementNum], src0Ub[dLoop * repeatElementNum], src1Ub, dRemain, sfaDealRows,
                repeatParams);
        }
    } else {
        BinaryRepeatParams repeatParams;
        repeatParams.dstRepStride = 8;
        repeatParams.src0RepStride = 8;
        repeatParams.src0BlkStride = 1;
        repeatParams.dstBlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 0;

        for (uint32_t i = 0; i < sfaDealRows; i++) {

            Mul(dstUb[i * sfaColumns], src0Ub[i * sfaColumns], src1Ub[i * blockElementNum], repeatElementNum, dLoop,
                repeatParams);

            if (dRemain > 0) {
                Mul(dstUb[i * sfaColumns + dLoop * repeatElementNum], src0Ub[i * sfaColumns + dLoop * repeatElementNum],
                    src1Ub[i * blockElementNum], dRemain, 1, repeatParams);
            }
        }
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::MergeKvCoop(const RunInfo& sfaVecRunInfo, uint32_t dstCore,
                                                           uint32_t halfIdx)
{
    int64_t s2ProcessSize = sfaVecRunInfo.actualSingleProcessSInnerSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * sfaVecConstInfo.sparseBlockSize);
    int64_t topkGmBaseOffset = 0;

#ifdef K11089_L78_SFA_TOPK
    const uint32_t topkRowStride = (sfaVecConstInfo.sparseBlockCount < 2048U) ? 2048U : sfaVecConstInfo.sparseBlockCount;
#else
    const uint32_t topkRowStride = sfaVecConstInfo.sparseBlockCount;
#endif
    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum =
            (sfaVecRunInfo.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(sfaVecRunInfo.bIdx - 1);
        topkGmBaseOffset += (actualSeqQPrefixSum + sfaVecRunInfo.gS1Idx / sfaVecConstInfo.gSize) *
                                sfaVecConstInfo.kvHeadNum * topkRowStride +
                            sfaVecRunInfo.n2Idx * topkRowStride;
    } else {
        topkGmBaseOffset += sfaVecRunInfo.bIdx * sfaVecConstInfo.qSeqSize * topkRowStride +
                            sfaVecRunInfo.gS1Idx / sfaVecConstInfo.gSize * topkRowStride;
    }
    int64_t mergeMte3Idx = 0;
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    int64_t s2IdxArray0 = -1;
    int64_t s2IdxArray1 = -1;
    bool needWaitMte3ToMte2 = true;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    uint64_t coreBase = (static_cast<uint64_t>(dstCore) * 4ULL + (sfaVecRunInfo.loop & 3U)) * 512ULL * 576ULL;
    int64_t s2GmStartOffset = (halfIdx == 0) ? 0 : CeilDiv(s2Pair, 2L) * 2 * sfaVecConstInfo.sparseBlockSize;
    int64_t s2GmLimit = (halfIdx == 0) ? CeilDiv(s2Pair, 2L) * 2 * sfaVecConstInfo.sparseBlockSize : s2ProcessSize;
    if (s2GmLimit > s2ProcessSize) {
        s2GmLimit = s2ProcessSize;
    }
    uint32_t activeBlocks = static_cast<uint32_t>((sfaVecRunInfo.curActualSeqLenOri + 127ULL) >> 7U);
    if constexpr (PAGE_ATTENTION) {
        if (sfaVecConstInfo.sparseBlockSize == 1 && sfaVecConstInfo.kvCacheBlockSize == 128 &&
            sfaVecConstInfo.kvHeadNum == 1 && sfaVecConstInfo.headDim == 512 &&
            sfaVecConstInfo.headDimRope == 64 && activeBlocks > 0U && activeBlocks <= 512U) {
            int64_t subCount = s2GmLimit - s2GmStartOffset;
            if (subCount > 0) {
                LocalTensor<int32_t> topkUb = v0ValidSizeUb_[512];
                LocalTensor<int32_t> blkUb = v0ValidSizeUb_[1024];
                uint32_t topkAlign = (static_cast<uint32_t>(subCount) + 7U) & ~7U;
                uint32_t blkAlign = (activeBlocks + 7U) & ~7U;
                DataCopy(topkUb,
                         topkGm_[topkGmBaseOffset + sfaVecRunInfo.s2Idx * sfaVecConstInfo.s2BaseSize + s2GmStartOffset],
                         topkAlign);
#ifdef K112130_VEC_RESOLVED_TOPK
                const bool preResolved = (treeMaskGmAddr_ & 1ULL) != 0ULL;
                if (!preResolved && sfaVecRunInfo.isFirstSInnerLoop) {
                    DataCopy(blkUb, blkTableGm_[sfaVecRunInfo.bIdx * sfaVecConstInfo.maxBlockNumPerBatch], blkAlign);
                }
#else
                if (sfaVecRunInfo.isFirstSInnerLoop) {
                    DataCopy(blkUb, blkTableGm_[sfaVecRunInfo.bIdx * sfaVecConstInfo.maxBlockNumPerBatch], blkAlign);
                }
#endif
                SetFlag<AscendC::HardEvent::MTE2_S>(1);
                WaitFlag<AscendC::HardEvent::MTE2_S>(1);
#ifdef K11090_FAST_SFA_VEC
                const uint32_t qRowIdx = static_cast<uint32_t>(sfaVecRunInfo.gS1Idx / sfaVecConstInfo.gSize);
                const int64_t prefixLen =
                    static_cast<int64_t>(sfaVecRunInfo.curActualSeqLenOri) - static_cast<int64_t>(sfaVecRunInfo.actS1Size);
                bool hasRowTreeMask = false;
                uint32_t rowTreeMask = 0xFFFFFFFFU;
                const uint64_t cleanTreeMaskAddr = treeMaskGmAddr_ & ~1ULL;
#ifdef K112130_VEC_RESOLVED_TOPK
                if (!preResolved && cleanTreeMaskAddr != 0ULL) {
#else
                if (cleanTreeMaskAddr != 0ULL) {
#endif
                    const __gm__ uint32_t* maskPtr = reinterpret_cast<const __gm__ uint32_t*>(cleanTreeMaskAddr);
                    if (maskPtr[0] != 0U && qRowIdx < 32U) {
                        hasRowTreeMask = true;
                        rowTreeMask = maskPtr[qRowIdx];
                    }
                }
                int64_t s2IdLimit = sfaVecRunInfo.curActualSeqLenOri;
                if (sfaVecConstInfo.sparseMode == 3) {
                    s2IdLimit = prefixLen + static_cast<int64_t>(qRowIdx) + 1;
                }
#ifdef K112130_VEC_RESOLVED_TOPK
                if (preResolved) {
                    s2IdLimit = 1;
                }
#endif
                int64_t maxTopkRemain =
                    static_cast<int64_t>(sfaVecConstInfo.sparseBlockCount) -
                    static_cast<int64_t>(sfaVecRunInfo.s2Idx * sfaVecConstInfo.s2BaseSize + s2GmStartOffset);
                LocalTensor<uint64_t> topkUb64 = topkUb.template ReinterpretCast<uint64_t>();
#if defined(K112174_FAST_MERGE_KV) && defined(K112130_VEC_RESOLVED_TOPK) && defined(K11120_SFA_GATHER_OPT)
                if (sizeof(KV_T) == 2 && preResolved) {
                    for (int64_t localIdx = 0; localIdx < subCount;) {
                        if (needWaitMte3ToMte2) {
                            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
                            needWaitMte3ToMte2 = false;
                        }
                        const uint64_t pairVal = topkUb64.GetValue(localIdx >> 1);
                        const int32_t ko0 = (localIdx < maxTopkRemain) ? static_cast<int32_t>(pairVal & 0xFFFFFFFFULL) : -1;
                        if (unlikely(ko0 == -1)) {
                            if (mte2Size > mte3Size) {
                                const int32_t m2m3Ev = static_cast<int32_t>(mergeMte3Idx & 1);
                                SetFlag<AscendC::HardEvent::MTE2_MTE3>(m2m3Ev);
                                WaitFlag<AscendC::HardEvent::MTE2_MTE3>(m2m3Ev);
                                uint32_t copyRows = static_cast<uint32_t>(mte2Size - mte3Size);
                                DataCopy(kvMergeBaseGm_[coreBase + (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDim],
                                         kvMergUb_[(mergeMte3Idx & 1) * 32 * 512], copyRows * sfaVecConstInfo.headDim);
                                if constexpr (SFAT::hasRope) {
                                    DataCopy(kvMergeBaseGm_[coreBase + 512ULL * 512ULL +
                                                            (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDimRope],
                                             ropeMergUb_[(mergeMte3Idx & 1) * 32 * 64],
                                             copyRows * sfaVecConstInfo.headDimRope);
                                }
                            }
                            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
                            mergeMte3Idx++;
                            break;
                        }
                        const int32_t ko1 = (localIdx + 1 < subCount && localIdx + 1 < maxTopkRemain)
                                                ? static_cast<int32_t>(pairVal >> 32ULL)
                                                : -2;
                        int64_t stepIdx = 2;
                        if (likely(ko0 >= 0 && ko1 >= 0 && ko0 != ko1)) {
                            const int64_t startGmOffset = (ko1 < ko0) ? static_cast<int64_t>(ko1) : static_cast<int64_t>(ko0);
                            const int64_t endGmOffset = (ko1 < ko0) ? static_cast<int64_t>(ko0) : static_cast<int64_t>(ko1);
                            const int64_t blkTableSrcStride = endGmOffset - startGmOffset - 1;
                            const int64_t ubKvOff = (mergeMte3Idx & 1) * 32 * 512 + (mte2Size - mte3Size) * 512;
                            const int64_t ubRopeOff = (mergeMte3Idx & 1) * 32 * 64 + (mte2Size - mte3Size) * 64;
                            if (blkTableSrcStride == 0) {
                                int64_t runPairs = 1;
                                const int64_t maxPairs = (32 - (mte2Size - mte3Size)) >> 1;
                                const int64_t remPairs = ((subCount < maxTopkRemain ? subCount : maxTopkRemain) - localIdx) >> 1;
                                const int64_t limPairs = (maxPairs < remPairs) ? maxPairs : remPairs;
                                while (runPairs < limPairs) {
                                    const uint64_t nextPair = topkUb64.GetValue((localIdx >> 1) + runPairs);
                                    const int32_t n0 = static_cast<int32_t>(nextPair & 0xFFFFFFFFULL);
                                    const int32_t n1 = static_cast<int32_t>(nextPair >> 32ULL);
                                    const int32_t exp0 = static_cast<int32_t>(startGmOffset + (runPairs << 1));
                                    if (!((n0 == exp0 && n1 == exp0 + 1) || (n1 == exp0 && n0 == exp0 + 1))) {
                                        break;
                                    }
                                    runPairs++;
                                }
                                DataCopy(kvMergUb_[ubKvOff], keyGm_[startGmOffset * 512], static_cast<uint32_t>(runPairs * 1024));
                                if constexpr (SFAT::hasRope) {
                                    DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64], static_cast<uint32_t>(runPairs * 128));
                                }
                                mte2Size += (runPairs << 1);
                                stepIdx = (runPairs << 1);
                            } else if (blkTableSrcStride <= 2047) {
                                DataCopy(kvMergUb_[ubKvOff], keyGm_[startGmOffset * 512],
                                         DataCopyParams{2, 32, static_cast<uint16_t>(blkTableSrcStride * 32), 0});
                                if constexpr (SFAT::hasRope) {
                                    DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64],
                                             DataCopyParams{2, 4, static_cast<uint16_t>(blkTableSrcStride * 4), 0});
                                }
                                mte2Size += 2;
                            } else {
                                DataCopy(kvMergUb_[ubKvOff], keyGm_[startGmOffset * 512], 512);
                                DataCopy(kvMergUb_[ubKvOff + 512], keyGm_[endGmOffset * 512], 512);
                                if constexpr (SFAT::hasRope) {
                                    if (likely(blkTableSrcStride <= 16383)) {
                                        DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64],
                                                 DataCopyParams{2, 4, static_cast<uint16_t>(blkTableSrcStride * 4), 0});
                                    } else {
                                        DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64], 64);
                                        DataCopy(ropeMergUb_[ubRopeOff + 64], keyRopeGm_[endGmOffset * 64], 64);
                                    }
                                }
                                mte2Size += 2;
                            }
                        } else {
                            if (ko0 >= 0) {
                                const int64_t ubKvOff = (mergeMte3Idx & 1) * 32 * 512 + (mte2Size - mte3Size) * 512;
                                const int64_t ubRopeOff = (mergeMte3Idx & 1) * 32 * 64 + (mte2Size - mte3Size) * 64;
                                DataCopy(kvMergUb_[ubKvOff], keyGm_[static_cast<int64_t>(ko0) * 512], 512);
                                if constexpr (SFAT::hasRope) {
                                    DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[static_cast<int64_t>(ko0) * 64], 64);
                                }
                                mte2Size += 1;
                            }
                            if (ko1 >= 0) {
                                const int64_t ubKvOff = (mergeMte3Idx & 1) * 32 * 512 + (mte2Size - mte3Size) * 512;
                                const int64_t ubRopeOff = (mergeMte3Idx & 1) * 32 * 64 + (mte2Size - mte3Size) * 64;
                                DataCopy(kvMergUb_[ubKvOff], keyGm_[static_cast<int64_t>(ko1) * 512], 512);
                                if constexpr (SFAT::hasRope) {
                                    DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[static_cast<int64_t>(ko1) * 64], 64);
                                }
                                mte2Size += 1;
                            }
                        }
                        localIdx += stepIdx;
                        if ((mte2Size - mte3Size + 2 > 32) || (localIdx >= subCount)) {
                            if (mte2Size > mte3Size) {
                                const int32_t m2m3Ev = static_cast<int32_t>(mergeMte3Idx & 1);
                                SetFlag<AscendC::HardEvent::MTE2_MTE3>(m2m3Ev);
                                WaitFlag<AscendC::HardEvent::MTE2_MTE3>(m2m3Ev);
                                uint32_t copyRows = static_cast<uint32_t>(mte2Size - mte3Size);
                                DataCopy(kvMergeBaseGm_[coreBase + (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDim],
                                         kvMergUb_[(mergeMte3Idx & 1) * 32 * 512], copyRows * sfaVecConstInfo.headDim);
                                if constexpr (SFAT::hasRope) {
                                    DataCopy(kvMergeBaseGm_[coreBase + 512ULL * 512ULL +
                                                            (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDimRope],
                                             ropeMergUb_[(mergeMte3Idx & 1) * 32 * 64],
                                             copyRows * sfaVecConstInfo.headDimRope);
                                }
                            }
                            mte3Size = mte2Size;
                            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
                            mergeMte3Idx++;
                            needWaitMte3ToMte2 = true;
                        }
                    }
                } else
#endif
                for (int64_t localIdx = 0; localIdx < subCount; localIdx += 2) {
                    if (needWaitMte3ToMte2) {
                        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
                        needWaitMte3ToMte2 = false;
                    }
                    const uint64_t pairVal = topkUb64.GetValue(localIdx >> 1);
                    const int32_t raw0 = static_cast<int32_t>(pairVal & 0xFFFFFFFFULL);
                    const int32_t raw1 = static_cast<int32_t>(pairVal >> 32ULL);
                    int64_t keyOffset1 = -1;
                    int64_t keyOffset2 = -1;
#ifdef K112130_VEC_RESOLVED_TOPK
                    if (preResolved) {
                        const int32_t ko0 = (localIdx < maxTopkRemain) ? raw0 : -1;
                        if (unlikely(ko0 == -1)) {
                            if (mte2Size > mte3Size) {
                                SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
                                WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
                                uint32_t copyRows = static_cast<uint32_t>(mte2Size - mte3Size);
                                DataCopy(kvMergeBaseGm_[coreBase + (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDim],
                                         kvMergUb_[(mergeMte3Idx & 1) * 32 * 512], copyRows * sfaVecConstInfo.headDim);
                                if constexpr (SFAT::hasRope) {
                                    DataCopy(kvMergeBaseGm_[coreBase + 512ULL * 512ULL +
                                                            (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDimRope],
                                             ropeMergUb_[(mergeMte3Idx & 1) * 32 * 64],
                                             copyRows * sfaVecConstInfo.headDimRope);
                                }
                            }
                            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
                            mergeMte3Idx++;
                            break;
                        }
                        const int32_t ko1 = (localIdx + 1 < subCount && localIdx + 1 < maxTopkRemain) ? raw1 : -2;
                        keyOffset1 = (ko0 >= 0) ? static_cast<int64_t>(ko0) : -1;
                        keyOffset2 = (ko1 >= 0) ? static_cast<int64_t>(ko1) : -1;
                        s2IdxArray0 = 0;
                        s2IdxArray1 = 0;
                    } else
#endif
                    {
                        s2IdxArray0 = (localIdx < maxTopkRemain) ? static_cast<int64_t>(raw0) : -1;
                        if (unlikely(s2IdxArray0 < 0)) {
                            if (mte2Size > mte3Size) {
                                SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
                                WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
                                uint32_t copyRows = static_cast<uint32_t>(mte2Size - mte3Size);
                                DataCopy(kvMergeBaseGm_[coreBase + (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDim],
                                         kvMergUb_[(mergeMte3Idx & 1) * 32 * 512], copyRows * sfaVecConstInfo.headDim);
                                if constexpr (SFAT::hasRope) {
                                    DataCopy(kvMergeBaseGm_[coreBase + 512ULL * 512ULL +
                                                            (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDimRope],
                                             ropeMergUb_[(mergeMte3Idx & 1) * 32 * 64],
                                             copyRows * sfaVecConstInfo.headDimRope);
                                }
                            }
                            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
                            mergeMte3Idx++;
                            break;
                        }
                        s2IdxArray1 = (localIdx + 1 < subCount && localIdx + 1 < maxTopkRemain) ?
                                          static_cast<int64_t>(raw1) :
                                          -1;
                        if (s2IdxArray0 >= 0 && s2IdxArray0 < s2IdLimit &&
                            likely(!hasRowTreeMask || s2IdxArray0 < prefixLen ||
                                   (static_cast<uint32_t>(s2IdxArray0 - prefixLen) < 32U &&
                                    ((rowTreeMask >> static_cast<uint32_t>(s2IdxArray0 - prefixLen)) & 1U) != 0U))) {
                            const uint32_t u0 = static_cast<uint32_t>(s2IdxArray0);
                            keyOffset1 = (static_cast<int64_t>(blkUb.GetValue(u0 >> 7U)) << 7) + (u0 & 127U);
                        }
                        if (s2IdxArray1 >= 0 && s2IdxArray1 < s2IdLimit &&
                            likely(!hasRowTreeMask || s2IdxArray1 < prefixLen ||
                                   (static_cast<uint32_t>(s2IdxArray1 - prefixLen) < 32U &&
                                    ((rowTreeMask >> static_cast<uint32_t>(s2IdxArray1 - prefixLen)) & 1U) != 0U))) {
                            const uint32_t u1 = static_cast<uint32_t>(s2IdxArray1);
                            keyOffset2 = (static_cast<int64_t>(blkUb.GetValue(u1 >> 7U)) << 7) + (u1 & 127U);
                        }
                    }
                    if (likely(keyOffset1 >= 0 || keyOffset2 >= 0)) {
                        int64_t diff =
                            (keyOffset1 > keyOffset2) ? (keyOffset1 - keyOffset2) : (keyOffset2 - keyOffset1);
                        int64_t blkTableSrcStride = diff - 1;
                        int64_t keySrcStride = blkTableSrcStride * 512 * sizeof(KV_T);
                        if (unlikely(keyOffset1 < 0 || keyOffset2 < 0 || keySrcStride >= INT32_MAX ||
                                     keySrcStride < 0)) {
                            CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, keyOffset1, s2IdLimit,
                                           sfaVecRunInfo);
                            CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray1, keyOffset2, s2IdLimit,
                                           sfaVecRunInfo);
                        } else {
                            int64_t startGmOffset = (keyOffset2 < keyOffset1) ? keyOffset2 : keyOffset1;
#ifdef K11120_SFA_GATHER_OPT
                            const int64_t endGmOffset = (keyOffset2 < keyOffset1) ? keyOffset1 : keyOffset2;
                            const int64_t ubKvOff = (mergeMte3Idx & 1) * 32 * 512 + (mte2Size - mte3Size) * 512;
                            const int64_t ubRopeOff = (mergeMte3Idx & 1) * 32 * 64 + (mte2Size - mte3Size) * 64;
                            if constexpr (sizeof(KV_T) == 2) {
#ifdef K112162_MERGE_CONTIG
                                if (blkTableSrcStride == 0) {
                                    DataCopy(kvMergUb_[ubKvOff], keyGm_[startGmOffset * 512], 1024);
                                    if constexpr (SFAT::hasRope) {
                                        DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64], 128);
                                    }
                                } else
#endif
                                if (blkTableSrcStride <= 2047) {
                                    DataCopy(kvMergUb_[ubKvOff], keyGm_[startGmOffset * 512],
                                             DataCopyParams{2, 32, static_cast<uint16_t>(blkTableSrcStride * 32), 0});
                                    if constexpr (SFAT::hasRope) {
                                        DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64],
                                                 DataCopyParams{2, 4, static_cast<uint16_t>(blkTableSrcStride * 4), 0});
                                    }
                                } else {
                                    DataCopy(kvMergUb_[ubKvOff], keyGm_[startGmOffset * 512], 512);
                                    DataCopy(kvMergUb_[ubKvOff + 512], keyGm_[endGmOffset * 512], 512);
                                    if constexpr (SFAT::hasRope) {
#ifdef K112174_FAST_MERGE_KV
                                        if (likely(blkTableSrcStride <= 16383)) {
                                            DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64],
                                                     DataCopyParams{2, 4, static_cast<uint16_t>(blkTableSrcStride * 4), 0});
                                        } else
#endif
                                        {
                                            DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64], 64);
                                            DataCopy(ropeMergUb_[ubRopeOff + 64], keyRopeGm_[endGmOffset * 64], 64);
                                        }
                                    }
                                }
                                mte2Size += 2;
                            } else if constexpr (sizeof(KV_T) == 1) {
                                if (blkTableSrcStride <= 4095) {
                                    DataCopy(kvMergUb_[ubKvOff], keyGm_[startGmOffset * 512],
                                             DataCopyParams{2, 16, static_cast<uint16_t>(blkTableSrcStride * 16), 0});
                                    if constexpr (SFAT::hasRope) {
                                        DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64],
                                                 DataCopyParams{2, 2, static_cast<uint16_t>(blkTableSrcStride * 2), 0});
                                    }
                                } else {
                                    DataCopy(kvMergUb_[ubKvOff], keyGm_[startGmOffset * 512], 512);
                                    DataCopy(kvMergUb_[ubKvOff + 512], keyGm_[endGmOffset * 512], 512);
                                    if constexpr (SFAT::hasRope) {
                                        DataCopy(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64], 64);
                                        DataCopy(ropeMergUb_[ubRopeOff + 64], keyRopeGm_[endGmOffset * 64], 64);
                                    }
                                }
                                mte2Size += 2;
                            } else {
                                DataCopyExtParams intriParams{2, 512 * sizeof(KV_T), static_cast<uint32_t>(keySrcStride), 0, 0};
                                DataCopyPadExtParams<KV_T> padParams;
                                DataCopyPad(kvMergUb_[ubKvOff], keyGm_[startGmOffset * 512], intriParams, padParams);
                                if constexpr (SFAT::hasRope) {
                                    intriParams.blockLen = 64 * sizeof(KV_T);
                                    intriParams.srcStride = blkTableSrcStride * 64 * sizeof(KV_T);
                                    DataCopyPad(ropeMergUb_[ubRopeOff], keyRopeGm_[startGmOffset * 64], intriParams, padParams);
                                }
                                mte2Size += 2;
                            }
#else
                            if constexpr (sizeof(KV_T) == 2) {
                                if (likely(blkTableSrcStride <= 2047)) {
                                    DataCopy(kvMergUb_[(mergeMte3Idx & 1) * 32 * 512 + (mte2Size - mte3Size) * 512],
                                             keyGm_[startGmOffset * 512],
                                             DataCopyParams{2, 32, static_cast<uint16_t>(blkTableSrcStride * 32), 0});
                                    if constexpr (SFAT::hasRope) {
                                        DataCopy(ropeMergUb_[(mergeMte3Idx & 1) * 32 * 64 + (mte2Size - mte3Size) * 64],
                                                 keyRopeGm_[startGmOffset * 64],
                                                 DataCopyParams{2, 4, static_cast<uint16_t>(blkTableSrcStride * 4), 0});
                                    }
                                    mte2Size += 2;
                                } else {
                                    DataCopyExtParams intriParams{2, 512 * sizeof(KV_T), static_cast<uint32_t>(keySrcStride), 0, 0};
                                    DataCopyPadExtParams<KV_T> padParams;
                                    DataCopyPad(kvMergUb_[(mergeMte3Idx & 1) * 32 * 512 + (mte2Size - mte3Size) * 512],
                                                keyGm_[startGmOffset * 512], intriParams, padParams);
                                    if constexpr (SFAT::hasRope) {
                                        intriParams.blockLen = 64 * sizeof(KV_T);
                                        intriParams.srcStride = blkTableSrcStride * 64 * sizeof(KV_T);
                                        DataCopyPad(ropeMergUb_[(mergeMte3Idx & 1) * 32 * 64 + (mte2Size - mte3Size) * 64],
                                                    keyRopeGm_[startGmOffset * 64], intriParams, padParams);
                                    }
                                    mte2Size += 2;
                                }
                            } else {
                                DataCopyExtParams intriParams{2, 512 * sizeof(KV_T), static_cast<uint32_t>(keySrcStride), 0, 0};
                                DataCopyPadExtParams<KV_T> padParams;
                                DataCopyPad(kvMergUb_[(mergeMte3Idx & 1) * 32 * 512 + (mte2Size - mte3Size) * 512],
                                            keyGm_[startGmOffset * 512], intriParams, padParams);
                                if constexpr (SFAT::hasRope) {
                                    intriParams.blockLen = 64 * sizeof(KV_T);
                                    intriParams.srcStride = blkTableSrcStride * 64 * sizeof(KV_T);
                                    DataCopyPad(ropeMergUb_[(mergeMte3Idx & 1) * 32 * 64 + (mte2Size - mte3Size) * 64],
                                                keyRopeGm_[startGmOffset * 64], intriParams, padParams);
                                }
                                mte2Size += 2;
                            }
#endif
                        }
                    }
#else
                int64_t s2IdLimit = sfaVecRunInfo.curActualSeqLenOri;
                if (sfaVecConstInfo.sparseMode == 3) {
                    s2IdLimit = sfaVecRunInfo.curActualSeqLenOri - sfaVecRunInfo.actS1Size +
                                sfaVecRunInfo.gS1Idx / sfaVecConstInfo.gSize + 1;
                }
                int64_t maxTopkRemain =
                    static_cast<int64_t>(sfaVecConstInfo.sparseBlockCount) -
                    static_cast<int64_t>(sfaVecRunInfo.s2Idx * sfaVecConstInfo.s2BaseSize + s2GmStartOffset);
                for (int64_t localIdx = 0; localIdx < subCount; localIdx += 2) {
                    if (needWaitMte3ToMte2) {
                        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
                        needWaitMte3ToMte2 = false;
                    }
                    s2IdxArray0 = (localIdx < maxTopkRemain) ? static_cast<int64_t>(topkUb.GetValue(localIdx)) : -1;
                    if (unlikely(s2IdxArray0 < 0)) {
                        if (mte2Size > mte3Size) {
                            SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
                            WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
                            uint32_t copyRows = static_cast<uint32_t>(mte2Size - mte3Size);
                            DataCopy(kvMergeBaseGm_[coreBase + (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDim],
                                     kvMergUb_[(mergeMte3Idx & 1) * 32 * 512], copyRows * sfaVecConstInfo.headDim);
                            if constexpr (SFAT::hasRope) {
                                DataCopy(kvMergeBaseGm_[coreBase + 512ULL * 512ULL +
                                                        (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDimRope],
                                         ropeMergUb_[(mergeMte3Idx & 1) * 32 * 64],
                                         copyRows * sfaVecConstInfo.headDimRope);
                            }
                        }
                        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
                        mergeMte3Idx++;
                        break;
                    }
                    s2IdxArray1 = (localIdx + 1 < subCount && localIdx + 1 < maxTopkRemain) ?
                                      static_cast<int64_t>(topkUb.GetValue(localIdx + 1)) :
                                      -1;
                    int64_t keyOffset1 = -1;
                    int64_t keyOffset2 = -1;
                    if (IsTreeKeyValid(s2IdxArray0, sfaVecRunInfo, s2IdLimit)) {
                        uint32_t u0 = static_cast<uint32_t>(s2IdxArray0);
                        keyOffset1 = (static_cast<int64_t>(blkUb.GetValue(u0 >> 7)) << 7) + (u0 & 127U);
                    }
                    if (IsTreeKeyValid(s2IdxArray1, sfaVecRunInfo, s2IdLimit)) {
                        uint32_t u1 = static_cast<uint32_t>(s2IdxArray1);
                        keyOffset2 = (static_cast<int64_t>(blkUb.GetValue(u1 >> 7)) << 7) + (u1 & 127U);
                    }
                    if (likely(keyOffset1 >= 0 || keyOffset2 >= 0)) {
                        int64_t diff =
                            (keyOffset1 > keyOffset2) ? (keyOffset1 - keyOffset2) : (keyOffset2 - keyOffset1);
                        int64_t blkTableSrcStride = diff - 1;
                        int64_t keySrcStride = blkTableSrcStride * 512 * sizeof(KV_T);
                        if (unlikely(keyOffset1 < 0 || keyOffset2 < 0 || keySrcStride >= INT32_MAX ||
                                     keySrcStride < 0)) {
                            CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, keyOffset1, s2IdLimit,
                                           sfaVecRunInfo);
                            CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray1, keyOffset2, s2IdLimit,
                                           sfaVecRunInfo);
                        } else {
                            DataCopyExtParams intriParams;
                            intriParams.blockLen = 512 * sizeof(KV_T);
                            intriParams.blockCount = 2;
                            intriParams.dstStride = 0;
                            intriParams.srcStride = keySrcStride;
                            DataCopyPadExtParams<KV_T> padParams;
                            int64_t startGmOffset = (keyOffset2 < keyOffset1) ? keyOffset2 : keyOffset1;
                            DataCopyPad(kvMergUb_[(mergeMte3Idx & 1) * 32 * 512 + (mte2Size - mte3Size) * 512],
                                        keyGm_[startGmOffset * 512], intriParams, padParams);
                            if constexpr (SFAT::hasRope) {
                                intriParams.blockLen = 64 * sizeof(KV_T);
                                intriParams.srcStride = blkTableSrcStride * 64 * sizeof(KV_T);
                                DataCopyPad(ropeMergUb_[(mergeMte3Idx & 1) * 32 * 64 + (mte2Size - mte3Size) * 64],
                                            keyRopeGm_[startGmOffset * 64], intriParams, padParams);
                            }
                            mte2Size += 2;
                        }
                    }
#endif
                    if ((mte2Size - mte3Size + 2 > 32) || (localIdx + 2 >= subCount)) {
                        if (mte2Size > mte3Size) {
                            SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
                            WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
                            uint32_t copyRows = static_cast<uint32_t>(mte2Size - mte3Size);
                            DataCopy(kvMergeBaseGm_[coreBase + (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDim],
                                     kvMergUb_[(mergeMte3Idx & 1) * 32 * 512], copyRows * sfaVecConstInfo.headDim);
                            if constexpr (SFAT::hasRope) {
                                DataCopy(kvMergeBaseGm_[coreBase + 512ULL * 512ULL +
                                                        (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDimRope],
                                         ropeMergUb_[(mergeMte3Idx & 1) * 32 * 64],
                                         copyRows * sfaVecConstInfo.headDimRope);
                            }
                        }
                        mte3Size = mte2Size;
                        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
                        mergeMte3Idx++;
                        needWaitMte3ToMte2 = true;
                    }
                }
            }
        } else {
            for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit;
                 s2GmOffsetArray += 2 * sfaVecConstInfo.sparseBlockSize) {
                if (needWaitMte3ToMte2) {
                    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
                    needWaitMte3ToMte2 = false;
                }
                GetRealS2Idx(s2GmOffsetArray, s2IdxArray0, topkGmBaseOffset, sfaVecRunInfo);
                if (unlikely(s2IdxArray0 < 0)) {
                    if (mte2Size > mte3Size) {
                        SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
                        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
                        uint32_t copyRows = static_cast<uint32_t>(mte2Size - mte3Size);
                        DataCopy(kvMergeBaseGm_[coreBase + (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDim],
                                 kvMergUb_[(mergeMte3Idx & 1) * 32 * 512], copyRows * sfaVecConstInfo.headDim);
                        if constexpr (SFAT::hasRope) {
                            DataCopy(kvMergeBaseGm_[coreBase + 512ULL * 512ULL +
                                                    (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDimRope],
                                     ropeMergUb_[(mergeMte3Idx & 1) * 32 * 64],
                                     copyRows * sfaVecConstInfo.headDimRope);
                        }
                    }
                    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
                    mergeMte3Idx++;
                    break;
                }
                GetRealS2Idx(s2GmOffsetArray + sfaVecConstInfo.sparseBlockSize, s2IdxArray1, topkGmBaseOffset,
                             sfaVecRunInfo);
                CopyInKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1, sfaVecRunInfo);
                if ((mte2Size - mte3Size + 2 * sfaVecConstInfo.sparseBlockSize > 32) ||
                    s2GmOffsetArray + 2 * sfaVecConstInfo.sparseBlockSize >= s2GmLimit) {
                    if (mte2Size > mte3Size) {
                        SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
                        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
                        uint32_t copyRows = static_cast<uint32_t>(mte2Size - mte3Size);
                        DataCopy(kvMergeBaseGm_[coreBase + (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDim],
                                 kvMergUb_[(mergeMte3Idx & 1) * 32 * 512], copyRows * sfaVecConstInfo.headDim);
                        if constexpr (SFAT::hasRope) {
                            DataCopy(kvMergeBaseGm_[coreBase + 512ULL * 512ULL +
                                                    (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDimRope],
                                     ropeMergUb_[(mergeMte3Idx & 1) * 32 * 64],
                                     copyRows * sfaVecConstInfo.headDimRope);
                        }
                    }
                    mte3Size = mte2Size;
                    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
                    mergeMte3Idx++;
                    needWaitMte3ToMte2 = true;
                }
            }
        }
    }

#ifdef K11090_FAST_SFA_VEC
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    if (unlikely(s2GmStartOffset + mte2Size < s2GmLimit)) {
        SetFlag<AscendC::HardEvent::MTE2_V>(0);
        WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        const int64_t remRows = s2GmLimit - (s2GmStartOffset + mte2Size);
        const uint32_t zeroRows = (remRows > 64) ? 64U : static_cast<uint32_t>(remRows);
        Duplicate(kvMergUb_, static_cast<KV_T>(0.0), zeroRows * sfaVecConstInfo.headDim);
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset += 64) {
            const uint32_t chunkR = (s2GmLimit - s2GmOffset > 64) ? 64U : static_cast<uint32_t>(s2GmLimit - s2GmOffset);
            DataCopy(kvMergeBaseGm_[coreBase + static_cast<uint64_t>(s2GmOffset) * sfaVecConstInfo.headDim],
                     kvMergUb_, chunkR * sfaVecConstInfo.headDim);
            if constexpr (SFAT::hasRope) {
                DataCopy(kvMergeBaseGm_[coreBase + 512ULL * sfaVecConstInfo.headDim +
                                        static_cast<uint64_t>(s2GmOffset) * sfaVecConstInfo.headDimRope],
                         kvMergUb_, chunkR * sfaVecConstInfo.headDimRope);
            }
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    }
#else
    if (unlikely(s2GmStartOffset + mte2Size < s2GmLimit)) {
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
        Duplicate(kvMergUb_, static_cast<KV_T>(0.0), sfaVecConstInfo.headDim);
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = sfaVecConstInfo.headDim * sizeof(KV_T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset++) {
            DataCopyPad(kvMergeBaseGm_[coreBase + s2GmOffset * sfaVecConstInfo.headDim], kvMergUb_, dataCopyParams);
        }
        if constexpr (SFAT::hasRope) {
            dataCopyParams.blockLen = sfaVecConstInfo.headDimRope * sizeof(KV_T);
            for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset++) {
                DataCopyPad(kvMergeBaseGm_[coreBase + 512ULL * sfaVecConstInfo.headDim +
                                           s2GmOffset * sfaVecConstInfo.headDimRope],
                            kvMergUb_, dataCopyParams);
            }
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
        mergeMte3Idx++;
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
#endif
    v0ValidSizeUb_.SetValue(sfaVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM, mte2Size);
    v0ValidSizeUb_.SetValue(sfaVecRunInfo.s2Idx & 3U, mte2Size);
    SetFlag<AscendC::HardEvent::S_MTE3>(1);
    WaitFlag<AscendC::HardEvent::S_MTE3>(1);
    uint64_t validBase =
        (static_cast<uint64_t>(dstCore) * 4ULL + (sfaVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM)) * 256ULL;
    DataCopy(kvValidSizeBaseGm_[validBase + halfIdx * 128ULL], v0ValidSizeUb_, 128);
    SetFlag<AscendC::HardEvent::MTE3_S>(SYNC_INPUT_V0BUF_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_S>(SYNC_INPUT_V0BUF_FLAG);
}

#endif
