

#ifndef SPARSE_FLASH_ATTENTION_SERVICE_CUBE_MLA_H
#define SPARSE_FLASH_ATTENTION_SERVICE_CUBE_MLA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_flash_attention_common.h"
#ifndef SFA_MM2_PAIR_LOAD
#define SFA_MM2_PAIR_LOAD 1
#endif

struct PAShape {
    uint32_t blockSize;
    uint32_t headNum;
    uint32_t headDim;
    uint32_t maxblockNumPerBatch;
    uint32_t actHeadDim;
    uint32_t copyRowNum;
    uint32_t copyRowNumAlign;
};

struct Position {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t s2Idx;
    uint32_t dIdx;
};


template <typename T>
__aicore__ inline void DataCopyGmNDToL1(LocalTensor<T>& l1Tensor, GlobalTensor<T>& gmTensor, uint32_t rowAct,
                                        uint32_t rowAlign,
                                        uint32_t col,
                                        uint32_t colStride)
{
    Nd2NzParams sfaCubeCopyParams;
    sfaCubeCopyParams.ndNum = 1;
    sfaCubeCopyParams.nValue = rowAct;

    sfaCubeCopyParams.dValue = col;
    sfaCubeCopyParams.srcDValue = colStride;
    sfaCubeCopyParams.dstNzC0Stride = rowAlign;
    sfaCubeCopyParams.dstNzNStride = 1;
    sfaCubeCopyParams.srcNdMatrixStride = 0;
    sfaCubeCopyParams.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmTensor, sfaCubeCopyParams);
}


template <typename T, SFA_LAYOUT SRC_LAYOUT>
__aicore__ inline void DataCopyPA(LocalTensor<T>& dstTensor,
                                  GlobalTensor<T>& srcTensor,
                                  GlobalTensor<int32_t>& blockTableGm,
                                  const PAShape& shape,
                                  const Position& startPos)
{
    uint64_t blockTableBaseOffset = startPos.bIdx * shape.maxblockNumPerBatch;
    uint32_t curS2Idx = startPos.s2Idx;
    uint32_t copyFinishRowCnt = 0;
    uint32_t blockElementCnt = 32 / sizeof(T);
    while (copyFinishRowCnt < shape.copyRowNum) {
        uint64_t reaminRowCnt = curS2Idx % shape.blockSize;
        uint64_t blockIdOffset = curS2Idx / shape.blockSize;
        uint64_t idInBlockTable =
            blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);

        uint32_t copyRowCnt = shape.blockSize - reaminRowCnt;
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum) {
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt;
        }
        uint64_t dStride = shape.headDim;
        uint64_t offset = idInBlockTable * shape.blockSize * shape.headNum * shape.headDim;

        if constexpr (SRC_LAYOUT == SFA_LAYOUT::BSND || SRC_LAYOUT == SFA_LAYOUT::TND) {
            offset += (uint64_t)(startPos.n2Idx * shape.headDim) + reaminRowCnt * shape.headDim * shape.headNum +
                      startPos.dIdx;
            dStride = shape.headDim * shape.headNum;
        } else {
            offset += (uint64_t)(startPos.n2Idx * shape.headDim * shape.blockSize) + reaminRowCnt * shape.headDim +
                      startPos.dIdx;
        }

        uint32_t dValue = shape.actHeadDim;
        uint32_t srcDValue = dStride;
        LocalTensor<T> tmpDstTensor = dstTensor[copyFinishRowCnt * blockElementCnt];
        GlobalTensor<T> tmpSrcTensor = srcTensor[offset];

        DataCopyGmNDToL1<T>(tmpDstTensor, tmpSrcTensor, copyRowCnt, shape.copyRowNumAlign, dValue, srcDValue);
        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
}

template <typename SFAT>
class SFAMatmulService {
public:

    using T = float;
    using Q_T = typename SFAT::queryType;
    using KV_T = typename SFAT::kvType;
    using OUT_T = typename SFAT::outputType;
    using MM_OUT_T = T;

    __aicore__ inline SFAMatmulService(){};
    __aicore__ inline void InitParams(const ConstInfo& sfaCubeConstInfo);
    __aicore__ inline void InitMm1GlobalTensor(GlobalTensor<Q_T> queryGm, GlobalTensor<Q_T> qRopeGm,
                                               GlobalTensor<KV_T> keyGm, GlobalTensor<KV_T> kRopeGm,
                                               GlobalTensor<MM_OUT_T> mm1ResGm);
    __aicore__ inline void InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm, GlobalTensor<KV_T> valueGm,
                                               GlobalTensor<MM_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void InitPageAttentionInfo(const GlobalTensor<KV_T>& kvMergeGm,
                                                 GlobalTensor<int32_t> blockTableGm, GlobalTensor<int32_t> topKGm,
                                                 uint32_t blockSize, uint32_t maxBlockNumPerBatch);
    __aicore__ inline void InitBuffers(TPipe* pipe);
    __aicore__ inline void UpdateKey(GlobalTensor<KV_T> keyGm);
    __aicore__ inline void UpdateValue(GlobalTensor<KV_T> valueGm);

    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void CalcTopKBlockInfo(const RunInfo& sfaCubeRunInfo, uint32_t& curTopKIdx,
                                             uint64_t& curOffsetInSparseBlock, uint32_t curSeqIdx, uint32_t& copyRowCnt,
                                             int64_t& idInTopK);
    __aicore__ inline void ComputeMm1(const RunInfo& sfaCubeRunInfo, const MSplitInfo sfaCubeSplitInfo);
    __aicore__ inline void ComputeMm1Coop(const RunInfo& sfaCubeRunInfo, const MSplitInfo sfaCubeSplitInfo,
                                          uint32_t cIdx, uint32_t mm1Slot = 0xFFFFFFFFU);
    __aicore__ inline void ComputeMm1NoRope(const RunInfo& sfaCubeRunInfo, const MSplitInfo sfaCubeSplitInfo);
    __aicore__ inline void ComputeMm2(const RunInfo& sfaCubeRunInfo, const MSplitInfo sfaCubeSplitInfo);

private:
    static constexpr bool PAGE_ATTENTION = SFAT::pageAttention;
    static constexpr int TEMPLATE_MODE = SFAT::templateMode;
    static constexpr bool FLASH_DECODE = SFAT::flashDecode;
    static constexpr SFA_LAYOUT LAYOUT_T = SFAT::layout;
    static constexpr SFA_LAYOUT KV_LAYOUT_T = SFAT::kvLayout;

    static constexpr uint32_t M_SPLIT_SIZE = 128;
    static constexpr uint32_t N_SPLIT_SIZE = 128;
    static constexpr uint32_t N_WORKSPACE_SIZE = 512;

    static constexpr uint32_t L1_BLOCK_SIZE = (64 * (512 + 64) * sizeof(Q_T));
    static constexpr uint32_t L1_BLOCK_OFFSET = 64 * (512 + 64);

    static constexpr uint32_t L0A_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0B_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0C_PP_SIZE = (64 * 1024);


    static constexpr uint32_t L1_EVENT0 = EVENT_ID2;
    static constexpr uint32_t L1_EVENT1 = EVENT_ID3;
    static constexpr uint32_t L1_EVENT2 = EVENT_ID4;
    static constexpr uint32_t L1_EVENT3 = EVENT_ID5;
    static constexpr uint32_t L1_EVENT4 = EVENT_ID6;
    static constexpr uint32_t L1_EVENT5 = EVENT_ID7;
    static constexpr uint32_t L1_EVENT6 = EVENT_ID1;


    static constexpr uint32_t L0AB_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0AB_EVENT1 = EVENT_ID4;

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true};
    static constexpr uint32_t mte21QPIds[4] = {L1_EVENT0, L1_EVENT1, L1_EVENT2, L1_EVENT3};
    static constexpr uint32_t sfaMte21KvEvents[3] = {L1_EVENT4, L1_EVENT5, L1_EVENT6};

    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    ConstInfo sfaCubeConstInfo{};


    uint32_t abL0BufIter = 0;
    uint32_t sfaL0CBufferIndex = 0;
    uint32_t qpL1BufIter = 0;
    uint32_t kvL1BufIter = -1;


    GlobalTensor<Q_T> queryGm;
    GlobalTensor<Q_T> qRopeGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> kRopeGm;
    GlobalTensor<MM_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> kvMergeGm_;
    GlobalTensor<KV_T> kvMergeBaseGm_;
    GlobalTensor<MM_OUT_T> mm1ResBaseGm_;


    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<MM_OUT_T> mm2ResGm;
    GlobalTensor<OUT_T> attentionOutGm;


    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> topKGm;

    TBuf<TPosition::A1> bufQPL1;
    TBuf<TPosition::A1> bufKVL1;
    TBuf<TPosition::A2> tmpBufL0A;
    TBuf<TPosition::B2> tmpBufL0B;
    TBuf<TPosition::CO1> tmpBufL0C;

    LocalTensor<Q_T> l1QPTensor;
    LocalTensor<Q_T> l1KVTensor;
    LocalTensor<KV_T> aL0TensorPingPong;
    LocalTensor<KV_T> bL0TensorPingPong;
    LocalTensor<MM_OUT_T> cL0TensorPingPong;


    __aicore__ inline uint32_t Mte1MmABEventId(uint32_t idx)
    {
        return (L0AB_EVENT0 + idx);
    }

    __aicore__ inline uint32_t GetQPL1RealIdx(uint32_t mIdx, uint32_t k1Idx)
    {
        uint32_t idxMap[] = {0, 2};
        return idxMap[mIdx % 2] + k1Idx;
    }

    __aicore__ inline void CopyGmToL1(LocalTensor<KV_T>& l1Tensor, GlobalTensor<KV_T>& gmSrcTensor, uint32_t srcN,
                                      uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyInMm1AToL1(LocalTensor<KV_T>& aL1Tensor, const RunInfo& sfaCubeRunInfo, uint32_t mSeqIdx,
                                          uint32_t mSizeAct, uint32_t headSize, uint32_t headOffset);
    __aicore__ inline void CopyInMm1ARopeToL1(LocalTensor<KV_T>& aL1Tensor, const RunInfo& sfaCubeRunInfo,
                                              uint32_t mSeqIdx, uint32_t mSizeAct);
    __aicore__ inline void CopyInMm1BToL1(LocalTensor<KV_T>& bL1Tensor, const uint64_t keyGmBaseOffset,
                                          uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                          uint32_t nActCopyRowCount, uint32_t headSize);
    __aicore__ inline void CopyInMm1BRopeToL1(LocalTensor<KV_T>& bL1Tensor, const uint64_t keyGmBaseOffset,
                                              uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                              uint32_t nActCopyRowCount, uint32_t headSize);
    __aicore__ inline void CopyInMm2AToL1(LocalTensor<KV_T>& aL1Tensor, const RunInfo& sfaCubeRunInfo, uint32_t mSeqIdx,
                                          uint32_t subMSizeAct, uint32_t nSize, uint32_t nOffset);
    __aicore__ inline void CopyInMm2BToL1(LocalTensor<KV_T>& bL1Tensor, const uint64_t valueGmBaseOffset,
                                          uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                          uint32_t nActCopyRowCount, uint32_t copyStartColumnCount,
                                          uint32_t copyColumnCount);
    __aicore__ inline void LoadDataMm1A(LocalTensor<KV_T>& aL0Tensor, LocalTensor<KV_T>& aL1Tensor, uint32_t idx,
                                        uint32_t kSplitSize, uint32_t mSize, uint32_t kSize);
    __aicore__ inline void LoadDataMm1B(LocalTensor<KV_T>& bL0Tensor, LocalTensor<KV_T>& bL1Tensor, uint32_t idx,
                                        uint32_t kSplitSize, uint32_t kSize, uint32_t nSize);
};

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::InitParams(const ConstInfo& sfaCubeConstInfo)
{
    this->sfaCubeConstInfo = sfaCubeConstInfo;
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::InitMm1GlobalTensor(GlobalTensor<Q_T> queryGm, GlobalTensor<Q_T> qRopeGm,
                                                                   GlobalTensor<KV_T> keyGm, GlobalTensor<KV_T> kRopeGm,
                                                                   GlobalTensor<MM_OUT_T> mm1ResGm)
{

    this->queryGm = queryGm;
    this->qRopeGm = qRopeGm;
    this->keyGm = keyGm;
    this->kRopeGm = kRopeGm;
    this->mm1ResGm = mm1ResGm;
    uint64_t coreIdx = GetBlockIdx();
    this->mm1ResBaseGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MM_OUT_T*>(mm1ResGm.GetPhyAddr(0)) -
                                        coreIdx * 2ULL * sfaCubeConstInfo.mmResUbSize);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm,
                                                                   GlobalTensor<KV_T> valueGm,
                                                                   GlobalTensor<MM_OUT_T> mm2ResGm,
                                                                   GlobalTensor<OUT_T> attentionOutGm)
{

    this->vec1ResGm = vec1ResGm;
    this->valueGm = valueGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::InitPageAttentionInfo(const GlobalTensor<KV_T>& kvMergeGm,
                                                                     GlobalTensor<int32_t> blockTableGm,
                                                                     GlobalTensor<int32_t> topKGm, uint32_t blockSize,
                                                                     uint32_t maxBlockNumPerBatch)
{
    this->blockTableGm = blockTableGm;
    this->topKGm = topKGm;
    this->kvCacheBlockSize = blockSize;
    this->maxBlockNumPerBatch = maxBlockNumPerBatch;
    this->kvMergeGm_ = kvMergeGm;
    uint64_t coreIdx = GetBlockIdx();
    this->kvMergeBaseGm_.SetGlobalBuffer(reinterpret_cast<__gm__ KV_T*>(kvMergeGm.GetPhyAddr(0)) -
                                         coreIdx * 512ULL * 576ULL * 4ULL);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::InitBuffers(TPipe* pipe)
{
#if defined(K112166_SFA_QV_L1) && defined(ATTENTION_HEADS) && (ATTENTION_HEADS <= 16)
    if constexpr (TEMPLATE_MODE == V_TEMPLATE && sizeof(Q_T) == 2 && sizeof(KV_T) == 2) {
#ifdef K112188_SFA_QVL1_ALL
        const bool enableQvL1 = (sfaCubeConstInfo.qSeqSize >= 1U);
#else
        const bool enableQvL1 = (sfaCubeConstInfo.qSeqSize >= 20U);
#endif
        if (enableQvL1) {
            pipe->InitBuffer(bufQPL1, 65536);
            l1QPTensor = bufQPL1.Get<Q_T>();
            pipe->InitBuffer(bufKVL1, L1_BLOCK_SIZE * 6);
            l1KVTensor = bufKVL1.Get<KV_T>();
        } else {
            pipe->InitBuffer(bufQPL1, L1_BLOCK_SIZE * 4);
            l1QPTensor = bufQPL1.Get<Q_T>();
            pipe->InitBuffer(bufKVL1, L1_BLOCK_SIZE * 3);
            l1KVTensor = bufKVL1.Get<KV_T>();
        }
    } else
#endif
    {
        pipe->InitBuffer(bufQPL1, L1_BLOCK_SIZE * 4);
        l1QPTensor = bufQPL1.Get<Q_T>();
        pipe->InitBuffer(bufKVL1, L1_BLOCK_SIZE * 3);
        l1KVTensor = bufKVL1.Get<KV_T>();
    }


    pipe->InitBuffer(tmpBufL0A, L0A_PP_SIZE * 2);
    aL0TensorPingPong = tmpBufL0A.Get<KV_T>();

    pipe->InitBuffer(tmpBufL0B, L0B_PP_SIZE * 2);
    bL0TensorPingPong = tmpBufL0B.Get<KV_T>();

    pipe->InitBuffer(tmpBufL0C, L0C_PP_SIZE * 2);
    cL0TensorPingPong = tmpBufL0C.Get<MM_OUT_T>();
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::UpdateKey(GlobalTensor<KV_T> keyGm)
{
    this->keyGm = keyGm;
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::UpdateValue(GlobalTensor<KV_T> valueGm)
{
    this->valueGm = valueGm;
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::AllocEventID()
{
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT2);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT3);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT4);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT5);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT6);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT1);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::FreeEventID()
{
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT2);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT3);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT4);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT5);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT6);
    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT1);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::CopyGmToL1(LocalTensor<KV_T>& l1Tensor, GlobalTensor<KV_T>& gmSrcTensor,
                                                          uint32_t srcN, uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams sfaCubeCopyParams;
    sfaCubeCopyParams.ndNum = 1;
    sfaCubeCopyParams.nValue = srcN;
    sfaCubeCopyParams.dValue = srcD;
    sfaCubeCopyParams.srcDValue = srcDstride;
    sfaCubeCopyParams.dstNzC0Stride = (srcN + 15) / 16 * 16;
    sfaCubeCopyParams.dstNzNStride = 1;
    sfaCubeCopyParams.srcNdMatrixStride = 0;
    sfaCubeCopyParams.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, sfaCubeCopyParams);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::CopyInMm1AToL1(LocalTensor<KV_T>& l1Tensor,
                                                              const RunInfo& sfaCubeRunInfo, uint32_t mSeqIdx,
                                                              uint32_t mSizeAct, uint32_t headSize, uint32_t headOffset)
{
    auto srcGm = queryGm[sfaCubeRunInfo.tensorAOffset + mSeqIdx * sfaCubeConstInfo.headDim + headOffset];
    CopyGmToL1(l1Tensor, srcGm, mSizeAct, headSize, sfaCubeConstInfo.headDim);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::CopyInMm1ARopeToL1(LocalTensor<KV_T>& l1Tensor,
                                                                  const RunInfo& sfaCubeRunInfo, uint32_t mSeqIdx,
                                                                  uint32_t mSizeAct)
{
    auto srcGm = qRopeGm[sfaCubeRunInfo.tensorARopeOffset + mSeqIdx * sfaCubeConstInfo.headDimRope];
    CopyGmToL1(l1Tensor, srcGm, mSizeAct, sfaCubeConstInfo.headDimRope, sfaCubeConstInfo.headDimRope);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::CopyInMm1BToL1(LocalTensor<KV_T>& bL1Tensor,
                                                              const uint64_t keyGmBaseOffset,
                                                              uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                                              uint32_t nActCopyRowCount, uint32_t headSize)
{
    uint64_t dStride = sfaCubeConstInfo.headDim;
    if constexpr (KV_LAYOUT_T == SFA_LAYOUT::BSND || KV_LAYOUT_T == SFA_LAYOUT::TND) {
        dStride = sfaCubeConstInfo.headDim * sfaCubeConstInfo.kvHeadNum;
    }

    uint32_t blockElementCnt = 32 / sizeof(KV_T);

    Nd2NzParams mm1Nd2NzParamsForB;
    mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
    mm1Nd2NzParamsForB.dValue = headSize;
    mm1Nd2NzParamsForB.srcDValue = dStride;
    mm1Nd2NzParamsForB.ndNum = 1;
    mm1Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
    mm1Nd2NzParamsForB.dstNzNStride = 1;
    mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
    mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
    DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], keyGm[keyGmBaseOffset], mm1Nd2NzParamsForB);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::CopyInMm1BRopeToL1(LocalTensor<KV_T>& bL1Tensor,
                                                                  const uint64_t kRopeGmBaseOffset,
                                                                  uint32_t copyTotalRowCntAlign,
                                                                  uint32_t copyStartRowCnt, uint32_t nActCopyRowCount,
                                                                  uint32_t headSize)
{
    uint64_t dStride = sfaCubeConstInfo.headDimRope;
    if constexpr (KV_LAYOUT_T == SFA_LAYOUT::BSND || KV_LAYOUT_T == SFA_LAYOUT::TND) {
        dStride = sfaCubeConstInfo.headDimRope * sfaCubeConstInfo.kvHeadNum;
    }

    uint32_t blockElementCnt = 32 / sizeof(KV_T);

    Nd2NzParams mm1Nd2NzParamsForB;
    mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
    mm1Nd2NzParamsForB.dValue = headSize;
    mm1Nd2NzParamsForB.ndNum = 1;
    mm1Nd2NzParamsForB.srcDValue = dStride;
    mm1Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
    mm1Nd2NzParamsForB.dstNzNStride = 1;
    mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
    mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
    DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], kRopeGm[kRopeGmBaseOffset], mm1Nd2NzParamsForB);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::LoadDataMm1A(LocalTensor<KV_T>& aL0Tensor, LocalTensor<KV_T>& aL1Tensor,
                                                            uint32_t idx, uint32_t kSplitSize, uint32_t mSize,
                                                            uint32_t kSize)
{
    LocalTensor<KV_T> srcTensor = aL1Tensor[mSize * kSplitSize * idx];
    LoadData3DParamsV2<KV_T> sfaMm1LoadParams;

    sfaMm1LoadParams.padList[0] = 0;
    sfaMm1LoadParams.padList[1] = 0;
    sfaMm1LoadParams.padList[2] = 0;
    sfaMm1LoadParams.padList[3] = 255;
    sfaMm1LoadParams.l1H = mSize / 16;
    sfaMm1LoadParams.l1W = 16;


    sfaMm1LoadParams.mExtension = mSize;
    sfaMm1LoadParams.kExtension = kSize;
    sfaMm1LoadParams.strideW = 1;
    sfaMm1LoadParams.strideH = 1;
    sfaMm1LoadParams.filterW = 1;
    sfaMm1LoadParams.filterSizeW = (1 >> 8) & 255;
    sfaMm1LoadParams.filterH = 1;
    sfaMm1LoadParams.filterSizeH = (1 >> 8) & 255;
    sfaMm1LoadParams.mStartPt = 0;
    sfaMm1LoadParams.kStartPt = 0;
    sfaMm1LoadParams.dilationFilterW = 1;
    sfaMm1LoadParams.dilationFilterH = 1;
    sfaMm1LoadParams.enTranspose = 0;
    sfaMm1LoadParams.fMatrixCtrl = 0;
    sfaMm1LoadParams.channelSize = kSize;
    LoadData<KV_T, LOAD3DV2_CONFIG>(aL0Tensor, srcTensor, sfaMm1LoadParams);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::LoadDataMm1B(LocalTensor<KV_T>& l0Tensor, LocalTensor<KV_T>& l1Tensor,
                                                            uint32_t idx, uint32_t kSplitSize, uint32_t kSize,
                                                            uint32_t nSize)
{

    LocalTensor<KV_T> srcTensor = l1Tensor[nSize * kSplitSize * idx];

    LoadData2DParams sfaMm1LoadBParams;
    sfaMm1LoadBParams.startIndex = 0;
    sfaMm1LoadBParams.repeatTimes = (nSize + 15) / 16 * kSize / (32 / sizeof(KV_T));
    sfaMm1LoadBParams.srcStride = 1;
    sfaMm1LoadBParams.dstGap = 0;
    sfaMm1LoadBParams.ifTranspose = false;
    LoadData(l0Tensor, srcTensor, sfaMm1LoadBParams);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::CopyInMm2AToL1(LocalTensor<KV_T>& aL1Tensor,
                                                              const RunInfo& sfaCubeRunInfo, uint32_t mSeqIdx,
                                                              uint32_t subMSizeAct, uint32_t nSize, uint32_t nOffset)
{
    auto srcGm = vec1ResGm[(sfaCubeRunInfo.loop % sfaCubeConstInfo.preLoadNum) * sfaCubeConstInfo.mmResUbSize +
                           mSeqIdx * sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign + nOffset];
    CopyGmToL1(aL1Tensor, srcGm, subMSizeAct, nSize, sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::CopyInMm2BToL1(LocalTensor<KV_T>& bL1Tensor,
                                                              const uint64_t valueGmBaseOffset,
                                                              uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                                              uint32_t nActCopyRowCount, uint32_t copyStartColumnCount,
                                                              uint32_t copyColumnCount)
{
    uint64_t step = sfaCubeConstInfo.headDim;
    if constexpr (KV_LAYOUT_T == SFA_LAYOUT::BSND || KV_LAYOUT_T == SFA_LAYOUT::TND) {
        step = sfaCubeConstInfo.headDim * sfaCubeConstInfo.kvHeadNum;
    }

    uint32_t blockElementCnt = 32 / sizeof(KV_T);

    Nd2NzParams mm1Nd2NzParamsForB;
    mm1Nd2NzParamsForB.ndNum = 1;
    mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
    mm1Nd2NzParamsForB.dValue = copyColumnCount;
    mm1Nd2NzParamsForB.srcDValue = step;
    mm1Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
    mm1Nd2NzParamsForB.dstNzNStride = 1;
    mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
    mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
    DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], valueGm[valueGmBaseOffset + copyStartColumnCount],
             mm1Nd2NzParamsForB);
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::CalcTopKBlockInfo(const RunInfo& sfaCubeRunInfo, uint32_t& curTopKIdx,
                                                                 uint64_t& curOffsetInSparseBlock, uint32_t curSeqIdx,
                                                                 uint32_t& copyRowCnt, int64_t& idInTopK)
{
    uint64_t blockBegin = idInTopK * sfaCubeConstInfo.sparseBlockSize;
    uint64_t blockEnd = (blockBegin + sfaCubeConstInfo.sparseBlockSize > sfaCubeRunInfo.threshold) ?
                            sfaCubeRunInfo.threshold :
                            blockBegin + sfaCubeConstInfo.sparseBlockSize;
    uint64_t blockLen = blockEnd - blockBegin;
    if (curOffsetInSparseBlock + copyRowCnt < blockLen) {
        curOffsetInSparseBlock += copyRowCnt;
        copyRowCnt = blockLen - curOffsetInSparseBlock;
    } else {
        for (uint64_t topkidx = curTopKIdx + 1; topkidx < sfaCubeConstInfo.sparseBlockCount; topkidx++) {
            int64_t sparseIndices = topKGm.GetValue(sfaCubeRunInfo.topKBaseOffset + topkidx);
            if (sparseIndices == -1) {
                break;
            }

            uint64_t blockBegin = sparseIndices * sfaCubeConstInfo.sparseBlockSize;
            if (blockBegin >= sfaCubeRunInfo.threshold) {
                continue;
            }
            uint64_t blockEnd = (blockBegin + sfaCubeConstInfo.sparseBlockSize > sfaCubeRunInfo.threshold) ?
                                    sfaCubeRunInfo.threshold :
                                    blockBegin + sfaCubeConstInfo.sparseBlockSize;
            uint64_t blockLen = blockEnd - blockBegin;
            curTopKIdx = topkidx;
            idInTopK = sparseIndices;
            curOffsetInSparseBlock = 0;
            copyRowCnt = blockLen;
            break;
        }
    }
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::ComputeMm1NoRope(const RunInfo& sfaCubeRunInfo,
                                                                const MSplitInfo sfaCubeSplitInfo)
{
    uint32_t mSize = sfaCubeSplitInfo.nBufferDealM;
    uint32_t mm1NrML1Size = M_SPLIT_SIZE;
    uint32_t mm1NrML1SizeAlign = SFAAlign(M_SPLIT_SIZE, 16U);
    uint32_t mm1NrML1Loops = (mSize + M_SPLIT_SIZE - 1) / M_SPLIT_SIZE;

    uint32_t nSize = sfaCubeRunInfo.actualSingleProcessSInnerSize;
    uint32_t mm1NrNL1Size = N_SPLIT_SIZE;
    uint32_t mm1NrNL1SizeAlign = SFAAlign(N_SPLIT_SIZE, 16U);
    uint32_t mm1NrNL1Loops = (nSize + N_SPLIT_SIZE - 1) / N_SPLIT_SIZE;

    constexpr uint32_t mm1NrKL1Loops = 2;
    constexpr uint32_t mm1NrKL0Size = 128;
    constexpr uint32_t mm1NrKL0Loops = 2;
    constexpr uint32_t MERGE_K_PITCH = 576;

    LocalTensor<KV_T> mm1NrBL1Tensor;
    LocalTensor<KV_T> kTensor;
    uint32_t mm1NrKa = 0, mm1NrKb = 0;

    uint32_t curTopKIdx = sfaCubeRunInfo.curTopKIdx;
    uint64_t curOffsetInSparseBlock = sfaCubeRunInfo.curOffsetInSparseBlock;
    uint32_t copyRowCnt = 0;
    int64_t idInTopK = topKGm.GetValue(sfaCubeRunInfo.topKBaseOffset + curTopKIdx);

    uint32_t curTopKIdxTmp = 0;
    uint64_t curOffsetInSparseBlockTmp = 0;
    uint32_t copyRowCntTmp = 0;
    int64_t idInTopKTmp = 0;

    for (uint32_t mm1NrNL1 = 0; mm1NrNL1 < mm1NrNL1Loops; mm1NrNL1++) {
        if (mm1NrNL1 == (mm1NrNL1Loops - 1)) {
            mm1NrNL1Size = nSize - (mm1NrNL1Loops - 1) * N_SPLIT_SIZE;
            mm1NrNL1SizeAlign = SFAAlign(mm1NrNL1Size, 16U);
        }
        curTopKIdxTmp = curTopKIdx;
        curOffsetInSparseBlockTmp = curOffsetInSparseBlock;
        copyRowCntTmp = copyRowCnt;
        idInTopKTmp = idInTopK;

        for (uint32_t mm1NrKL1 = 0; mm1NrKL1 < mm1NrKL1Loops; mm1NrKL1++) {
            kvL1BufIter++;
            mm1NrKb = kvL1BufIter % 3;
            WaitFlag<HardEvent::MTE1_MTE2>(sfaMte21KvEvents[mm1NrKb]);
            mm1NrBL1Tensor = l1KVTensor[mm1NrKb * L1_BLOCK_OFFSET];

            uint32_t curSeqIdx = sfaCubeRunInfo.s2BatchOffset + mm1NrNL1 * N_SPLIT_SIZE;
            uint32_t copyFinishRowCnt = 0;
            curTopKIdx = curTopKIdxTmp;
            curOffsetInSparseBlock = curOffsetInSparseBlockTmp;
            copyRowCnt = copyRowCntTmp;
            idInTopK = idInTopKTmp;
            if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                Nd2NzParams sfaCubeCopyParams;
                sfaCubeCopyParams.ndNum = 1;
                sfaCubeCopyParams.nValue = mm1NrNL1Size;
                sfaCubeCopyParams.dValue = sfaCubeConstInfo.headDim >> 1;
                sfaCubeCopyParams.srcDValue = sfaCubeConstInfo.headDim;
                sfaCubeCopyParams.dstNzC0Stride = mm1NrNL1SizeAlign;
                sfaCubeCopyParams.dstNzNStride = 1;
                sfaCubeCopyParams.srcNdMatrixStride = 0;
                sfaCubeCopyParams.dstNzMatrixStride = 0;
                DataCopy(mm1NrBL1Tensor,
                         kvMergeGm_[sfaCubeRunInfo.loop % 4 * N_WORKSPACE_SIZE * MERGE_K_PITCH +
                                    mm1NrKL1 * (sfaCubeConstInfo.headDim >> 1) +
                                    mm1NrNL1 * N_SPLIT_SIZE * sfaCubeConstInfo.headDim],
                         sfaCubeCopyParams);
            } else {
                while (copyFinishRowCnt < mm1NrNL1Size) {
                    CalcTopKBlockInfo(sfaCubeRunInfo, curTopKIdx, curOffsetInSparseBlock, curSeqIdx, copyRowCnt,
                                      idInTopK);
                    if (copyFinishRowCnt + copyRowCnt > mm1NrNL1Size) {
                        copyRowCnt = mm1NrNL1Size - copyFinishRowCnt;
                    }
                    if constexpr (PAGE_ATTENTION) {
                        Position startPos;
                        startPos.bIdx = sfaCubeRunInfo.bIdx;
                        startPos.n2Idx = sfaCubeRunInfo.n2Idx;
                        startPos.s2Idx = idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock;
                        startPos.dIdx = mm1NrKL1 * 256;
                        PAShape shape;
                        shape.blockSize = kvCacheBlockSize;
                        shape.headNum = sfaCubeConstInfo.kvHeadNum;
                        shape.headDim = sfaCubeConstInfo.headDim;
                        shape.actHeadDim = 256;
                        shape.maxblockNumPerBatch = maxBlockNumPerBatch;
                        shape.copyRowNum = copyRowCnt;
                        shape.copyRowNumAlign = mm1NrNL1SizeAlign;
                        kTensor = mm1NrBL1Tensor[copyFinishRowCnt * 16];
                        DataCopyPA<KV_T, KV_LAYOUT_T>(kTensor, keyGm, blockTableGm, shape, startPos);
                    } else {
                        uint64_t keyOffset = sfaCubeRunInfo.tensorBOffset;
                        if constexpr (KV_LAYOUT_T == SFA_LAYOUT::BSND || KV_LAYOUT_T == SFA_LAYOUT::TND) {
                            keyOffset += (idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                         sfaCubeConstInfo.kvHeadNum * sfaCubeConstInfo.headDim;
                        } else {
                            keyOffset += (idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                         sfaCubeConstInfo.headDim;
                        }
                        CopyInMm1BToL1(mm1NrBL1Tensor, keyOffset + mm1NrKL1 * 256, mm1NrNL1SizeAlign, copyFinishRowCnt,
                                       copyRowCnt, 256);
                    }
                    copyFinishRowCnt += copyRowCnt;
                    curSeqIdx += copyRowCnt;
                }
            }

            SetFlag<HardEvent::MTE2_MTE1>(sfaMte21KvEvents[mm1NrKb]);
            WaitFlag<HardEvent::MTE2_MTE1>(sfaMte21KvEvents[mm1NrKb]);
            mm1NrML1Size = M_SPLIT_SIZE;
            mm1NrML1SizeAlign = SFAAlign(M_SPLIT_SIZE, 16U);
            for (uint32_t mm1NrML1 = 0; mm1NrML1 < mm1NrML1Loops; mm1NrML1++) {
                uint32_t aL1PaddingSize = 0;
                if (mm1NrML1 == (mm1NrML1Loops - 1)) {
                    mm1NrML1Size = mSize - (mm1NrML1Loops - 1) * M_SPLIT_SIZE;
                    mm1NrML1SizeAlign = SFAAlign(mm1NrML1Size, 16U);
                    aL1PaddingSize = (M_SPLIT_SIZE - mm1NrML1SizeAlign) * 256;
                }
                uint32_t mIdx = qpL1BufIter + mm1NrML1;
                mm1NrKa = GetQPL1RealIdx(mIdx, mm1NrKL1);
                LocalTensor<Q_T> mm1NrAL1Tensor =
                    l1QPTensor[mm1NrKa * L1_BLOCK_OFFSET + (1 - mm1NrKL1) * aL1PaddingSize];
                if (mm1NrNL1 == 0) {
                    if (mm1NrKL1 == 0) {
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[mm1NrKa]);
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[mm1NrKa + 1]);
                    }
                    CopyInMm1AToL1(mm1NrAL1Tensor, sfaCubeRunInfo,
                                   sfaCubeSplitInfo.nBufferStartM + mm1NrML1 * M_SPLIT_SIZE, mm1NrML1Size, 256,
                                   mm1NrKL1 * 256);
                    SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[mm1NrKa]);
                    WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[mm1NrKa]);
                }

                LocalTensor mm1NrCL0Tensor =
                    cL0TensorPingPong[(sfaL0CBufferIndex % 2) * (L0C_PP_SIZE / sizeof(MM_OUT_T))];
                for (uint32_t mm1NrKL0 = 0; mm1NrKL0 < mm1NrKL0Loops; mm1NrKL0++) {
                    WaitFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    LocalTensor<KV_T> mm1NrAL0Tensor =
                        aL0TensorPingPong[(abL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];
                    LoadDataMm1A(mm1NrAL0Tensor, mm1NrAL1Tensor, mm1NrKL0, mm1NrKL0Size, mm1NrML1SizeAlign,
                                 mm1NrKL0Size);
                    LocalTensor<KV_T> mm1NrBL0Tensor =
                        bL0TensorPingPong[(abL0BufIter % 2) * (L0B_PP_SIZE / sizeof(KV_T))];
                    LoadDataMm1B(mm1NrBL0Tensor, mm1NrBL1Tensor, mm1NrKL0, mm1NrKL0Size, mm1NrKL0Size,
                                 mm1NrNL1SizeAlign);
                    SetFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));
                    WaitFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));

                    MmadParams mm1NrMmadParams;
                    mm1NrMmadParams.m = mm1NrML1SizeAlign;
                    mm1NrMmadParams.n = mm1NrNL1SizeAlign;
                    mm1NrMmadParams.k = mm1NrKL0Size;
                    mm1NrMmadParams.cmatrixInitVal = (mm1NrKL1 == 0 && mm1NrKL0 == 0);
                    mm1NrMmadParams.cmatrixSource = false;
                    mm1NrMmadParams.unitFlag = (mm1NrKL1 == 1 && mm1NrKL0 == (mm1NrKL0Loops - 1)) ? 0b11 : 0b10;
                    Mmad(mm1NrCL0Tensor, mm1NrAL0Tensor, mm1NrBL0Tensor, mm1NrMmadParams);
                    if ((mm1NrMmadParams.m / 16) * (mm1NrMmadParams.n / 16) < 10) {
                        PipeBarrier<PIPE_M>();
                    }
                    SetFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    abL0BufIter++;
                }

                if (mm1NrNL1 == (mm1NrNL1Loops - 1)) {
                    SetFlag<HardEvent::MTE1_MTE2>(mte21QPIds[mm1NrKa]);
                }
                if (mm1NrKL1 == 1) {
                    FixpipeParamsV220 mm1NrFixParams;
                    mm1NrFixParams.srcStride = mm1NrML1SizeAlign;
                    mm1NrFixParams.nSize = mm1NrNL1SizeAlign;
                    mm1NrFixParams.mSize = mm1NrML1SizeAlign;
                    mm1NrFixParams.dstStride = sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign;
                    mm1NrFixParams.unitFlag = 0b11;
                    mm1NrFixParams.ndNum = 1;
                    Fixpipe(
                        mm1ResGm[(sfaCubeRunInfo.loop % (sfaCubeConstInfo.preLoadNum)) * sfaCubeConstInfo.mmResUbSize +
                                 mm1NrNL1 * N_SPLIT_SIZE +
                                 (sfaCubeSplitInfo.nBufferStartM + mm1NrML1 * M_SPLIT_SIZE) *
                                     sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign],
                        mm1NrCL0Tensor, mm1NrFixParams);
                }
                if (mm1NrML1Loops == 2) {
                    sfaL0CBufferIndex++;
                }
            }
            SetFlag<HardEvent::MTE1_MTE2>(sfaMte21KvEvents[mm1NrKb]);
        }
        if (mm1NrML1Loops == 1) {
            sfaL0CBufferIndex++;
        }
    }
    qpL1BufIter += mm1NrML1Loops;
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::ComputeMm1(const RunInfo& sfaCubeRunInfo,
                                                          const MSplitInfo sfaCubeSplitInfo)
{

    uint32_t mSize = sfaCubeSplitInfo.nBufferDealM;
    uint32_t mL1Size = M_SPLIT_SIZE;
    uint32_t mL1SizeAlign = SFAAlign(M_SPLIT_SIZE, 16U);
    uint32_t sfaML1LoopCount = (mSize + M_SPLIT_SIZE - 1) / M_SPLIT_SIZE;

    uint32_t nSize = sfaCubeRunInfo.actualSingleProcessSInnerSize;
    uint32_t nL1Size = N_SPLIT_SIZE;
    uint32_t nL1SizeAlign = SFAAlign(N_SPLIT_SIZE, 16U);
    uint32_t nL1Loops = (nSize + N_SPLIT_SIZE - 1) / N_SPLIT_SIZE;

    uint32_t kSize = 576;
    uint32_t kL1Size = 288;
    uint32_t kL1Loops = 2;

    uint32_t kL0Size = 96;
    uint32_t kL0Loops = (kL1Size + kL0Size - 1) / kL0Size;

    LocalTensor<KV_T> bL1Tensor;
    LocalTensor<KV_T> kRopeTensor;
    LocalTensor<KV_T> kTensor;

    uint32_t sfaQueryBufferIndex = 0, sfaKvBufferIndex = 0;

#if defined(K112166_SFA_QV_L1) && defined(K11110_SFA_PIPE) && defined(ATTENTION_HEADS) && (ATTENTION_HEADS <= 16)
    if constexpr (TEMPLATE_MODE == V_TEMPLATE && sizeof(Q_T) == 2 && sizeof(KV_T) == 2) {
        const uint32_t totalStepsQv = nL1Loops * kL1Loops;
#ifdef K112188_SFA_QVL1_ALL
        const bool enableQvL1 = (sfaCubeConstInfo.qSeqSize >= 1U);
#else
        const bool enableQvL1 = (sfaCubeConstInfo.qSeqSize >= 20U);
#endif
        if (enableQvL1 && mSize <= 16U) {
            if (totalStepsQv == 0U) {
                return;
            }
            const bool useVCache =
                (!sfaCubeRunInfo.useCoopMm1) &&
                (sfaCubeConstInfo.headDim == 512U) &&
                (N_SPLIT_SIZE == 128U) &&
                (sfaCubeSplitInfo.nBufferStartM == 0U) &&
                (sfaCubeSplitInfo.nBufferDealM == sfaCubeRunInfo.actMBaseSize) &&
                (sfaCubeRunInfo.actualSingleProcessSInnerSize == 512U);
            const uint32_t vSlot = sfaCubeRunInfo.loop & 1U;
            const uint32_t vEv = vSlot ? L1_EVENT6 : L1_EVENT1;
            struct KvTicket {
                uint32_t l1Off;
                uint32_t ev;
                bool cached;
            };
            auto issue_kv_mm1_qv = [&](uint32_t step) __aicore__ -> KvTicket {
                const uint32_t sn = step >> 1U;
                const uint32_t sk = step & 1U;
                const uint32_t snSize =
                    (sn + 1U == nL1Loops) ? (nSize - (nL1Loops - 1U) * N_SPLIT_SIZE) : N_SPLIT_SIZE;
                const uint32_t snAlign = SFAAlign(snSize, 16U);
                uint32_t l1Off = 0U;
                uint32_t ev = L1_EVENT4;
                bool isCached = false;
                if (useVCache && (step == 0U || step == 2U)) {
                    if (step == 0U) {
                        WaitFlag<HardEvent::MTE1_MTE2>(vEv);
                    }
                    l1Off = (2U + vSlot * 2U + sn) * L1_BLOCK_OFFSET;
                    ev = vEv;
                    isCached = true;
                } else {
                    const uint32_t sSlot = kvL1BufIter & 1U;
                    kvL1BufIter++;
                    l1Off = sSlot * L1_BLOCK_OFFSET;
                    ev = sSlot ? L1_EVENT5 : L1_EVENT4;
                    WaitFlag<HardEvent::MTE1_MTE2>(ev);
                }
                LocalTensor<KV_T> dst = l1KVTensor[l1Off];
                Nd2NzParams cp;
                cp.ndNum = 1;
                cp.nValue = snSize;
                cp.dstNzC0Stride = snAlign;
                cp.dstNzNStride = 1;
                cp.srcNdMatrixStride = 0;
                cp.dstNzMatrixStride = 0;
                const uint64_t baseOff = (sfaCubeRunInfo.loop % 4U) * N_WORKSPACE_SIZE * kSize;
                if (sk == 0U) {
                    cp.dValue = sfaCubeConstInfo.headDim >> 1;
                    cp.srcDValue = sfaCubeConstInfo.headDim;
                    DataCopy(dst, kvMergeGm_[baseOff + sn * N_SPLIT_SIZE * sfaCubeConstInfo.headDim], cp);
                    cp.dValue = sfaCubeConstInfo.headDimRope >> 1;
                    cp.srcDValue = sfaCubeConstInfo.headDimRope;
                    DataCopy(dst[snAlign * (sfaCubeConstInfo.headDim >> 1)],
                             kvMergeGm_[baseOff + N_WORKSPACE_SIZE * sfaCubeConstInfo.headDim +
                                        sn * N_SPLIT_SIZE * sfaCubeConstInfo.headDimRope],
                             cp);
                } else {
                    LocalTensor<Q_T> kTmp = dst[(sfaCubeConstInfo.headDimRope >> 1) * snAlign];
                    cp.dValue = sfaCubeConstInfo.headDim >> 1;
                    cp.srcDValue = sfaCubeConstInfo.headDim;
                    DataCopy(kTmp,
                             kvMergeGm_[baseOff + (sfaCubeConstInfo.headDim >> 1) +
                                        sn * N_SPLIT_SIZE * sfaCubeConstInfo.headDim],
                             cp);
                    cp.dValue = sfaCubeConstInfo.headDimRope >> 1;
                    cp.srcDValue = sfaCubeConstInfo.headDimRope;
                    DataCopy(dst,
                             kvMergeGm_[baseOff + N_WORKSPACE_SIZE * sfaCubeConstInfo.headDim +
                                        (sfaCubeConstInfo.headDimRope >> 1) +
                                        sn * N_SPLIT_SIZE * sfaCubeConstInfo.headDimRope],
                             cp);
                }
                SetFlag<HardEvent::MTE2_MTE1>(ev);
                return KvTicket{l1Off, ev, isCached};
            };
            mL1Size = mSize;
            mL1SizeAlign = 16U;
            const uint32_t qTag =
                0x80000000U |
                (static_cast<uint32_t>((sfaCubeRunInfo.tensorAOffset >> 9U) & 0xFFFFFU)) |
                ((sfaCubeSplitInfo.nBufferStartM & 31U) << 20U) |
                ((mSize & 31U) << 25U);
            const bool needLoadQ = (qpL1BufIter != qTag);
            qpL1BufIter = qTag;
            KvTicket curTicket = issue_kv_mm1_qv(0U);
            WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT0);
            if (needLoadQ) {
                CopyInMm1AToL1(l1QPTensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM, mL1Size, 256, 0);
                LocalTensor<Q_T> qRopeTensor = l1QPTensor[4096U];
                CopyInMm1ARopeToL1(qRopeTensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM, mL1Size);
                LocalTensor<Q_T> qHiTensor = l1QPTensor[5120U];
                CopyInMm1AToL1(qHiTensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM, mL1Size, 256, 256);
                SetFlag<HardEvent::MTE2_MTE1>(L1_EVENT0);
                WaitFlag<HardEvent::MTE2_MTE1>(L1_EVENT0);
            }
            for (uint32_t nL1 = 0U; nL1 < nL1Loops; nL1++) {
                if (nL1 + 1U == nL1Loops) {
                    nL1Size = nSize - (nL1Loops - 1U) * N_SPLIT_SIZE;
                    nL1SizeAlign = SFAAlign(nL1Size, 16U);
                }
                for (uint32_t kL1 = 0U; kL1 < kL1Loops; kL1++) {
                    const uint32_t curStep = nL1 * 2U + kL1;
                    LocalTensor<Q_T> aL1Tensor = l1QPTensor[kL1 * 4608U];
                    const KvTicket nextTicket =
                        (curStep + 1U < totalStepsQv) ? issue_kv_mm1_qv(curStep + 1U) : KvTicket{0U, L1_EVENT4, false};
                    WaitFlag<HardEvent::MTE2_MTE1>(curTicket.ev);
                    bL1Tensor = l1KVTensor[curTicket.l1Off];
                    LocalTensor cL0Tensor =
                        cL0TensorPingPong[(sfaL0CBufferIndex % 2U) * (L0C_PP_SIZE / sizeof(MM_OUT_T))];
                    for (uint32_t kL0 = 0U; kL0 < kL0Loops; kL0++) {
                        WaitFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2U));
                        LocalTensor<KV_T> aL0Tensor =
                            aL0TensorPingPong[(abL0BufIter % 2U) * (L0A_PP_SIZE / sizeof(KV_T))];
                        LoadDataMm1A(aL0Tensor, aL1Tensor, kL0, kL0Size, mL1SizeAlign, kL0Size);
                        LocalTensor<KV_T> bL0Tensor =
                            bL0TensorPingPong[(abL0BufIter % 2U) * (L0B_PP_SIZE / sizeof(KV_T))];
                        LoadDataMm1B(bL0Tensor, bL1Tensor, kL0, kL0Size, kL0Size, nL1SizeAlign);
                        SetFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2U));
                        WaitFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2U));
                        MmadParams sfaMmadParams;
                        sfaMmadParams.m = mL1SizeAlign;
                        sfaMmadParams.n = nL1SizeAlign;
                        sfaMmadParams.k = kL0Size;
                        sfaMmadParams.cmatrixInitVal = (kL1 == 0U && kL0 == 0U);
                        sfaMmadParams.cmatrixSource = false;
                        sfaMmadParams.unitFlag = (kL1 == 1U && kL0 + 1U == kL0Loops) ? 0b11 : 0b10;
                        Mmad(cL0Tensor, aL0Tensor, bL0Tensor, sfaMmadParams);
                        if ((sfaMmadParams.m / 16U) * (sfaMmadParams.n / 16U) < 10U) {
                            PipeBarrier<PIPE_M>();
                        }
                        SetFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2U));
                        abL0BufIter++;
                    }
                    if (nL1 + 1U == nL1Loops && kL1 == 1U) {
                        SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT0);
                    }
                    if (kL1 == 1U) {
                        FixpipeParamsV220 fixParams;
                        fixParams.srcStride = mL1SizeAlign;
                        fixParams.nSize = nL1SizeAlign;
                        fixParams.mSize = mL1SizeAlign;
                        fixParams.dstStride = sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign;
                        fixParams.unitFlag = 0b11;
                        fixParams.ndNum = 1;
                        Fixpipe(
                            mm1ResGm[(sfaCubeRunInfo.loop % sfaCubeConstInfo.preLoadNum) * sfaCubeConstInfo.mmResUbSize +
                                     nL1 * N_SPLIT_SIZE +
                                     sfaCubeSplitInfo.nBufferStartM * sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign],
                            cL0Tensor, fixParams);
                    }
                    if (!curTicket.cached) {
                        SetFlag<HardEvent::MTE1_MTE2>(curTicket.ev);
                    }
                    curTicket = nextTicket;
                }
                sfaL0CBufferIndex++;
            }
            return;
        }
    }
#endif
#ifdef K11110_SFA_PIPE
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        const uint32_t totalSteps = nL1Loops * kL1Loops;
        if (sfaML1LoopCount == 1U && totalSteps > 0U) {
            auto issue_kv_mm1 = [&](uint32_t step) __aicore__ -> uint32_t {
                const uint32_t sn = step >> 1U;
                const uint32_t sk = step & 1U;
                const uint32_t snSize =
                    (sn + 1U == nL1Loops) ? (nSize - (nL1Loops - 1U) * N_SPLIT_SIZE) : N_SPLIT_SIZE;
                const uint32_t snAlign = SFAAlign(snSize, 16U);
                kvL1BufIter++;
                const uint32_t bufIdx = kvL1BufIter % 3U;
                WaitFlag<HardEvent::MTE1_MTE2>(sfaMte21KvEvents[bufIdx]);
                LocalTensor<KV_T> dst = l1KVTensor[bufIdx * L1_BLOCK_OFFSET];
                Nd2NzParams cp;
                cp.ndNum = 1;
                cp.nValue = snSize;
                cp.dstNzC0Stride = snAlign;
                cp.dstNzNStride = 1;
                cp.srcNdMatrixStride = 0;
                cp.dstNzMatrixStride = 0;
                const uint64_t baseOff = (sfaCubeRunInfo.loop % 4U) * N_WORKSPACE_SIZE * kSize;
                if (sk == 0U) {
                    cp.dValue = sfaCubeConstInfo.headDim >> 1;
                    cp.srcDValue = sfaCubeConstInfo.headDim;
                    DataCopy(dst, kvMergeGm_[baseOff + sn * N_SPLIT_SIZE * sfaCubeConstInfo.headDim], cp);
                    cp.dValue = sfaCubeConstInfo.headDimRope >> 1;
                    cp.srcDValue = sfaCubeConstInfo.headDimRope;
                    DataCopy(dst[snAlign * (sfaCubeConstInfo.headDim >> 1)],
                             kvMergeGm_[baseOff + N_WORKSPACE_SIZE * sfaCubeConstInfo.headDim +
                                        sn * N_SPLIT_SIZE * sfaCubeConstInfo.headDimRope],
                             cp);
                } else {
                    LocalTensor<Q_T> kTmp = dst[(sfaCubeConstInfo.headDimRope >> 1) * snAlign];
                    cp.dValue = sfaCubeConstInfo.headDim >> 1;
                    cp.srcDValue = sfaCubeConstInfo.headDim;
                    DataCopy(kTmp,
                             kvMergeGm_[baseOff + (sfaCubeConstInfo.headDim >> 1) +
                                        sn * N_SPLIT_SIZE * sfaCubeConstInfo.headDim],
                             cp);
                    cp.dValue = sfaCubeConstInfo.headDimRope >> 1;
                    cp.srcDValue = sfaCubeConstInfo.headDimRope;
                    DataCopy(dst,
                             kvMergeGm_[baseOff + N_WORKSPACE_SIZE * sfaCubeConstInfo.headDim +
                                        (sfaCubeConstInfo.headDimRope >> 1) +
                                        sn * N_SPLIT_SIZE * sfaCubeConstInfo.headDimRope],
                             cp);
                }
                SetFlag<HardEvent::MTE2_MTE1>(sfaMte21KvEvents[bufIdx]);
                return bufIdx;
            };
            mL1Size = mSize;
            mL1SizeAlign = SFAAlign(mL1Size, 16U);
            const uint32_t aL1PaddingSize = (M_SPLIT_SIZE - mL1SizeAlign) * 288U;
            uint32_t curKvBuf = issue_kv_mm1(0U);
            for (uint32_t nL1 = 0U; nL1 < nL1Loops; nL1++) {
                if (nL1 + 1U == nL1Loops) {
                    nL1Size = nSize - (nL1Loops - 1U) * N_SPLIT_SIZE;
                    nL1SizeAlign = SFAAlign(nL1Size, 16U);
                }
                for (uint32_t kL1 = 0U; kL1 < kL1Loops; kL1++) {
                    const uint32_t curStep = nL1 * 2U + kL1;
                    sfaQueryBufferIndex = GetQPL1RealIdx(qpL1BufIter, kL1);
                    LocalTensor<Q_T> aL1Tensor =
                        l1QPTensor[sfaQueryBufferIndex * L1_BLOCK_OFFSET + (1U - kL1) * aL1PaddingSize];
                    if (nL1 == 0U) {
                        if (kL1 == 0U) {
                            WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[sfaQueryBufferIndex]);
                            WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[sfaQueryBufferIndex + 1U]);
                            CopyInMm1AToL1(aL1Tensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM, mL1Size, 256, 0);
                            LocalTensor<Q_T> qRopeTensor = aL1Tensor[mL1SizeAlign * 256];
                            CopyInMm1ARopeToL1(qRopeTensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM, mL1Size);
                        } else {
                            LocalTensor<Q_T> qTmpTensor = aL1Tensor[mL1SizeAlign * 32];
                            CopyInMm1AToL1(qTmpTensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM, mL1Size, 256, 256);
                        }
                        SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[sfaQueryBufferIndex]);
                    }
                    const uint32_t nextKvBuf = (curStep + 1U < totalSteps) ? issue_kv_mm1(curStep + 1U) : 0U;
                    WaitFlag<HardEvent::MTE2_MTE1>(sfaMte21KvEvents[curKvBuf]);
                    if (nL1 == 0U) {
                        WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[sfaQueryBufferIndex]);
                    }
                    bL1Tensor = l1KVTensor[curKvBuf * L1_BLOCK_OFFSET];
                    LocalTensor cL0Tensor =
                        cL0TensorPingPong[(sfaL0CBufferIndex % 2U) * (L0C_PP_SIZE / sizeof(MM_OUT_T))];
                    for (uint32_t kL0 = 0U; kL0 < kL0Loops; kL0++) {
                        WaitFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2U));
                        LocalTensor<KV_T> aL0Tensor =
                            aL0TensorPingPong[(abL0BufIter % 2U) * (L0A_PP_SIZE / sizeof(KV_T))];
                        LoadDataMm1A(aL0Tensor, aL1Tensor, kL0, kL0Size, mL1SizeAlign, kL0Size);
                        LocalTensor<KV_T> bL0Tensor =
                            bL0TensorPingPong[(abL0BufIter % 2U) * (L0B_PP_SIZE / sizeof(KV_T))];
                        LoadDataMm1B(bL0Tensor, bL1Tensor, kL0, kL0Size, kL0Size, nL1SizeAlign);
                        SetFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2U));
                        WaitFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2U));
                        MmadParams sfaMmadParams;
                        sfaMmadParams.m = mL1SizeAlign;
                        sfaMmadParams.n = nL1SizeAlign;
                        sfaMmadParams.k = kL0Size;
                        sfaMmadParams.cmatrixInitVal = (kL1 == 0U && kL0 == 0U);
                        sfaMmadParams.cmatrixSource = false;
                        sfaMmadParams.unitFlag = (kL1 == 1U && kL0 + 1U == kL0Loops) ? 0b11 : 0b10;
                        Mmad(cL0Tensor, aL0Tensor, bL0Tensor, sfaMmadParams);
                        if ((sfaMmadParams.m / 16U) * (sfaMmadParams.n / 16U) < 10U) {
                            PipeBarrier<PIPE_M>();
                        }
                        SetFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2U));
                        abL0BufIter++;
                    }
                    if (nL1 + 1U == nL1Loops) {
                        SetFlag<HardEvent::MTE1_MTE2>(mte21QPIds[sfaQueryBufferIndex]);
                    }
                    if (kL1 == 1U) {
                        FixpipeParamsV220 fixParams;
                        fixParams.srcStride = mL1SizeAlign;
                        fixParams.nSize = nL1SizeAlign;
                        fixParams.mSize = mL1SizeAlign;
                        fixParams.dstStride = sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign;
                        fixParams.unitFlag = 0b11;
                        fixParams.ndNum = 1;
                        Fixpipe(
                            mm1ResGm[(sfaCubeRunInfo.loop % sfaCubeConstInfo.preLoadNum) * sfaCubeConstInfo.mmResUbSize +
                                     nL1 * N_SPLIT_SIZE +
                                     sfaCubeSplitInfo.nBufferStartM * sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign],
                            cL0Tensor, fixParams);
                    }
                    SetFlag<HardEvent::MTE1_MTE2>(sfaMte21KvEvents[curKvBuf]);
                    curKvBuf = nextKvBuf;
                }
                sfaL0CBufferIndex++;
            }
            qpL1BufIter += 1U;
            return;
        }
    }
#endif

    uint32_t curTopKIdx = sfaCubeRunInfo.curTopKIdx;
    uint64_t curOffsetInSparseBlock = sfaCubeRunInfo.curOffsetInSparseBlock;
    uint32_t copyRowCnt = 0;
    int64_t idInTopK = topKGm.GetValue(sfaCubeRunInfo.topKBaseOffset + curTopKIdx);

    uint32_t curTopKIdxTmp = 0;
    uint64_t curOffsetInSparseBlockTmp = 0;
    uint32_t copyRowCntTmp = 0;
    int64_t idInTopKTmp = 0;


    for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) {
        if (nL1 == (nL1Loops - 1)) {

            nL1Size = nSize - (nL1Loops - 1) * N_SPLIT_SIZE;
            nL1SizeAlign = SFAAlign(nL1Size, 16U);
        }
        curTopKIdxTmp = curTopKIdx;
        curOffsetInSparseBlockTmp = curOffsetInSparseBlock;
        copyRowCntTmp = copyRowCnt;
        idInTopKTmp = idInTopK;

        for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) {
            kvL1BufIter++;
            uint32_t sfaKvBufferIndex = kvL1BufIter % 3;
            WaitFlag<HardEvent::MTE1_MTE2>(sfaMte21KvEvents[sfaKvBufferIndex]);

            bL1Tensor = l1KVTensor[sfaKvBufferIndex * L1_BLOCK_OFFSET];


            uint32_t curSeqIdx = sfaCubeRunInfo.s2BatchOffset + nL1 * N_SPLIT_SIZE;
            uint32_t copyFinishRowCnt = 0;
            curTopKIdx = curTopKIdxTmp;
            curOffsetInSparseBlock = curOffsetInSparseBlockTmp;
            copyRowCnt = copyRowCntTmp;
            idInTopK = idInTopKTmp;
            if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                if (kL1 == 0) {
                    Nd2NzParams sfaCubeCopyParams;
                    sfaCubeCopyParams.ndNum = 1;
                    sfaCubeCopyParams.nValue = nL1Size;
                    sfaCubeCopyParams.dValue = sfaCubeConstInfo.headDim >> 1;
                    sfaCubeCopyParams.srcDValue = sfaCubeConstInfo.headDim;
                    sfaCubeCopyParams.dstNzC0Stride = nL1SizeAlign;
                    sfaCubeCopyParams.dstNzNStride = 1;
                    sfaCubeCopyParams.srcNdMatrixStride = 0;
                    sfaCubeCopyParams.dstNzMatrixStride = 0;
                    DataCopy(bL1Tensor,
                             kvMergeGm_[sfaCubeRunInfo.loop % 4 * N_WORKSPACE_SIZE * kSize +
                                        nL1 * N_SPLIT_SIZE * sfaCubeConstInfo.headDim],
                             sfaCubeCopyParams);
                    sfaCubeCopyParams.dValue = sfaCubeConstInfo.headDimRope >> 1;
                    sfaCubeCopyParams.srcDValue = sfaCubeConstInfo.headDimRope;
                    DataCopy(bL1Tensor[nL1SizeAlign * (sfaCubeConstInfo.headDim >> 1)],
                             kvMergeGm_[sfaCubeRunInfo.loop % 4 * N_WORKSPACE_SIZE * kSize +
                                        N_WORKSPACE_SIZE * sfaCubeConstInfo.headDim +
                                        nL1 * N_SPLIT_SIZE * sfaCubeConstInfo.headDimRope],
                             sfaCubeCopyParams);
                } else {
                    LocalTensor<Q_T> kTmpTensor = bL1Tensor[(sfaCubeConstInfo.headDimRope >> 1) * nL1SizeAlign];
                    Nd2NzParams sfaCubeCopyParams;
                    sfaCubeCopyParams.ndNum = 1;
                    sfaCubeCopyParams.nValue = nL1Size;
                    sfaCubeCopyParams.dValue = sfaCubeConstInfo.headDim >> 1;
                    sfaCubeCopyParams.srcDValue = sfaCubeConstInfo.headDim;
                    sfaCubeCopyParams.dstNzC0Stride = nL1SizeAlign;
                    sfaCubeCopyParams.dstNzNStride = 1;
                    sfaCubeCopyParams.srcNdMatrixStride = 0;
                    sfaCubeCopyParams.dstNzMatrixStride = 0;
                    DataCopy(
                        kTmpTensor,
                        kvMergeGm_[sfaCubeRunInfo.loop % 4 * N_WORKSPACE_SIZE * kSize +
                                   (sfaCubeConstInfo.headDim >> 1) + nL1 * N_SPLIT_SIZE * sfaCubeConstInfo.headDim],
                        sfaCubeCopyParams);
                    sfaCubeCopyParams.dValue = sfaCubeConstInfo.headDimRope >> 1;
                    sfaCubeCopyParams.srcDValue = sfaCubeConstInfo.headDimRope;
                    DataCopy(
                        bL1Tensor,
                        kvMergeGm_[sfaCubeRunInfo.loop % 4 * N_WORKSPACE_SIZE * kSize +
                                   N_WORKSPACE_SIZE * sfaCubeConstInfo.headDim + (sfaCubeConstInfo.headDimRope >> 1) +
                                   nL1 * N_SPLIT_SIZE * sfaCubeConstInfo.headDimRope],
                        sfaCubeCopyParams);
                }
            } else {
                while (copyFinishRowCnt < nL1Size) {
                    CalcTopKBlockInfo(sfaCubeRunInfo, curTopKIdx, curOffsetInSparseBlock, curSeqIdx, copyRowCnt,
                                      idInTopK);
                    if (copyFinishRowCnt + copyRowCnt > nL1Size) {
                        copyRowCnt = nL1Size - copyFinishRowCnt;
                    }


                    if constexpr (PAGE_ATTENTION) {
                        Position startPos;
                        startPos.bIdx = sfaCubeRunInfo.bIdx;
                        startPos.n2Idx = sfaCubeRunInfo.n2Idx;
                        startPos.s2Idx = idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock;

                        startPos.dIdx = kL1 * 256;
                        Position ropeStartPos = startPos;
                        ropeStartPos.dIdx = kL1 * 32;
                        PAShape shape;
                        shape.blockSize = kvCacheBlockSize;
                        shape.headNum = sfaCubeConstInfo.kvHeadNum;
                        shape.headDim = sfaCubeConstInfo.headDim;
                        shape.actHeadDim = 256;
                        shape.maxblockNumPerBatch = maxBlockNumPerBatch;
                        shape.copyRowNum = copyRowCnt;
                        shape.copyRowNumAlign = nL1SizeAlign;
                        PAShape ropeShape = shape;
                        ropeShape.headDim = sfaCubeConstInfo.headDimRope;
                        ropeShape.actHeadDim = 32;
                        if (kL1 == 0) {
                            kTensor = bL1Tensor[copyFinishRowCnt * 16];
                            DataCopyPA<KV_T, KV_LAYOUT_T>(kTensor, keyGm, blockTableGm, shape, startPos);
                            kRopeTensor = bL1Tensor[(nL1SizeAlign * (BlockAlign<KV_T>(sfaCubeConstInfo.headDim) >> 1)) +
                                                    copyFinishRowCnt * 16];
                            DataCopyPA<KV_T, KV_LAYOUT_T>(kRopeTensor, kRopeGm, blockTableGm, ropeShape, ropeStartPos);
                        } else {
                            kRopeTensor = bL1Tensor[copyFinishRowCnt * 16];
                            DataCopyPA<KV_T, KV_LAYOUT_T>(kRopeTensor, kRopeGm, blockTableGm, ropeShape, ropeStartPos);
                            LocalTensor<Q_T> kTmpTensor = bL1Tensor[32 * nL1SizeAlign + copyFinishRowCnt * 16];
                            DataCopyPA<KV_T, KV_LAYOUT_T>(kTmpTensor, keyGm, blockTableGm, shape, startPos);
                        }
                    } else {
                        uint64_t keyOffset = sfaCubeRunInfo.tensorBOffset;
                        uint64_t kRopeOffset = sfaCubeRunInfo.tensorBRopeOffset;
                        if constexpr (KV_LAYOUT_T == SFA_LAYOUT::BSND || KV_LAYOUT_T == SFA_LAYOUT::TND) {
                            keyOffset += (idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                         sfaCubeConstInfo.kvHeadNum * sfaCubeConstInfo.headDim;
                            kRopeOffset += (idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                           sfaCubeConstInfo.kvHeadNum * sfaCubeConstInfo.headDimRope;
                        } else {
                            keyOffset += (idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                         sfaCubeConstInfo.headDim;
                            kRopeOffset += (idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                           sfaCubeConstInfo.headDimRope;
                        }

                        if (kL1 == 0) {
                            CopyInMm1BToL1(bL1Tensor, keyOffset, nL1SizeAlign, copyFinishRowCnt, copyRowCnt, 256);
                            kRopeTensor = bL1Tensor[nL1SizeAlign * (BlockAlign<KV_T>(sfaCubeConstInfo.headDim) >> 1)];
                            CopyInMm1BRopeToL1(kRopeTensor, kRopeOffset, nL1SizeAlign, copyFinishRowCnt, copyRowCnt,
                                               32);
                        } else {
                            kRopeTensor = bL1Tensor;
                            CopyInMm1BRopeToL1(kRopeTensor, kRopeOffset + 32, nL1SizeAlign, copyFinishRowCnt,
                                               copyRowCnt, 32);
                            LocalTensor<Q_T> kTmpTensor = bL1Tensor[nL1SizeAlign * 32];
                            CopyInMm1BToL1(kTmpTensor, keyOffset + 256, nL1SizeAlign, copyFinishRowCnt, copyRowCnt,
                                           256);
                        }
                    }


                    copyFinishRowCnt += copyRowCnt;
                    curSeqIdx += copyRowCnt;
                }
            }

            SetFlag<HardEvent::MTE2_MTE1>(sfaMte21KvEvents[sfaKvBufferIndex]);
            WaitFlag<HardEvent::MTE2_MTE1>(sfaMte21KvEvents[sfaKvBufferIndex]);
            mL1Size = M_SPLIT_SIZE;
            mL1SizeAlign = SFAAlign(M_SPLIT_SIZE, 16U);
            for (uint32_t mL1 = 0; mL1 < sfaML1LoopCount; mL1++) {
                uint32_t aL1PaddingSize = 0;
                if (mL1 == (sfaML1LoopCount - 1)) {

                    mL1Size = mSize - (sfaML1LoopCount - 1) * M_SPLIT_SIZE;
                    mL1SizeAlign = SFAAlign(mL1Size, 16U);

                    aL1PaddingSize = (M_SPLIT_SIZE - mL1SizeAlign) * 288;
                }


                uint32_t mIdx = qpL1BufIter + mL1;
                sfaQueryBufferIndex = GetQPL1RealIdx(mIdx, kL1);
                LocalTensor<Q_T> aL1Tensor =
                    l1QPTensor[sfaQueryBufferIndex * L1_BLOCK_OFFSET + (1 - kL1) * aL1PaddingSize];
                if (nL1 == 0) {
                    if (kL1 == 0) {
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[sfaQueryBufferIndex]);
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[sfaQueryBufferIndex + 1]);
                        CopyInMm1AToL1(aL1Tensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE,
                                       mL1Size, 256, 0);

                        LocalTensor<Q_T> qRopeTensor = aL1Tensor[mL1SizeAlign * 256];
                        CopyInMm1ARopeToL1(qRopeTensor, sfaCubeRunInfo,
                                           sfaCubeSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE, mL1Size);
                    } else {

                        LocalTensor<Q_T> qTmpTensor = aL1Tensor[mL1SizeAlign * 32];
                        CopyInMm1AToL1(qTmpTensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE,
                                       mL1Size, 256, 256);
                    }
                    SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[sfaQueryBufferIndex]);
                    WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[sfaQueryBufferIndex]);
                }


                LocalTensor cL0Tensor =
                    cL0TensorPingPong[(sfaL0CBufferIndex % 2) *
                                      (L0C_PP_SIZE / sizeof(MM_OUT_T))];
                for (uint32_t kL0 = 0; kL0 < kL0Loops; kL0++) {
                    WaitFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(abL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];
                    LoadDataMm1A(aL0Tensor, aL1Tensor, kL0, kL0Size, mL1SizeAlign, kL0Size);
                    LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(abL0BufIter % 2) * (L0B_PP_SIZE / sizeof(KV_T))];
                    LoadDataMm1B(bL0Tensor, bL1Tensor, kL0, kL0Size, kL0Size, nL1SizeAlign);
                    SetFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));
                    WaitFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));


                    MmadParams sfaMmadParams;
                    sfaMmadParams.m = mL1SizeAlign;
                    sfaMmadParams.n = nL1SizeAlign;
                    sfaMmadParams.k = kL0Size;
                    sfaMmadParams.cmatrixInitVal = (kL1 == 0 && kL0 == 0);
                    sfaMmadParams.cmatrixSource = false;
                    sfaMmadParams.unitFlag =
                        (kL1 == 1 && kL0 == (kL0Loops - 1)) ? 0b11 : 0b10;
                    Mmad(cL0Tensor, aL0Tensor, bL0Tensor, sfaMmadParams);

                    if ((sfaMmadParams.m / 16) * (sfaMmadParams.n / 16) < 10) {
                        PipeBarrier<PIPE_M>();
                    }
                    SetFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    abL0BufIter++;
                }

                if (nL1 == (nL1Loops - 1)) {
                    SetFlag<HardEvent::MTE1_MTE2>(
                        mte21QPIds[sfaQueryBufferIndex]);
                }

                if (kL1 == 1) {
                    FixpipeParamsV220 fixParams;
                    fixParams.srcStride = mL1SizeAlign;
                    fixParams.nSize = nL1SizeAlign;
                    fixParams.mSize = mL1SizeAlign;

                    fixParams.dstStride = sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign;
                    fixParams.unitFlag = 0b11;
                    fixParams.ndNum = 1;


                    Fixpipe(
                        mm1ResGm[(sfaCubeRunInfo.loop % (sfaCubeConstInfo.preLoadNum)) * sfaCubeConstInfo.mmResUbSize +
                                 nL1 * N_SPLIT_SIZE +
                                 (sfaCubeSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE) *
                                     sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign],
                        cL0Tensor, fixParams);
                }
                if (sfaML1LoopCount == 2) {
                    sfaL0CBufferIndex++;
                }
            }
            SetFlag<HardEvent::MTE1_MTE2>(sfaMte21KvEvents[sfaKvBufferIndex]);
        }
        if (sfaML1LoopCount == 1) {
            sfaL0CBufferIndex++;
        }
    }
    qpL1BufIter += sfaML1LoopCount;
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::ComputeMm2(const RunInfo& sfaCubeRunInfo,
                                                          const MSplitInfo sfaCubeSplitInfo)
{
    uint32_t mSize = sfaCubeSplitInfo.nBufferDealM;
    uint32_t mSizeAlign = (mSize + 16 - 1) / 16;
    uint32_t sfaML1LoopCount = (mSize + M_SPLIT_SIZE - 1) / M_SPLIT_SIZE;
    uint32_t mL1SizeAlign = M_SPLIT_SIZE;
    uint32_t mL1Size = M_SPLIT_SIZE;

    uint32_t nSize = BlockAlign<KV_T>(sfaCubeConstInfo.headDim);
    uint32_t nL1Loops = (nSize + N_SPLIT_SIZE - 1) / N_SPLIT_SIZE;
    uint32_t nL1SizeAlign = N_SPLIT_SIZE;
    uint32_t nL1Size = N_SPLIT_SIZE;

    uint32_t kSize = sfaCubeRunInfo.actualSingleProcessSInnerSize;
    uint32_t kL1Size = 256;
    uint32_t kL1SizeAlign = SFAAlign(kL1Size, 16U);
    uint32_t kL1Loops = (kSize + kL1Size - 1) / kL1Size;
    uint32_t kL0Size = 128;
    uint32_t kL0Loops = (kL1Size + kL0Size - 1) / kL0Size;
    uint32_t kL0SizeAlign = kL0Size;
    LocalTensor<KV_T> bL1Tensor;
    LocalTensor<KV_T> subvTensor;


    uint32_t sfaQueryBufferIndex = 0, sfaKvBufferIndex = 0;
    uint32_t mBaseIdx = qpL1BufIter;
#if defined(K112166_SFA_QV_L1) && defined(K11110_SFA_PIPE) && defined(ATTENTION_HEADS) && (ATTENTION_HEADS <= 16)
    if constexpr (TEMPLATE_MODE == V_TEMPLATE && sizeof(Q_T) == 2 && sizeof(KV_T) == 2) {
        const uint32_t totalStepsQv = nL1Loops * kL1Loops;
#ifdef K112188_SFA_QVL1_ALL
        const bool enableQvL1 = (sfaCubeConstInfo.qSeqSize >= 1U);
#else
        const bool enableQvL1 = (sfaCubeConstInfo.qSeqSize >= 20U);
#endif
        if (enableQvL1 && mSize <= 16U) {
            if (totalStepsQv == 0U) {
#ifdef K112516_SFA_V2
                CrossCoreWaitFlag(sfaCubeConstInfo.syncV1C2);
#endif
                return;
            }
            const bool useVCache =
                (!sfaCubeRunInfo.useCoopMm1) &&
                (sfaCubeConstInfo.headDim == 512U) &&
                (N_SPLIT_SIZE == 128U) &&
                (sfaCubeSplitInfo.nBufferStartM == 0U) &&
                (sfaCubeSplitInfo.nBufferDealM == sfaCubeRunInfo.actMBaseSize) &&
                (sfaCubeRunInfo.actualSingleProcessSInnerSize == 512U);
            const uint32_t vSlot = sfaCubeRunInfo.loop & 1U;
            const uint32_t vEv = vSlot ? L1_EVENT6 : L1_EVENT1;
            const uint32_t vBaseOff = (2U + vSlot * 2U) * L1_BLOCK_OFFSET;
            struct KvTicket2 {
                uint32_t l1Off;
                uint32_t ev;
                bool cached;
            };
            auto issue_kv_mm2_qv = [&](uint32_t sn, uint32_t sk1) __aicore__ -> KvTicket2 {
                if (useVCache && sk1 == 0U && sn < 2U) {
                    return KvTicket2{vBaseOff + sn * 16384U, vEv, true};
                }
                const uint32_t snSize =
                    (sn + 1U == nL1Loops) ? (nSize - (nL1Loops - 1U) * N_SPLIT_SIZE) : N_SPLIT_SIZE;
                const uint32_t sk1Size = (sk1 + 1U == kL1Loops) ? (kSize - (kL1Loops - 1U) * 256U) : 256U;
                const uint32_t skL0Loops = (sk1Size + 127U) / 128U;
                const uint32_t kOff = sk1 * 2U;
                const uint32_t sSlot = kvL1BufIter & 1U;
                kvL1BufIter++;
                const uint32_t sEv = sSlot ? L1_EVENT5 : L1_EVENT4;
                WaitFlag<HardEvent::MTE1_MTE2>(sEv);
                LocalTensor<KV_T> dst = l1KVTensor[sSlot * L1_BLOCK_OFFSET];
                if (SFA_MM2_PAIR_LOAD && IsSameType<KV_T, bfloat16_t>::value &&
                    sfaCubeConstInfo.headDim == 512U && N_SPLIT_SIZE == 128U &&
                    snSize == 128U && sk1Size == 256U && skL0Loops == 2U) {
                    const uint64_t offset = static_cast<uint64_t>(kOff) * 128ULL * 512ULL + sn * 128ULL;
                    const auto address = sfaCubeRunInfo.useCoopKv
                        ? kvMergeBaseGm_.GetPhyAddr(
                              static_cast<uint64_t>(sfaCubeRunInfo.coopKvSlot) * N_WORKSPACE_SIZE * 576ULL + offset)
                        : kvMergeGm_.GetPhyAddr((sfaCubeRunInfo.loop % 4U) * N_WORKSPACE_SIZE * 576ULL + offset);
                    GlobalTensor<uint32_t> source;
                    source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address), 65344);
                    Nd2NzParams pair{2, 128, 64, 32768, 256, 128, 1, 8192};
                    DataCopy(dst.template ReinterpretCast<uint32_t>(), source, pair);
                } else {
                    for (uint32_t kl = 0U; kl < skL0Loops; kl++) {
                        const uint32_t klSize = (kl + 1U == skL0Loops) ? (sk1Size - (skL0Loops - 1U) * 128U) : 128U;
                        const uint32_t klAlign = SFAAlign(klSize, 16U);
                        Nd2NzParams cp;
                        cp.ndNum = 1;
                        cp.nValue = klSize;
                        cp.dValue = N_SPLIT_SIZE;
                        cp.srcDValue = sfaCubeConstInfo.headDim;
                        cp.dstNzC0Stride = klAlign;
                        cp.dstNzNStride = 1;
                        cp.srcNdMatrixStride = 0;
                        cp.dstNzMatrixStride = 0;
                        const uint64_t rowOff =
                            static_cast<uint64_t>(kOff + kl) * 128ULL * sfaCubeConstInfo.headDim + sn * N_SPLIT_SIZE;
                        DataCopy(dst[kl * 128U * N_SPLIT_SIZE],
                                 sfaCubeRunInfo.useCoopKv
                                     ? kvMergeBaseGm_[static_cast<uint64_t>(sfaCubeRunInfo.coopKvSlot) *
                                                          N_WORKSPACE_SIZE * 576ULL +
                                                      rowOff]
                                     : kvMergeGm_[(sfaCubeRunInfo.loop % 4U) * N_WORKSPACE_SIZE * 576ULL + rowOff],
                                 cp);
                    }
                }
                SetFlag<HardEvent::MTE2_MTE1>(sEv);
                return KvTicket2{sSlot * L1_BLOCK_OFFSET, sEv, false};
            };
            mL1Size = mSize;
            mL1SizeAlign = 16U;
#ifdef K112516_SFA_V2
            bool hasPre1 = false;
            bool hasPre3 = false;
            bool hasPre4 = false;
            KvTicket2 curTicket{0U, L1_EVENT4, false};
            KvTicket2 preTicket1{0U, L1_EVENT4, false};
            KvTicket2 preTicket3{0U, L1_EVENT5, false};
            KvTicket2 preTicket4{0U, L1_EVENT4, false};
            if (!sfaCubeRunInfo.useCoopMm1) {
                curTicket = issue_kv_mm2_qv(0U, 0U);
                if (totalStepsQv > 1U) {
                    const uint32_t qk1 = (1U < kL1Loops) ? 1U : 0U;
                    const uint32_t qn1 = (1U < kL1Loops) ? 0U : 1U;
                    preTicket1 = issue_kv_mm2_qv(qn1, qk1);
                    hasPre1 = true;
                    if (useVCache && totalStepsQv == 8U && nL1Loops == 4U && kL1Loops == 2U) {
                        preTicket3 = issue_kv_mm2_qv(1U, 1U);
                        hasPre3 = true;
                    }
                }
                CrossCoreWaitFlag(sfaCubeConstInfo.syncV1C2);
            } else {
                CrossCoreWaitFlag(sfaCubeConstInfo.syncV1C2);
                curTicket = issue_kv_mm2_qv(0U, 0U);
            }
#else
            KvTicket2 curTicket = issue_kv_mm2_qv(0U, 0U);
#endif
            for (uint32_t nL1 = 0U; nL1 < nL1Loops; nL1++) {
                if (nL1 + 1U == nL1Loops) {
                    nL1Size = nSize - (nL1Loops - 1U) * N_SPLIT_SIZE;
                    nL1SizeAlign = SFAAlign(nL1Size, 16U);
                }
                for (uint32_t k1 = 0U; k1 < kL1Loops; k1++) {
                    kL1Size = (k1 + 1U == kL1Loops) ? (kSize - (kL1Loops - 1U) * 256U) : 256U;
                    kL0Loops = (kL1Size + 127U) / 128U;
                    const uint32_t pEv = k1 ? L1_EVENT3 : L1_EVENT2;
                    LocalTensor<KV_T> aL1Tensor = l1QPTensor[16384U + k1 * 8192U];
                    if (nL1 == 0U) {
                        WaitFlag<HardEvent::MTE1_MTE2>(pEv);
                        CopyInMm2AToL1(aL1Tensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM, mL1Size, kL1Size,
                                       256U * k1);
                        SetFlag<HardEvent::MTE2_MTE1>(pEv);
                    }
                    uint32_t next_k1 = k1 + 1U;
                    uint32_t next_n = nL1;
                    if (next_k1 == kL1Loops) {
                        next_k1 = 0U;
                        next_n++;
                    }
#ifdef K112516_SFA_V2
                    KvTicket2 nextTicket{0U, L1_EVENT4, false};
                    if (nL1 == 0U && k1 == 0U && hasPre1) {
                        nextTicket = preTicket1;
                    } else if (nL1 == 1U && k1 == 0U && hasPre3) {
                        preTicket4 = issue_kv_mm2_qv(2U, 0U);
                        hasPre4 = true;
                        nextTicket = preTicket3;
                    } else if (nL1 == 1U && k1 == 1U && hasPre4) {
                        nextTicket = preTicket4;
                    } else if (next_n < nL1Loops) {
                        nextTicket = issue_kv_mm2_qv(next_n, next_k1);
                    }
#else
                    const KvTicket2 nextTicket =
                        (next_n < nL1Loops) ? issue_kv_mm2_qv(next_n, next_k1) : KvTicket2{0U, L1_EVENT4, false};
#endif
                    if (!curTicket.cached) {
                        WaitFlag<HardEvent::MTE2_MTE1>(curTicket.ev);
                    }
                    if (nL1 == 0U) {
                        WaitFlag<HardEvent::MTE2_MTE1>(pEv);
                    }
                    bL1Tensor = l1KVTensor[curTicket.l1Off];
                    const uint32_t kL0Step = curTicket.cached ? L1_BLOCK_OFFSET : 16384U;
                    LocalTensor cL0Tensor =
                        cL0TensorPingPong[(sfaL0CBufferIndex % 2U) * (L0C_PP_SIZE / sizeof(MM_OUT_T))];
                    for (uint32_t kL0 = 0U; kL0 < kL0Loops; kL0++) {
                        kL0Size = (kL0 + 1U == kL0Loops) ? (kL1Size - (kL0Loops - 1U) * 128U) : 128U;
                        kL0SizeAlign = SFAAlign(kL0Size, 16U);
                        WaitFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2U));
                        LocalTensor<KV_T> bL0Tensor =
                            bL0TensorPingPong[(abL0BufIter % 2U) * (L0B_PP_SIZE / sizeof(KV_T))];
                        LoadData3DParamsV2<KV_T> sfaMm2LoadBParams;
                        sfaMm2LoadBParams.l1H = kL0SizeAlign / 16U;
                        sfaMm2LoadBParams.l1W = 16U;
                        sfaMm2LoadBParams.padList[0] = 0;
                        sfaMm2LoadBParams.padList[1] = 0;
                        sfaMm2LoadBParams.padList[2] = 0;
                        sfaMm2LoadBParams.padList[3] = 255;
                        sfaMm2LoadBParams.mStartPt = 0;
                        sfaMm2LoadBParams.kStartPt = 0;
                        sfaMm2LoadBParams.mExtension = kL0SizeAlign;
                        sfaMm2LoadBParams.kExtension = nL1SizeAlign;
                        sfaMm2LoadBParams.strideW = 1;
                        sfaMm2LoadBParams.strideH = 1;
                        sfaMm2LoadBParams.filterW = 1;
                        sfaMm2LoadBParams.filterSizeW = false;
                        sfaMm2LoadBParams.filterH = 1;
                        sfaMm2LoadBParams.filterSizeH = false;
                        sfaMm2LoadBParams.dilationFilterW = 1;
                        sfaMm2LoadBParams.dilationFilterH = 1;
                        sfaMm2LoadBParams.enTranspose = 1;
                        sfaMm2LoadBParams.fMatrixCtrl = 0;
                        sfaMm2LoadBParams.channelSize = nL1SizeAlign;
                        LoadData<KV_T, LOAD3DV2_CONFIG>(bL0Tensor, bL1Tensor[kL0 * kL0Step], sfaMm2LoadBParams);

                        LocalTensor<KV_T> aL0Tensor =
                            aL0TensorPingPong[(abL0BufIter % 2U) * (L0A_PP_SIZE / sizeof(KV_T))];
                        LoadData3DParamsV2<KV_T> sfaMm2LoadAParams;
                        sfaMm2LoadAParams.padList[0] = 0;
                        sfaMm2LoadAParams.padList[1] = 0;
                        sfaMm2LoadAParams.padList[2] = 0;
                        sfaMm2LoadAParams.padList[3] = 255;
                        sfaMm2LoadAParams.l1H = mL1SizeAlign / 16U;
                        sfaMm2LoadAParams.l1W = 16U;
                        sfaMm2LoadAParams.mExtension = mL1SizeAlign;
                        sfaMm2LoadAParams.kExtension = kL0SizeAlign;
                        sfaMm2LoadAParams.mStartPt = 0;
                        sfaMm2LoadAParams.kStartPt = 0;
                        sfaMm2LoadAParams.strideW = 1;
                        sfaMm2LoadAParams.strideH = 1;
                        sfaMm2LoadAParams.filterW = 1;
                        sfaMm2LoadAParams.filterSizeW = false;
                        sfaMm2LoadAParams.filterH = 1;
                        sfaMm2LoadAParams.filterSizeH = false;
                        sfaMm2LoadAParams.enTranspose = 0;
                        sfaMm2LoadAParams.fMatrixCtrl = 0;
                        sfaMm2LoadAParams.dilationFilterW = 1;
                        sfaMm2LoadAParams.dilationFilterH = 1;
                        sfaMm2LoadAParams.channelSize = kL0SizeAlign;
                        LoadData<KV_T, LOAD3DV2_CONFIG>(aL0Tensor, aL1Tensor[kL0 * 128U * mL1SizeAlign],
                                                        sfaMm2LoadAParams);

                        SetFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2U));
                        WaitFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2U));

                        MmadParams sfaMmadParams;
                        sfaMmadParams.m = mL1SizeAlign;
                        sfaMmadParams.n = nL1SizeAlign;
                        sfaMmadParams.k = kL0Size;
                        sfaMmadParams.cmatrixInitVal = (kL0 == 0U && k1 == 0U);
                        sfaMmadParams.cmatrixSource = false;
                        sfaMmadParams.unitFlag = ((k1 + 1U == kL1Loops) && (kL0 + 1U == kL0Loops)) ? 0b11 : 0b10;
                        Mmad(cL0Tensor, aL0Tensor, bL0Tensor, sfaMmadParams);
                        if (((sfaMmadParams.m / 16U) * (sfaMmadParams.n / 16U)) < 10U) {
                            PipeBarrier<PIPE_M>();
                        }
                        SetFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2U));
                        abL0BufIter++;
                    }
                    if (nL1 + 1U == nL1Loops) {
                        SetFlag<HardEvent::MTE1_MTE2>(pEv);
                    }
                    if (k1 + 1U == kL1Loops) {
                        if (nL1 == 0U) {
                            CrossCoreWaitFlag(sfaCubeConstInfo.syncV1NupdateC2);
                        }
                        SetAtomicAdd<MM_OUT_T>();
                        FixpipeParamsV220 fixParams;
                        fixParams.nSize = nL1SizeAlign;
                        fixParams.mSize = mL1SizeAlign;
                        fixParams.srcStride = mL1SizeAlign;
                        fixParams.dstStride = nSize;
                        fixParams.ndNum = 1;
                        fixParams.unitFlag = 0b11;
                        uint64_t mm2Offset = sfaCubeSplitInfo.nBufferStartM * nSize + nL1 * N_SPLIT_SIZE;
                        Fixpipe(mm2ResGm[(sfaCubeRunInfo.bn2IdxInCurCore % sfaCubeConstInfo.preLoadNum) *
                                             sfaCubeConstInfo.bmm2ResUbSize +
                                         mm2Offset],
                                cL0Tensor, fixParams);
                        SetAtomicNone();
                    }
                    if (curTicket.cached) {
                        if (nL1 == 1U && k1 == 0U) {
                            SetFlag<HardEvent::MTE1_MTE2>(vEv);
                        }
                    } else {
                        SetFlag<HardEvent::MTE1_MTE2>(curTicket.ev);
                    }
                    curTicket = nextTicket;
                }
                sfaL0CBufferIndex++;
            }
            return;
        }
    }
#endif
#ifdef K112516_SFA_V2
    CrossCoreWaitFlag(sfaCubeConstInfo.syncV1C2);
#endif
#ifdef K11110_SFA_PIPE
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        const uint32_t totalSteps = nL1Loops * kL1Loops;
        if (sfaML1LoopCount == 1U && totalSteps > 0U) {
            auto issue_kv_mm2 = [&](uint32_t sn, uint32_t sk1) __aicore__ -> uint32_t {
                const uint32_t snSize =
                    (sn + 1U == nL1Loops) ? (nSize - (nL1Loops - 1U) * N_SPLIT_SIZE) : N_SPLIT_SIZE;
                const uint32_t sk1Size = (sk1 + 1U == kL1Loops) ? (kSize - (kL1Loops - 1U) * 256U) : 256U;
                const uint32_t skL0Loops = (sk1Size + 127U) / 128U;
                const uint32_t kOff = sk1 * 2U;
                kvL1BufIter++;
                const uint32_t bufIdx = kvL1BufIter % 3U;
                WaitFlag<HardEvent::MTE1_MTE2>(sfaMte21KvEvents[bufIdx]);
                LocalTensor<KV_T> dst = l1KVTensor[bufIdx * L1_BLOCK_OFFSET];
                if (SFA_MM2_PAIR_LOAD && IsSameType<KV_T, bfloat16_t>::value &&
                    sfaCubeConstInfo.headDim == 512U && N_SPLIT_SIZE == 128U &&
                    snSize == 128U && sk1Size == 256U && skL0Loops == 2U) {
                    const uint64_t offset = static_cast<uint64_t>(kOff) * 128ULL * 512ULL + sn * 128ULL;
                    const auto address = sfaCubeRunInfo.useCoopKv
                        ? kvMergeBaseGm_.GetPhyAddr(
                              static_cast<uint64_t>(sfaCubeRunInfo.coopKvSlot) * N_WORKSPACE_SIZE * 576ULL + offset)
                        : kvMergeGm_.GetPhyAddr((sfaCubeRunInfo.loop % 4U) * N_WORKSPACE_SIZE * 576ULL + offset);
                    GlobalTensor<uint32_t> source;
                    source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address), 65344);
                    Nd2NzParams pair{2, 128, 64, 32768, 256, 128, 1, 8192};
                    DataCopy(dst.template ReinterpretCast<uint32_t>(), source, pair);
                } else {
                    for (uint32_t kl = 0U; kl < skL0Loops; kl++) {
                        const uint32_t klSize = (kl + 1U == skL0Loops) ? (sk1Size - (skL0Loops - 1U) * 128U) : 128U;
                        const uint32_t klAlign = SFAAlign(klSize, 16U);
                        Nd2NzParams cp;
                        cp.ndNum = 1;
                        cp.nValue = klSize;
                        cp.dValue = N_SPLIT_SIZE;
                        cp.srcDValue = sfaCubeConstInfo.headDim;
                        cp.dstNzC0Stride = klAlign;
                        cp.dstNzNStride = 1;
                        cp.srcNdMatrixStride = 0;
                        cp.dstNzMatrixStride = 0;
                        const uint64_t rowOff =
                            static_cast<uint64_t>(kOff + kl) * 128ULL * sfaCubeConstInfo.headDim + sn * N_SPLIT_SIZE;
                        DataCopy(dst[kl * 128U * N_SPLIT_SIZE],
                                 sfaCubeRunInfo.useCoopKv
                                     ? kvMergeBaseGm_[static_cast<uint64_t>(sfaCubeRunInfo.coopKvSlot) *
                                                          N_WORKSPACE_SIZE * 576ULL +
                                                      rowOff]
                                     : kvMergeGm_[(sfaCubeRunInfo.loop % 4U) * N_WORKSPACE_SIZE * 576ULL + rowOff],
                                 cp);
                    }
                }
                SetFlag<HardEvent::MTE2_MTE1>(sfaMte21KvEvents[bufIdx]);
                return bufIdx;
            };
            mL1Size = mSize;
            mL1SizeAlign = SFAAlign(mL1Size, 16U);
            uint32_t curKvBuf = issue_kv_mm2(0U, 0U);
            for (uint32_t nL1 = 0U; nL1 < nL1Loops; nL1++) {
                if (nL1 + 1U == nL1Loops) {
                    nL1Size = nSize - (nL1Loops - 1U) * N_SPLIT_SIZE;
                    nL1SizeAlign = SFAAlign(nL1Size, 16U);
                }
                for (uint32_t k1 = 0U; k1 < kL1Loops; k1++) {
                    kL1Size = (k1 + 1U == kL1Loops) ? (kSize - (kL1Loops - 1U) * 256U) : 256U;
                    kL0Loops = (kL1Size + 127U) / 128U;
                    sfaQueryBufferIndex = GetQPL1RealIdx(mBaseIdx, k1);
                    LocalTensor<KV_T> aL1Tensor = l1QPTensor[sfaQueryBufferIndex * L1_BLOCK_OFFSET];
                    if (nL1 == 0U) {
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[sfaQueryBufferIndex]);
                        CopyInMm2AToL1(aL1Tensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM, mL1Size, kL1Size,
                                       256U * k1);
                        SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[sfaQueryBufferIndex]);
                    }
                    uint32_t next_k1 = k1 + 1U;
                    uint32_t next_n = nL1;
                    if (next_k1 == kL1Loops) {
                        next_k1 = 0U;
                        next_n++;
                    }
                    const uint32_t nextKvBuf = (next_n < nL1Loops) ? issue_kv_mm2(next_n, next_k1) : 0U;
                    WaitFlag<HardEvent::MTE2_MTE1>(sfaMte21KvEvents[curKvBuf]);
                    if (nL1 == 0U) {
                        WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[sfaQueryBufferIndex]);
                    }
                    bL1Tensor = l1KVTensor[curKvBuf * L1_BLOCK_OFFSET];
                    LocalTensor cL0Tensor =
                        cL0TensorPingPong[(sfaL0CBufferIndex % 2U) * (L0C_PP_SIZE / sizeof(MM_OUT_T))];
                    for (uint32_t kL0 = 0U; kL0 < kL0Loops; kL0++) {
                        kL0Size = (kL0 + 1U == kL0Loops) ? (kL1Size - (kL0Loops - 1U) * 128U) : 128U;
                        kL0SizeAlign = SFAAlign(kL0Size, 16U);
                        WaitFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2U));
                        LocalTensor<KV_T> bL0Tensor =
                            bL0TensorPingPong[(abL0BufIter % 2U) * (L0B_PP_SIZE / sizeof(KV_T))];
                        LoadData3DParamsV2<KV_T> sfaMm2LoadBParams;
                        sfaMm2LoadBParams.l1H = kL0SizeAlign / 16U;
                        sfaMm2LoadBParams.l1W = 16U;
                        sfaMm2LoadBParams.padList[0] = 0;
                        sfaMm2LoadBParams.padList[1] = 0;
                        sfaMm2LoadBParams.padList[2] = 0;
                        sfaMm2LoadBParams.padList[3] = 255;
                        sfaMm2LoadBParams.mStartPt = 0;
                        sfaMm2LoadBParams.kStartPt = 0;
                        sfaMm2LoadBParams.mExtension = kL0SizeAlign;
                        sfaMm2LoadBParams.kExtension = nL1SizeAlign;
                        sfaMm2LoadBParams.strideW = 1;
                        sfaMm2LoadBParams.strideH = 1;
                        sfaMm2LoadBParams.filterW = 1;
                        sfaMm2LoadBParams.filterSizeW = false;
                        sfaMm2LoadBParams.filterH = 1;
                        sfaMm2LoadBParams.filterSizeH = false;
                        sfaMm2LoadBParams.dilationFilterW = 1;
                        sfaMm2LoadBParams.dilationFilterH = 1;
                        sfaMm2LoadBParams.enTranspose = 1;
                        sfaMm2LoadBParams.fMatrixCtrl = 0;
                        sfaMm2LoadBParams.channelSize = nL1SizeAlign;
                        LoadData<KV_T, LOAD3DV2_CONFIG>(bL0Tensor, bL1Tensor[kL0 * 128U * 128U], sfaMm2LoadBParams);

                        LocalTensor<KV_T> aL0Tensor =
                            aL0TensorPingPong[(abL0BufIter % 2U) * (L0A_PP_SIZE / sizeof(KV_T))];
                        LoadData3DParamsV2<KV_T> sfaMm2LoadAParams;
                        sfaMm2LoadAParams.padList[0] = 0;
                        sfaMm2LoadAParams.padList[1] = 0;
                        sfaMm2LoadAParams.padList[2] = 0;
                        sfaMm2LoadAParams.padList[3] = 255;
                        sfaMm2LoadAParams.l1H = mL1SizeAlign / 16U;
                        sfaMm2LoadAParams.l1W = 16U;
                        sfaMm2LoadAParams.mExtension = mL1SizeAlign;
                        sfaMm2LoadAParams.kExtension = kL0SizeAlign;
                        sfaMm2LoadAParams.mStartPt = 0;
                        sfaMm2LoadAParams.kStartPt = 0;
                        sfaMm2LoadAParams.strideW = 1;
                        sfaMm2LoadAParams.strideH = 1;
                        sfaMm2LoadAParams.filterW = 1;
                        sfaMm2LoadAParams.filterSizeW = false;
                        sfaMm2LoadAParams.filterH = 1;
                        sfaMm2LoadAParams.filterSizeH = false;
                        sfaMm2LoadAParams.enTranspose = 0;
                        sfaMm2LoadAParams.fMatrixCtrl = 0;
                        sfaMm2LoadAParams.dilationFilterW = 1;
                        sfaMm2LoadAParams.dilationFilterH = 1;
                        sfaMm2LoadAParams.channelSize = kL0SizeAlign;
                        LoadData<KV_T, LOAD3DV2_CONFIG>(aL0Tensor, aL1Tensor[kL0 * 128U * mL1SizeAlign],
                                                        sfaMm2LoadAParams);

                        SetFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2U));
                        WaitFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2U));

                        MmadParams sfaMmadParams;
                        sfaMmadParams.m = mL1SizeAlign;
                        sfaMmadParams.n = nL1SizeAlign;
                        sfaMmadParams.k = kL0Size;
                        sfaMmadParams.cmatrixInitVal = (kL0 == 0U && k1 == 0U);
                        sfaMmadParams.cmatrixSource = false;
                        sfaMmadParams.unitFlag = ((k1 + 1U == kL1Loops) && (kL0 + 1U == kL0Loops)) ? 0b11 : 0b10;
                        Mmad(cL0Tensor, aL0Tensor, bL0Tensor, sfaMmadParams);
                        if (((sfaMmadParams.m / 16U) * (sfaMmadParams.n / 16U)) < 10U) {
                            PipeBarrier<PIPE_M>();
                        }
                        SetFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2U));
                        abL0BufIter++;
                    }
                    if (nL1 + 1U == nL1Loops) {
                        SetFlag<HardEvent::MTE1_MTE2>(mte21QPIds[sfaQueryBufferIndex]);
                    }
                    if (k1 + 1U == kL1Loops) {
                        if (nL1 == 0U) {
                            CrossCoreWaitFlag(sfaCubeConstInfo.syncV1NupdateC2);
                        }
                        SetAtomicAdd<MM_OUT_T>();
                        FixpipeParamsV220 fixParams;
                        fixParams.nSize = nL1SizeAlign;
                        fixParams.mSize = mL1SizeAlign;
                        fixParams.srcStride = mL1SizeAlign;
                        fixParams.dstStride = nSize;
                        fixParams.ndNum = 1;
                        fixParams.unitFlag = 0b11;
                        uint64_t mm2Offset = sfaCubeSplitInfo.nBufferStartM * nSize + nL1 * N_SPLIT_SIZE;
                        Fixpipe(mm2ResGm[(sfaCubeRunInfo.bn2IdxInCurCore % sfaCubeConstInfo.preLoadNum) *
                                             sfaCubeConstInfo.bmm2ResUbSize +
                                         mm2Offset],
                                cL0Tensor, fixParams);
                        SetAtomicNone();
                    }
                    SetFlag<HardEvent::MTE1_MTE2>(sfaMte21KvEvents[curKvBuf]);
                    curKvBuf = nextKvBuf;
                }
                sfaL0CBufferIndex++;
            }
            qpL1BufIter += 1U;
            return;
        }
    }
#endif
    for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) {
        if (nL1 == (nL1Loops - 1)) {

            nL1Size = nSize - (nL1Loops - 1) * N_SPLIT_SIZE;
            nL1SizeAlign = SFAAlign(nL1Size, 16U);
        }


        kL1Size = 256;
        kL1SizeAlign = SFAAlign(kL1Size, 16U);

        uint32_t curTopKIdx = sfaCubeRunInfo.curTopKIdx;
        uint64_t curOffsetInSparseBlock = sfaCubeRunInfo.curOffsetInSparseBlock;
        uint32_t copyRowCnt = 0;
        int64_t idInTopK = 0;
        if constexpr (TEMPLATE_MODE != V_TEMPLATE) {
            idInTopK = topKGm.GetValue(sfaCubeRunInfo.topKBaseOffset + curTopKIdx);
        }

        for (uint32_t k1 = 0; k1 < kL1Loops; k1++) {
            if (k1 == (kL1Loops - 1)) {

                kL1Size = kSize - (kL1Loops - 1) * 256;
                kL1SizeAlign = SFAAlign(kL1Size, 16U);
            }
            kvL1BufIter++;
            uint32_t sfaKvBufferIndex = kvL1BufIter % 3;
            WaitFlag<HardEvent::MTE1_MTE2>(sfaMte21KvEvents[sfaKvBufferIndex]);
            bL1Tensor = l1KVTensor[sfaKvBufferIndex * L1_BLOCK_OFFSET];
            uint32_t kOffset = k1 * kL0Loops;
            kL0Size = 128;

            kL0Loops = (kL1Size + kL0Size - 1) / kL0Size;
            kL0SizeAlign = kL0Size;
            for (uint32_t kL1 = kOffset; kL1 < kL0Loops + kOffset; kL1++) {
                if (kL1 == kOffset + kL0Loops - 1) {

                    kL0Size = kL1Size - (kL0Loops - 1) * kL0Size;
                    kL0SizeAlign = SFAAlign(kL0Size, 16U);
                }

                uint32_t curSeqIdx = sfaCubeRunInfo.s2BatchOffset + (kL1 - kOffset) * 128 + k1 * 256;
                uint32_t copyFinishRowCnt = 0;
                if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                    if (SFA_MM2_PAIR_LOAD && IsSameType<KV_T, bfloat16_t>::value &&
                        sfaCubeConstInfo.headDim == 512 &&
                        N_SPLIT_SIZE == 128 && nL1Size == 128 && kL1Size == 256 && kL0Loops == 2) {
                        if (kL1 == kOffset) {
                            const uint64_t offset = static_cast<uint64_t>(kOffset) * 128 * 512 + nL1 * 128;
                            const auto address = sfaCubeRunInfo.useCoopKv ?
                                kvMergeBaseGm_.GetPhyAddr(static_cast<uint64_t>(sfaCubeRunInfo.coopKvSlot) *
                                                         N_WORKSPACE_SIZE * 576 + offset) :
                                kvMergeGm_.GetPhyAddr(sfaCubeRunInfo.loop % 4 * N_WORKSPACE_SIZE * 576 + offset);
                            GlobalTensor<uint32_t> source;
                            source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address), 65344);
                            Nd2NzParams pair{2, 128, 64, 32768, 256, 128, 1, 8192};
                            DataCopy(bL1Tensor.template ReinterpretCast<uint32_t>(), source, pair);
                        }
                    } else {
                        Nd2NzParams sfaCubeCopyParams;
                        sfaCubeCopyParams.ndNum = 1;
                        sfaCubeCopyParams.nValue = kL0Size;
                        sfaCubeCopyParams.dValue = N_SPLIT_SIZE;
                        sfaCubeCopyParams.srcDValue = sfaCubeConstInfo.headDim;
                        sfaCubeCopyParams.dstNzC0Stride = kL0SizeAlign;
                        sfaCubeCopyParams.dstNzNStride = 1;
                        sfaCubeCopyParams.srcNdMatrixStride = 0;
                        sfaCubeCopyParams.dstNzMatrixStride = 0;
                        DataCopy(bL1Tensor[(kL1 - kOffset) * 128 * N_SPLIT_SIZE],
                                 (sfaCubeRunInfo.useCoopKv ?
                                      kvMergeBaseGm_[static_cast<uint64_t>(sfaCubeRunInfo.coopKvSlot) * N_WORKSPACE_SIZE * 576 +
                                                     kL1 * 128 * sfaCubeConstInfo.headDim + nL1 * N_SPLIT_SIZE] :
                                      kvMergeGm_[sfaCubeRunInfo.loop % 4 * N_WORKSPACE_SIZE * 576 +
                                                 kL1 * 128 * sfaCubeConstInfo.headDim + nL1 * N_SPLIT_SIZE]),
                                 sfaCubeCopyParams);
                    }
                } else {
                    while (copyFinishRowCnt < kL0Size) {
                        CalcTopKBlockInfo(sfaCubeRunInfo, curTopKIdx, curOffsetInSparseBlock, curSeqIdx, copyRowCnt,
                                          idInTopK);

                        if (copyFinishRowCnt + copyRowCnt > kL0Size) {
                            copyRowCnt = kL0Size - copyFinishRowCnt;
                        }

                        if constexpr (PAGE_ATTENTION) {
                            Position startPos;
                            startPos.bIdx = sfaCubeRunInfo.bIdx;
                            startPos.n2Idx = sfaCubeRunInfo.n2Idx;
                            startPos.s2Idx = idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock;
                            startPos.dIdx =
                                nL1 * N_SPLIT_SIZE;
                            PAShape shape;
                            shape.blockSize = kvCacheBlockSize;
                            shape.headNum = sfaCubeConstInfo.kvHeadNum;
                            shape.headDim = sfaCubeConstInfo.headDim;
                            shape.actHeadDim = nL1Size;
                            shape.maxblockNumPerBatch = maxBlockNumPerBatch;
                            shape.copyRowNum = copyRowCnt;
                            shape.copyRowNumAlign = kL0SizeAlign;
                            subvTensor = bL1Tensor[(kL1 - kOffset) * 128 * N_SPLIT_SIZE + copyFinishRowCnt * 16];
                            DataCopyPA<KV_T, KV_LAYOUT_T>(subvTensor, valueGm, blockTableGm, shape, startPos);
                        } else {
                            uint64_t valueOffset = sfaCubeRunInfo.tensorBOffset;
                            if constexpr (KV_LAYOUT_T == SFA_LAYOUT::BSND || KV_LAYOUT_T == SFA_LAYOUT::TND) {
                                valueOffset += (idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                               sfaCubeConstInfo.kvHeadNum * sfaCubeConstInfo.headDim;
                            } else {
                                valueOffset += (idInTopK * sfaCubeConstInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                               sfaCubeConstInfo.headDim;
                            }

                            subvTensor = bL1Tensor[(kL1 - kOffset) * 128 * N_SPLIT_SIZE];
                            CopyInMm2BToL1(subvTensor, valueOffset, kL0SizeAlign, copyFinishRowCnt, copyRowCnt,
                                           nL1 * N_SPLIT_SIZE, nL1Size);
                        }

                        copyFinishRowCnt += copyRowCnt;
                        curSeqIdx += copyRowCnt;
                    }
                }
            }
            SetFlag<HardEvent::MTE2_MTE1>(sfaMte21KvEvents[sfaKvBufferIndex]);
            WaitFlag<HardEvent::MTE2_MTE1>(sfaMte21KvEvents[sfaKvBufferIndex]);
            mL1SizeAlign = M_SPLIT_SIZE;
            mL1Size = M_SPLIT_SIZE;
            for (uint32_t mL1 = 0; mL1 < sfaML1LoopCount; mL1++) {
                if (mL1 == (sfaML1LoopCount - 1)) {

                    mL1Size = mSize - (sfaML1LoopCount - 1) * M_SPLIT_SIZE;
                    mL1SizeAlign = SFAAlign(mL1Size, 16U);
                }

                uint32_t mIdx = mBaseIdx + mL1;
                sfaQueryBufferIndex = GetQPL1RealIdx(mIdx, k1);
                LocalTensor<KV_T> aL1Tensor = l1QPTensor[sfaQueryBufferIndex * L1_BLOCK_OFFSET];
                if (nL1 == 0) {
                    WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[sfaQueryBufferIndex]);
                    CopyInMm2AToL1(aL1Tensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE,
                                   mL1Size, kL1Size, 256 * k1);
                    SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[sfaQueryBufferIndex]);
                    WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[sfaQueryBufferIndex]);
                }

                LocalTensor cL0Tensor =
                    cL0TensorPingPong[(sfaL0CBufferIndex % 2) *
                                      (L0C_PP_SIZE / sizeof(MM_OUT_T))];
                uint32_t baseK = 128;
                uint32_t baseN = 128;
                kL0Size = 128;
                kL0SizeAlign = kL0Size;
                for (uint32_t kL0 = 0; kL0 < kL0Loops; kL0++) {
                    if (kL0 + 1 == kL0Loops) {
                        kL0Size = kL1Size - (kL0Loops - 1) * kL0Size;
                        kL0SizeAlign = SFAAlign(kL0Size, 16U);
                    }
                    WaitFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(abL0BufIter % 2) * (L0B_PP_SIZE / sizeof(KV_T))];
                    LoadData3DParamsV2<KV_T> sfaMm2LoadBParams;
                    sfaMm2LoadBParams.l1H = kL0SizeAlign / 16;
                    sfaMm2LoadBParams.l1W = 16;
                    sfaMm2LoadBParams.padList[0] = 0;
                    sfaMm2LoadBParams.padList[1] = 0;
                    sfaMm2LoadBParams.padList[2] = 0;
                    sfaMm2LoadBParams.padList[3] = 255;

                    sfaMm2LoadBParams.mStartPt = 0;
                    sfaMm2LoadBParams.kStartPt = 0;
                    sfaMm2LoadBParams.mExtension = kL0SizeAlign;
                    sfaMm2LoadBParams.kExtension = nL1SizeAlign;
                    sfaMm2LoadBParams.strideW = 1;
                    sfaMm2LoadBParams.strideH = 1;
                    sfaMm2LoadBParams.filterW = 1;
                    sfaMm2LoadBParams.filterSizeW = false;
                    sfaMm2LoadBParams.filterH = 1;
                    sfaMm2LoadBParams.filterSizeH = false;
                    sfaMm2LoadBParams.dilationFilterW = 1;
                    sfaMm2LoadBParams.dilationFilterH = 1;
                    sfaMm2LoadBParams.enTranspose = 1;
                    sfaMm2LoadBParams.fMatrixCtrl =
                        0;
                    sfaMm2LoadBParams.channelSize =
                        nL1SizeAlign;
                    LoadData<KV_T, LOAD3DV2_CONFIG>(bL0Tensor, bL1Tensor[kL0 * baseK * baseN], sfaMm2LoadBParams);

                    LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(abL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];
                    LoadData3DParamsV2<KV_T> sfaMm2LoadAParams;
                    sfaMm2LoadAParams.padList[0] = 0;
                    sfaMm2LoadAParams.padList[1] = 0;
                    sfaMm2LoadAParams.padList[2] = 0;
                    sfaMm2LoadAParams.padList[3] = 255;
                    sfaMm2LoadAParams.l1H = mL1SizeAlign / 16;
                    sfaMm2LoadAParams.l1W = 16;

                    sfaMm2LoadAParams.mExtension = mL1SizeAlign;
                    sfaMm2LoadAParams.kExtension = kL0SizeAlign;
                    sfaMm2LoadAParams.strideW = 1;
                    sfaMm2LoadAParams.strideH = 1;
                    sfaMm2LoadAParams.mStartPt = 0;
                    sfaMm2LoadAParams.kStartPt = 0;
                    sfaMm2LoadAParams.filterW = 1;
                    sfaMm2LoadAParams.filterSizeW = false;
                    sfaMm2LoadAParams.filterH = 1;
                    sfaMm2LoadAParams.filterSizeH = false;
                    sfaMm2LoadAParams.enTranspose = 0;
                    sfaMm2LoadAParams.fMatrixCtrl = 0;
                    sfaMm2LoadAParams.dilationFilterW = 1;
                    sfaMm2LoadAParams.dilationFilterH = 1;
                    sfaMm2LoadAParams.channelSize =
                        kL0SizeAlign;
                    LoadData<KV_T, LOAD3DV2_CONFIG>(aL0Tensor, aL1Tensor[kL0 * baseK * mL1SizeAlign],
                                                    sfaMm2LoadAParams);
                    SetFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));
                    WaitFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));

                    MmadParams sfaMmadParams;
                    sfaMmadParams.m = mL1SizeAlign;
                    sfaMmadParams.n = nL1SizeAlign;
                    sfaMmadParams.k = kL0Size;
                    sfaMmadParams.cmatrixInitVal = (kL0 == 0 && k1 == 0);
                    sfaMmadParams.cmatrixSource = false;
                    sfaMmadParams.unitFlag = ((k1 == (kL1Loops - 1)) && (kL0 == (kL0Loops - 1))) ? 0b11 : 0b10;

                    Mmad(cL0Tensor, aL0Tensor, bL0Tensor, sfaMmadParams);
                    if (((sfaMmadParams.m / 16) * (sfaMmadParams.n / 16)) < 10) {
                        PipeBarrier<PIPE_M>();
                    }
                    SetFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    abL0BufIter++;
                }

                if (nL1 == (nL1Loops - 1)) {
                    SetFlag<HardEvent::MTE1_MTE2>(
                        mte21QPIds[sfaQueryBufferIndex]);
                }

                if (k1 == (kL1Loops - 1)) {
                    if (nL1 == 0 && mL1 == 0) {
                        CrossCoreWaitFlag(sfaCubeConstInfo.syncV1NupdateC2);
                    }

                    SetAtomicAdd<MM_OUT_T>();

                    FixpipeParamsV220 fixParams;
                    fixParams.nSize = nL1SizeAlign;
                    fixParams.mSize = mL1SizeAlign;
                    fixParams.srcStride = mL1SizeAlign;
                    fixParams.dstStride = nSize;
                    fixParams.ndNum = 1;
                    fixParams.unitFlag = 0b11;

                    uint64_t mm2Offset =
                        (sfaCubeSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE) * nSize + nL1 * N_SPLIT_SIZE;
                    Fixpipe(mm2ResGm[(sfaCubeRunInfo.bn2IdxInCurCore % (sfaCubeConstInfo.preLoadNum)) *
                                         sfaCubeConstInfo.bmm2ResUbSize +
                                     mm2Offset],
                            cL0Tensor, fixParams);
                    SetAtomicNone();
                }

                if (sfaML1LoopCount == 2) {
                    sfaL0CBufferIndex++;
                }
            }
            SetFlag<HardEvent::MTE1_MTE2>(sfaMte21KvEvents[sfaKvBufferIndex]);
        }

        if (sfaML1LoopCount == 1) {
            sfaL0CBufferIndex++;
        }
    }
    qpL1BufIter += sfaML1LoopCount;
}

template <typename SFAT>
__aicore__ inline void SFAMatmulService<SFAT>::ComputeMm1Coop(const RunInfo& sfaCubeRunInfo,
                                                              const MSplitInfo sfaCubeSplitInfo, uint32_t cIdx,
                                                              uint32_t mm1Slot)
{
    if (mm1Slot == 0xFFFFFFFFU) {
        mm1Slot = cIdx;
    }
    uint32_t mSize = sfaCubeSplitInfo.nBufferDealM;
    uint32_t mL1Size = M_SPLIT_SIZE;
    uint32_t mL1SizeAlign = SFAAlign(M_SPLIT_SIZE, 16U);
    uint32_t sfaML1LoopCount = (mSize + M_SPLIT_SIZE - 1) / M_SPLIT_SIZE;

    uint32_t nSize = sfaCubeRunInfo.actualSingleProcessSInnerSize;
    uint32_t nL1Size = N_SPLIT_SIZE;
    uint32_t nL1SizeAlign = SFAAlign(N_SPLIT_SIZE, 16U);
    uint32_t nL1Loops = (nSize + N_SPLIT_SIZE - 1) / N_SPLIT_SIZE;

    uint32_t kSize = 576;
    uint32_t kL1Loops = 2;
    uint32_t kL0Size = 96;
    uint32_t kL0Loops = 3;

    LocalTensor<KV_T> bL1Tensor;
    uint32_t sfaQueryBufferIndex = 0;
    uint64_t kvBase = static_cast<uint64_t>(cIdx) * N_WORKSPACE_SIZE * kSize;
    uint64_t mm1Base = static_cast<uint64_t>(mm1Slot) * sfaCubeConstInfo.mmResUbSize;
#ifdef K11110_SFA_PIPE
    const uint32_t totalStepsCoop = nL1Loops * kL1Loops;
    if (sfaML1LoopCount == 1U && totalStepsCoop > 0U) {
        auto issue_kv_coop = [&](uint32_t step) __aicore__ -> uint32_t {
            const uint32_t sn = step >> 1U;
            const uint32_t sk = step & 1U;
            const uint32_t snSize =
                (sn + 1U == nL1Loops) ? (nSize - (nL1Loops - 1U) * N_SPLIT_SIZE) : N_SPLIT_SIZE;
            const uint32_t snAlign = SFAAlign(snSize, 16U);
            kvL1BufIter++;
            const uint32_t bufIdx = kvL1BufIter % 3U;
            WaitFlag<HardEvent::MTE1_MTE2>(sfaMte21KvEvents[bufIdx]);
            LocalTensor<KV_T> dst = l1KVTensor[bufIdx * L1_BLOCK_OFFSET];
            Nd2NzParams cp;
            cp.ndNum = 1;
            cp.nValue = snSize;
            cp.dstNzC0Stride = snAlign;
            cp.dstNzNStride = 1;
            cp.srcNdMatrixStride = 0;
            cp.dstNzMatrixStride = 0;
            if (sk == 0U) {
                cp.dValue = sfaCubeConstInfo.headDim >> 1;
                cp.srcDValue = sfaCubeConstInfo.headDim;
                DataCopy(dst, kvMergeBaseGm_[kvBase + sn * N_SPLIT_SIZE * sfaCubeConstInfo.headDim], cp);
                cp.dValue = sfaCubeConstInfo.headDimRope >> 1;
                cp.srcDValue = sfaCubeConstInfo.headDimRope;
                DataCopy(dst[snAlign * (sfaCubeConstInfo.headDim >> 1)],
                         kvMergeBaseGm_[kvBase + N_WORKSPACE_SIZE * sfaCubeConstInfo.headDim +
                                        sn * N_SPLIT_SIZE * sfaCubeConstInfo.headDimRope],
                         cp);
            } else {
                LocalTensor<Q_T> kTmp = dst[(sfaCubeConstInfo.headDimRope >> 1) * snAlign];
                cp.dValue = sfaCubeConstInfo.headDim >> 1;
                cp.srcDValue = sfaCubeConstInfo.headDim;
                DataCopy(kTmp,
                         kvMergeBaseGm_[kvBase + (sfaCubeConstInfo.headDim >> 1) +
                                        sn * N_SPLIT_SIZE * sfaCubeConstInfo.headDim],
                         cp);
                cp.dValue = sfaCubeConstInfo.headDimRope >> 1;
                cp.srcDValue = sfaCubeConstInfo.headDimRope;
                DataCopy(dst,
                         kvMergeBaseGm_[kvBase + N_WORKSPACE_SIZE * sfaCubeConstInfo.headDim +
                                        (sfaCubeConstInfo.headDimRope >> 1) +
                                        sn * N_SPLIT_SIZE * sfaCubeConstInfo.headDimRope],
                         cp);
            }
            SetFlag<HardEvent::MTE2_MTE1>(sfaMte21KvEvents[bufIdx]);
            return bufIdx;
        };
        mL1Size = mSize;
        mL1SizeAlign = SFAAlign(mL1Size, 16U);
        const uint32_t aL1PaddingSize = (M_SPLIT_SIZE - mL1SizeAlign) * 288U;
#if defined(K112166_SFA_QV_L1) && defined(ATTENTION_HEADS) && (ATTENTION_HEADS <= 16)
        if constexpr (sizeof(Q_T) == 2 && sizeof(KV_T) == 2) {
#ifdef K112188_SFA_QVL1_ALL
            const bool enableQvL1 = (sfaCubeConstInfo.qSeqSize >= 1U);
#else
            const bool enableQvL1 = (sfaCubeConstInfo.qSeqSize >= 20U);
#endif
            if (enableQvL1 && mSize <= 16U) {
                auto coop_slot_ev = [](uint32_t sl) __aicore__ -> uint32_t {
                    return (sl == 0U) ? L1_EVENT4 : ((sl == 1U) ? L1_EVENT5 : ((sl == 2U) ? L1_EVENT1 : L1_EVENT6));
                };
                auto issue_kv_coop_qv = [&](uint32_t step) __aicore__ -> uint32_t {
                    const uint32_t sn = step >> 1U;
                    const uint32_t sk = step & 1U;
                    const uint32_t snSize =
                        (sn + 1U == nL1Loops) ? (nSize - (nL1Loops - 1U) * N_SPLIT_SIZE) : N_SPLIT_SIZE;
                    const uint32_t snAlign = SFAAlign(snSize, 16U);
                    const uint32_t sSlot = kvL1BufIter & 3U;
                    kvL1BufIter++;
                    const uint32_t sEv = coop_slot_ev(sSlot);
                    WaitFlag<HardEvent::MTE1_MTE2>(sEv);
                    LocalTensor<KV_T> dst = l1KVTensor[sSlot * L1_BLOCK_OFFSET];
                    Nd2NzParams cp;
                    cp.ndNum = 1;
                    cp.nValue = snSize;
                    cp.dstNzC0Stride = snAlign;
                    cp.dstNzNStride = 1;
                    cp.srcNdMatrixStride = 0;
                    cp.dstNzMatrixStride = 0;
                    if (sk == 0U) {
                        cp.dValue = sfaCubeConstInfo.headDim >> 1;
                        cp.srcDValue = sfaCubeConstInfo.headDim;
                        DataCopy(dst, kvMergeBaseGm_[kvBase + sn * N_SPLIT_SIZE * sfaCubeConstInfo.headDim], cp);
                        cp.dValue = sfaCubeConstInfo.headDimRope >> 1;
                        cp.srcDValue = sfaCubeConstInfo.headDimRope;
                        DataCopy(dst[snAlign * (sfaCubeConstInfo.headDim >> 1)],
                                 kvMergeBaseGm_[kvBase + N_WORKSPACE_SIZE * sfaCubeConstInfo.headDim +
                                                sn * N_SPLIT_SIZE * sfaCubeConstInfo.headDimRope],
                                 cp);
                    } else {
                        LocalTensor<Q_T> kTmp = dst[(sfaCubeConstInfo.headDimRope >> 1) * snAlign];
                        cp.dValue = sfaCubeConstInfo.headDim >> 1;
                        cp.srcDValue = sfaCubeConstInfo.headDim;
                        DataCopy(kTmp,
                                 kvMergeBaseGm_[kvBase + (sfaCubeConstInfo.headDim >> 1) +
                                                sn * N_SPLIT_SIZE * sfaCubeConstInfo.headDim],
                                 cp);
                        cp.dValue = sfaCubeConstInfo.headDimRope >> 1;
                        cp.srcDValue = sfaCubeConstInfo.headDimRope;
                        DataCopy(dst,
                                 kvMergeBaseGm_[kvBase + N_WORKSPACE_SIZE * sfaCubeConstInfo.headDim +
                                                (sfaCubeConstInfo.headDimRope >> 1) +
                                                sn * N_SPLIT_SIZE * sfaCubeConstInfo.headDimRope],
                                 cp);
                    }
                    SetFlag<HardEvent::MTE2_MTE1>(sEv);
                    return sSlot;
                };
                mL1Size = mSize;
                mL1SizeAlign = 16U;
                const uint32_t qTag =
                    0x80000000U |
                    (static_cast<uint32_t>((sfaCubeRunInfo.tensorAOffset >> 9U) & 0xFFFFFU)) |
                    ((sfaCubeSplitInfo.nBufferStartM & 31U) << 20U) |
                    ((mSize & 31U) << 25U);
                const bool needLoadQ = (qpL1BufIter != qTag);
                qpL1BufIter = qTag;
                uint32_t curSlot = issue_kv_coop_qv(0U);
                uint32_t prefSlot1 = (totalStepsCoop > 1U) ? issue_kv_coop_qv(1U) : 0U;
                WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT0);
                if (needLoadQ) {
                    CopyInMm1AToL1(l1QPTensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM, mL1Size, 256, 0);
                    LocalTensor<Q_T> qRopeTensor = l1QPTensor[4096U];
                    CopyInMm1ARopeToL1(qRopeTensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM, mL1Size);
                    LocalTensor<Q_T> qHiTensor = l1QPTensor[5120U];
                    CopyInMm1AToL1(qHiTensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM, mL1Size, 256, 256);
                    SetFlag<HardEvent::MTE2_MTE1>(L1_EVENT0);
                    WaitFlag<HardEvent::MTE2_MTE1>(L1_EVENT0);
                }
                for (uint32_t nL1 = 0U; nL1 < nL1Loops; nL1++) {
                    if (nL1 + 1U == nL1Loops) {
                        nL1Size = nSize - (nL1Loops - 1U) * N_SPLIT_SIZE;
                        nL1SizeAlign = SFAAlign(nL1Size, 16U);
                    }
                    for (uint32_t kL1 = 0U; kL1 < kL1Loops; kL1++) {
                        const uint32_t curStep = nL1 * 2U + kL1;
                        LocalTensor<Q_T> aL1Tensor = l1QPTensor[kL1 * 4608U];
                        const uint32_t nextSlot = prefSlot1;
                        prefSlot1 = (curStep + 2U < totalStepsCoop) ? issue_kv_coop_qv(curStep + 2U) : 0U;
                        const uint32_t curEv = coop_slot_ev(curSlot);
                        WaitFlag<HardEvent::MTE2_MTE1>(curEv);
                        bL1Tensor = l1KVTensor[curSlot * L1_BLOCK_OFFSET];
                        LocalTensor cL0Tensor =
                            cL0TensorPingPong[(sfaL0CBufferIndex % 2U) * (L0C_PP_SIZE / sizeof(MM_OUT_T))];
                        for (uint32_t kL0 = 0U; kL0 < kL0Loops; kL0++) {
                            WaitFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2U));
                            LocalTensor<KV_T> aL0Tensor =
                                aL0TensorPingPong[(abL0BufIter % 2U) * (L0A_PP_SIZE / sizeof(KV_T))];
                            LoadDataMm1A(aL0Tensor, aL1Tensor, kL0, kL0Size, mL1SizeAlign, kL0Size);
                            LocalTensor<KV_T> bL0Tensor =
                                bL0TensorPingPong[(abL0BufIter % 2U) * (L0B_PP_SIZE / sizeof(KV_T))];
                            LoadDataMm1B(bL0Tensor, bL1Tensor, kL0, kL0Size, kL0Size, nL1SizeAlign);
                            SetFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2U));
                            WaitFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2U));
                            MmadParams sfaMmadParams;
                            sfaMmadParams.m = mL1SizeAlign;
                            sfaMmadParams.n = nL1SizeAlign;
                            sfaMmadParams.k = kL0Size;
                            sfaMmadParams.cmatrixInitVal = (kL1 == 0U && kL0 == 0U);
                            sfaMmadParams.cmatrixSource = false;
                            sfaMmadParams.unitFlag = (kL1 == 1U && kL0 + 1U == kL0Loops) ? 0b11 : 0b10;
                            Mmad(cL0Tensor, aL0Tensor, bL0Tensor, sfaMmadParams);
                            if (((sfaMmadParams.m / 16U) * (sfaMmadParams.n / 16U)) < 10U) {
                                PipeBarrier<PIPE_M>();
                            }
                            SetFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2U));
                            abL0BufIter++;
                        }
                        if (nL1 + 1U == nL1Loops && kL1 == 1U) {
                            SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT0);
                        }
                        if (kL1 == 1U) {
                            FixpipeParamsV220 fixParams;
                            fixParams.nSize = nL1SizeAlign;
                            fixParams.mSize = mL1SizeAlign;
                            fixParams.srcStride = mL1SizeAlign;
                            fixParams.dstStride = sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign;
                            fixParams.unitFlag = 0b11;
                            fixParams.ndNum = 1;
                            Fixpipe(mm1ResBaseGm_[mm1Base + nL1 * N_SPLIT_SIZE +
                                                  sfaCubeSplitInfo.nBufferStartM *
                                                      sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign],
                                    cL0Tensor, fixParams);
                        }
                        SetFlag<HardEvent::MTE1_MTE2>(curEv);
                        curSlot = nextSlot;
                    }
                    sfaL0CBufferIndex++;
                }
                return;
            }
        }
#endif
        uint32_t curKvBuf = issue_kv_coop(0U);
        for (uint32_t nL1 = 0U; nL1 < nL1Loops; nL1++) {
            if (nL1 + 1U == nL1Loops) {
                nL1Size = nSize - (nL1Loops - 1U) * N_SPLIT_SIZE;
                nL1SizeAlign = SFAAlign(nL1Size, 16U);
            }
            for (uint32_t kL1 = 0U; kL1 < kL1Loops; kL1++) {
                const uint32_t curStep = nL1 * 2U + kL1;
                sfaQueryBufferIndex = GetQPL1RealIdx(qpL1BufIter, kL1);
                LocalTensor<Q_T> aL1Tensor =
                    l1QPTensor[sfaQueryBufferIndex * L1_BLOCK_OFFSET + (1U - kL1) * aL1PaddingSize];
                if (nL1 == 0U) {
                    if (kL1 == 0U) {
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[sfaQueryBufferIndex]);
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[sfaQueryBufferIndex + 1U]);
                        CopyInMm1AToL1(aL1Tensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM, mL1Size, 256, 0);
                        LocalTensor<Q_T> qRopeTensor = aL1Tensor[mL1SizeAlign * 256];
                        CopyInMm1ARopeToL1(qRopeTensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM, mL1Size);
                    } else {
                        LocalTensor<Q_T> qTmpTensor = aL1Tensor[mL1SizeAlign * 32];
                        CopyInMm1AToL1(qTmpTensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM, mL1Size, 256, 256);
                    }
                    SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[sfaQueryBufferIndex]);
                }
                const uint32_t nextKvBuf = (curStep + 1U < totalStepsCoop) ? issue_kv_coop(curStep + 1U) : 0U;
                WaitFlag<HardEvent::MTE2_MTE1>(sfaMte21KvEvents[curKvBuf]);
                if (nL1 == 0U) {
                    WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[sfaQueryBufferIndex]);
                }
                bL1Tensor = l1KVTensor[curKvBuf * L1_BLOCK_OFFSET];
                LocalTensor cL0Tensor =
                    cL0TensorPingPong[(sfaL0CBufferIndex % 2U) * (L0C_PP_SIZE / sizeof(MM_OUT_T))];
                for (uint32_t kL0 = 0U; kL0 < kL0Loops; kL0++) {
                    WaitFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2U));
                    LocalTensor<KV_T> aL0Tensor =
                        aL0TensorPingPong[(abL0BufIter % 2U) * (L0A_PP_SIZE / sizeof(KV_T))];
                    LoadDataMm1A(aL0Tensor, aL1Tensor, kL0, kL0Size, mL1SizeAlign, kL0Size);
                    LocalTensor<KV_T> bL0Tensor =
                        bL0TensorPingPong[(abL0BufIter % 2U) * (L0B_PP_SIZE / sizeof(KV_T))];
                    LoadDataMm1B(bL0Tensor, bL1Tensor, kL0, kL0Size, kL0Size, nL1SizeAlign);
                    SetFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2U));
                    WaitFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2U));
                    MmadParams sfaMmadParams;
                    sfaMmadParams.m = mL1SizeAlign;
                    sfaMmadParams.n = nL1SizeAlign;
                    sfaMmadParams.k = kL0Size;
                    sfaMmadParams.cmatrixInitVal = (kL1 == 0U && kL0 == 0U);
                    sfaMmadParams.cmatrixSource = false;
                    sfaMmadParams.unitFlag = (kL1 == 1U && kL0 + 1U == kL0Loops) ? 0b11 : 0b10;
                    Mmad(cL0Tensor, aL0Tensor, bL0Tensor, sfaMmadParams);
                    if (((sfaMmadParams.m / 16U) * (sfaMmadParams.n / 16U)) < 10U) {
                        PipeBarrier<PIPE_M>();
                    }
                    SetFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2U));
                    abL0BufIter++;
                }
                if (nL1 + 1U == nL1Loops) {
                    SetFlag<HardEvent::MTE1_MTE2>(mte21QPIds[sfaQueryBufferIndex]);
                }
                if (kL1 == 1U) {
                    FixpipeParamsV220 fixParams;
                    fixParams.nSize = nL1SizeAlign;
                    fixParams.mSize = mL1SizeAlign;
                    fixParams.srcStride = mL1SizeAlign;
                    fixParams.dstStride = sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign;
                    fixParams.unitFlag = 0b11;
                    fixParams.ndNum = 1;
                    Fixpipe(mm1ResBaseGm_[mm1Base + nL1 * N_SPLIT_SIZE +
                                          sfaCubeSplitInfo.nBufferStartM *
                                              sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign],
                            cL0Tensor, fixParams);
                }
                SetFlag<HardEvent::MTE1_MTE2>(sfaMte21KvEvents[curKvBuf]);
                curKvBuf = nextKvBuf;
            }
            sfaL0CBufferIndex++;
        }
        qpL1BufIter += 1U;
        return;
    }
#endif

    for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) {
        if (nL1 == (nL1Loops - 1)) {
            nL1Size = nSize - (nL1Loops - 1) * N_SPLIT_SIZE;
            nL1SizeAlign = SFAAlign(nL1Size, 16U);
        }
        for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) {
            kvL1BufIter++;
            uint32_t sfaKvBufferIndex = kvL1BufIter % 3;
            WaitFlag<HardEvent::MTE1_MTE2>(sfaMte21KvEvents[sfaKvBufferIndex]);
            bL1Tensor = l1KVTensor[sfaKvBufferIndex * L1_BLOCK_OFFSET];
            if (kL1 == 0) {
                Nd2NzParams sfaCubeCopyParams;
                sfaCubeCopyParams.ndNum = 1;
                sfaCubeCopyParams.nValue = nL1Size;
                sfaCubeCopyParams.dValue = sfaCubeConstInfo.headDim >> 1;
                sfaCubeCopyParams.srcDValue = sfaCubeConstInfo.headDim;
                sfaCubeCopyParams.dstNzC0Stride = nL1SizeAlign;
                sfaCubeCopyParams.dstNzNStride = 1;
                sfaCubeCopyParams.srcNdMatrixStride = 0;
                sfaCubeCopyParams.dstNzMatrixStride = 0;
                DataCopy(bL1Tensor, kvMergeBaseGm_[kvBase + nL1 * N_SPLIT_SIZE * sfaCubeConstInfo.headDim],
                         sfaCubeCopyParams);
                sfaCubeCopyParams.dValue = sfaCubeConstInfo.headDimRope >> 1;
                sfaCubeCopyParams.srcDValue = sfaCubeConstInfo.headDimRope;
                DataCopy(bL1Tensor[nL1SizeAlign * (sfaCubeConstInfo.headDim >> 1)],
                         kvMergeBaseGm_[kvBase + N_WORKSPACE_SIZE * sfaCubeConstInfo.headDim +
                                        nL1 * N_SPLIT_SIZE * sfaCubeConstInfo.headDimRope],
                         sfaCubeCopyParams);
            } else {
                LocalTensor<Q_T> kTmpTensor = bL1Tensor[(sfaCubeConstInfo.headDimRope >> 1) * nL1SizeAlign];
                Nd2NzParams sfaCubeCopyParams;
                sfaCubeCopyParams.ndNum = 1;
                sfaCubeCopyParams.nValue = nL1Size;
                sfaCubeCopyParams.dValue = sfaCubeConstInfo.headDim >> 1;
                sfaCubeCopyParams.srcDValue = sfaCubeConstInfo.headDim;
                sfaCubeCopyParams.dstNzC0Stride = nL1SizeAlign;
                sfaCubeCopyParams.dstNzNStride = 1;
                sfaCubeCopyParams.srcNdMatrixStride = 0;
                sfaCubeCopyParams.dstNzMatrixStride = 0;
                DataCopy(kTmpTensor,
                         kvMergeBaseGm_[kvBase + (sfaCubeConstInfo.headDim >> 1) +
                                        nL1 * N_SPLIT_SIZE * sfaCubeConstInfo.headDim],
                         sfaCubeCopyParams);
                sfaCubeCopyParams.dValue = sfaCubeConstInfo.headDimRope >> 1;
                sfaCubeCopyParams.srcDValue = sfaCubeConstInfo.headDimRope;
                DataCopy(bL1Tensor,
                         kvMergeBaseGm_[kvBase + N_WORKSPACE_SIZE * sfaCubeConstInfo.headDim +
                                        (sfaCubeConstInfo.headDimRope >> 1) +
                                        nL1 * N_SPLIT_SIZE * sfaCubeConstInfo.headDimRope],
                         sfaCubeCopyParams);
            }
            SetFlag<HardEvent::MTE2_MTE1>(sfaMte21KvEvents[sfaKvBufferIndex]);
            WaitFlag<HardEvent::MTE2_MTE1>(sfaMte21KvEvents[sfaKvBufferIndex]);
            mL1Size = M_SPLIT_SIZE;
            mL1SizeAlign = SFAAlign(M_SPLIT_SIZE, 16U);
            for (uint32_t mL1 = 0; mL1 < sfaML1LoopCount; mL1++) {
                uint32_t aL1PaddingSize = 0;
                if (mL1 == (sfaML1LoopCount - 1)) {
                    mL1Size = mSize - (sfaML1LoopCount - 1) * M_SPLIT_SIZE;
                    mL1SizeAlign = SFAAlign(mL1Size, 16U);
                    aL1PaddingSize = (M_SPLIT_SIZE - mL1SizeAlign) * 288;
                }
                uint32_t mIdx = qpL1BufIter + mL1;
                sfaQueryBufferIndex = GetQPL1RealIdx(mIdx, kL1);
                LocalTensor<Q_T> aL1Tensor =
                    l1QPTensor[sfaQueryBufferIndex * L1_BLOCK_OFFSET + (1 - kL1) * aL1PaddingSize];
                if (nL1 == 0) {
                    if (kL1 == 0) {
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[sfaQueryBufferIndex]);
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[sfaQueryBufferIndex + 1]);
                        CopyInMm1AToL1(aL1Tensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE,
                                       mL1Size, 256, 0);
                        LocalTensor<Q_T> qRopeTensor = aL1Tensor[mL1SizeAlign * 256];
                        CopyInMm1ARopeToL1(qRopeTensor, sfaCubeRunInfo,
                                           sfaCubeSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE, mL1Size);
                    } else {
                        LocalTensor<Q_T> qTmpTensor = aL1Tensor[mL1SizeAlign * 32];
                        CopyInMm1AToL1(qTmpTensor, sfaCubeRunInfo, sfaCubeSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE,
                                       mL1Size, 256, 256);
                    }
                    SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[sfaQueryBufferIndex]);
                    WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[sfaQueryBufferIndex]);
                }
                LocalTensor cL0Tensor =
                    cL0TensorPingPong[(sfaL0CBufferIndex % 2) * (L0C_PP_SIZE / sizeof(MM_OUT_T))];
                for (uint32_t kL0 = 0; kL0 < kL0Loops; kL0++) {
                    WaitFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(abL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];
                    LoadDataMm1A(aL0Tensor, aL1Tensor, kL0, kL0Size, mL1SizeAlign, kL0Size);
                    LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(abL0BufIter % 2) * (L0B_PP_SIZE / sizeof(KV_T))];
                    LoadDataMm1B(bL0Tensor, bL1Tensor, kL0, kL0Size, kL0Size, nL1SizeAlign);
                    SetFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));
                    WaitFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));

                    MmadParams sfaMmadParams;
                    sfaMmadParams.m = mL1SizeAlign;
                    sfaMmadParams.n = nL1SizeAlign;
                    sfaMmadParams.k = kL0Size;
                    sfaMmadParams.cmatrixInitVal = (kL1 == 0 && kL0 == 0);
                    sfaMmadParams.cmatrixSource = false;
                    sfaMmadParams.unitFlag = (kL1 == 1 && kL0 == (kL0Loops - 1)) ? 0b11 : 0b10;
                    Mmad(cL0Tensor, aL0Tensor, bL0Tensor, sfaMmadParams);

                    if (((sfaMmadParams.m / 16) * (sfaMmadParams.n / 16)) < 10) {
                        PipeBarrier<PIPE_M>();
                    }
                    SetFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    abL0BufIter++;
                }
                if (nL1 == (nL1Loops - 1)) {
                    SetFlag<HardEvent::MTE1_MTE2>(mte21QPIds[sfaQueryBufferIndex]);
                }
                if (kL1 == 1) {
                    FixpipeParamsV220 fixParams;
                    fixParams.nSize = nL1SizeAlign;
                    fixParams.mSize = mL1SizeAlign;
                    fixParams.srcStride = mL1SizeAlign;
                    fixParams.dstStride = sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign;
                    fixParams.unitFlag = 0b11;
                    fixParams.ndNum = 1;
                    Fixpipe(mm1ResBaseGm_[mm1Base + nL1 * N_SPLIT_SIZE +
                                          (sfaCubeSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE) *
                                              sfaCubeRunInfo.actualSingleProcessSInnerSizeAlign],
                            cL0Tensor, fixParams);
                }
                if (sfaML1LoopCount == 2) {
                    sfaL0CBufferIndex++;
                }
            }
            SetFlag<HardEvent::MTE1_MTE2>(sfaMte21KvEvents[sfaKvBufferIndex]);
        }
        if (sfaML1LoopCount == 1) {
            sfaL0CBufferIndex++;
        }
    }
    qpL1BufIter += sfaML1LoopCount;
}

#endif
