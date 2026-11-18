

#ifndef SERVICE_SCATTER_CACHE_ARCH22_H
#define SERVICE_SCATTER_CACHE_ARCH22_H

#include "mla_prolog_comm_arch22.h"

namespace MlaProlog {
struct ScatterCacheParams {
    int64_t blockSize;
    int64_t paTokenIndex;
    int64_t row;
    int64_t col;
    int64_t stride;
    int64_t cacheStride0;
    int64_t seqLength;
    int64_t tokenIndex;
};

__aicore__ inline int64_t GetCacheOffset(int64_t paTokenIndex, int64_t blockSize, int64_t tokenStride,
                                         int64_t cacheStride0)
{
    if (blockSize <= 0) {
        return paTokenIndex * cacheStride0;
    }
    int64_t blockIndex = paTokenIndex / blockSize;
    int64_t tokenIndexInBlock = paTokenIndex % blockSize;
    return blockIndex * cacheStride0 + tokenIndexInBlock * tokenStride;
}


template <typename T, bool IS_NZ>
__aicore__ inline void ScatterCacheUnAligned(const GlobalTensor<T>& cacheGm, const LocalTensor<T>& inputLocal,
                                             const ScatterCacheParams& scatterCacheParams)
{
    if (scatterCacheParams.paTokenIndex < 0) {
        return;
    }
    if constexpr (!IS_NZ) {
        int64_t cacheOffset = GetCacheOffset(scatterCacheParams.paTokenIndex, scatterCacheParams.blockSize,
                                             scatterCacheParams.stride, scatterCacheParams.cacheStride0);

        DataCopyParams dataCopyParams{1, static_cast<uint16_t>(scatterCacheParams.col * sizeof(T)), 0, 0};
        DataCopyPad(cacheGm[cacheOffset], inputLocal, dataCopyParams);
    }
}

template <typename T, bool IS_NZ>
__aicore__ inline void ScatterCacheMultiRows(GlobalTensor<T>& cacheGm, const LocalTensor<T>& inputLocal,
                                             const ScatterCacheParams& scatterCacheParams, int64_t rowsInCurBatch,
                                             int64_t a22CacheOffset, int64_t nextBatchOffset)
{
    if (a22CacheOffset < 0) {
        return;
    }
    int64_t copyCnt = rowsInCurBatch * scatterCacheParams.col;

    if constexpr (!IS_NZ) {
        DataCopy(cacheGm[a22CacheOffset], inputLocal, copyCnt);
        if (rowsInCurBatch != scatterCacheParams.row) {
            DataCopy(cacheGm[nextBatchOffset], inputLocal[copyCnt],
                     (scatterCacheParams.row - rowsInCurBatch) * scatterCacheParams.col);
        }
    } else {
        constexpr uint8_t a22Col0 = ALIGN_BLOCK_SIZE / sizeof(T);
        DataCopyParams copyParams{static_cast<uint16_t>(scatterCacheParams.col / a22Col0), 1, 0,
                                  static_cast<uint16_t>(scatterCacheParams.blockSize - 1)};
        DataCopy(cacheGm[a22CacheOffset], inputLocal, copyParams);
        if (rowsInCurBatch != scatterCacheParams.row) {
            for (int a22Row = 0; a22Row < scatterCacheParams.row - rowsInCurBatch; ++a22Row) {
                DataCopy(cacheGm[nextBatchOffset + a22Col0 * a22Row], inputLocal[copyCnt + a22Col0 * a22Row],
                         copyParams);
            }
        }
    }
}

template <typename T, bool IS_NZ>
__aicore__ inline void ScatterCache(const GlobalTensor<T>& cacheGm, const LocalTensor<T>& inputLocal,
                                    const ScatterCacheParams& scatterCacheParams)
{
    if (scatterCacheParams.paTokenIndex < 0) {
        return;
    }
    if constexpr (!IS_NZ) {
        int64_t a22CacheOffset = GetCacheOffset(scatterCacheParams.paTokenIndex, scatterCacheParams.blockSize,
                                                scatterCacheParams.stride, scatterCacheParams.cacheStride0);
        DataCopy(cacheGm[a22CacheOffset], inputLocal, scatterCacheParams.col);
    } else {
        constexpr uint8_t a22Col0 = ALIGN_BLOCK_SIZE / sizeof(T);
        int64_t a22CacheOffset = GetCacheOffset(scatterCacheParams.paTokenIndex, scatterCacheParams.blockSize, a22Col0,
                                                scatterCacheParams.cacheStride0);
        DataCopyParams copyParams{static_cast<uint16_t>(scatterCacheParams.col / a22Col0), 1, 0,
                                  static_cast<uint16_t>(scatterCacheParams.blockSize - 1)};
        DataCopy(cacheGm[a22CacheOffset], inputLocal, copyParams);
    }
}

template <typename T, bool IS_NZ>
__aicore__ inline void MaterializeOffsetsWithHeadSize(int64_t pageId, int64_t tokenOffsetInPage, int64_t rowsThisStep,
                                                      bool spill, int64_t nextPageId, int64_t headSize,
                                                      int64_t cacheStride0, CkvkrParams& ckvkrParams)
{
    ckvkrParams.rowsInCurBatch = rowsThisStep;
    if constexpr (IS_NZ) {
        constexpr uint8_t a22Col0 = ALIGN_BLOCK_SIZE / sizeof(T);
        ckvkrParams.cacheOffset = pageId * cacheStride0 + tokenOffsetInPage * a22Col0;
    } else {
        ckvkrParams.cacheOffset = pageId * cacheStride0 + tokenOffsetInPage * headSize;
    }
    ckvkrParams.nextBatchOffset = (spill && nextPageId >= 0) ? nextPageId * cacheStride0 : 0;
}

}

#endif
