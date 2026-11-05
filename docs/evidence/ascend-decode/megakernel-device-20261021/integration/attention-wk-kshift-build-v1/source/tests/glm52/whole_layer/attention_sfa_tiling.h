#ifndef TESTS_GLM52_WHOLE_LAYER_ATTENTION_SFA_TILING_H
#define TESTS_GLM52_WHOLE_LAYER_ATTENTION_SFA_TILING_H

#include <cstdint>

// Named members consumed by the arch22 SFA device templates. The standalone
// kernel constructs this object locally; it does not deserialize a CANN op's
// generated tiling blob. Keep members in sync with the recorded vendor source.
struct SparseFlashAttentionBaseParamsMla {
    uint32_t batchSize, seqSize, qSeqSize;
    int64_t blockSize;
    uint32_t maxBlockNumPerBatch;
    float scaleValue;
    uint32_t nNumOfQInOneGroup, actualLenDimsQ, actualLenDimsKV;
    uint32_t outputLayout, sparseMode;
    int64_t preTokens, nextTokens;
    uint32_t attentionMode, returnSoftmaxLse;
    int64_t sparseBlockSize;
    uint32_t sparseBlockCount, isActualLenDimsNull, isActualLenDimsKVNull;
    uint32_t keyStride0;
};

struct SparseFlashAttentionSplitKVParamsMla {
    uint32_t s2, accumOutSize, logSumExpSize;
};

struct SparseFlashAttentionSingleCoreParamsMla {
    uint32_t usedCoreNum;
};

struct SparseFlashAttentionSingleCoreTensorSizeMla {
    uint32_t mmResUbSize, bmm2ResUbSize;
};

struct SparseFlashAttentionInnerSplitParams {
    uint32_t mBaseSize, s2BaseSize;
};

struct SparseFlashAttentionTilingDataMla {
    SparseFlashAttentionBaseParamsMla baseParams;
    SparseFlashAttentionSplitKVParamsMla splitKVParams;
    SparseFlashAttentionSingleCoreParamsMla singleCoreParams;
    SparseFlashAttentionSingleCoreTensorSizeMla singleCoreTensorSize;
    SparseFlashAttentionInnerSplitParams innerSplitParams;
};

#endif
