

#ifndef MLA_PROLOG_TILING_DATA_H
#define MLA_PROLOG_TILING_DATA_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace optiling {

struct MlaPrologBaseParams {
    uint32_t batchSize;
    uint32_t stepBatchSize;
    uint32_t mSubSize;
    uint32_t mSubCoreNum;
    uint32_t stepNumHeadDequant;
    uint32_t tokenSize;
    uint32_t seq1Size;
    uint32_t seq2Size;
    uint32_t headSizeX;
    uint32_t headSizeCq;
    uint32_t headSizeCkv;
    uint32_t headSizeQc;
    uint32_t headSizeQr;
    uint32_t headSizeKr;
    uint32_t numHeadSize;
    uint32_t numHeadKvSize;
    uint32_t dimHeadSizeQc;
    uint32_t dimHeadRope;
    uint32_t blockNum;
    uint32_t blockSize;
    uint32_t mm1BlockNum;
    uint32_t mm2BlockNum;
    uint32_t mm3BlockNum;
    uint32_t mm4BlockNum;
    uint32_t vectorBlockNum;
    uint32_t mm1SingleCoreN;
    uint32_t mm2SingleCoreN;
    uint32_t mm3SingleCoreN;
    uint32_t mm4SingleCoreBatch;
    uint32_t dtileSize;
    uint32_t kvQuantMode;
    uint32_t tileSize;
    uint32_t ckvkrRepoMode;
    uint32_t quantScaleRepoMode;
    uint32_t queryNormFlag;
    float reciprocalCq;
    float epsilonCq;
    float reciprocalCkv;
    float epsilonCkv;
    float kNopeClipAlpha;
    float qcQrScale;
    float kcScale;
    uint16_t isQcQrScaleEnable;
    uint16_t isKcScaleEnable;

    uint16_t mm1NGroup;
    uint16_t mm1KGroup;
    uint32_t mm1SingleCoreK;
    uint16_t mm2NGroup;
    uint16_t mm2KGroup;
    uint32_t mm2SingleCoreK;
    uint64_t kvCacheStride0;
    uint64_t krCacheStride0;
};


struct MlaPrologTilingData {
    MlaPrologBaseParams baseParams;
};
}

#endif
