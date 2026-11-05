#ifndef TESTS_GLM52_WHOLE_LAYER_INDEXER_TILING_H
#define TESTS_GLM52_WHOLE_LAYER_INDEXER_TILING_H

#include <cstdint>

// Plain device-side structs consumed by the explicit vendor template
// instances. These are not serialized CANN host tiling blobs.
struct QLITilingData {
    uint32_t bSize, tSize, n2Size, gSize, s1Size, s2Size, sparseCount;
    uint32_t keyStride0, keyDequantScaleStride0, usedCoreNum, blockSize;
    uint32_t maxBlockNumPerBatch, sparseMode;
};

struct LITilingData {
    uint32_t bSize, n2Size, gSize, s1Size, s2Size, sparseCount;
    uint32_t usedCoreNum, blockSize, maxBlockNumPerBatch, sparseMode;
    int64_t preTokens, nextTokens;
    uint32_t returnValue, keyStride0;
};

#endif
