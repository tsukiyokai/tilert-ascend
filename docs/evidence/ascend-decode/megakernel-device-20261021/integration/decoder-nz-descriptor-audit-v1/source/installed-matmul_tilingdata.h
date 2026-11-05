/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_tilingdata.h
 * \brief
 */

#ifndef LIB_MATMUL_MATMUL_TILINGDATA_H
#define LIB_MATMUL_MATMUL_TILINGDATA_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TCubeTiling)
TILING_DATA_FIELD_DEF(int32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int32_t, M);
TILING_DATA_FIELD_DEF(int32_t, N);
TILING_DATA_FIELD_DEF(int32_t, Ka);
TILING_DATA_FIELD_DEF(int32_t, Kb);
TILING_DATA_FIELD_DEF(int32_t, singleCoreM);
TILING_DATA_FIELD_DEF(int32_t, singleCoreN);
TILING_DATA_FIELD_DEF(int32_t, singleCoreK);
TILING_DATA_FIELD_DEF(int32_t, baseM);
TILING_DATA_FIELD_DEF(int32_t, baseN);
TILING_DATA_FIELD_DEF(int32_t, baseK);
TILING_DATA_FIELD_DEF(int32_t, depthA1);
TILING_DATA_FIELD_DEF(int32_t, depthB1);
TILING_DATA_FIELD_DEF(int32_t, stepM);
TILING_DATA_FIELD_DEF(int32_t, stepN);
TILING_DATA_FIELD_DEF(int32_t, isBias);
TILING_DATA_FIELD_DEF(int32_t, transLength);
TILING_DATA_FIELD_DEF(int32_t, iterateOrder);
TILING_DATA_FIELD_DEF(int32_t, shareMode);
TILING_DATA_FIELD_DEF(int32_t, shareL1Size);
TILING_DATA_FIELD_DEF(int32_t, shareL0CSize);
TILING_DATA_FIELD_DEF(int32_t, shareUbSize);
TILING_DATA_FIELD_DEF(int32_t, batchM);
TILING_DATA_FIELD_DEF(int32_t, batchN);
TILING_DATA_FIELD_DEF(int32_t, singleBatchM);
TILING_DATA_FIELD_DEF(int32_t, singleBatchN);
TILING_DATA_FIELD_DEF(int32_t, stepKa);
TILING_DATA_FIELD_DEF(int32_t, stepKb);
TILING_DATA_FIELD_DEF(int32_t, depthAL1CacheUB);
TILING_DATA_FIELD_DEF(int32_t, depthBL1CacheUB);
TILING_DATA_FIELD_DEF(int32_t, dbL0A);
TILING_DATA_FIELD_DEF(int32_t, dbL0B);
TILING_DATA_FIELD_DEF(int32_t, dbL0C);
TILING_DATA_FIELD_DEF(int32_t, ALayoutInfoB);
TILING_DATA_FIELD_DEF(int32_t, ALayoutInfoS);
TILING_DATA_FIELD_DEF(int32_t, ALayoutInfoN);
TILING_DATA_FIELD_DEF(int32_t, ALayoutInfoG);
TILING_DATA_FIELD_DEF(int32_t, ALayoutInfoD);
TILING_DATA_FIELD_DEF(int32_t, BLayoutInfoB);
TILING_DATA_FIELD_DEF(int32_t, BLayoutInfoS);
TILING_DATA_FIELD_DEF(int32_t, BLayoutInfoN);
TILING_DATA_FIELD_DEF(int32_t, BLayoutInfoG);
TILING_DATA_FIELD_DEF(int32_t, BLayoutInfoD);
TILING_DATA_FIELD_DEF(int32_t, CLayoutInfoB);
TILING_DATA_FIELD_DEF(int32_t, CLayoutInfoS1);
TILING_DATA_FIELD_DEF(int32_t, CLayoutInfoN);
TILING_DATA_FIELD_DEF(int32_t, CLayoutInfoG);
TILING_DATA_FIELD_DEF(int32_t, CLayoutInfoS2);
TILING_DATA_FIELD_DEF(int32_t, BatchNum);
TILING_DATA_FIELD_DEF(int32_t, mxTypePara);
END_TILING_DATA_DEF;
} // namespace optiling

#endif // LIB_MATMUL_MATMUL_TILINGDATA_H
