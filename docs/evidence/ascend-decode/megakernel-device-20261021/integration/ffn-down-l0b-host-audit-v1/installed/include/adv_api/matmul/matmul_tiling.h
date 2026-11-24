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
 * \file matmul_tiling.h
 * \brief
 */

#ifndef LIB_MATMUL_MATMUL_TILING_H
#define LIB_MATMUL_MATMUL_TILING_H

#include <cstdint>
#include <string>
#include "matmul_tiling_base.h"
#include "matmul_tilingdata.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace matmul_tiling {
/**
 * @class MatmulApiTiling
 * @brief single core matmul tiling
 */
class MatmulApiTiling : public MatmulApiTilingBase {
public:
    MatmulApiTiling() {};
    explicit MatmulApiTiling(const platform_ascendc::PlatformAscendC& ascendcPlatform)
        : MatmulApiTilingBase(ascendcPlatform) {};
    explicit MatmulApiTiling(const PlatformInfo& platform) : MatmulApiTilingBase(platform) {};
    ~MatmulApiTiling() override = default;

    /**
     * @brief Get caculated tiling information
     * @param [in] tiling the structure to store the tiling result defined on the Host side
     * @return return 0 if success, else return -1
     */
    int64_t GetTiling(optiling::TCubeTiling& tiling) override;

    /**
     * @brief Get caculated tiling information
     * @param [in] tiling the structure to store the tiling result defined on the Kernel side
     * @return return 0 if success, else return -1
     */
    int64_t GetTiling(AscendC::tiling::TCubeTiling& tiling) override;

    /**
     * @brief Whether can enable L1BankConflictOptimise
     * @return return true if can, else return false
     */
    bool EnableL1BankConflictOptimise();

protected:
    int64_t Compute() override;
};
} // namespace matmul_tiling

extern "C" {
/**
 * @brief After invoke GetTiling function, obtain the used size of L1/UB/L0C buffer based on TCubeTiling
 * @param [in] tiling tiling information defined on the Host side
 * @param [in] bufSize the structure to store the used size of L1/UB/L0C buffer
 * @return return 0 if success, else return -1
 */
int32_t MatmulGetTmpBufSize(optiling::TCubeTiling& tiling, matmul_tiling::SysTilingTempBufSize& bufSize);
/**
 * @brief After invoke GetTiling function, obtain the used size of L1/UB/L0C buffer based on TCubeTiling
 * @param [in] tiling tiling information defined on the Kernel side
 * @param [in] bufSize the structure to store the used sizeof L1/UB/L0C buffer
 * @return return 0 if success, else return -1
 */
int32_t MatmulGetTmpBufSizeV2(AscendC::tiling::TCubeTiling& tiling, matmul_tiling::SysTilingTempBufSize& bufSize);
};

#endif // MATMUL_API_TILING_H
