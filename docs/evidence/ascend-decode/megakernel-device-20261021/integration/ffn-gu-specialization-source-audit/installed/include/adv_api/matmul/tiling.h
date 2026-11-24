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
 * \file tiling.h
 * \brief
 */
#ifndef LIB_MATMUL_TILING_H
#define LIB_MATMUL_TILING_H
#include "kernel_basic_intf.h"
#include "../../../impl/adv_api/detail/matmul/utils/matmul_config_impl.h"

/**
 * @brief Get normal config with custom configuration
 * @param [in] intrinsicsLimit: whether to enable looping data transfer from Global Memory to L1 Buffer
 *                              when the inner axis of input matrix on a single core is greater than or equal to 65535
 * @param [in] batchLoop: whether multi-batch input and multi-batch output are enabled, only for BatchMatmul
 * @param [in] isVecND2NZ: whether to enable the conversion from ND to NZ format using vector instructions
 * @param [in] bmmMode: set the relationship between multi-batch data in A/B matrix and the size of L1 Buffer
 *                      when BatchMatmul Layout is NORMAL
 * @param [in] isMsgReuse: whether the dataPtr in the callback function set by SetSelfDefineData directly passes a value
 * @param [in] iterateOrder: iterate order for matmul operation
 * @param [in] scheduleType: set matmul data transfer mode
 * @param [in] enUnitFlag: whether to enable UnitFlag
 * @param [in] enableMixDualMaster: whether to enable MixDualMaster
 * @param [in] bmmOutMode: set multi-batch output mode
 * @return MatmulConfig with normal setting
 */
__aicore__ constexpr MatmulConfig GetNormalConfig(
    const bool intrinsicsLimit = false, const bool batchLoop = false, const bool isVecND2NZ = false,
    const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1, const bool isMsgReuse = true,
    const IterateOrder iterateOrder = IterateOrder::UNDEF,
    const ScheduleType scheduleType = ScheduleType::INNER_PRODUCT, const bool enUnitFlag = true,
    const bool enableMixDualMaster = false, const BatchOutMode bmmOutMode = BatchOutMode::SINGLE_BATCH)
{
    return {
        .doNorm = true,
        .doBasicBlock = false,
        .doMultiDataLoad = false,
        .basicM = 0,
        .basicN = 0,
        .basicK = 0,
        .intrinsicsCheck = intrinsicsLimit,
        .isNBatch = batchLoop,
        .enVecND2NZ = isVecND2NZ,
        .doSpecialBasicBlock = false,
        .doMTE2Preload = 0,
        .singleCoreM = 0,
        .singleCoreN = 0,
        .singleCoreK = 0,
        .stepM = 0,
        .stepN = 0,
        .baseMN = 0,
        .singleCoreMN = 0,
        .enUnitFlag = enUnitFlag,
        .isPerTensor = false,
        .hasAntiQuantOffset = false,
        .doIBShareNorm = false,
        .doSpecialMDL = false,
        .enableInit = true,
        .batchMode = bmmMode,
        .enableEnd = true,
        .enableGetTensorC = true,
        .enableSetOrgShape = true,
        .enableSetBias = true,
        .enableSetTail = true,
        .enableQuantVector = true,
        .enableSetDefineData = true,
        .iterateMode = IterateMode::ITERATE_MODE_DEFAULT,
        .enableReuse = isMsgReuse,
        .enableUBReuse = true,
        .enableL1CacheUB = false,
        .intraBlockPartSum = false,
        .iterateOrder = iterateOrder,
        .scheduleType = scheduleType,
        .enableDoubleCache = false,
        .isBiasBatch = true,
        .enableStaticPadZeros = false,
        .isPartialOutput = false,
        .enableMixDualMaster = enableMixDualMaster,
        .isA2B2Shared = false,
        .isEnableChannelSplit = false,
        .enableKdimReorderLoad = false,
        .isCO1Shared = false,
        .sharedCO1BufferSize = SHARED_CO1_BUFFER_SIZE,
        .bmmOutMode = bmmOutMode,
        .enableL1BankConflictOptimise = false,
        .enableRelu = false,
        .decompMode = DecompressionMode::DECOMP_NONE};
}

/**
 * @brief Get MDL config with custom configuration
 * @param [in] intrinsicsLimit: whether to enable looping data transfer from Global Memory to L1 Buffer
 *                              when the inner axis of input matrix on a single core is greater than or equal to 65535
 * @param [in] batchLoop: whether multi-batch input and multi-batch output are enabled, only for BatchMatmul
 * @param [in] doMTE2Preload: enable M/N direction preload
 * @param [in] isVecND2NZ: whether to enable the conversion from ND to NZ format using vector instructions
 * @param [in] isPerTensor: whether to enable B matrix per tensor quantization when the input is A16W8
 * @param [in] hasAntiQuantOffset: whether to enable B matrix quantization by using offset coefficient
 *                                 when the input is A16W8
 * @param [in] enUnitFlag: whether to enable UnitFlag
 * @param [in] isMsgReuse: whether the dataPtr in the callback function set by SetSelfDefineData directly passes a value
 * @param [in] enableUBReuse: whether to reuse Unified Buffer space
 * @param [in] enableL1CacheUB: whether to enable L1 Buffer caching for Unified Buffer compute blocks
 * @param [in] enableMixDualMaster: whether to enable MixDualMaster
 * @param [in] enableKdimReorderLoad: whether to enable KdimReorder
 * @param [in] enableL1BankConflictOptimise: whether to enable L1BankConflictOptimise
 * @return MatmulConfig with MDL setting
 */
__aicore__ constexpr MatmulConfig GetMDLConfig(
    const bool intrinsicsLimit = false, const bool batchLoop = false, const uint32_t doMTE2Preload = 0,
    const bool isVecND2NZ = false, bool isPerTensor = false, bool hasAntiQuantOffset = false,
    const bool enUnitFlag = false, const bool isMsgReuse = true, const bool enableUBReuse = true,
    const bool enableL1CacheUB = false, const bool enableMixDualMaster = false,
    const bool enableKdimReorderLoad = false, const bool enableL1BankConflictOptimise = false)
{
    return {
        .doNorm = false,
        .doBasicBlock = false,
        .doMultiDataLoad = true,
        .basicM = 0,
        .basicN = 0,
        .basicK = 0,
        .intrinsicsCheck = intrinsicsLimit,
        .isNBatch = batchLoop,
        .enVecND2NZ = isVecND2NZ,
        .doSpecialBasicBlock = false,
        .doMTE2Preload = doMTE2Preload,
        .singleCoreM = 0,
        .singleCoreN = 0,
        .singleCoreK = 0,
        .stepM = 0,
        .stepN = 0,
        .baseMN = 0,
        .singleCoreMN = 0,
        .enUnitFlag = enUnitFlag,
        .isPerTensor = isPerTensor,
        .hasAntiQuantOffset = hasAntiQuantOffset,
        .doIBShareNorm = false,
        .doSpecialMDL = false,
        .enableInit = true,
        .batchMode = BatchMode::NONE,
        .enableEnd = true,
        .enableGetTensorC = true,
        .enableSetOrgShape = true,
        .enableSetBias = true,
        .enableSetTail = true,
        .enableQuantVector = true,
        .enableSetDefineData = true,
        .iterateMode = IterateMode::ITERATE_MODE_DEFAULT,
        .enableReuse = isMsgReuse,
        .enableUBReuse = enableUBReuse,
        .enableL1CacheUB = enableL1CacheUB,
        .intraBlockPartSum = false,
        .iterateOrder = IterateOrder::UNDEF,
        .scheduleType = ScheduleType::INNER_PRODUCT,
        .enableDoubleCache = false,
        .isBiasBatch = true,
        .enableStaticPadZeros = false,
        .isPartialOutput = false,
        .enableMixDualMaster = enableMixDualMaster,
        .isA2B2Shared = false,
        .isEnableChannelSplit = false,
        .enableKdimReorderLoad = enableKdimReorderLoad,
        .isCO1Shared = false,
        .sharedCO1BufferSize = SHARED_CO1_BUFFER_SIZE,
        .bmmOutMode = BatchOutMode::SINGLE_BATCH,
        .enableL1BankConflictOptimise = enableL1BankConflictOptimise,
        .enableRelu = false,
        .decompMode = DecompressionMode::DECOMP_NONE};
}

/**
 * @brief Get special MDL config with custom configuration
 * @param [in] intrinsicsLimit: whether to enable looping data transfer from Global Memory to L1 Buffer
 *                              when the inner axis of input matrix on a single core is greater than or equal to 65535
 * @param [in] batchLoop: whether multi-batch input and multi-batch output are enabled, only for BatchMatmul
 * @param [in] doMTE2Preload: enable M/N direction preload
 * @param [in] isVecND2NZ: whether to enable the conversion from ND to NZ format using vector instructions
 * @param [in] isPerTensor: whether to enable B matrix per tensor quantization when the input is A16W8
 * @param [in] hasAntiQuantOffset: whether to enable B matrix quantization by using offset coefficient
 *                                 when the input is A16W8
 * @return MatmulConfig with special MDL setting
 */
__aicore__ constexpr MatmulConfig GetSpecialMDLConfig(
    const bool intrinsicsLimit = false, const bool batchLoop = false, const uint32_t doMTE2Preload = 0,
    const bool isVecND2NZ = false, bool isPerTensor = false, bool hasAntiQuantOffset = false)
{
    return {
        .doNorm = false,
        .doBasicBlock = false,
        .doMultiDataLoad = false,
        .basicM = 0,
        .basicN = 0,
        .basicK = 0,
        .intrinsicsCheck = intrinsicsLimit,
        .isNBatch = batchLoop,
        .enVecND2NZ = isVecND2NZ,
        .doSpecialBasicBlock = false,
        .doMTE2Preload = doMTE2Preload,
        .singleCoreM = 0,
        .singleCoreN = 0,
        .singleCoreK = 0,
        .stepM = 0,
        .stepN = 0,
        .baseMN = 0,
        .singleCoreMN = 0,
        .enUnitFlag = false,
        .isPerTensor = isPerTensor,
        .hasAntiQuantOffset = hasAntiQuantOffset,
        .doIBShareNorm = false,
        .doSpecialMDL = true,
        .enableInit = true,
        .batchMode = BatchMode::NONE,
        .enableEnd = true,
        .enableGetTensorC = true,
        .enableSetOrgShape = true,
        .enableSetBias = true,
        .enableSetTail = true,
        .enableQuantVector = true,
        .enableSetDefineData = true,
        .iterateMode = IterateMode::ITERATE_MODE_DEFAULT,
        .enableReuse = true,
        .enableUBReuse = true,
        .enableL1CacheUB = false,
        .intraBlockPartSum = false,
        .iterateOrder = IterateOrder::UNDEF,
        .scheduleType = ScheduleType::INNER_PRODUCT,
        .enableDoubleCache = false,
        .isBiasBatch = true,
        .enableStaticPadZeros = false,
        .isPartialOutput = false,
        .enableMixDualMaster = false,
        .isA2B2Shared = false,
        .isEnableChannelSplit = false,
        .enableKdimReorderLoad = false,
        .isCO1Shared = false,
        .sharedCO1BufferSize = SHARED_CO1_BUFFER_SIZE,
        .bmmOutMode = BatchOutMode::SINGLE_BATCH,
        .enableL1BankConflictOptimise = false,
        .enableRelu = false,
        .decompMode = DecompressionMode::DECOMP_NONE};
}

/**
 * @brief Get basic config with custom configuration
 * @param [in] basicM: size of M-axis shape involved in once mmad instruction
 * @param [in] basicN: size of N-axis shape involved in once mmad instruction
 * @param [in] basicK: size of K-axis shape involved in once mmad instruction
 * @param [in] intrinsicsLimit: whether to enable looping data transfer from Global Memory to L1 Buffer
 *                              when the inner axis of input matrix on a single core is greater than or equal to 65535
 * @param [in] batchLoop: whether multi-batch input and multi-batch output are enabled, only for BatchMatmul
 * @param [in] bmmMode: set the relationship between multi-batch data in A/B matrix and the size of L1 Buffer
 *                      when BatchMatmul Layout is NORMAL
 * @return MatmulConfig with basic setting
 */
__aicore__ constexpr MatmulConfig GetBasicConfig(
    const uint32_t basicM, const uint32_t basicN, const uint32_t basicK, const bool intrinsicsLimit = false,
    const bool batchLoop = false, const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1)
{
    return {
        .doNorm = false,
        .doBasicBlock = true,
        .doMultiDataLoad = false,
        .basicM = basicM,
        .basicN = basicN,
        .basicK = basicK,
        .intrinsicsCheck = intrinsicsLimit,
        .isNBatch = batchLoop,
        .enVecND2NZ = false,
        .doSpecialBasicBlock = false,
        .doMTE2Preload = 0,
        .singleCoreM = 0,
        .singleCoreN = 0,
        .singleCoreK = 0,
        .stepM = 0,
        .stepN = 0,
        .baseMN = 0,
        .singleCoreMN = 0,
        .enUnitFlag = false,
        .isPerTensor = false,
        .hasAntiQuantOffset = false,
        .doIBShareNorm = false,
        .doSpecialMDL = false,
        .enableInit = true,
        .batchMode = bmmMode,
        .enableEnd = true,
        .enableGetTensorC = true,
        .enableSetOrgShape = true,
        .enableSetBias = true,
        .enableSetTail = true,
        .enableQuantVector = true,
        .enableSetDefineData = true,
        .iterateMode = IterateMode::ITERATE_MODE_DEFAULT,
        .enableReuse = true,
        .enableUBReuse = true,
        .enableL1CacheUB = false,
        .intraBlockPartSum = false,
        .iterateOrder = IterateOrder::UNDEF,
        .scheduleType = ScheduleType::INNER_PRODUCT,
        .enableDoubleCache = false,
        .isBiasBatch = true,
        .enableStaticPadZeros = false,
        .isPartialOutput = false,
        .enableMixDualMaster = false,
        .isA2B2Shared = false,
        .isEnableChannelSplit = false,
        .enableKdimReorderLoad = false,
        .isCO1Shared = false,
        .sharedCO1BufferSize = SHARED_CO1_BUFFER_SIZE,
        .bmmOutMode = BatchOutMode::SINGLE_BATCH,
        .enableL1BankConflictOptimise = false,
        .enableRelu = false,
        .decompMode = DecompressionMode::DECOMP_NONE};
}

/**
 * @brief Get IBShare normal config with custom configuration
 * @param [in] intrinsicsLimit: whether to enable looping data transfer from Global Memory to L1 Buffer
 *                              when the inner axis of input matrix on a single core is greater than or equal to 65535
 * @param [in] batchLoop: whether multi-batch input and multi-batch output are enabled, only for BatchMatmul
 * @param [in] isVecND2NZ: whether to enable the conversion from ND to NZ format using vector instructions
 * @param [in] bmmMode: set the relationship between multi-batch data in A/B matrix and the size of L1 Buffer
 *                      when BatchMatmul Layout is NORMAL
 * @param [in] isDoubleCache: whether L1 Buffer caches two blocks of data at the same time after using IBShare
 * @param [in] enUnitFlag: whether to enable UnitFlag
 * @return MatmulConfig with IBShare normal setting
 */
__aicore__ constexpr MatmulConfig GetIBShareNormConfig(
    const bool intrinsicsLimit = false, const bool batchLoop = false, const bool isVecND2NZ = false,
    const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1, const bool isDoubleCache = false,
    const bool enUnitFlag = true)
{
    return {
        .doNorm = false,
        .doBasicBlock = false,
        .doMultiDataLoad = false,
        .basicM = 0,
        .basicN = 0,
        .basicK = 0,
        .intrinsicsCheck = intrinsicsLimit,
        .isNBatch = batchLoop,
        .enVecND2NZ = isVecND2NZ,
        .doSpecialBasicBlock = false,
        .doMTE2Preload = false,
        .singleCoreM = 0,
        .singleCoreN = 0,
        .singleCoreK = 0,
        .stepM = 0,
        .stepN = 0,
        .baseMN = 0,
        .singleCoreMN = 0,
        .enUnitFlag = enUnitFlag,
        .isPerTensor = false,
        .hasAntiQuantOffset = false,
        .doIBShareNorm = true,
        .doSpecialMDL = false,
        .enableInit = true,
        .batchMode = bmmMode,
        .enableEnd = true,
        .enableGetTensorC = true,
        .enableSetOrgShape = true,
        .enableSetBias = true,
        .enableSetTail = true,
        .enableQuantVector = true,
        .enableSetDefineData = true,
        .iterateMode = IterateMode::ITERATE_MODE_DEFAULT,
        .enableReuse = true,
        .enableUBReuse = true,
        .enableL1CacheUB = false,
        .intraBlockPartSum = false,
        .iterateOrder = IterateOrder::UNDEF,
        .scheduleType = ScheduleType::INNER_PRODUCT,
        .enableDoubleCache = isDoubleCache,
        .isBiasBatch = true,
        .enableStaticPadZeros = false,
        .isPartialOutput = false,
        .enableMixDualMaster = false,
        .isA2B2Shared = false,
        .isEnableChannelSplit = false,
        .enableKdimReorderLoad = false,
        .isCO1Shared = false,
        .sharedCO1BufferSize = SHARED_CO1_BUFFER_SIZE,
        .bmmOutMode = BatchOutMode::SINGLE_BATCH,
        .enableL1BankConflictOptimise = false,
        .enableRelu = false,
        .decompMode = DecompressionMode::DECOMP_NONE};
}

constexpr MatmulConfig CFG_NORM = GetNormalConfig();
constexpr MatmulConfig CFG_MDL = GetMDLConfig();
constexpr MatmulConfig MM_CFG_BB = GetBasicConfig(128, 128, 128);
constexpr MatmulConfig CFG_IBSHARE_NORM = GetIBShareNormConfig();

/**
 * @brief Get matmul config with custom params
 * \tparam [in] configMode: MatmulConfig template
 * \tparam [in] ArgTypes: variable template params
 * @param [in] args: variable params, one or more of
 *                   MatmulShapeParams/MatmulQuantParams/MatmulBatchParams/MatmulFuncParams in any order
 * @return MatmulConfig with custom setting
 */
template <MatmulConfigMode configMode, typename... ArgTypes>
__aicore__ inline constexpr MatmulConfig GetMMConfig(ArgTypes&&... args)
{
    MatmulConfig mmConfig = CFG_NORM;
    if constexpr (configMode == MatmulConfigMode::CONFIG_MDL) {
        mmConfig = CFG_MDL;
    } else if constexpr (configMode == MatmulConfigMode::CONFIG_SPECIALMDL) {
        mmConfig = GetSpecialMDLConfig();
    } else if constexpr (configMode == MatmulConfigMode::CONFIG_IBSHARE) {
        mmConfig = CFG_IBSHARE_NORM;
    }
    GetMMConfigImpl(mmConfig, args...);
    return mmConfig;
}

/**
 * @struct MatmulApiStaticTiling
 * @brief Constant tiling struct, include a set of constant tiling parameters and a MatmulConfig
 */
struct MatmulApiStaticTiling {
    int32_t usedCoreNum = -1;     ///< number of AICore used
    int32_t M = -1;               ///< size of original A matrix M-axis shape
    int32_t N = -1;               ///< size of original B matrix N-axis shape
    int32_t Ka = -1;              ///< size of original A matrix K-axis shape
    int32_t Kb = -1;              ///< size of original B matrix K-axis shape
    int32_t singleCoreM = -1;     ///< size of M-axis shape within a single core
    int32_t singleCoreN = -1;     ///< size of N-axis shape within a single core
    int32_t singleCoreK = -1;     ///< size of K-axis shape within a single core
    int32_t baseM = -1;           ///< size of M-axis shape involved in once mmad instruction
    int32_t baseN = -1;           ///< size of N-axis shape involved in once mmad instruction
    int32_t baseK = -1;           ///< size of K-axis shape involved in once mmad instruction
    int32_t depthA1 = -1;         ///< number of baseM * baseK full loaded in A1
    int32_t depthB1 = -1;         ///< number of baseK * baseN full loaded in B1
    int32_t stepM = -1;           ///< multiple of baseM for A matrix in M-direction of A1
    int32_t stepN = -1;           ///< multiple of baseN for B matrix in N-direction of B1
    int32_t isBias = -1;          ///< whether to enable bias
    int32_t transLength = -1;     ///< size of UB temporary space during the calculation
    int32_t iterateOrder = -1;    ///< order of each Iterate
    int32_t shareMode = -1;       ///< reserved parameter
    int32_t shareL1Size = -1;     ///< reserved parameter
    int32_t shareL0CSize = -1;    ///< reserved parameter
    int32_t shareUbSize = -1;     ///< reserved parameter
    int32_t stepKa = -1;          ///< multiple of baseK for A matrix in K-direction of A1
    int32_t stepKb = -1;          ///< multiple of baseK for B matrix in K-direction of B1
    int32_t depthAL1CacheUB = -1; ///< number of A matrix L1 Buffer caching for Unified Buffer compute blocks
    int32_t depthBL1CacheUB = -1; ///< number of B matrix L1 Buffer caching for Unified Buffer compute blocks
    int32_t dbL0A = -1;           ///< whether A matrix MTE1 is using double buffer
    int32_t dbL0B = -1;           ///< whether B matrix MTE1 is using double buffer
    int32_t dbL0C = -1;           ///< whether MMAD is using double buffer
    int32_t ALayoutInfoB = -1;    ///< B-axis information of A matrix layout
    int32_t ALayoutInfoS = -1;    ///< S-axis information of A matrix layout
    int32_t ALayoutInfoN = -1;    ///< N-axis information of A matrix layout
    int32_t ALayoutInfoG = -1;    ///< G-axis information of A matrix layout
    int32_t ALayoutInfoD = -1;    ///< D-axis information of A matrix layout
    int32_t BLayoutInfoB = -1;    ///< B-axis information of B matrix layout
    int32_t BLayoutInfoS = -1;    ///< S-axis information of B matrix layout
    int32_t BLayoutInfoN = -1;    ///< N-axis information of B matrix layout
    int32_t BLayoutInfoG = -1;    ///< G-axis information of B matrix layout
    int32_t BLayoutInfoD = -1;    ///< D-axis information of B matrix layout
    int32_t CLayoutInfoB = -1;    ///< B-axis information of C matrix layout
    int32_t CLayoutInfoS1 = -1;   ///< S1-axis information of C matrix layout
    int32_t CLayoutInfoN = -1;    ///< N-axis information of C matrix layout
    int32_t CLayoutInfoG = -1;    ///< G-axis information of C matrix layout
    int32_t CLayoutInfoS2 = -1;   ///< S2-axis information of C matrix layout
    int32_t BatchNum = -1;        ///< batch number of Batch Matmul
    int32_t mxTypePara =
        -1; ///< multiple of scaleA/scaleB load size in L1 compared to the load size of A/B matrix in L1
    MatmulConfig cfg = CFG_NORM; ///< MatmulConfig parameter
};

#endif // LIB_MATMUL_TILING_H
