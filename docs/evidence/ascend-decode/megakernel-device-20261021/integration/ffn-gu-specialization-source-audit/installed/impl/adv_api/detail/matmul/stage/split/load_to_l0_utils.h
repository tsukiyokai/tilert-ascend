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
 * \file load_to_l0_utils.h
 * \brief
 */

#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message("impl/adv_api/detail/matmul/stage/split/load_to_l0_utils.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"adv_api/matmul/matmul.h\"\" and use public functions or variables defined in interface headers files.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_STAGE_SPLIT_LOAD_TO_L0_UTILS_H__
#endif

#ifndef IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_UTILS_H
#define IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_UTILS_H

#include "../../feature_trait/matmul_feature_trait.h"

namespace AscendC {
namespace Impl {
namespace Detail {
constexpr uint16_t HW_N0 = 16;
constexpr uint16_t HW_M0 = 16;
constexpr uint16_t ALIGN_NUM = 16;
constexpr uint64_t M_POS_BIT = 48;
constexpr uint64_t K_POS_BIT = 32;
constexpr uint64_t M_STEP_BIT = 16;
constexpr uint8_t INDEX_SHIFT = 2;
constexpr uint8_t M_STEP_MIN_VAL_B8 = 2;
constexpr uint8_t M_STEP_MIN_VAL_B4 = 4;
constexpr uint8_t K_STEP_MIN_VAL_B32 = 2;
constexpr uint8_t padList[4] = {0, 0, 0, 0};

enum class LoadInstrType {
    LOAD2D,
    LOAD3DV2,
    LOAD2DTRANSPOSE,
    LOAD2DV2,
};

enum class GemvMode {
    MATRIX,
    VECTOR,
    SCALAR,
};

template <typename A_T, const auto& MM_CFG>
__aicore__ inline constexpr LoadInstrType GetLoadInstrType()
{
    if constexpr (MatmulFeatureTrait<MM_CFG>::IsSupportLoad2dV2()) {
        return LoadInstrType::LOAD2DV2;
    }

    if constexpr (MatmulFeatureTrait<MM_CFG>::IsSupportLoad2dTranspose() &&
        IsSameTypeV<A_T, int8_t>) {
            return LoadInstrType::LOAD2DTRANSPOSE;
    }

    if constexpr (MatmulFeatureTrait<MM_CFG>::IsSupportLoad3dV2()) {
        return LoadInstrType::LOAD3DV2;
    }

    return LoadInstrType::LOAD2D;
}

template <typename A_TYPE>
__aicore__ inline constexpr GemvMode GetGemvMode() {
    return (A_TYPE::format == CubeFormat::VECTOR) ? GemvMode::VECTOR :
        ((A_TYPE::format == CubeFormat::SCALAR) ? GemvMode::SCALAR : GemvMode::MATRIX);
}

enum class LoadL0bInstrType {
    LOAD2D,
    LOAD3DV2,
    LOAD2DTRANSPOSE,
    LOAD2DV2,
};

template <typename B_T, const auto& MM_CFG>
__aicore__ inline constexpr LoadL0bInstrType GetLoadL0bInstrType()
{
    if constexpr (AscendC::Impl::Detail::MatmulFeatureTrait<MM_CFG>::IsSupportLoad2dV2()) {
        return LoadL0bInstrType::LOAD2DV2;
    }

    if constexpr (AscendC::Impl::Detail::MatmulFeatureTrait<MM_CFG>::IsSupportLoad2dTranspose() &&
        (IsSameTypeV<B_T, int8_t> || IsSameTypeV<B_T, int4b_t>)) {
            return LoadL0bInstrType::LOAD2DTRANSPOSE;
    }

    if constexpr (AscendC::Impl::Detail::MatmulFeatureTrait<MM_CFG>::IsSupportLoad3dV2()) {
        return LoadL0bInstrType::LOAD3DV2;
    }

    return LoadL0bInstrType::LOAD2D;
}

template <typename INPUT_TYPE>
__aicore__ inline constexpr auto GetAuxDataType()
{
#if defined(__NPU_ARCH__)
    // Mx auxData is fp8, sparse auxData is uint8_t
    if constexpr (HasSparseIndex<INPUT_TYPE>()) {
        uint8_t auxData = 0;
        return auxData;
    } else if constexpr (HasScalePosition<INPUT_TYPE>::value) {
        uint8_t mxData = get_imm(0);
        fp8_e8m0_t mxType = *(reinterpret_cast<fp8_e8m0_t*>(&mxData));
        return mxType;
    } else {
        uint8_t defaultData = 0;
        return defaultData;
    }
#else
    uint8_t defaultData = 0;
    return defaultData;
#endif
}
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif
#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_STAGE_SPLIT_LOAD_TO_L0_UTILS_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_STAGE_SPLIT_LOAD_TO_L0_UTILS_H__
#endif // IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_UTILS_H
