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
 * \file load_to_l0a_load3dv2.h
 * \brief
 */

#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message( \
    "impl/adv_api/detail/matmul/stage/split/load_to_l0a/load_to_l0a_loadInstr.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"adv_api/matmul/matmul.h\"\" and use public functions or variables defined in interface headers files.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_LOAD_TO_L0A_LOADINSTR_H__
#endif

#ifndef IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_LOADINSTR_H
#define IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_LOADINSTR_H

#include "load_to_l0a_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, typename A_TYPE, const auto& MM_CFG>
class LoadToL0A<
    IMPL, A_TYPE, MM_CFG,
    enable_if_t<
        GetGemvMode<A_TYPE>() == GemvMode::MATRIX &&
        !(DoMatmulBasicBlock(MM_CFG) || DoMatmulSpecialBasicBlock(MM_CFG)) &&
        (MatmulFeatureTrait<MM_CFG>::IsSupportLoad3dV2()) && !(MatmulFeatureTrait<MM_CFG>::IsSupportLoad2dV2())>> {
    using A_T = typename A_TYPE::T;
    using AuxDtype = decltype(GetAuxDataType<A_TYPE>());

public:
    __aicore__ inline LoadToL0A(){};
    __aicore__ inline ~LoadToL0A(){};

    __aicore__ inline void SetScalar(A_T scalar){};

    __aicore__ inline void Prepare(bool isATranspose, uint16_t aL1K, uint16_t aL1M) const
    {
        if constexpr (!isFmatrixUpdate_) {
            SetFmatrix(isATranspose, aL1K, aL1M);
        }
    }

    __aicore__ inline void Load(
        const LocalTensor<A_T>& l0A, const LocalTensor<A_T>& l1A, uint16_t aL1M, uint16_t aL1K, uint16_t madM,
        uint16_t madK, uint16_t aL1MOffset, uint16_t aL1KOffset, bool isATranspose,
        const LocalTensor<AuxDtype>& l1AAuxMatrix = {}, uint16_t aAuxL1K = 0, uint16_t aAuxL1KOffset = 0) const
    {
        if constexpr (isFmatrixUpdate_) {
            SetFmatrix(isATranspose, aL1K, aL1M);
        }
        if (isATranspose) {
            TransLoadDataToL0(l0A, l1A, aL1M, aL1K, madM, madK, aL1MOffset, aL1KOffset);
        } else {
            LoadDataToL0(l0A, l1A, aL1K, madM, madK, aL1MOffset, aL1KOffset);
        }
    }

private:
    __aicore__ inline void SetFmatrix(bool isATranspose, uint16_t aL1K, uint16_t aL1M) const
    {
        // fmatrix w should be 16 aligned
        uint16_t wAlign;
        if (isATranspose) {
            wAlign = CeilAlign(aL1K, HW_M0);
        } else {
            wAlign = CeilAlign(aL1M, HW_M0);
        }
        Load3DSetFMatrixCal(1, wAlign, padList);
    }

    __aicore__ inline void TransLoadDataToL0(
        const LocalTensor<A_T>& l0A, const LocalTensor<A_T>& l1A, uint16_t aL1M, uint16_t aL1K, uint16_t madM,
        uint16_t madK, uint16_t aL1MOffset, uint16_t aL1KOffset) const
    {
        // K_axis is m direction, and M_axis is k direction in load3d intrin
        if constexpr (GetLoadInstrType<typename A_TYPE::T, MM_CFG>() == LoadInstrType::LOAD2DTRANSPOSE) {
            uint16_t l0aLoop = Ceil(madM, c0Size_);
            uint8_t l0aRepeat = Ceil(madK, c0Size_);
            uint64_t l0aSrcAddrStride = aL1K * c0Size_;
            uint64_t c0Square = c0Size_ * c0Size_;
            uint64_t l0aDstAddrStride = l0aRepeat * c0Square;

            uint64_t l1aOffset = aL1MOffset * aL1K + aL1KOffset * c0Size_;
            uint64_t l0aOffset = 0;
            // startIndex, repeatTimes, srcStride, dstGap, dstFracGap, addrMode
            LoadData2dTransposeParams loadData2dTransposeParams{
                0, l0aRepeat, 1, 0, static_cast<uint16_t>(l0aRepeat - 1), inc};
            for (uint16_t i = 0; i < l0aLoop; ++i) {
                LoadDataWithTranspose(l0A[l0aOffset], l1A[l1aOffset], loadData2dTransposeParams);
                l1aOffset += l0aSrcAddrStride;
                l0aOffset += l0aDstAddrStride;
            }
        } else {
            // format(K, M), K, M need to be 16 aligned for f32
            uint16_t madMAlign = CeilAlign(madM, ALIGN_NUM);
            uint16_t mStep = CeilAlign(madK, HW_M0);
            uint16_t aL1MAlign = CeilAlign(aL1M, ALIGN_NUM);
            LoadData3DParamsV2Pro loadData3DV2;
            loadData3DV2.channelSize = aL1MAlign;
            loadData3DV2.extConfig = ((uint64_t)aL1KOffset << M_POS_BIT) | ((uint64_t)aL1MOffset << K_POS_BIT) |
                                     ((uint64_t)mStep << M_STEP_BIT) | (uint64_t)madMAlign;
            loadData3DV2.enTranspose = true;
            LoadData<A_T>(l0A[0], l1A[0], loadData3DV2);
        }
    }

    __aicore__ inline void LoadDataToL0(
        const LocalTensor<A_T>& l0A, const LocalTensor<A_T>& l1A, uint16_t aL1K, uint16_t madM, uint16_t madK,
        uint16_t aL1MOffset, uint16_t aL1KOffset) const
    {
        // format(M, K), K_axis is k direction, and M_axis is m direction in load3d intrin
        uint16_t madMAlign = CeilAlign(madM, HW_M0);
        uint16_t kStep = CeilAlign<uint16_t>(madK, c0Size_);
        uint16_t aL1KAlign = CeilAlign<uint16_t>(aL1K, c0Size_);
        LoadData3DParamsV2Pro loadData3DV2;
        loadData3DV2.channelSize = aL1KAlign;
        loadData3DV2.extConfig = ((uint64_t)aL1MOffset << M_POS_BIT) | ((uint64_t)aL1KOffset << K_POS_BIT) |
                                 ((uint64_t)madMAlign << M_STEP_BIT) | (uint64_t)kStep;
        LoadData<A_T>(l0A[0], l1A[0], loadData3DV2);
    }

private:
    static constexpr bool isFmatrixUpdate_ = !MatmulFeatureTrait<MM_CFG>::IsSupportFmatrixB();
    constexpr static int32_t c0Size_ = AuxGetC0Size<A_T>();
};

} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif
#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_LOAD_TO_L0A_LOADINSTR_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_LOAD_TO_L0A_LOADINSTR_H__
#endif // IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_LOADINSTR_H
