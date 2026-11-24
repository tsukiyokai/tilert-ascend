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
 * \file load_to_l0a_load2d.h
 * \brief
 */

#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message("impl/adv_api/detail/matmul/stage/split/load_to_l0a/load_to_l0a_load2d.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"adv_api/matmul/matmul.h\"\" and use public functions or variables defined in interface headers files.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_LOAD_TO_L0A_LOAD2D_H__
#endif

#ifndef IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_LOAD2D_H
#define IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_LOAD2D_H

#include "load_to_l0a_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, typename A_TYPE, const auto& MM_CFG>
class LoadToL0A<IMPL, A_TYPE, MM_CFG,
    enable_if_t<(GetGemvMode<A_TYPE>() == GemvMode::MATRIX) &&
                !(DoMatmulBasicBlock(MM_CFG) || DoMatmulSpecialBasicBlock(MM_CFG)) &&
                (GetLoadInstrType<typename A_TYPE::T, MM_CFG>() == LoadInstrType::LOAD2D)>>
{
    using A_T = typename A_TYPE::T;
    using AuxDtype = decltype(GetAuxDataType<A_TYPE>());
public:
    __aicore__ inline LoadToL0A() {};
    __aicore__ inline ~LoadToL0A() {};

    __aicore__ inline void SetScalar(A_T scalar) {};

    __aicore__ inline void Prepare(bool isATranspose, uint16_t aL1K, uint16_t aL1M) const {};

    __aicore__ inline void Load(const LocalTensor<A_T> &dst, const LocalTensor<A_T> &aMatrix,
     uint16_t aL1M, uint16_t aL1K, uint16_t madM, uint16_t madK, uint16_t aL1MOffset, uint16_t aL1KOffset,
     bool isATranspose, const LocalTensor<AuxDtype> &l1AAuxMatrix = {}, uint16_t aAuxL1K = 0,
     uint16_t aAuxL1KOffset = 0) const
    {
        uint16_t blockUseM = Ceil(madM, BLOCK_CUBE);
        uint16_t blockUseK = Ceil(madK, c0Size_);
        int srcL1Offset = 0;
        if constexpr(PhyPosIsL1(A_TYPE::pos)) {
            if (isATranspose) {
                srcL1Offset = aL1K * aL1MOffset + c0Size_ * aL1KOffset;
            } else {
                srcL1Offset = c0Size_ * aL1MOffset + aL1M * aL1KOffset;
            }
        }
        if (isATranspose) {
            TransLoadDataToL0(dst, aMatrix, aL1K, blockUseM, blockUseK, srcL1Offset);
        } else {
            LoadDataToL0(dst, aMatrix, aL1M, isATranspose, blockUseM, blockUseK, srcL1Offset);
        }
    }
private:
    constexpr static int32_t factor_ = AuxGetFactor<A_T>();
    constexpr static int32_t c0Size_ = AuxGetC0Size<A_T>();

     __aicore__ inline void TransLoadDataToL0(const LocalTensor<A_T> &dst, const LocalTensor<A_T> &aMatrix,
      uint16_t aL1K, uint16_t blockUseM, uint16_t blockUseK, int srcL1Offset) const
     {
        // startIndex, repeatTimes, srcStride, sid, dstGap, ifTranspose, addrmode
        LoadData2dParams loadDataParams{0, static_cast<uint8_t>(blockUseK), 1, 0, 0, true, 0};
        int dstOffset = blockUseK * CUBE_MAX_SIZE / factor_; // madk
        int srcOffset = aL1K * c0Size_; // aL1K
        if constexpr (!PhyPosIsL1(A_TYPE::pos)) {
            srcOffset = blockUseK * c0Size_ * BLOCK_CUBE;
        }

        if (blockUseK == 1) {
            loadDataParams.repeatTimes = blockUseM;
            LoadData(dst, aMatrix[srcL1Offset], loadDataParams);
        } else {
            for (int i = 0; i < blockUseM; i++) {
                LoadData(dst[i * dstOffset], aMatrix[srcL1Offset + i * srcOffset], loadDataParams);
            }
        }
     }

     __aicore__ inline void LoadDataToL0(const LocalTensor<A_T> &dst, const LocalTensor<A_T> &aMatrix,
     uint16_t aL1M, bool isATranspose, uint16_t blockUseM, uint16_t blockUseK, int srcL1Offset) const
     {
        LoadData2dParams loadDataParams;
        int dstOffset = blockUseK * CUBE_MAX_SIZE / factor_;
        int srcOffset = CUBE_MAX_SIZE / factor_;
        loadDataParams.repeatTimes = blockUseK;
        if constexpr (PhyPosIsL1(A_TYPE::pos)) {
            // all A matrix is in L1 buffer
            loadDataParams.srcStride = Ceil(aL1M, BLOCK_CUBE);
        } else {
            loadDataParams.srcStride = blockUseM;
        }
        loadDataParams.ifTranspose = false;

        if (blockUseK == 1) {
            loadDataParams.repeatTimes = blockUseM;
            loadDataParams.srcStride = 1;
            LoadData(dst, aMatrix[srcL1Offset], loadDataParams);
        } else {
            for (int i = 0; i < blockUseM; i++) {
                LoadData(dst[i * dstOffset], aMatrix[srcL1Offset + i * srcOffset], loadDataParams);
            }
        }
     }
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif
#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_LOAD_TO_L0A_LOAD2D_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_LOAD_TO_L0A_LOAD2D_H__
#endif // IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_LOAD2D_H
