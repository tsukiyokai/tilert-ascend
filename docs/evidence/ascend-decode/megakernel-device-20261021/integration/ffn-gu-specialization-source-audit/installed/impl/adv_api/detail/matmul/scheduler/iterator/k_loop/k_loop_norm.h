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
 * \file k_loop_norm.h
 * \brief
 */
#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message( \
    "impl/adv_api/detail/matmul/scheduler/iterator/k_loop/k_loop_norm.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"adv_api/matmul/matmul.h\"\" and use public functions or variables defined in interface headers files.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_SCHEDULER_ITERATOR_K_LOOP_K_LOOP_NORM_H__
#endif

#ifndef IMPL_MATMUL_SCHEDULER_ITERATOR_K_LOOP_K_LOOP_NORM_H
#define IMPL_MATMUL_SCHEDULER_ITERATOR_K_LOOP_K_LOOP_NORM_H

#include "k_loop_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    KLoop is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    KLoop is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, typename TRANS_T, class A_TYPE, const auto& MM_CFG>
class KLoop<
    IMPL, TRANS_T, A_TYPE, MM_CFG,
    enable_if_t<
        isNormEnableScheduler<A_TYPE, MM_CFG> || IsBmmEnableScheduler<A_TYPE, MM_CFG> || IsBasicBlockEnable<MM_CFG> ||
        DoMatmulIBShareNorm(MM_CFG)>> {
    MATMUL_USE_MODULE(MatmulShapeTiling);

public:
    __aicore__ inline KLoop() = default;
    __aicore__ inline ~KLoop() = default;

    __aicore__ inline void Init(int32_t singleShape) { SetSingleShape(singleShape); }

    __aicore__ inline void SetSingleShape(int32_t singleShape)
    {
        if constexpr (IsBasicK(MM_CFG)) {
            kIter_ = 1;
        } else {
            kIter_ = Ceil(
                static_cast<uint32_t>(singleShape),
                static_cast<uint32_t>(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK()));
        }
        ASCENDC_ASSERT(
            (kIter_ > 0), { KERNEL_LOG(KERNEL_ERROR, "kIter_ is %d , which should be larger than 0", kIter_); });
        if constexpr (DoMatmulIBShareNorm(MM_CFG)) {
            if constexpr (A_TYPE::ibShare) {
                if (MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetDepthA1() <
                    kIter_ * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepM()) {
                    // k not full load && tiling_.GetDepthA1() == 1
                    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "Unsupported k not full load."); });
                }
            } else {
                if (MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetDepthB1() <
                    kIter_ * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepN()) {
                    // k not full load && tiling_.GetDepthB1() == 1
                    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "Unsupported k not full load."); });
                }
            }
        }

        if constexpr (NoTailK(MM_CFG)) {
            tailK_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
        } else {
            tailK_ = singleShape % MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
            if (tailK_ == 0) {
                tailK_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
            }
        }
        if constexpr (IsBasic(MM_CFG)) {
            if constexpr (ToMatmulConfig(MM_CFG).enableSetTail) {
                tileShape_ = tailK_;
            } else {
                tileShape_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
            }
            tileBlockShape_ = Ceil(tileShape_, c0Size_);
        }
    }

    __aicore__ inline void OuterStart()
    {
        outerIdx_ = 0;
        UpdateOuterParams();
    }

    __aicore__ inline bool OuterNext()
    {
        outerIdx_++;
        UpdateOuterParams();
        return !OuterEnd();
    }

    __aicore__ inline bool OuterEnd() { return outerIdx_ == kIter_; }

    __aicore__ inline bool FirstOuterIter() const { return outerIdx_ == 0; }

    __aicore__ inline bool LastOuterIter() const { return outerIdx_ + 1 == kIter_; }

    __aicore__ inline void InnerStart(){};

    __aicore__ inline bool InnerNext() { return false; }

    __aicore__ inline bool InnerEnd() { return false; }

    __aicore__ inline bool FirstInnerIter() const { return outerIdx_ == 0; }

    __aicore__ inline uint32_t GetTotalIter() const { return kIter_; }

    __aicore__ inline uint32_t GetOuterIter() const { return kIter_; }

    __aicore__ inline uint32_t GetInnerIter() const { return 1; }

    __aicore__ inline uint32_t GetOuterIdx() const { return outerIdx_; }

    __aicore__ inline uint32_t GetInnerStartIdx() const { return outerIdx_; }

    __aicore__ inline uint32_t GetInnerIdx() const { return outerIdx_; }

    __aicore__ inline int32_t GetTileShapeA() const { return tileShape_; }

    __aicore__ inline int32_t GetTileShapeB() const { return tileShape_; }

    __aicore__ inline int32_t GetTileBlockShapeA() const { return tileBlockShape_; }

    __aicore__ inline int32_t GetTileBlockShapeB() const { return tileBlockShape_; }

    __aicore__ inline int32_t GetBaseShape() const { return tileShape_; }

    __aicore__ inline int32_t GetBaseBlockShape() const { return tileBlockShape_; }

private:
    __aicore__ inline void UpdateOuterParams()
    {
        if constexpr (NoTailK(MM_CFG)) {
            if constexpr (ToMatmulConfig(MM_CFG).enableSetTail) {
                tileShape_ =
                    (outerIdx_ == kIter_ - 1) ? tailK_ : MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
            } else {
                tileShape_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
            }
        } else {
            tileShape_ = (outerIdx_ == kIter_ - 1) ? tailK_ : MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
        }
        tileBlockShape_ = Ceil(tileShape_, c0Size_);
    }

private:
    int32_t tailK_;
    int32_t tileShape_;      // kL1 length, in normal version, kaL1 = kbL1
    int32_t tileBlockShape_; // kL1 block num
    uint32_t kIter_;         // total iteration counts
    uint32_t outerIdx_{0};   // current outer loop index

    constexpr static int32_t c0Size_ = AuxGetC0Size<TRANS_T>();
};

} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif // _K_LOOP_NORM_H_

#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_SCHEDULER_ITERATOR_K_LOOP_K_LOOP_NORM_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_SCHEDULER_ITERATOR_K_LOOP_K_LOOP_NORM_H__
#endif
