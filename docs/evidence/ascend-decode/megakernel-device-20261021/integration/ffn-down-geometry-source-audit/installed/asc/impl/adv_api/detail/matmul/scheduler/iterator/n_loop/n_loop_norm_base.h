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
 * \file n_loop_norm_base.h
 * \brief
 */

#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message( \
    "impl/adv_api/detail/matmul/scheduler/iterator/n_loop/n_loop_norm_base.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"adv_api/matmul/matmul.h\"\" and use public functions or variables defined in interface headers files.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_SCHEDULER_ITERATOR_N_LOOP_N_LOOP_NORM_BASE_H__
#endif

#ifndef IMPL_MATMUL_SCHEDULER_ITERATOR_N_LOOP_N_LOOP_NORM_BASE_H
#define IMPL_MATMUL_SCHEDULER_ITERATOR_N_LOOP_N_LOOP_NORM_BASE_H

#include "n_loop_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    NLoopNormBase is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    NLoopNormBase is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class A_TYPE, const auto& MM_CFG>
class NLoopNormBase {
    MATMUL_USE_MODULE(MatmulShapeTiling);

public:
    __aicore__ inline NLoopNormBase() = default;
    __aicore__ inline ~NLoopNormBase() = default;

    __aicore__ inline void Init(int32_t singleShape) { SetSingleShape(singleShape); }

    __aicore__ inline void SetSingleShape(int32_t singleShape)
    {
        if constexpr (ToMatmulConfig(MM_CFG).singleCoreN != 0 && !ToMatmulConfig(MM_CFG).enableSetTail) {
            SetSingleShapeFromCFG();
        } else {
            SetSingleShapeFromTiling(singleShape);
        }
        ASCENDC_ASSERT((totalIter_ > 0), {
            KERNEL_LOG(
                KERNEL_ERROR, "invalid singleCoreN, totalIter_ is %d , which should be larger than 0", totalIter_);
        });
    }

    __aicore__ inline uint32_t GetTotalIter() const { return totalIter_; }

    __aicore__ inline bool OuterNext()
    {
        if constexpr (IsBasicN(MM_CFG)) {
            return false;
        } else {
            innerStartIdx_ += innerIter_;
            UpdateOuterParams();
            return !OuterEnd();
        }
    }

    __aicore__ inline void OuterStart()
    {
        innerStartIdx_ = 0;
        UpdateOuterParams();
    }

    __aicore__ inline bool OuterEnd()
    {
        if constexpr (IsBasicN(MM_CFG)) {
            return true;
        } else {
            return innerStartIdx_ >= totalIter_;
        }
    }

    __aicore__ inline uint32_t GetOuterIdx() const
    {
        return Ceil(innerStartIdx_, MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepN());
    }

    __aicore__ inline uint32_t GetOuterIter() const
    {
        return Ceil(totalIter_, MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepN());
    }

    __aicore__ inline int32_t GetTileShape() const { return baseShape_; }

    __aicore__ inline int32_t GetTileBlockShape() const { return baseBlockShape_; }

    __aicore__ inline int32_t GetTailShape() const { return tailBaseShape_; }

    __aicore__ inline bool InnerNext()
    {
        if constexpr (IsBasicN(MM_CFG)) {
            return false;
        } else {
            ++innerIndex_;
            UpdateInnerParams();
            return !InnerEnd();
        }
    }

    __aicore__ inline void InnerStart()
    {
        innerIndex_ = innerStartIdx_;
        UpdateInnerParams();
    }

    __aicore__ inline bool InnerEnd() { return innerIndex_ >= innerStartIdx_ + innerIter_; }

    __aicore__ inline uint32_t GetInnerIdx() const { return innerIndex_; }

    __aicore__ inline uint32_t GetInnerIter() const { return innerIter_; }

    __aicore__ inline int32_t GetBaseShape() const { return baseShape_; }

    __aicore__ inline int32_t GetBaseBlockShape() const { return baseBlockShape_; }

protected:
    __aicore__ inline void SetSingleShapeFromCFG()
    {
        totalIter_ = GetNIter(MM_CFG);
        tailBaseShape_ = ToMatmulConfig(MM_CFG).singleCoreN % ToMatmulConfig(MM_CFG).basicN;
        if (tailBaseShape_ == 0) {
            tailBaseShape_ = ToMatmulConfig(MM_CFG).basicN;
        }
    }

    __aicore__ inline void SetSingleShapeFromTiling(int32_t singleShape)
    {
        totalIter_ = Ceil(
            static_cast<uint32_t>(singleShape),
            static_cast<uint32_t>(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN()));
        tailBaseShape_ = singleShape % MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
        if (tailBaseShape_ == 0) {
            tailBaseShape_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
        }
    }

    __aicore__ inline void UpdateOuterParams()
    {
        if constexpr (IsBasicN(MM_CFG)) {
            innerIter_ = 1;
        } else {
            innerIter_ = (totalIter_ - innerStartIdx_) >
                                 static_cast<uint32_t>(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepN()) ?
                             MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepN() :
                             (totalIter_ - innerStartIdx_);
        }
    }

    __aicore__ inline void UpdateInnerParams()
    {
        if constexpr (NoTailN(MM_CFG)) {
            baseShape_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
        } else if constexpr (IsBasicN(MM_CFG)) {
            baseShape_ = tailBaseShape_;
        } else {
            baseShape_ = (innerIndex_ + 1 == totalIter_) ? tailBaseShape_ :
                                                           MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
        }
        baseBlockShape_ = Ceil(baseShape_, BLOCK_CUBE);
    }

    uint32_t totalIter_;
    // InnerLoop
    uint32_t innerIndex_ = 0;
    uint32_t innerIter_;
    uint32_t innerStartIdx_ = 0;
    // Shape
    int32_t baseShape_;
    int32_t baseBlockShape_;
    int32_t tailBaseShape_;
};

} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif // _N_LOOP_NORM_BASE_H

#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_SCHEDULER_ITERATOR_N_LOOP_N_LOOP_NORM_BASE_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_SCHEDULER_ITERATOR_N_LOOP_N_LOOP_NORM_BASE_H__
#endif
