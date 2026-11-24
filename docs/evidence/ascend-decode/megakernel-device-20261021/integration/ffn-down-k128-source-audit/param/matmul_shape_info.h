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
 * \file matmul_shape_info.h
 * \brief matmul shape info manager
 */

#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message( \
    "impl/adv_api/detail/matmul/param/matmul_shape_info.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"adv_api/matmul/matmul.h\"\" and use public functions or variables defined in interface headers files.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_PARAM_MATMUL_SHAPE_INFO_H__
#endif

#ifndef IMPL_MATMUL_PARAM_MATMUL_SHAPE_INFO_H
#define IMPL_MATMUL_PARAM_MATMUL_SHAPE_INFO_H

#include "../utils/matmul_module.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, typename A_TYPE, const auto& MM_CFG>
class MatmulShapeInfoBase {
    using L0cT = typename GetMmDstType<typename A_TYPE::T>::Type;
    using SrcT = typename A_TYPE::T;
    MATMUL_USE_MODULE(MatmulShapeTiling);

public:
    __aicore__ inline void SetTransposeA(bool isTransposeA = false) { isTransposeA_ = isTransposeA; }

    __aicore__ inline void SetTransposeB(bool isTransposeB = false) { isTransposeB_ = isTransposeB; }

    __aicore__ inline void SetOrgM(int orgM) { M_ = orgM; }

    __aicore__ inline void SetOrgN(int orgN) { N_ = orgN; }

    __aicore__ inline void SetOrgKa(int orgKa) { Ka_ = orgKa; }

    __aicore__ inline void SetOrgKb(int orgKb) { Kb_ = orgKb; }

    __aicore__ inline void SetOrgKc(int orgKc) { Kc_ = orgKc; }

    __aicore__ inline void SetOrgShape(int orgM, int orgN, int orgKa, int orgKb, int orgKc)
    {
        M_ = orgM;
        N_ = orgN;
        Ka_ = orgKa;
        Kb_ = orgKb;
        Kc_ = orgKc;
    }

    __aicore__ inline void CheckOrgShape(int orgM, int orgN, int orgKa, int orgKb, int orgKc)
    {
        ASCENDC_ASSERT((orgM > 0), { KERNEL_LOG(KERNEL_ERROR, "orgM is %d , which should be larger than 0", orgM); });
        ASCENDC_ASSERT((orgN > 0), { KERNEL_LOG(KERNEL_ERROR, "orgN is %d , which should be larger than 0", orgN); });
        ASCENDC_ASSERT(
            (orgKa > 0), { KERNEL_LOG(KERNEL_ERROR, "orgKa is %d , which should be larger than 0", orgKa); });
        ASCENDC_ASSERT(
            (orgKb > 0), { KERNEL_LOG(KERNEL_ERROR, "orgKb is %d , which should be larger than 0", orgKb); });
    }

    __aicore__ inline void CheckTailShape(int tailM, int tailN, int tailK)
    {
        ASCENDC_ASSERT(
            (tailM >= -1), { KERNEL_LOG(KERNEL_ERROR, "tailM is %d , which should be not less than -1", tailM); });
        ASCENDC_ASSERT(
            (tailN >= -1), { KERNEL_LOG(KERNEL_ERROR, "tailN is %d , which should be not less than -1", tailN); });
        ASCENDC_ASSERT(
            (tailK >= -1), { KERNEL_LOG(KERNEL_ERROR, "tailK is %d , which should be not less than -1", tailK); });
        const auto tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        if constexpr (DoMatmulIBShareNorm(MM_CFG)) {
            ASCENDC_ASSERT((tiling.GetSingleCoreM() >= tailM), {
                KERNEL_LOG(KERNEL_ERROR, "tailM is %d , which should be not more than singleCoreM_", tailM);
            });
            ASCENDC_ASSERT((tiling.GetSingleCoreN() >= tailN), {
                KERNEL_LOG(KERNEL_ERROR, "tailN is %d , which should be not more than singleCoreN_", tailN);
            });
            ASCENDC_ASSERT((tiling.GetSingleCoreK() >= tailK), {
                KERNEL_LOG(KERNEL_ERROR, "tailK is %d , which should be not more than singleCoreK_", tailK);
            });
        }
    }

    __aicore__ inline void CheckSpecificShape()
    {
        if constexpr (DoMatmulBasicBlock(MM_CFG) || DoMatmulSpecialBasicBlock(MM_CFG)) {
            if constexpr (A_TYPE::format != CubeFormat::VECTOR) {
                ASCENDC_ASSERT((GetSingleCoreM() % ToMatmulConfig(MM_CFG).basicM == 0), {
                    KERNEL_LOG(
                        KERNEL_ERROR,
                        "singleCoreM is %d, basicM is %d, singleCoreM should be a multiple of basicM in Basic Block "
                        "mode.",
                        GetSingleCoreM(), ToMatmulConfig(MM_CFG).basicM);
                });
            }
            ASCENDC_ASSERT((GetSingleCoreN() % ToMatmulConfig(MM_CFG).basicN == 0), {
                KERNEL_LOG(
                    KERNEL_ERROR,
                    "singleCoreN is %d, basicN is %d, singleCoreN should be a multiple of basicN in Basic Block mode.",
                    GetSingleCoreN(), ToMatmulConfig(MM_CFG).basicN);
            });
        }
    }

    __aicore__ inline void SetSingleCoreM(int singleCoreM) { singleCoreM_ = singleCoreM; }

    __aicore__ inline void SetSingleCoreN(int singleCoreN) { singleCoreN_ = singleCoreN; }

    __aicore__ inline void SetSingleCoreK(int singleCoreK) { singleCoreK_ = singleCoreK; }

    __aicore__ inline void SetSingleShape(int singleCoreM, int singleCoreN, int singleCoreK)
    {
        singleCoreM_ = singleCoreM;
        singleCoreN_ = singleCoreN;
        singleCoreK_ = singleCoreK;
    }

    __aicore__ inline void InitParams()
    {
        SetTransposeA(false);
        SetTransposeB(false);

        const auto& tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        SetOrgShape(tiling.GetM(), tiling.GetN(), tiling.GetKa(), tiling.GetKb(), tiling.GetN());
        if constexpr (
            DoMatmulSpecialBasicBlock(MM_CFG) && ToMatmulConfig(MM_CFG).singleCoreM != 0 &&
            ToMatmulConfig(MM_CFG).singleCoreN != 0 && ToMatmulConfig(MM_CFG).singleCoreK != 0) {
            SetSingleShape(
                ToMatmulConfig(MM_CFG).singleCoreM, ToMatmulConfig(MM_CFG).singleCoreN,
                ToMatmulConfig(MM_CFG).singleCoreK);
        } else {
            SetSingleShape(tiling.GetSingleCoreM(), tiling.GetSingleCoreN(), tiling.GetSingleCoreK());
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline bool IsTransposeA() const
    {
        return isTransposeA_;
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline bool IsTransposeB() const
    {
        return isTransposeB_;
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetOrgM()
    {
        return M_;
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetOrgN()
    {
        return N_;
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetOrgKa()
    {
        return Ka_;
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetOrgKb()
    {
        return Kb_;
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetOrgKc()
    {
        return Kc_;
    }

    template <bool IS_INTRA_BLOCK = false, bool IS_BASIC = false>
    __aicore__ inline int32_t GetSingleCoreM() const
    {
        if constexpr (IS_BASIC) {
            return ToMatmulConfig(MM_CFG).singleCoreM;
        } else {
            return singleCoreM_;
        }
    }

    template <bool IS_INTRA_BLOCK = false, bool IS_BASIC = false>
    __aicore__ inline int32_t GetSingleCoreN() const
    {
        if constexpr (IS_BASIC) {
            return ToMatmulConfig(MM_CFG).singleCoreN;
        } else {
            return singleCoreN_;
        }
    }

    template <bool IS_INTRA_BLOCK = false, bool IS_BASIC = false>
    __aicore__ inline int32_t GetSingleCoreK() const
    {
        if constexpr (IS_BASIC) {
            return ToMatmulConfig(MM_CFG).singleCoreK;
        } else {
            return singleCoreK_;
        }
    }

protected:
    bool isTransposeA_{false}; // whether A matrix need to transpose
    bool isTransposeB_{false}; // whether B matrix need to transpose
    int M_;
    int N_;
    int Ka_;
    int Kb_;
    int Kc_;
    int singleCoreM_;
    int singleCoreN_;
    int singleCoreK_;
};

template <typename IMPL, typename A_TYPE, const auto& MM_CFG, typename = void>
class MatmulShapeInfo : public MatmulShapeInfoBase<IMPL, A_TYPE, MM_CFG> {
    using L0cT = typename GetMmDstType<typename A_TYPE::T>::Type;
    using SrcT = typename A_TYPE::T;
    MATMUL_USE_MODULE(MatmulShapeTiling);

public:
    using BASE_MODULE = MatmulShapeInfoBase<IMPL, A_TYPE, MM_CFG>;
};

template <typename IMPL, typename A_TYPE, const auto& MM_CFG>
class MatmulShapeInfo<IMPL, A_TYPE, MM_CFG, enable_if_t<IsIntrablock<MM_CFG>>>
    : public MatmulShapeInfoBase<IMPL, A_TYPE, MM_CFG> {
    using L0cT = typename GetMmDstType<typename A_TYPE::T>::Type;
    using SrcT = typename A_TYPE::T;
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulSubBlockInfo);

public:
    using BASE_MODULE = MatmulShapeInfoBase<IMPL, A_TYPE, MM_CFG>;

    __aicore__ inline MatmulShapeInfo() = default;
    __aicore__ inline ~MatmulShapeInfo() = default;
    __aicore__ inline void SetTransposeA(bool isTransposeA = false)
    {
        if (MATMUL_MODULE(MatmulSubBlockInfo)->GetSubBlockIdx() == 0) {
            BASE_MODULE::isTransposeA_ = isTransposeA;
        } else {
            intraBlock.isTransposeA = isTransposeA;
        }
    }

    __aicore__ inline void SetTransposeB(bool isTransposeB = false)
    {
        if (MATMUL_MODULE(MatmulSubBlockInfo)->GetSubBlockIdx() == 0) {
            BASE_MODULE::isTransposeB_ = isTransposeB;
        } else {
            intraBlock.isTransposeB = isTransposeB;
        }
    }

    __aicore__ inline void SetOrgShape(int orgM, int orgN, int orgKa, int orgKb, int orgKc)
    {
        if (MATMUL_MODULE(MatmulSubBlockInfo)->GetSubBlockIdx() == 0) {
            BASE_MODULE::M_ = orgM;
            BASE_MODULE::N_ = orgN;
            BASE_MODULE::Ka_ = orgKa;
            BASE_MODULE::Kb_ = orgKb;
            BASE_MODULE::Kc_ = orgKc;
        } else {
            intraBlock.M = orgM;
            intraBlock.N = orgN;
            intraBlock.Ka = orgKa;
            intraBlock.Kb = orgKb;
            intraBlock.Kc = orgKc;
        }
    }

    __aicore__ inline void SetSingleShape(int singleCoreM, int singleCoreN, int singleCoreK)
    {
        if (MATMUL_MODULE(MatmulSubBlockInfo)->GetSubBlockIdx() == 0) {
            BASE_MODULE::singleCoreM_ = singleCoreM;
            BASE_MODULE::singleCoreN_ = singleCoreN;
            BASE_MODULE::singleCoreK_ = singleCoreK;
        } else {
            intraBlock.singleCoreM = singleCoreM;
            intraBlock.singleCoreN = singleCoreN;
            intraBlock.singleCoreK = singleCoreK;
        }
    }

    __aicore__ inline void InitParams()
    {
        SetTransposeA(false);
        SetTransposeB(false);

        const auto& tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        SetOrgShape(tiling.GetM(), tiling.GetN(), tiling.GetKa(), tiling.GetKb(), tiling.GetN());
        SetSingleShape(tiling.GetSingleCoreM(), tiling.GetSingleCoreN(), tiling.GetSingleCoreK());

        MATMUL_MODULE(MatmulShapeTiling)->template CheckTiling<SrcT, L0cT>();
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline bool IsTransposeA() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return intraBlock.isTransposeA;
        } else {
            return BASE_MODULE::isTransposeA_;
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline bool IsTransposeB() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return intraBlock.isTransposeB;
        } else {
            return BASE_MODULE::isTransposeB_;
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetOrgM() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return intraBlock.M;
        } else {
            return BASE_MODULE::M_;
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetOrgN() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return intraBlock.N;
        } else {
            return BASE_MODULE::N_;
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetOrgKa() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return intraBlock.Ka;
        } else {
            return BASE_MODULE::Ka_;
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetOrgKb() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return intraBlock.Kb;
        } else {
            return BASE_MODULE::Kb_;
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetOrgKc() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return intraBlock.Kc;
        } else {
            return BASE_MODULE::Kc_;
        }
    }

    template <bool IS_INTRA_BLOCK = false, bool IS_BASIC = false>
    __aicore__ inline int32_t GetSingleCoreM() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return intraBlock.singleCoreM;
        } else {
            return BASE_MODULE::singleCoreM_;
        }
    }

    template <bool IS_INTRA_BLOCK = false, bool IS_BASIC = false>
    __aicore__ inline int32_t GetSingleCoreN() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return intraBlock.singleCoreN;
        } else {
            return BASE_MODULE::singleCoreN_;
        }
    }

    template <bool IS_INTRA_BLOCK = false, bool IS_BASIC = false>
    __aicore__ inline int32_t GetSingleCoreK() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return intraBlock.singleCoreK;
        } else {
            return BASE_MODULE::singleCoreK_;
        }
    }

private:
    struct IntraBlock {
        __aicore__ inline IntraBlock(){};
        int M;
        int N;
        int Ka;
        int Kb;
        int Kc;
        int singleCoreM;
        int singleCoreN;
        int singleCoreK;
        bool isTransposeA = false;
        bool isTransposeB = false;
    };

    IntraBlock intraBlock;
};

template <typename IMPL, typename A_TYPE, const auto& MM_CFG>
class MatmulShapeInfo<IMPL, A_TYPE, MM_CFG, enable_if_t<HasScalePosition<A_TYPE>::value>>
    : public MatmulShapeInfoBase<IMPL, A_TYPE, MM_CFG> {
    using L0cT = typename GetMmDstType<typename A_TYPE::T>::Type;
    using SrcT = typename A_TYPE::T;
    MATMUL_USE_MODULE(MatmulShapeTiling);

public:
    using BASE_MODULE = MatmulShapeInfoBase<IMPL, A_TYPE, MM_CFG>;

    __aicore__ inline MatmulShapeInfo() = default;
    __aicore__ inline ~MatmulShapeInfo() = default;

    __aicore__ inline void SetTransposeScaleA(bool isTransposeScaleA = false)
    {
        isTransposeScaleA_ = isTransposeScaleA;
    }

    __aicore__ inline void SetTransposeScaleB(bool isTransposeScaleB = false)
    {
        isTransposeScaleB_ = isTransposeScaleB;
    }

    __aicore__ inline void InitParams()
    {
        BASE_MODULE::isTransposeA_ = false;
        BASE_MODULE::isTransposeB_ = false;
        SetTransposeScaleA(false);
        SetTransposeScaleB(true);

        const auto& tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        BASE_MODULE::SetOrgShape(tiling.GetM(), tiling.GetN(), tiling.GetKa(), tiling.GetKb(), tiling.GetN());
        BASE_MODULE::SetSingleShape(tiling.GetSingleCoreM(), tiling.GetSingleCoreN(), tiling.GetSingleCoreK());

        MATMUL_MODULE(MatmulShapeTiling)->template CheckTiling<SrcT, L0cT>();
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline bool IsTransposeScaleA() const
    {
        return isTransposeScaleA_;
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline bool IsTransposeScaleB() const
    {
        return isTransposeScaleB_;
    }

private:
    bool isTransposeScaleA_{false};
    bool isTransposeScaleB_{true};
};
} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif // IMPL_MATMUL_PARAM_MATMUL_SHAPE_INFO_H

#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_PARAM_MATMUL_SHAPE_INFO_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_PARAM_MATMUL_SHAPE_INFO_H__
#endif
