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
 * \file constant_tiling.h
 * \brief
 */
#ifndef LIB_MATMUL_CONSTANT_TILING_H
#define LIB_MATMUL_CONSTANT_TILING_H

#include "../../../impl/adv_api/tiling/matmul/matmul_constant_tiling_impl.h"

namespace AscendC {
/**
 * @brief Retrieves constantized Matmul Tiling parameters during compilation
 *
 * This interface is used to obtain constantized Matmul Tiling parameters at compile time,
 * which can be used for matrix multiplication operations with fixed configurations.
 *
 * @tparam A_TYPE Type information of matrix A, defined through MatmulType
 * @tparam B_TYPE Type information of matrix B, defined through MatmulType
 * @tparam C_TYPE Type information of matrix C, defined through MatmulType
 * @tparam BIAS_TYPE Type information of BIAS matrix, defined through MatmulType
 *
 * @param[in] mmCFG Input MatmulConfig template.
 * @param[in] l1Size Available L1 size, default value is L1_SIZE
 *
 * @return MatmulApiStaticTiling Constantized Matmul Tiling parameters obtained
 */
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ constexpr MatmulApiStaticTiling GetMatmulApiTiling(const MatmulConfig& mmCFG, int32_t l1Size = Impl::L1_SIZE)
{
    MatmulApiStaticTiling tiling;
    tiling.cfg = mmCFG;
    if ((mmCFG.singleCoreM == 0) || (mmCFG.singleCoreN == 0) || (mmCFG.singleCoreK == 0)) {
        if (mmCFG.basicM != 0 && mmCFG.basicN != 0 && mmCFG.basicK != 0) {
            tiling.baseM = mmCFG.basicM;
            tiling.baseN = mmCFG.basicN;
            tiling.baseK = mmCFG.basicK;
            tiling.dbL0A = GetL0ADb<A_TYPE>(mmCFG, TOTAL_L0A_SIZE);
            tiling.dbL0B = GetL0BDb<B_TYPE>(mmCFG, TOTAL_L0B_SIZE);
            tiling.isBias = mmCFG.enableSetBias;
        }
        return tiling;
    }
    L1Status l1Factor = GetL1Factor<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmCFG, l1Size);
    // when enable constant tiling, user need update orgShape
    tiling.M = mmCFG.singleCoreM;
    tiling.N = mmCFG.singleCoreN;
    tiling.Ka = mmCFG.singleCoreK;
    tiling.Kb = mmCFG.singleCoreK;
    tiling.singleCoreM = mmCFG.singleCoreM;
    tiling.singleCoreN = mmCFG.singleCoreN;
    tiling.singleCoreK = mmCFG.singleCoreK;
    tiling.baseM = mmCFG.basicM;
    tiling.baseN = mmCFG.basicN;
    tiling.baseK = mmCFG.basicK;
    tiling.isBias = mmCFG.enableSetBias;
    tiling.stepM = l1Factor.mAL1;
    tiling.stepN = l1Factor.nBL1;
    int32_t reduceC0Size = GetReduceC0Size<typename A_TYPE::T>();
    if (!CalcAL1FullLoadTiling<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(l1Size, tiling)) {
        int32_t kL0 = GetKL0<A_TYPE>(mmCFG);
        tiling.stepKa = CeilNoLog<int32_t>(l1Factor.kAL1, kL0);
        tiling.stepKb = CeilNoLog<int32_t>(l1Factor.kBL1, kL0);
        tiling.depthA1 = CeilNoLog<int32_t>(l1Factor.kAL1, kL0) * l1Factor.mAL1 * l1Factor.dbAL1;
        tiling.depthB1 = CeilNoLog<int32_t>(l1Factor.kBL1, kL0) * l1Factor.nBL1 * l1Factor.dbBL1;
    }
    tiling.iterateOrder = GetIterateOrder<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(l1Factor, mmCFG);
    tiling.dbL0A = GetL0ADb<A_TYPE>(mmCFG, TOTAL_L0A_SIZE);
    tiling.dbL0B = GetL0BDb<B_TYPE>(mmCFG, TOTAL_L0B_SIZE);
    GetMxMatmulApiTiling<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tiling, l1Size);
    // keep the same with runtime tiling, fix l0c db
    tiling.dbL0C = 1;
    tiling.transLength = GetTransLength<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmCFG, l1Factor);
    tiling.shareMode = 0;
    tiling.shareL1Size =
        GetL1UsedSize<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmCFG, l1Factor, tiling.depthA1, tiling.depthB1);
    tiling.shareL0CSize = mmCFG.basicM * mmCFG.basicN * GetBitSize<float>() / ONE_BYTE_BIT_SIZE;
    // tiling constant not support v200
    tiling.shareUbSize = 0;
    return tiling;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class SingleShape, class L1Shape, class BaseShape>
__aicore__ constexpr MatmulApiStaticTiling GetMatmulApiTiling(const MatmulConfig& mmCFG, int32_t l1Size = Impl::L1_SIZE)
{
    constexpr auto singleM = Std::tuple_element<0, SingleShape>::type::value;
    constexpr auto singleN = Std::tuple_element<1, SingleShape>::type::value;
    constexpr auto singleKa = Std::tuple_element<2, SingleShape>::type::value;
    constexpr auto singleKb = []() {
        if constexpr (Std::tuple_size_v<SingleShape> > 3) {
            return Std::tuple_element<3, SingleShape>::type::value;
        } else {
            return singleKa;
        }
    }();
    constexpr auto l1M = Std::tuple_element<0, L1Shape>::type::value;
    constexpr auto l1N = Std::tuple_element<1, L1Shape>::type::value;
    constexpr auto l1Ka = Std::tuple_element<2, L1Shape>::type::value;
    constexpr auto l1Kb = []() {
        if constexpr (Std::tuple_size_v<L1Shape> > 3) {
            return Std::tuple_element<3, L1Shape>::type::value;
        } else {
            return l1Ka;
        }
    }();
    constexpr auto baseM = Std::tuple_element<0, BaseShape>::type::value;
    constexpr auto baseN = Std::tuple_element<1, BaseShape>::type::value;
    constexpr auto baseK = Std::tuple_element<2, BaseShape>::type::value;

    if constexpr (l1M == 0 || l1N == 0 || l1Ka == 0 || l1Kb == 0) {
        return GetMatmulApiTiling<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmCFG, l1Size);
    }
    MatmulApiStaticTiling tiling;
    tiling.cfg = mmCFG;
    tiling.baseM = baseM;
    tiling.baseN = baseN;
    tiling.baseK = baseK;
    tiling.dbL0A = 2 * baseM * baseK * GetBitSize<typename A_TYPE::T>() / ONE_BYTE_BIT_SIZE <= TOTAL_L0A_SIZE ?
                       Impl::DB_ON :
                       Impl::DB_OFF;
    tiling.dbL0B = 2 * baseK * baseN * GetBitSize<typename B_TYPE::T>() / ONE_BYTE_BIT_SIZE <= TOTAL_L0B_SIZE ?
                       Impl::DB_ON :
                       Impl::DB_OFF;
    tiling.isBias = mmCFG.enableSetBias;
    tiling.M = singleM;
    tiling.N = singleN;
    tiling.Ka = singleKa;
    tiling.Kb = singleKb;
    tiling.singleCoreM = singleM;
    tiling.singleCoreN = singleN;
    tiling.singleCoreK = singleKa;
    tiling.stepM = CeilNoLog<int32_t>(l1M, baseM);
    tiling.stepN = CeilNoLog<int32_t>(l1N, baseN);
    tiling.stepKa = CeilNoLog<int32_t>(l1Ka, baseK);
    tiling.stepKb = CeilNoLog<int32_t>(l1Kb, baseK);
    tiling.depthA1 = tiling.stepM * tiling.stepKa * 2; // 2 DoubleBuffer
    tiling.depthB1 = tiling.stepN * tiling.stepKb * 2; // 2 DoubleBuffer
    tiling.iterateOrder = 0;
    // keep the same with runtime tiling, fix l0c db
    tiling.dbL0C = 1;
    int32_t biasLength = 0;
    if (mmCFG.enableSetBias) {
        if constexpr (PhyPosIsL1(BIAS_TYPE::pos)) {
            biasLength = 0;
        } else {
            int32_t channelWiseSize = GetChannelWise<BIAS_TYPE>(mmCFG) * 1 * GetTypeSize<typename BIAS_TYPE::T>();
            biasLength = tiling.stepN * baseN * channelWiseSize;
        }
    }
    // C matrix ND2NZ
    int32_t c1Length = 0;
    if constexpr (C_TYPE::format == CubeFormat::ND || C_TYPE::pos == TPosition::GM) {
        c1Length = baseM * baseN * GetBitSize<typename C_TYPE::T>() / ONE_BYTE_BIT_SIZE;
    }
    tiling.transLength = MaxValue<int32_t>(c1Length, biasLength);
    tiling.shareMode = 0;
    tiling.shareL1Size = GetL1UsedSize<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmCFG, tiling.depthA1, tiling.depthB1);
    tiling.shareL0CSize = baseM * baseN * GetBitSize<float>() / ONE_BYTE_BIT_SIZE;
    tiling.shareUbSize = 0;
    return tiling;
}
} // namespace AscendC
#endif // LIB_MATMUL_CONSTANT_TILING_H
