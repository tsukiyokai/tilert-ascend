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
 * \file kernel_operator_vec_reduce_impl.h
 * \brief
 */
#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message( \
    "impl/basic_api/dav_c220/kernel_operator_vec_reduce_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"basic_api/kernel_tpipe.h\"\" and use public functions or variables defined in interface headers files.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_REDUCE_IMPL_H__
#endif
#ifndef ASCENDC_MODULE_OPERATOR_VEC_REDUCE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_REDUCE_IMPL_H

#include "../../../include/basic_api/kernel_tpipe.h"

namespace AscendC {
template <typename T>
__aicore__ inline void BlockReduceSumIntrinsicsImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const int32_t repeatTime, const int32_t dstRepStride,
    const int32_t srcBlkStride, const int32_t srcRepStride)
{
    vcgadd(dstLocal, srcLocal, repeatTime, dstRepStride, srcBlkStride, srcRepStride);
}

template <typename T>
__aicore__ inline void BlockReduceMaxIntrinsicsImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const int32_t repeatTime, const int32_t dstRepStride,
    const int32_t srcBlkStride, const int32_t srcRepStride)
{
    vcgmax(dstLocal, srcLocal, repeatTime, dstRepStride, srcBlkStride, srcRepStride);
}

template <typename T>
__aicore__ inline void BlockReduceMinIntrinsicsImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const int32_t repeatTime, const int32_t dstRepStride,
    const int32_t srcBlkStride, const int32_t srcRepStride)
{
    vcgmin(dstLocal, srcLocal, repeatTime, dstRepStride, srcBlkStride, srcRepStride);
}

template <typename T>
__aicore__ inline void PairReduceSumIntrinsicsImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const int32_t repeatTime, const int32_t dstRepStride,
    const int32_t srcBlkStride, const int32_t srcRepStride)
{
    vcpadd(dstLocal, srcLocal, repeatTime, dstRepStride, srcBlkStride, srcRepStride);
}

template <typename T>
__aicore__ inline void RepeatReduceSumIntrinsicsImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const int32_t repeatTime, const int32_t srcBlkStride,
    const int32_t dstRepStride, const int32_t srcRepStride)
{
    vcadd(dstLocal, srcLocal, repeatTime, dstRepStride, srcBlkStride, srcRepStride, 0);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceSumImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const int32_t repeatTime, const int32_t mask,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        BlockReduceSumIntrinsicsImpl(dstLocal, srcLocal, repeatTime, dstRepStride, srcBlkStride, srcRepStride);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMaxImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const int32_t repeatTime, const int32_t mask,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        BlockReduceMaxIntrinsicsImpl(dstLocal, srcLocal, repeatTime, dstRepStride, srcBlkStride, srcRepStride);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMinImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const int32_t repeatTime, const int32_t mask,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        BlockReduceMinIntrinsicsImpl(dstLocal, srcLocal, repeatTime, dstRepStride, srcBlkStride, srcRepStride);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void PairReduceSumImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const int32_t repeatTime, const int32_t mask,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        PairReduceSumIntrinsicsImpl(dstLocal, srcLocal, repeatTime, dstRepStride, srcBlkStride, srcRepStride);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceSumImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const int32_t repeatTime, const uint64_t mask[],
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        BlockReduceSumIntrinsicsImpl(dstLocal, srcLocal, repeatTime, dstRepStride, srcBlkStride, srcRepStride);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMaxImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const int32_t repeatTime, const uint64_t mask[],
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        BlockReduceMaxIntrinsicsImpl(dstLocal, srcLocal, repeatTime, dstRepStride, srcBlkStride, srcRepStride);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMinImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const int32_t repeatTime, const uint64_t mask[],
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        BlockReduceMinIntrinsicsImpl(dstLocal, srcLocal, repeatTime, dstRepStride, srcBlkStride, srcRepStride);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void PairReduceSumImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const int32_t repeatTime, const uint64_t mask[],
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        PairReduceSumIntrinsicsImpl(dstLocal, srcLocal, repeatTime, dstRepStride, srcBlkStride, srcRepStride);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void RepeatReduceSumImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const int32_t repeatTime, const int32_t elemsInOneRepeat,
    const int32_t dstBlkStride, const int32_t srcBlkStride, const int32_t dstRepStride, const int32_t srcRepStride)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(elemsInOneRepeat);
        RepeatReduceSumIntrinsicsImpl(dstLocal, srcLocal, repeatTime, srcBlkStride, dstRepStride, srcRepStride);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void ReduceSumImpl(__ubuf__ T* dst, __ubuf__ T* src, uint32_t count)
{
    if ASCEND_IS_AIV {
        if constexpr (isSetMask) {
            set_mask_count();
            set_vector_mask(0, count);
        }
        vcadd(dst, src, 1, DEFAULT_REDUCE_DST_REP_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, 1);
        auto eventIdVToS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        int64_t accVal = get_acc_val();
        *(dst) = *(reinterpret_cast<T*>(&accVal));
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        event_t eventIdSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
        if constexpr (isSetMask) {
            set_mask_norm();
            set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        }
    }
}

/* **************************************** Whole Reduce Interface ****************************************** */
template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMaxImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, struct ReduceRepeatParams& params, const ReduceOrder order)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(params.highMask, params.lowMask);
        if (order == ReduceOrder::ORDER_VALUE_INDEX) {
            vcmax(
                dstLocal, srcLocal, params.repeatTimes, params.dstRepStride, params.srcBlkStride, params.srcRepStride,
                Order_t::VALUE_INDEX);
        } else if (order == ReduceOrder::ORDER_INDEX_VALUE) {
            vcmax(
                dstLocal, srcLocal, params.repeatTimes, params.dstRepStride, params.srcBlkStride, params.srcRepStride,
                Order_t::INDEX_VALUE);
        } else if (order == ReduceOrder::ORDER_ONLY_VALUE) {
            vcmax(
                dstLocal, srcLocal, params.repeatTimes, params.dstRepStride, params.srcBlkStride, params.srcRepStride,
                Order_t::ONLY_VALUE);
        } else {
            vcmax(
                dstLocal, srcLocal, params.repeatTimes, params.dstRepStride, params.srcBlkStride, params.srcRepStride,
                Order_t::ONLY_INDEX);
        }
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMaxImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const uint64_t mask[], const int32_t repeatTime,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride, const ReduceOrder order)
{
    ReduceRepeatParams params(mask, repeatTime, dstRepStride, srcBlkStride, srcRepStride);
    WholeReduceMaxImpl<T, isSetMask>(dstLocal, srcLocal, params, order);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMaxImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const int32_t mask, const int32_t repeatTime,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride, const ReduceOrder order)
{
    ReduceRepeatParams params(mask, repeatTime, dstRepStride, srcBlkStride, srcRepStride);
    WholeReduceMaxImpl<T, isSetMask>(dstLocal, srcLocal, params, order);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMinImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, struct ReduceRepeatParams& params, const ReduceOrder order)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(params.highMask, params.lowMask);
        if (order == ReduceOrder::ORDER_VALUE_INDEX) {
            vcmin(
                dstLocal, srcLocal, params.repeatTimes, params.dstRepStride, params.srcBlkStride, params.srcRepStride,
                Order_t::VALUE_INDEX);
        } else if (order == ReduceOrder::ORDER_INDEX_VALUE) {
            vcmin(
                dstLocal, srcLocal, params.repeatTimes, params.dstRepStride, params.srcBlkStride, params.srcRepStride,
                Order_t::INDEX_VALUE);
        } else if (order == ReduceOrder::ORDER_ONLY_VALUE) {
            vcmin(
                dstLocal, srcLocal, params.repeatTimes, params.dstRepStride, params.srcBlkStride, params.srcRepStride,
                Order_t::ONLY_VALUE);
        } else {
            vcmin(
                dstLocal, srcLocal, params.repeatTimes, params.dstRepStride, params.srcBlkStride, params.srcRepStride,
                Order_t::ONLY_INDEX);
        }
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMinImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const uint64_t mask[], const int32_t repeatTime,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride, const ReduceOrder order)
{
    struct ReduceRepeatParams params(mask, repeatTime, dstRepStride, srcBlkStride, srcRepStride);
    WholeReduceMinImpl<T, isSetMask>(dstLocal, srcLocal, params, order);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMinImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const int32_t mask, const int32_t repeatTime,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride, const ReduceOrder order)
{
    struct ReduceRepeatParams params(mask, repeatTime, dstRepStride, srcBlkStride, srcRepStride);
    WholeReduceMinImpl<T, isSetMask>(dstLocal, srcLocal, params, order);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceSumImpl(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, struct ReduceRepeatParams& params)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(params.highMask, params.lowMask);
        vcadd(dstLocal, srcLocal, params.repeatTimes, params.dstRepStride, params.srcBlkStride, params.srcRepStride, 0);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceSumImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const uint64_t mask[], const int32_t repeatTime,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    struct ReduceRepeatParams params(mask, repeatTime, dstRepStride, srcBlkStride, srcRepStride);
    WholeReduceSumImpl<T, isSetMask>(dstLocal, srcLocal, params);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceSumImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const uint32_t mask, const int32_t repeatTime,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    struct ReduceRepeatParams params(mask, repeatTime, dstRepStride, srcBlkStride, srcRepStride);
    WholeReduceSumImpl<T, isSetMask>(dstLocal, srcLocal, params);
}

/* **************************************** Reduce Interface ****************************************** */
template <typename T>
__aicore__ inline void ReduceMaxIntrinsicsImpl(
    __ubuf__ T* sharedTmpBuffer, __ubuf__ T* srcLocal, const int32_t repeatTime, const int32_t srcRepStride)
{
    vcmax(sharedTmpBuffer, srcLocal, repeatTime, 1, 1, srcRepStride, Order_t::VALUE_INDEX);
}

template <typename T>
__aicore__ inline void ReduceMinIntrinsicsImpl(
    __ubuf__ T* sharedTmpBuffer, __ubuf__ T* srcLocal, const int32_t repeatTime, const int32_t srcRepStride)
{
    vcmin(sharedTmpBuffer, srcLocal, repeatTime, 1, 1, srcRepStride, Order_t::VALUE_INDEX);
}

template <typename T>
__aicore__ inline void ReduceSumIntrinsicsImpl(
    __ubuf__ T* sharedTmpBuffer, __ubuf__ T* srcLocal, const int32_t repeatTime, const int32_t srcRepStride)
{
    vcadd(sharedTmpBuffer, srcLocal, repeatTime, 1, 1, srcRepStride, 0);
}

template <typename T>
__aicore__ inline void ReduceSumSecondStep(
    __ubuf__ T* dstLocal, __ubuf__ T* sharedTmpBuffer, struct ReduceRepeatParams& params)
{
    int32_t dstOffset = 0;
    int32_t srcOffset = 0;
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T); // fp16=128 , fp32=64
    int32_t newRepeatTimes = params.repeatTimes / elementNumPerRep;
    int32_t leftData = params.repeatTimes % elementNumPerRep;

    uint64_t highMask = 0;
    uint64_t lowMask = 0;

#if ASCENDC_CPU_DEBUG == 0        // or 910B soc
    lowMask = params.repeatTimes; // MASK[31:0] is used to indicate the exact number of elments to be operated on by
                                  // SIMD instructions

    // set CTRL[56] as 1,for counter mask
    SetMaskCount();

    AscendCUtils::SetMask<T>(highMask, lowMask);
    ReduceSumIntrinsicsImpl<T>(sharedTmpBuffer, sharedTmpBuffer, 1, DEFAULT_REPEAT_STRIDE);

    SetMaskNorm();
#else
    if (newRepeatTimes != 0) {
        highMask = (sizeof(T) == sizeof(half)) ? FULL_MASK : 0;
        lowMask = FULL_MASK;

        AscendCUtils::SetMask<T>(highMask, lowMask);
        ReduceSumIntrinsicsImpl<T>(sharedTmpBuffer, sharedTmpBuffer, newRepeatTimes, DEFAULT_REPEAT_STRIDE);
    }
    highMask = 0;
    lowMask = 0;

    if (leftData > 0) { // has_tail
        srcOffset = elementNumPerRep * newRepeatTimes;
        highMask = (leftData > HALF_MASK_LEN) ? (((static_cast<uint64_t>(1)) << (leftData - HALF_MASK_LEN)) - 1) : 0;
        lowMask = (leftData > HALF_MASK_LEN) ? FULL_MASK : (((static_cast<uint64_t>(1)) << leftData) - 1);

        AscendCUtils::SetMask<T>(highMask, lowMask);
        ReduceSumIntrinsicsImpl<T>(dstLocal, sharedTmpBuffer + srcOffset, 1, DEFAULT_REPEAT_STRIDE);

        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        *(sharedTmpBuffer + newRepeatTimes) = *dstLocal;
        if (newRepeatTimes != 0) {
            event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventIdSToV);
            WaitFlag<HardEvent::S_V>(eventIdSToV);
        }
    }
#endif
}

/* **************************************** Reduce Interface ****************************************** */
template <typename T>
__aicore__ inline void CreateSpecialFormatMask(const int32_t& maskLen, uint64_t& highMask, uint64_t& lowMask)
{
    // create mask in the "0101010101" format
    int32_t halfLen = HALF_MASK_LEN / 2;
    for (int32_t i = 0; i < maskLen - halfLen; i++) {
        highMask = highMask << 2;
        highMask = highMask | 1;
    }
    int32_t lowMaskRange = maskLen >= halfLen ? halfLen : maskLen;
    for (int32_t i = 0; i < lowMaskRange; i++) {
        lowMask = lowMask << 2;
        lowMask = lowMask | 1;
    }
}

template <typename T>
__aicore__ inline void ReduceOperation(
    __ubuf__ T* sharedTmpBuffer, __ubuf__ T* srcLocal, const int32_t repeatTime, const int32_t srcRepStride,
    const uint64_t& highMask, const uint64_t& lowMask, const ReduceMode& mode)
{
    AscendCUtils::SetMask<T>(highMask, lowMask);
    switch (mode) {
        case ReduceMode::REDUCE_MAX:
            ReduceMaxIntrinsicsImpl(sharedTmpBuffer, srcLocal, repeatTime, srcRepStride);
            break;
        case ReduceMode::REDUCE_MIN:
            ReduceMinIntrinsicsImpl(sharedTmpBuffer, srcLocal, repeatTime, srcRepStride);
            break;
        case ReduceMode::REDUCE_SUM:
            ReduceSumIntrinsicsImpl(sharedTmpBuffer, srcLocal, repeatTime, srcRepStride);
            break;
        default:
            break;
    }
}

template <typename T>
__aicore__ inline void ReduceImplFirstStep(
    __ubuf__ T* sharedTmpBuffer, __ubuf__ T* srcLocal, struct ReduceRepeatParams& params, const ReduceMode& mode,
    int32_t& curData)
{
    int32_t dstOffset = 0;
    int32_t srcOffset = 0;
    int32_t range = params.repeatTimes / MAX_REPEAT_TIMES;

    for (int32_t index = 0; index < range; index++) {
        dstOffset = index * MAX_REPEAT_TIMES * VREDUCE_PER_REP_OUTPUT;
        srcOffset = index * MAX_REPEAT_TIMES * params.srcRepStride * ONE_BLK_SIZE / sizeof(T);
        ReduceOperation<T>(
            sharedTmpBuffer + dstOffset, srcLocal + srcOffset, MAX_REPEAT_TIMES, params.srcRepStride, params.highMask,
            params.lowMask, mode);
    }
    int32_t leftRepeatTimes = params.repeatTimes % MAX_REPEAT_TIMES;
    if (leftRepeatTimes > 0) {
        dstOffset = range * MAX_REPEAT_TIMES * VREDUCE_PER_REP_OUTPUT;
        srcOffset = range * MAX_REPEAT_TIMES * params.srcRepStride * ONE_BLK_SIZE / sizeof(T);
        ReduceOperation<T>(
            sharedTmpBuffer + dstOffset, srcLocal + srcOffset, leftRepeatTimes, params.srcRepStride, params.highMask,
            params.lowMask, mode);
    }
    curData = VREDUCE_PER_REP_OUTPUT * params.repeatTimes;
}

template <typename T>
__aicore__ inline void ReduceImplSecondStep(
    __ubuf__ T* sharedTmpBuffer, const ReduceMode& mode, int32_t& curData, int32_t preStartPos, int32_t secondStartPos)
{
    int32_t dstOffset = 0;
    int32_t srcOffset = 0;
    int32_t newMaskLen = 0;
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    int32_t newRepeatTimes = curData / elementNumPerRep;
    int32_t leftData = curData % elementNumPerRep;
    uint64_t highMask = 0;
    uint64_t lowMask = 0;
    int32_t bodyOutputCount = 0;
    int32_t tailOutputCount = 0;

    if (newRepeatTimes >= 1) {
        highMask = (sizeof(T) == sizeof(half)) ? 0x5555555555555555 : 0;
        lowMask = 0x5555555555555555;

        ReduceOperation<T>(
            sharedTmpBuffer + secondStartPos, sharedTmpBuffer + preStartPos, newRepeatTimes, DEFAULT_REPEAT_STRIDE,
            highMask, lowMask, mode);
        bodyOutputCount = newRepeatTimes * VREDUCE_PER_REP_OUTPUT;
    }
    highMask = 0;
    lowMask = 0;

    if (leftData > 0) {
        newMaskLen = leftData / VREDUCE_PER_REP_OUTPUT;
        // create mask in the "0101010101" format
        CreateSpecialFormatMask<T>(newMaskLen, highMask, lowMask);

        dstOffset = secondStartPos + bodyOutputCount;
        srcOffset = preStartPos + newRepeatTimes * elementNumPerRep;
        ReduceOperation<T>(
            sharedTmpBuffer + dstOffset, sharedTmpBuffer + srcOffset, 1, DEFAULT_REPEAT_STRIDE, highMask, lowMask,
            mode);
        tailOutputCount = VREDUCE_PER_REP_OUTPUT;
    }

    curData = bodyOutputCount + tailOutputCount;
}

template <typename T>
__aicore__ inline void GetIndex(
    __ubuf__ T* sharedTmpBuffer, int32_t secondStartPos, int32_t& secondIndex, int32_t& thirdIndex)
{
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    if (sizeof(T) == sizeof(half)) {
        thirdIndex = *reinterpret_cast<__ubuf__ uint16_t*>(sharedTmpBuffer + secondStartPos + 1);
        ASCENDC_CHECK_VALUE_RANGE(thirdIndex, 0, elementNumPerRep - 1, "thirdIndex", "GetIndex");
        secondIndex = *reinterpret_cast<__ubuf__ uint16_t*>(sharedTmpBuffer + thirdIndex + 1);
        ASCENDC_CHECK_VALUE_RANGE(secondIndex, 0, elementNumPerRep - 1, "secondIndex", "GetIndex");
    } else {
        thirdIndex = *reinterpret_cast<__ubuf__ uint32_t*>(sharedTmpBuffer + secondStartPos + 1);
        ASCENDC_CHECK_VALUE_RANGE(thirdIndex, 0, elementNumPerRep - 1, "thirdIndex", "GetIndex");
        secondIndex = *reinterpret_cast<__ubuf__ uint32_t*>(sharedTmpBuffer + thirdIndex + 1);
        ASCENDC_CHECK_VALUE_RANGE(secondIndex, 0, elementNumPerRep - 1, "secondIndex", "GetIndex");
    }
}

template <typename T>
__aicore__ inline void GetIndex(
    __ubuf__ T* sharedTmpBuffer, int32_t secondStartPos, int32_t thirdStartPos, int32_t& firstIndex,
    int32_t& secondIndex, int32_t& thirdIndex)
{
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    using U = typename Conditional<sizeof(T) == B16_BYTE_SIZE, uint16_t, uint32_t>::type;
    thirdIndex = *reinterpret_cast<__ubuf__ U*>(sharedTmpBuffer + thirdStartPos + 1);
    ASCENDC_CHECK_VALUE_RANGE(thirdIndex, 0, elementNumPerRep - 1, "thirdIndex", "GetIndex with firstIndex");
    secondIndex = *reinterpret_cast<__ubuf__ U*>(sharedTmpBuffer + secondStartPos + thirdIndex + 1);
    ASCENDC_CHECK_VALUE_RANGE(secondIndex, 0, elementNumPerRep - 1, "secondIndex", "GetIndex with firstIndex");
    firstIndex = *reinterpret_cast<__ubuf__ U*>(
        sharedTmpBuffer + elementNumPerRep * (thirdIndex / VREDUCE_PER_REP_OUTPUT) + secondIndex + 1);
    ASCENDC_CHECK_VALUE_RANGE(firstIndex, 0, elementNumPerRep - 1, "firstIndex", "GetIndex with firstIndex");
}

template <typename T>
__aicore__ inline void GetIndex(
    __ubuf__ T* sharedTmpBuffer, int32_t secondStartPos, int32_t thirdStartPos, int32_t fourthStartPos,
    int32_t& firstIndex, int32_t& secondIndex, int32_t& thirdIndex, int32_t& fourthIndex)
{
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    if (sizeof(T) == sizeof(half)) {
        fourthIndex = *reinterpret_cast<__ubuf__ uint16_t*>(sharedTmpBuffer + fourthStartPos + 1);
        ASCENDC_ASSERT(((fourthIndex >= 0) && (fourthIndex < elementNumPerRep)), {
            KERNEL_LOG(
                KERNEL_ERROR, "fourthIndex is %d, which should be in range [0, %d])", fourthIndex, elementNumPerRep);
        });
        thirdIndex = *reinterpret_cast<__ubuf__ uint16_t*>(sharedTmpBuffer + thirdStartPos + fourthIndex + 1);
        ASCENDC_ASSERT(((thirdIndex >= 0) && (thirdIndex < elementNumPerRep)), {
            KERNEL_LOG(
                KERNEL_ERROR, "thirdIndex is %d, which should be in range [0, %d])", thirdIndex, elementNumPerRep);
        });
        secondIndex = *reinterpret_cast<__ubuf__ uint16_t*>(
            sharedTmpBuffer + secondStartPos + elementNumPerRep * (fourthIndex / VREDUCE_PER_REP_OUTPUT) + thirdIndex +
            1);
        ASCENDC_ASSERT(((secondIndex >= 0) && (secondIndex < elementNumPerRep)), {
            KERNEL_LOG(
                KERNEL_ERROR, "secondIndex is %d, which should be in range [0, %d])", secondIndex, elementNumPerRep);
        });
        firstIndex = *reinterpret_cast<__ubuf__ uint16_t*>(
            sharedTmpBuffer +
            elementNumPerRep * (elementNumPerRep * (fourthIndex / VREDUCE_PER_REP_OUTPUT) + thirdIndex) /
                VREDUCE_PER_REP_OUTPUT +
            secondIndex + 1);
        ASCENDC_ASSERT(((firstIndex >= 0) && (firstIndex < elementNumPerRep)), {
            KERNEL_LOG(
                KERNEL_ERROR, "firstIndex is %d, which should be in range [0, %d])", firstIndex, elementNumPerRep);
        });
    } else {
        fourthIndex = *reinterpret_cast<__ubuf__ uint32_t*>(sharedTmpBuffer + fourthStartPos + 1);
        ASCENDC_ASSERT(((fourthIndex >= 0) && (fourthIndex < elementNumPerRep)), {
            KERNEL_LOG(
                KERNEL_ERROR, "fourthIndex is %d, which should be in range [0, %d])", fourthIndex, elementNumPerRep);
        });
        thirdIndex = *reinterpret_cast<__ubuf__ uint32_t*>(sharedTmpBuffer + thirdStartPos + fourthIndex + 1);
        ASCENDC_ASSERT(((thirdIndex >= 0) && (thirdIndex < elementNumPerRep)), {
            KERNEL_LOG(
                KERNEL_ERROR, "thirdIndex is %d, which should be in range [0, %d])", thirdIndex, elementNumPerRep);
        });
        secondIndex = *reinterpret_cast<__ubuf__ uint32_t*>(
            sharedTmpBuffer + secondStartPos + elementNumPerRep * (fourthIndex / VREDUCE_PER_REP_OUTPUT) + thirdIndex +
            1);
        ASCENDC_ASSERT(((secondIndex >= 0) && (secondIndex < elementNumPerRep)), {
            KERNEL_LOG(
                KERNEL_ERROR, "secondIndex is %d, which should be in range [0, %d])", secondIndex, elementNumPerRep);
        });
        firstIndex = *reinterpret_cast<__ubuf__ uint32_t*>(
            sharedTmpBuffer +
            elementNumPerRep * (elementNumPerRep * (fourthIndex / VREDUCE_PER_REP_OUTPUT) + thirdIndex) /
                VREDUCE_PER_REP_OUTPUT +
            secondIndex + 1);
        ASCENDC_ASSERT(((firstIndex >= 0) && (firstIndex < elementNumPerRep)), {
            KERNEL_LOG(
                KERNEL_ERROR, "firstIndex is %d, which should be in range [0, %d])", firstIndex, elementNumPerRep);
        });
    }
}

template <typename T>
__aicore__ inline void ReduceImplThirdStep(
    __ubuf__ T* dstLocal, __ubuf__ T* sharedTmpBuffer, const int32_t srcRepStride, const ReduceMode& mode,
    int32_t& curData, int32_t& secondStartPos, int32_t& thirdStartPos)
{
    int32_t preNum = 0;
    int32_t firstIndex = 0;
    int32_t secondIndex = 0;
    int32_t thirdIndex = 0;
    int32_t fourthIndex = 0;
    int32_t dstOffset = 0;
    int32_t srcOffset = 0;
    uint64_t highMask = 0;
    uint64_t lowMask = 0;

    int32_t offsetNumPerRep = ONE_BLK_SIZE / sizeof(T) * srcRepStride;
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    if (curData == VREDUCE_PER_REP_OUTPUT) {
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);

        GetIndex<T>(sharedTmpBuffer, secondStartPos, secondIndex, thirdIndex);
        preNum = offsetNumPerRep * (thirdIndex / VREDUCE_PER_REP_OUTPUT);
        int32_t redultIndex = secondIndex + preNum;
        *dstLocal = *(sharedTmpBuffer + secondStartPos);
        *(dstLocal + 1) = *reinterpret_cast<T*>(&redultIndex);
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        event_t eventIdSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
        return;
    }

    int32_t newMaskLen = curData / VREDUCE_PER_REP_OUTPUT;
    CreateSpecialFormatMask<T>(newMaskLen, highMask, lowMask);
    if (curData > elementNumPerRep) {
        ReduceImplSecondStep<T>(sharedTmpBuffer, mode, curData, secondStartPos, thirdStartPos);

        int32_t fourthStartPos =
            (((thirdStartPos + curData) * sizeof(T) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE) * ONE_BLK_SIZE / sizeof(T);
        dstOffset = fourthStartPos;
        srcOffset = thirdStartPos;
        PipeBarrier<PIPE_V>();
        ReduceOperation<T>(
            sharedTmpBuffer + dstOffset, sharedTmpBuffer + srcOffset, 1, DEFAULT_REPEAT_STRIDE, highMask, lowMask,
            mode);
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);

        *dstLocal = *(sharedTmpBuffer + dstOffset);

        GetIndex<T>(
            sharedTmpBuffer, secondStartPos, thirdStartPos, fourthStartPos, firstIndex, secondIndex, thirdIndex,
            fourthIndex);
        preNum = offsetNumPerRep *
                 (elementNumPerRep * (elementNumPerRep * (fourthIndex / VREDUCE_PER_REP_OUTPUT) + thirdIndex) /
                      VREDUCE_PER_REP_OUTPUT +
                  secondIndex) /
                 VREDUCE_PER_REP_OUTPUT;
    } else {
        dstOffset = thirdStartPos;
        srcOffset = secondStartPos;

        ReduceOperation<T>(
            sharedTmpBuffer + dstOffset, sharedTmpBuffer + srcOffset, 1, DEFAULT_REPEAT_STRIDE, highMask, lowMask,
            mode);
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);

        *dstLocal = *(sharedTmpBuffer + thirdStartPos);

        GetIndex<T>(sharedTmpBuffer, secondStartPos, thirdStartPos, firstIndex, secondIndex, thirdIndex);
        preNum = offsetNumPerRep * (elementNumPerRep * (thirdIndex / VREDUCE_PER_REP_OUTPUT) + secondIndex) /
                 VREDUCE_PER_REP_OUTPUT;
    }

    int32_t redultIndex = firstIndex + preNum;
    *(dstLocal + 1) = *reinterpret_cast<T*>(&redultIndex);
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    event_t eventIdSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
}

template <typename T>
__aicore__ inline void ReduceSumFirstStep(
    __ubuf__ T* sharedTmpBuffer, __ubuf__ T* srcLocal, struct ReduceRepeatParams& params)
{
    int32_t dstOffset = 0;
    int32_t srcOffset = 0;
    int32_t range = params.repeatTimes / MAX_REPEAT_TIMES;

    for (int32_t index = 0; index < range; index++) {
        dstOffset = index * MAX_REPEAT_TIMES;
        srcOffset = index * MAX_REPEAT_TIMES * (params.srcRepStride * ONE_BLK_SIZE / sizeof(T));
        ReduceOperation<T>(
            sharedTmpBuffer + dstOffset, srcLocal + srcOffset, MAX_REPEAT_TIMES, params.srcRepStride, params.highMask,
            params.lowMask, ReduceMode::REDUCE_SUM);
    }

    int32_t leftRepeatTimes = params.repeatTimes % MAX_REPEAT_TIMES;
    if (leftRepeatTimes > 0) {
        dstOffset = range * MAX_REPEAT_TIMES;
        srcOffset = range * MAX_REPEAT_TIMES * (params.srcRepStride * ONE_BLK_SIZE / sizeof(T));
        ReduceOperation<T>(
            sharedTmpBuffer + dstOffset, srcLocal + srcOffset, leftRepeatTimes, params.srcRepStride, params.highMask,
            params.lowMask, ReduceMode::REDUCE_SUM);
    }
}

template <typename T>
__aicore__ inline void ReduceSumFinalStep(__ubuf__ T* dstLocal, __ubuf__ T* sharedTmpBuffer, int32_t& secondResultNum)
{
    uint64_t highMask = 0;
    uint64_t lowMask = 0;
    if (secondResultNum == 1) {
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        *(dstLocal) = *(sharedTmpBuffer);
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        event_t eventIdSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
    } else {
        highMask = (secondResultNum > HALF_MASK_LEN) ?
                       (((static_cast<uint64_t>(1)) << (secondResultNum - HALF_MASK_LEN)) - 1) :
                       0;
        lowMask = (secondResultNum > HALF_MASK_LEN) ? FULL_MASK : (((static_cast<uint64_t>(1)) << secondResultNum) - 1);
        ReduceOperation<T>(
            dstLocal, sharedTmpBuffer, 1, DEFAULT_REPEAT_STRIDE, highMask, lowMask, ReduceMode::REDUCE_SUM);
    }
}

template <typename T>
__aicore__ inline void ReduceSumImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, __ubuf__ T* sharedTmpBuffer, struct ReduceRepeatParams& params)
{
    ReduceSumFirstStep<T>(sharedTmpBuffer, srcLocal, params);
    PipeBarrier<PIPE_V>();
    ReduceSumSecondStep<T>(dstLocal, sharedTmpBuffer, params);
    PipeBarrier<PIPE_V>();
    int32_t secondResultNum = DivCeil(params.repeatTimes, ONE_REPEAT_BYTE_SIZE / sizeof(T));
    ReduceSumFinalStep<T>(dstLocal, sharedTmpBuffer, secondResultNum);
}

template <typename T>
__aicore__ inline void ReduceImplSecondStepNoIndex(
    __ubuf__ T* sharedTmpBuffer, const ReduceMode& mode, int32_t& curData)
{
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T); // fp16=128,fp32=64
    int32_t newRepeatTimes = curData / elementNumPerRep;
    int32_t leftData = curData % elementNumPerRep;
    uint64_t highMask = 0, lowMask = 0;
    if (newRepeatTimes != 0) {
        CreateSpecialFormatMask<T>(elementNumPerRep / VREDUCE_PER_REP_OUTPUT, highMask, lowMask);
        ReduceOperation<T>(
            sharedTmpBuffer, sharedTmpBuffer, newRepeatTimes, DEFAULT_REPEAT_STRIDE, highMask, lowMask, mode);
    }
    highMask = 0;
    lowMask = 0;
    if (leftData > 0) {
        CreateSpecialFormatMask<T>(leftData / VREDUCE_PER_REP_OUTPUT, highMask, lowMask);
        ReduceOperation<T>(
            sharedTmpBuffer + newRepeatTimes * VREDUCE_PER_REP_OUTPUT,
            sharedTmpBuffer + newRepeatTimes * elementNumPerRep, 1, DEFAULT_REPEAT_STRIDE, highMask, lowMask, mode);
        newRepeatTimes += 1;
    }
    curData = newRepeatTimes * VREDUCE_PER_REP_OUTPUT;
}

template <typename T>
__aicore__ inline void ReduceImplThirdStepNoIndex(
    __ubuf__ T* dstLocal, __ubuf__ T* sharedTmpBuffer, const ReduceMode& mode, int32_t& curData)
{
    uint64_t highMask = 0;
    uint64_t lowMask = 0;

    CreateSpecialFormatMask<T>(curData / VREDUCE_PER_REP_OUTPUT, highMask, lowMask);
    ReduceOperation<T>(sharedTmpBuffer, sharedTmpBuffer, 1, DEFAULT_REPEAT_STRIDE, highMask, lowMask, mode);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    *dstLocal = *sharedTmpBuffer;
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    event_t eventIdSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
}

template <typename T>
__aicore__ inline void ReduceImplWithIndex(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, __ubuf__ T* sharedTmpBuffer, struct ReduceRepeatParams& params,
    const ReduceMode& mode)
{
    if ASCEND_IS_AIV {
        if (params.repeatTimes == 1) {
            ReduceOperation<T>(dstLocal, srcLocal, 1, params.srcRepStride, params.highMask, params.lowMask, mode);
        } else {
            int32_t curData = 0;
            ReduceImplFirstStep<T>(sharedTmpBuffer, srcLocal, params, mode, curData);
            PipeBarrier<PIPE_V>();
            int32_t secondStartPos =
                ((curData * sizeof(T) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE) * ONE_BLK_SIZE / sizeof(T);
            ReduceImplSecondStep<T>(sharedTmpBuffer, mode, curData, 0, secondStartPos);
            PipeBarrier<PIPE_V>();
            int32_t thirdStartPos =
                (((secondStartPos + curData) * sizeof(T) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE) * ONE_BLK_SIZE / sizeof(T);
            ReduceImplThirdStep<T>(
                dstLocal, sharedTmpBuffer, params.srcRepStride, mode, curData, secondStartPos, thirdStartPos);
        }
    }
}

template <typename T>
__aicore__ inline void ReduceImplNoIndex(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, __ubuf__ T* sharedTmpBuffer, struct ReduceRepeatParams& params,
    const ReduceMode& mode)
{
    if ASCEND_IS_AIV {
        if (params.repeatTimes == 1) {
            ReduceOperation<T>(
                sharedTmpBuffer, srcLocal, 1, params.srcRepStride, params.highMask, params.lowMask, mode);
            event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIdVToS);
            WaitFlag<HardEvent::V_S>(eventIdVToS);
            *dstLocal = *sharedTmpBuffer;
            event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            event_t eventIdSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_V>(eventIdSToV);
            WaitFlag<HardEvent::S_V>(eventIdSToV);
            SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
        } else {
            if (mode == ReduceMode::REDUCE_SUM) {
                ReduceSumImpl<T>(dstLocal, srcLocal, sharedTmpBuffer, params);
            } else {
                int32_t curData = 0;
                ReduceImplFirstStep<T>(sharedTmpBuffer, srcLocal, params, mode, curData);
                PipeBarrier<PIPE_V>();
                ReduceImplSecondStepNoIndex<T>(sharedTmpBuffer, mode, curData);

                int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T); // fp16=128,fp32=64
                if (curData <= elementNumPerRep) {
                    PipeBarrier<PIPE_V>();
                    ReduceImplThirdStepNoIndex<T>(dstLocal, sharedTmpBuffer, mode, curData);
                    return;
                }
                PipeBarrier<PIPE_V>();
                ReduceImplSecondStepNoIndex<T>(sharedTmpBuffer, mode, curData);
                if (curData <= elementNumPerRep) {
                    PipeBarrier<PIPE_V>();
                    ReduceImplThirdStepNoIndex<T>(dstLocal, sharedTmpBuffer, mode, curData);
                }
            }
        }
    }
}
template <typename T>
__aicore__ inline void ReduceImpl(
    __ubuf__ T* dstLocal, __ubuf__ T* srcLocal, __ubuf__ T* sharedTmpBuffer, struct ReduceRepeatParams& params,
    bool calIndex, const ReduceMode& mode)
{
    if (calIndex) {
        ReduceImplWithIndex<T>(dstLocal, srcLocal, sharedTmpBuffer, params, mode);
    } else {
        ReduceImplNoIndex<T>(dstLocal, srcLocal, sharedTmpBuffer, params, mode);
    }
}

template <typename T>
__aicore__ inline void ReduceTailCompute(
    const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<T>& sharedTmpBuffer, const int32_t count,
    bool calIndex, const ReduceMode& mode)
{
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(PrimT<T>); // fp16=128 , fp32=64
    int32_t repeatTime = count / elementNumPerRep;
    int32_t tailCount = count % elementNumPerRep; // tailCount  <= 128/64 repeat=1

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    PrimT<T> bodyValue = dst.GetValue(0);
    PrimT<T> bodyIndex = dst.GetValue(1);

    struct ReduceRepeatParams tailParams(
        tailCount, 1, DEFAULT_REDUCE_DST_REP_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);

    ReduceImpl<PrimT<T>>(
        (__ubuf__ PrimT<T>*)dst.GetPhyAddr(), // reuse dst
        (__ubuf__ PrimT<T>*)src.GetPhyAddr(elementNumPerRep * repeatTime),
        (__ubuf__ PrimT<T>*)sharedTmpBuffer.GetPhyAddr(), tailParams, calIndex, mode);
    eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    PrimT<T> tailValue = dst.GetValue(0);
    PrimT<T> tailIndex = dst.GetValue(1);

    // bodyresult tailresult need vcmin/vcmax again
    struct ReduceRepeatParams lastParams(
        2, 1, DEFAULT_REDUCE_DST_REP_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    sharedTmpBuffer.SetValue(0, bodyValue);
    sharedTmpBuffer.SetValue(1, tailValue);
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    ReduceImpl<PrimT<T>>(
        (__ubuf__ PrimT<T>*)dst.GetPhyAddr(), (__ubuf__ PrimT<T>*)sharedTmpBuffer.GetPhyAddr(),
        (__ubuf__ PrimT<T>*)sharedTmpBuffer.GetPhyAddr(), lastParams, calIndex, mode);
    if (calIndex) {
        eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        PrimT<T> lastIndexVal = dst.GetValue(1);
        uint32_t newIndex = 0;
        uint32_t lastIndex = 0;
        if (sizeof(PrimT<T>) == sizeof(half)) {
            lastIndex = *reinterpret_cast<uint16_t*>(&lastIndexVal);
            newIndex = elementNumPerRep * repeatTime + *reinterpret_cast<uint16_t*>(&tailIndex);
        } else {
            lastIndex = *reinterpret_cast<uint32_t*>(&lastIndexVal);
            newIndex = elementNumPerRep * repeatTime + *reinterpret_cast<uint32_t*>(&tailIndex);
        }
        if (lastIndex == 1) {
            dst.SetValue(1, *reinterpret_cast<PrimT<T>*>(&newIndex));
        } else {
            dst.SetValue(1, bodyIndex);
        }
        eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        event_t eventIdSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
    }
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(uint32_t& maxMinValue, uint32_t& maxMinIndex)
{
    int64_t maxMinCnt = get_max_min_cnt();
    if constexpr (IsSameType<T, half>::value) {
        constexpr uint64_t valueMask = 0xffff;
        maxMinValue = (static_cast<uint64_t>(maxMinCnt) & valueMask);
    } else {
        constexpr uint64_t valueMask = 0xffffffff;
        maxMinValue = (static_cast<uint64_t>(maxMinCnt) & valueMask);
    }
    constexpr uint64_t indexBit = 32;
    maxMinIndex = (static_cast<uint64_t>(maxMinCnt) >> indexBit);
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(uint32_t& maxMinValue)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "GetReduceMaxMinCount with only maxMinValue");
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(T& maxMinValue, T& maxMinIndex)
{
    int64_t maxMinCnt = get_max_min_cnt();
    uint32_t maxVal = 0;
    uint32_t maxIdx = 0;
    if constexpr (IsSameType<T, half>::value) {
        constexpr uint64_t valueMask = 0xffff;
        maxVal = (static_cast<uint64_t>(maxMinCnt) & valueMask);
    } else {
        constexpr uint64_t valueMask = 0xffffffff;
        maxVal = (static_cast<uint64_t>(maxMinCnt) & valueMask);
    }
    maxMinValue = *(reinterpret_cast<T*>(&maxVal));

    constexpr uint64_t indexBit = 32;
    maxIdx = (static_cast<uint64_t>(maxMinCnt) >> indexBit);
    maxMinIndex = *(reinterpret_cast<T*>(&maxIdx));
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(T& maxMinValue)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "GetReduceMaxMinCount with only maxMinValue");
}

template <typename T>
__aicore__ inline T GetAccValImpl()
{
    int64_t accVal = get_acc_val();
    return *(reinterpret_cast<T*>(&accVal));
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_REDUCE_IMPL_H
#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_REDUCE_IMPL_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_REDUCE_IMPL_H__
#endif
