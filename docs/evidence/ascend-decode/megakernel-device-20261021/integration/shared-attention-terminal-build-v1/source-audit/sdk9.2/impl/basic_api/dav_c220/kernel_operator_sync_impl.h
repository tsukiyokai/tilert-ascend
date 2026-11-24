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
 * \file kernel_operator_sync_impl.h
 * \brief
 */
#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message( \
    "impl/basic_api/dav_c220/kernel_operator_sync_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"basic_api/kernel_tpipe.h\"\" and use public functions or variables defined in interface headers files.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_SYNC_IMPL_H__
#endif
#ifndef ASCENDC_MODULE_OPERATOR_SYNC_IMPL_H
#define ASCENDC_MODULE_OPERATOR_SYNC_IMPL_H

#include "../utils/kernel_utils_constants.h"
#include "../../../include/basic_api/kernel_tpipe.h"
#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_TPIPE_H__
#endif
#include "../../../include/basic_api/kernel_common.h"
namespace AscendC {

__aicore__ inline void ClcSyncCount(
    __gm__ int32_t* localSyncGM, __ubuf__ int32_t* ubWorkspaceAddr, const int32_t blockIdx, const int32_t totalBlocks,
    bool isFirst, int32_t& count)
{
    if (isFirst) {
        __ubuf__ int32_t* localUbAddr = ubWorkspaceAddr + (blockIdx * DEFAULT_BLK_NUM);
        *(reinterpret_cast<__ubuf__ int32_t*>(localUbAddr)) = 1;
        event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        copy_ubuf_to_gm(
            static_cast<__gm__ void*>(localSyncGM), static_cast<__ubuf__ void*>(localUbAddr), 0, 1, 1, 0, 0);
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        count = 1;
        for (int32_t i = 0; i < totalBlocks; i++) {
            if (i != blockIdx) {
                count += *(reinterpret_cast<__ubuf__ int32_t*>(ubWorkspaceAddr) + i * ONE_BLK_FLOAT_NUM);
            }
        }
    } else {
        for (int32_t i = 0; i < totalBlocks; i++) {
            count += *(reinterpret_cast<__ubuf__ int32_t*>(ubWorkspaceAddr) + i * ONE_BLK_FLOAT_NUM);
        }
    }
}

__aicore__ inline int64_t GetBlockNum();
template <bool isAIVOnly = true>
__aicore__ inline void SoftSyncAllImpl(
    __gm__ int32_t* gmWorkspaceAddr, __ubuf__ int32_t* ubWorkspaceAddr, const int usedCores)
{
    if ASCEND_IS_AIC {
        return;
    }
    __sync_all_stub(usedCores, isAIVOnly);
    PipeBarrier<PIPE_ALL>();

    int32_t totalBlocks = isAIVOnly ? GetBlockNum() : (GetTaskRationImpl() * GetBlockNum());
    totalBlocks = usedCores != 0 ? usedCores : totalBlocks;
#ifdef __ASCENDC_ENABLE_SUPER_KERNEL__
    int32_t blockIdx = GetBlockIdxImpl();
#else
    int32_t blockIdx = isAIVOnly ? get_block_idx() : GetBlockIdxImpl();
#endif

    __gm__ int32_t* localSyncGM = gmWorkspaceAddr + (blockIdx * DEFAULT_BLK_NUM);
    __ubuf__ int32_t* localUbAddr = ubWorkspaceAddr + (blockIdx * DEFAULT_BLK_NUM);

    copy_gm_to_ubuf(static_cast<__ubuf__ void*>(localUbAddr), static_cast<__gm__ void*>(localSyncGM), 0, 1, 1, 0, 0);
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    int32_t curValue = *(reinterpret_cast<__ubuf__ int32_t*>(localUbAddr)) + 1;
    bool isFirst = curValue == 1 ? true : false;
    *(reinterpret_cast<__ubuf__ int32_t*>(localUbAddr)) = curValue;
    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    copy_ubuf_to_gm(static_cast<__gm__ void*>(localSyncGM), static_cast<__ubuf__ void*>(localUbAddr), 0, 1, 1, 0, 0);
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    int32_t totalBlockCount = ONE_BLK_FLOAT_NUM * totalBlocks;
    uint16_t blockLen = totalBlockCount / AscendCUtils::GetC0Count(sizeof(int32_t));
    while (true) {
        copy_gm_to_ubuf(
            static_cast<__ubuf__ void*>(ubWorkspaceAddr), static_cast<__gm__ void*>(gmWorkspaceAddr), 0, 1, blockLen, 0,
            0);
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        int32_t count = 0;
        ClcSyncCount(localSyncGM, ubWorkspaceAddr, blockIdx, totalBlocks, isFirst, count);
        event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
        WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);
        if (count >= (totalBlocks * curValue)) {
            break;
        }
    }
    __sync_all_stub(usedCores, isAIVOnly);
}

constexpr uint16_t SYNC_AIV_FLAG = 12;
constexpr uint16_t SYNC_AIC_FLAG = 11;
constexpr uint16_t SYNC_AIC_AIV_FLAG = 13;
constexpr uint16_t SYNC_AIV_ONLY_ALL = 14;
constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;

__aicore__ inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId)
{
    return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) + ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}

template <pipe_t AIV_PIPE, pipe_t AIC_PIPE>
__aicore__ inline void SetNextTaskStartV3Impl(uint32_t earlyStartConfig)
{
    if ASCEND_IS_AIC {
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIC_TO_AIC_SET) {
            ffts_cross_core_sync(AIC_PIPE, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIC_FLAG)); // SET:C--->C
        }
    }
    if ASCEND_IS_AIV {
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIV_TO_AIV_SET) {
            ffts_cross_core_sync(AIV_PIPE, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIV_ONLY_ALL)); // SET: V--->V
        }
    }
}

template <pipe_t AIV_PIPE = PIPE_MTE3, pipe_t AIC_PIPE = PIPE_FIX, bool FORCE = false>
__aicore__ inline void SetNextTaskStartImpl()
{
#if defined(__ASCENDC_SUPERKERNEL_EARLY_START_V1) || defined(__ASCENDC_SUPERKERNEL_EARLY_START_V2)
    if ASCEND_IS_AIC {
        ffts_cross_core_sync(AIC_PIPE, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIC_FLAG));
    }
    if ASCEND_IS_AIV {
        ffts_cross_core_sync(AIV_PIPE, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIV_ONLY_ALL));
    }
#elif defined(__ASCENDC_SUPERKERNEL_EARLY_START_V3)
    SetNextTaskStartV3Impl<AIV_PIPE, AIC_PIPE>(g_super_kernel_early_start_config);
#else
    if constexpr (FORCE) {
        SetNextTaskStartV3Impl<AIV_PIPE, AIC_PIPE>(g_super_kernel_early_start_config);
    }
#endif
}

__aicore__ inline void WaitPreTaskEndV2Impl()
{
#ifdef __ASCENDC_SUPERKERNEL_EARLY_START_V2
    if (g_super_kernel_early_start_config == ASCENDC_SUPER_KERNEL_EARLY_START_AIC_TO_AIC) {
        if ASCEND_IS_AIC {
            wait_flag_dev(AscendC::SYNC_AIC_FLAG);
        }
    } else if (g_super_kernel_early_start_config == ASCENDC_SUPER_KERNEL_EARLY_START_AIV_TO_AIV) {
        if ASCEND_IS_AIV {
            wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);
        }
    } else if (g_super_kernel_early_start_config == ASCENDC_SUPER_KERNEL_EARLY_START_AIC_TO_AIV) {
        if ASCEND_IS_AIC {
            wait_flag_dev(AscendC::SYNC_AIC_FLAG);
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIC_AIV_FLAG));
        }
        if ASCEND_IS_AIV {
            wait_flag_dev(AscendC::SYNC_AIC_AIV_FLAG);
        }
    } else if (g_super_kernel_early_start_config == ASCENDC_SUPER_KERNEL_EARLY_START_AIV_TO_AIC) {
        if ASCEND_IS_AIC {
            wait_flag_dev(AscendC::SYNC_AIV_FLAG);
        }
        if ASCEND_IS_AIV {
            wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG));
        }
    } else if (g_super_kernel_early_start_config == ASCENDC_SUPER_KERNEL_EARLY_START_MIX_TO_AIV) {
        if ASCEND_IS_AIC {
            wait_flag_dev(AscendC::SYNC_AIC_FLAG);
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIC_AIV_FLAG));
        }
        if ASCEND_IS_AIV {
            wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);
            wait_flag_dev(AscendC::SYNC_AIC_AIV_FLAG);
        }
    } else if (g_super_kernel_early_start_config == ASCENDC_SUPER_KERNEL_EARLY_START_MIX_TO_AIC) {
        if ASCEND_IS_AIC {
            wait_flag_dev(AscendC::SYNC_AIC_FLAG);
            wait_flag_dev(AscendC::SYNC_AIV_FLAG);
        }
        if ASCEND_IS_AIV {
            wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG));
        }
    } else if (g_super_kernel_early_start_config == ASCENDC_SUPER_KERNEL_EARLY_START_AIC_TO_MIX) {
        if ASCEND_IS_AIC {
            wait_flag_dev(AscendC::SYNC_AIC_FLAG);
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIC_AIV_FLAG));
        }
        if ASCEND_IS_AIV {
            wait_flag_dev(AscendC::SYNC_AIC_AIV_FLAG);
        }
    } else if (g_super_kernel_early_start_config == ASCENDC_SUPER_KERNEL_EARLY_START_AIV_TO_MIX) {
        if ASCEND_IS_AIC {
            wait_flag_dev(AscendC::SYNC_AIV_FLAG);
        }
        if ASCEND_IS_AIV {
            wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG));
        }
    } else if (g_super_kernel_early_start_config == ASCENDC_SUPER_KERNEL_EARLY_START_MIX_TO_MIX) {
        if ASCEND_IS_AIC {
            wait_flag_dev(AscendC::SYNC_AIC_FLAG);
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIC_AIV_FLAG));
            wait_flag_dev(AscendC::SYNC_AIV_FLAG);
        }
        if ASCEND_IS_AIV {
            wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG));
            wait_flag_dev(AscendC::SYNC_AIC_AIV_FLAG);
        }
    }
    return;
#endif
}

__aicore__ inline void WaitPreTaskEndV3Impl(uint32_t earlyStartConfig)
{
    if ASCEND_IS_AIC {
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIC_TO_AIC_WAIT) {
            wait_flag_dev(AscendC::SYNC_AIC_FLAG); // WAIT: C--->C
        }
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIC_TO_AIV_SET) {
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIC_AIV_FLAG)); // SET: C--->V
        }
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIV_TO_AIC_WAIT) {
            wait_flag_dev(AscendC::SYNC_AIV_FLAG); // WAIT: V--->C
        }
    }
    if ASCEND_IS_AIV {
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIV_TO_AIV_WAIT) {
            wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL); // WAIT: V--->V
        }
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIV_TO_AIC_SET) {
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG)); // SET: V--->C
        }
        if (earlyStartConfig & Internal::ASCENDC_SUPER_KERNEL_EARLY_START_MASK_AIC_TO_AIV_WAIT) {
            wait_flag_dev(AscendC::SYNC_AIC_AIV_FLAG); // WAIT: C--->V
        }
    }
}

// optimize if-else scalar cost, will be called only in superkernel.cpp
template <int8_t earlyStartConfig = -1, bool FORCE = false>
__aicore__ inline void WaitPreTaskEndImpl()
{
#ifdef __ASCENDC_SUPERKERNEL_EARLY_START_V1
    if ASCEND_IS_AIC {
        wait_flag_dev(AscendC::SYNC_AIC_FLAG);
        ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIC_AIV_FLAG));
        wait_flag_dev(AscendC::SYNC_AIV_FLAG);
    }
    if ASCEND_IS_AIV {
        wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);
        ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG));
        wait_flag_dev(AscendC::SYNC_AIC_AIV_FLAG);
    }
#elif defined(__ASCENDC_SUPERKERNEL_EARLY_START_V2)
    if constexpr (earlyStartConfig == -1) {
        WaitPreTaskEndV2Impl();
    } else if constexpr (earlyStartConfig == ASCENDC_SUPER_KERNEL_EARLY_START_AIC_TO_AIC) {
        if ASCEND_IS_AIC {
            wait_flag_dev(AscendC::SYNC_AIC_FLAG);
        }
    } else if constexpr (earlyStartConfig == ASCENDC_SUPER_KERNEL_EARLY_START_AIV_TO_AIV) {
        if ASCEND_IS_AIV {
            wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);
        }
    } else if constexpr (earlyStartConfig == ASCENDC_SUPER_KERNEL_EARLY_START_AIC_TO_AIV) {
        if ASCEND_IS_AIC {
            wait_flag_dev(AscendC::SYNC_AIC_FLAG);
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIC_AIV_FLAG));
        }
        if ASCEND_IS_AIV {
            wait_flag_dev(AscendC::SYNC_AIC_AIV_FLAG);
        }
    } else if constexpr (earlyStartConfig == ASCENDC_SUPER_KERNEL_EARLY_START_AIV_TO_AIC) {
        if ASCEND_IS_AIC {
            wait_flag_dev(AscendC::SYNC_AIV_FLAG);
        }
        if ASCEND_IS_AIV {
            wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG));
        }
    } else if constexpr (earlyStartConfig == ASCENDC_SUPER_KERNEL_EARLY_START_MIX_TO_AIV) {
        if ASCEND_IS_AIC {
            wait_flag_dev(AscendC::SYNC_AIC_FLAG);
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIC_AIV_FLAG));
        }
        if ASCEND_IS_AIV {
            wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);
            wait_flag_dev(AscendC::SYNC_AIC_AIV_FLAG);
        }
    } else if constexpr (earlyStartConfig == ASCENDC_SUPER_KERNEL_EARLY_START_MIX_TO_AIC) {
        if ASCEND_IS_AIC {
            wait_flag_dev(AscendC::SYNC_AIC_FLAG);
            wait_flag_dev(AscendC::SYNC_AIV_FLAG);
        }
        if ASCEND_IS_AIV {
            wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG));
        }
    } else if constexpr (earlyStartConfig == ASCENDC_SUPER_KERNEL_EARLY_START_AIC_TO_MIX) {
        if ASCEND_IS_AIC {
            wait_flag_dev(AscendC::SYNC_AIC_FLAG);
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIC_AIV_FLAG));
        }
        if ASCEND_IS_AIV {
            wait_flag_dev(AscendC::SYNC_AIC_AIV_FLAG);
        }
    } else if constexpr (earlyStartConfig == ASCENDC_SUPER_KERNEL_EARLY_START_AIV_TO_MIX) {
        if ASCEND_IS_AIC {
            wait_flag_dev(AscendC::SYNC_AIV_FLAG);
        }
        if ASCEND_IS_AIV {
            wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG));
        }
    } else if constexpr (earlyStartConfig == ASCENDC_SUPER_KERNEL_EARLY_START_MIX_TO_MIX) {
        if ASCEND_IS_AIC {
            wait_flag_dev(AscendC::SYNC_AIC_FLAG);
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIC_AIV_FLAG));
            wait_flag_dev(AscendC::SYNC_AIV_FLAG);
        }
        if ASCEND_IS_AIV {
            wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG));
            wait_flag_dev(AscendC::SYNC_AIC_AIV_FLAG);
        }
    }
#elif defined(__ASCENDC_SUPERKERNEL_EARLY_START_V3)
    WaitPreTaskEndV3Impl(g_super_kernel_early_start_config);
#else
    if constexpr (FORCE) {
        WaitPreTaskEndV3Impl(g_super_kernel_early_start_config);
    }
#endif
}

template <bool isAIVOnly = true>
__aicore__ inline void SyncAllImpl()
{
#if defined(__ASCENDC_SUPERKERNEL_AUTO_SYNC_ALL__)
    if (g_superKernelAutoSyncAllEnable) {
        if (GetBlockIdxImpl() == 0) {
            g_superKernelAutoSyncAllSyncIdx += 1;
            // |    32bits  |   16bits  |   8bits   |   8bits   |
            // |  sync idx  | sync type |   is end  | is valid  |
            *reinterpret_cast<__gm__ uint64_t*>(g_superKernelAutoSyncAllConfigGmAddr) =
                (static_cast<uint64_t>(g_superKernelAutoSyncAllSyncIdx)
                 << SK_AUTO_SYNC_ALL_SYNC_IDX_CONFIG_BIT_OFFSET) |
                (static_cast<uint64_t>(isAIVOnly) << SK_AUTO_SYNC_ALL_SYNC_TYPE_CONFIG_BIT_OFFSET) |
                static_cast<uint64_t>(SK_AUTO_SYNC_ALL_VALID_MAGIC_NUM);
            dcci(
                reinterpret_cast<__gm__ uint64_t*>(g_superKernelAutoSyncAllConfigGmAddr),
                cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
        }
    }
#endif

    PipeBarrier<PIPE_ALL>();
    if constexpr (isAIVOnly) {
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_AIV_ONLY_ALL));
        wait_flag_dev(SYNC_AIV_ONLY_ALL);
        return;
    }
    // trans software sync to hardware sync
    if ASCEND_IS_AIC {
        wait_flag_dev(SYNC_AIV_FLAG);
        ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, SYNC_AIC_FLAG));
        wait_flag_dev(SYNC_AIC_FLAG);
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIC_AIV_FLAG));
        return;
    }
    if ASCEND_IS_AIV {
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIV_FLAG));
        wait_flag_dev(SYNC_AIC_AIV_FLAG);
        return;
    }
}

#if defined(__ASCENDC_SUPERKERNEL_AUTO_SYNC_ALL__)
template <bool isAIVOnly = true>
__aicore__ inline void SuperKernelAutoSyncAllImpl()
{
    PipeBarrier<PIPE_ALL>();
    if constexpr (isAIVOnly) {
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_AIV_ONLY_ALL));
        wait_flag_dev(SYNC_AIV_ONLY_ALL);
        return;
    }
    // trans software sync to hardware sync
    if ASCEND_IS_AIC {
        wait_flag_dev(SYNC_AIV_FLAG);
        ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, SYNC_AIC_FLAG));
        wait_flag_dev(SYNC_AIC_FLAG);
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIC_AIV_FLAG));
        return;
    }
    if ASCEND_IS_AIV {
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIV_FLAG));
        wait_flag_dev(SYNC_AIC_AIV_FLAG);
        return;
    }
}

template <bool isAIVOnly = true>
__aicore__ inline void SuperKernelAutoSyncAllEndImpl()
{
    if (g_superKernelAutoSyncAllEnable) {
        if (GetBlockIdxImpl() == 0) {
            g_superKernelAutoSyncAllSyncIdx += 1;
            // |    32bits  |   16bits  |   8bits   |   8bits   |
            // |  sync idx  | sync type |   is end  | is valid  |
            *reinterpret_cast<__gm__ uint64_t*>(g_superKernelAutoSyncAllConfigGmAddr) =
                (static_cast<uint64_t>(g_superKernelAutoSyncAllSyncIdx)
                 << SK_AUTO_SYNC_ALL_SYNC_IDX_CONFIG_BIT_OFFSET) |
                (static_cast<uint64_t>(1) << SK_AUTO_SYNC_ALL_IS_END_CONFIG_BIT_OFFSET) |
                static_cast<uint64_t>(SK_AUTO_SYNC_ALL_VALID_MAGIC_NUM);
            dcci(
                reinterpret_cast<__gm__ uint64_t*>(g_superKernelAutoSyncAllConfigGmAddr),
                cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
        }
    }
}

__aicore__ inline void SuperKernelAutoSyncAllComplementImpl()
{
    if (!g_superKernelAutoSyncAllEnable) {
        return;
    }
    while (1) {
        SuperKernelAutoSyncAllDcciBarrier();
        dcci(
            reinterpret_cast<__gm__ uint64_t*>(g_superKernelAutoSyncAllConfigGmAddr), cache_line_t::SINGLE_CACHE_LINE,
            dcci_dst_t::CACHELINE_OUT);
        SuperKernelAutoSyncAllDcciBarrier();

        uint64_t tmpAutoSyncAllConfig = *reinterpret_cast<__gm__ uint64_t*>(g_superKernelAutoSyncAllConfigGmAddr);
        uint32_t currentSyncIdx =
            static_cast<uint32_t>(tmpAutoSyncAllConfig >> SK_AUTO_SYNC_ALL_SYNC_IDX_CONFIG_BIT_OFFSET);
        uint16_t syncType =
            static_cast<uint16_t>((tmpAutoSyncAllConfig & 0xffff0000) >> SK_AUTO_SYNC_ALL_SYNC_TYPE_CONFIG_BIT_OFFSET);
        uint8_t isSyncEnd =
            static_cast<uint8_t>((tmpAutoSyncAllConfig & 0xff00) >> SK_AUTO_SYNC_ALL_IS_END_CONFIG_BIT_OFFSET);
        uint8_t validNum = static_cast<uint8_t>(tmpAutoSyncAllConfig & 0xff);
        if (validNum != SK_AUTO_SYNC_ALL_VALID_MAGIC_NUM) {
            trap();
        }

        if (isSyncEnd == 1) {
            // is end
            g_superKernelAutoSyncAllSyncIdx += 1;
            break;
        }

        if (currentSyncIdx > g_superKernelAutoSyncAllSyncIdx) {
            if (syncType == 1) {
                // isAIVOnly
                SuperKernelAutoSyncAllImpl<true>();
            } else {
                // mix
                SuperKernelAutoSyncAllImpl<false>();
            }
            g_superKernelAutoSyncAllSyncIdx += 1;
        }
    }
}
#endif

template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void NotifyEventImpl(uint16_t flagId)
{
    ffts_cross_core_sync(pipe, GetffstMsg(modeId, flagId));
}

template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void WaitEventImpl(uint16_t flagId)
{
    (void)modeId;
    wait_flag_dev(flagId);
}

__aicore__ inline void SetSyncBaseAddrImpl(uint64_t config) { set_ffts_base_addr(config); }

__aicore__ inline void SetSyncBaseAddr(uint64_t config) { SetSyncBaseAddrImpl(config); }
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_SYNC_IMPL_H
#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_SYNC_IMPL_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_SYNC_IMPL_H__
#endif
