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
 * \file tbuf_pool_l0_base.h
 * \brief
 */
#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message( \
    "impl/adv_api/detail/matmul/resource/l0_buffer/tbuf_pool_l0_base.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"adv_api/matmul/matmul.h\"\" and use public functions or variables defined in interface headers files.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_RESOURCE_L0_BUFFER_TBUF_POOL_L0_BASE_H__
#endif

#ifndef IMPL_MATMUL_RESOURCE_L0_BUFFER_TBUF_POOL_L0_BASE_H
#define IMPL_MATMUL_RESOURCE_L0_BUFFER_TBUF_POOL_L0_BASE_H

#include "tbuf_pool_l0_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    TBufPoolL0Base is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    TBufPoolL0Base is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, typename A_TYPE, typename B_TYPE, const auto& MM_CFG>
class TBufPoolL0Base {
    MATMUL_USE_MODULE(MatmulShapeTiling);

public:
    __aicore__ inline TBufPoolL0Base() = default;
    __aicore__ inline ~TBufPoolL0Base() = default;
    __aicore__ inline void Init()
    {
        bool isL0Db;
        if constexpr (NormInitScene<MM_CFG> && Impl::Detail::MatmulFeatureTrait<MM_CFG>().IsSupportMNL0DB()) {
            isL0Db = true;
        } else if constexpr (
            IsBasic(MM_CFG) && NormInitScene<MM_CFG> &&
            Impl::Detail::MatmulFeatureTrait<MM_CFG>().IsSupportLoad2dV2()) {
            isL0Db = false;
        } else {
            const auto& tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
            isL0Db = (tiling.GetDbL0A() - 1) & (tiling.GetDbL0B() - 1);
        }

        useL0PingPong_ = static_cast<uint16_t>(isL0Db);
        GetTPipePtr()->InitBuffer(l0aBuf_, L0AUF_SIZE);
        GetTPipePtr()->InitBuffer(l0bBuf_, L0BUF_SIZE);
    }

    __aicore__ inline void SetDBFlag(bool isL0Db = true) { useL0PingPong_ = static_cast<uint16_t>(isL0Db); }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline TBufPoolL0Base& Allocate()
    {
        if constexpr (!IS_INTRA_BLOCK) {
            WaitFlag<HardEvent::M_MTE1>(l0PingPongFlag_);
        }
        return *this;
    }

    template <TPosition Pos, typename T, bool IS_INTRA_BLOCK = false>
    __aicore__ inline LocalTensor<T> GetBuffer(uint8_t subIdx = 0)
    {
        LocalTensor<T> tempTensor;
        if constexpr (Pos == TPosition::A2) {
            tempTensor = l0aBuf_.Get<T>();
            if (l0PingPongFlag_ != 0) {
                if constexpr (IsSupportB4<T>()) {
                    tempTensor = tempTensor[L0AUF_SIZE / sizeof(T)];
                } else {
                    tempTensor = tempTensor[L0AUF_SIZE / 2 / sizeof(T)];
                }
            }
        } else {
            tempTensor = l0bBuf_.Get<T>();
            if constexpr (IS_INTRA_BLOCK) {
                if (subIdx != 0) {
                    tempTensor = tempTensor[L0BUF_SIZE / 2 / sizeof(T)];
                }
            } else {
                if (l0PingPongFlag_ != 0) {
                    if constexpr (IsSupportB4<T>()) {
                        tempTensor = tempTensor[L0BUF_SIZE / sizeof(T)];
                    } else {
                        tempTensor = tempTensor[L0BUF_SIZE / 2 / sizeof(T)];
                    }
                }
            }
        }
        return tempTensor;
    }

    template <TPosition Pos>
    __aicore__ inline bool Hit(uint32_t pos = 0)
    {
        return false;
    }

    __aicore__ inline void ResetCache() {}

    __aicore__ inline void EnQue() { SetFlag<HardEvent::MTE1_M>(l0PingPongFlag_); }

    __aicore__ inline void DeQue() { WaitFlag<HardEvent::MTE1_M>(l0PingPongFlag_); }

    __aicore__ inline void Free()
    {
        SetFlag<HardEvent::M_MTE1>(l0PingPongFlag_);
        l0PingPongFlag_ = useL0PingPong_ - l0PingPongFlag_;
    }

protected:
    TBuf<TPosition::A2> l0aBuf_;
    TBuf<TPosition::B2> l0bBuf_;
    uint16_t l0PingPongFlag_{0};
    uint16_t useL0PingPong_{1};
};

} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif // IMPL_MATMUL_RESOURCE_L0_BUFFER_TBUF_POOL_L0_BASE_H

#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_RESOURCE_L0_BUFFER_TBUF_POOL_L0_BASE_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_RESOURCE_L0_BUFFER_TBUF_POOL_L0_BASE_H__
#endif
