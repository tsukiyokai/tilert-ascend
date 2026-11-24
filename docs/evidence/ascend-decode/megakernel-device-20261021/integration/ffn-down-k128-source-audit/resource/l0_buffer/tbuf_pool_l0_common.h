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
 * \file tbuf_pool_l0_common.h
 * \brief
 */
#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message( \
    "impl/adv_api/detail/matmul/resource/l0_buffer/tbuf_pool_l0_common.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"adv_api/matmul/matmul.h\"\" and use public functions or variables defined in interface headers files.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_RESOURCE_L0_BUFFER_TBUF_POOL_L0_COMMON_H__
#endif

#ifndef IMPL_MATMUL_RESOURCE_L0_BUFFER_TBUF_POOL_L0_COMMON_H
#define IMPL_MATMUL_RESOURCE_L0_BUFFER_TBUF_POOL_L0_COMMON_H

#include "tbuf_pool_l0_intf.h"
#include "tbuf_pool_l0_base.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    TBufPoolL0 is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    TBufPoolL0 is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, typename A_TYPE, typename B_TYPE, const auto& MM_CFG>
class TBufPoolL0<
    IMPL, A_TYPE, B_TYPE, MM_CFG,
    enable_if_t<
        !MatmulFeatureTrait<MM_CFG>::IsNeedUB() &&
        (DoMatmulMDL(MM_CFG) || isNormEnableScheduler<A_TYPE, MM_CFG> || IsBmmEnableScheduler<A_TYPE, MM_CFG> ||
         DoMatmulSpecialMDL(MM_CFG) || IsBasicBlockEnable<MM_CFG> || DoMatmulIBShareNorm(MM_CFG) ||
         IsIntrablock<MM_CFG>)&&!IsA2B2Shared(MM_CFG) &&
        !IsL0Cache<A_TYPE, MM_CFG>()>> : public TBufPoolL0Base<IMPL, A_TYPE, B_TYPE, MM_CFG> {
public:
    using BASE_MODULE = AscendC::Impl::Detail::TBufPoolL0Base<IMPL, A_TYPE, B_TYPE, MM_CFG>;
    __aicore__ inline TBufPoolL0() = default;
    __aicore__ inline ~TBufPoolL0() = default;
};

} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif // IMPL_MATMUL_RESOURCE_L0_BUFFER_TBUF_POOL_L0_COMMON_H

#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_RESOURCE_L0_BUFFER_TBUF_POOL_L0_COMMON_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_DETAIL_MATMUL_RESOURCE_L0_BUFFER_TBUF_POOL_L0_COMMON_H__
#endif
