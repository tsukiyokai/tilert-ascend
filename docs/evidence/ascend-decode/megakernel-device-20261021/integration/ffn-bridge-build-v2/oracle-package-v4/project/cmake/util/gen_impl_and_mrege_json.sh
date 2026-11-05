#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

project_path=$1
build_path=$2
vendor_name=customize
if [[ ! -d "$project_path" ]]; then
    echo "[ERROR] No project path is provided"
    exit 1
fi

if [[ ! -d "$build_path" ]]; then
    echo "[ERROR] No build path is provided"
    exit 1
fi

# copy aicpu kernel so operators
if [[ -d "${project_path}/cpukernel/aicpu_kernel_lib" ]]; then
    cp -f ${project_path}/cpukernel/aicpu_kernel_lib/* ${build_path}/makepkg/packages/vendors/$vendor_name/op_impl/cpu/aicpu_kernel/impl
    rm -rf ${project_path}/cpukernel/aicpu_kernel_lib
fi
