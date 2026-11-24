#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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

import json
import sys
import os


def read_json(file):
    with open(file, "r") as fd:
        config = json.load(fd)
    return config


def get_config_opts(file):
    config = read_json(file)

    src_dir = os.path.abspath(os.path.dirname(file))
    opts = ""

    for conf in config:
        if conf == "configurePresets":
            for node in config[conf]:
                macros = node.get("cacheVariables")
                if macros is not None:
                    for key in macros:
                        opts += "-D{}={} ".format(key, macros[key]["value"])

    opts = opts.replace("${sourceDir}", src_dir)
    print(opts)


if __name__ == "__main__":
    get_config_opts(sys.argv[1])
