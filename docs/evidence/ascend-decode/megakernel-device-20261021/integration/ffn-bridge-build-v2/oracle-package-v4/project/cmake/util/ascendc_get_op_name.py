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
"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""

import configparser
import argparse


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ini-file", help="op info ini.")
    return parser.parse_args()


if __name__ == "__main__":
    args = args_parse()
    op_config = configparser.ConfigParser()
    op_config.read(args.ini_file)
    for section in op_config.sections():
        print(section, end="-")
        print(op_config.get(section, "opFile.value"), end="\n")
