#!/usr/bin/env bash
set -eo pipefail
source /workspace/environment/activate.sh
set -u
artifact=/workspace/offline/sfa-c1-bounds-20261019-codex
arm=${1:?original or bounds}
case "$arm" in original|bounds) ;; *) exit 2;; esac
snapshot=$artifact/source/csrc
overlay=$artifact/$arm/vendors/custom_transformer
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=
export CPATH="$snapshot/common/include:$snapshot/third_party/catlass/include${CPATH:+:$CPATH}"
export ASCEND_CUSTOM_OPP_PATH="$overlay${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export HI_PYTHON=python3 TILINGKEY_PAR_COMPILE=1 BIN_FILENAME_HASHED=1
opc "$overlay/op_impl/ai_core/tbe/custom_transformer_impl/dynamic/sparse_flash_attention.py" \
  --main_func=sparse_flash_attention --input_param="$artifact/input-param.json" \
  --soc_version=Ascend910_9391 --output="$artifact/$arm/kernel" \
  --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic
