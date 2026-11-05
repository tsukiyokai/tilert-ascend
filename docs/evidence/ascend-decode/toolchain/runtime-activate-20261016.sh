#!/usr/bin/env bash

if declare -F deactivate >/dev/null; then
    deactivate
fi
unset ASCEND_TOOLKIT_HOME ASCEND_HOME_PATH ASCEND_AICPU_PATH ASCEND_OPP_PATH
unset ASCEND_CUSTOM_OPP_PATH TOOLCHAIN_HOME PYTHONPATH CMAKE_PREFIX_PATH
unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT
export PATH=/opt/openEuler/gcc-toolset-12/root/usr/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/opt/openEuler/gcc-toolset-12/root/usr/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
export PYTHONNOUSERSITE=1
source /opt/ascend/set_env.sh
source /opt/ascend/nnal/atb/set_env.sh --cxx_abi=1
source /opt/trt-asc/venv/bin/activate
export PYTHONPATH="${PYTHONPATH%:}"
export CC=/opt/openEuler/gcc-toolset-12/root/usr/bin/gcc
export CXX=/opt/openEuler/gcc-toolset-12/root/usr/bin/g++
export ASCEND_CUSTOM_OPP_PATH=/workspace/src/vllm-ascend-unified/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
