#!/usr/bin/env bash
# Run inside a3-84:trt-asc. No install target or device compilation.
set -e -o pipefail
source /workspace/environment/activate.sh
set -u
snapshot=${1:?prepared isolated csrc snapshot}
build=${2:-$snapshot/build}
cmake -S "$snapshot" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/openEuler/gcc-toolset-12/root/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/opt/openEuler/gcc-toolset-12/root/usr/bin/g++ \
  -DBUILD_OPEN_PROJECT=ON -DVENDOR_NAME=sfa_c1_fullhost \
  -DASCEND_OP_NAME="$(cat /workspace/offline/sfa-c1-fullhost-20261019-codex/op-selectors.txt)" \
  -DASCEND_COMPUTE_UNIT=ascend910_93 \
  -DCANN_3RD_LIB_PATH="$snapshot/third_party" \
  -DCUSTOM_ASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  -DCHECK_COMPATIBLE=true -DENABLE_BUILT_IN=OFF \
  -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=OFF \
  -DENABLE_BUILD_PKG=ON -DCMAKE_INSTALL_PREFIX="$snapshot/output"
cmake --build "$build" --target cust_opmaster -j8
sha256sum "$build/libcust_opmaster_rt2.0.so"
