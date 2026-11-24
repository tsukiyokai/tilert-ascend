#!/usr/bin/env bash
# Sourced by the standalone and Torch build entry points.
set -euo pipefail
routed_here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
routed_cann=${CANN_ROOT:-${ASCEND_HOME_PATH:-}}
if [[ -z "$routed_cann" ]]; then
  for candidate in /opt/ascend/cann-9.2.0 \
      /usr/local/Ascend/ascend-toolkit/latest; do
    if [[ -x "$candidate/bin/bisheng" ]]; then
      routed_cann=$candidate
      break
    fi
  done
fi
if [[ -z "$routed_cann" ]]; then
  echo 'CANN toolkit not found; set CANN_ROOT to the installed toolkit.' >&2
  exit 2
fi
routed_bisheng=${BISHENG_BIN:-$routed_cann/bin/bisheng}
if [[ ! -x "$routed_bisheng" ]]; then
  echo "AscendC compiler not found: $routed_bisheng" >&2
  exit 2
fi
routed_target=${CANN_TARGET_DIR:-$routed_cann/$(uname -m)-linux}
if [[ ! -d "$routed_target" ]]; then
  echo "CANN target directory not found: $routed_target" >&2
  echo 'Set CANN_TARGET_DIR explicitly when cross-compiling.' >&2
  exit 2
fi
routed_cxx=${CXX:-g++}
routed_arch=${NPU_ARCH:-dav-2201}
routed_device_includes=(-I "$routed_here" -I "$routed_cann/include")
for relative in asc/impl/adv_api asc/impl/basic_api asc/impl/c_api \
    asc/impl/basic_api/reg_compute asc/impl/simt_api asc/impl/utils asc \
    asc/include asc/include/adv_api asc/include/basic_api asc/include/c_api \
    asc/include/basic_api/reg_compute asc/include/simt_api asc/include/utils \
    tikcpp/tikcfw tikcpp/tikcfw/impl tikcpp/tikcfw/interface tikcpp/tikcfw/lib; do
  if [[ -d "$routed_target/$relative" ]]; then
    routed_device_includes+=(-I "$routed_target/$relative")
  fi
done
routed_host_includes=(-I "$routed_here" -I "$routed_cann/include"
  -I "$routed_target/tikcpp/tikcfw"
  -I "$routed_target/tikcpp/tikcfw/lib"
  -I "$routed_target/tikcpp/tikcfw/impl"
  -I "$routed_target/tikcpp/tikcfw/interface")
routed_libs=(-L "$routed_target/lib64" -L "$routed_cann/lib64"
  -ltiling_api -lplatform -lascendcl -lruntime -lunified_dlog -lc_sec
  -lregister -lprofapi)
routed_device_flags=(-xcce "--npu-arch=$routed_arch" -std=c++17 -O2
  -ffp-contract=off -fPIC)
if [[ ${SANITIZE:-0} == 1 ]]; then
  routed_device_flags+=(--cce-enable-sanitizer)
fi
if [[ ${DEBUG:-0} == 1 ]]; then
  routed_device_flags+=(-g)
fi
routed_compile() {
  local logfile=$1
  shift
  if ! "$@" > "$logfile" 2>&1; then
    tail -n 40 "$logfile" >&2
    return 1
  fi
}
