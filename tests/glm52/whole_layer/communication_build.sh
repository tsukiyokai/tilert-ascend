#!/usr/bin/env bash
set -euo pipefail
communication_here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
communication_cann=${CANN_ROOT:-${ASCEND_HOME_PATH:-/opt/ascend/cann-9.2.0}}
communication_target=${CANN_TARGET_DIR:-$communication_cann/$(uname -m)-linux}
communication_shmem=${SHMEM_SOURCE:?Set SHMEM_SOURCE to the matching SHMEM source tree.}
communication_build=${BUILD_DIR:-$communication_here/communication_build}
communication_bisheng=${BISHENG_BIN:-$communication_cann/bin/bisheng}
communication_cxx=${CXX:-g++}
communication_includes=(-I "$communication_here" -I "$communication_cann/include"
  -I "$communication_shmem/include" -I "$communication_shmem/src/device")
for relative in asc/impl/adv_api asc/impl/basic_api asc/impl/c_api \
    asc/impl/basic_api/reg_compute asc/impl/simt_api asc/impl/utils asc \
    asc/include asc/include/adv_api asc/include/basic_api asc/include/c_api \
    asc/include/basic_api/reg_compute asc/include/simt_api asc/include/utils \
    tikcpp/tikcfw tikcpp/tikcfw/impl tikcpp/tikcfw/interface tikcpp/tikcfw/lib; do
  if [[ -d "$communication_target/$relative" ]]; then
    communication_includes+=(-I "$communication_target/$relative")
  fi
done
mkdir -p -- "$communication_build"
{
  date --utc --iso-8601=seconds
  "$communication_bisheng" --version
  sha256sum "$communication_here"/communication*.h \
    "$communication_here"/communication*.cpp \
    "$communication_here/communication_build.sh"
} > "$communication_build/identity.txt"
if [[ -f "$communication_shmem/SOURCE_COMMIT" ]]; then
  cat "$communication_shmem/SOURCE_COMMIT" >> "$communication_build/identity.txt"
fi
find "$communication_shmem/include" "$communication_shmem/src/device" \
  "$communication_shmem/src/host_device" -type f -print0 \
  | sort -z | xargs -0 sha256sum > "$communication_build/shmem_headers.sha256"
if ! "$communication_bisheng" -xcce --npu-arch=dav-2201 -std=c++17 -O2 \
    -ffp-contract=off -fPIC "${communication_includes[@]}" \
    -c "$communication_here/communication_probe.cpp" \
    -o "$communication_build/communication_probe.o" \
    > "$communication_build/device_compile.log" 2>&1; then
  tail -n 60 "$communication_build/device_compile.log" >&2
  exit 1
fi
if ! "$communication_cxx" -std=c++17 -Wall -Wextra -Werror \
    -Wno-missing-field-initializers -fPIC -I "$communication_cann/include" \
    -I "$communication_shmem/include" -c \
    "$communication_here/communication_host.cpp" \
    -o "$communication_build/communication_host.o" \
    > "$communication_build/host_compile.log" 2>&1; then
  tail -n 60 "$communication_build/host_compile.log" >&2
  exit 1
fi
if ! "$communication_cxx" -std=c++17 -O2 -fPIC \
    "${communication_includes[@]}" -c \
    "$communication_here/communication_probe_host.cpp" \
    -o "$communication_build/communication_probe_host.o" \
    > "$communication_build/probe_host_compile.log" 2>&1; then
  tail -n 60 "$communication_build/probe_host_compile.log" >&2
  exit 1
fi
communication_shmem_library=${SHMEM_LIBRARY:-}
if [[ -z "$communication_shmem_library" ]]; then
  printf '%s\n' 'Communication objects compiled; set SHMEM_LIBRARY to link the runnable probe.'
  exit 0
fi
if ! "$communication_cxx" -shared \
    "$communication_build/communication_probe.o" \
    "$communication_build/communication_host.o" \
    "$communication_build/communication_probe_host.o" \
    -o "$communication_build/communication_probe.so" \
    "$communication_shmem_library" \
    -L "$communication_target/lib64" -L "$communication_cann/lib64" \
    -ltiling_api -lplatform -lascendcl -lruntime -lunified_dlog -lc_sec \
    -lregister -lprofapi > "$communication_build/link.log" 2>&1; then
  tail -n 60 "$communication_build/link.log" >&2
  exit 1
fi
sha256sum "$communication_shmem_library" \
  "$communication_build/communication_probe.so" \
  >> "$communication_build/identity.txt"
printf 'Built %s\n' "$communication_build/communication_probe.so"
