#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/build_common.sh"
routed_out=${1:-$routed_here/build}
mkdir -p "$routed_out"
routed_compile "$routed_out/kernel.log" "$routed_bisheng" \
  "${routed_device_flags[@]}" "${routed_device_includes[@]}" \
  -c "$routed_here/routed_int4_kernel.cpp" -o "$routed_out/kernel.o"
routed_compile "$routed_out/tiling.log" "$routed_cxx" -std=c++17 -O2 \
  "${routed_host_includes[@]}" -c "$routed_here/routed_int4_tiling.cpp" \
  -o "$routed_out/tiling.o"
routed_compile "$routed_out/main.log" "$routed_cxx" -std=c++17 -O2 \
  "${routed_host_includes[@]}" -c "$routed_here/routed_int4_main.cpp" \
  -o "$routed_out/main.o"
routed_compile "$routed_out/link.log" "$routed_cxx" \
  "$routed_out/kernel.o" "$routed_out/tiling.o" "$routed_out/main.o" \
  -o "$routed_out/routed_int4" "${routed_libs[@]}"
printf 'Built %s\n' "$routed_out/routed_int4"
