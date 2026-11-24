#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/build_common.sh"
routed_out=${1:-$routed_here/build-reference}
routed_python=${PYTHON_BIN:-python3}
routed_site=${SITE_PACKAGES:-}
if [[ -z "$routed_site" ]]; then
  routed_site=$("$routed_python" -c \
    'import pathlib, torch; print(pathlib.Path(torch.__file__).parent.parent)')
fi
routed_abi=${TORCH_CXX11_ABI:-}
if [[ -z "$routed_abi" ]]; then
  routed_abi=$("$routed_python" -c \
    'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
fi
mkdir -p "$routed_out"
routed_compile "$routed_out/compile.log" "$routed_cxx" -std=c++17 -O2 -fPIC \
  "-D_GLIBCXX_USE_CXX11_ABI=$routed_abi" \
  -I "$routed_site/torch/include" \
  -I "$routed_site/torch/include/torch/csrc/api/include" \
  -I "$routed_site/torch_npu/include" "${routed_host_includes[@]}" \
  -c "$routed_here/nz_reference.cpp" -o "$routed_out/nz_reference.o"
routed_compile "$routed_out/link.log" "$routed_cxx" -shared \
  "$routed_out/nz_reference.o" -o "$routed_out/nz_reference.so" \
  -L "$routed_target/lib64" -L "$routed_cann/lib64" -lopapi -lascendcl \
  -L "$routed_site/torch/lib" -L "$routed_site/torch_npu/lib" \
  -ltorch -ltorch_cpu -lc10 -ltorch_npu
printf 'Built %s\n' "$routed_out/nz_reference.so"
