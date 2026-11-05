#!/usr/bin/env bash
# Build the front stage. MEGAKERNEL_FRONT_SPLIT=1 gives the control arm, which
# issues one invocation a projection instead of one over their concatenation.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../routed_int4/build_common.sh"

# Outputs live under MEGAKERNEL_BUILD_ROOT so a read-only or synchronised
# source tree never holds them.
root=${MEGAKERNEL_BUILD_ROOT:-$here}
variant_flags=()
arm=fused
if [[ ${MEGAKERNEL_FRONT_SPLIT:-0} == 1 ]]; then
  variant_flags=(-DMEGAKERNEL_FRONT_SPLIT)
  arm=split
fi
# int8 operands match Prolog's; MEGAKERNEL_FRONT_BF16=1 keeps the earlier
# comparison available.
dtype=int8
if [[ ${MEGAKERNEL_FRONT_BF16:-0} == 1 ]]; then
  variant_flags+=(-DMEGAKERNEL_FRONT_BF16)
  dtype=bf16
fi
out=${1:-$root/build-$dtype-$arm}

python_bin=${PYTHON_BIN:-python3}
site=${SITE_PACKAGES:-$("$python_bin" -c \
  'import pathlib, torch; print(pathlib.Path(torch.__file__).parent.parent)')}
abi=${TORCH_CXX11_ABI:-$("$python_bin" -c \
  'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')}
torch_includes=(-I "$site/torch/include"
  -I "$site/torch/include/torch/csrc/api/include"
  -I "$site/torch_npu/include")

mkdir -p "$out"
routed_compile "$out/kernel.log" "$routed_bisheng" \
  "${routed_device_flags[@]}" -I "$here" "${routed_device_includes[@]}" \
  "${variant_flags[@]}" -c "$here/front_kernel.cpp" -o "$out/kernel.o"
routed_compile "$out/tiling.log" "$routed_cxx" -std=c++17 -O2 -fPIC \
  -I "$here" "${routed_host_includes[@]}" "${variant_flags[@]}" \
  -c "$here/front_tiling.cpp" -o "$out/tiling.o"
routed_compile "$out/binding.log" "$routed_cxx" -std=c++17 -O2 -fPIC \
  "-D_GLIBCXX_USE_CXX11_ABI=$abi" "${torch_includes[@]}" \
  -I "$here" "${routed_host_includes[@]}" "${variant_flags[@]}" \
  -c "$here/front_binding.cpp" -o "$out/binding.o"
routed_compile "$out/link.log" "$routed_cxx" -shared \
  "$out/kernel.o" "$out/tiling.o" "$out/binding.o" \
  -o "$out/megakernel_front.so" "${routed_libs[@]}" \
  -L "$site/torch/lib" -L "$site/torch_npu/lib" \
  -ltorch -ltorch_cpu -lc10 -ltorch_npu
printf 'Built %s\n' "$out/megakernel_front.so"
