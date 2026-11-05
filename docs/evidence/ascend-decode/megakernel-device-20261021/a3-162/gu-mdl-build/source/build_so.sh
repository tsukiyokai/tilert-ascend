#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/build_common.sh"
routed_variant_flags=()
case ${ROUTED_PARTIALS:-int32} in
  int32) routed_default_out=$routed_here/build ;;
  fp16)
    routed_variant_flags=(-DROUTED_FIXPIPE_HALF)
    routed_default_out=$routed_here/build-fixpipe
    ;;
  *) echo 'ROUTED_PARTIALS must be int32 or fp16' >&2; exit 2 ;;
esac
if [[ ${ROUTED_CONTROL:-0} == 1 ]]; then
  if [[ ${ROUTED_PARTIALS:-int32} != int32 ]]; then
    echo 'ROUTED_CONTROL requires INT32 partials' >&2
    exit 2
  fi
  routed_variant_flags+=(-DROUTED_CONTROL)
  routed_default_out=$routed_here/build-control
fi
if [[ ${ROUTED_ROW_BALANCED:-0} == 1 ]]; then
  if [[ ${ROUTED_PARTIALS:-int32} != int32 || ${ROUTED_CONTROL:-0} != 0 ]]; then
    echo 'ROUTED_ROW_BALANCED requires INT32 partials and no control variant' >&2
    exit 2
  fi
  routed_variant_flags+=(-DROUTED_ROW_BALANCED)
  if [[ ${ROUTED_BALANCED_NAMESPACE:-1} == 1 ]]; then
    routed_variant_flags+=(-DROUTED_BALANCED_NAMESPACE)
    routed_default_out=$routed_here/build-balanced
  else
    routed_default_out=$routed_here/build-balanced-promoted
  fi
fi
if [[ ${ROUTED_SHARED_PIPELINE:-0} == 1 ]]; then
  if [[ ${ROUTED_PARTIALS:-int32} != int32 || ${ROUTED_CONTROL:-0} != 0 \
        || ${ROUTED_ROW_BALANCED:-0} != 1 ]]; then
    echo 'ROUTED_SHARED_PIPELINE requires the balanced INT32 schedule' >&2
    exit 2
  fi
  routed_variant_flags+=(-DROUTED_SHARED_PIPELINE -DSHARED_W8_GROUPED_QUANT=1)
  routed_default_out=$routed_here/build-pipeline
fi
if [[ ${ROUTED_GU_MDL:-0} == 1 ]]; then
  if [[ ${ROUTED_PARTIALS:-int32} != int32 || ${ROUTED_CONTROL:-0} != 0 \
        || ${ROUTED_ROW_BALANCED:-0} != 1 \
        || ${ROUTED_SHARED_PIPELINE:-0} != 0 ]]; then
    echo 'ROUTED_GU_MDL requires the separate balanced INT32 kernel' >&2
    exit 2
  fi
  routed_variant_flags+=(-DROUTED_GU_MDL)
  routed_default_out=$routed_here/build-gu-mdl
fi
routed_out=${1:-$routed_default_out}
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
for required in "$routed_site/torch/include" "$routed_site/torch_npu/include"; do
  if [[ ! -d "$required" ]]; then
    echo "Torch extension headers not found: $required" >&2
    exit 2
  fi
done
routed_torch_includes=(-I "$routed_site/torch/include"
  -I "$routed_site/torch/include/torch/csrc/api/include"
  -I "$routed_site/torch_npu/include")
mkdir -p "$routed_out"
routed_compile "$routed_out/kernel.log" "$routed_bisheng" \
  "${routed_device_flags[@]}" "${routed_device_includes[@]}" \
  "${routed_variant_flags[@]}" \
  -c "$routed_here/routed_int4_kernel.cpp" -o "$routed_out/kernel.o"
routed_compile "$routed_out/tiling.log" "$routed_cxx" -std=c++17 -O2 -fPIC \
  "${routed_host_includes[@]}" "${routed_variant_flags[@]}" \
  -c "$routed_here/routed_int4_tiling.cpp" \
  -o "$routed_out/tiling.o"
routed_extra_objects=()
if [[ ${ROUTED_SHARED_PIPELINE:-0} == 1 ]]; then
  routed_compile "$routed_out/shared_tiling.log" "$routed_cxx" -std=c++17 -O2 -fPIC \
    "${routed_host_includes[@]}" "${routed_variant_flags[@]}" \
    -c "$routed_here/routed_int4_pipeline_tiling.cpp" \
    -o "$routed_out/shared_tiling.o"
  routed_extra_objects+=("$routed_out/shared_tiling.o")
fi
routed_compile "$routed_out/binding.log" "$routed_cxx" -std=c++17 -O2 -fPIC \
  "-D_GLIBCXX_USE_CXX11_ABI=$routed_abi" "${routed_torch_includes[@]}" \
  "${routed_host_includes[@]}" "${routed_variant_flags[@]}" \
  -c "$routed_here/routed_int4_binding.cpp" \
  -o "$routed_out/binding.o"
routed_compile "$routed_out/link.log" "$routed_cxx" -shared \
  "$routed_out/kernel.o" "$routed_out/tiling.o" "$routed_out/binding.o" \
  "${routed_extra_objects[@]}" \
  -o "$routed_out/routed_int4_torch.so" "${routed_libs[@]}" \
  -L "$routed_site/torch/lib" -L "$routed_site/torch_npu/lib" \
  -ltorch -ltorch_cpu -lc10 -ltorch_npu
printf 'Built %s\n' "$routed_out/routed_int4_torch.so"
