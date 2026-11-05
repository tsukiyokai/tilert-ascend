#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/build_common.sh"
routed_out=${1:-$routed_here/build-pipeline}
ROUTED_ROW_BALANCED=1 ROUTED_SHARED_PIPELINE=1 \
  bash "$routed_here/build_so.sh" "$routed_out"
"$routed_bisheng" --version > "$routed_out/compiler.txt"
sha256sum "$routed_here"/routed_int4_{binding.cpp,contract.h,dispatch.h,kernel.cpp,kernel.h,tiling.cpp,trace.h,profile.h} \
  "$routed_here"/routed_int4_pipeline{.h,_native.h,_tiling.cpp} \
  "$routed_here"/{build_common.sh,build_so.sh,build_pipeline.sh} \
  "$routed_here"/../shared_w8/{shared_w8.h,shared_w8_native.h} \
  "$routed_here"/../mla_tail/{mla_tail_dequant.h,mla_tail_native.h,mla_tail.h} \
  "$routed_out"/{kernel.o,tiling.o,shared_tiling.o,binding.o,routed_int4_torch.so} \
  > "$routed_out/identity.sha256"
