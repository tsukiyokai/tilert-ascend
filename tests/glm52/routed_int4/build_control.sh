#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/build_common.sh"
routed_out=${1:-$routed_here/build-control}
ROUTED_CONTROL=1 bash "$routed_here/build_so.sh" "$routed_out"
"$routed_bisheng" --version > "$routed_out/compiler.txt"
sha256sum "$routed_here"/routed_int4_{binding.cpp,contract.h,dispatch.h,kernel.cpp,kernel.h,tiling.cpp,trace.h,profile.h} \
  "$routed_here"/{build_common.sh,build_so.sh,build_control.sh,check.py} \
  "$routed_out"/{kernel.o,tiling.o,binding.o,routed_int4_torch.so} \
  > "$routed_out/identity.sha256"
