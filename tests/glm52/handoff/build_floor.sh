#!/usr/bin/env bash
# One die, no HCCL, no model. Build with the same include roots the SHMEM
# examples need: the device headers live under the source tree, not the
# install tree.
set -uo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${1:-/tmp/handoff-build}
H=/opt/ascend/cann-9.2.0/aarch64-linux

mkdir -p "$out"
incs=""
for d in asc/impl/adv_api asc/impl/basic_api asc/impl/c_api \
         asc/impl/basic_api/reg_compute asc/impl/simt_api asc/impl/utils asc \
         asc/include asc/include/adv_api asc/include/basic_api \
         asc/include/c_api asc/include/basic_api/reg_compute \
         asc/include/simt_api asc/include/utils tikcpp/tikcfw \
         tikcpp/tikcfw/impl tikcpp/tikcfw/interface tikcpp/tikcfw/lib; do
  incs="$incs -I $H/$d"
done
incs="$incs -I /opt/ascend/cann-9.2.0/include -I $here"

/opt/ascend/cann-9.2.0/bin/bisheng -xcce --cce-aicore-arch=dav-c220 \
  -std=c++17 -O2 $incs \
  "$here/floor_kernel.cpp" "$here/floor_main.cpp" \
  -L/opt/ascend/cann-9.2.0/aarch64-linux/lib64 -lascendcl -lruntime \
  -o "$out/floor" 2>&1 | grep -vE "macro-redefined|^In file|^ +from" | head -20

[ -x "$out/floor" ] && echo "built $out/floor" || echo "build failed"
