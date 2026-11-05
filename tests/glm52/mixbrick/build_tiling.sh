#!/usr/bin/env bash
# Host-only: the cube tiling a hand-written MIX kernel needs.
#
# The link line is the point of this script. `tiling_api` is a static archive
# whose `TilingDef` symbols live in `register`, its logging in `unified_dlog`
# and its `memcpy_s` in `c_sec`; none of that is visible from the headers.
# Run the result under `set_env.sh`, or `libruntime_common` fails to resolve
# `ErrorManager::ATCReportErrMessage`.
set -uo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${1:-/tmp/mixbrick-build}
H=/opt/ascend/cann-9.2.0/aarch64-linux

mkdir -p "$out"
incs="-I $H/tikcpp/tikcfw -I $H/tikcpp/tikcfw/lib -I $H/tikcpp/tikcfw/impl"
incs="$incs -I $H/tikcpp/tikcfw/interface -I /opt/ascend/cann-9.2.0/include"

g++ -std=c++17 -O2 $incs "$here/check_tiling.cpp" \
  -L"$H/lib64" -L/opt/ascend/cann-9.2.0/lib64 \
  -ltiling_api -lplatform -lascendcl -lruntime -lunified_dlog -lc_sec \
  -lregister -o "$out/check_tiling" > "$out/tiling_build.log" 2>&1

[ -x "$out/check_tiling" ] && echo "built $out/check_tiling" \
  || { echo "build failed"; tail -5 "$out/tiling_build.log"; }
