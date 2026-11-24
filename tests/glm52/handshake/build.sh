#!/usr/bin/env bash
# Three translation units, two compilers. `bisheng` owns the kernel because
# only it understands `<<<>>>` and the device intrinsics; `g++` owns the host
# tiling because `matmul_tiling.h` pulls in the operator registry's types.
# Link with `g++`: every device symbol the kernel object needs comes from
# `ascendcl` and `runtime`. Run the result under `set_env.sh`.
set -uo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${1:-/tmp/handshake-build}
H=/opt/ascend/cann-9.2.0/aarch64-linux

mkdir -p "$out"
device_incs=""
for d in asc/impl/adv_api asc/impl/basic_api asc/impl/c_api \
         asc/impl/basic_api/reg_compute asc/impl/simt_api asc/impl/utils asc \
         asc/include asc/include/adv_api asc/include/basic_api \
         asc/include/c_api asc/include/basic_api/reg_compute \
         asc/include/simt_api asc/include/utils tikcpp/tikcfw \
         tikcpp/tikcfw/impl tikcpp/tikcfw/interface tikcpp/tikcfw/lib; do
  device_incs="$device_incs -I $H/$d"
done
device_incs="$device_incs -I /opt/ascend/cann-9.2.0/include -I $here"
# `mssanitizer` refuses a kernel built without this, with "No active sanitizer
# tool on kernel". Opt in with SANITIZE=1; it changes the binary, so never
# time an instrumented build.
sanitize=${SANITIZE:+--cce-enable-sanitizer}
# DEBUG=1 adds device debug info so `msdebug` can break inside the kernel.
# ARCH overrides the architecture flag: `--cce-aicore-arch=dav-c220` silently
# drops `__schedmode__`, while `--npu-arch=dav-2201` makes it reach the
# kernel metadata, which the runtime log reports as schemMode.
debug=${DEBUG:+-g}
arch=${ARCH:---npu-arch=dav-2201}
host_incs="-I $H/tikcpp/tikcfw -I $H/tikcpp/tikcfw/lib -I $H/tikcpp/tikcfw/impl"
host_incs="$host_incs -I $H/tikcpp/tikcfw/interface"
host_incs="$host_incs -I /opt/ascend/cann-9.2.0/include -I $here"

/opt/ascend/cann-9.2.0/bin/bisheng -xcce "$arch" \
  -std=c++17 -O2 ${sanitize:-} ${debug:-} $device_incs \
  -c "$here/handshake_kernel.cpp" \
  -o "$out/kernel.o" > "$out/kernel_build.log" 2>&1
g++ -std=c++17 -O2 $host_incs -c "$here/handshake_main.cpp" \
  -o "$out/main.o" > "$out/main_build.log" 2>&1
g++ "$out/kernel.o" "$out/main.o" \
  -L"$H/lib64" -L/opt/ascend/cann-9.2.0/lib64 \
  -ltiling_api -lplatform -lascendcl -lruntime -lunified_dlog -lc_sec \
  -lregister -lprofapi -o "$out/handshake" > "$out/link.log" 2>&1

[ -x "$out/handshake" ] && echo "built $out/handshake" || {
  echo "build failed"
  for log in kernel_build tiling_build main_build link; do
    [ -s "$out/$log.log" ] && { echo "--- $log"; grep -E "error" "$out/$log.log" | head -6; }
  done
}
