#!/usr/bin/env bash
set -euo pipefail

shared_here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
shared_out=${1:-$shared_here/build}
shared_contract=${2:-scalar}
case "$shared_contract" in
    scalar) shared_quant_flag=-DSHARED_W8_GROUPED_QUANT=0 ;;
    grouped) shared_quant_flag=-DSHARED_W8_GROUPED_QUANT=1 ;;
    *) echo 'Activation contract must be scalar or grouped.' >&2; exit 2 ;;
esac
shared_cann=${CANN_ROOT:-${ASCEND_HOME_PATH:-}}
if [[ -z "$shared_cann" || ! -d "$shared_cann" ]]; then
    echo 'Set CANN_ROOT to the matching CANN installation.' >&2
    exit 2
fi
shared_arch=$shared_cann/$(uname -m)-linux
shared_bisheng=${BISHENG:-$shared_cann/compiler/ccec_compiler/bin/bisheng}
if [[ ! -x "$shared_bisheng" ]]; then
    shared_bisheng=$shared_cann/bin/bisheng
fi
mkdir -p "$shared_out/sources/shared_w8" "$shared_out/sources/mla_tail"
cp "$shared_here"/shared_w8*.h "$shared_here"/shared_w8*.cpp \
   "$shared_here/build.sh" "$shared_out/sources/shared_w8/"
cp "$shared_here/../mla_tail/mla_tail.h" \
   "$shared_here/../mla_tail/mla_tail_native.h" \
   "$shared_here/../mla_tail/mla_tail_dequant.h" \
   "$shared_out/sources/mla_tail/"
shared_includes=(-I "$shared_out/sources/shared_w8" -I "$shared_cann/include")
for relative in asc asc/include asc/include/adv_api asc/include/basic_api \
    asc/include/c_api asc/include/simt_api asc/include/utils \
    asc/include/basic_api/reg_compute asc/impl/adv_api asc/impl/basic_api \
    asc/impl/c_api asc/impl/simt_api asc/impl/utils \
    asc/impl/basic_api/reg_compute tikcpp/tikcfw tikcpp/tikcfw/lib \
    tikcpp/tikcfw/impl tikcpp/tikcfw/interface; do
    if [[ -d "$shared_arch/$relative" ]]; then
        shared_includes+=(-I "$shared_arch/$relative")
    fi
done
"$shared_bisheng" -xcce --npu-arch=dav-2201 -std=c++17 -O2 -fPIC \
    -ffp-contract=off "$shared_quant_flag" "${shared_includes[@]}" \
    -c "$shared_out/sources/shared_w8/shared_w8_kernel.cpp" \
    -o "$shared_out/kernel.o"
"${CXX:-g++}" -std=c++17 -O2 -fPIC "$shared_quant_flag" "${shared_includes[@]}" \
    -c "$shared_out/sources/shared_w8/shared_w8_host.cpp" -o "$shared_out/host.o"
"${CXX:-g++}" -shared "$shared_out/kernel.o" "$shared_out/host.o" \
    -o "$shared_out/shared_w8.so" \
    -L "$shared_arch/lib64" -L "$shared_cann/lib64" \
    -ltiling_api -lplatform -lascendcl -lruntime -lunified_dlog -lc_sec \
    -lregister -lprofapi
"${PYTHON_BIN:-python3}" - "$shared_out" "$shared_cann" "$shared_bisheng" "$shared_contract" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

out, cann, compiler = map(Path, sys.argv[1:4])
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "cann_root": str(cann),
    "activation_contract": sys.argv[4],
    "compiler": str(compiler),
    "compiler_version": subprocess.check_output(
        [str(compiler), "--version"], text=True),
    "source_sha256": {str(path.relative_to(out)): digest(path)
                      for path in sorted((out / "sources").rglob("*"))
                      if path.is_file()},
    "library_sha256": digest(out / "shared_w8.so"),
}
(out / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
printf 'Built %s\n' "$shared_out/shared_w8.so"
