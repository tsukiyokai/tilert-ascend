#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
diagnostic=${MLA_TAIL_DIAGNOSTIC:-profile}
case "$diagnostic" in
    profile|dequant) ;;
    *) echo 'MLA_TAIL_DIAGNOSTIC must be profile or dequant.' >&2; exit 2 ;;
esac
out=${1:-$here/build-$diagnostic}
python_bin=${PYTHON_BIN:-${PYTHON:-python3}}
cann_root=${CANN_ROOT:-${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}}}
if [[ -z "$cann_root" || ! -d "$cann_root" ]]; then
    echo 'Set CANN_ROOT or ASCEND_HOME_PATH to a CANN installation.' >&2
    exit 2
fi
device_compiler=${BISHENG:-$cann_root/compiler/ccec_compiler/bin/bisheng}
if [[ ! -x "$device_compiler" ]]; then
    device_compiler=$cann_root/bin/bisheng
fi
if [[ ! -x "$device_compiler" ]]; then
    echo 'CANN bisheng compiler was not found.' >&2
    exit 2
fi
arch_root="$cann_root/$(uname -m)-linux"
if [[ ! -d "$arch_root" ]]; then
    arch_root=$cann_root
fi
includes=(-I "$here" -I "$cann_root/include")
for relative in asc asc/include asc/include/adv_api asc/include/basic_api \
                asc/include/c_api asc/include/simt_api asc/include/utils \
                asc/include/basic_api/reg_compute asc/impl/adv_api \
                asc/impl/basic_api asc/impl/c_api asc/impl/simt_api \
                asc/impl/utils asc/impl/basic_api/reg_compute \
                tikcpp/tikcfw tikcpp/tikcfw/lib tikcpp/tikcfw/impl \
                tikcpp/tikcfw/interface; do
    if [[ -d "$arch_root/$relative" ]]; then
        includes+=(-I "$arch_root/$relative")
    fi
done
mkdir -p "$out"
"$device_compiler" -xcce --npu-arch=dav-2201 -std=c++17 -O2 -fPIC \
    "${includes[@]}" -c "$here/mla_tail_${diagnostic}_kernel.cpp" \
    -o "$out/$diagnostic.o"
"${CXX:-g++}" -shared "$out/$diagnostic.o" \
    -o "$out/mla_tail_$diagnostic.so" \
    -L "$arch_root/lib64" -L "$cann_root/lib64" \
    -lascendcl -lruntime -lunified_dlog -lprofapi
"$python_bin" "$here/profile_provenance.py" --cann "$cann_root" \
    --compiler "$device_compiler" --output "$out/build-manifest.json" \
    --kind "$diagnostic"
echo "built $out/mla_tail_$diagnostic.so"
