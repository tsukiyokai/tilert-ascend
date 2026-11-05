#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${1:-$here/indexer_build}
python_bin=${PYTHON_BIN:-${PYTHON:-python3}}
cann_root=${CANN_ROOT:-${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}}}
vendor_root=${OPS_TRANSFORMER_ROOT:?Set OPS_TRANSFORMER_ROOT to the recorded ops-transformer source}
compiler_host=${CXX:-g++}
if [[ -z "$cann_root" || ! -d "$cann_root" ]]; then
    echo 'Set CANN_ROOT or source CANN set_env.sh.' >&2
    exit 2
fi
device_compiler=${BISHENG:-}
if [[ -z "$device_compiler" ]]; then
    for candidate in "$cann_root/bin/bisheng" \
                     "$cann_root/compiler/ccec_compiler/bin/bisheng"; do
        if [[ -x "$candidate" ]]; then
            device_compiler=$candidate
            break
        fi
    done
fi
if [[ ! -x "$device_compiler" ]]; then
    echo 'CANN bisheng was not found; set BISHENG explicitly.' >&2
    exit 2
fi
arch_root="$cann_root/$(uname -m)-linux"
if [[ ! -d "$arch_root" ]]; then
    arch_root=$cann_root
fi
includes=(-I "$here" -I "$cann_root/include" -I "$vendor_root")
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
flags=(-xcce --npu-arch=dav-2201 -std=c++17 -O2 -fPIC)
if [[ ${SANITIZE:-0} == 1 ]]; then
    flags+=(--cce-enable-sanitizer)
fi
mkdir -p "$out"
"$device_compiler" "${flags[@]}" "${includes[@]}" \
    -c "$here/indexer_kernel.cpp" -o "$out/indexer_kernel.o"
if [[ ${INDEXER_KERNEL_ONLY:-0} != 1 ]]; then
    package_info=$($python_bin - <<'PY'
import importlib.util
from pathlib import Path
import torch

spec = importlib.util.find_spec("torch_npu")
if spec is None or spec.origin is None:
    raise SystemExit("The selected Python must provide torch_npu.")
print(Path(torch.__file__).resolve().parent)
print(Path(spec.origin).resolve().parent)
print(int(torch._C._GLIBCXX_USE_CXX11_ABI))
PY
)
    mapfile -t packages <<< "$package_info"
    torch_root=${packages[0]}
    npu_root=${packages[1]}
    "$compiler_host" -std=c++17 -O2 -fPIC \
        "-D_GLIBCXX_USE_CXX11_ABI=${packages[2]}" \
        "${includes[@]}" -I "$torch_root/include" \
        -I "$torch_root/include/torch/csrc/api/include" \
        -I "$npu_root/include" -c "$here/indexer_binding.cpp" \
        -o "$out/indexer_binding.o"
    "$compiler_host" -shared "$out/indexer_kernel.o" "$out/indexer_binding.o" \
        -o "$out/whole_layer_indexer.so" \
        -L "$arch_root/lib64" -L "$cann_root/lib64" \
        -L "$torch_root/lib" -L "$npu_root/lib" \
        -lascendcl -lruntime -lunified_dlog -lc_sec -lregister -lprofapi \
        -ltorch -ltorch_cpu -lc10 -ltorch_npu
fi
"$python_bin" "$here/indexer_provenance.py" \
    --vendor "$vendor_root" --cann "$cann_root" --output "$out/provenance.json"
echo "built indexer artifacts in $out"
