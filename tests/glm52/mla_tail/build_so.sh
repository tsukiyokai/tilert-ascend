#!/usr/bin/env bash
# Compile against the local CANN and the selected Python's torch/torch_npu.
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${1:-$here/build}
python_bin=${PYTHON_BIN:-${PYTHON:-python3}}
compiler_host=${CXX:-g++}
cann_root=${CANN_ROOT:-${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}}}
if [[ -z "$cann_root" ]]; then
    for candidate in /usr/local/Ascend/ascend-toolkit/latest \
                     /usr/local/Ascend/cann /opt/ascend/cann-9.2.0; do
        if [[ -d "$candidate" ]]; then
            cann_root=$candidate
            break
        fi
    done
fi
if [[ -z "$cann_root" || ! -d "$cann_root" ]]; then
    echo 'Set CANN_ROOT or ASCEND_HOME_PATH to a CANN installation.' >&2
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
if [[ -z "$device_compiler" || ! -x "$device_compiler" ]]; then
    echo 'CANN bisheng compiler was not found; set BISHENG explicitly.' >&2
    exit 2
fi

arch_root="$cann_root/$(uname -m)-linux"
if [[ ! -d "$arch_root" ]]; then
    arch_root=$cann_root
fi
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
mapfile -t package_lines <<< "$package_info"
torch_root=${package_lines[0]}
torch_npu_root=${package_lines[1]}
torch_abi=${package_lines[2]}

device_includes=(-I "$here" -I "$cann_root/include")
host_includes=(-I "$here" -I "$cann_root/include")
for relative in asc asc/include asc/include/adv_api asc/include/basic_api \
                asc/include/c_api asc/include/simt_api asc/include/utils \
                asc/include/basic_api/reg_compute asc/impl/adv_api \
                asc/impl/basic_api asc/impl/c_api asc/impl/simt_api \
                asc/impl/utils asc/impl/basic_api/reg_compute \
                tikcpp/tikcfw tikcpp/tikcfw/lib tikcpp/tikcfw/impl \
                tikcpp/tikcfw/interface; do
    if [[ -d "$arch_root/$relative" ]]; then
        device_includes+=(-I "$arch_root/$relative")
        host_includes+=(-I "$arch_root/$relative")
    fi
done
torch_includes=(-I "$torch_root/include"
                -I "$torch_root/include/torch/csrc/api/include"
                -I "$torch_npu_root/include")
device_flags=(-xcce --npu-arch=dav-2201 -std=c++17 -O2 -fPIC)
if [[ ${SANITIZE:-0} == 1 ]]; then
    device_flags+=(--cce-enable-sanitizer)
fi
mkdir -p "$out"
"$device_compiler" "${device_flags[@]}" "${device_includes[@]}" \
    -c "$here/mla_tail_kernel.cpp" -o "$out/kernel.o"
"$compiler_host" -std=c++17 -O2 -fPIC "${host_includes[@]}" \
    -c "$here/mla_tail_tiling.cpp" -o "$out/tiling.o"
"$compiler_host" -std=c++17 -O2 -fPIC \
    "-D_GLIBCXX_USE_CXX11_ABI=$torch_abi" \
    "${host_includes[@]}" "${torch_includes[@]}" \
    -c "$here/mla_tail_binding.cpp" -o "$out/binding.o"
"$compiler_host" -shared "$out/kernel.o" "$out/tiling.o" "$out/binding.o" \
    -o "$out/mla_tail_torch.so" \
    -L "$arch_root/lib64" -L "$cann_root/lib64" \
    -L "$torch_root/lib" -L "$torch_npu_root/lib" \
    -ltiling_api -lplatform -lascendcl -lruntime -lunified_dlog -lc_sec \
    -lregister -lprofapi -ltorch -ltorch_cpu -lc10 -ltorch_npu
echo "built $out/mla_tail_torch.so"
