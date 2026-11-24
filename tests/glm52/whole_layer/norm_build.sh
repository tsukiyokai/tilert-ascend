#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${1:-$here/norm_build}
python_bin=${PYTHON_BIN:-python3}
vendor_root=${VLLM_ASCEND_ROOT:?Set VLLM_ASCEND_ROOT to the baseline source checkout}
vendor_kernel="$vendor_root/csrc/moe/add_rms_norm_bias/op_kernel"
source "$here/../routed_int4/build_common.sh"

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
mkdir -p "$out"
"$routed_bisheng" "${routed_device_flags[@]}" \
    "${routed_device_includes[@]}" -I "$here" -I "$vendor_kernel" \
    -c "$here/norm_kernel.cpp" -o "$out/norm_kernel.o"
"$routed_cxx" -std=c++17 -O2 -fPIC \
    "-D_GLIBCXX_USE_CXX11_ABI=${packages[2]}" \
    "${routed_host_includes[@]}" -I "$here" -I "$torch_root/include" \
    -I "$torch_root/include/torch/csrc/api/include" -I "$npu_root/include" \
    -c "$here/norm_binding.cpp" -o "$out/norm_binding.o"
"$routed_cxx" -shared "$out/norm_kernel.o" "$out/norm_binding.o" \
    -o "$out/whole_layer_norm.so" "${routed_libs[@]}" \
    -L "$torch_root/lib" -L "$npu_root/lib" \
    -ltorch -ltorch_cpu -lc10 -ltorch_npu
"$python_bin" - "$here" "$vendor_kernel" "$out" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

source, vendor, output = map(Path, sys.argv[1:])
paths = list(source.glob("norm_*.*"))
paths += [vendor / name for name in (
    "add_rms_norm_bias_single_n.h", "rms_norm_base.h", "reduce_common.h")]
paths += list(output.glob("*.o")) + list(output.glob("*.so"))
report = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
          for path in sorted(paths)}
(output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
PY
echo "built $out/whole_layer_norm.so"
