#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${1:?Usage: build.sh NEW_OUTPUT BRIDGE_BUILD_V2 BODY_BUILD_V2}
bridge=${2:?Usage: build.sh NEW_OUTPUT BRIDGE_BUILD_V2 BODY_BUILD_V2}
body=${3:?Usage: build.sh NEW_OUTPUT BRIDGE_BUILD_V2 BODY_BUILD_V2}
python_bin=${PYTHON_BIN:-python3}
"$python_bin" "$here/prepare.py" "$bridge" "$body" "$out"
out=$(cd -- "$out" && pwd)
source "$out/body/source/routed_int4/build_common.sh"
own="$out/source"

"$python_bin" - "$out/torch-build.json" <<'PY'
import importlib.util
import json
from pathlib import Path
import sys
import torch

spec = importlib.util.find_spec("torch_npu")
Path(sys.argv[1]).write_text(json.dumps({
    "torch": str(Path(torch.__file__).resolve().parent),
    "torch_npu": str(Path(spec.origin).resolve().parent),
    "abi": int(torch._C._GLIBCXX_USE_CXX11_ABI),
}) + "\n")
PY
mapfile -t packages < <("$python_bin" - "$out/torch-build.json" <<'PY'
import json
from pathlib import Path
import sys
data = json.loads(Path(sys.argv[1]).read_text())
for key in ("torch", "torch_npu", "abi"):
    print(data[key])
PY
)
macros=(-DROUTED_ROW_BALANCED=1 -DROUTED_SHARED_PIPELINE=1
        -DROUTED_INPUT_FRONT=1 -DSHARED_W8_GROUPED_QUANT=1)

run_step() {
    local label=$1
    shift
    "$python_bin" - "$out/commands.jsonl" "$label" "$@" <<'PY'
import json
from pathlib import Path
import sys
with Path(sys.argv[1]).open("a") as stream:
    stream.write(json.dumps({"step": sys.argv[2], "argv": sys.argv[3:]}) + "\n")
PY
    routed_compile "$out/$label.log" "$@"
}

run_step kernel "$routed_bisheng" "${routed_device_flags[@]}" \
    "${routed_device_includes[@]}" "${macros[@]}" \
    -I "$out/bridge/vendor" -I "$own" \
    -c "$own/kernel.cpp" -o "$out/kernel.o"
run_step bridge-tiling "$routed_cxx" -std=c++17 -O2 -fPIC \
    "${routed_host_includes[@]}" "${macros[@]}" \
    -c "$out/bridge/source/ffn_bridge/tiling.cpp" -o "$out/bridge-tiling.o"
run_step routed-tiling "$routed_cxx" -std=c++17 -O2 -fPIC \
    "${routed_host_includes[@]}" "${macros[@]}" \
    -c "$out/body/source/routed_int4/routed_int4_tiling.cpp" -o "$out/routed-tiling.o"
run_step shared-tiling "$routed_cxx" -std=c++17 -O2 -fPIC \
    "${routed_host_includes[@]}" "${macros[@]}" \
    -c "$out/body/source/routed_int4/routed_int4_pipeline_tiling.cpp" -o "$out/shared-tiling.o"
run_step binding "$routed_cxx" -std=c++17 -O2 -fPIC \
    "-D_GLIBCXX_USE_CXX11_ABI=${packages[2]}" "${macros[@]}" \
    "${routed_host_includes[@]}" -I "$own" -I "${packages[0]}/include" \
    -I "${packages[0]}/include/torch/csrc/api/include" -I "${packages[1]}/include" \
    -c "$own/binding.cpp" -o "$out/binding.o"
run_step link "$routed_cxx" -shared "$out/kernel.o" "$out/bridge-tiling.o" \
    "$out/routed-tiling.o" "$out/shared-tiling.o" "$out/binding.o" \
    -o "$out/ffn_composed.so" "${routed_libs[@]}" \
    -L "${packages[0]}/lib" -L "${packages[1]}/lib" \
    -ltorch -ltorch_cpu -lc10 -ltorch_npu
"$routed_bisheng" --version > "$out/compiler.txt"
"$python_bin" - "$out" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import torch
import torch_npu

output = Path(sys.argv[1])
report = json.loads((output / "source-transforms.json").read_text())
for name, expected in report["sources"].items():
    if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Frozen source changed during build: {name}")
library = output / "ffn_composed.so"
torch.ops.load_library(str(library))
if torch.npu.is_initialized():
    raise RuntimeError("CPU build initialized NPU")
report.update({
    "namespace": "tilert_ffn_composed", "torch": torch.__version__,
    "torch_npu": torch_npu.__version__, "npu_initialized": False,
    "scratch_tensors": {"bridge": 12, "body": 20},
    "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for pattern in ("*.o", "*.so") for p in output.glob(pattern)},
    "schemas": {name: str(getattr(torch.ops.tilert_ffn_composed, name).default._schema)
                for name in ("prepare", "forward_out")},
    "device_validation": "not run",
})
(output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"library": str(library),
                  "sha256": report["artifacts"]["ffn_composed.so"],
                  "npu_initialized": False}))
PY
