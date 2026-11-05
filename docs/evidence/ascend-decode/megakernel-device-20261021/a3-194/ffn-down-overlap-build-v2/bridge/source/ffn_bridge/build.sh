#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${1:-$here/build-v1}
python_bin=${PYTHON_BIN:-python3}
source "$here/../../routed_int4/build_common.sh"
if [[ -e "$out" ]]; then
    echo "Choose a new build directory; $out already exists." >&2
    exit 2
fi
mkdir -p "$out/source/ffn_bridge" "$out/vendor"
for name in contract.h native.h kernel.cpp tiling.cpp binding.cpp build.sh \
            prepare_vendor.py fixtures.py check.py contract_test.cpp test_contract.py README.md; do
    cp "$here/$name" "$out/source/ffn_bridge/$name"
done
cp "$here/../norm_native.h" "$here/../norm_contract.h" "$out/source/"
cp "$here/../../routed_int4/build_common.sh" "$out/source/build_common.sh"
if [[ -n ${BRIDGE_VENDOR_DIR:-} ]]; then
    cp "$BRIDGE_VENDOR_DIR/"* "$out/vendor/"
else
    "$python_bin" "$here/prepare_vendor.py" \
        "${VLLM_ASCEND_ROOT:?Set VLLM_ASCEND_ROOT or BRIDGE_VENDOR_DIR}" "$out/vendor"
fi

mapfile -t packages < <("$python_bin" - <<'PY'
import importlib.util
from pathlib import Path
import torch
spec = importlib.util.find_spec("torch_npu")
if spec is None or spec.origin is None:
    raise SystemExit("Torch-NPU is required for the binding")
print(Path(torch.__file__).resolve().parent)
print(Path(spec.origin).resolve().parent)
print(int(torch._C._GLIBCXX_USE_CXX11_ABI))
PY
)
torch_root=${packages[0]}
npu_root=${packages[1]}
own="$out/source/ffn_bridge"

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
    "${routed_device_includes[@]}" -I "$own" -I "$out/vendor" \
    -c "$own/kernel.cpp" -o "$out/kernel.o"
run_step tiling "$routed_cxx" -std=c++17 -O2 -fPIC \
    "${routed_host_includes[@]}" -I "$own" \
    -c "$own/tiling.cpp" -o "$out/tiling.o"
run_step binding "$routed_cxx" -std=c++17 -O2 -fPIC \
    "-D_GLIBCXX_USE_CXX11_ABI=${packages[2]}" \
    "${routed_host_includes[@]}" -I "$own" -I "$torch_root/include" \
    -I "$torch_root/include/torch/csrc/api/include" -I "$npu_root/include" \
    -c "$own/binding.cpp" -o "$out/binding.o"
run_step link "$routed_cxx" -shared "$out/kernel.o" "$out/tiling.o" \
    "$out/binding.o" -o "$out/ffn_bridge.so" "${routed_libs[@]}" \
    -L "$torch_root/lib" -L "$npu_root/lib" -ltorch -ltorch_cpu -lc10 -ltorch_npu
"$routed_bisheng" --version > "$out/compiler.txt"
"$python_bin" - "$out" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import torch
import torch_npu

output = Path(sys.argv[1]).resolve()
paths = sorted(p for sub in ("source", "vendor")
               for p in (output / sub).rglob("*") if p.is_file())
source_hashes = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in paths}
artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for pattern in ("*.o", "*.so") for p in output.glob(pattern)}
torch.ops.load_library(str(output / "ffn_bridge.so"))
schemas = {name: str(getattr(torch.ops.tilert_ffn_bridge, name).default._schema)
           for name in ("prepare", "forward_out", "topk_out", "finalize_out")}
if torch.npu.is_initialized():
    raise RuntimeError("CPU build unexpectedly initialized NPU")
manifest = {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "npu_initialized": False, "sources": source_hashes,
            "artifacts": artifacts, "schemas": schemas,
            "device_validation": "not run"}
(output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"library": str(output / "ffn_bridge.so"),
                  "sha256": artifacts["ffn_bridge.so"],
                  "npu_initialized": False}))
PY
