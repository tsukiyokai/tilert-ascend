#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/build_common.sh"
if [[ ${ROUTED_PARTIALS:-int32} != int32 ]]; then
  echo 'The Active-list enumeration candidate requires INT32 partials.' >&2
  exit 2
fi
routed_live=$routed_here
routed_out=${1:-$routed_live/build-pipeline-front-active}
routed_base=${2:-$routed_live/build-pipeline-front}
routed_python=${PYTHON_BIN:-python3}
mkdir "$routed_out"
routed_out=$(cd -- "$routed_out" && pwd)
routed_base=$(cd -- "$routed_base" && pwd)

"$routed_python" - "$routed_live" "$routed_base" "$routed_out" <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

live, baseline, output = map(Path, sys.argv[1:])
manifest = json.loads((baseline / "build-manifest.json").read_text())
expected_kernel = "186b700ccf392e8bf8232679b14d6b21f3b63566077f66e7eae88f9c866fa695"
assert manifest["namespace"] == "tilert_pipeline_front"
assert manifest["own_sources"]["routed_int4/routed_int4_kernel.cpp"] == expected_kernel
for name, expected in manifest["own_sources"].items():
    source = baseline / "source" / name
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, name
    target = output / "source" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

for name in ("routed_int4_active_list.h", "active_list_test.cpp", "active_list.patch",
             "build_pipeline_front_active.sh"):
    shutil.copy2(live / name, output / "source/routed_int4" / name)
patch = output / "source/routed_int4/active_list.patch"
result = subprocess.run(
    ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patch)],
    cwd=output / "source/routed_int4", text=True, capture_output=True)
(output / "patch.log").write_text(result.stdout + result.stderr)
result.check_returncode()
parent_kernel = (baseline / "source/routed_int4/routed_int4_kernel.cpp").read_text()
new_kernel = (output / "source/routed_int4/routed_int4_kernel.cpp").read_text()
def numerical_body(kernel):
    start = kernel.index("__aicore__ inline void Project(")
    return kernel[start:kernel.index("}  // namespace\n")]
assert numerical_body(parent_kernel) == numerical_body(new_kernel)

# Keep both Torch schemas and all launch/tiling symbols independent when
# loading the accepted library and this candidate into one process.
renames = {
    "routed_shared_front_build_routed_tiling": "routed_shared_front_active_build_routed_tiling",
    "routed_shared_front_build_tiling": "routed_shared_front_active_build_tiling",
    "routed_shared_front_launch": "routed_shared_front_active_launch",
    "routed_shared_front_fused": "routed_shared_front_active_fused",
    "tilert_pipeline_front": "tilert_pipeline_front_active",
}
changed = {}
for name in ("routed_int4_binding.cpp", "routed_int4_kernel.cpp",
             "routed_int4_kernel.h", "routed_int4_pipeline.h"):
    path = output / "source/routed_int4" / name
    content = path.read_text()
    before = content
    for old_symbol, new_symbol in renames.items():
        content = content.replace(old_symbol, new_symbol)
    assert content != before, name
    path.write_text(content)
    changed[name] = hashlib.sha256(path.read_bytes()).hexdigest()
(output / "parent-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(output / "source-transforms.json").write_text(json.dumps({
    "parent_namespace": manifest["namespace"],
    "parent_kernel_sha256": expected_kernel,
    "kernel_patch": "source/routed_int4/active_list.patch",
    "symbol_renames": renames,
    "renamed_sources": changed,
}, indent=2) + "\n")
PY

source "$routed_out/source/routed_int4/build_common.sh"
routed_site=${SITE_PACKAGES:-}
if [[ -z "$routed_site" ]]; then
  routed_site=$("$routed_python" -c \
    'import pathlib, torch; print(pathlib.Path(torch.__file__).parent.parent)')
fi
routed_abi=${TORCH_CXX11_ABI:-}
if [[ -z "$routed_abi" ]]; then
  routed_abi=$("$routed_python" -c \
    'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
fi
routed_flags=(-DROUTED_INPUT_FRONT -DROUTED_SHARED_PIPELINE
  -DROUTED_ROW_BALANCED -DROUTED_BALANCED_NAMESPACE
  -DSHARED_W8_GROUPED_QUANT=1 -DROUTED_ACTIVE_LIST)
routed_torch_includes=(-I "$routed_site/torch/include"
  -I "$routed_site/torch/include/torch/csrc/api/include"
  -I "$routed_site/torch_npu/include")
sha256sum "$routed_out/source"/{routed_int4,shared_w8,mla_tail}/* \
  > "$routed_out/source-identity.sha256"
routed_compile "$routed_out/contract.log" "$routed_cxx" -std=c++17 -O2 \
  "${routed_flags[@]}" "$routed_here/active_list_test.cpp" \
  -o "$routed_out/active_list_test"
"$routed_out/active_list_test" >> "$routed_out/contract.log"
routed_compile "$routed_out/kernel.log" "$routed_bisheng" \
  "${routed_device_flags[@]}" "${routed_device_includes[@]}" \
  "${routed_flags[@]}" -c "$routed_here/routed_int4_kernel.cpp" \
  -o "$routed_out/kernel.o"
routed_compile "$routed_out/tiling.log" "$routed_cxx" -std=c++17 -O2 -fPIC \
  "${routed_host_includes[@]}" "${routed_flags[@]}" \
  -c "$routed_here/routed_int4_tiling.cpp" -o "$routed_out/tiling.o"
routed_compile "$routed_out/shared_tiling.log" "$routed_cxx" -std=c++17 -O2 -fPIC \
  "${routed_host_includes[@]}" "${routed_flags[@]}" \
  -c "$routed_here/routed_int4_pipeline_tiling.cpp" \
  -o "$routed_out/shared_tiling.o"
routed_compile "$routed_out/binding.log" "$routed_cxx" -std=c++17 -O2 -fPIC \
  "-D_GLIBCXX_USE_CXX11_ABI=$routed_abi" "${routed_torch_includes[@]}" \
  "${routed_host_includes[@]}" "${routed_flags[@]}" \
  -c "$routed_here/routed_int4_binding.cpp" -o "$routed_out/binding.o"
routed_compile "$routed_out/link.log" "$routed_cxx" -shared \
  "$routed_out/kernel.o" "$routed_out/tiling.o" "$routed_out/binding.o" \
  "$routed_out/shared_tiling.o" -o "$routed_out/routed_int4_torch.so" \
  "${routed_libs[@]}" -L "$routed_site/torch/lib" \
  -L "$routed_site/torch_npu/lib" -ltorch -ltorch_cpu -lc10 -ltorch_npu
routed_compile "$routed_out/tiling_dump.log" "$routed_cxx" -std=c++17 -O2 \
  "${routed_host_includes[@]}" "${routed_flags[@]}" \
  "$routed_here/dump_tiling.cpp" "$routed_out/tiling.o" \
  -o "$routed_out/tiling_dump" "${routed_libs[@]}"
"$routed_out/tiling_dump" > "$routed_out/tiling.jsonl"
"$routed_bisheng" --version > "$routed_out/compiler.txt"
sha256sum -c "$routed_out/source-identity.sha256" \
  > "$routed_out/source-verification.log"
sha256sum "$routed_out/source"/{routed_int4,shared_w8,mla_tail}/* \
  "$routed_out"/{kernel.o,tiling.o,shared_tiling.o,binding.o,routed_int4_torch.so} \
  > "$routed_out/identity.sha256"
"$routed_python" - "$routed_target/asc" "$routed_out" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

sdk, output = map(Path, sys.argv[1:])
parent = json.loads((output / "parent-manifest.json").read_text())
tilings = [json.loads(line) for line in (output / "tiling.jsonl").read_text().splitlines()]
assert tilings == parent["tilings"], "candidate changed the projection tilings"
headers = (
    "include/adv_api/matmul/matmul_intf.h",
    "impl/adv_api/detail/matmul/param/matmul_shape_info.h",
    "impl/adv_api/detail/matmul/scheduler/iterator/k_loop/k_loop_norm.h",
    "impl/adv_api/detail/matmul/stage/copy_cube_out/copy_cube_out_fixpipe.h",
    "impl/adv_api/detail/matmul/stage/copy_cube_in/base/copy_cube_in_norm.h",
    "impl/adv_api/detail/matmul/stage/copy_cube_in/copy_tile_to_cube/data_copy_wrapper_utils.h",
)
report = dict(parent)
report.update({
    "variant": "Ascending active expert enumeration with unchanged matrix and row ownership",
    "namespace": "tilert_pipeline_front_active",
    "c_abi": ["routed_shared_front_active_launch",
              "routed_shared_front_active_build_routed_tiling",
              "routed_shared_front_active_build_tiling"],
    "compiled_from_snapshot": str(output / "source"),
    "flags": parent["flags"] + ["ROUTED_ACTIVE_LIST"],
    "own_sources": {
        str(path.relative_to(output / "source")): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((output / "source").glob("*/*"))
    },
    "sdk_headers": {
        str(sdk / name): hashlib.sha256((sdk / name).read_bytes()).hexdigest()
        for name in headers
    },
    "scratch_tensors": 21,
    "active_list_slot": 20,
    "active_list_shape": [264],
    "active_list_dtype": "int32",
    "active_list_bytes": 1056,
    "active_list_layout": "[0]=count, [1:8]=0, [8:8+count]=ascending expert IDs, remaining IDs=0",
    "original_scratch_slots_unchanged": list(range(20)),
    "phase0_ub_bytes_per_aiv": 83968,
    "resident_ub_bytes_per_aiv": 113408,
    "active_producers": ["from_bf16", "from_dispatch"],
    "extra_scratch_bytes_m12": 1056,
    "extra_scratch_bytes_m32": 1056,
    "tilings_equal_to_parent": True,
    "tilings": tilings,
})
(output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
PY
printf 'Built %s\n' "$routed_out/routed_int4_torch.so"
