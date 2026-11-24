#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/build_common.sh"
if [[ ${ROUTED_PARTIALS:-int32} != int32 ]]; then
  echo 'The GU split-K candidate requires INT32 partials.' >&2
  exit 2
fi
routed_live=$routed_here
routed_out=${1:-$routed_live/build-pipeline-front-splitk}
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

for name in ("routed_int4_gu_splitk.h", "gu_splitk_test.cpp", "gu_splitk.patch",
             "build_pipeline_front_splitk.sh"):
    shutil.copy2(live / name, output / "source/routed_int4" / name)
patch = output / "source/routed_int4/gu_splitk.patch"
result = subprocess.run(
    ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patch)],
    cwd=output / "source/routed_int4", text=True, capture_output=True)
(output / "patch.log").write_text(result.stdout + result.stderr)
result.check_returncode()

# Keep both Torch schemas and all launch/tiling symbols independent when
# loading the accepted library and this candidate into one process.
renames = {
    "routed_shared_front_build_routed_tiling": "routed_shared_front_splitk_build_routed_tiling",
    "routed_shared_front_build_tiling": "routed_shared_front_splitk_build_tiling",
    "routed_shared_front_launch": "routed_shared_front_splitk_launch",
    "routed_shared_front_fused": "routed_shared_front_splitk_fused",
    "tilert_pipeline_front": "tilert_pipeline_front_splitk",
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
    "kernel_patch": "source/routed_int4/gu_splitk.patch",
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
  -DSHARED_W8_GROUPED_QUANT=1 -DROUTED_GU_SPLIT_K)
routed_torch_includes=(-I "$routed_site/torch/include"
  -I "$routed_site/torch/include/torch/csrc/api/include"
  -I "$routed_site/torch_npu/include")
sha256sum "$routed_out/source"/{routed_int4,shared_w8,mla_tail}/* \
  > "$routed_out/source-identity.sha256"
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
routed_compile "$routed_out/contract.log" "$routed_cxx" -std=c++17 -O2 \
  "${routed_flags[@]}" "$routed_here/gu_splitk_test.cpp" \
  -o "$routed_out/gu_splitk_test"
"$routed_out/gu_splitk_test" >> "$routed_out/contract.log"
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
    "variant": "Three K2048 GU parts with INT32 sum for 24 groups and at most 8 active experts",
    "namespace": "tilert_pipeline_front_splitk",
    "c_abi": ["routed_shared_front_splitk_launch",
              "routed_shared_front_splitk_build_routed_tiling",
              "routed_shared_front_splitk_build_tiling"],
    "compiled_from_snapshot": str(output / "source"),
    "flags": parent["flags"] + ["ROUTED_GU_SPLIT_K"],
    "own_sources": {
        str(path.relative_to(output / "source")): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((output / "source").glob("*/*"))
    },
    "sdk_headers": {
        str(sdk / name): hashlib.sha256((sdk / name).read_bytes()).hexdigest()
        for name in headers
    },
    "gu_parts": 3,
    "gu_part_inner": 2048,
    "gu_original_inner": 6144,
    "gu_columns": 256,
    "gu_splitk_blocks": 24,
    "gu_splitk_maximum_active_experts": 8,
    "gu_product_shape": "[6 * M * 8, 256] INT32; first 2 * M * 8 rows hold the complete sum",
    "gu_sum_rounding": "Two INT32 Adds before the original INT32-to-FP32 and FP16 partial rounding",
    "extra_scratch_bytes_m12": 393216,
    "extra_scratch_bytes_m32": 1048576,
    "tilings_equal_to_parent": True,
    "tilings": tilings,
})
(output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
PY
printf 'Built %s\n' "$routed_out/routed_int4_torch.so"
