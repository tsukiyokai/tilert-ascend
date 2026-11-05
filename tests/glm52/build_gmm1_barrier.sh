#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 || (${3:-} != "" && ${3:-} != --compact-experts) ]]; then
    echo "Usage: $0 VLLM_ASCEND_SOURCE NEW_OUTPUT_DIRECTORY [--compact-experts]" >&2
    exit 2
fi
: "${ASCEND_HOME_PATH:?Source the CANN/Python activation before this script}"

source_dir=$(realpath "$1")
output_dir=$(realpath -m "$2")
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# The barrier is the correctness patch; the three after it are the speed
# patches measured at 1.468 ms a round, and the overlay that produced that
# figure is byte-identical to the barrier header plus exactly these three.
# gmm1-single-chunk is superseded by gmm1-straight-line, which rewrites the
# same branch, and the two cannot both apply.
patches=(
    "$script_dir/patches/gmm1-swiglu-barrier.patch"
    "$script_dir/../../patches/cann-gmm-swiglu-v2/gmm1-straight-line.patch"
    "$script_dir/../../patches/cann-gmm-swiglu-v2/gmm1-chunk-span.patch"
    "$script_dir/../../patches/cann-gmm-swiglu-v2/gmm1-tiling-rowcount.patch"
)
op=grouped_matmul_swiglu_quant_v2
kernel_header="gmm/$op/op_kernel/${op}_a8w4_msd_post.h"
pipeline_header="gmm/$op/op_kernel/${op}_a8w4_msd_pipeline.h"
headers=("$kernel_header" "$pipeline_header")
if [[ ${3:-} == --compact-experts ]]; then
    patches+=("$script_dir/../../patches/cann-gmm-swiglu-v2/gmm1-active-groups.patch")
    headers+=("gmm/$op/op_kernel/${op}_a8w4_msd_mid.h"
              "gmm/$op/op_kernel/${op}_utils.h")
fi
original_vendor="$source_dir/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
binary_dir="op_impl/ai_core/tbe/kernel/ascend910_93/$op"
variant=GroupedMatmulSwigluQuantV2_fa3d6d3de6e1f32e170e39d2ddd3a20e
installed_headers="op_impl/ai_core/tbe/custom_transformer_impl/ascendc/$op"
wrapper="op_impl/ai_core/tbe/custom_transformer_impl/dynamic/$op.py"

if [[ ! -f "$source_dir/csrc/$kernel_header" ]]; then
    echo "GMM1 source not found: $source_dir/csrc/$kernel_header" >&2
    exit 1
fi
if [[ -e "$output_dir" ]]; then
    echo "Output directory must be new: $output_dir" >&2
    exit 1
fi
case "$output_dir/" in
    "$source_dir/"*)
        echo "Output directory must be outside the original source" >&2
        exit 1
        ;;
esac

build_source="$output_dir/source/csrc"
mkdir -p "$build_source"
tar -C "$source_dir/csrc" --exclude='./build' --exclude='./build_cache' \
    --exclude='./output' --exclude='./build_out' --exclude=.git \
    --exclude=__pycache__ -cf - . | tar -C "$build_source" -xf -
sha256sum "$source_dir/csrc/$kernel_header" \
    "$source_dir/csrc/$pipeline_header" "${patches[@]}" \
    "$script_dir/build_gmm1_barrier.sh" "$original_vendor/$wrapper" \
    "$original_vendor/$binary_dir/$variant.json" \
    "$build_source/common/include/cann_compat.h" > "$output_dir/inputs.sha256"
for header in "${headers[@]}"; do
    cmp "$build_source/$header" "$original_vendor/$installed_headers/${header##*/}"
done
cmp "$build_source/gmm/$op/op_kernel/$op.cpp" \
    "$original_vendor/$installed_headers/$op.cpp"
for patch_file in "${patches[@]}"; do
    strip=1
    if grep -q '^+++ b/csrc/' "$patch_file"; then
        strip=2
    fi
    patch --batch --fuzz=0 -p"$strip" -d "$build_source" < "$patch_file"
done
if [[ ${3:-} == --compact-experts ]]; then
    headers+=("gmm/$op/op_kernel/active_groups.h")
fi

overlay="$output_dir/overlay/vendors/custom_transformer"
mkdir -p "$(dirname -- "$overlay")"
cp -a --reflink=auto "$original_vendor" "$overlay"
for header in "${headers[@]}"; do
    cp "$build_source/$header" "$overlay/$installed_headers/"
done
# opc compiles the entry from the installed copy, so a patch that touches it
# takes effect only once that copy is replaced too. Without this the build
# succeeds, the kernel is byte-identical to the unpatched one, and the
# measurement reports no difference.
cp "$build_source/gmm/$op/op_kernel/$op.cpp" "$overlay/$installed_headers/"

kernel_output="$output_dir/kernel"
mkdir -p "$kernel_output"
python3 - "$original_vendor/$binary_dir/$variant.json" \
    "$output_dir/input-param.json" <<'PY'
import json
from pathlib import Path
import sys

descriptor = json.loads(Path(sys.argv[1]).read_text())
support = descriptor["supportInfo"]
parameters = {
    "op_type": "GroupedMatmulSwigluQuantV2",
    "op_list": [{
        "bin_filename": descriptor["binFileName"],
        **{key: support[key] for key in ("inputs", "outputs", "attrs")},
    }],
}
Path(sys.argv[2]).write_text(json.dumps(parameters, indent=2) + "\n")
PY

(
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    export CPATH="$build_source/common/include:$build_source/third_party/catlass/include${CPATH:+:$CPATH}"
    export ASCEND_CUSTOM_OPP_PATH="$overlay${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
    export HI_PYTHON=python3 TILINGKEY_PAR_COMPILE=1 BIN_FILENAME_HASHED=1
    opc "$overlay/$wrapper" --main_func="$op" \
        --input_param="$output_dir/input-param.json" --soc_version=Ascend910_9391 \
        --output="$kernel_output" --impl_mode=high_performance,optional \
        --simplified_key_mode=0 --op_mode=dynamic
) 2>&1 | tee "$output_dir/build.log"

python3 - "$original_vendor/$binary_dir/$variant.json" \
    "$kernel_output/$variant.json" "$kernel_output/$variant.o" \
    "$overlay/$binary_dir/$variant.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

original, compiled = (json.loads(Path(path).read_text()) for path in sys.argv[1:3])
with Path(sys.argv[3]).open("rb") as stream:
    elf_sha = hashlib.file_digest(stream, "sha256").hexdigest()
assert compiled["sha256"] == elf_sha
for key in original.keys() | compiled.keys():
    if key not in ("sha256", "filePath"):
        assert original.get(key) == compiled.get(key), key
original["sha256"] = elf_sha
Path(sys.argv[4]).write_text(json.dumps(original, indent=2) + "\n")
PY
cp "$kernel_output/$variant.o" "$overlay/$binary_dir/$variant.o"
sha256sum "$output_dir/input-param.json" "$kernel_output/$variant.o" \
    "$kernel_output/$variant.json" "$overlay/$wrapper" \
    "$build_source/$kernel_header" "$build_source/$pipeline_header" \
    "$build_source/gmm/$op/op_kernel/$op.cpp" \
    "$overlay/$installed_headers/$op.cpp" \
    "$overlay/$binary_dir/$variant.o" \
    "$overlay/$binary_dir/$variant.json" \
    "$original_vendor/op_api/lib/libcust_opapi.so" \
    "$overlay/op_api/lib/libcust_opapi.so" \
    "$original_vendor/op_impl/ai_core/tbe/op_tiling/liboptiling.so" \
    "$overlay/op_impl/ai_core/tbe/op_tiling/liboptiling.so" > "$output_dir/artifacts.sha256"
for header in "${headers[@]}"; do
    sha256sum "$build_source/$header" "$overlay/$installed_headers/${header##*/}"
done >> "$output_dir/artifacts.sha256"
printf 'ASCEND_CUSTOM_OPP_PATH=%s:%s\n' "$overlay" "$original_vendor"
