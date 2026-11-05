#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || (${2:-} != "" && ${2:-} != --compact-experts && ${2:-} != --adaptive-tiles) ]]; then
    echo "Usage: $0 NEW_OUTPUT_DIRECTORY [--compact-experts|--adaptive-tiles]" >&2
    exit 2
fi
: "${ASCEND_OPP_PATH:?Activate CANN first}"
output=$(realpath -m "$1")
if [[ -e "$output" ]]; then
    echo "Output directory must be new: $output" >&2
    exit 1
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
patch_file="$script_dir/../../patches/cann-grouped-matmul/gmm2-active-groups.patch"
tile_patch="$script_dir/../../patches/cann-grouped-matmul/gmm2-adaptive-tiles.patch"
builtin="$ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe"
variant=GroupedMatmul_a0be48d01b162435398ed52e1b6182d4
original="$builtin/kernel/ascend910_93/ops_transformer/grouped_matmul/$variant.json"
source_dir="$output/source/ops_transformer"
overlay="$output/overlay/vendors/tilert_gmm2"
kernel_dir="$overlay/op_impl/ai_core/tbe/kernel"

# The compact table uses unused B114 workspace; pin its allocation contract.
python3 - "$builtin" "$original" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

builtin, descriptor_path = map(Path, sys.argv[1:])
expected = {
    descriptor_path.with_suffix(".o"):
        "fa20701770aa8a575946ad502fcc770184b2a21b4d4e727e4c864aab0faa34d7",
    builtin / "op_tiling/lib/linux/aarch64/liboptiling.so":
        "0d603b1fe5375b069d870747cb6ef3da588db9e6c7e4d86bdd7b507aba0b7ac3",
    builtin / "op_tiling/lib/linux/aarch64/libopmaster_ct.so":
        "c86d2007196069ed8de09c9b16cfe4b0b1abe7adf646167f5e461087d2c7138d",
}
for path, sha in expected.items():
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != sha:
        raise SystemExit(f"CANN 9.2.0 B114 identity mismatch: {path}")
descriptor = json.loads(descriptor_path.read_text())
assert descriptor["sha256"] == expected[descriptor_path.with_suffix(".o")]
PY

mkdir -p "$source_dir/dynamic" "$source_dir/ascendc" "$output/kernel" \
    "$kernel_dir/ascend910_93/grouped_matmul" "$kernel_dir/config/ascend910_93"
cp "$builtin/impl/ops_transformer/dynamic/grouped_matmul.py" "$source_dir/dynamic/"
cp -a "$builtin/impl/ops_transformer/ascendc/grouped_matmul" "$source_dir/ascendc/"
ln -s "$builtin/impl/ops_transformer/ascendc/common" "$source_dir/ascendc/common"
ln -s "$builtin/impl/ops_transformer/ascendc/3rd" "$source_dir/ascendc/3rd"
sha256sum "$original" "$patch_file" "$tile_patch" "$script_dir/build_gmm2.sh" \
    "$builtin/op_tiling/lib/linux/aarch64/"*.so > "$output/inputs.sha256"
if [[ ${2:-} != "" ]]; then
    patch --batch --fuzz=0 -p1 -d "$source_dir/ascendc/grouped_matmul/op_kernel" < "$patch_file"
fi
if [[ ${2:-} == --adaptive-tiles ]]; then
    patch --batch --fuzz=0 -p1 -d "$source_dir/ascendc/grouped_matmul/op_kernel" < "$tile_patch"
fi
python3 - "$original" "$builtin/kernel/config/ascend910_93/ops_transformer/grouped_matmul.json" \
    "$output" <<'PY'
import json
from pathlib import Path
import sys

descriptor_path = Path(sys.argv[1])
original, config = (json.loads(Path(p).read_text()) for p in sys.argv[1:3])
out = Path(sys.argv[3])
support = original["supportInfo"]
params = {"op_type": "GroupedMatmul", "op_list": [{
    "bin_filename": original["binFileName"],
    **{key: support[key] for key in ("inputs", "outputs", "attrs")},
}]}
(out / "input-param.json").write_text(json.dumps(params, indent=2) + "\n")
entries = [item for item in config["binList"]
           if original["binFileName"] in item["binInfo"]["jsonFilePath"]]
assert len(entries) == 1
path = out / ("overlay/vendors/tilert_gmm2/op_impl/ai_core/tbe/kernel/"
              "config/ascend910_93/grouped_matmul.json")
path.write_text(json.dumps(config, indent=2) + "\n")
# Keep every other dtype/layout variant available through the same overlay.
target = out / ("overlay/vendors/tilert_gmm2/op_impl/ai_core/tbe/kernel/"
                "ascend910_93/grouped_matmul")
for source in descriptor_path.parent.iterdir():
    if source.stem != original["binFileName"]:
        (target / source.name).symlink_to(source)
PY
(
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    export HI_PYTHON=python3 TILINGKEY_PAR_COMPILE=1 BIN_FILENAME_HASHED=1
    opc "$source_dir/dynamic/grouped_matmul.py" --main_func=grouped_matmul \
        --input_param="$output/input-param.json" --soc_version=Ascend910_9391 \
        --output="$output/kernel" --impl_mode=high_performance,optional \
        --simplified_key_mode=0 --op_mode=dynamic
) 2>&1 | tee "$output/build.log"
python3 - "$original" "$output/kernel/$variant.json" "$output/kernel/$variant.o" \
    "$kernel_dir/ascend910_93/grouped_matmul/$variant.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

original, compiled = (json.loads(Path(p).read_text()) for p in sys.argv[1:3])
sha = hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest()
assert sha == compiled["sha256"]
for key in original.keys() | compiled.keys():
    if key not in ("sha256", "filePath"):
        assert original.get(key) == compiled.get(key), key
original["sha256"] = sha
Path(sys.argv[4]).write_text(json.dumps(original, indent=2) + "\n")
PY
cp "$output/kernel/$variant.o" "$kernel_dir/ascend910_93/grouped_matmul/"
sha256sum "$output/input-param.json" "$output/kernel/$variant.o" \
    "$kernel_dir/ascend910_93/grouped_matmul/$variant.json" \
    "$source_dir/dynamic/grouped_matmul.py" \
    "$source_dir/ascendc/grouped_matmul/op_kernel/"*.{cpp,h} > "$output/artifacts.sha256"
printf 'ASCEND_CUSTOM_OPP_PATH=%s:%s\n' "$overlay" "${ASCEND_CUSTOM_OPP_PATH:-}"
