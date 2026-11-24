#!/usr/bin/env bash
set -eo pipefail
source /workspace/environment/activate.sh
set -u
root=/workspace/experiments/gmm-compact-20261020
output=$root/build-v1
original=/workspace/experiments/gmm-fastpath-20261018/prod-kernel/overlay/vendors/custom_transformer
op=grouped_matmul_swiglu_quant_v2
variant=GroupedMatmulSwigluQuantV2_fa3d6d3de6e1f32e170e39d2ddd3a20e
binary=op_impl/ai_core/tbe/kernel/ascend910_93/$op
headers=op_impl/ai_core/tbe/custom_transformer_impl/ascendc/$op
wrapper=op_impl/ai_core/tbe/custom_transformer_impl/dynamic/$op.py
test ! -e "$output"
mkdir -p "$output/overlay/vendors" "$output/kernel"
cp -a --reflink=auto "$original" "$output/overlay/vendors/"
overlay=$output/overlay/vendors/custom_transformer
cp "$root"/candidate/*.h "$overlay/$headers/"
python - "$original/$binary/$variant.json" "$output/input-param.json" <<'PY'
import json
from pathlib import Path
import sys
descriptor = json.loads(Path(sys.argv[1]).read_text())
support = descriptor['supportInfo']
params = {'op_type': 'GroupedMatmulSwigluQuantV2', 'op_list': [{
    'bin_filename': descriptor['binFileName'],
    **{key: support[key] for key in ('inputs', 'outputs', 'attrs')},
}]}
Path(sys.argv[2]).write_text(json.dumps(params, indent=2) + '\n')
PY
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 HI_PYTHON=python3 TILINGKEY_PAR_COMPILE=1 BIN_FILENAME_HASHED=1
export CPATH=/workspace/src/vllm-ascend-unified/csrc/common/include:/workspace/src/vllm-ascend-unified/csrc/third_party/catlass/include
export ASCEND_CUSTOM_OPP_PATH="$overlay:$ASCEND_CUSTOM_OPP_PATH"
cd "$output"
opc "$overlay/$wrapper" --main_func="$op" --input_param="$output/input-param.json" \
    --soc_version=Ascend910_9391 --output="$output/kernel" \
    --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic \
    > "$output/build.log" 2>&1
python - "$original/$binary/$variant.json" "$output/kernel/$variant.json" \
    "$output/kernel/$variant.o" "$overlay/$binary/$variant.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
original, compiled = (json.loads(Path(path).read_text()) for path in sys.argv[1:3])
sha = hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest()
assert compiled['sha256'] == sha
for key in original.keys() | compiled.keys():
    if key not in ('sha256', 'filePath'):
        assert original.get(key) == compiled.get(key), key
original['sha256'] = sha
Path(sys.argv[4]).write_text(json.dumps(original, indent=2) + '\n')
PY
cp "$output/kernel/$variant.o" "$overlay/$binary/$variant.o"
sha256sum "$root"/candidate/*.h "$output/kernel/$variant.o" \
    "$overlay/op_api/lib/libcust_opapi.so" \
    "$overlay/op_impl/ai_core/tbe/op_tiling/liboptiling.so" > "$output/artifacts.sha256"
cat "$output/artifacts.sha256"
