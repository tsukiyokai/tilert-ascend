#!/usr/bin/env bash
set -eo pipefail
source /workspace/environment/activate.sh
set -u
build=/workspace/experiments/gmm2-compact-20261020/teacher-counts-screen/fixed-wide-buffer-build
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 HI_PYTHON=python3
export TILINGKEY_PAR_COMPILE=1 BIN_FILENAME_HASHED=1
opc "$build/source/ops_transformer/dynamic/grouped_matmul.py" \
    --main_func=grouped_matmul --input_param="$build/input-param.json" \
    --soc_version=Ascend910_9391 --output="$build/kernel" \
    --impl_mode=high_performance,optional --simplified_key_mode=0 \
    --op_mode=dynamic 2>&1 | tee "$build/build.log"
python3 - "$build" <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import sys

build = Path(sys.argv[1])
name = "GroupedMatmul_a0be48d01b162435398ed52e1b6182d4"
binary = build / "kernel" / (name + ".o")
descriptor = json.loads(binary.with_suffix(".json").read_text())
sha = hashlib.sha256(binary.read_bytes()).hexdigest()
assert descriptor["sha256"] == sha
target = build / "overlay/vendors/tilert_gmm2/op_impl/ai_core/tbe/kernel/ascend910_93/grouped_matmul" / binary.name
original = json.loads(target.with_suffix(".json").read_text())
for key in original.keys() | descriptor.keys():
    if key not in ("sha256", "filePath"):
        assert original.get(key) == descriptor.get(key), key
original["sha256"] = sha
target.with_suffix(".json").write_text(json.dumps(original, indent=2) + "\n")
shutil.copyfile(binary, target)
print("paired_buffer_fixed_elf_sha256=" + sha, flush=True)
PY
