#!/usr/bin/env bash
set -eo pipefail
source /workspace/environment/activate.sh
set -u
root=/workspace/experiments/gmm2-compact-20261020
out="$root/teacher-counts-screen"
vendor="$out/fixed-wide-buffer-build/overlay/vendors/tilert_gmm2"
kernel="$vendor/op_impl/ai_core/tbe/kernel/ascend910_93/grouped_matmul/GroupedMatmul_a0be48d01b162435398ed52e1b6182d4.o"
python3 - "$kernel" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

kernel = Path(sys.argv[1])
digest = hashlib.sha256(kernel.read_bytes()).hexdigest()
assert digest == "ae1a0296f8f051c0701a21f84d83769f902c6fe6c7321a08f7c262745483922d"
assert json.loads(kernel.with_suffix(".json").read_text())["sha256"] == digest
PY
export ASCEND_CUSTOM_OPP_PATH="$vendor:${ASCEND_CUSTOM_OPP_PATH:-}"
export ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
cd "$out"
exec python3 /workspace/experiments/gmm-compact-20261020/watch_kernel.py \
    "$kernel" "$out/device-access.json" \
    python3 "$out/bench_counts.py" --device "${1:-npu:15}" \
    --output "$out/device-result.json"
