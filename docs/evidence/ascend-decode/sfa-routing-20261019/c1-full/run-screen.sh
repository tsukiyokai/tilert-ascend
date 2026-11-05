#!/usr/bin/env bash
set -eo pipefail
if [[ $# != 3 ]]; then echo "Usage: $0 NPU_ID NEW_OUTPUT_DIR v1|bounds" >&2; exit 2; fi
device=$1
output=$2
arm=$3
source /workspace/environment/activate.sh
set -u
artifact=/workspace/offline/sfa-c1-bounds-20261019-codex
original_vendor=/workspace/src/vllm-ascend-unified/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
case "$arm" in v1) key=1158;; bounds) key=134;; *) exit 2;; esac
vendor=$artifact/$arm/vendors/custom_transformer
ASCEND_CUSTOM_OPP_PATH="$vendor:$original_vendor:${ASCEND_CUSTOM_OPP_PATH:-}" \
ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1 \
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
timeout --kill-after=15s 180s python3 "$artifact/bench-sfa-c1.py" \
  --device "$device" --input "$artifact/cpu-fixture/input.pt" \
  --output "$output" 2>&1 | tee "$output.launch.log"
python3 "$artifact/audit-launch.py" "$output" "$vendor" "$key"
