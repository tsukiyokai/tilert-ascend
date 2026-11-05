#!/usr/bin/env bash
set -eo pipefail
source /workspace/environment/activate.sh
set -u

base=/workspace/experiments/m32-full-teacher-20261020/norm-fusion/compiled-check
vendor=/workspace/experiments/gmm-fastpath-20261018/prod-kernel/overlay/vendors/custom_transformer
label=${1:?Usage: bash run-acl-pair.sh OUTPUT_LABEL NPU [SOURCE]}
device=${2:?Choose the NPU after inventory}
source_path=${3:-/workspace/src/TileRT-Ascend}
case "$label" in
    *[!a-zA-Z0-9_-]*|'') exit 2 ;;
esac
for arm in on off; do
    test ! -e "$base/$label-$arm"
done

export ASCEND_CUSTOM_OPP_PATH="$vendor:$ASCEND_CUSTOM_OPP_PATH"
export ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
for arm in on off; do
    timeout --signal=TERM --kill-after=15s 300 python3 "$base/check-acl.py" \
        --mode device --flag "$arm" --device "$device" \
        --source "$source_path" --output "$base/$label-$arm"
done
