#!/usr/bin/env bash
set -eo pipefail
source /workspace/environment/activate.sh
set -u
base=/workspace/experiments/m32-full-teacher-20261020/fia-row-correction
export PYTHONPATH=/workspace/experiments/m32-full-teacher-20261020/source:$PYTHONPATH
vendor=/workspace/experiments/gmm-fastpath-20261018/prod-kernel/overlay/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH="$vendor:$ASCEND_CUSTOM_OPP_PATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0
exec timeout --signal=TERM --kill-after=10s 300 \
  /opt/trt-asc/venv/bin/python "$base/replay.py" device \
  --die 15 --output "$base/device-01"
