#!/usr/bin/env bash
set -eo pipefail
source /workspace/environment/activate.sh
set -u
base=/workspace/experiments/layer-major-local-attention-20261020
device=${1:?explicit device, e.g. npu:15}
out=${2:-$base/device-01}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0
timeout --signal=TERM --kill-after=10s 300 \
  python "$base/probe.py" --device "$device" --quota-mib 512 --output "$out"
