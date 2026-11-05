#!/usr/bin/env bash
set -euo pipefail
base=/workspace/experiments/layer-major-m1-m32-20261020
docker exec trt-asc env TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 \
  /opt/trt-asc/venv/bin/python "$base/compare.py" \
  "$base/old/device-78x32-sfa-01" \
  /workspace/experiments/layer-major-m32-20261020/sfa-8 \
  "$base/result.json"
