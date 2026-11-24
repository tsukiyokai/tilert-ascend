#!/usr/bin/env bash
set -euo pipefail
base=/workspace/experiments/m32-full-teacher-20261020/exact-sharing
docker exec trt-asc env TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 /opt/trt-asc/venv/bin/python \
  "$base/screen.py" \
  /workspace/experiments/m32-full-teacher-20261020/sfa-78/rank-0.json \
  "$base/result.json"
