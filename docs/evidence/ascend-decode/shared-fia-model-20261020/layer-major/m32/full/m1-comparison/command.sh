#!/usr/bin/env bash
set -euo pipefail
base=/workspace/experiments/m32-full-teacher-20261020
docker exec trt-asc /usr/bin/python3 -c '
from pathlib import Path
import json
base = Path("/workspace/experiments/m32-full-teacher-20261020")
exit_code = int((base / "sfa-78.exit").read_text().strip())
reports = [json.loads((base / "sfa-78" / f"rank-{rank}.json").read_text())
           for rank in range(16)]
assert all(report["full_model_executed"] and report["layers"] == 78
           and report["steps"] == 32 for report in reports)
print(json.dumps({"model_exit": exit_code, "complete_rank_reports": 16}))
'
docker exec trt-asc env TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 /opt/trt-asc/venv/bin/python \
  "$base/m1-comparison/compare.py" \
  "$base/m1-comparison/old/device-78x32-sfa-01" \
  "$base/sfa-78" "$base/m1-comparison/result.json" --layers 78
