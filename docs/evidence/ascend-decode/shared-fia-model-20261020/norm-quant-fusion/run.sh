#!/usr/bin/env bash
set -eo pipefail
source /workspace/environment/activate.sh
set -u
base=/workspace/experiments/m32-full-teacher-20261020/norm-fusion
vendor=/workspace/experiments/gmm-fastpath-20261018/prod-kernel/overlay/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH="$vendor:$ASCEND_CUSTOM_OPP_PATH"
export ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
python3 - "$base" <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

root = Path(sys.argv[1])
prepared = json.loads((root / 'prepared.json').read_text())
expected = '23cb609c55a74eb8f0840e58444170b776b8c1b5258dc99a295930b157fae675'
assert hashlib.sha256((root / 'probe.py').read_bytes()).hexdigest() == expected
assert prepared['script_sha256'] == expected
samples = []
for sample in range(2):
    raw = subprocess.check_output(['npu-smi', 'info'], text=True)
    (root / f'device-01-preflight-{sample}.txt').write_text(raw)
    rows = re.findall(
        r'\|\s+(\d+)\s+(\d+)\s+\|\s+[0-9A-Fa-f:.]+\s+\|\s+(\d+)'
        r'\s+0\s*/\s*0\s+(\d+)\s*/\s*(\d+)', raw,
    )
    chosen = [row for row in rows if int(row[1]) == 9]
    assert len(chosen) == 1
    samples.append(chosen[0])
    if sample == 0:
        time.sleep(2)
passed = all(int(row[4]) - int(row[3]) >= 1536 for row in samples)
(root / 'device-01-preflight.json').write_text(json.dumps({
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'die': 9, 'samples': samples, 'passed': passed,
    'torch_quota_bytes': 256 * 1024 ** 2,
    'scope': 'Single-die numerical contrast; no timing',
}, indent=2) + '\n')
if not passed:
    raise SystemExit('Insufficient memory for single-die contrast')
PY
timeout --signal=TERM --kill-after=15s 180 python3 "$base/probe.py" run \
    --device npu:9 --output "$base/device-01"
