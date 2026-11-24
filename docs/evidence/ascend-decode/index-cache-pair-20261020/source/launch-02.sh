#!/usr/bin/env bash
set -eo pipefail
source /workspace/environment/activate.sh
export ASCEND_CUSTOM_OPP_PATH="/workspace/experiments/gmm-fastpath-20261018/prod-kernel/overlay/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH:-}"
export ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0
python3 - <<'PY'
import json
from pathlib import Path
import subprocess
import sys
root = Path('/workspace/experiments/index-cache-pair-20261020')
receipt = json.loads((root / 'receipt.json').read_text())
audit = receipt['launch_audit']
command = [sys.executable, str(root / 'probe.py'), '--run-npu', '--device', '15',
           '--output', str(root / 'device-screen-02')]
for i, target in enumerate(audit['targets']):
    command = [sys.executable, audit['watcher'], target['path'],
               str(root / f'device-access-02-{i}.json'), *command]
raise SystemExit(subprocess.call(command))
PY
