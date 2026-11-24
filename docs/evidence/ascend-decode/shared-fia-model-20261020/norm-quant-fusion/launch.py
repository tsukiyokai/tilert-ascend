"""Retain one bounded numerical contrast's terminal process result."""
import json
from pathlib import Path
import subprocess

root = Path('/home/shanshan/trt-asc-194/experiments/m32-full-teacher-20261020/norm-fusion')
with (root / 'device-01.log').open('x') as log:
    result = subprocess.run([
        'docker', 'exec', 'trt-asc', 'bash',
        '/workspace/experiments/m32-full-teacher-20261020/norm-fusion/run.sh',
    ], stdout=log, stderr=subprocess.STDOUT)
(root / 'device-01-launch-exit.json').write_text(
    json.dumps({'returncode': result.returncode}) + '\n',
)
print(json.dumps({'returncode': result.returncode}), flush=True)
raise SystemExit(result.returncode)
