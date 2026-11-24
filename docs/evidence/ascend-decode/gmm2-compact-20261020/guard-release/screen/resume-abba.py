"""Run the four release arms only after a fresh die0 occupancy check."""

import datetime
import json
from pathlib import Path
import re
import subprocess
import time


ROOT = Path('/home/shanshan/trt-asc-194/experiments/gmm2-compact-20261020/guard-release-screen')
for arm in ('B2', 'A2'):
    samples = []
    for _ in range(2):
        raw = subprocess.check_output(['npu-smi', 'info'], text=True)
        matches = []
        for line in raw.splitlines():
            parts = [value.strip() for value in line.split('|') if value.strip()]
            if len(parts) == 3 and re.fullmatch(r'\d+\s+0', parts[0]):
                used, total = map(int, re.search(r'(\d+)\s*/\s*(\d+)\s*$', parts[2]).groups())
                matches.append({'aicore': int(parts[2].split()[0]),
                                'free_mib': total - used})
        assert len(matches) == 1
        samples.append(matches[0])
        (ROOT / f'{arm}-smi-{len(samples)}.txt').write_text(raw)
        time.sleep(1)
    preflight = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'die': 0, 'samples': samples}
    (ROOT / f'{arm}-preflight.json').write_text(json.dumps(preflight, indent=2))
    assert all(s['aicore'] == 0 and s['free_mib'] >= 6000 for s in samples), preflight
    command = ('timeout --kill-after=10 300 bash '
               '/workspace/experiments/gmm2-compact-20261020/'
               f'guard-release-screen/run-arm.sh {arm} npu:0')
    print(json.dumps({'arm': arm, 'event': 'launch', 'preflight': preflight}), flush=True)
    with (ROOT / f'{arm}.log').open('w') as log:
        result = subprocess.run(['docker', 'exec', 'trt-asc', 'bash', '-lc', command],
                                stdout=log, stderr=subprocess.STDOUT)
    (ROOT / f'{arm}-exit.json').write_text(json.dumps({'returncode': result.returncode}))
    print(json.dumps({'arm': arm, 'event': 'exit', 'returncode': result.returncode}), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
