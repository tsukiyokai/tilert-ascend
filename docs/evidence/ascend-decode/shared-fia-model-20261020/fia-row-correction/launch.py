"""Run the reviewed functional replay after a fresh die15 preflight."""
import datetime
import json
from pathlib import Path
import re
import subprocess
import time

root = Path('/home/shanshan/trt-asc-194/experiments/m32-full-teacher-20261020/fia-row-correction')
assert (root / 'cpu-ready.json').is_file()
assert not (root / 'device-01').exists()
pattern = (r'\|\s+(\d+)\s+(\d+)\s+\|\s+([0-9A-Fa-f:.]+)\s+\|\s+(\d+)'
           r'\s+0\s*/\s*0\s+(\d+)\s*/\s*(\d+)')
samples = []
for sample in range(2):
    raw = subprocess.check_output(['npu-smi', 'info'], text=True)
    (root / f'preflight-{sample}.txt').write_text(raw)
    rows = re.findall(pattern, raw)
    assert len(rows) == 16, 'Could not map all16 logical dies'
    by_die = {int(physical_id): (int(chip), bus, int(util), int(used), int(total))
              for chip, physical_id, bus, util, used, total in rows}
    assert set(by_die) == set(range(16)), 'Ambiguous logical-die mapping'
    chip, bus, util, used, total = by_die[15]
    samples.append({'logical_die': 15, 'chip': chip, 'bus_id': bus,
                    'aicore_pct': util, 'free_mib': total - used})
    if sample == 0:
        time.sleep(1)
passed = all(row['aicore_pct'] == 0 and row['free_mib'] >= 4096
             for row in samples)
(root / 'preflight.json').write_text(json.dumps({
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'passed': passed, 'samples': samples, 'torch_quota_mib': 512,
}, indent=2) + '\n')
if not passed:
    raise SystemExit('Die15 utilization or free-memory preflight failed')
with (root / 'device-01.log').open('x') as log:
    result = subprocess.run([
        'docker', 'exec', 'trt-asc', 'bash',
        '/workspace/experiments/m32-full-teacher-20261020/fia-row-correction/run.sh',
    ], stdout=log, stderr=subprocess.STDOUT)
(root / 'launch-exit.json').write_text(json.dumps({
    'returncode': result.returncode,
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
}) + '\n')
raise SystemExit(result.returncode)
