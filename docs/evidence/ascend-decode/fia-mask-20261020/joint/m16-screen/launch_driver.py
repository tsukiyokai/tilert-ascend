import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path('/tmp/fia-m16-launch-20261020')
ROOT.mkdir(exist_ok=True)
TARGET = '/workspace/experiments/fia-m16-20261020'

def run(args, timeout=15):
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'rc': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}

def rows(raw):
    result = {}
    for line in raw.splitlines():
        ps = [v.strip() for v in line.split('|') if v.strip()]
        if len(ps) != 3 or not re.fullmatch(r'\d+\s+\d+', ps[0]):
            continue
        m = re.search(r'(\d+)\s*/\s*(\d+)\s*$', ps[2])
        if m:
            used, total = map(int, m.groups())
            result[int(ps[0].split()[1])] = {
                'aicore_pct': int(ps[2].split()[0]), 'free_mib': total-used,
                'used_mib': used, 'total_mib': total}
    assert len(result) == 16
    return result

pre = [run(['npu-smi', 'info'])]
time.sleep(1)
pre.append(run(['npu-smi', 'info']))
parsed = [rows(v['stdout']) for v in pre]
eligible = [d for d in range(16) if min(v[d]['free_mib'] for v in parsed) >= 2048]
assert eligible, 'No die has 2GiB headroom'
quiet = [d for d in eligible if all(v[d]['aicore_pct'] == 0 for v in parsed)]
die = 15 if 15 in quiet else min(quiet or eligible,
    key=lambda d: (max(v[d]['aicore_pct'] for v in parsed),
                   -min(v[d]['free_mib'] for v in parsed)))
timing = die in quiet
audit = {'scope': 'Shared single-die; two launch snapshots; unlocked clocks; no reservation',
         'host': 'a3-194', 'die': die, 'timing_enabled': timing,
         'precheck': pre, 'chosen_die': [v[die] for v in parsed],
         'processes': run(['ps', '-eo', 'pid,ppid,comm']),
         'containers': run(['docker', 'ps', '--format', '{{.Names}}\t{{.Status}}\t{{.Image}}']),
         'memory': Path('/proc/meminfo').read_text()}
read = run(['docker', 'exec', 'trt-asc', 'cat', TARGET + '/screen.py'])
assert read['rc'] == 0
source = read['stdout']
assert hashlib.sha256(source.encode()).hexdigest() == '61b5d3964c00d3d27bc3b48f264407558419eb56570cbb4a6b4e00ac71b6f6dc'
if die != 15:
    source = source.replace("'device': 15", "'device': " + str(die))
    source = source.replace('set_device(15)', 'set_device(' + str(die) + ')')
    source = source.replace('get_device_properties(15)', 'get_device_properties(' + str(die) + ')')
    source = source.replace("torch.device('npu', 15)", "torch.device('npu', " + str(die) + ')')
if not timing:
    begin = source.index('        start = torch_npu.npu.Event(enable_timing=True)')
    finish = source.index('\nexcept Exception:', begin)
    source = (source[:begin] + "        assert sum(report['attention_calls_attempted'].values()) == 124\n"
              "        report['status'] = 'bounded_correctness_completed_timing_deferred'\n" + source[finish:])
launched = ROOT / 'screen-launched.py'
launched.write_text(source)
audit['screen_launched_sha256'] = hashlib.sha256(launched.read_bytes()).hexdigest()
cp = run(['docker', 'cp', str(launched), 'trt-asc:' + TARGET + '/screen-launched.py'])
assert cp['rc'] == 0, cp
(ROOT / 'launch-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps({'die':die, 'timing_enabled':timing, 'chosen_die':audit['chosen_die'],
                  'script_sha256':audit['screen_launched_sha256']}), flush=True)
started = time.perf_counter()
execution = run(['docker', 'exec', 'trt-asc', 'bash', TARGET + '/run.sh'], timeout=120)
audit['execution'] = execution
audit['execution_wall_s'] = time.perf_counter()-started
audit['postcheck'] = run(['npu-smi', 'info'])
audit['post_processes'] = run(['ps', '-eo', 'pid,ppid,comm'])
(ROOT / 'launch-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
cp = run(['docker', 'cp', str(ROOT / 'launch-audit.json'), 'trt-asc:' + TARGET + '/device/launch-audit.json'])
print(json.dumps({'rc':execution['rc'], 'execution_wall_s':audit['execution_wall_s'],
                  'audit_copy_rc':cp['rc']}), flush=True)
raise SystemExit(execution['rc'])
