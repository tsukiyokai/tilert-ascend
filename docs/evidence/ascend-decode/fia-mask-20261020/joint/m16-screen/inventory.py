import concurrent.futures
import datetime
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path('/tmp/fia-m16-20261020/inventory')
ROOT.mkdir(exist_ok=True)
HOSTS = ['a3-gs', 'a3-78', 'a3-84', 'a3-162', 'a3-194', 'a3-187']
REMOTE = r'''
import datetime, json, pathlib, subprocess, time

def run(args, timeout=12):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {'rc': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except Exception as e:
        return {'error': str(e)}
r = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
r['smi'] = [run(['npu-smi', 'info'])]
time.sleep(1)
r['smi'].append(run(['npu-smi', 'info']))
r['containers'] = run(['docker', 'ps', '--format', '{{.Names}}\t{{.Status}}\t{{.Image}}'])
r['processes'] = run(['ps', '-eo', 'pid,ppid,comm'])
r['memory'] = pathlib.Path('/proc/meminfo').read_text()
r['stacks'] = {}
for line in r['containers'].get('stdout', '').splitlines():
    name = line.split('\t')[0]
    if name.startswith('trt-asc'):
        code = "import importlib.metadata as m,json,pathlib; p=pathlib.Path('/opt/ascend/cann-9.2.0/aarch64-linux/ascend_toolkit_install.info'); print(json.dumps({'torch':m.version('torch'),'torch_npu':m.version('torch_npu'),'cann_info':p.read_text() if p.exists() else None,'vllm_source':pathlib.Path('/workspace/src/vllm-unified/vllm').is_dir(),'ascend_source':pathlib.Path('/workspace/src/vllm-ascend-unified/vllm_ascend').is_dir()}))"
        r['stacks'][name] = run(['docker', 'exec', name, 'python', '-c', code])
print(json.dumps(r))
'''

def parse_smi(value):
    rows = []
    for line in value.splitlines():
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) != 3 or not re.fullmatch(r'\d+\s+\d+', parts[0]):
            continue
        m = re.search(r'(\d+)\s*/\s*(\d+)\s*$', parts[2])
        if m is None:
            continue
        chip, die = map(int, parts[0].split())
        used, total = map(int, m.groups())
        rows.append({'die': die, 'chip': chip, 'aicore_pct': int(parts[2].split()[0]),
                     'used_mib': used, 'total_mib': total, 'free_mib': total-used})
    return rows

def fetch(host):
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=7', host,
                        'python3 -c ' + shlex.quote(REMOTE)], capture_output=True,
                       text=True, timeout=55)
    raw = {'host': host, 'rc': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    (ROOT / (host + '.raw.json')).write_text(json.dumps(raw, indent=2) + '\n')
    if p.returncode:
        return {'host': host, 'rc': p.returncode, 'stderr': p.stderr}
    r = json.loads(p.stdout)
    (ROOT / (host + '.json')).write_text(json.dumps(r, indent=2) + '\n')
    samples = [parse_smi(v.get('stdout', '')) for v in r['smi']]
    assert all(len(v) == 16 for v in samples), (host, samples)
    rows = []
    for die in range(16):
        values = [next(v for v in sample if v['die'] == die) for sample in samples]
        rows.append({'die': die, 'aicore_pct': [v['aicore_pct'] for v in values],
                     'free_mib_min': min(v['free_mib'] for v in values),
                     'used_mib_max': max(v['used_mib'] for v in values)})
    stacks = {}
    for name, v in r['stacks'].items():
        try:
            stacks[name] = json.loads(v['stdout'])
        except Exception:
            stacks[name] = v
    return {'host': host, 'utc': r['utc'], 'dies': rows, 'stacks': stacks,
            'containers': r['containers'],
            'mem_available': next(v for v in r['memory'].splitlines() if v.startswith('MemAvailable'))}

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    fs = {pool.submit(fetch, h): h for h in HOSTS}
    for f in concurrent.futures.as_completed(fs):
        try:
            results.append(f.result())
        except Exception as e:
            results.append({'host': fs[f], 'error': repr(e)})
results.sort(key=lambda r: HOSTS.index(r['host']))
summary = {'scope': 'One bounded parallel six-host inventory; two read-only npu-smi samples per host one second apart',
           'local_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'hosts': results}
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
for r in results:
    print(json.dumps({'host':r['host'], 'error':r.get('error'), 'utc':r.get('utc'),
                      'dies':r.get('dies'), 'stacks':r.get('stacks'),
                      'mem_available':r.get('mem_available')}))
