"""Run the frozen three-rank layer fixture after CPU and idle-device checks."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time


ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
LOCAL = Path('/home/shanshan/.cache/tilert-megakernel-20261021/tp-tree-staging-v1')
TASK = '/home/shanshan/tilert-megakernel-20261021'
REMOTE = TASK + '/vendor-hook-tp-tree-v1'
DEVICES = (4, 6, 7)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('eager', 'graph'), required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--candidate-sha256', required=True)
    parser.add_argument('--cpu-report-sha256', required=True)
    parser.add_argument('--attempt', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    name = 'vendor-hook-tp-tree-' + args.kind + '-v' + str(args.attempt)
    assert not (ROOT / (name + '.command.json')).exists(), 'Use a new run identity'
    assert digest(LOCAL / 'manifest.json') == args.manifest_sha256
    pins = dict(line.split('  ', 1)[::-1]
                for line in (LOCAL / 'frozen-inputs.sha256').read_text().splitlines())
    for filename, expected in pins.items():
        assert digest(LOCAL / filename) == expected, filename
    assert pins['layer_tp_chain.so'] == args.candidate_sha256
    assert pins['complete_ffn.so'] == '381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2'
    assert pins['layer_local_chain.so'] == '609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f'
    cpu_path = ROOT / 'tp-tree-cpu-v1/report.json'
    assert digest(cpu_path) == args.cpu_report_sha256
    cpu = json.loads(cpu_path.read_text())
    assert cpu['pass'] and cpu['manifest_sha256'] == args.manifest_sha256
    assert cpu['candidate_sha256'] == args.candidate_sha256
    assert not cpu['device_execution'] and not cpu['mapped_davinci_devices']
    assert not cpu['npu_initialized_before'] and not cpu['npu_initialized_after']
    assert (cpu['torch'], cpu['torch_npu']) == ('2.12.0+cu130', '2.12.0')
    spec = importlib.util.spec_from_file_location('idle_gate', ROOT / 'two-die-idle-gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    gate.DEVICES = DEVICES
    gate.LOG = ROOT / (name + '.admission.jsonl')
    expected = {REMOTE + '/' + filename: sha for filename, sha in pins.items()}
    expected.update({TASK + '/communication-deps/shmem-build/lib/' + filename: sha
                     for filename, sha in cpu['shmem_libraries'].items()})
    expected[TASK + '/logs/attention-wk-m1-build-v1/whole_layer_indexed_attention.so'] = (
        '8deff750464e23bb871302c8626b86d37b60c28a00991d550f95dfcfd47ecb6f')
    oracle_path = Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/'
                       'ffn-bridge-build-v2/oracle-package-v4/manifest.json')
    oracle = json.loads(oracle_path.read_text())['installed_sha256']
    expected.update({TASK + '/tests/glm52/whole_layer/ffn_bridge/oracle-package-v4/install/' + filename: sha
                     for filename, sha in oracle.items()})
    code = ('import hashlib,json; from pathlib import Path; expected=json.loads('
            + repr(json.dumps(expected)) + '); '
            'bad=[n for n,h in expected.items() '
            'if hashlib.sha256(Path(n).read_bytes()).hexdigest()!=h]; '
            'assert not bad,bad; print("Verified device inputs",len(expected))')
    subprocess.run(gate.SSH + ['python3', '-'], input=code,
                   text=True, check=True)
    memory = subprocess.check_output(gate.SSH + ['cat', '/proc/meminfo'], text=True)
    available = int(next(line.split()[1] for line in memory.splitlines()
                         if line.startswith('MemAvailable:')))
    assert available > 96 * 1024 * 1024, available
    (ROOT / (name + '.host-memory.json')).write_text(json.dumps({
        'utc_epoch': time.time(), 'available_kib': available,
        'container_limit_gib': 64}, indent=2) + '\n')
    assert gate.vacant(), 'First idle-device sample refused'
    time.sleep(5)
    assert gate.vacant(), 'Second idle-device sample refused'
    launcher = 'device_tp_graph_run.sh' if args.kind == 'graph' else 'device_tp_run.sh'
    command = ['bash', REMOTE + '/' + launcher, *map(str, DEVICES), name,
               '--mode', 'int8', '--forward-widths', '1', '12', '--forward-replays', '8',
               '--graph-residual-modes', 'provided', '--indexed-library',
               '/evidence/attention-wk-m1-build-v1/whole_layer_indexed_attention.so']
    (ROOT / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
    (ROOT / (name + '.inputs.json')).write_text(json.dumps({
        'fixture_manifest_sha256': args.manifest_sha256,
        'candidate_sha256': args.candidate_sha256,
        'cpu_report_sha256': args.cpu_report_sha256,
        'runner_sha256': digest(Path(__file__))}, indent=2) + '\n')
    print('Starting three-rank layer', args.kind, 'on physical dies', DEVICES, flush=True)
    for label in ('start', 'end'):
        sample = subprocess.run(gate.SSH + ['npu-smi', 'info'],
                                text=True, capture_output=True, timeout=20)
        (ROOT / (name + '.telemetry-' + label + '.json')).write_text(json.dumps({
            'epoch': time.time(), 'returncode': sample.returncode,
            'stdout': sample.stdout, 'stderr': sample.stderr,
            'clock_control': 'No clock settings changed'}, indent=2) + '\n')
        if label == 'start':
            with (ROOT / (name + '.launcher.log')).open('w') as log:
                result = subprocess.Popen(gate.SSH + [shlex.join(command)],
                                          stdout=log, stderr=subprocess.STDOUT)
                container = 'shanshan-tilert-decoder212-' + name
                observer = '''import json,subprocess,time
from pathlib import Path
name=CONTAINER_NAME
command=['docker','inspect','--format','{{json .State}}',name]
p=subprocess.run(command,text=True,capture_output=True,timeout=10)
record={'epoch':time.time(),'container':name,'returncode':p.returncode,'stderr':p.stderr}
if p.returncode == 0:
 state=json.loads(p.stdout); record['state']=state; pid=state['Pid']
 process=Path('/proc')/str(pid)
 if pid and process.exists():
  record['status']='\n'.join(line for line in (process/'status').read_text().splitlines() if line.startswith(('Name:','State:','VmRSS:','Threads:')))
  record['cgroup']=(process/'cgroup').read_text()
  counters={}
  for line in record['cgroup'].splitlines():
   _,controllers,relative=line.split(':',2)
   base=Path('/sys/fs/cgroup')/relative.lstrip('/') if not controllers else Path('/sys/fs/cgroup')/controllers/relative.lstrip('/')
   for key in ('memory.current','memory.max','memory.events','memory.usage_in_bytes','memory.limit_in_bytes','memory.failcnt','pids.current','pids.events','cpu.stat'):
    path=base/key
    if path.is_file(): counters[str(path)]=path.read_text()
  record['counters']=counters
print(json.dumps(record))
'''.replace('CONTAINER_NAME', repr(container))
                while result.poll() is None:
                    try:
                        result.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        observation = subprocess.run(
                            gate.SSH + ['python3', '-'], input=observer,
                            text=True, capture_output=True, timeout=20)
                        with (ROOT / (name + '.container-state.jsonl')).open('a') as stream:
                            stream.write(json.dumps({
                                'epoch': time.time(), 'exit_code': observation.returncode,
                                'stdout': observation.stdout,
                                'stderr': observation.stderr}) + '\n')
    print('Three-rank layer', args.kind, 'exit', result.returncode, flush=True)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
