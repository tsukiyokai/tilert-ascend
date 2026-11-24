"""Admit one die for the frozen N512/N1024 local-layer graph comparison."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time


ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
TASK = '/home/shanshan/tilert-megakernel-20261021'
NAME = 'layer-down-n1024-timing-v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


handoff_path = ROOT / 'layer-down-n1024-timing-handoff.json'
assert digest(handoff_path) == '680e5b6041f556d39d406b72bdb11f1204e0a76769c569166c0c53ae6fe4c48b'
handoff = json.loads(handoff_path.read_text())
local = Path(handoff['frozen_local'])
remote = handoff['frozen_remote'].split(':', 1)[1]
assert digest(local / 'manifest.json') == handoff['manifest_sha256']
frozen = dict(line.split('  ', 1)[::-1]
              for line in (local / 'frozen-inputs.sha256').read_text().splitlines())
assert len(frozen) == 890
for filename, expected in frozen.items():
    assert digest(local / filename) == expected, filename
assert frozen['layer_local_chain.so'] == '609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f'
assert frozen['local_chain_host.py'] == '70bfd7393e7ceb2a6921100be6057f0cea3f922c52d6e3df2e02ea398c66e6f9'
assert frozen['complete_ffn.so'] == 'bbd52e425a1e94bb199e464e3bf6001c4420e8238afe993e8741e38af50b77ef'
assert frozen['control_ffn.so'] == 'c90b3666a1697070eda19d5b3e133e9ed58f36fe429abefceae6ffe102a6a032'
assert digest(handoff['cpu_report']) == handoff['cpu_report_sha256']
cpu = json.loads(Path(handoff['cpu_report']).read_text())
assert cpu['pass'] and cpu['contracts_run'] == 18
assert cpu['decoder_files_verified'] == 890 and cpu['runtime_files_verified'] == 7953
assert not cpu['npu_initialized_before'] and not cpu['npu_initialized_after']
assert not cpu['device_execution'] and not cpu['mapped_davinci_devices']
assert cpu['libraries'] == handoff['libraries']
assert len(cpu['cli_controls']) == 4 and len(cpu['schemas']) == 11
assert cpu['decoder_manifest_sha256'] == handoff['manifest_sha256']
cpu_command = json.loads((Path(handoff['cpu_report']).parent / 'command.json').read_text())
assert not any(value == '--device' or value.startswith('--device=') for value in cpu_command)

spec = importlib.util.spec_from_file_location('idle_gate', ROOT / 'two-die-idle-gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
gate.DEVICES = (4,)
gate.LOG = ROOT / (NAME + '.admission.jsonl')


def verify_remote(directory, identities):
    source = (
        'from pathlib import Path; import hashlib,json; p=Path(' + repr(directory) + '); '
        'expected=json.loads(' + repr(json.dumps(identities)) + '); '
        'bad=[n for n,h in expected.items() if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h]; '
        'assert not bad,bad; print("Verified files",len(expected))')
    subprocess.run(gate.SSH + ['python3', '-c', shlex.quote(source)], check=True)


verify_remote(remote, frozen)
oracle = json.loads(Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-bridge-build-v2/oracle-package-v4/manifest.json').read_text())
assert len(oracle['installed_sha256']) == 51
verify_remote(TASK + '/tests/glm52/whole_layer/ffn_bridge/oracle-package-v4/install',
              oracle['installed_sha256'])
verify_remote(TASK + '/logs/attention-wk-m1-build-v1', {
    'whole_layer_indexed_attention.so': '8deff750464e23bb871302c8626b86d37b60c28a00991d550f95dfcfd47ecb6f',
    'manifest.json': 'db2f5af406d2bfd728f2e6f4550b336e0b8a3d1aee9682fe06b2815f6298ece9'})
memory = subprocess.check_output(gate.SSH + ['cat', '/proc/meminfo'], text=True)
available = int(next(line.split()[1] for line in memory.splitlines()
                     if line.startswith('MemAvailable:')))
assert available > 96 * 1024 * 1024, available
(ROOT / (NAME + '.host-memory.json')).write_text(json.dumps({
    'utc_epoch': time.time(), 'available_kib': available, 'container_limit_gib': 64}, indent=2) + '\n')
assert gate.vacant(), 'First idle sample refused'
time.sleep(5)
assert gate.vacant(), 'Second idle sample refused'
command = handoff['argv']
assert command[:4] == ['bash', remote + '/device_local_chain_run.sh', '4', NAME]
(ROOT / (NAME + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')


def telemetry(label):
    command = gate.SSH + ['npu-smi', 'info']
    sample = subprocess.run(command, capture_output=True, text=True, timeout=20)
    (ROOT / (NAME + '.telemetry-' + label + '.json')).write_text(json.dumps({
        'epoch': time.time(), 'command': command, 'returncode': sample.returncode,
        'stdout': sample.stdout, 'stderr': sample.stderr,
        'clock_control': 'No clock settings changed'}, indent=2) + '\n')


print('Starting N512/N1024 whole-layer paired graphs on physical die 4', flush=True)
telemetry('start')
with (ROOT / (NAME + '.launcher.log')).open('w') as log:
    result = subprocess.run(gate.SSH + [shlex.join(command)], stdout=log, stderr=subprocess.STDOUT)
telemetry('end')
print('Local-layer Down timing device exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
