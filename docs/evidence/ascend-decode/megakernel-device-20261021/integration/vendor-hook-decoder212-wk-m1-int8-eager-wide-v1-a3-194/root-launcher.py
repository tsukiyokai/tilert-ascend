import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

root = Path('/tmp/tilert-megakernel-validation-20261021')
spec = importlib.util.spec_from_file_location('idle_gate', root / 'run-pending-single-die.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
gate.SSH[-1] = 'a3-194'
task = '/home/shanshan/tilert-megakernel-20261021'
local = Path('/tmp/tilert-decoder212-storage-forward-staging-v2')
remote = task + '/vendor-hook-decoder212-storage-forward-v2'
for name in ('device_forward_storage_async.py', 'decoder_forward.py', 'decoder_storage_forward.py', 'nz_alias.so', 'nz-alias-manifest.json', 'device_storage_forward_run.sh', 'manifest.json', 'frozen-inputs.sha256'):
    expected = hashlib.sha256((local / name).read_bytes()).hexdigest()
    actual = subprocess.check_output(gate.SSH + ['sha256sum', remote + '/' + name], text=True).split()[0]
    assert actual == expected, name
manifest = json.loads(Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-bridge-build-v2/oracle-package-v4/manifest.json').read_text())
verify = 'import hashlib,json; from pathlib import Path; p=Path(' + repr(task + '/tests/glm52/whole_layer/ffn_bridge/oracle-package-v4/install') + '); expected=json.loads(' + repr(json.dumps(manifest['installed_sha256'])) + '); bad=[n for n,h in expected.items() if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h]; assert not bad,bad; print("Verified oracle files",len(expected))'
subprocess.run(gate.SSH + ['python3', '-c', shlex.quote(verify)], check=True)
candidate = task + '/logs/attention-wk-m1-build-v1'
for name, expected in {
    'whole_layer_indexed_attention.so': '8deff750464e23bb871302c8626b86d37b60c28a00991d550f95dfcfd47ecb6f',
    'manifest.json': 'db2f5af406d2bfd728f2e6f4550b336e0b8a3d1aee9682fe06b2815f6298ece9',
}.items():
    actual = subprocess.check_output(gate.SSH + ['sha256sum', candidate + '/' + name], text=True).split()[0]
    assert actual == expected, name
memory = subprocess.check_output(gate.SSH + ['cat', '/proc/meminfo'], text=True)
available = int(next(line.split()[1] for line in memory.splitlines() if line.startswith('MemAvailable:')))
assert available > 96 * 1024 * 1024, available
name = 'vendor-hook-decoder212-wk-m1-int8-eager-wide-v1'
(root / (name + '.host-memory.json')).write_text(json.dumps({'utc_epoch': time.time(), 'available_kib': available, 'container_limit_gib': 64}, indent=2) + '\n')
gate_files = {'decoder_gates.py': '0fba3c1c24893ae4da6d955c303aca23dce2af9da20e1064ca74a708e529e6b3', 'manifest.json':'81f4329aa130f07e580782e5956afc86539a1793da917361e14324c427acc6e8'}
for filename, expected in gate_files.items():
    assert hashlib.sha256((root/'wk-m1-decoder-gates-v1'/filename).read_bytes()).hexdigest()==expected
    actual=subprocess.check_output(gate.SSH+['sha256sum',task+'/wk-m1-decoder-gates-v1/'+filename],text=True).split()[0]
    assert actual==expected
assert gate.vacant()
time.sleep(5)
assert gate.vacant()
users = subprocess.run(gate.SSH + ['fuser', '/dev/davinci4'], capture_output=True, text=True)
assert users.returncode == 1 and not users.stdout.strip() and not users.stderr.strip(), users
command = ['python3', task + '/wk-m1-decoder-gates-v1/decoder_gates.py', 'eager', '4', name, '--execute']
(root / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
print('Starting original asynchronous two-layer decoder with corrected NZ storage descriptors with one HCCL metadata participant on physical die4', flush=True)
with (root / (name + '.launcher.log')).open('w') as log:
    result = subprocess.run(gate.SSH + [shlex.join(command)], stdout=log, stderr=subprocess.STDOUT)
print('Decoder asynchronous forward device exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
