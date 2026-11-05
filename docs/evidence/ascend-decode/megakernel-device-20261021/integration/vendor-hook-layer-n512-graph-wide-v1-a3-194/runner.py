import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time
import sys

root = Path('/tmp/tilert-megakernel-validation-20261021')
spec = importlib.util.spec_from_file_location('idle_gate', root / 'run-pending-single-die.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
gate.SSH[-1] = 'a3-194'
task = '/home/shanshan/tilert-megakernel-20261021'
local = Path('/tmp/tilert-layer-n512-graph-staging-v1')
remote = task + '/vendor-hook-layer-n512-graph-v1'
expected_manifest = '6988778ffa2a0b593ed8924d3386a9a2efb6ffa2a6a2957781f1dced56ce49a5'
assert hashlib.sha256((local/'manifest.json').read_bytes()).hexdigest() == expected_manifest
frozen = dict(line.split('  ', 1)[::-1] for line in (local/'frozen-inputs.sha256').read_text().splitlines())
for filename, expected in frozen.items():
    assert hashlib.sha256((local/filename).read_bytes()).hexdigest() == expected, filename
assert frozen['complete_ffn.so'] == 'c90b3666a1697070eda19d5b3e133e9ed58f36fe429abefceae6ffe102a6a032'
assert frozen['layer_local_chain.so'] == '6a326292287a17254b48376400e1b852093ebfe10fbf228938336693402aa528'
assert frozen['local_chain_host.py'] == 'e5d0179b36bf983ba2b7f522cc0b35143d0e898c0ce3af8f72452f75438f2cb2'
verify = 'from pathlib import Path; import hashlib,json; p=Path(' + repr(remote) + '); identities=json.loads(' + repr(json.dumps(frozen)) + '); bad=[n for n,h in identities.items() if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h]; assert not bad,bad; print("Verified full Decoder graph files",len(identities))'
subprocess.run(gate.SSH + ['python3', '-c', shlex.quote(verify)], check=True)
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
name = 'vendor-hook-layer-n512-graph-wide-v1'
(root / (name + '.host-memory.json')).write_text(json.dumps({'utc_epoch': time.time(), 'available_kib': available, 'container_limit_gib': 64}, indent=2) + '\n')
assert gate.vacant()
time.sleep(5)
assert gate.vacant()
users = subprocess.run(gate.SSH + ['fuser', '/dev/davinci4'], capture_output=True, text=True)
assert users.returncode == 1 and not users.stdout.strip() and not users.stderr.strip(), users
command = ['bash', remote + '/device_local_graph_run.sh', '4', name,
           '--mode', 'int8', '--forward-widths', '1', '2', '12', '32', '--forward-replays', '8',
           '--graph-residual-modes', 'provided', '--indexed-library',
           '/evidence/attention-wk-m1-build-v1/whole_layer_indexed_attention.so']
(root / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
print('Starting official ACLGraphWrapper local layer chain numerical comparison on physical die4', flush=True)
with (root / (name + '.launcher.log')).open('w') as log:
    result = subprocess.run(gate.SSH + [shlex.join(command)], stdout=log, stderr=subprocess.STDOUT)
print('Local layer chain device exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
