import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time
import sys

root = Path('/tmp/tilert-megakernel-validation-20261021')
spec = importlib.util.spec_from_file_location('idle_gate', root / 'two-die-idle-gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
gate.SSH[-1] = 'a3-194'
task = '/home/shanshan/tilert-megakernel-20261021'
local = Path('/tmp/tilert-tp-graph-staging-v1')
remote = task + '/vendor-hook-tp-graph-v1'
expected_manifest = '546f56c9bc0bb47d67d3848da7acf3d64f66409fde53784b11986aad24d27ea7'
assert hashlib.sha256((local/'manifest.json').read_bytes()).hexdigest() == expected_manifest
frozen = dict(line.split('  ', 1)[::-1] for line in (local/'frozen-inputs.sha256').read_text().splitlines())
for filename, expected in frozen.items():
    assert hashlib.sha256((local/filename).read_bytes()).hexdigest() == expected, filename
assert frozen['complete_ffn.so'] == '381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2'
assert frozen['layer_local_chain.so'] == '609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f'
assert frozen['local_chain_host.py'] == '70bfd7393e7ceb2a6921100be6057f0cea3f922c52d6e3df2e02ea398c66e6f9'
assert frozen['layer_tp_chain.so'] == 'c2b72e469c3ee80a5fe7d9df4a3cceedc0e65cca95e6f426aea762c213392462'
assert frozen['tp_host.py'] == 'c3caf00eaf135c963c5ef5e990ae309531e9802aa03574edbbdc0fd310534493'
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
name = 'vendor-hook-tp-graph-m12-v1'
(root / (name + '.host-memory.json')).write_text(json.dumps({'utc_epoch': time.time(), 'available_kib': available, 'container_limit_gib': 64}, indent=2) + '\n')
assert gate.vacant()
time.sleep(5)
assert gate.vacant()
command = ['bash', remote + '/device_tp_graph_run.sh', '6', '7', name,
           '--mode', 'int8', '--forward-widths', '12', '--forward-replays', '2',
           '--graph-residual-modes', 'provided', '--indexed-library',
           '/evidence/attention-wk-m1-build-v1/whole_layer_indexed_attention.so']
(root / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
cpu = json.loads((root / 'tp-graph-cpu-v1/report.json').read_text())
assert cpu['pass']
print('Starting two-rank whole-layer official graph comparison on physical dies6/7', flush=True)
def telemetry(label):
    command = gate.SSH + ['npu-smi', 'info']
    sample = subprocess.run(command, capture_output=True, text=True, timeout=20)
    (root / (name + '.telemetry-' + label + '.json')).write_text(json.dumps({
        'epoch': time.time(), 'command': command, 'returncode': sample.returncode,
        'stdout': sample.stdout, 'stderr': sample.stderr,
        'clock_control': 'No clock settings changed by this experiment'}, indent=2) + '\n')
telemetry('start')
with (root / (name + '.launcher.log')).open('w') as log:
    result = subprocess.run(gate.SSH + [shlex.join(command)], stdout=log, stderr=subprocess.STDOUT)
telemetry('end')
print('Two-rank layer device exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
