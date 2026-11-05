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
local = Path('/tmp/vendor-hook-decoder212-graph-index-v3')
remote = task + '/vendor-hook-decoder212-graph-index-v3'
for name in ('device_graph_index.py', 'decoder_graph_index_diagnostic.py', 'decoder_graph.py', 'decoder_forward.py', 'decoder_storage_graph.py', 'nz_alias.so', 'nz-alias-manifest.json', 'device_graph_index_run.sh', 'manifest.json', 'frozen-inputs.sha256'):
    expected = hashlib.sha256((local / name).read_bytes()).hexdigest()
    actual = subprocess.check_output(gate.SSH + ['sha256sum', remote + '/' + name], text=True).split()[0]
    assert actual == expected, name
manifest = json.loads(Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-bridge-build-v2/oracle-package-v4/manifest.json').read_text())
verify = 'import hashlib,json; from pathlib import Path; p=Path(' + repr(task + '/tests/glm52/whole_layer/ffn_bridge/oracle-package-v4/install') + '); expected=json.loads(' + repr(json.dumps(manifest['installed_sha256'])) + '); bad=[n for n,h in expected.items() if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h]; assert not bad,bad; print("Verified oracle files",len(expected))'
subprocess.run(gate.SSH + ['python3', '-c', shlex.quote(verify)], check=True)
candidate = task + '/logs/attention-wk-m12-build-v1'
for name, expected in {
    'whole_layer_indexed_attention.so': '5f8868bf96efcfd5a53f17890ce8b619287d4463386a2f020c80546332edfabc',
    'manifest.json': '8a6641c3a9fcd04c36f66e0d2d08b6a967667f5082f478ced5fd8ba94d5ac90c',
}.items():
    actual = subprocess.check_output(gate.SSH + ['sha256sum', candidate + '/' + name], text=True).split()[0]
    assert actual == expected, name
for name, expected in {
    'manifest.json': 'dc370c4463f4401374d2ad645ba32380c84b4e53463d52228a588207cfe446d1',
    'frozen-inputs.sha256': 'ee01eec16060ad2e6d366accc7e85240d9af8e00e412ed635457ce2681c8e492',
    'decoder_graph_index_diagnostic.py': '637aa45165021ad4ce970e2475a6cb369f8c215be746d53449e94a34857f38f5',
    'device_graph_index_run.sh': 'b9db0f2e644c7970189ef1a6646c843d6a8d2abc22ebbbc4cd3da01a917dd169',
}.items():
    assert hashlib.sha256((local / name).read_bytes()).hexdigest() == expected, name
for name, expected in {
    'decoder-graph-int8-m1-provided-r2-native.pt': 'e84b65134ba32de1168e285ab32ce9773e2011b68c5c2d0879bfbc1b6839905a',
    'decoder-graph.json': 'a47fa508986e46bbbe0ceacfc6a85e7538f381f9d254ff8217d8efb1d05a9010',
}.items():
    path = task + '/logs/vendor-hook-decoder212-graph-int8-wide-v1/' + name
    actual = subprocess.check_output(gate.SSH + ['sha256sum', path], text=True).split()[0]
    assert actual == expected, name
memory = subprocess.check_output(gate.SSH + ['cat', '/proc/meminfo'], text=True)
available = int(next(line.split()[1] for line in memory.splitlines() if line.startswith('MemAvailable:')))
assert available > 96 * 1024 * 1024, available
name = 'vendor-hook-decoder212-graph-index-int8-m1-v3'
(root / (name + '.host-memory.json')).write_text(json.dumps({'utc_epoch': time.time(), 'available_kib': available, 'container_limit_gib': 64}, indent=2) + '\n')
assert gate.vacant()
time.sleep(5)
assert gate.vacant()
users = subprocess.run(gate.SSH + ['fuser', '/dev/davinci4'], capture_output=True, text=True)
assert users.returncode == 1 and not users.stdout.strip() and not users.stderr.strip(), users
command = ['bash', remote + '/device_graph_index_run.sh', '4', name, '--mode', 'int8', '--forward-widths', '1', '--forward-replays', '3', '--graph-residual-modes', 'provided', '--indexed-library', '/evidence/attention-wk-m12-build-v1/whole_layer_indexed_attention.so', '--graph-index-parent', '/evidence/vendor-hook-decoder212-graph-int8-wide-v1/decoder-graph-int8-m1-provided-r2-native.pt', '--graph-index-parent-report', '/evidence/vendor-hook-decoder212-graph-int8-wide-v1/decoder-graph.json']
(root / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
print('Starting retained-indexer diagnostic of original Decoder graph M1 failure on physical die4', flush=True)
with (root / (name + '.launcher.log')).open('w') as log:
    result = subprocess.run(gate.SSH + [shlex.join(command)], stdout=log, stderr=subprocess.STDOUT)
print('Decoder asynchronous forward device exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
