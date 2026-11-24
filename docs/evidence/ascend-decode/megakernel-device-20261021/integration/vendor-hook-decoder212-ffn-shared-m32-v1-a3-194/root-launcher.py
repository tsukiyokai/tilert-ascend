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
local = Path('/tmp/tilert-decoder212-ffn-diagnostic-staging-v1')
remote = task + '/vendor-hook-decoder212-ffn-diagnostic-v1'
frozen = dict(line.split('  ', 1)[::-1] for line in (local / 'frozen-inputs.sha256').read_text().splitlines())
for name in ('device_ffn_diagnostic.py', 'decoder_ffn.py', 'decoder_ffn_diagnostic.py', 'decoder_storage_ffn_diagnostic.py', 'ffn_active_token_join.so', 'nz_alias.so', 'nz-alias-manifest.json', 'device_ffn_diagnostic_run.sh', 'manifest.json', 'frozen-inputs.sha256'):
    expected = hashlib.sha256((local / name).read_bytes()).hexdigest() if (local / name).exists() else frozen[name]
    actual = subprocess.check_output(gate.SSH + ['sha256sum', remote + '/' + name], text=True).split()[0]
    assert actual == expected, name
manifest = json.loads(Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-bridge-build-v2/oracle-package-v4/manifest.json').read_text())
verify = 'import hashlib,json; from pathlib import Path; p=Path(' + repr(task + '/tests/glm52/whole_layer/ffn_bridge/oracle-package-v4/install') + '); expected=json.loads(' + repr(json.dumps(manifest['installed_sha256'])) + '); bad=[n for n,h in expected.items() if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h]; assert not bad,bad; print("Verified oracle files",len(expected))'
subprocess.run(gate.SSH + ['python3', '-c', shlex.quote(verify)], check=True)
memory = subprocess.check_output(gate.SSH + ['cat', '/proc/meminfo'], text=True)
available = int(next(line.split()[1] for line in memory.splitlines() if line.startswith('MemAvailable:')))
assert available > 96 * 1024 * 1024, available
name = 'vendor-hook-decoder212-ffn-shared-m32-v1'
(root / (name + '.host-memory.json')).write_text(json.dumps({'utc_epoch': time.time(), 'available_kib': available, 'container_limit_gib': 64}, indent=2) + '\n')
assert gate.vacant()
time.sleep(5)
assert gate.vacant()
users = subprocess.run(gate.SSH + ['fuser', '/dev/davinci4'], capture_output=True, text=True)
assert users.returncode == 1 and not users.stdout.strip() and not users.stderr.strip(), users
command = ['bash', remote + '/device_ffn_diagnostic_run.sh', '4', name, '--mode', 'int8', '--forward-widths', '32', '--forward-replays', '3', '--ffn-shared-parent', '/evidence/vendor-hook-decoder212-ffn-int8-wide-v2/decoder-ffn-layer6-m32-r2.pt']
(root / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
print('Starting shared original FFN observation versus native payload on physical die4', flush=True)
with (root / (name + '.launcher.log')).open('w') as log:
    result = subprocess.run(gate.SSH + [shlex.join(command)], stdout=log, stderr=subprocess.STDOUT)
print('Decoder asynchronous forward device exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
