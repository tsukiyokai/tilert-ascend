import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

root = Path('/tmp/tilert-megakernel-validation-20261021')
spec = importlib.util.spec_from_file_location('idle_gate', root / 'two-die-idle-gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
gate.SSH[-1] = 'a3-194'
gate.DEVICES = (4,)
gate.LOG = root / 'ffn-down-n1024-hot-b24-v1-admission.jsonl'
task = '/home/shanshan/tilert-megakernel-20261021'
local = root / 'vendor-hook-decoder212-ffn-down-n1024-hot-v1'
remote = task + '/vendor-hook-decoder212-ffn-down-n1024-hot-v1'
expected_manifest = '7799dff345f63271a65ca86df6efd8217a75b96aa13401a37b282d8cffcb7212'
assert hashlib.sha256((local / 'manifest.json').read_bytes()).hexdigest() == expected_manifest
frozen = dict(line.split('  ', 1)[::-1] for line in (local / 'frozen-inputs.sha256').read_text().splitlines())
for filename, expected in frozen.items():
    assert hashlib.sha256((local / filename).read_bytes()).hexdigest() == expected, filename
assert frozen['ffn_down_n1024.so'] == 'bbd52e425a1e94bb199e464e3bf6001c4420e8238afe993e8741e38af50b77ef'
assert frozen['decoder_ffn.py'] == 'e55024e0276e828b0746d5fedd02a83352770d11aa9bd5d4c81616f911dad1b0'
verify = 'from pathlib import Path; import hashlib,json; p=Path(' + repr(remote) + '); identities=json.loads(' + repr(json.dumps(frozen)) + '); bad=[n for n,h in identities.items() if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h]; assert not bad,bad; print("Verified original FFN fixture files",len(identities))'
subprocess.run(gate.SSH + ['python3', '-c', shlex.quote(verify)], check=True)
cpu = json.loads((root / 'ffn-down-n1024-hot-cpu-v1/cpu-preflight.json').read_text())
assert cpu['pass'] and not cpu['npu_initialized']
assert cpu['torch'] == '2.12.0+cu130' and cpu['torch_npu'] == '2.12.0'
assert cpu['library_sha256'] == frozen['ffn_down_n1024.so']
manifest = json.loads(Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-bridge-build-v2/oracle-package-v4/manifest.json').read_text())
verify = 'import hashlib,json; from pathlib import Path; p=Path(' + repr(task + '/tests/glm52/whole_layer/ffn_bridge/oracle-package-v4/install') + '); expected=json.loads(' + repr(json.dumps(manifest['installed_sha256'])) + '); bad=[n for n,h in expected.items() if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h]; assert not bad,bad; print("Verified oracle files",len(expected))'
subprocess.run(gate.SSH + ['python3', '-c', shlex.quote(verify)], check=True)
memory = subprocess.check_output(gate.SSH + ['cat', '/proc/meminfo'], text=True)
available = int(next(line.split()[1] for line in memory.splitlines() if line.startswith('MemAvailable:')))
assert available > 96 * 1024 * 1024, available
name = 'ffn-down-n1024-hot-b24-v1'
(root / (name + '.host-memory.json')).write_text(json.dumps({'utc_epoch': time.time(), 'available_kib': available, 'container_limit_gib': 64}, indent=2) + '\n')
assert gate.vacant()
time.sleep(5)
assert gate.vacant()
command = ['bash', remote + '/device_ffn_run.sh', '4', name, '--mode', 'int8', '--forward-widths', '17', '32', '--forward-replays', '8', '--native-blocks', '24']
(root / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
print('Starting actual decoder norm/MLP versus complete native FFN on physical die4', flush=True)
with (root / (name + '.launcher.log')).open('w') as log:
    result = subprocess.run(gate.SSH + [shlex.join(command)], stdout=log, stderr=subprocess.STDOUT)
print('Decoder asynchronous forward device exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
