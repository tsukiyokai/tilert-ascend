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
local = Path('/tmp/tilert-wk-m12-kshift-v1')
remote = '/home/shanshan/tilert-megakernel-20261021/wk-m12-kshift-v1'
manifest = json.loads((local / 'manifest.json').read_text())
for name, expected in manifest['sources'].items():
    assert hashlib.sha256((local / name).read_bytes()).hexdigest() == expected, name
    actual = subprocess.check_output(gate.SSH + ['sha256sum', remote + '/' + name], text=True).split()[0]
    assert actual == expected, name
for name in ('manifest.json', 'frozen-inputs.sha256'):
    expected = hashlib.sha256((local / name).read_bytes()).hexdigest()
    actual = subprocess.check_output(gate.SSH + ['sha256sum', remote + '/' + name], text=True).split()[0]
    assert actual == expected, name
assert manifest['capture_sha256'] == 'fb187d933ca499bec2ecc938184949775025ccbc90e0de894503e1dba2bae0bb'
assert gate.vacant()
time.sleep(5)
assert gate.vacant()
users = subprocess.run(gate.SSH + ['fuser', '/dev/davinci4'], capture_output=True, text=True)
assert users.returncode == 1 and not users.stdout.strip() and not users.stderr.strip(), users
name = 'captured-wk-m12-kshift-v1'
command = ['bash', remote + '/run_kshift.sh', '4', name]
(root / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
print('Starting captured M12 natural/shifted K traversal on idle physical die4', flush=True)
with (root / (name + '.launcher.log')).open('w') as log:
    result = subprocess.run(gate.SSH + [shlex.join(command)], stdout=log, stderr=subprocess.STDOUT)
print('Captured M12 K-traversal exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
