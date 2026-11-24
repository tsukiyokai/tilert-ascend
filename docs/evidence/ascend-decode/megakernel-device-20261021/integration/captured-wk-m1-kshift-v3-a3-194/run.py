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
local = root / 'wk-m1-kshift-v3'
remote = '/home/shanshan/tilert-megakernel-20261021/wk-m1-kshift-v3'
manifest = json.loads((local / 'manifest.json').read_text())
for name, expected in manifest['sources'].items():
    assert hashlib.sha256((local / name).read_bytes()).hexdigest() == expected, name
    actual = subprocess.check_output(gate.SSH + ['sha256sum', remote + '/' + name], text=True).split()[0]
    assert actual == expected, name
for name in ('manifest.json', 'frozen-inputs.sha256'):
    expected = hashlib.sha256((local / name).read_bytes()).hexdigest()
    actual = subprocess.check_output(gate.SSH + ['sha256sum', remote + '/' + name], text=True).split()[0]
    assert actual == expected, name
assert manifest['parent_manifest_sha256'] == '162f84f3aa282feec5d011926faf04132f5873f96a9e0498299b89de8188d2f8'
assert manifest['library_sha256'] == '8493189945cd922ac2cff9a7653bccd1b5ad03bcf7574ea1278ef91fd44594a8'
assert hashlib.sha256((local/'manifest.json').read_bytes()).hexdigest() == '2dfb247ebd29be6b5fb4834a99c3fe5c554208df05c4ffbd57a30d67a8c5b8e4'
assert manifest['sources']['kshift_check.py'] == '0bec490083d74c526c6b55b456b5a26b693b776168cdf7e48f7efc9d8aa69564'
assert gate.vacant()
time.sleep(5)
assert gate.vacant()
users = subprocess.run(gate.SSH + ['fuser', '/dev/davinci4'], capture_output=True, text=True)
assert users.returncode == 1 and not users.stdout.strip() and not users.stderr.strip(), users
name = 'captured-wk-m1-kshift-v3'
command = ['bash', remote + '/run_kshift.sh', '4', name]
(root / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
print('Starting captured M1 natural/shifted K traversal on idle physical die4', flush=True)
with (root / (name + '.launcher.log')).open('w') as log:
    result = subprocess.run(gate.SSH + [shlex.join(command)], stdout=log, stderr=subprocess.STDOUT)
print('Captured M1 K-traversal exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
