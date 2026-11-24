import importlib.util
from pathlib import Path
import subprocess
import time

root = Path('/tmp/tilert-megakernel-validation-20261021')
spec = importlib.util.spec_from_file_location('idle_gate', root / 'run-pending-single-die.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
gate.SSH[-1] = 'a3-194'
remote = '/home/shanshan/tilert-megakernel-20261021'
for name, expected in [('check.py', 'df015159cf56cebf7ce4d0b36f7f3db0270195cfca855873412ca77c55021798'), ('run.sh', '46b23bfc8905db0de97cd5a8a7be1d3dd3e0b8a0f56f7f6afdadb8d5123351a4')]:
    actual = subprocess.check_output(gate.SSH + ['sha256sum', remote + '/vendor-hook-runtime/tests/glm52/whole_layer/indexer_wk_probe/' + name], text=True).split()[0]
    assert actual == expected, name
assert gate.vacant()
time.sleep(5)
assert gate.vacant()
users = subprocess.run(gate.SSH + ['fuser', '/dev/davinci4'], capture_output=True, text=True)
assert users.returncode == 1 and not users.stdout.strip() and not users.stderr.strip(), users
script = 'set -e\ncd ' + remote + '\nbash vendor-hook-runtime/tests/glm52/whole_layer/indexer_wk_probe/run.sh 4 captured-wk-v2 > logs/captured-wk-v2-launcher.log 2>&1\n'
print('Starting captured WK v2 on a3-194 physical die4', flush=True)
result = subprocess.run(gate.SSH + ['bash', '-s'], input=script, text=True)
print('WK launcher exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
