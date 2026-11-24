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
for name, expected in [('check_fp32.py', '642d054c7e61579b8fbddd705a9e3ffdbafecb94a4df65ff95ae8955f07e0184'), ('run_fp32.sh', '4c7d497fb7a4f5d5f201fde21e897e2c503f2a30799736750527e2bb95b5d0c1')]:
    actual = subprocess.check_output(gate.SSH + ['sha256sum', remote + '/vendor-hook-runtime/tests/glm52/whole_layer/indexer_wk_probe/' + name], text=True).split()[0]
    assert actual == expected, name
assert gate.vacant()
time.sleep(5)
assert gate.vacant()
users = subprocess.run(gate.SSH + ['fuser', '/dev/davinci4'], capture_output=True, text=True)
assert users.returncode == 1 and not users.stdout.strip() and not users.stderr.strip(), users
script = 'set -e\ncd ' + remote + '\nbash vendor-hook-runtime/tests/glm52/whole_layer/indexer_wk_probe/run_fp32.sh 4 captured-wk-fp32-v1 > logs/captured-wk-fp32-v1-launcher.log 2>&1\n'
print('Starting captured WK FP32-output probe on a3-194 physical die4', flush=True)
result = subprocess.run(gate.SSH + ['bash', '-s'], input=script, text=True)
print('WK launcher exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
