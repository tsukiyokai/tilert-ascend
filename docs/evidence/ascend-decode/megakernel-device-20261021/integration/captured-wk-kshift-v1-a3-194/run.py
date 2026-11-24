import importlib.util
from pathlib import Path
import json
import shlex
import subprocess
import time

root = Path('/tmp/tilert-megakernel-validation-20261021')
spec = importlib.util.spec_from_file_location('idle_gate', root / 'run-pending-single-die.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
gate.SSH[-1] = 'a3-194'
task_root = '/home/shanshan/tilert-megakernel-20261021'
probe = task_root + '/vendor-hook-runtime/tests/glm52/whole_layer/indexer_wk_probe'
identities = {
    probe + '/run_kshift.sh': 'd2738ac047680f2c8763a401eedfe6eebc90c0798785214d0251fa3ab6fd0499',
    probe + '/kshift_check.py': 'ec5c14dd9299215a7ac1729034824204fc990c70927c7cc41917f02e11973d2c',
    probe + '/check.py': 'df015159cf56cebf7ce4d0b36f7f3db0270195cfca855873412ca77c55021798',
    probe + '/build-kshift-v2/wk_kshift.so': 'd0a7b654c9b02b253554c169ca774288f64b1f018cf9229c8340ce846a4807f3',
    task_root + '/logs/vendor-hook-index-diagnostic-bf16-v13/outer-index-diagnostic.pt': '0b6e5a0198e0423f4a1c644016089cc2723ddfd7768f8b4d11b7ac4e25008725',
}
for path, expected in identities.items():
    actual = subprocess.check_output(gate.SSH + ['sha256sum', path], text=True).split()[0]
    assert actual == expected, path
assert gate.vacant(), 'Die is occupied'
time.sleep(5)
assert gate.vacant(), 'Die is occupied'
users = subprocess.run(gate.SSH + ['fuser', '/dev/davinci4'], capture_output=True, text=True)
assert users.returncode == 1 and not users.stdout.strip() and not users.stderr.strip(), users
name = 'captured-wk-kshift-v1'
command = ['bash', probe + '/run_kshift.sh', '4', name]
(root / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
print('Starting captured natural/shifted WK on a3-194 physical die4', flush=True)
with (root / (name + '.launcher.log')).open('w') as output:
    result = subprocess.run(gate.SSH + [shlex.join(command)], stdout=output, stderr=subprocess.STDOUT)
print('WK Kshift exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
