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
    probe + '/dispatch.py': '644aad02ec5c162c04ef8a235961467017ed4ccf56eed2e41c740c3fb0eb6886',
    probe + '/run_dispatch.sh': '30b4467e68f71d262a6e67a4610e785a9e4af3380dea5b39b769e854c80e92d0',
    probe + '/check.py': 'df015159cf56cebf7ce4d0b36f7f3db0270195cfca855873412ca77c55021798',
    task_root + '/logs/vendor-hook-index-diagnostic-bf16-v13/outer-index-diagnostic.pt': '0b6e5a0198e0423f4a1c644016089cc2723ddfd7768f8b4d11b7ac4e25008725',
}
for path, expected in identities.items():
    actual = subprocess.check_output(gate.SSH + ['sha256sum', path], text=True).split()[0]
    assert actual == expected, path
for kind in ('full-bf16', 'full-fp32', 'split-bf16'):
    assert gate.vacant()
    time.sleep(5)
    assert gate.vacant()
    users = subprocess.run(gate.SSH + ['fuser', '/dev/davinci4'], capture_output=True, text=True)
    assert users.returncode == 1 and not users.stdout.strip() and not users.stderr.strip(), users
    name = 'captured-wk-dispatch-' + kind + '-v1'
    command = ['bash', probe + '/run_dispatch.sh', '4', name, kind]
    (root / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
    print('Starting WK dispatch probe', kind, 'on a3-194 physical die4', flush=True)
    with (root / (name + '.launcher.log')).open('w') as output:
        result = subprocess.run(gate.SSH + [shlex.join(command)], stdout=output, stderr=subprocess.STDOUT)
    print('WK dispatch probe', kind, 'exit', result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
