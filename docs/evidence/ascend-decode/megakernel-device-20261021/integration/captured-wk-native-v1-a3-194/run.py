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
for name, expected in [('native_check.py', '0cbb52cd75af04150f86875f113ab579cbfcf1fb00190e6e4279b58d5cb536c5'), ('check.py', 'df015159cf56cebf7ce4d0b36f7f3db0270195cfca855873412ca77c55021798'), ('run_native.sh', 'e58df5d9d744004011f26c693dbef93ebf50c238670f8aa74ea814d7f9c662ac'), ('build-native-v2/wk_geometry.so', 'f28e12ef02f256f37c7ba30485fa72516a8bba5d373c43476f45bceb80970a5d')]:
    actual = subprocess.check_output(gate.SSH + ['sha256sum', remote + '/vendor-hook-runtime/tests/glm52/whole_layer/indexer_wk_probe/' + name], text=True).split()[0]
    assert actual == expected, name
assert gate.vacant()
time.sleep(5)
assert gate.vacant()
users = subprocess.run(gate.SSH + ['fuser', '/dev/davinci4'], capture_output=True, text=True)
assert users.returncode == 1 and not users.stdout.strip() and not users.stderr.strip(), users
script = 'set -e\ncd ' + remote + '\nbash vendor-hook-runtime/tests/glm52/whole_layer/indexer_wk_probe/run_native.sh 4 captured-wk-native-v1 > logs/captured-wk-native-v1-launcher.log 2>&1\n'
print('Starting native WK geometry probe on a3-194 physical die4', flush=True)
result = subprocess.run(gate.SSH + ['bash', '-s'], input=script, text=True)
print('WK launcher exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
