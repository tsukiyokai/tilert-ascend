import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
TASK = '/home/shanshan/tilert-megakernel-20261021'
NAME = 'vendor-hook-decoder212-ffn-down-n512-v1'
REMOTE = TASK + '/' + NAME
LOCAL = Path('/tmp/tilert-decoder212-ffn-down-n512-staging-v1')
LOCAL.mkdir()
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes']
files = ['device_ffn.py', 'decoder_ffn.py', 'decoder_storage_ffn.py',
         'ffn_down_n512.so', 'nz_alias.so', 'nz-alias-manifest.json',
         'device_ffn_run.sh', 'manifest.json', 'frozen-inputs.sha256',
         'cpu_preflight.py', 'test_scale.py', 'stage_down_n512.py', 'expected-tilings.json']
raw = subprocess.check_output(SSH + ['a3-194', shlex.join(['tar', '-czf', '-', '-C', REMOTE] + files)])
with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as tar:
    tar.extractall(LOCAL, filter='data')
frozen = dict(line.split('  ', 1)[::-1] for line in (LOCAL/'frozen-inputs.sha256').read_text().splitlines())
for name in files:
    if name != 'frozen-inputs.sha256':
        assert hashlib.sha256((LOCAL/name).read_bytes()).hexdigest() == frozen[name], name
subprocess.run(SSH + ['a3-78', 'mkdir', REMOTE], check=True)
subprocess.run(SSH + ['a3-78', shlex.join(['tar', '-xzf', '-', '-C', REMOTE])], input=raw, check=True)
command = json.loads((ROOT / 'decoder-ffn-original-shared-cpu-v1.command.json').read_text())
command = [s.replace('ffn-original-shared', 'ffn-down-n512') for s in command]
name = 'decoder-ffn-down-n512-cpu-v1'
(ROOT / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
with (ROOT / (name + '.log')).open('w') as log:
    subprocess.run(SSH + ['a3-78', shlex.join(command)], check=True, stdout=log, stderr=subprocess.STDOUT)
for name in ('cpu-tests.log', 'cpu-preflight.json'):
    data = subprocess.check_output(SSH + ['a3-78', 'cat', REMOTE + '/' + name])
    (LOCAL / name).write_bytes(data)
report = json.loads((LOCAL/'cpu-preflight.json').read_text())
assert report['pass'] and not report['npu_initialized']
assert report['library_sha256'] == 'c90b3666a1697070eda19d5b3e133e9ed58f36fe429abefceae6ffe102a6a032'
out = Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/decoder-ffn-down-n512-build-v1')
out.mkdir()
for name in files + ['cpu-tests.log', 'cpu-preflight.json']:
    if not name.endswith('.so'):
        shutil.copy2(LOCAL/name, out/name)
shutil.copy2(__file__, out/'root-preflight.py')
print(json.dumps({'pass': True, 'files_frozen': len(frozen), 'library_sha256': report['library_sha256'],
                  'manifest_sha256': hashlib.sha256((LOCAL/'manifest.json').read_bytes()).hexdigest(),
                  'decoder_sha256': frozen['decoder_ffn.py'], 'npu_initialized': False}, indent=2))
