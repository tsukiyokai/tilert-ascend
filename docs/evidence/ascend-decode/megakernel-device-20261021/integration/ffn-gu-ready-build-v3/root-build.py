import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


root = Path('/tmp/tilert-megakernel-validation-20261021')
source = Path('tests/glm52/whole_layer/ffn_gu_ready').resolve()
task = '/home/shanshan/tilert-megakernel-20261021'
remote = task + '/tests/glm52/whole_layer/ffn_gu_ready'
ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes', 'a3-78']
files = ['prepare.py', 'build.py', 'test_prepare.py', 'README.md', 'gu_ready_contract.h', 'gu_ready_native.h']
subprocess.run(ssh + ['test', '-d', remote], check=True)
archive = io.BytesIO()
with tarfile.open(fileobj=archive, mode='w:gz') as tar:
    for name in files:
        tar.add(source / name, arcname=name)
subprocess.run(ssh + [shlex.join(['tar', '-xzf', '-', '-C', remote])],
               input=archive.getvalue(), check=True)
for name in files:
    expected = hashlib.sha256((source / name).read_bytes()).hexdigest()
    actual = subprocess.check_output(ssh + ['sha256sum', remote + '/' + name],
                                     text=True).split()[0]
    assert actual == expected, name
inner = '''set -e
source /opt/ascend/cann-9.2.0/set_env.sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FFN_GU_READY_PARENT=/workspace/tests/glm52/whole_layer/ffn_original_shared_terminal/build-v1 python3 /candidate/test_prepare.py -v > /candidate/cpu-tests.log 2>&1
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 /candidate/build.py /workspace/tests/glm52/whole_layer/ffn_original_shared_terminal/build-v1 /candidate/build-v3
'''
command = ['docker', 'run', '--rm', '--pull=never', '--read-only', '--network', 'none',
    '--name', 'shanshan-tilert-ffn-gu-ready-build-v3',
    '--workdir', '/tmp', '--cpus', '1', '--memory', '12g', '--pids-limit', '128',
    '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g',
    '-v', task + ':/workspace:ro', '-v', remote + ':/candidate:rw',
    '-v', '/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0:/opt/ascend/cann-9.2.0:ro',
    '--entrypoint', '/bin/bash',
    'sha256:7448df539bcb40397063c4b7e83a6a1590ba20bcc16639c0aa83f1c60f1a0449', '-c', inner]
(root / 'ffn-gu-ready-build-v3.command.json').write_text(
    json.dumps(command, indent=2) + '\n')
with (root / 'ffn-gu-ready-build-v3.launcher.log').open('w') as log:
    result = subprocess.run(ssh + [shlex.join(command)], stdout=log, stderr=subprocess.STDOUT)
print('FFN original shared CPU build exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
