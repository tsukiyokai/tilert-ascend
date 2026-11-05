import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

root = Path('/tmp/tilert-megakernel-validation-20261021')
source = Path('tests/glm52/whole_layer/attention_shared_descriptor').resolve()
task = '/home/shanshan/tilert-megakernel-20261021'
remote = task + '/tests/glm52/whole_layer/attention_shared_descriptor'
ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes', 'a3-78']
parent = task + '/logs/runtime212-build-v1/attention/shared/whole_layer_full_attention.so'
assert subprocess.check_output(ssh + ['sha256sum', parent], text=True).split()[0] == '64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c'
files = ['descriptor.h', 'prepare.py', 'build.py', 'test_prepare.py', 'descriptor_probe.cpp']
subprocess.run(ssh + ['test', '-d', remote], check=True)
archive = io.BytesIO()
with tarfile.open(fileobj=archive, mode='w:gz') as tar:
    for name in files:
        tar.add(source / name, arcname=name)
subprocess.run(ssh + [shlex.join(['tar', '-xzf', '-', '-C', remote])], input=archive.getvalue(), check=True)
for name in files:
    expected = hashlib.sha256((source / name).read_bytes()).hexdigest()
    actual = subprocess.check_output(ssh + ['sha256sum', remote + '/' + name], text=True).split()[0]
    assert actual == expected, name
inner = """set -e
source /opt/ascend/cann-9.2.0/set_env.sh
export PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/tmp/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OPS_TRANSFORMER_ROOT=/workspace/attention_vendor
export SHARED_DESCRIPTOR_PARENT=/workspace/logs/runtime212-build-v1/attention
python3 /candidate/test_prepare.py -v > /candidate/cpu-tests-v2.log 2>&1
python3 /candidate/build.py /workspace/logs/runtime212-build-v1/attention /candidate/build-v1 /workspace/logs/runtime212-build-v1/attention/shared/whole_layer_full_attention.so
"""
command = ['docker', 'run', '--rm', '--pull=never', '--read-only', '--network', 'none',
    '--workdir', '/tmp', '--cpus', '2', '--memory', '12g', '--pids-limit', '128',
    '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g',
    '-v', task + ':/workspace:ro', '-v', remote + ':/candidate:rw',
    '-v', '/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0:/opt/ascend/cann-9.2.0:ro',
    '--entrypoint', '/bin/bash',
    'sha256:7448df539bcb40397063c4b7e83a6a1590ba20bcc16639c0aa83f1c60f1a0449', '-c', inner]
(root / 'shared-attention-descriptor-build-v2.command.json').write_text(json.dumps(command, indent=2) + '\n')
with (root / 'shared-attention-descriptor-build-v2.launcher.log').open('w') as log:
    result = subprocess.run(ssh + [shlex.join(command)], stdout=log, stderr=subprocess.STDOUT)
print('Shared attention descriptor CPU build exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
