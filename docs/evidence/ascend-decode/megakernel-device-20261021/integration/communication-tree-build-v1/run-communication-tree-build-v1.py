import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
name = 'communication-tree-build-v1'
local = ROOT / name
remote = '/home/shanshan/tilert-megakernel-20261021/' + name
control = ROOT / 'communication-order-v2/communication_probe.so'
ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes', 'a3-78']
subprocess.run(ssh + ['mkdir', remote], check=True)
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
    for child in local.iterdir():
        tar.add(child, arcname=child.name)
    tar.add(control, arcname='accepted-control.so')
subprocess.run(ssh + [shlex.join(['tar', '-xzf', '-', '-C', remote])],
               input=buffer.getvalue(), check=True)
command = ['docker', 'run', '--rm', '--pull=never', '--name',
           'shanshan-tilert-' + name, '--read-only', '--network', 'none',
           '--workdir', '/tmp', '--cpus', '1', '--memory', '8g',
           '--pids-limit', '256', '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g',
           '-v', remote + ':/candidate:rw',
           '-v', '/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0:/opt/ascend/cann-9.2.0:ro',
           '-v', '/home/shanshan/tilert-megakernel-20261021/communication-deps/shmem:/shmem:ro',
           '-v', '/home/shanshan/tilert-megakernel-20261021/communication-deps/shmem-build/lib:/shmem-lib:ro',
           '-v', '/home/shanshan/tilert-megakernel-20261021/tests/glm52/whole_layer/communication_build-reuse/shmem_headers.sha256:/shmem-headers.sha256:ro',
           '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
           '-v', '/usr/local/dcmi:/usr/local/dcmi:ro',
           '--entrypoint', '/bin/bash',
           'sha256:7448df539bcb40397063c4b7e83a6a1590ba20bcc16639c0aa83f1c60f1a0449',
           '-c', 'set -e\nsource /opt/ascend/cann-9.2.0/set_env.sh\n'
           'LD_LIBRARY_PATH=/shmem-lib:${LD_LIBRARY_PATH} '
           'PYTHONDONTWRITEBYTECODE=1 ASCEND_PROCESS_LOG_PATH=/tmp/ascend-log '
           'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
           'python3 /candidate/source/build.py /candidate '
           '--cann /opt/ascend/cann-9.2.0 --shmem /shmem --libraries /shmem-lib '
           '--headers /shmem-headers.sha256 --control /candidate/accepted-control.so\n']
(ROOT / (name + '-command.json')).write_text(json.dumps(command, indent=2) + '\n')
with (ROOT / (name + '-launcher.log')).open('w') as log:
    result = subprocess.run(ssh + [shlex.join(command)], stdout=log, stderr=subprocess.STDOUT)
collect = "import io,sys,tarfile; from pathlib import Path; p=Path(" + repr(remote) + "); t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz'); [t.add(f,arcname=f.name) for f in p.iterdir() if f.is_file() and (f.suffix in ('.o','.so','.log','.jsonl','.d') or f.name=='build.json')]; t.close()"
raw = subprocess.check_output(ssh + ['python3', '-c', shlex.quote(collect)])
results = local / 'results'
results.mkdir()
with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as tar:
    tar.extractall(results, filter='data')
print('Tree CPU build exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
