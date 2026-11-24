import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

root = Path('/tmp/tilert-megakernel-validation-20261021')
name = 'layer-local-chain-host-cpu-v1'
local = root / name
local.mkdir(exist_ok=False)
source = Path('/home/shanshan/repo/tile-ai/TileRT-Ascend/tests/glm52/whole_layer/layer_local_chain')
for filename in ('host.py', 'host_cpu_check.py'):
    shutil.copy2(source / filename, local / filename)
task = '/home/shanshan/tilert-megakernel-20261021'
remote = task + '/' + name
ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes', 'a3-78']
subprocess.run(ssh + ['mkdir', remote], check=True)
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
    for path in local.iterdir():
        archive.add(path, arcname=path.name)
subprocess.run(ssh + [shlex.join(['tar', '-xzf', '-', '-C', remote])],
               input=buffer.getvalue(), check=True)
command = ['docker', 'run', '--rm', '--pull=never', '--name',
           'shanshan-tilert-' + name, '--read-only', '--network', 'none',
           '--workdir', '/tmp', '--cpus', '1', '--memory', '8g',
           '--pids-limit', '128', '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g',
           '-v', remote + ':/candidate:rw',
           '-v', task + '/layer-local-chain-build-v4:/library:ro',
           '-v', '/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0:/opt/ascend/cann-9.2.0:ro',
           '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
           '-v', '/usr/local/dcmi:/usr/local/dcmi:ro',
           '--entrypoint', '/bin/bash',
           'sha256:7448df539bcb40397063c4b7e83a6a1590ba20bcc16639c0aa83f1c60f1a0449',
           '-c', 'set -e\nsource /opt/ascend/cann-9.2.0/set_env.sh\n'
           'PYTHONDONTWRITEBYTECODE=1 ASCEND_PROCESS_LOG_PATH=/tmp/ascend-log '
           'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
           'python3 /candidate/host_cpu_check.py '
           '--library /library/layer_local_chain.so '
           '--sha256 609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f '
           '--output /candidate/report.json\n']
(local / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
with (local / 'run.log').open('w') as output:
    result = subprocess.run(ssh + [shlex.join(command)],
                            stdout=output, stderr=subprocess.STDOUT)
collect = "import sys,tarfile; from pathlib import Path; p=Path(" + repr(remote) + "); t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz'); [t.add(f,arcname=f.name) for f in p.iterdir() if f.is_file()]; t.close()"
raw = subprocess.check_output(ssh + ['python3', '-c', shlex.quote(collect)])
with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
    archive.extractall(local, filter='data')
print('Host C ABI CPU gate exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
