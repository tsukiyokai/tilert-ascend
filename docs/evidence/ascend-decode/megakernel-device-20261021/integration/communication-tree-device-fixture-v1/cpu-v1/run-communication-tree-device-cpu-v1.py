import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT=Path('/tmp/tilert-megakernel-validation-20261021')
STAGE='communication-tree-device-v1'
NAME='communication-tree-device-cpu-v1'
local=ROOT/STAGE
task='/home/shanshan/tilert-megakernel-20261021'
remote=task+'/'+STAGE
cpu=task+'/'+NAME
ssh=['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes','a3-194']
subprocess.run(ssh+[shlex.join(['mkdir',remote,cpu])],check=True)
data=io.BytesIO()
with tarfile.open(fileobj=data,mode='w:gz') as tar:
 for p in local.iterdir():tar.add(p,arcname=p.name)
subprocess.run(ssh+[shlex.join(['tar','-xzf','-','-C',remote])],input=data.getvalue(),check=True)
command=['docker','run','--rm','--pull=never','--read-only','--network','none',
 '--name','shanshan-tilert-'+NAME,'--cpus','1','--memory','8g','--pids-limit','128',
 '--tmpfs','/tmp:rw,exec,nosuid,nodev,size=1g','--workdir','/tmp',
 '-v',remote+':/candidate:ro','-v',cpu+':/cpu:rw',
 '-v',task+'/communication-deps/shmem-build/lib:/shmem:ro',
 '-v','/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0:/opt/ascend/cann-9.2.0:ro',
 '--entrypoint','/bin/bash',
 'sha256:1910b1acedb37acc42beda9cb695c2cf84c57c81d8dd8b5191a98a97bc6d5fe6','-c',
 'set -euo pipefail\nsource /opt/ascend/cann-9.2.0/set_env.sh\n'
 '(cd /candidate && sha256sum --quiet -c frozen-inputs.sha256)\n'
 'LD_LIBRARY_PATH=/shmem:${LD_LIBRARY_PATH} '
 'PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
 'PYTHONPATH=/candidate ASCEND_PROCESS_LOG_PATH=/tmp/ascend-log '
 'python3 /candidate/device_cpu_check.py --fixture /candidate --libraries /shmem '
 '--output /cpu/report.json\n']
(ROOT/(NAME+'-command.json')).write_text(json.dumps(command,indent=2)+'\n')
with (ROOT/(NAME+'-launcher.log')).open('w') as log:
 result=subprocess.run(ssh+[shlex.join(command)],stdout=log,stderr=subprocess.STDOUT)
localcpu=ROOT/NAME;localcpu.mkdir()
collect="import sys,tarfile; from pathlib import Path; p=Path("+repr(cpu)+"); t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz'); [t.add(f,arcname=f.name) for f in p.iterdir() if f.is_file()]; t.close()"
raw=subprocess.check_output(ssh+['python3','-c',shlex.quote(collect)])
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:tar.extractall(localcpu,filter='data')
print('Device fixture CPU gate exit',result.returncode,flush=True)
raise SystemExit(result.returncode)
