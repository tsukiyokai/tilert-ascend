import io,json,shlex,subprocess,tarfile
from pathlib import Path
root=Path('/tmp/tilert-megakernel-validation-20261021')
source=Path('/home/shanshan/repo/tile-ai/TileRT-Ascend/tests/glm52/whole_layer/layer_tp_chain')
name='layer-tp-chain-host-cpu-v1'
local=root/name
local.mkdir()
for name_ in ('host.py','host_cpu_check.py'):
 (local/name_).write_bytes((source/name_).read_bytes())
task='/home/shanshan/tilert-megakernel-20261021'
remote=task+'/'+name
ssh=['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes','a3-78']
subprocess.run(ssh+['mkdir',remote],check=True)
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz') as t:
 for f in local.iterdir():t.add(f,arcname=f.name)
subprocess.run(ssh+[shlex.join(['tar','-xzf','-','-C',remote])],input=buffer.getvalue(),check=True)
inner=('set -e\nsource /opt/ascend/cann-9.2.0/set_env.sh\n'
 'LD_LIBRARY_PATH=/shmem-lib:${LD_LIBRARY_PATH} PYTHONDONTWRITEBYTECODE=1 '
 'ASCEND_PROCESS_LOG_PATH=/tmp/log OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
 'python3 /check/host_cpu_check.py --library /tp/layer_tp_chain.so '
 '--library-sha256 c2b72e469c3ee80a5fe7d9df4a3cceedc0e65cca95e6f426aea762c213392462 '
 '--local /local/layer_local_chain.so --output /check/report.json\n')
command=['docker','run','--rm','--pull=never','--name','shanshan-tilert-'+name,
 '--read-only','--network','none','--workdir','/tmp','--cpus','1','--memory','8g','--pids-limit','256',
 '--tmpfs','/tmp:rw,exec,nosuid,nodev,size=1g',
 '-v',remote+':/check:rw','-v',task+'/layer-tp-chain-build-v1:/tp:ro',
 '-v',task+'/layer-local-chain-build-v4:/local:ro',
 '-v',task+'/communication-deps/shmem-build/lib:/shmem-lib:ro',
 '-v','/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0:/opt/ascend/cann-9.2.0:ro',
 '-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
 '-v','/usr/local/dcmi:/usr/local/dcmi:ro','--entrypoint','/bin/bash',
 'sha256:7448df539bcb40397063c4b7e83a6a1590ba20bcc16639c0aa83f1c60f1a0449','-c',inner]
(local/'command.json').write_text(json.dumps(command,indent=2)+'\n')
with (local/'run.log').open('w') as f:
 result=subprocess.run(ssh+[shlex.join(command)],stdout=f,stderr=subprocess.STDOUT)
collect="from pathlib import Path;import sys;sys.stdout.buffer.write(Path("+repr(remote+'/report.json')+").read_bytes())"
if result.returncode==0:
 (local/'report.json').write_bytes(subprocess.check_output(ssh+['python3','-c',shlex.quote(collect)]))
print('TP host CPU gate exit',result.returncode,flush=True)
raise SystemExit(result.returncode)
