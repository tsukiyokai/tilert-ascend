import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
ROOT=Path('/tmp/tilert-megakernel-validation-20261021')
NAME='communication-tree-device-cpu-v2'
task='/home/shanshan/tilert-megakernel-20261021'
ssh=['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes','a3-194']
subprocess.run(ssh+[shlex.join(['mkdir',task+'/'+NAME])],check=True)
command=json.loads((ROOT/'communication-tree-device-cpu-v1-command.json').read_text())
command=[value.replace('communication-tree-device-cpu-v1',NAME) for value in command]
i=command.index('--entrypoint')
command[i:i]=['-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
              '-v','/usr/local/dcmi:/usr/local/dcmi:ro']
(ROOT/(NAME+'-command.json')).write_text(json.dumps(command,indent=2)+'\n')
with (ROOT/(NAME+'-launcher.log')).open('w') as log:
 result=subprocess.run(ssh+[shlex.join(command)],stdout=log,stderr=subprocess.STDOUT)
local=ROOT/NAME;local.mkdir()
collect="import sys,tarfile; from pathlib import Path; p=Path("+repr(task+'/'+NAME)+"); t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz'); [t.add(f,arcname=f.name) for f in p.iterdir() if f.is_file()]; t.close()"
raw=subprocess.check_output(ssh+['python3','-c',shlex.quote(collect)])
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:tar.extractall(local,filter='data')
print('Device fixture CPU gate exit',result.returncode,flush=True)
raise SystemExit(result.returncode)
