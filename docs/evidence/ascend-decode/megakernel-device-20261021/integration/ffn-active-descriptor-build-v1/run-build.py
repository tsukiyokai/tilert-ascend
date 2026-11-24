from pathlib import Path
import json
import shlex
import subprocess

root=Path('/tmp/tilert-megakernel-validation-20261021')
name='ffn-active-descriptor-build-v1'
command=json.loads((root/'ffn-active-build-v1.command.json').read_text())
command[command.index('--name')+1]='shanshan-tilert-'+name
command[-1]='''set -e
source /opt/ascend/cann-9.2.0/set_env.sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FFN_ACTIVE_DESCRIPTOR_PARENT=/workspace/tests/glm52/whole_layer/ffn_active/build-v1 python3 /workspace/tests/glm52/whole_layer/ffn_active_descriptor/test_prepare.py -v
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 /workspace/tests/glm52/whole_layer/ffn_active_descriptor/build.py /workspace/tests/glm52/whole_layer/ffn_active/build-v1 /workspace/tests/glm52/whole_layer/ffn_active_descriptor/build-v1
'''
(root/(name+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
with (root/(name+'.launcher.log')).open('w') as log:
 result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes','7.150.10.78',shlex.join(command)],stdout=log,stderr=subprocess.STDOUT)
print('CPU combined build exit',result.returncode,flush=True)
raise SystemExit(result.returncode)
