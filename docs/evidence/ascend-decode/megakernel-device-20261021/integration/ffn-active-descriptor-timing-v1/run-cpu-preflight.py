from pathlib import Path
import json,shlex,subprocess
root=Path('/tmp/tilert-megakernel-validation-20261021')
remote='/home/shanshan/tilert-megakernel-20261021'
ssh=['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes','7.150.10.78']
name='ffn-active-descriptor-preflight-v1'
subprocess.run(ssh+['mkdir',remote+'/logs/'+name],check=True)
command=json.loads((root/'ffn-active-build-v1.command.json').read_text())
command[command.index('--name')+1]='shanshan-tilert-'+name
command[command.index(remote+':/workspace:rw')]=remote+':/workspace:ro'
command[command.index('--entrypoint'):command.index('--entrypoint')]=['-v',remote+'/logs/'+name+':/evidence:rw']
command[-1]='''set -e
source /opt/ascend/cann-9.2.0/set_env.sh
PYTHONDONTWRITEBYTECODE=1 FFN_VARIANT_TIMING_PARENT=/workspace/tests/glm52/whole_layer/ffn_active_descriptor/timing-v1/parent/bench.py FFN_FROZEN_BUILD=/workspace/tests/glm52/whole_layer/ffn_composed/build-v2 FFN_ACTIVE_BUILD=/workspace/tests/glm52/whole_layer/ffn_active/build-v1 FFN_ACTIVE_DESCRIPTOR_BUILD=/workspace/tests/glm52/whole_layer/ffn_active_descriptor/build-v1 FFN_BASE_ABI_PROBE=/workspace/tests/glm52/whole_layer/ffn_active_descriptor/timing-v1/parent/bench_abi_probe.cpp python3 /workspace/tests/glm52/whole_layer/ffn_active_descriptor/timing-v1/test_bench.py -v
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bash /workspace/tests/glm52/whole_layer/ffn_active_descriptor/timing-v1/bench_run.sh preflight /evidence
'''
(root/(name+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
with (root/(name+'.launcher.log')).open('w') as log:
 r=subprocess.run(ssh+[shlex.join(command)],stdout=log,stderr=subprocess.STDOUT)
print('Combined benchmark CPU preflight exit',r.returncode)
raise SystemExit(r.returncode)
