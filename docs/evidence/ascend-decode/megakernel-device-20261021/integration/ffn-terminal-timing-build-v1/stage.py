from pathlib import Path
import hashlib
import io
import json
import shlex
import subprocess
import tarfile

root = Path('/tmp/tilert-megakernel-validation-20261021')
source = root / 'ffn-terminal-timing-v1'
task = '/home/shanshan/tilert-megakernel-20261021'
remote = task + '/tests/glm52/whole_layer/ffn_terminal/timing-v1'
manifest = json.loads((source/'manifest.json').read_text())
expected = dict(manifest['sources'])
expected['manifest.json'] = hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()
archive = io.BytesIO()
with tarfile.open(fileobj=archive, mode='w:gz') as tar:
    for name in expected:
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == expected[name], name
        tar.add(source/name, arcname=name)
for host in ['a3-78', 'a3-194']:
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes', host]
    subprocess.run(ssh + ['mkdir', remote], check=True)
    subprocess.run(ssh + [shlex.join(['tar', '-xzf', '-', '-C', remote])], input=archive.getvalue(), check=True)
    verify = 'import hashlib,json; from pathlib import Path; base=Path('+repr(remote)+'); expected=json.loads('+repr(json.dumps(expected))+'); bad=[p for p,h in expected.items() if hashlib.sha256((base/p).read_bytes()).hexdigest()!=h]; assert not bad,bad; print("Verified timing files",len(expected))'
    subprocess.run(ssh + ['python3', '-c', shlex.quote(verify)], check=True)
inner = '''set -e
source /opt/ascend/cann-9.2.0/set_env.sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FFN_ACTIVE_TOKEN_JOIN_TIMING_PARENT=/candidate/parent/bench.py FFN_FROZEN_BUILD=/workspace/tests/glm52/whole_layer/ffn_composed/build-v2 FFN_ACTIVE_TOKEN_JOIN_BASE_BUILD=/workspace/tests/glm52/whole_layer/ffn_active_token_join/build-v1 FFN_TERMINAL_BUILD=/workspace/tests/glm52/whole_layer/ffn_terminal/build-v1 FFN_BASE_ABI_PROBE=/candidate/parent/bench_abi_probe.cpp python3 /candidate/test_bench.py -v > /candidate/cpu-tests.log 2>&1
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bash /candidate/bench_run.sh preflight /candidate/preflight
'''
command = ['docker','run','--rm','--pull=never','--read-only','--network','none','--name','shanshan-tilert-ffn-terminal-timing-cpu-v1','--workdir','/tmp','--cpus','1','--memory','8g','--pids-limit','128','--tmpfs','/tmp:rw,exec,nosuid,nodev,size=1g','-v',task+':/workspace:ro','-v',remote+':/candidate:rw','-v','/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0:/opt/ascend/cann-9.2.0:ro','--entrypoint','/bin/bash','sha256:7448df539bcb40397063c4b7e83a6a1590ba20bcc16639c0aa83f1c60f1a0449','-c',inner]
(root/'ffn-terminal-timing-cpu-v1.command.json').write_text(json.dumps(command,indent=2)+'\n')
with (root/'ffn-terminal-timing-cpu-v1.launcher.log').open('w') as log:
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes','a3-78',shlex.join(command)],stdout=log,stderr=subprocess.STDOUT)
print('ARM ABI and Torch preflight exit',result.returncode,flush=True)
raise SystemExit(result.returncode)
