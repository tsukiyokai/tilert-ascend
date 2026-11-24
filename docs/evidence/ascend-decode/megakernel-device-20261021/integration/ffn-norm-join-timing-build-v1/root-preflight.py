import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
LOCAL = ROOT / 'ffn-norm-join-timing-v1'
TASK = '/home/shanshan/tilert-megakernel-20261021'
REMOTE = TASK + '/tests/glm52/whole_layer/ffn_norm_join/timing-v1'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes']
stream = io.BytesIO()
with tarfile.open(fileobj=stream, mode='w:gz') as tar:
    for path in LOCAL.rglob('*'):
        if path.is_file():
            tar.add(path, arcname=str(path.relative_to(LOCAL)))
raw = stream.getvalue()
subprocess.run(SSH + ['a3-78', 'mkdir', REMOTE], check=True)
subprocess.run(SSH + ['a3-78', shlex.join(['tar', '-xzf', '-', '-C', REMOTE])], input=raw, check=True)
inner = '''set -e
source /opt/ascend/cann-9.2.0/set_env.sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FFN_NORM_JOIN_TIMING_PARENT=/candidate/parent/bench.py FFN_FROZEN_BUILD=/workspace/tests/glm52/whole_layer/ffn_composed/build-v2 FFN_NORM_JOIN_BASE_BUILD=/workspace/tests/glm52/whole_layer/ffn_original_shared_gu_static/build-v1 FFN_NORM_JOIN_BUILD=/workspace/tests/glm52/whole_layer/ffn_norm_join/build-v1 FFN_BASE_ABI_PROBE=/candidate/parent/bench_abi_probe.cpp python3 /candidate/test_bench.py -v > /candidate/cpu-arm.log 2>&1
mkdir /candidate/preflight
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bash /candidate/bench_run.sh preflight /candidate/preflight > /candidate/preflight.log 2>&1
'''
command = ['docker', 'run', '--rm', '--pull=never', '--read-only', '--network', 'none',
           '--name', 'shanshan-tilert-ffn-norm-join-timing-cpu-v1',
           '--workdir', '/tmp', '--cpus', '1', '--memory', '8g', '--pids-limit', '128',
           '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g', '-v', TASK+':/workspace:ro',
           '-v', REMOTE+':/candidate:rw',
           '-v', '/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0:/opt/ascend/cann-9.2.0:ro',
           '--entrypoint', '/bin/bash',
           'sha256:7448df539bcb40397063c4b7e83a6a1590ba20bcc16639c0aa83f1c60f1a0449', '-c', inner]
(ROOT / 'ffn-norm-join-timing-cpu-v1.command.json').write_text(json.dumps(command, indent=2)+'\n')
with (ROOT / 'ffn-norm-join-timing-cpu-v1.log').open('w') as log:
    result = subprocess.run(SSH + ['a3-78', shlex.join(command)], stdout=log, stderr=subprocess.STDOUT)
if result.returncode:
    raise SystemExit(result.returncode)
for name in ('cpu-arm.log', 'preflight.log', 'preflight/result.json'):
    data = subprocess.check_output(SSH + ['a3-78', 'cat', REMOTE+'/'+name])
    target = LOCAL/name
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(data)
report = json.loads((LOCAL/'preflight/result.json').read_text())
assert report['cpu_preflight'] and not report['npu_initialized']
assert report['arm_names'] == {'fused': 'norm_join', 'separate': 'original_shared_gu_static_parent'}
manifest = json.loads((LOCAL/'manifest.json').read_text())
for name, expected in manifest['sources'].items():
    assert hashlib.sha256((LOCAL/name).read_bytes()).hexdigest()==expected,name
subprocess.run(SSH + ['a3-194', 'mkdir', REMOTE], check=True)
subprocess.run(SSH + ['a3-194', shlex.join(['tar', '-xzf', '-', '-C', REMOTE])], input=raw, check=True)
for name, expected in manifest['sources'].items():
    actual = subprocess.check_output(SSH + ['a3-194', 'sha256sum', REMOTE+'/'+name], text=True).split()[0]
    assert actual == expected, name
out = Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-norm-join-timing-build-v1')
shutil.copytree(LOCAL, out)
shutil.copy2(__file__, out/'root-preflight.py')
print(json.dumps({'pass':True,'cpu_contract_tests':6,'npu_initialized':False,
                  'bench_sha256':manifest['sources']['bench.py'],
                  'runner_sha256':manifest['sources']['bench_run.sh'],
                  'manifest_sha256':hashlib.sha256((LOCAL/'manifest.json').read_bytes()).hexdigest()},indent=2))
