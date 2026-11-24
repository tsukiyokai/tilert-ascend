import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

root = Path('/tmp/tilert-megakernel-validation-20261021')
spec = importlib.util.spec_from_file_location('idle_gate', root / 'run-pending-single-die.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
gate.SSH[-1] = 'a3-194'
task = '/home/shanshan/tilert-megakernel-20261021'
candidate = task + '/tests/glm52/whole_layer/attention_terminal'
identities = json.loads(Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/shared-attention-terminal-build-v1/deployed-sha256.json').read_text())
verify = 'import hashlib,json; from pathlib import Path; p=Path(' + repr(candidate) + '); expected=json.loads(' + repr(json.dumps(identities)) + '); bad=[n for n,h in expected.items() if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h]; assert not bad,bad; print("Verified shared terminal files",len(expected))'
subprocess.run(gate.SSH + ['python3', '-c', shlex.quote(verify)], check=True)
control = task + '/vendor-hook-runtime212/native/shared/whole_layer_full_attention.so'
assert subprocess.check_output(gate.SSH + ['sha256sum', control], text=True).split()[0] == '64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c'
timing_files = json.loads(Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/shared-attention-terminal-timing-build-v1/deployed-sha256.json').read_text())
verify_timing = 'import hashlib,json; from pathlib import Path; p=Path(' + repr(candidate + '/timing-v1') + '); expected=json.loads(' + repr(json.dumps(timing_files)) + '); bad=[n for n,h in expected.items() if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h]; assert not bad,bad; print("Verified timing files",len(expected))'
subprocess.run(gate.SSH + ['python3', '-c', shlex.quote(verify_timing)], check=True)
assert gate.vacant()
time.sleep(5)
assert gate.vacant()
users = subprocess.run(gate.SSH + ['fuser', '/dev/davinci4'], capture_output=True, text=True)
assert users.returncode == 1 and not users.stdout.strip() and not users.stderr.strip(), users
name = 'shared-attention-terminal212-timing-v1'
subprocess.run(gate.SSH + ['mkdir', task + '/logs/' + name], check=True)
inner = """set -e
source /opt/ascend/cann-9.2.0/set_env.sh
PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/tmp/cache ASCEND_PROCESS_LOG_PATH=/evidence/driver-log OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 env -u ASCEND_DOCKER_RUNTIME python3 /workspace/tests/glm52/whole_layer/attention_terminal/timing-v1/bench.py --library /workspace/tests/glm52/whole_layer/attention_terminal/build-v1/shared_attention_terminal.so --control-library /workspace/vendor-hook-runtime212/native/shared/whole_layer_full_attention.so --helpers /workspace/tests/glm52/whole_layer/attention_terminal/build-v1/source/tests/glm52/whole_layer --rows 12 --blocks 24 --replays 8 --sequence 8192 --banks 16 --samples 30 --iterations 30 --output /evidence/result.json
"""
command = ['docker', 'run', '--rm', '--pull=never', '--name', 'shanshan-tilert-' + name, '--read-only', '--network', 'none', '--workdir', '/tmp', '--cpus', '2', '--memory', '12g', '--pids-limit', '128', '--shm-size', '256m', '--device', '/dev/davinci4', '--device', '/dev/davinci_manager', '--device', '/dev/devmm_svm', '--device', '/dev/hisi_hdc', '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g', '--tmpfs', '/root/ascend/log:rw,nosuid,nodev,size=128m', '--tmpfs', '/root/Ascend:rw,nosuid,nodev,size=128m', '--tmpfs', '/root/.tvm_test_data:rw,nosuid,nodev,size=128m', '-v', task + ':/workspace:ro', '-v', task + '/logs/' + name + ':/evidence:rw', '-v', '/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0:/opt/ascend/cann-9.2.0:ro', '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro', '-v', '/usr/local/dcmi:/usr/local/dcmi:ro', '--entrypoint', '/bin/bash', 'sha256:1910b1acedb37acc42beda9cb695c2cf84c57c81d8dd8b5191a98a97bc6d5fe6', '-c', inner]
(root / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
print('Starting shared attention terminal same-address paired timing on physical die4', flush=True)
with (root / (name + '.launcher.log')).open('w') as log:
    result = subprocess.run(gate.SSH + [shlex.join(command)], stdout=log, stderr=subprocess.STDOUT)
print('Shared attention terminal device exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
