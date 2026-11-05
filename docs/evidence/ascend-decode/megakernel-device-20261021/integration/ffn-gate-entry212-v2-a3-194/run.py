import importlib.util
from pathlib import Path
import subprocess
import time
import shlex
import json

root = Path('/tmp/tilert-megakernel-validation-20261021')
spec = importlib.util.spec_from_file_location('idle_gate', root / 'run-pending-single-die.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
gate.SSH[-1] = 'a3-194'
task_root = '/home/shanshan/tilert-megakernel-20261021'
identities = {'logs/ffn-gate-entry-v1-harness/gate_probe.py': '0bf8e940836905d37e396e04bea1440485e0472e0c68cdb6b4b738b8e7105276', 'logs/ffn-gate-entry-v1-harness/fixtures.py': 'c0b010dadbdfe3e47918e5a84184ab295d4ac3a4da818066bee93e5e215b05f9'}
for name, expected in identities.items():
    actual = subprocess.check_output(gate.SSH + ["sha256sum", task_root + "/" + name], text=True).split()[0]
    assert actual == expected, name
assert gate.vacant()
time.sleep(5)
assert gate.vacant()
users = subprocess.run(gate.SSH + ['fuser', '/dev/davinci4'], capture_output=True, text=True)
assert users.returncode == 1 and not users.stdout.strip() and not users.stderr.strip(), users
name = 'ffn-gate-entry212-v2'
subprocess.run(gate.SSH + ['mkdir', task_root + '/logs/' + name], check=True)
inner = """source /opt/ascend/cann-9.2.0/set_env.sh
PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/tmp/cache ASCEND_PROCESS_LOG_PATH=/evidence/driver-log OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 env -u ASCEND_DOCKER_RUNTIME python3 /workspace/logs/ffn-gate-entry-v1-harness/gate_probe.py --rows 1 12 32 --layouts nd nz --output /evidence
"""
command = ['docker', 'run', '--rm', '--pull=never', '--name', 'shanshan-tilert-' + name, '--read-only', '--network', 'none', '--workdir', '/tmp', '--cpus', '2', '--memory', '12g', '--pids-limit', '128', '--shm-size', '256m', '--device', '/dev/davinci4', '--device', '/dev/davinci_manager', '--device', '/dev/devmm_svm', '--device', '/dev/hisi_hdc', '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g', '--tmpfs', '/root/ascend/log:rw,nosuid,nodev,size=128m', '--tmpfs', '/root/Ascend:rw,nosuid,nodev,size=128m', '--tmpfs', '/root/.tvm_test_data:rw,nosuid,nodev,size=128m', '-v', task_root + ':/workspace:ro', '-v', task_root + '/logs/' + name + ':/evidence:rw', '-v', '/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0:/opt/ascend/cann-9.2.0:ro', '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro', '-v', '/usr/local/dcmi:/usr/local/dcmi:ro', '--entrypoint', '/bin/bash', 'sha256:1910b1acedb37acc42beda9cb695c2cf84c57c81d8dd8b5191a98a97bc6d5fe6', '-c', inner]
print('Starting actual FP32 gate API probe on a3-194 physical die4', flush=True)
(root / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
with (root / (name + '.launcher.log')).open('w') as output:
    result = subprocess.run(gate.SSH + [shlex.join(command)], stdout=output, stderr=subprocess.STDOUT)
print('Gate-entry launcher exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
