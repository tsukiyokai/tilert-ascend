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
identities = {
    'logs/ffn-bridge-runtime212-v1/check.py': '6a36334a25bbde2f76c8749a0a385db48078deb3549c3a321aed1d43c4089afa',
    'logs/ffn-bridge-runtime212-v1/native/ffn_bridge.so': '57392089102d11b0d3c9389228345406e7473c50b9877876332ba147dba55e6c',
    'logs/ffn-bridge-runtime212-v1/official_bridge_oracle.so': 'a7dc6b3e54d41ecf7068d5c7e18d96513c827f97c62039b5483ea905178d0708',
}
for name, expected in identities.items():
    actual = subprocess.check_output(gate.SSH + ['sha256sum', task_root + '/' + name], text=True).split()[0]
    assert actual == expected, name
manifest_path = Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-bridge-build-v2/oracle-package-v4/manifest.json')
manifest = json.loads(manifest_path.read_text())
verify = 'import hashlib,json; from pathlib import Path; base=Path(' + repr(task_root + '/tests/glm52/whole_layer/ffn_bridge/oracle-package-v4/install') + '); expected=json.loads(' + repr(json.dumps(manifest['installed_sha256'])) + '); bad=[name for name,digest in expected.items() if hashlib.sha256((base/name).read_bytes()).hexdigest()!=digest]; assert not bad,bad; print("Verified installed oracle files",len(expected))'
subprocess.run(gate.SSH + ['python3', '-c', shlex.quote(verify)], check=True)
assert gate.vacant()
time.sleep(5)
assert gate.vacant()
users = subprocess.run(gate.SSH + ['fuser', '/dev/davinci4'], capture_output=True, text=True)
assert users.returncode == 1 and not users.stdout.strip() and not users.stderr.strip(), users
name = 'ffn-bridge212-m1b16-v1'
subprocess.run(gate.SSH + ['mkdir', task_root + '/logs/' + name], check=True)
inner = """source /opt/ascend/cann-9.2.0/set_env.sh
PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/tmp/cache ASCEND_PROCESS_LOG_PATH=/evidence/driver-log OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 env -u ASCEND_DOCKER_RUNTIME python3 /workspace/logs/ffn-bridge-runtime212-v1/check.py --library /workspace/logs/ffn-bridge-runtime212-v1/native/ffn_bridge.so --oracle-library /workspace/logs/ffn-bridge-runtime212-v1/official_bridge_oracle.so --oracle-prefix /workspace/tests/glm52/whole_layer/ffn_bridge/oracle-package-v4/install/packages/vendors/tilert_bridge_oracle --gate-weight-layout nd --rows 1 --blocks 16 --replays 8 --output /evidence
"""
command = ['docker', 'run', '--rm', '--pull=never', '--name', 'shanshan-tilert-' + name, '--read-only', '--network', 'none', '--workdir', '/tmp', '--cpus', '2', '--memory', '12g', '--pids-limit', '128', '--shm-size', '256m', '--device', '/dev/davinci4', '--device', '/dev/davinci_manager', '--device', '/dev/devmm_svm', '--device', '/dev/hisi_hdc', '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g', '--tmpfs', '/root/ascend/log:rw,nosuid,nodev,size=128m', '--tmpfs', '/root/Ascend:rw,nosuid,nodev,size=128m', '--tmpfs', '/root/.tvm_test_data:rw,nosuid,nodev,size=128m', '-v', task_root + ':/workspace:ro', '-v', task_root + '/logs/' + name + ':/evidence:rw', '-v', '/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0:/opt/ascend/cann-9.2.0:ro', '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro', '-v', '/usr/local/dcmi:/usr/local/dcmi:ro', '--entrypoint', '/bin/bash', 'sha256:1910b1acedb37acc42beda9cb695c2cf84c57c81d8dd8b5191a98a97bc6d5fe6', '-c', inner]
print('Starting complete FFN bridge M1 official gate on a3-194 physical die4', flush=True)
(root / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
with (root / (name + '.launcher.log')).open('w') as output:
    result = subprocess.run(gate.SSH + [shlex.join(command)], stdout=output, stderr=subprocess.STDOUT)
print('Bridge launcher exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
