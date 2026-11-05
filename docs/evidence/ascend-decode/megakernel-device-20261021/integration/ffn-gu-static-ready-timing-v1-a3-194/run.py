import importlib.util
from pathlib import Path
import subprocess
import time
import shlex
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("kind", choices=("gu_static_ready",))
args = parser.parse_args()

root = Path('/tmp/tilert-megakernel-validation-20261021')
spec = importlib.util.spec_from_file_location('idle_gate', root / 'run-pending-single-die.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
gate.SSH[-1] = 'a3-194'
task_root = '/home/shanshan/tilert-megakernel-20261021'
identities = {'tests/glm52/whole_layer/ffn_composed/build-v2/harness-v1/check.py': 'd5c1f9f3214a1461d0aff29e166ec781d894c6bd991369bbac27f80baed3c9bb', 'logs/ffn-bridge-runtime212-v1/check.py': '6a36334a25bbde2f76c8749a0a385db48078deb3549c3a321aed1d43c4089afa', 'logs/ffn-bridge-runtime212-v1/fixtures.py': 'c0b010dadbdfe3e47918e5a84184ab295d4ac3a4da818066bee93e5e215b05f9', 'logs/ffn-bridge-runtime212-v1/gate_probe.py': '0bf8e940836905d37e396e04bea1440485e0472e0c68cdb6b4b738b8e7105276', 'tests/glm52/whole_layer/ffn_composed/build-v2/body/source/routed_int4/check.py': '4a28c5cdbbd4e7e4d71ea22e8b2e01e6337aacb0c7c2887442c3d25d1c2fd0fd', 'tests/glm52/whole_layer/ffn_composed/build-v2/body/source/routed_int4/check_pipeline.py': '3eb8c327b01da2671d77a66f38e4bc201c4c727f113bae38a4023172ee344f25', 'tests/glm52/whole_layer/ffn_composed/build-v2/ffn_composed.so': '3673c2a73b2f4b5a4a93b3c2543a4aca35b45773d6a53fc043f27e59a2482655', 'logs/ffn-bridge-runtime212-v1/native/ffn_bridge.so': '57392089102d11b0d3c9389228345406e7473c50b9877876332ba147dba55e6c', 'tests/glm52/whole_layer/routed_body/build-v2/routed_int4_torch.so': '6dcb8401cf722f9b3c3650d2432321ce6100d79aa98ff329d3a17a63a0f1d6c4', 'tests/glm52/whole_layer/ffn_gu_static_ready/build-v1/ffn_gu_static_ready.so': 'affc88ba3f4bdf30b569ec14cc58a6c3d67a32b23fbb1a58f959571a7890d488', 'tests/glm52/whole_layer/ffn_gu_static_ready/timing-v1/bench.py': 'cb80928abcf598c1876ec186a2cf915712205501677039f81cc4121ef9d16f0b', 'tests/glm52/whole_layer/ffn_gu_static_ready/timing-v1/bench_run.sh': '4b1ccf19bf7fc912ed4ea40d7696f1c0cb43954340c6052ce4be79509b33e9e0'}
identities['tests/glm52/whole_layer/ffn_original_shared_gu_static/build-v1/ffn_original_shared_gu_static.so'] = '381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2'
for name, expected in identities.items():
    actual = subprocess.check_output(gate.SSH + ['sha256sum', task_root + '/' + name], text=True).split()[0]
    assert actual == expected, name
assert gate.vacant()
time.sleep(5)
assert gate.vacant()
users = subprocess.run(gate.SSH + ['fuser', '/dev/davinci4'], capture_output=True, text=True)
assert users.returncode == 1 and not users.stdout.strip() and not users.stderr.strip(), users
name = 'ffn-gu-static-ready-timing-v1'
subprocess.run(gate.SSH + ['mkdir', task_root + '/logs/' + name], check=True)
inner = ('source /opt/ascend/cann-9.2.0/set_env.sh\n'
    'PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/tmp/cache '
    'ASCEND_PROCESS_LOG_PATH=/evidence/driver-log OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
    'env -u ASCEND_DOCKER_RUNTIME bash /workspace/tests/glm52/whole_layer/ffn_gu_static_ready/'
    'timing-v1/bench_run.sh device /evidence\n')
command = ['docker', 'run', '--rm', '--pull=never', '--name', 'shanshan-tilert-' + name, '--read-only', '--network', 'none', '--workdir', '/tmp', '--cpus', '2', '--memory', '12g', '--pids-limit', '128', '--shm-size', '256m', '--device', '/dev/davinci4', '--device', '/dev/davinci_manager', '--device', '/dev/devmm_svm', '--device', '/dev/hisi_hdc', '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g', '--tmpfs', '/root/ascend/log:rw,nosuid,nodev,size=128m', '--tmpfs', '/root/Ascend:rw,nosuid,nodev,size=128m', '--tmpfs', '/root/.tvm_test_data:rw,nosuid,nodev,size=128m', '-v', task_root + ':/workspace:ro', '-v', task_root + '/logs/' + name + ':/evidence:rw', '-v', '/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0:/opt/ascend/cann-9.2.0:ro', '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro', '-v', '/usr/local/dcmi:/usr/local/dcmi:ro', '--entrypoint', '/bin/bash', 'sha256:1910b1acedb37acc42beda9cb695c2cf84c57c81d8dd8b5191a98a97bc6d5fe6', '-c', inner]
print('Starting paired', args.kind, 'versus original-shared parent FFN on a3-194 physical die4', flush=True)
(root / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
with (root / (name + '.launcher.log')).open('w') as output:
    result = subprocess.run(gate.SSH + [shlex.join(command)], stdout=output, stderr=subprocess.STDOUT)
print('Paired FFN timing exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
