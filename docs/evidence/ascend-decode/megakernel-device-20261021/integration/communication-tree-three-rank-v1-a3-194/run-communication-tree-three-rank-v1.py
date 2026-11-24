"""Run the three-rank serial/tree/HCCL diagnostic on admitted idle dies."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
TASK = '/home/shanshan/tilert-megakernel-20261021'
NAME = 'communication-tree-three-rank-v1'
LOCAL = ROOT / 'communication-tree-device-v1'
REMOTE = TASK + '/communication-tree-device-v1'
MANIFEST = '18d57bafa45fb629965916d0d380e014686dd322c575c469ee6327caa4cc98e8'
SHMEM = {
    'libshmem.so': 'cf9b853bb7612ce1e44853b1d16851016c4a2c279bdc9bb32da0640bc411f532',
    'libshmem_utils.so': '743ea9661dbc6707fba5030967a22c59ebfdd0daed4884fdff39da12ae008ac0',
    'aclshmem_bootstrap_config_store.so': 'ef9160f3706a7eac573ff0a83160ae2d38579550dd923f45b330f309a33d6541',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert digest(LOCAL / 'manifest.json') == MANIFEST
pins = dict(line.split('  ', 1)[::-1]
            for line in (LOCAL / 'frozen-inputs.sha256').read_text().splitlines())
assert len(pins) == 43
for name, expected in pins.items():
    assert digest(LOCAL / name) == expected, name
cpu_path = ROOT / 'communication-tree-device-cpu-v2/report.json'
assert digest(cpu_path) == '30898f1ca654512e7ae62247388a143929c272ee049ff20f2fe63d30ccc52348'
cpu = json.loads(cpu_path.read_text())
assert cpu['pass'] and cpu['same_shmem_instance']
assert not cpu['npu_initialized_before'] and not cpu['npu_initialized_after']
assert not cpu['mapped_davinci_devices'] and not cpu['device_execution']
assert cpu['shmem_initialize_calls'] == 0
assert (cpu['torch'], cpu['torch_npu']) == ('2.12.0+cu130', '2.12.0')
assert cpu['fixture_manifest_sha256'] == MANIFEST
assert cpu['driver_sha256'] == pins['communication_tree_devicecheck.py']
assert cpu['parent_sha256'] == pins['parent_probe.so']
assert cpu['candidate_sha256'] == pins['candidate_probe.so']
assert cpu['driver_cpu_contract']['pass']
assert cpu['shmem_libraries'] == SHMEM
spec = importlib.util.spec_from_file_location('idle_gate', ROOT / 'two-die-idle-gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
gate.DEVICES = (4, 6, 7)
gate.LOG = ROOT / (NAME + '.admission.jsonl')
expected = {REMOTE + '/' + n: h for n, h in pins.items()}
expected.update({TASK + '/communication-deps/shmem-build/lib/' + n: h
                 for n, h in SHMEM.items()})
code = ('import hashlib,json; from pathlib import Path; expected=json.loads('
        + repr(json.dumps(expected)) + '); '
        'bad=[n for n,h in expected.items() '
        'if hashlib.sha256(Path(n).read_bytes()).hexdigest()!=h]; '
        'assert not bad,bad; print("Verified probe and SHMEM files",len(expected))')
subprocess.run(gate.SSH + [shlex.join(['python3', '-c', code])], check=True)

script = '''set -euo pipefail
(cd /candidate && sha256sum --quiet -c frozen-inputs.sha256)
source /opt/ascend/cann-9.2.0/set_env.sh
test ! -e /evidence/communication-tree-three-rank-v1
test ! -e /evidence/communication-tree-three-rank-v1-driver
mkdir /evidence/communication-tree-three-rank-v1 /evidence/communication-tree-three-rank-v1-driver
LD_LIBRARY_PATH=/shmem:${LD_LIBRARY_PATH:-} PYTHONPATH=/candidate \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
GLOO_SOCKET_IFNAME=lo HCCL_OP_EXPANSION_MODE=AIV HCCL_DETERMINISTIC=true \
ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=0 \
ASCEND_PROCESS_LOG_PATH=/evidence/communication-tree-three-rank-v1-driver \
env -u ASCEND_DOCKER_RUNTIME -u ASCEND_RT_VISIBLE_DEVICES \
-u HCCL_ALGO -u HCCL_ALGO_CONFIG -u HCCL_BUFFSIZE -u ASCEND_LAUNCH_BLOCKING \
timeout --signal=TERM --kill-after=20s 300s \
python3 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=3 \
/candidate/communication_tree_devicecheck.py \
--parent /candidate/parent_probe.so --candidate /candidate/candidate_probe.so \
--candidate-sha256 a0bd2462e2f999280720fa5060466159a2eab13ce5541639ddc7afef35a45ebf \
--endpoint tcp://127.0.0.1:28766 \
--physical-devices 4 6 7 --output /evidence/communication-tree-three-rank-v1
'''
command = ['docker', 'run', '--rm', '--pull=never', '--read-only',
           '--network', 'none', '--name', 'shanshan-tilert-' + NAME,
           '--workdir', '/tmp', '--cpus', '4', '--memory', '12g',
           '--pids-limit', '512', '--shm-size', '256m']
for name in ('4', '6', '7', '_manager'):
    command += ['--device', '/dev/davinci' + name]
command += ['--device', '/dev/devmm_svm', '--device', '/dev/hisi_hdc',
            '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g',
            '--tmpfs', '/root/ascend/log:rw,nosuid,nodev,size=128m',
            '--tmpfs', '/root/Ascend:rw,nosuid,nodev,size=128m',
            '-v', REMOTE + ':/candidate:ro',
            '-v', TASK + '/communication-deps/shmem-build/lib:/shmem:ro',
            '-v', TASK + '/logs:/evidence:rw',
            '-v', '/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0:/opt/ascend/cann-9.2.0:ro',
            '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
            '-v', '/usr/local/dcmi:/usr/local/dcmi:ro',
            '--entrypoint', '/bin/bash',
            'sha256:1910b1acedb37acc42beda9cb695c2cf84c57c81d8dd8b5191a98a97bc6d5fe6',
            '-lc', script]
assert gate.vacant(), 'First three-die idle sample refused'
time.sleep(5)
assert gate.vacant(), 'Second three-die idle sample refused'
(ROOT / (NAME + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
print('Starting three-rank tree probe on physical dies 4/6/7', flush=True)
with (ROOT / (NAME + '.launcher.log')).open('w') as log:
    result = subprocess.run(gate.SSH + [shlex.join(command)],
                            stdout=log, stderr=subprocess.STDOUT)
print('Three-rank tree probe exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
