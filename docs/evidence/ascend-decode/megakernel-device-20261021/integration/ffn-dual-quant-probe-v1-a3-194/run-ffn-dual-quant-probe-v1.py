import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

parser = argparse.ArgumentParser()
parser.add_argument('--wide', action='store_true')
args = parser.parse_args()
root = Path('/tmp/tilert-megakernel-validation-20261021')
name = 'ffn-dual-quant-probe-' + ('wide' if args.wide else 'm12') + '-v1'
task = '/home/shanshan/tilert-megakernel-20261021'
relative = 'tests/glm52/whole_layer/ffn_dual_quant'
spec = importlib.util.spec_from_file_location('idle_gate', root / 'run-pending-single-die.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
gate.SSH[-1] = 'a3-194'
identities = {
 relative + '/build-v1/manifest.json': '2a5009cb99dbb366919cca1713645400adbef59e302e774a5160394b482bb222',
 relative + '/probe-build-v1/manifest.json': 'ca037e2dbd81642310f50fb2aac32ec973c765e84a9ce17306caec8aaa90bc24',
 relative + '/build-v1/ffn_dual_quant.so': '715060c0c50a275855eb757cd85128a6c5586deeaa7757dc2945b92bee64d69b',
 relative + '/probe-build-v1/quant_probe.so': '41bc2af880217a0520e6dfc8764de6068e1e570bc08ea14e2349333549126c65',
}
for local in ('ffn-dual-quant-build-v1', 'ffn-dual-quant-probe-build-v1'):
 manifest = json.loads((root / local / 'manifest.json').read_text())
 for kind in ('sources', 'artifacts'):
  for path, expected in manifest[kind].items():
   assert hashlib.sha256((root / local / path).read_bytes()).hexdigest() == expected, (local, path)
for path, expected in identities.items():
 actual = subprocess.check_output(gate.SSH + ['sha256sum', task + '/' + path], text=True).split()[0]
 assert actual == expected, path
assert gate.vacant(), 'die4 not vacant at first sample'
time.sleep(5)
assert gate.vacant(), 'die4 not vacant at second sample'
users = subprocess.run(gate.SSH + ['fuser', '/dev/davinci4'], capture_output=True, text=True)
assert users.returncode == 1 and not users.stdout.strip() and not users.stderr.strip(), users
remote = task + '/logs/' + name
subprocess.run(gate.SSH + ['mkdir', remote], check=True)
argv = ['python3', '/workspace/' + relative + '/probe-build-v1/probe/check.py',
 '--candidate-build', '/workspace/' + relative + '/build-v1',
 '--probe-build', '/workspace/' + relative + '/probe-build-v1',
 '--rows', '1,2,12,32' if args.wide else '12',
 '--blocks', '16,24' if args.wide else '24', '--output', '/evidence/report.json']
inner = ('set -e\nsource /opt/ascend/cann-9.2.0/set_env.sh\n'
 'PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/tmp/cache '
 'ASCEND_PROCESS_LOG_PATH=/evidence/driver-log OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
 + shlex.join(argv) + '\n')
command = ['docker', 'run', '--rm', '--pull=never', '--name', 'shanshan-tilert-' + name,
 '--read-only', '--network', 'none', '--workdir', '/tmp', '--cpus', '2', '--memory', '12g',
 '--pids-limit', '128', '--shm-size', '256m', '--device', '/dev/davinci4',
 '--device', '/dev/davinci_manager', '--device', '/dev/devmm_svm', '--device', '/dev/hisi_hdc',
 '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g',
 '--tmpfs', '/root/ascend/log:rw,nosuid,nodev,size=128m',
 '--tmpfs', '/root/Ascend:rw,nosuid,nodev,size=128m',
 '-v', task + ':/workspace:ro', '-v', remote + ':/evidence:rw',
 '-v', '/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0:/opt/ascend/cann-9.2.0:ro',
 '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
 '-v', '/usr/local/dcmi:/usr/local/dcmi:ro', '--entrypoint', '/bin/bash',
 'sha256:1910b1acedb37acc42beda9cb695c2cf84c57c81d8dd8b5191a98a97bc6d5fe6', '-c', inner]
(root / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
print('Starting direct quantization probe on a3-194 physical die4', flush=True)
with (root / (name + '.launcher.log')).open('w') as stream:
 result = subprocess.run(gate.SSH + [shlex.join(command)], stdout=stream, stderr=subprocess.STDOUT)
raw = subprocess.check_output(gate.SSH + [shlex.join(['tar', '-czf', '-', '-C', remote, '.'])])
output = root / name
output.mkdir()
with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
 archive.extractall(output, filter='data')
print('Direct quantization probe exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
