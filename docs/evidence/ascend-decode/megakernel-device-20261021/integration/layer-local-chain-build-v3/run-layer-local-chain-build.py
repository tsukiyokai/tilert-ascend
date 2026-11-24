import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
version = sys.argv[1]
assert version.isdecimal()
name = 'layer-local-chain-build-v' + version
local = ROOT / name
remote = '/home/shanshan/tilert-megakernel-20261021/' + name
source = Path('/home/shanshan/repo/tile-ai/TileRT-Ascend/tests/glm52/whole_layer/layer_local_chain')
subprocess.run([sys.executable, str(source / 'prepare.py'),
                str(ROOT / 'attention-wk-m1-build-v1'),
                str(ROOT / 'ffn-original-shared-gu-static-build-v1'), str(local)], check=True)
ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes', 'a3-78']
subprocess.run(ssh + ['mkdir', remote], check=True)
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
    for child in local.iterdir():
        tar.add(child, arcname=child.name)
subprocess.run(ssh + [shlex.join(['tar', '-xzf', '-', '-C', remote])],
               input=buffer.getvalue(), check=True)
command = ['python3', remote + '/source/build.py', remote,
           '--cann', '/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0',
           '--vendor', '/home/shanshan/tilert-megakernel-20261021/attention_vendor']
(ROOT / (name + '-command.json')).write_text(json.dumps(command, indent=2) + '\n')
with (ROOT / (name + '-launcher.log')).open('w') as log:
    result = subprocess.run(ssh + [shlex.join(command)], stdout=log, stderr=subprocess.STDOUT)
collect = "import io,sys,tarfile; from pathlib import Path; p=Path(" + repr(remote) + "); t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz'); [t.add(f,arcname=f.name) for f in p.iterdir() if f.is_file() and (f.suffix in ('.o','.so','.log','.jsonl') or f.name=='build.json')]; t.close()"
raw = subprocess.check_output(ssh + ['python3', '-c', shlex.quote(collect)])
results = local / 'results'
results.mkdir()
with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as tar:
    tar.extractall(results, filter='data')
print('Local layer CPU build exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
