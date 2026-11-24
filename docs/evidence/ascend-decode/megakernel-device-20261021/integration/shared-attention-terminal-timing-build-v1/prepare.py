import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

ROOT=Path('/tmp/tilert-megakernel-validation-20261021')
OLD=Path('tests/glm52/whole_layer/attention_shared_descriptor')
NEW=Path('tests/glm52/whole_layer/attention_terminal')
LOCAL=ROOT/'shared-attention-terminal-timing-v1'
LOCAL.mkdir()
SHA='2a2deacf850ce108825002559a1a44e08d6176fdc1c9a092955acc6840eb1ea4'
PARENT_SHA='ff36c603205f43420b2181ae7193a3b6b4b8a3a2445db84a3cf74840664c4d64'
TASK='/home/shanshan/tilert-megakernel-20261021'
REMOTE=TASK+'/tests/glm52/whole_layer/attention_terminal/timing-v1'
SSH=['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes']
originals={'bench.py':OLD/'bench.py', 'test_bench.py':OLD/'test_bench.py',
           'cpu_preflight.py':ROOT/'shared-attention-descriptor-timing-v1/cpu_preflight.py'}
parent_hashes={}
for name,path in originals.items():
    source=path.read_text()
    parent_hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    candidate=(source.replace(PARENT_SHA,SHA)
               .replace('tilert_shared_descriptor','tilert_attention_terminal')
               .replace('shared_attention_descriptor.so','shared_attention_terminal.so')
               .replace('SHARED_DESCRIPTOR','ATTENTION_TERMINAL')
               .replace('shared-attention-descriptor','shared-attention-terminal'))
    (LOCAL/name).write_text(candidate)
    assert not (NEW/name).exists()
    (NEW/name).write_text(candidate)
assert parent_hashes['bench.py']=='809e8abef5fd979f19a62de29818d19f45eb3ba8a41dfa7b04d87899e1207246'
identities={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in LOCAL.iterdir()}
manifest={'candidate_sha256':SHA,'parent_library_sha256':'64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c',
          'parent_benchmark_source_sha256':parent_hashes,'sources':identities,
          'scope':'Same unmodified matched-address benchmark; only candidate namespace and identities change.'}
(LOCAL/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=io.BytesIO()
with tarfile.open(fileobj=archive,mode='w:gz') as tar:
    for path in LOCAL.iterdir():tar.add(path,arcname=path.name)
subprocess.run(SSH+['a3-78',shlex.join(['mkdir',REMOTE])],check=True)
subprocess.run(SSH+['a3-78',shlex.join(['tar','-xzf','-','-C',REMOTE])],input=archive.getvalue(),check=True)
inner='''set -e
source /opt/ascend/cann-9.2.0/set_env.sh
PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/tmp/cache OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 /candidate/timing-v1/cpu_preflight.py --build /candidate/build-v1 --control-library /workspace/logs/runtime212-build-v1/attention/shared/whole_layer_full_attention.so --output /candidate/timing-v1/cpu-preflight.json
ATTENTION_TERMINAL_BUILD=/candidate/build-v1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 /candidate/test_check.py -v
'''
# The numerical checker CPU contracts run in the same actual ARM/Torch environment.
for name in ('prepare_check.py','test_check.py'):
    subprocess.run(SSH+['a3-78',shlex.join(['tee',REMOTE.rsplit('/',1)[0]+'/'+name])],input=(NEW/name).read_bytes(),stdout=subprocess.DEVNULL,check=True)
command=['docker','run','--rm','--pull=never','--read-only','--network','none','--workdir','/tmp','--cpus','1','--memory','12g','--pids-limit','128',
 '--tmpfs','/tmp:rw,exec,nosuid,nodev,size=1g','-v',TASK+':/workspace:ro','-v',REMOTE.rsplit('/',1)[0]+':/candidate:rw',
 '-v','/home/shanshan/megamoe-mask-only-20261021/sdk/cann-9.2.0:/opt/ascend/cann-9.2.0:ro','--entrypoint','/bin/bash',
 'sha256:7448df539bcb40397063c4b7e83a6a1590ba20bcc16639c0aa83f1c60f1a0449','-c',inner]
(LOCAL/'cpu-command.json').write_text(json.dumps(command,indent=2)+'\n')
with (LOCAL/'cpu-tests.log').open('w') as log:
    subprocess.run(SSH+['a3-78',shlex.join(command)],stdout=log,stderr=subprocess.STDOUT,check=True)
raw=subprocess.check_output(SSH+['a3-78','cat',REMOTE+'/cpu-preflight.json'])
(LOCAL/'cpu-preflight.json').write_bytes(raw)
report=json.loads(raw)
assert report['pass'] and not report['npu_initialized']
identities={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in LOCAL.iterdir()}
archive=io.BytesIO()
with tarfile.open(fileobj=archive,mode='w:gz') as tar:
    for name in identities:tar.add(LOCAL/name,arcname=name)
subprocess.run(SSH+['a3-194',shlex.join(['mkdir',REMOTE])],check=True)
subprocess.run(SSH+['a3-194',shlex.join(['tar','-xzf','-','-C',REMOTE])],input=archive.getvalue(),check=True)
verify='import hashlib,json;from pathlib import Path;p=Path('+repr(REMOTE)+');e=json.loads('+repr(json.dumps(identities))+');bad=[n for n,h in e.items() if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h];assert not bad,bad;print("Verified",len(e),"timing files")'
subprocess.run(SSH+['a3-194','python3','-c',shlex.quote(verify)],check=True)
EVIDENCE=Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/shared-attention-terminal-timing-build-v1')
shutil.copytree(LOCAL,EVIDENCE)
(EVIDENCE/'deployed-sha256.json').write_text(json.dumps(identities,indent=2)+'\n')
shutil.copy2(__file__, EVIDENCE/'prepare.py')
print(json.dumps({'bench_sha256':identities['bench.py'],'cpu_tests':report['cpu_tests'],'npu_initialized':False},indent=2))
