import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
CANONICAL = Path('tests/glm52/whole_layer/attention_terminal').resolve()
EVIDENCE = Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/shared-attention-terminal-build-v1')
TASK = '/home/shanshan/tilert-megakernel-20261021'
REMOTE = TASK + '/tests/glm52/whole_layer/attention_terminal'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes']
LOCAL = ROOT / 'shared-attention-terminal-runtime-v1'
LOCAL.mkdir()
raw = subprocess.check_output(SSH + ['a3-78', shlex.join(['tar', '-czf', '-', '-C', REMOTE, '.'])])
with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as tar:
    tar.extractall(LOCAL, filter='data')
manifest=json.loads((LOCAL/'build-v1/manifest.json').read_text())
assert manifest['parent_co_load'] and manifest['npu_initialized'] is False
assert len(manifest['sources']) == 26
for name,sha in manifest['sources'].items():
    assert hashlib.sha256((LOCAL/'build-v1/source'/name).read_bytes()).hexdigest() == sha
for name,sha in manifest['artifacts'].items():
    assert hashlib.sha256((LOCAL/'build-v1'/name).read_bytes()).hexdigest() == sha
for name in ('prepare_check.py','test_check.py'):
    shutil.copy2(CANONICAL/name, LOCAL/name)
shutil.copytree(ROOT/'shared-attention-terminal-check-v1', LOCAL/'check-v1')
shutil.copy2(CANONICAL/'test_check.py',LOCAL/'check-v1/test_check.py')
identities={str(path.relative_to(LOCAL)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(LOCAL.rglob('*')) if path.is_file()}
archive=io.BytesIO()
with tarfile.open(fileobj=archive, mode='w:gz') as tar:
    for name in identities:
        tar.add(LOCAL/name,arcname=name)
subprocess.run(SSH+['a3-194',shlex.join(['mkdir',REMOTE])],check=True)
subprocess.run(SSH+['a3-194',shlex.join(['tar','-xzf','-','-C',REMOTE])],input=archive.getvalue(),check=True)
verify='import hashlib,json;from pathlib import Path;p=Path('+repr(REMOTE)+');e=json.loads('+repr(json.dumps(identities))+');bad=[n for n,h in e.items() if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h];assert not bad,bad;print("Verified",len(e),"deployed files")'
subprocess.run(SSH+['a3-194','python3','-c',shlex.quote(verify)],check=True)
EVIDENCE.mkdir()
for name in identities:
    path=LOCAL/name
    if path.suffix in ('.so','.o'):
        continue
    target=EVIDENCE/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,target)
(EVIDENCE/'deployed-sha256.json').write_text(json.dumps(identities,indent=2)+'\n')
for suffix in ('command.json','launcher.log'):
    shutil.copy2(ROOT/('shared-attention-terminal-build-v1.'+suffix),EVIDENCE/('root-'+suffix))
shutil.copy2(__file__, EVIDENCE/'deploy.py')
print(json.dumps({'candidate':manifest['artifacts']['shared_attention_terminal.so'],
 'checker':identities['check-v1/check.py'],'sources':len(manifest['sources']),
 'files':len(identities)},indent=2))
