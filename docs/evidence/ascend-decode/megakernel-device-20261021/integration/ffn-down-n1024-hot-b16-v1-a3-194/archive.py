import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

ROOT=Path('/tmp/tilert-megakernel-validation-20261021')
NAME='ffn-down-n1024-hot-b16-v1'
RAW=ROOT/(NAME+'-raw')
RAW.mkdir()
OUT=Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration')/(NAME+'-a3-194')
OUT.mkdir()
ssh=['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes','a3-194']
for leaf,remote in [('report',NAME),('driver',NAME+'-driver'),('extra-info',NAME+'-extra-info')]:
    dst=RAW/leaf
    dst.mkdir()
    raw=subprocess.check_output(ssh+[shlex.join(['tar','-czf','-','-C','/home/shanshan/tilert-megakernel-20261021/logs/'+remote,'.'])])
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:tar.extractall(dst,filter='data')
identities={}
for path in RAW.rglob('*'):
    if not path.is_file():continue
    name=str(path.relative_to(RAW))
    identities[name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    if path.suffix in ('.json','.jsonl','.log','.txt'):
        rel=path.relative_to(RAW/'report') if path.is_relative_to(RAW/'report') else path.relative_to(RAW)
        dest=OUT/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
(OUT/'raw-artifacts.json').write_text(json.dumps({'raw_root':str(RAW),'files':identities},indent=2)+'\n')
for suffix in ('command.json','launcher.log','host-memory.json'):
    shutil.copy2(ROOT/(NAME+'.'+suffix),OUT/('root-'+suffix))
shutil.copy2(ROOT/'run-ffn-down-n1024-hot-b16-v1.py',OUT/'root-launcher.py')
shutil.copy2(__file__,OUT/'archive.py')
shutil.copy2(ROOT/'two-die-idle-gate.py',OUT/'idle-gate.py')
shutil.copy2(ROOT/'ffn-down-n1024-hot-b16-v1-admission.jsonl',OUT/'admission.jsonl')
print({'files':len(identities),'bytes':sum(v['bytes'] for v in identities.values()),'out':str(OUT)})
