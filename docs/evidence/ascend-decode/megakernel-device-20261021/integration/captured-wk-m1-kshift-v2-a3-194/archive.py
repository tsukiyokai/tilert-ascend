import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

root = Path('/tmp/tilert-megakernel-validation-20261021')
name = 'captured-wk-m1-kshift-v2'
raw = root / (name + '-raw')
raw.mkdir()
data = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes',
    'a3-194', shlex.join(['tar', '-czf', '-', '-C', '/home/shanshan/tilert-megakernel-20261021/logs/' + name, '.'])])
with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
    archive.extractall(raw, filter='data')
out = Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration') / (name + '-a3-194')
out.mkdir()
identities = {}
for path in raw.rglob('*'):
    if path.is_file():
        rel = path.relative_to(raw)
        identities[str(rel)] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
        if path.suffix in ('.json', '.jsonl', '.log', '.txt'):
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
r = json.loads((raw/'wk-kshift.json').read_text())
assert not r['pass'] and not r['completed'] and r['cases'] == []
assert r['production_reproduction']['bitwise_equal']
assert r['library']['sha256'] == '601c05b229019cb2672c6715babaaf684a49b64c9a412cec5b47c16763b8b59c'
assert 'Kernel left nonfinite output elements' in r['error']
summary = {'pass': False, 'completed': False, 'exit': 1,
           'scope': r['scope'], 'original_F_linear_reproduced': True,
           'accepted_natural_or_shifted_cases': 0,
           'failure': 'The first natural call left nonfinite output elements; the numerical control was not accepted.',
           'raw_outputs_retained': True, 'raw_calls': r['raw_calls'],
           'library_sha256': r['library']['sha256'],
           'report_sha256': hashlib.sha256((raw/'wk-kshift.json').read_bytes()).hexdigest(),
           'physical_die': 4, 'host': 'a3-194',
           'raw_root': str(raw), 'raw_files': identities}
(out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
for source, target in ((name+'.command.json','command.json'),(name+'.launcher.log','launcher.log'),
                       ('run-wk-m1-kshift-v2.py','run.py')):
    shutil.copy2(root/source,out/target)
shutil.copy2(__file__,out/'archive.py')
print(json.dumps({key:summary[key] for key in ('pass','failure','report_sha256')},indent=2))
