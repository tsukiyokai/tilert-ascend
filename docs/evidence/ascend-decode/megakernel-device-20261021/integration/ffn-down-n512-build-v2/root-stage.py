"""Verify the frozen build, then stage the unchanged original FFN oracle."""
import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

root = Path('/tmp/tilert-megakernel-validation-20261021')
task = '/home/shanshan/tilert-megakernel-20261021'
name = 'ffn_down_n512'
local = root / 'ffn-down-n512-build-v2'
remote = task + '/tests/glm52/whole_layer/' + name
ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes']
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = json.loads((local / 'manifest.json').read_text())
identities = {**manifest['sources'], **manifest['artifacts'], 'manifest.json': digest(local / 'manifest.json')}
for filename, expected in identities.items():
    assert digest(local / filename) == expected, filename
library = manifest['artifacts'][name + '.so']
assert library == 'c90b3666a1697070eda19d5b3e133e9ed58f36fe429abefceae6ffe102a6a032'
assert manifest['parent_co_load'] and not manifest['npu_initialized']
assert manifest['parent_library_sha256'] == '381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2'
verify = ('import hashlib,json; from pathlib import Path; base=Path(' + repr(remote + '/build-v2')
          + '); identities=json.loads(' + repr(json.dumps(identities))
          + '); bad=[name for name,expected in identities.items() if hashlib.sha256((base/name).read_bytes()).hexdigest()!=expected]; assert not bad,bad; print("Verified frozen files",len(identities))')
subprocess.run(ssh + ['a3-194', 'python3', '-c', shlex.quote(verify)], check=True)
evidence = Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-down-n512-build-v2')
evidence.mkdir()
for path in local.rglob('*'):
    if path.is_file() and path.suffix in ('.py', '.json', '.jsonl', '.log', '.txt', '.md', '.h', '.cpp', '.sh', '.patch'):
        output = evidence / path.relative_to(local)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, output)
shutil.copy2(__file__, evidence / 'root-stage.py')
stage = Path('tests/glm52/whole_layer/ffn_down_n512/stage_decoder.py')
subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes', str(stage), 'a3-194:' + remote + '/stage_decoder.py'], check=True)
command = ['python3', remote + '/stage_decoder.py', '--parent', task + '/vendor-hook-decoder212-ffn-v2',
           '--library', remote + '/build-v2/' + name + '.so', '--library-sha256', library,
           '--output', task + '/vendor-hook-decoder212-ffn-down-n512-v1']
(root / 'ffn-down-n512-stage-v1.command.json').write_text(json.dumps(command, indent=2) + '\n')
with (root / 'ffn-down-n512-stage-v1.log').open('w') as log:
    subprocess.run(ssh + ['a3-194', shlex.join(command)], check=True, stdout=log, stderr=subprocess.STDOUT)
print(json.dumps({'pass': True, 'sources': len(manifest['sources']), 'artifacts': len(manifest['artifacts']),
                  'library_sha256': library, 'manifest_sha256': identities['manifest.json']}))
