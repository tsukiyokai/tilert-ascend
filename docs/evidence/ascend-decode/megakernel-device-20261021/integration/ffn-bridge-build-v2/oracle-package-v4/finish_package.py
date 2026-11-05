"""Resume a completed kernel build after restoring SDK template executable modes."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

output = Path('/workspace/tests/glm52/whole_layer/ffn_bridge/oracle-package-v4')
template = Path('/opt/ascend/cann-9.2.0/tools/op_project_templates/ascendc/customize')
manifest = output / 'manifest.json'
report = json.loads(manifest.read_text())
assert not report['complete'] and len(report['commands']) == 3
fixes = {}
for source in template.rglob('*'):
    if not source.is_file() or not source.stat().st_mode & 0o111:
        continue
    relative = source.relative_to(template)
    target = output / 'project' / relative
    expected = report['template_sources'][str(Path('template') / relative)]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
        target.chmod(source.stat().st_mode & 0o777)
        fixes[str(relative)] = oct(source.stat().st_mode & 0o777)
script = Path('/workspace/tests/glm52/whole_layer/ffn_bridge/build_oracle_package.py')
assert hashlib.sha256(script.read_bytes()).hexdigest() == report['script_sha256']
shutil.copyfile(script, output / 'build_oracle_package.py')
shutil.copyfile(__file__, output / 'finish_package.py')
report['permission_repair'] = {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'files': fixes, 'scope': 'restore SDK template executable modes; all contents unchanged'}
for label, command in (
    ('resume-build', ['cmake','--build',str(output/'build'),'-j1']),
    ('resume-install', ['cmake','--install',str(output/'build')])):
    report['commands'].append(command)
    manifest.write_text(json.dumps(report,indent=2)+'\n')
    with (output/(label+'.log')).open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
for key, directory in (('installed_sha256','install'),('project_sha256','project')):
    base=output/directory
    report[key]={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(base.rglob('*')) if p.is_file()}
report['complete']=True
manifest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'complete':True,'manifest':str(manifest),'installed_files':len(report['installed_sha256'])}))
