import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

ROOT=Path('/tmp/tilert-megakernel-validation-20261021')
NAME='shared-attention-terminal212-wide-v1'
OUT=Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration')/(NAME+'-a3-194')
OUT.mkdir()
remote='/home/shanshan/tilert-megakernel-20261021/logs/'+NAME
raw=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes','a3-194',shlex.join(['tar','-czf','-','-C',remote,'.'])])
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:tar.extractall(OUT,filter='data')
report=json.loads((OUT/'result.json').read_text())
assert report['provenance']['library_sha256']=='2a2deacf850ce108825002559a1a44e08d6176fdc1c9a092955acc6840eb1ea4'
assert report['harness_sha256']=='30102cd3f32718eb3dde6fe7fcab3dcaf9eec451bc7c192460684a03b41df9a7'
assert set(report['comparison_libraries'].values())=={'64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c'}
assert {(c['width'],c['blocks']) for c in report['cases']}=={(m,b) for m in (1,2,12,32) for b in (16,24)}
assert len(report['cases'])==8
boundaries={'output','latent','kv_cache','rope_cache','full_consumer','control_consumer'}
checks=0
for case in report['cases']:
    assert case['pass'] and case['capture'] and case['sequence']==8192
    assert case['requested_workspace_bytes']['full']==case['requested_workspace_bytes']['separate']
    assert len(case['checks'])==8
    for index,check in enumerate(case['checks']):
        assert check['replay']==index and check['pass']
        assert set(check)-{'replay','pass'}==boundaries
        for boundary in boundaries:
            metric=check[boundary]
            assert metric['pass'] and metric['bitwise_equal']
            assert metric['bitwise_different_bytes']==metric['different_elements']==0
            checks+=1
summary={'pass':True,'scope':'Standalone shared-attention terminal completion vs frozen64108, including immediate same-graph device output consumers',
 'geometries':8,'inputs_per_geometry':8,'raw_byte_checks':checks,'consumer_checks':8*8*2,
 'candidate_sha256':'2a2deacf850ce108825002559a1a44e08d6176fdc1c9a092955acc6840eb1ea4',
 'parent_sha256':'64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c',
 'checker_sha256':report['harness_sha256'],
 'report_sha256':hashlib.sha256((OUT/'result.json').read_bytes()).hexdigest()}
assert checks==384
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for suffix in ('command.json','launcher.log'):
    shutil.copy2(ROOT/(NAME+'.'+suffix),OUT/('root-'+suffix))
shutil.copy2(ROOT/('run-'+NAME+'.py'),OUT/'launch.py')
shutil.copy2(__file__,OUT/'audit.py')
print(json.dumps(summary,indent=2))
