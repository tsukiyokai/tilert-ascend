from pathlib import Path
import hashlib,io,json,shlex,shutil,subprocess,tarfile
root=Path('/tmp/tilert-megakernel-validation-20261021')
name='ffn-down-batch212-m12-v1'
out=Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration')/(name+'-a3-194')
out.mkdir()
raw=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes','a3-194',shlex.join(['tar','-czf','-','-C','/home/shanshan/tilert-megakernel-20261021/logs/'+name,'.'])])
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:tar.extractall(out,filter='data')
r=json.loads((out/'result.json').read_text())
assert r['pass'] and r['oracle_mode']=='both' and r['replays']==8 and len(r['cases'])==16
sha='c83c9cb66f9109b810023d95794a4907734e73f66b9526ffd0c3b5987e63d984'
assert sha in r['libraries'].values()
count=0
for case in r['cases']:
    assert case['pass'] and case['first_mismatch'] is None and len(case['checks'])==30
    for check in case['checks'].values():
        assert check['bitwise_equal'] and check['bitwise_different_bytes']==0
        assert check.get('finite',True)
        count+=1
assert count==480
summary={'pass':True,'rows':12,'blocks':24,'raw_byte_boundary_checks':count,
 'candidate_replays':8,'reference_replays':16,'library_sha256':sha,
 'scope':r['scope'],'oracle_mode':'both','performance':'unmeasured',
 'report_sha256':hashlib.sha256((out/'result.json').read_bytes()).hexdigest()}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for suffix in ('command.json','launcher.log'):shutil.copy2(root/(name+'.'+suffix),out/('root-'+suffix))
shutil.copy2(root/('run-'+name+'.py'),out/'launch.py')
shutil.copy2(__file__,out/'audit.py')
print(json.dumps(summary,indent=2))
