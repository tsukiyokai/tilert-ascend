import hashlib,json,shutil
from pathlib import Path
out=Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-terminal212-m12-v1-a3-194')
r=json.loads((out/'result.json').read_text())
assert r['pass'] and r['oracle_mode']=='both' and r['replays']==8
assert r['queued_clone_producers'] and len(r['cases'])==16
sha='b2efdb8d0f6bf1a7dbb7c46cb9610ea0ab37098940081b4975d4196d0330738c'
assert any(v==sha for v in r['libraries'].values())
consumers={'consumer.bridge.combined','consumer.bridge.scaled_routed','consumer.body.routes_bf16','consumer.body.routed_output'}
count=0
for case in r['cases']:
    assert case['pass'] and case['first_mismatch'] is None
    assert len(case['checks'])==35 and consumers<=set(case['checks'])
    assert 'body.active_experts' in case['checks']
    for check in case['checks'].values():
        assert check['bitwise_equal'] and check['bitwise_different_bytes']==0
        assert check.get('finite',True)
        count+=1
assert count==560
summary={'pass':True,'rows':12,'blocks':24,'raw_byte_boundary_checks':count,
 'immediate_consumer_checks':64,'candidate_replays':8,'reference_replays':16,
 'library_sha256':sha,'scope':r['scope'],'oracle_mode':'both',
 'active_table':'All INT32[264] words including padding',
 'report_sha256':hashlib.sha256((out/'result.json').read_bytes()).hexdigest()}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
shutil.copy2(__file__,out/'audit.py')
print(json.dumps(summary,indent=2))
