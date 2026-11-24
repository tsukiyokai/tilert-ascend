import hashlib
import json
from pathlib import Path
import shutil

OUT=Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-wk-m12-int8-smoke-v1-a3-194')
load=lambda name:json.loads((OUT/name).read_text())
correction=load('nz-storage-correction.json')
assert correction['completed'] and correction['additional_hbm_bytes']==0
assert correction['library_sha256']=='d420080a55500e4d908161bcd23b6bf92f4a5c9ee8d8f0d3020a1ba2a2a9ad4a'
assert {(b['layer'],b['parameter']) for b in correction['banks']}=={(l,n) for l in (6,7) for n in ('w13_weight','w2_weight')}
assert len(correction['banks'])==4
for bank in correction['banks']:
    before,after=bank['before'],bank['after']
    k,n=(6144,256) if bank['parameter']=='w13_weight' else (128,6144)
    assert before['logical_shape']==after['logical_shape']==[256,k,n//8]
    assert before['storage_sizes']==[256,n//64,k//16,16,32]
    assert after['storage_sizes']==[256,n//64,k//16,16,8]
    assert after['base_sizes']==after['logical_shape']
    assert after['base_strides']==after['logical_stride']
    assert after['descriptor_scalar_type']==after['logical_scalar_type']==[3]
    assert before['descriptor_scalar_type']==[1]
    assert before['storage_impl']!=after['storage_impl']
    assert before['data_ptr']==after['data_ptr']
    assert before['storage_bytes']==after['storage_bytes']==[256*k*n//2]
    stable=lambda value:{k:v for k,v in value.items() if k!='tensor_impl_use_count'}
    assert stable(before)==stable(bank['source_after'])
    assert stable(after)==stable(bank['installed_parameter'])
    assert bank['source_after']['tensor_impl_use_count'][0]>before['tensor_impl_use_count'][0]
    assert bank['raw_bytes_equal'] and bank['new_device_allocation_bytes']==0
    assert len(bank['raw_sha256_before'])==64 and bank['raw_sha256_before']==bank['raw_sha256_after']
ranks=[]
for rank in range(16):
    r=load(f'rank-{rank:02d}.json')
    assert r['rank']==rank and r['rank_zero_status']=={'success':True}
    assert r['npu_initialized_before'] is False
    assert r['npu_initialized_after']==(rank==0)
    assert r['stages'] and all(s['success'] for s in r['stages'])
    ranks.append(rank)
assert not (OUT/'decoder-diagnostic.json').exists()
forward=load('decoder-forward.json')
assert [x['sha256'] for x in forward['native_libraries']] == ['5f8868bf96efcfd5a53f17890ce8b619287d4463386a2f020c80546332edfabc', '64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c']
assert forward['pass'] and forward['mode']=='int8' and forward['widths']==[12] and forward['iterations']==1
assert forward['tp_geometry']==16 and forward['tp_collectives_executed']==0
assert forward['topk_owners_before_cache_binding']==forward['topk_owners_after_cache_binding']
assert len({v['data_ptr'] for v in forward['topk_owners_after_cache_binding']})==1
assert len(forward['cases'])==1
case=forward['cases'][0]
assert case['pass'] and case['rows']==12 and case['phase']=='comparison'
assert case['native_calls']==[{'rows':12,'mode':'int8'},{'rows':12,'mode':'shared'}]
assert len(case['checks'])==25
for name,value in case['checks'].items():
    assert value['pass'] and value['bitwise_equal'],name
    assert value['bitwise_different_bytes']==value['different_elements']==0,name
summary={
 'pass':True,'scope':forward['scope'],
 'rows':12,'iterations':1,'layers':[6,7],'raw_byte_checks':25,
 'corrected_weight_banks':4,'weight_raw_bytes_preserved':True,
 'diagnostic_pre_post_sync_retained':False,
 'gloo_ranks_completed':16,'npu_ranks':[0],'physical_dies':1,
 'tp_geometry':16,'tp_collectives_executed':0,
 'native_libraries':forward['native_libraries'],
 'alias_library_sha256':correction['library_sha256'],
 'report_sha256':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in
                  ('decoder-forward.json','nz-storage-correction.json')},
 'next_gate':'Exercise all supported row counts and changing inputs, then whole-layer graph replay.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
shutil.copy2(__file__,OUT/'audit.py')
print(json.dumps(summary,indent=2))
