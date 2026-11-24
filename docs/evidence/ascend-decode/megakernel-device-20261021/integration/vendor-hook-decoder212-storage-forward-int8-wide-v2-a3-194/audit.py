import hashlib
import json
from pathlib import Path
import shutil

OUT=Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-storage-forward-int8-wide-v2-a3-194')
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
forward=load('decoder-forward.json')
assert forward['pass'] is False and forward['mode']=='int8'
assert forward['widths']==[1,2,12,32] and forward['iterations']==8
assert forward['tp_geometry']==16 and forward['tp_collectives_executed']==0
assert len(forward['cases'])==17
assert not (OUT/'decoder-diagnostic.json').exists()
failures=[]
checks=0
for index,case in enumerate(forward['cases']):
    rows=1 if index<8 else 2 if index<16 else 12
    assert case['rows']==rows and case['iteration']==index%8
    assert case['pass']==(index<16)
    assert case['native_calls']==[{'rows':rows,'mode':'int8'},{'rows':rows,'mode':'shared'}]
    assert len(case['checks'])==25+(case['initial_residual']=='provided')
    for name,value in case['checks'].items():
        checks+=1
        if not value['pass']:
            failures.append({'rows':rows,'iteration':case['iteration'],'boundary':name,**value})
        else:
            assert value['bitwise_equal'] and value['different_elements']==0
            assert value['bitwise_different_bytes']==0
assert len(failures)==3 and checks==433
assert {f['boundary'] for f in failures}=={'layer6.topk','layer7.topk','layer6.index_cache'}
assert all(f['rows']==12 and f['iteration']==0 for f in failures)
summary={
 'pass':False,'scope':forward['scope'],
 'passed_cases':[{'rows':m,'iterations':8} for m in (1,2)],
 'failed_case':{'rows':12,'iteration':0},'unreached_rows':[32],
 'raw_byte_checks':checks,'raw_byte_passes':checks-len(failures),
 'failures':failures,
 'both_layers_output_and_residual_byte_equal':True,
 'corrected_weight_banks':4,'weight_raw_bytes_preserved':True,
 'diagnostic_pre_post_sync_retained':False,
 'physical_dies':1,'tp_geometry':16,'tp_collectives_executed':0,
 'alias_library_sha256':correction['library_sha256'],
 'report_sha256':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in
                  ('decoder-forward.json','nz-storage-correction.json')},
 'next_gate':'Locate the one-byte index-cache mismatch and dependent Top-K differences without weakening byte acceptance.'}
(OUT/'failure-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
shutil.copy2(__file__,OUT/'audit.py')
print(json.dumps(summary,indent=2))
