import hashlib
import json
from pathlib import Path
import shutil

OUT=Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-index-int8-m12-v1-a3-194')
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
diagnostic=load('decoder-index-diagnostic.json')
forward=load('decoder-forward.json')
assert not forward['pass'] and len(forward['cases'])==1
assert forward['widths']==[12] and forward['iterations']==1
assert diagnostic['original_failure_reproduction']['pass']
assert all(v['bitwise_equal'] for v in diagnostic['original_failure_reproduction']['checks'].values())
assert diagnostic['calls']=={'vendor':1,'native':1,'rope_snapshots':2}
assert diagnostic['first_unequal_boundary_in_recorded_order']=='key_projection'
expected={'key_projection':2,'key_normalized':2,'key_rope':1,'key_hadamard':6,'key':1}
assert {n:v['different_elements'] for n,v in diagnostic['checks'].items() if not v['bitwise_equal']}==expected
raw=Path('/tmp/tilert-megakernel-validation-20261021/vendor-hook-decoder212-index-int8-m12-v1-raw/report/decoder-index-diagnostic.pt')
assert hashlib.sha256(raw.read_bytes()).hexdigest()==diagnostic['dump']['sha256']
summary={
 'diagnostic_complete':True,'original_decoder_gate_pass':False,
 'original_failure_reproduced_bitwise':True,
 'first_unequal_boundary':'key_projection',
 'boundary_different_elements':expected,
 'key_projection_differences':diagnostic['checks']['key_projection']['first_differences'],
 'final_key_difference':diagnostic['checks']['key']['first_differences'],
 'inputs_and_weights_byte_equal':True,
 'query_path_and_scales_byte_equal':True,
 'both_layers_original_ffn_preserved':True,
 'rows':12,'iteration':0,'layers':[6,7],'physical_dies':1,
 'tp_geometry':16,'tp_collectives_executed':0,
 'diagnostic_perturbation':diagnostic['perturbation'],
 'dump_sha256':diagnostic['dump']['sha256'],
 'report_sha256':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ('decoder-index-diagnostic.json','decoder-forward.json','nz-storage-correction.json')},
 'next_gate':'Reconcile the actual WK accumulation and BF16 rounding path using these same inputs.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
shutil.copy2(__file__,OUT/'audit.py')
print(json.dumps(summary,indent=2))
