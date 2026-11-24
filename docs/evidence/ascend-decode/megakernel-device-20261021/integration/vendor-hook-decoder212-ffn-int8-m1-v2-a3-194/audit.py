import hashlib
import json
from pathlib import Path
import shutil

OUT=Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-ffn-int8-m1-v2-a3-194')
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
ffn=load('decoder-ffn.json')
assert ffn['pass'] and ffn['widths']==[1] and ffn['iterations']==1
assert ffn['library_sha256']=='8430ef17d068ce5a1cdbe4915ba189ca38c666322d8726a42091f8acb3b36f84'
assert ffn['blocks']==24 and ffn['tp_geometry']==16 and ffn['tp_collectives_executed']==0
assert ffn['native_graph_replay'] and not ffn['reference_graph_replay']
assert not ffn['attention_executed'] and ffn['synthetic_parameters']
assert len(ffn['layers'])==2 and {v['layer'] for v in ffn['layers']}=={6,7}
for layer in ffn['layers']:
    banks={b['parameter']:b for b in correction['banks'] if b['layer']==layer['layer']}
    assert layer['routed_weight_data_ptrs']==[banks[n]['after']['data_ptr'][0] for n in ('w13_weight','w2_weight')]
    assert layer['gmm1']['sha256']==layer['gmm2']['sha256']==layer['scale_source_sha256']=='efdcc610c05a44952c2548e0d8b0cd48ca7704c3e2cf8247f8f08f5e765b830b'
    assert layer['epsilon']==1e-5
assert len(ffn['cases'])==2
boundaries={'normalized','retained','logits','shared','routed','combined','pre_ar2'}
for case in ffn['cases']:
    assert case['pass'] and case['rows']==1 and case['iteration']==0
    assert set(case['checks'])==boundaries
    assert all(v['pass'] and v['bitwise_equal'] and v['bitwise_different_bytes']==0 and v['different_elements']==0 for v in case['checks'].values())
result=load('rank-00.json')['result']
assert result['complete_ffn_executed'] and not result['full_decoder_forward_executed']
summary={
 'pass':True,'scope':ffn['scope'],'rows':[1],'iterations':1,'layers':[6,7],
 'raw_byte_boundary_checks':14,'boundaries':sorted(boundaries),
 'original_ffn_oracle':'Actual DecoderLayer post_attention_layernorm and original MLP, including original GMM1/GMM2/shared operators',
 'candidate_library_sha256':ffn['library_sha256'],
 'corrected_weight_banks':4,'original_nz_allocations_and_raw_bytes_preserved':True,
 'scale_encoding':'FP32 bit patterns decoded from the original zero-extended INT64 carriers',
 'native_graph_replay':True,'reference_graph_replay':False,
 'synthetic_parameters':True,'full_decoder_executed':False,
 'physical_dies':1,'npu_ranks':[0],'completed_gloo_ranks':16,
 'tp_geometry':16,'tp_collectives_executed':0,
 'report_sha256':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ('decoder-ffn.json','nz-storage-correction.json')},
 'next_gate':'All supported widths and changing inputs, followed by actual decoder integration.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
shutil.copy2(__file__,OUT/'audit.py')
print(json.dumps(summary,indent=2))
