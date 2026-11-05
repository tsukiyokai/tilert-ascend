import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys

BLOCKS = int(sys.argv[1])
assert BLOCKS in (16, 24)
NAME = f'ffn-down-k128-hot-b{BLOCKS}-v1'
OUT=Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration')/(NAME+'-a3-194')
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
assert ffn['pass'] and ffn['widths']==[17,32] and ffn['iterations']==8
assert ffn['library_sha256']=='0842cd0f3a946c39aefb3e926e01eb870a61d6d1ad8689aebcf52a1b3daa2b65'
assert ffn['blocks']==BLOCKS and ffn['tp_geometry']==16 and ffn['tp_collectives_executed']==0
assert ffn['native_graph_replay'] and not ffn['reference_graph_replay']
assert not ffn['attention_executed'] and ffn['synthetic_parameters']
assert len(ffn['layers'])==2 and {v['layer'] for v in ffn['layers']}=={6,7}
for layer in ffn['layers']:
    banks={b['parameter']:b for b in correction['banks'] if b['layer']==layer['layer']}
    assert layer['routed_weight_data_ptrs']==[banks[n]['after']['data_ptr'][0] for n in ('w13_weight','w2_weight')]
    assert layer['gmm1']['sha256']==layer['gmm2']['sha256']==layer['scale_source_sha256']=='efdcc610c05a44952c2548e0d8b0cd48ca7704c3e2cf8247f8f08f5e765b830b'
    assert layer['epsilon']==1e-5
assert len(ffn['down_tiling_checks']) == 36
assert sum(t['stage']=='prepared' for t in ffn['down_tiling_checks']) == 4
assert sum(t['stage']=='replay' for t in ffn['down_tiling_checks']) == 32
for tiling in ffn['down_tiling_checks']:
    assert tiling['pass'] and tiling['bytes']==200
    assert tiling['sha256']=='c2b73af9d605ad8e64ea551f288d188a0cf636828a5a0471ea7c63d616f50999'
assert len(ffn['cases'])==32
assert {(c['layer'],c['rows'],c['iteration']) for c in ffn['cases']}=={(l,m,i) for l in (6,7) for m in (17,32) for i in range(8)}
boundaries={'normalized','retained','logits','shared','routed','combined','pre_ar2'}
consumers={'combined','routed','shared','retained'}
assert set(ffn['immediate_graph_consumers'])==consumers
boundaries |= {'consumer.' + name for name in consumers}
metadata = {'route_ids', 'route_probabilities', 'route_counts', 'active_experts'}
assert set(ffn['metadata_checks']) == metadata
boundaries |= metadata
assert ffn['script_sha256'] == '21c094ced4e644b296ebd4f974cfffe68431ad02d4c8245a8bce22904bff2923'
bias_sha = hashlib.sha256(struct.pack('<256f', *([4.0] * 8 + [-4.0] * 248))).hexdigest()
for layer in ffn['layers']:
    bias = layer['hot_bias']
    assert bias['address'] > 0 and bias['dtype'] == 'torch.float32' and bias['shape'] == [256]
    assert bias['sha256'] == bias_sha and bias['selected_experts'] == list(range(8))
    assert bias['selected_bias'] == 4.0 and bias['other_bias'] == -4.0 and bias['router_scale'] == 1.0

for case in ffn['cases']:
    assert case['pass'] and case['rows'] in (17,32) and case['iteration'] in range(8)
    assert set(case['checks'])==boundaries
    hot = case['hot_route']
    assert hot['pass'] and hot['every_token_selects_first_eight_once'] and hot['finite_probabilities']
    assert hot['active_experts'] == list(range(8))
    assert hot['counts'] == [case['rows']] * 8 + [0] * 248
    assert hot['maximum_count'] == case['rows']
    assert hot['msd_rows_per_active_expert'] == 2 * case['rows']
    assert hot['down_m_tiles_per_expert'] == 2

    assert all(v['pass'] and v['bitwise_equal'] and v['bitwise_different_bytes']==0 and v['different_elements']==0 for v in case['checks'].values())
result=load('rank-00.json')['result']
assert result['complete_ffn_executed'] and not result['full_decoder_forward_executed']
summary={
 'pass':True,'scope':ffn['scope'],'rows':[17,32],'iterations':8,'layers':[6,7],
 'raw_byte_boundary_checks':480,'immediate_consumer_checks':128,'boundaries':sorted(boundaries),
 'original_ffn_oracle':'Actual DecoderLayer post_attention_layernorm and original MLP, including original GMM1/GMM2/shared operators',
 'candidate_library_sha256':ffn['library_sha256'],
 'corrected_weight_banks':4,'original_nz_allocations_and_raw_bytes_preserved':True,
 'scale_encoding':'FP32 bit patterns decoded from the original zero-extended INT64 carriers',
 'native_graph_replay':True,'reference_graph_replay':False,
 'synthetic_parameters':True,'full_decoder_executed':False,
 'full_down_tiling_byte_checks':36,'down_tiling_bytes':200,'down_tiling_sha256':'c2b73af9d605ad8e64ea551f288d188a0cf636828a5a0471ea7c63d616f50999',
 'physical_dies':1,'npu_ranks':[0],'completed_gloo_ranks':16,
 'tp_geometry':16,'tp_collectives_executed':0,
 'report_sha256':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ('decoder-ffn.json','nz-storage-correction.json')},
 'native_blocks': BLOCKS, 'forced_active_experts': 8, 'hot_route_cases': 32,
 'metadata_byte_checks': 128, 'original_bias_owner_preserved': True,
 'both_down_m_tiles_exercised': True,
 'next_gate':'One-launch local layer composition with the accepted original Top-K copy.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
shutil.copy2(__file__,OUT/'audit.py')
print(json.dumps(summary,indent=2))
