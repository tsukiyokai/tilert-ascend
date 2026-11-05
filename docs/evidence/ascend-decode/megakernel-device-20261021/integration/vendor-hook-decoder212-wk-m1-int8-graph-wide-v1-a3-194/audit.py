import hashlib
import json
from pathlib import Path
import shutil

OUT=Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-wk-m1-int8-graph-wide-v1-a3-194')
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
graph = load('decoder-graph.json')
assert graph['pass'] and graph['mode'] == 'int8'
assert graph['widths'] == [1,2,12,32] and graph['iterations'] == 8
assert graph['residual_modes'] == ['none', 'provided']
assert graph['tp_geometry'] == 16 and graph['tp_collectives_executed'] == 0
assert graph['host_sequence_bound'] == 8192 and graph['page_table_columns'] == 64
assert not graph['updatable_graph_branch']
assert graph['decoder_forward']['sha256'] == '96f4f755f62caa85869fb884209e7f8869882ddb6efbd40c930778b27bd3e39d'
assert graph['script']['sha256'] == '12bd361bdf9fd8b7890dadafaf701e260eb38124553b8e0a3d477039b2b07016'
assert [v['sha256'] for v in graph['native_libraries']] == [
    '8deff750464e23bb871302c8626b86d37b60c28a00991d550f95dfcfd47ecb6f',
    '64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c']
expected_captures = {(rows, residual, arm) for rows in (1,2,12,32) for residual in ('none', 'provided')
                     for arm in ('vendor', 'native')}
assert len(graph['captures']) == len(expected_captures)
assert {(c['rows'], c['initial_residual'], c['arm'])
        for c in graph['captures']} == expected_captures
for capture in graph['captures']:
    assert capture['runnable_calls'] == capture['capture_counter_delta'] == 1
    assert capture['graph_type'] == 'UpdatableGraph'
    assert capture['capture_stream'] != capture['caller_stream']
    assert len(capture['output_owner_ptrs']) == 2
    modes = [] if capture['arm'] == 'vendor' else ['int8', 'shared']
    assert capture['native_python_modes'] == modes
    if capture['arm'] == 'native':
        assert capture['workspace_reserve_claimed'] is True
    for layer in ('layer6', 'layer7'):
        traces = capture['tp_cuts'][layer]['tp_cuts']
        assert [v['phase'] for v in traces] == ['attention', 'ffn']
        assert [v['action'] for v in traces] == ['external_post_ar1', 'local_pre_ar2']
        assert capture['tp_cuts'][layer]['residual_is_post_norm_owner']
expected_cases = {(rows, residual, i) for rows in (1,2,12,32) for residual in ('none', 'provided')
                  for i in range(8)}
assert len(graph['cases']) == len(expected_cases)
assert {(c['rows'], c['initial_residual'], c['iteration'])
        for c in graph['cases']} == expected_cases
count = replays = 0
for case in graph['cases']:
    assert case['pass'] and case['phase'] == 'comparison'
    assert set(case['arms']) == {'vendor', 'native'}
    length = 8192 - case['iteration'] % 11
    inputs = case['inputs']
    assert inputs['actual_main_lengths'] == [[length], [length]]
    assert inputs['host_main_lengths'] == [[length], [length]]
    assert inputs['actual_index_length'] == [length]
    assert inputs['host_max_seq_len_bounds'] == [8192, 8192]
    tensors = inputs['tensors']
    main = tensors['model.layers.6.self_attn.attn:block_table']['sha256']
    index = tensors['model.layers.6.self_attn.indexer.k_cache:block_table']['sha256']
    assert main != index
    for arm in case['arms'].values():
        assert arm['pass'] and arm['replay_count'] == case['iteration'] + 1
        assert arm['python_calls_during_replay'] == arm['native_python_calls_during_replay'] == 0
        assert arm['output'] == case['eager_output']
        assert len(arm['checks']) == 25 + (case['initial_residual'] == 'provided')
        for name, value in arm['checks'].items():
            assert value['pass'] and value['bitwise_equal'], name
            assert value['bitwise_different_bytes'] == value['different_elements'] == 0
        count += len(arm['checks'])
        replays += 1
for rows in (1,2,12,32):
    for residual in ('none', 'provided'):
        selected = [c for c in graph['cases'] if c['initial_residual'] == residual and c['rows'] == rows]
        assert len({c['eager_output']['sha256'] for c in selected}) == 8
        for layer in ('layer6', 'layer7'):
            assert len({c['eager_attention'][layer]['sha256'] for c in selected}) == 8
        for case in selected:
            assert case['eager_output']['nonzero_elements'] > 0
            assert all(v['nonzero_elements'] > 0 for v in case['eager_attention'].values())
assert count == 3264 and replays == 128
summary = {
    'pass': True, 'scope': graph['scope'], 'rows': [1,2,12,32],
    'iterations_per_residual_mode': 8, 'residual_modes': graph['residual_modes'],
    'real_graph_captures': len(graph['captures']), 'explicit_graph_replays': replays,
    'raw_byte_boundary_checks': count, 'python_calls_during_replay': 0,
    'actual_lengths_and_distinct_page_maps_change': True,
    'strong_output_owners_and_poisoned_observations': True,
    'native_capture_reserves_claimed': True, 'original_ffn_disposal': 'Unchanged',
    'original_decoder_module_sha256': graph['decoder_forward']['sha256'],
    'native_libraries': graph['native_libraries'],
    'physical_dies': 1, 'completed_gloo_ranks': 16, 'npu_ranks': [0],
    'tp_geometry': 16, 'tp_collectives_executed': 0,
    'report_sha256': hashlib.sha256((OUT / 'decoder-graph.json').read_bytes()).hexdigest(),
    'next_gate': 'Complete native FFN integration with original Decoder graph',
}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
shutil.copy2(__file__, OUT / 'audit.py')
print(json.dumps(summary, indent=2))
