import hashlib
import json
from pathlib import Path
import shutil

OUT=Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-graph-index-int8-m1-v3-a3-194')
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
import torch

ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
NAME = 'vendor-hook-decoder212-graph-index-int8-m1-v3'
raw = ROOT / (NAME + '-raw') / 'report'
diagnostic = load('decoder-graph-index-diagnostic.json')
graph = load('decoder-graph.json')
assert diagnostic['intermediate_evidence_accepted']
assert diagnostic['original_graph_pass'] is False
assert diagnostic['expected_numerical_failure_handled']
assert diagnostic['first_unequal_boundary_in_recorded_order'] == 'key_projection'
assert diagnostic['original_graph']['sha256'] == '12bd361bdf9fd8b7890dadafaf701e260eb38124553b8e0a3d477039b2b07016'
assert diagnostic['script']['sha256'] == '637aa45165021ad4ce970e2475a6cb369f8c215be746d53449e94a34857f38f5'
parent = ROOT / 'vendor-hook-decoder212-graph-int8-wide-v1-raw/report/decoder-graph-int8-m1-provided-r2-native.pt'
assert hashlib.sha256(parent.read_bytes()).hexdigest() == diagnostic['parent_failure_sha256'] == 'e84b65134ba32de1168e285ab32ce9773e2011b68c5c2d0879bfbc1b6839905a'
dump = raw / 'decoder-graph-index-diagnostic.pt'
assert hashlib.sha256(dump.read_bytes()).hexdigest() == diagnostic['dump']['sha256']
saved = torch.load(dump, map_location='cpu', weights_only=True)
original = torch.load(parent, map_location='cpu', weights_only=True)


def equal_bytes(first, second):
    return (first.dtype == second.dtype and first.shape == second.shape
            and torch.equal(first.contiguous().view(torch.uint8),
                            second.contiguous().view(torch.uint8)))


reproduced = 0
for collection in ('plan', 'expected', 'actual'):
    assert saved['failure'][collection].keys() == original[collection].keys()
    for name, value in original[collection].items():
        assert equal_bytes(saved['failure'][collection][name], value), (collection, name)
        reproduced += 1
assert reproduced == 60
assert len(diagnostic['original_failure_reproduction']['checks']) == reproduced
assert all(value['bitwise_equal'] for value in diagnostic['original_failure_reproduction']['checks'].values())
assert all(diagnostic['original_failure_reproduction']['case_and_live_producer_witnesses'].values())
assert len(diagnostic['live_input_checks']) == len(diagnostic['live_producer_checks']) == 4
assert all(value['bitwise_equal'] for value in diagnostic['live_input_checks'].values())
assert all(value['bitwise_equal'] for value in diagnostic['live_producer_checks'].values())
slots = original['plan']['index_slots'].long()
assert slots.tolist() == [1021]
for arm, oracle in (('vendor', 'expected'), ('native', 'actual')):
    values = saved[arm]
    assert equal_bytes(values['hidden'], original[oracle]['layer6.input_norm.0'])
    assert equal_bytes(values['index_slots'], original['plan']['index_slots'])
    assert equal_bytes(values['key'], original[oracle]['layer6.index_cache'].reshape(-1, 128)[slots])
    assert equal_bytes(values['key_scale'], original[oracle]['layer6.index_scale'].reshape(-1)[slots])
for name in ('hidden', 'key_weight', 'query_weight', 'query_weight_scale',
             'gamma', 'beta', 'cosine', 'sine', 'index_slots'):
    assert equal_bytes(saved['native'][name], saved['vendor'][name]), name
assert (saved['native']['key_projection'] != saved['vendor']['key_projection']).nonzero().tolist() == [[0, 25]]
assert (saved['native']['key'] != saved['vendor']['key']).nonzero().tolist() == [[0, 46]]
assert graph['pass'] is False and graph['widths'] == [1]
assert graph['residual_modes'] == ['provided'] and graph['iterations'] == 3
assert len(graph['cases']) == 3 and len(graph['captures']) == 2
assert graph['tp_collectives_executed'] == 0
checks = [(case['iteration'], arm, name, value)
          for case in graph['cases'] for arm, result in case['arms'].items()
          for name, value in result['checks'].items()]
assert len(checks) == 156
failed = [(iteration, arm, name) for iteration, arm, name, value in checks
          if not value['bitwise_equal']]
assert failed == [(2, 'native', 'layer6.index_cache')]
for case in graph['cases']:
    for result in case['arms'].values():
        assert result['python_calls_during_replay'] == 0
        assert result['native_python_calls_during_replay'] == 0
for capture in graph['captures']:
    assert capture['runnable_calls'] == capture['capture_counter_delta'] == 1
    if capture['arm'] == 'native':
        assert capture['workspace_reserve_claimed']
summary = {
    'diagnostic_accepted': True, 'original_graph_pass': False,
    'rows': 1, 'initial_residual': 'provided', 'failed_iteration': 2,
    'raw_reproduced_fixture_tensors': reproduced, 'live_input_checks': 4,
    'cache_producer_checks': 4, 'graph_boundary_checks': 156,
    'graph_boundaries_equal': 155, 'completed_gloo_ranks': 16,
    'physical_dies': 1, 'npu_ranks': [0], 'tp_collectives_executed': 0,
    'first_differing_recorded_boundary': 'WK BF16 projection',
    'projection_coordinate': [0, 25],
    'projection_values': {arm: saved[arm]['key_projection'][0, 25].item()
                          for arm in ('vendor', 'native')},
    'cache_coordinate': [7, 125, 0, 46], 'physical_slot': 1021,
    'cache_values': {arm: saved[arm]['key'][0, 46].item()
                    for arm in ('vendor', 'native')},
    'report_sha256': hashlib.sha256((OUT / 'decoder-graph-index-diagnostic.json').read_bytes()).hexdigest(),
    'graph_sha256': hashlib.sha256((OUT / 'decoder-graph.json').read_bytes()).hexdigest(),
    'dump_sha256': diagnostic['dump']['sha256'],
    'interpretation': 'Matching inputs and retained producer/cache bytes locate the first recorded difference at WK projection. The device accumulation path still requires a captured-input comparison.',
}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
shutil.copy2(__file__, OUT / 'audit.py')
print(json.dumps(summary, indent=2))
