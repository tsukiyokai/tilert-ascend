"""Check two-rank HCCL-versus-SHMEM layer results before archiving them."""

import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
REPO = Path('/home/shanshan/repo/tile-ai/TileRT-Ascend')
NAME = sys.argv[1]
WIDTHS = list(map(int, sys.argv[2].split(',')))
ITERATIONS = int(sys.argv[3])
RAW = ROOT / (NAME + '-raw')
DIRECTORY = RAW / NAME
FROZEN = Path('/tmp/tilert-tp-chain-staging-v2')
LIBRARIES = {
    '609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f',
    'c2b72e469c3ee80a5fe7d9df4a3cceedc0e65cca95e6f426aea762c213392462',
    '8deff750464e23bb871302c8626b86d37b60c28a00991d550f95dfcfd47ecb6f',
    '64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c',
    '381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2',
}
SHARDED = {
    'self_attn.q_b_proj.weight', 'self_attn.kv_b_proj.weight',
    'self_attn.o_proj.weight', 'mlp.shared_experts.gate_up_proj.weight',
    'mlp.shared_experts.down_proj.weight',
    'mlp.experts.routed_experts.w13_weight', 'mlp.experts.routed_experts.w2_weight',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(value):
    assert value['pass'] and value['bitwise_equal']
    assert value['bitwise_different_bytes'] == 0


assert digest(FROZEN / 'manifest.json') == '4d608c5a7a382a37002035026871b4cb3a64c20c2947242f15a38692d635507f'
manifest = json.loads((FROZEN / 'manifest.json').read_text())
for name, expected in manifest['files_sha256'].items():
    assert digest(FROZEN / name) == expected, name
driver = json.loads((DIRECTORY / 'summary.json').read_text())
assert driver['success'] and driver['exitcodes'] == [0] * 16
assert driver['script_sha256'] == manifest['files_sha256']['tp_device.py']
assert driver['arguments']['physical_devices'] == [6, 7]
assert driver['arguments']['forward_widths'] == WIDTHS
assert driver['arguments']['forward_replays'] == ITERATIONS
workers = []
for rank in range(16):
    worker = json.loads((DIRECTORY / f'rank-{rank:02d}.json').read_text())
    assert worker['success'] and worker['rank'] == rank
    assert worker['torch'] == '2.12.0+cu130' and worker['torch_npu'] == '2.12.0'
    assert not worker['npu_initialized_before']
    assert worker['npu_initialized_after'] == (rank < 2)
    assert worker['device_participant'] == (rank < 2)
    assert all(stage['success'] for stage in worker['stages'])
    assert worker['all_rank_statuses'] == [
        {'rank': value, 'success': True} for value in range(16)]
    stages = {row['name']: row for row in worker['stages']}
    assert stages['actual_tp16_gloo_geometry']['cpu_rank_sum'] == 120
    assert stages['actual_tp16_gloo_geometry']['tp_rank'] == rank
    if rank < 2:
        mapping = stages['physical_device_mapping']
        assert mapping['logical_to_physical'] == [6, 7]
        assert mapping['logical_rank'] == rank and mapping['can_access_peer'] == 1
        group = stages['actual_mc2_and_hccl2_groups']
        assert group['mc2_ranks'] == [rank] and group['oracle_hccl_ranks'] == [0, 1]
        assert group['oracle_rank'] == rank and group['tp_constructor_size'] == 16
        assert not group['tp16_device_collective']
        assert worker['result']['two_rank_component_chain_executed']
        assert not worker['result']['full_decoder_forward_executed']
    workers.append(stages)
for layer in (6, 7):
    rows = [{value['name']: value for value in stages[f'layer{layer}_move_parameters']['parameters']}
            for stages in workers[:2]]
    left, right = rows
    assert left.keys() == right.keys() and SHARDED <= left.keys()
    for name in left:
        for field in ('shape', 'dtype', 'bytes'):
            assert left[name][field] == right[name][field]
        assert (left[name]['sha256'] != right[name]['sha256']) == (name in SHARDED)
        assert left[name]['replicated'] == right[name]['replicated'] == (name not in SHARDED)
counts = {'raw_byte_checks': 0, 'hccl_ordered_checks': 0}
rank_reports = []
for rank in (0, 1):
    path = DIRECTORY / f'device-rank{rank}'
    report = json.loads((path / 'tp-chain.json').read_text())
    assert report['pass'] and report['rank'] == rank and report['ranks'] == 2
    assert report['constructor_tp_size'] == 16 and report['tp16_collectives_executed'] == 0
    assert report['eager'] and not report['graph_capture'] and not report['timing']
    assert not report['original_decoder_forward_called']
    assert report['widths'] == WIDTHS and report['iterations'] == ITERATIONS
    assert report['blocks'] == 24 and set(report['libraries'].values()) == LIBRARIES
    for key, filename in (('script', 'tp_chain.py'), ('parent_phases', 'tp_parent.py'),
                          ('tp_host', 'tp_host.py')):
        assert report[key]['sha256'] == manifest['files_sha256'][filename]
    assert report['actual_decoder']['sha256'] == '96f4f755f62caa85869fb884209e7f8869882ddb6efbd40c930778b27bd3e39d'
    assert report['window']['bytes'] == 1573184 and report['window']['capacity'] == 32
    assert report['window']['address'] > 0 and report['window']['stream'] > 0
    assert len(report['cases']) == ITERATIONS * len(WIDTHS)
    for width in WIDTHS:
        cases = [row for row in report['cases'] if row['rows'] == width]
        assert [row['iteration'] for row in cases] == list(range(ITERATIONS))
        for key in ('consumer.output', 'consumer.retained'):
            assert len({row['outputs'][key]['sha256'] for row in cases}) == ITERATIONS
            assert all(row['outputs'][key]['finite'] and row['outputs'][key]['nonzero_elements'] > 0
                       for row in cases)
        for key in ('hidden', 'slot_i64:0', 'slot_i64:2'):
            assert len({row['inputs']['tensors'][key]['sha256'] for row in cases}) == ITERATIONS
        for row in cases:
            assert row['pass'] and len(row['checks']) == 113
            for value in row['checks'].values():
                exact(value)
                counts['raw_byte_checks'] += 1
            reductions = row['hccl_ordered_checks']
            assert [(value['layer'], value['phase']) for value in reductions] == [
                (layer, phase) for layer in (6, 7) for phase in ('ar1', 'ar2')]
            for value in reductions:
                exact(value['check'])
                assert len(value['partials']) == 2
                assert value['partials'][0]['sha256'] != value['partials'][1]['sha256']
                counts['hccl_ordered_checks'] += 1
            left, right = row['rank_witnesses']
            assert left['rank'] == 0 and right['rank'] == 1
            for key in ('input_hidden', 'input_residual', 'layer6.retained', 'layer7.retained',
                        'layer6.ar1', 'layer6.ar2', 'layer7.ar1', 'layer7.ar2'):
                assert left[key] == right[key]
            for key in ('layer6.attention_partial', 'layer6.ffn_partial',
                        'layer7.attention_partial', 'layer7.ffn_partial'):
                assert left[key] != right[key]
    nz = json.loads((path / 'nz-storage-correction.json').read_text())
    assert nz['completed'] and nz['additional_hbm_bytes'] == 0 and len(nz['banks']) == 4
    for bank in nz['banks']:
        assert bank['raw_bytes_equal'] and bank['raw_sha256_before'] == bank['raw_sha256_after']
        assert bank['new_device_allocation_bytes'] == 0
        assert bank['before']['data_ptr'] == bank['after']['data_ptr']
    rank_reports.append(report)
for left, right in zip(*(report['cases'] for report in rank_reports)):
    assert left['rank_witnesses'] == right['rank_witnesses']
expected_cases = 2 * len(WIDTHS) * ITERATIONS
assert counts == {'raw_byte_checks': expected_cases * 113,
                  'hccl_ordered_checks': expected_cases * 4}
summary = {
    'scope': 'Two controlled TP16-shaped shards: local components plus HCCL2 versus complete layers with SHMEM2',
    'host': 'a3-194', 'physical_dies': [6, 7], 'rows': WIDTHS,
    'inputs_per_width': ITERATIONS, 'device_ranks': 2, 'cpu_geometry_peers': 14,
    'pass': True, **counts, 'defined_boundaries_per_rank_case': 113,
    'four_reductions_per_two_layer_case': True, 'actual_hccl_oracle': True,
    'different_rank_partials': True, 'same_replicated_and_reduced_values': True,
    'window_bytes_per_rank': 1573184, 'host_epoch_reset': False,
    'graph_capture': False, 'timing': False, 'checkpoint_loaded': False,
    'tp16_device_collectives': 0, 'candidate_library_sha256': manifest['tp_library_sha256'],
    'report_sha256': {str(rank): digest(DIRECTORY / f'device-rank{rank}' / 'tp-chain.json')
                      for rank in (0, 1)},
    'fixture_manifest_sha256': digest(FROZEN / 'manifest.json'),
}
destination = REPO / 'docs/evidence/ascend-decode/megakernel-device-20261021/integration' / (NAME + '-a3-194')
destination.mkdir(exist_ok=False)
for directory in RAW.iterdir():
    shutil.copytree(directory, destination / directory.name)
for suffix in ('command.json', 'launcher.log', 'host-memory.json', 'telemetry-start.json', 'telemetry-end.json'):
    shutil.copy2(ROOT / (NAME + '.' + suffix), destination / suffix)
for source, filename in ((Path(__file__), 'audit.py'), (ROOT / 'archive-local-fusion-device.py', 'archive.py'),
                         (ROOT / 'two-die-idle-gate.py', 'idle-gate.py'),
                         (ROOT / 'two-die-admission.jsonl', 'admission.jsonl'),
                         (ROOT / ('run-' + NAME.removeprefix('vendor-hook-') + '.py'), 'runner.py')):
    shutil.copy2(source, destination / filename)
(destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
