"""Recompute the two-layer paired graph result from the recorded device reports."""

import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
REPO = Path('/home/shanshan/repo/tile-ai/TileRT-Ascend')
NAME = 'vendor-hook-layer-native-timing-v3'
RAW = ROOT / (NAME + '-raw')
REPORT = RAW / NAME
FROZEN = Path('/tmp/tilert-layer-native-timing-staging-v3')
TILING_SHA = 'bb8ef4481b8c63b00dc43940975ee5b172818bcd228e8bc553b22234cbb2a244'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(checks, count):
    assert len(checks) == count
    for key, value in checks.items():
        assert value['pass'] and value['bitwise_equal'], key
        assert value['bitwise_different_bytes'] == 0, key


def tilings(records, addresses):
    assert len(records) == 2
    assert [value['address'] for value in records] == addresses
    for value in records:
        assert value['pass'] and value['bytes'] == 200
        assert value['sha256'] == TILING_SHA
    return len(records)


def changes(records, output_key):
    assert len({row[output_key]['sha256'] for row in records}) == 8
    for row in records:
        assert row[output_key]['nonzero_elements'] > 0 and row[output_key]['finite']


manifest = json.loads((FROZEN / 'manifest.json').read_text())
assert digest(FROZEN / 'manifest.json') == 'a69b6cae51cb0435b5756599772bc98d22546c9e2531a3d435dd1f4796ca990f'
for name, expected in manifest['files_sha256'].items():
    assert digest(FROZEN / name) == expected, name
report = json.loads((REPORT / 'local-chain.json').read_text())
driver = json.loads((REPORT / 'summary.json').read_text())
assert driver['success'] and driver['exitcodes'] == [0] * 16
assert driver['arguments']['forward_widths'] == [12]
assert report['pass'] and report['widths'] == [12] and report['mode'] == 'int8'
assert report['blocks'] == 24 and report['iterations'] == 8
assert report['tp_geometry'] == 16 and report['tp_collectives_executed'] == 0
assert report['same_addresses'] and report['fresh_output_poison'] and report['timing']
assert len(report['cases']) == 16 and len(report['prepared']) == 2
assert len(report['timings']) == 2 and len(report['down_tiling_checks']) == 32
for key, expected in {
    'script': manifest['files_sha256']['local_chain.py'],
    'decoder_class': '96f4f755f62caa85869fb884209e7f8869882ddb6efbd40c930778b27bd3e39d',
    'input_norm': '75491119b6b494d6d399a552bf005d57db40bfe2a11d5ae13259c68f824e02b4',
    'attention_packing': '336eaff0de153d32439d4d2b0557522b313a24a7e5bef82b8dc11f1257511c01',
    'ffn_packing': manifest['files_sha256']['ffn_host_adapter.py'],
}.items():
    assert report[key]['sha256'] == expected, key
assert set(report['libraries'].values()) == {
    '6a326292287a17254b48376400e1b852093ebfe10fbf228938336693402aa528',
    '8deff750464e23bb871302c8626b86d37b60c28a00991d550f95dfcfd47ecb6f',
    '64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c',
    'c90b3666a1697070eda19d5b3e133e9ed58f36fe429abefceae6ffe102a6a032',
}
counts = {'eager_raw': 0, 'native_component_raw': 0, 'captured_raw': 0, 'event_tail_raw': 0, 'down_tiling_200_byte': 0}
results = []
for arena in (0, 1):
    prepared = report['prepared'][arena]
    assert prepared['arena'] == arena and prepared['rows'] == 12
    first, second = prepared['owners']
    assert first['attention_mode'] == 'int8' and second['attention_mode'] == 'shared'
    assert first['bridge'][10] == second['norm'][0]
    assert first['bridge'][1] == second['norm'][1]
    assert first['bridge'][10][0] != first['bridge'][1][0]
    assert first['norm'][6][1] == [12, 6144] and first['norm'][6][3] == 'torch.int8'
    assert first['sync'] and first['sync'] == second['sync']
    assert first['persistent_topk'] == second['persistent_topk']
    assert first['persistent_topk'][1] == [12, 1, 2048]
    assert first['persistent_topk'][3] == 'torch.int32'
    assert first['persistent_topk'][5] == 0 and first['persistent_topk'][0] % 32 == 0
    assert first['internal_topk'][0] != first['persistent_topk'][0]
    assert second['internal_topk'] is None
    addresses = [owner['body'][7][0] for owner in prepared['owners']]
    counts['down_tiling_200_byte'] += tilings(prepared['down_tiling'], addresses)
    records = [row for row in report['down_tiling_checks'] if row['arena'] == arena]
    assert [(row['iteration'], row['arm']) for row in records] == [
        (iteration, arm) for iteration in range(8) for arm in ('separate', 'combined')]
    for row in records:
        counts['down_tiling_200_byte'] += tilings(row['layers'], addresses)
    cases = [row for row in report['cases'] if row['arena'] == arena]
    assert [row['iteration'] for row in cases] == list(range(8))
    for row in cases:
        assert row['pass'] and row['unused_internal_topk_unchanged']
        assert row['inactive_persistent_rows_unchanged'] and row['candidate_topk_copy_calls'] == 0
        exact(row['checks'], 109)
        counts['eager_raw'] += len(row['checks'])
        assert all(value['finite'] for value in row['candidate_diagnostics'].values())
    for key in ('consumer.output', 'consumer.retained'):
        changes([row['outputs'] for row in cases], key)
    timing = report['timings'][arena]
    assert timing == json.loads((REPORT / f'layer-pair-timing-arena{arena}.json').read_text())
    assert timing['arena'] == arena and timing['rows'] == 12 and timing['blocks'] == 24
    assert timing['pass'] and timing['same_host_record_bytes']
    assert timing['scratch_owners'] == prepared['owners']
    assert timing['weight_banks'] == 1 and timing['weight_sets'] == 2
    assert timing['unit'] == 'us per indexed/shared two-layer pair'
    assert timing['graph_pair_calls'] == 16 and timing['graph_replays_per_event'] == 30
    assert timing['samples_per_arm'] == 30 and timing['warm_replays_before_event'] == 3
    assert not timing['profiler_enabled'] and timing['copies_and_poison_outside_events']
    assert timing['observation_copies_inside_graph'] == 0
    assert timing['topk_copy_calls_per_pair'] == {'control': 1, 'candidate': 1}
    assert timing['norm_owner_copies_per_layer'] == {'control': 0, 'candidate': 0}
    assert timing['comparison_mode'] == 'native-stages'
    assert [row['iteration'] for row in timing['native_component_checks']] == list(range(8))
    for row, original in zip(timing['native_component_checks'], cases):
        exact(row['checks'], 109)
        counts['native_component_raw'] += len(row['checks'])
        counts['down_tiling_200_byte'] += tilings(row['down_tiling'], addresses)
        assert row['inputs'] == original['inputs']
    assert timing['script_sha256'] == manifest['files_sha256']['layer_pair_timing.py']
    assert [row['iteration'] for row in timing['checks']] == list(range(8, 16))
    for row in timing['checks']:
        exact(row['checks'], 109)
        counts['captured_raw'] += len(row['checks'])
        for arm in ('candidate', 'control'):
            counts['down_tiling_200_byte'] += tilings(row['down_tiling'][arm], addresses)
    changes(timing['checks'], 'output')
    changes(timing['checks'], 'retained')
    for sequence in (cases, timing['checks']):
        for field in ('hidden', 'slot_i64:0', 'slot_i64:2'):
            assert len({row['inputs']['tensors'][field]['sha256'] for row in sequence}) == 8
        assert len({json.dumps(row['inputs']['actual_main_lengths']) for row in sequence}) == 8
    assert [row['sample'] for row in timing['samples']] == list(range(30))
    arm_values = {'control': [], 'candidate': []}
    savings = []
    for sample, row in enumerate(timing['samples']):
        assert row['order'] == (['candidate', 'control'] if sample % 2 == 0 else ['control', 'candidate'])
        for arm in arm_values:
            value = row['per_pair_us'][arm]
            assert math.isfinite(value) and value > 0
            assert value == row['event_total_us'][arm] / (30 * 16)
            exact(row['tail_checks'][arm], 5)
            assert row['tail_checks'][arm]['layer6.active_list']['elements'] == 264
            assert row['tail_checks'][arm]['layer7.active_list']['elements'] == 264
            counts['event_tail_raw'] += 5
            counts['down_tiling_200_byte'] += tilings(row['down_tiling'][arm], addresses)
            arm_values[arm].append(value)
        delta = row['per_pair_us']['control'] - row['per_pair_us']['candidate']
        assert delta == row['paired_saving_us']
        savings.append(delta)
    for arm, values in arm_values.items():
        assert statistics.median(values) == timing['summary'][arm]['p50']
        assert statistics.pstdev(values) == timing['summary'][arm]['std']
    assert statistics.median(savings) == timing['paired_saving_us']['p50']
    assert sum(value > 0 for value in savings) == timing['candidate_wins']
    assert [sum(values) for values in timing['route_counts']] == [96, 96]
    assert [sum(value > 0 for value in values) for values in timing['route_counts']] == [85, 23]
    results.append({'arena': arena, 'control_p50_us': statistics.median(arm_values['control']),
                    'candidate_p50_us': statistics.median(arm_values['candidate']),
                    'paired_saving_p50_us': statistics.median(savings),
                    'candidate_wins': timing['candidate_wins'], 'pairs': 30,
                    'resident_weight_bytes': timing['resident_weight_bytes'],
                    'active_routed_matrix_bytes': timing['active_routed_matrix_bytes']})
assert counts == {'eager_raw': 1744, 'native_component_raw': 1744, 'captured_raw': 1744,
                  'event_tail_raw': 600, 'down_tiling_200_byte': 404}
for rank in range(16):
    row = json.loads((REPORT / f'rank-{rank:02d}.json').read_text())
    assert row['torch'] == '2.12.0+cu130' and row['torch_npu'] == '2.12.0'
    assert not row['npu_initialized_before'] and row['npu_initialized_after'] == (rank == 0)
    assert row['rank_zero_status']['success'] and all(value['success'] for value in row['stages'])
nz = json.loads((REPORT / 'nz-storage-correction.json').read_text())
assert nz['completed'] and nz['additional_hbm_bytes'] == 0 and len(nz['banks']) == 4
for bank in nz['banks']:
    assert bank['raw_bytes_equal'] and bank['raw_sha256_before'] == bank['raw_sha256_after']
    assert bank['new_device_allocation_bytes'] == 0
    assert bank['before']['data_ptr'] == bank['after']['data_ptr']
summary = {
    'scope': 'Two N512 local layers, same owners: three native norm/attention/FFN launches versus one 609 launch per layer',
    'host': 'a3-194', 'physical_die': 4, 'rows': 12, 'blocks': 24,
    'unit': 'us per indexed/shared two-layer pair', 'numerical_pass': True,
    'whole_layer_faster_in_both_arenas': all(row['paired_saving_p50_us'] > 0 for row in results),
    'norm_output_owner_copies_inside_timing': 0, 'topk_copies_per_pair_in_both_arms': 1,
    'candidate_library_sha256': manifest['control_library_sha256'],
    'norm_library_sha256': manifest['norm_library_sha256'],
    'attention_library_sha256': {key: value for key, value in report['libraries'].items() if 'attention' in key},
    'ffn_library_sha256': report['libraries']['/decoder/complete_ffn.so'],
    'raw_checks': counts, 'arenas': results, 'weight_banks': 1,
    'shared_between_arenas': 'Weights, persistent Top-K owner and caches',
    'tp_contract': 'Both reductions are identity', 'checkpoint_loaded': False,
    'report_sha256': digest(REPORT / 'local-chain.json'),
    'fixture_manifest_sha256': digest(FROZEN / 'manifest.json'),
}
destination = REPO / 'docs/evidence/ascend-decode/megakernel-device-20261021/integration' / (NAME + '-a3-194')
destination.mkdir(exist_ok=False)
for directory in RAW.iterdir():
    shutil.copytree(directory, destination / directory.name)
for suffix in ('command.json', 'launcher.log', 'host-memory.json', 'telemetry-start.json', 'telemetry-end.json'):
    shutil.copy2(ROOT / (NAME + '.' + suffix), destination / suffix)
for source, filename in ((Path(__file__), 'audit.py'),
                         (ROOT / 'archive-local-fusion-device.py', 'archive.py'),
                         (ROOT / 'run-layer-native-timing-v3.py', 'runner.py')):
    shutil.copy2(source, destination / filename)
(destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
