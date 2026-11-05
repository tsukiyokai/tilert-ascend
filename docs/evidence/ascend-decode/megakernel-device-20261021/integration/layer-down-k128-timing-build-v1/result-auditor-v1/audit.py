"""Recompute the one-variable Down geometry comparison from device reports."""

import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
REPO = Path('/home/shanshan/repo/tile-ai/TileRT-Ascend')
NAME = 'layer-down-k128-timing-v1'
RAW = ROOT / (NAME + '-raw')
REPORT = RAW / NAME
FROZEN = Path('/tmp/tilert-layer-down-k128-timing-staging-v1')
TILINGS = {
    'control': 'bb8ef4481b8c63b00dc43940975ee5b172818bcd228e8bc553b22234cbb2a244',
    'candidate': 'c2b73af9d605ad8e64ea551f288d188a0cf636828a5a0471ea7c63d616f50999',
}
COUNTS = dict.fromkeys(('eager_raw', 'native_component_raw', 'control_owner_raw',
                       'captured_raw', 'post_timing_raw', 'event_tail_raw',
                       'down_tiling_200_byte', 'complete_packet_comparisons'), 0)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(checks, count, category):
    assert len(checks) == count, (category, len(checks))
    for name, value in checks.items():
        assert value['pass'] and value['bitwise_equal'], name
        assert value['bitwise_different_bytes'] == 0, name
    COUNTS[category] += count


def tiling(value, arm, address):
    assert value['pass'] and value['bytes'] == 200
    assert value['sha256'] == TILINGS[arm] and value['address'] == address
    COUNTS['down_tiling_200_byte'] += 1


def packet(value, reference=None):
    if reference is not None:
        assert value == reference, 'A complete record or tiling owner changed'
    record = value['records']
    assert record['only_different_field'] == 'body.scratch[7]' and record['byte_offset'] == 624
    assert record['config_bytes_equal'] and record['all_other_address_bytes_equal']
    assert len(record['config']) == 98
    first, second = (record['addresses'][arm] for arm in ('control', 'candidate'))
    assert len(first) == len(second) == 116 and first.keys() == second.keys()
    assert [name for name in first if first[name] != second[name]] == ['body.scratch[7]']
    assert len(value['tilings']) == 2
    for arm, row in zip(('control', 'candidate'), value['tilings']):
        assert row['arm'] == arm
        tiling(row, arm, record['addresses'][arm]['body.scratch[7]'])
    COUNTS['complete_packet_comparisons'] += 1


def packets(values, references):
    assert len(values) == len(references) == 2
    for value, reference in zip(values, references):
        packet(value, reference)


def changes(rows, field):
    values = [row[field] for row in rows]
    assert len({value['sha256'] for value in values}) == 8
    assert all(value['finite'] and value['nonzero_elements'] > 0 for value in values)


assert digest(FROZEN / 'manifest.json') == '006088bf725f910bb2578eaed567bcfe4e396a1e349444c3792ee34906972fa8'
manifest = json.loads((FROZEN / 'manifest.json').read_text())
for line in (FROZEN / 'frozen-inputs.sha256').read_text().splitlines():
    sha, name = line.split('  ', 1)
    assert digest(FROZEN / name) == sha, name
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
    '609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f',
    '8deff750464e23bb871302c8626b86d37b60c28a00991d550f95dfcfd47ecb6f',
    '64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c',
    '0842cd0f3a946c39aefb3e926e01eb870a61d6d1ad8689aebcf52a1b3daa2b65',
}
results = []
for arena in (0, 1):
    prepared = report['prepared'][arena]
    assert prepared['arena'] == arena and prepared['rows'] == 12
    first, second = prepared['owners']
    assert first['attention_mode'] == 'int8' and second['attention_mode'] == 'shared'
    assert first['bridge'][10] == second['norm'][0]
    assert first['bridge'][1] == second['norm'][1]
    assert first['bridge'][10][0] != first['bridge'][1][0]
    assert first['sync'] and first['sync'] == second['sync']
    addresses = [owner['body'][7][0] for owner in prepared['owners']]
    for row, address in zip(prepared['down_tiling'], addresses):
        tiling(row, 'candidate', address)
    records = [row for row in report['down_tiling_checks'] if row['arena'] == arena]
    assert [(row['iteration'], row['arm']) for row in records] == [
        (iteration, arm) for iteration in range(8) for arm in ('separate', 'combined')]
    for row in records:
        for value, address in zip(row['layers'], addresses):
            tiling(value, 'candidate', address)
    cases = [row for row in report['cases'] if row['arena'] == arena]
    assert [row['iteration'] for row in cases] == list(range(8))
    for row in cases:
        assert row['pass']
        exact(row['checks'], 109, 'eager_raw')
        assert all(value['finite'] for value in row['candidate_diagnostics'].values())
    for name in ('consumer.output', 'consumer.retained'):
        changes([row['outputs'] for row in cases], name)
    timing = report['timings'][arena]
    assert timing == json.loads((REPORT / f'layer-pair-timing-arena{arena}.json').read_text())
    assert timing['pass'] and timing['arena'] == arena and timing['rows'] == 12
    assert timing['blocks'] == 24 and timing['scratch_owners'] == prepared['owners']
    assert not timing['same_host_record_bytes'] and timing['comparison_mode'] == 'down-tiling'
    assert timing['weight_banks'] == 1 and timing['weight_sets'] == 2
    assert (timing['graph_pair_calls'], timing['graph_replays_per_event'],
            timing['samples_per_arm'], timing['warm_replays_before_event']) == (16, 30, 30, 3)
    assert not timing['profiler_enabled'] and timing['copies_and_poison_outside_events']
    assert timing['observation_copies_inside_graph'] == 0
    assert timing['topk_copy_calls_per_pair'] == {'control': 1, 'candidate': 1}
    assert timing['norm_owner_copies_per_layer'] == {'control': 0, 'candidate': 0}
    assert timing['script_sha256'] == manifest['files_sha256']['layer_pair_timing.py']
    reference = timing['prepared_records']
    packets(reference, reference)
    for value, address in zip(reference, addresses):
        assert value['records']['addresses']['candidate']['body.scratch[7]'] == address
    assert [row['iteration'] for row in timing['native_component_checks']] == list(range(8))
    for row, original in zip(timing['native_component_checks'], cases):
        exact(row['checks'], 109, 'native_component_raw')
        packets(row['down_tiling'], reference)
        assert row['inputs'] == original['inputs']
    assert [(row['iteration'], row['layer']) for row in timing['control_owner_checks']] == [
        (iteration, layer) for iteration in range(8) for layer in (6, 7)]
    for row in timing['control_owner_checks']:
        exact(row['checks'], 31, 'control_owner_raw')
        packet(row['records'], reference[row['layer'] - 6])
    for arm in ('control', 'candidate'):
        packets(timing['capture_records'][arm], reference)
    assert [row['iteration'] for row in timing['checks']] == list(range(8, 16))
    for row in timing['checks']:
        exact(row['checks'], 109, 'captured_raw')
        for arm in ('control', 'candidate'):
            packets(row['down_tiling'][arm], reference)
    changes(timing['checks'], 'output')
    changes(timing['checks'], 'retained')
    for sequence in (cases, timing['checks']):
        for name in ('hidden', 'slot_i64:0', 'slot_i64:2'):
            assert len({row['inputs']['tensors'][name]['sha256'] for row in sequence}) == 8
        assert len({json.dumps(row['inputs']['actual_main_lengths']) for row in sequence}) == 8
    assert [row['sample'] for row in timing['samples']] == list(range(30))
    arms, savings = {'control': [], 'candidate': []}, []
    for sample, row in enumerate(timing['samples']):
        assert row['order'] == (['candidate', 'control'] if sample % 2 == 0 else ['control', 'candidate'])
        for arm in arms:
            value = row['event_total_us'][arm] / (30 * 16)
            assert math.isfinite(value) and value > 0 and row['per_pair_us'][arm] == value
            exact(row['tail_checks'][arm], 5, 'event_tail_raw')
            for layer in (6, 7):
                assert row['tail_checks'][arm][f'layer{layer}.active_list']['elements'] == 264
            packets(row['down_tiling'][arm], reference)
            arms[arm].append(value)
        delta = arms['control'][-1] - arms['candidate'][-1]
        assert row['paired_saving_us'] == delta
        savings.append(delta)
    assert [row['arm'] for row in timing['post_timing_checks']] == ['control', 'candidate']
    for row in timing['post_timing_checks']:
        exact(row['checks'], 109, 'post_timing_raw')
        packets(row['records'], reference)
    for arm, values in arms.items():
        assert statistics.median(values) == timing['summary'][arm]['p50']
        assert statistics.pstdev(values) == timing['summary'][arm]['std']
    assert statistics.median(savings) == timing['paired_saving_us']['p50']
    assert sum(value > 0 for value in savings) == timing['candidate_wins']
    assert [sum(values) for values in timing['route_counts']] == [96, 96]
    results.append({'arena': arena, 'control_p50_us': statistics.median(arms['control']),
                    'candidate_p50_us': statistics.median(arms['candidate']),
                    'paired_saving_p50_us': statistics.median(savings),
                    'candidate_wins': timing['candidate_wins'], 'pairs': 30,
                    'resident_weight_bytes': timing['resident_weight_bytes'],
                    'active_experts': [sum(value > 0 for value in counts) for counts in timing['route_counts']],
                    'active_routed_matrix_bytes': timing['active_routed_matrix_bytes']})
assert COUNTS == {'eager_raw': 1744, 'native_component_raw': 1744, 'control_owner_raw': 992,
                  'captured_raw': 1744, 'post_timing_raw': 436, 'event_tail_raw': 600,
                  'down_tiling_200_byte': 844, 'complete_packet_comparisons': 388}, COUNTS
for rank in range(16):
    row = json.loads((REPORT / f'rank-{rank:02d}.json').read_text())
    assert row['torch'] == '2.12.0+cu130' and row['torch_npu'] == '2.12.0'
    assert not row['npu_initialized_before'] and row['npu_initialized_after'] == (rank == 0)
    assert row['rank_zero_status']['success'] and all(value['success'] for value in row['stages'])
nz = json.loads((REPORT / 'nz-storage-correction.json').read_text())
assert nz['completed'] and nz['additional_hbm_bytes'] == 0 and len(nz['banks']) == 4
for bank in nz['banks']:
    assert bank['raw_bytes_equal'] and bank['raw_sha256_before'] == bank['raw_sha256_after']
    assert bank['new_device_allocation_bytes'] == 0 and bank['before']['data_ptr'] == bank['after']['data_ptr']
summary = {
    'scope': 'Same 609 local-layer launch and Top-K copy; only N512/K128 versus N1024/K128 Down tiling differs',
    'host': 'a3-194', 'physical_die': 4, 'rows': 12, 'blocks': 24,
    'unit': 'us per indexed/shared two-layer pair', 'numerical_pass': True,
    'candidate_faster_in_both_arenas': all(row['paired_saving_p50_us'] > 0 for row in results),
    'only_different_record_field': 'body.scratch[7]', 'down_tiling_sha256': TILINGS,
    'layer_library_sha256': '609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f',
    'raw_checks': COUNTS, 'arenas': results, 'weight_banks': 1,
    'tp_contract': 'Both reductions are identity', 'checkpoint_loaded': False,
    'report_sha256': digest(REPORT / 'local-chain.json'),
    'fixture_manifest_sha256': digest(FROZEN / 'manifest.json'),
}
destination = REPO / 'docs/evidence/ascend-decode/megakernel-device-20261021/integration' / (NAME + '-a3-194')
destination.mkdir(exist_ok=False)
for directory in RAW.iterdir():
    shutil.copytree(directory, destination / directory.name)
for suffix in ('command.json', 'launcher.log', 'host-memory.json', 'telemetry-start.json', 'telemetry-end.json', 'admission.jsonl'):
    shutil.copy2(ROOT / (NAME + '.' + suffix), destination / suffix)
for source, filename in ((Path(__file__), 'audit.py'),
                         (ROOT / 'archive-local-fusion-device.py', 'archive.py'),
                         (ROOT / 'run-layer-down-k128-timing-v1.py', 'runner.py')):
    shutil.copy2(source, destination / filename)
(destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
