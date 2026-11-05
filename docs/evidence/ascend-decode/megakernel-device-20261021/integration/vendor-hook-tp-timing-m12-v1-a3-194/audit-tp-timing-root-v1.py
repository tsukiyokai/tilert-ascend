"""Independently reconstruct paired rank maxima and every reported byte metric."""

import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
REPORT = ROOT / 'vendor-hook-tp-timing-m12-v1-raw/vendor-hook-tp-timing-m12-v1'


def exact(checks, count):
    assert len(checks) == count
    for name, value in checks.items():
        assert value['pass'] and value['bitwise_equal'], name
        assert value['bitwise_different_bytes'] == 0, name


driver = json.loads((REPORT / 'summary.json').read_text())
assert driver['success'] and driver['exitcodes'] == [0] * 16
arenas, reports = {}, []
counts = {'production_byte_checks': 0, 'literal_byte_checks': 0,
          'event_tail_byte_checks': 0, 'capture_unchanged_boundaries': 0}
for rank in (0, 1):
    parent_path = REPORT / f'device-rank{rank}/tp-chain-timing.json'
    parent = json.loads(parent_path.read_text())
    assert parent['pass'] and parent['rank'] == rank
    assert parent['ranks'] == 2 and parent['constructor_tp_size'] == 16
    assert parent['tp16_collectives_executed'] == 0
    assert not set(parent['private_arena_allocations'][0]) & set(parent['private_arena_allocations'][1])
    reports.append(hashlib.sha256(parent_path.read_bytes()).hexdigest())
    for arena in (0, 1):
        path = REPORT / f'device-rank{rank}/tp-pair-timing-arena{arena}.json'
        report = json.loads(path.read_text())
        arenas[rank, arena] = report
        assert report['pass'] and report['rows'] == 12 and report['blocks'] == 24
        assert report['same_production_addresses'] and len(report['production_owners']) == 113
        assert not report['norm_owner_copies'] and not report['observation_copies_inside_graph']
        assert report['production_topk_copies_per_pair'] == 1
        assert not report['window_epoch_reset']
        assert len(set(report['collective_temporary_addresses'].values())) == 4
        assert report['collective_temporary_bytes'] == 4 * 12 * 6144 * 4
        assert len(report['captures']) == 4
        for capture in report['captures']:
            assert capture['runnable_calls'] == capture['capture_counter_delta'] == 1
            assert capture['unchanged_capture_boundaries'] == 113
            assert capture['capture_stream'] == capture['window_stream'] == report['window_stream']
            counts['capture_unchanged_boundaries'] += 113
        exact(report['native_control_check'], 113)
        counts['production_byte_checks'] += 113
        literal = [*report['literal_conversion_checks'], *report['timed_literal_conversions']]
        assert [row['iteration'] for row in report['checks']] == list(range(8))
        for case in report['checks']:
            literal.extend(case['literal_conversions'])
            assert len(case['checks']) == 4
            for observed in case['checks']:
                exact(observed['checks'], 113)
                counts['production_byte_checks'] += 113
        assert len(report['idempotence_checks']) == 6 and len(report['postchecks']) == 2
        for case in [*report['idempotence_checks'], *report['postchecks']]:
            exact(case['checks'], 113)
            counts['production_byte_checks'] += 113
        assert len(literal) == 40
        for case in literal:
            exact(case['checks'], 4)
            assert case['partials'][0]['sha256'] != case['partials'][1]['sha256']
            counts['literal_byte_checks'] += 4
        for collection, calls in (('samples', 16), ('one_pair_sensitivity', 1)):
            assert len(report[collection]) == 30
            for sample, row in enumerate(report[collection]):
                assert row['sample'] == sample and row['pair_calls'] == calls
                assert row['order'] == (['candidate', 'control'] if sample % 2 == 0 else ['control', 'candidate'])
                values = {}
                for arm in ('control', 'candidate'):
                    observed = row['arms'][arm]
                    intervals = observed['rank_intervals']
                    assert [value['rank'] for value in intervals] == [0, 1]
                    assert all(value['key'] == [arena, calls, sample, arm] for value in intervals)
                    durations = [value['duration_us'] for value in intervals]
                    assert all(math.isfinite(value) and value > 0 for value in durations)
                    assert max(durations) == observed['maximum_event_us']
                    values[arm] = max(durations) / (calls * 30)
                    assert values[arm] == observed['per_pair_us']
                    exact(observed['tail_checks'], 9)
                    counts['event_tail_byte_checks'] += 9
                assert row['paired_saving_us'] == values['control'] - values['candidate']
        for graph in report['graph_execution_counts']:
            assert graph['captures'] == 1 and graph['python_replay_calls'] == 0
            assert graph['replays'] == (1002 if graph['pair_calls'] == 16 else 998)
            expected = 4 * graph['pair_calls'] * graph['replays'] if graph['arm'] == 'candidate' else 0
            assert graph['expected_candidate_epoch_increments_from_calls'] == expected
for rank in range(16):
    report = json.loads((REPORT / f'rank-{rank:02d}.json').read_text())
    assert report['success'] and not report['npu_initialized_before']
    assert report['npu_initialized_after'] == (rank < 2)
    assert all(value['success'] for value in report['stages'])
assert counts == {'production_byte_checks': 18532, 'literal_byte_checks': 640,
                  'event_tail_byte_checks': 4320, 'capture_unchanged_boundaries': 1808}, counts
result = []
for arena in (0, 1):
    row = {'arena': arena}
    for collection in ('samples', 'one_pair_sensitivity'):
        left, right = (arenas[rank, arena][collection] for rank in (0, 1))
        assert left == right
        savings = [sample['paired_saving_us'] for sample in left]
        row[collection] = {
            'control_p50_us': statistics.median(sample['arms']['control']['per_pair_us'] for sample in left),
            'candidate_p50_us': statistics.median(sample['arms']['candidate']['per_pair_us'] for sample in left),
            'paired_saving_p50_us': statistics.median(savings),
            'wins': sum(value > 0 for value in savings), 'pairs': 30,
        }
    result.append(row)
summary = {'pass': True, 'scope': 'Two TP16-shaped shards; per-arm maximum of two device event intervals',
           'report_sha256': reports, 'raw_checks': counts, 'arenas': result,
           'full_model_tpot_measured': False, 'device_epoch_counter_read': False}
(ROOT / 'tp-timing-root-independent-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
