"""Audit every numerical check and independently recompute paired timings."""

import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


output = Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/shared-attention-terminal212-timing-v1-a3-194')
path = output / 'result.json'
report = json.loads(path.read_text())
assert report['pass']
assert report['script_sha256'] == 'c79a60350411ae22030159f263e2549fac8ec5ee9c698d1a9ef44195f5890859'
assert report['library_sha256'] == {
    'control': '64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c',
    'candidate': '2a2deacf850ce108825002559a1a44e08d6176fdc1c9a092955acc6840eb1ea4',
}
assert report['calls_per_graph'] == 16 and report['graph_replays_per_event'] == 30
assert report['logical_weight_bytes'] > 192 * 1024 ** 2
assert {owner['owner'] for owner in report['owners']} == {'control', 'candidate'}
boundaries = {'output', 'latent', 'kv_cache', 'rope_cache'}


def audit_checks(checks):
    assert set(checks) == boundaries
    for check in checks.values():
        assert check['pass'] and check['bitwise_equal']
        assert check['bitwise_different_bytes'] == 0 and check['different_elements'] == 0
    return len(checks)


raw_checks = 0
tail_checks = 0
owners = []
for owner in report['owners']:
    assert owner['pass'] and owner['same_arguments_for_both_arms']
    assert owner['python_calls_after_capture_and_replays'] == {'control': 4, 'candidate': 4}
    assert len(owner['argument_descriptors']) == 16
    addresses = []
    for values in owner['argument_descriptors']:
        assert len(values) == 32
        assert values[0]['shape'] == [12, 6144] and values[30] == 24
        addresses.extend(values[index]['address'] for index in (*range(2, 11), 20, 21, 22))
    assert len(set(addresses)) == 16 * 12
    for key, inputs in (('prechecks', range(8)), ('postchecks', [3])):
        assert {(case['input'], case['bank']) for case in owner[key]} == {
            (replay, bank) for replay in inputs for bank in range(16)}
        assert len(owner[key]) == len(inputs) * 16
        raw_checks += sum(audit_checks(case['checks']) for case in owner[key])
    assert len(owner['samples']) == 30
    savings = []
    for pair, sample in enumerate(owner['samples']):
        assert sample['pair'] == pair
        assert sample['order'] == (['control', 'candidate'] if pair % 2 == 0
                                   else ['candidate', 'control'])
        times = sample['per_call_us']
        assert set(times) == {'control', 'candidate'}
        assert all(math.isfinite(value) and value > 0 for value in times.values())
        saving = times['control'] - times['candidate']
        assert math.isclose(saving, sample['paired_saving_us'], abs_tol=1e-12)
        savings.append(saving)
        assert set(sample['tail_checks']) == {'control', 'candidate'}
        tail_checks += sum(audit_checks(checks) for checks in sample['tail_checks'].values())
    computed = {
        'control_median_us': statistics.median(sample['per_call_us']['control']
                                               for sample in owner['samples']),
        'candidate_median_us': statistics.median(sample['per_call_us']['candidate']
                                                 for sample in owner['samples']),
        'paired_median_saving_us': statistics.median(savings),
        'paired_minimum_saving_us': min(savings),
        'paired_maximum_saving_us': max(savings),
        'paired_wins': sum(value > 0 for value in savings),
    }
    assert computed == owner['summary']
    owners.append({'owner': owner['owner'], **computed})
summary = {
    'pass': True, 'numerical_pass': True, 'performance_selected': True,
    'scope': report['scope'], 'owners': owners,
    'per_bank_raw_byte_checks': raw_checks, 'event_tail_raw_byte_checks': tail_checks,
    'logical_weight_bytes': report['logical_weight_bytes'],
    'graph_captures': 4, 'timed_graph_replays': 2 * 30 * 2 * 30,
    'calls_per_graph': 16, 'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'decision': 'Terminal completion improves paired medians in both arenas, with 52 of 60 pairs faster; eligible for this local synthetic geometry. Actual engine integration is not measured.',
}
assert raw_checks == 1152 and tail_checks == 480
assert all(owner['paired_wins'] == 26 and owner['paired_median_saving_us'] > 0 for owner in owners)
(output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
shutil.copy2(__file__, output / 'audit.py')
print(json.dumps(summary, indent=2))
