import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
BASE = Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration')
NAME = 'ffn-gu-static212-wide-v1'
OUT = BASE / (NAME + '-a3-194')
SHA = '05b3427aa29630c9af16ea829e2ecec8833d31a32615e6a052f6f4e9867ca6b2'
CONSUMERS = {
    'consumer.bridge.combined', 'consumer.bridge.scaled_routed',
    'consumer.body.routes_bf16', 'consumer.body.routed_output',
}
reports = {
    'wide/result.json': {(m, b) for m in (1, 2, 32) for b in (16, 24)},
    'm12b16/result.json': {(12, 16)},
}
summaries = []
for name, geometries in reports.items():
    path = OUT / name
    report = json.loads(path.read_text())
    assert report['pass'] and report['oracle_mode'] == 'both'
    assert report['queued_clone_producers'] and report['replays'] == 8
    assert SHA in report['libraries'].values()
    expected = {
        f'b{b}-m{m}-r{iteration}-{oracle}'
        for m, b in geometries for iteration in range(8)
        for oracle in ('orchestration', 'production')
    }
    assert len(report['cases']) == len(expected)
    assert {case['name'] for case in report['cases']} == expected
    for case in report['cases']:
        assert case['pass'] and case['first_mismatch'] is None
        assert len(case['checks']) == 35
        assert CONSUMERS <= set(case['checks'])
        assert 'body.active_experts' in case['checks']
        for boundary, check in case['checks'].items():
            assert check['bitwise_equal'], (case['name'], boundary)
            assert check['bitwise_different_bytes'] == 0
            assert check.get('finite', True)
    summaries.append({
        'report': name,
        'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'geometries': sorted(geometries),
        'raw_byte_boundary_checks': len(expected) * 35,
        'immediate_consumer_checks': len(expected) * 4,
    })
prior = json.loads((BASE / 'ffn-gu-static212-m12-v1-a3-194/summary.json').read_text())
assert prior['pass'] and prior['library_sha256'] == SHA
assert prior['raw_byte_boundary_checks'] == 560
summary = {
    'pass': True, 'library_sha256': SHA,
    'rows': [1, 2, 12, 32], 'blocks': [16, 24],
    'iterations_per_geometry': 8, 'oracle_mode': 'both',
    'new_raw_byte_boundary_checks': sum(s['raw_byte_boundary_checks'] for s in summaries),
    'new_immediate_consumer_checks': sum(s['immediate_consumer_checks'] for s in summaries),
    'all_eight_geometry_raw_byte_checks': 4480,
    'all_eight_geometry_immediate_consumer_checks': 512,
    'prior_m12_b24_summary': '../ffn-gu-static212-m12-v1-a3-194/summary.json',
    'reports': summaries,
    'scope': report['scope'],
}
assert summary['new_raw_byte_boundary_checks'] + prior['raw_byte_boundary_checks'] == 4480
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
for source, dest in ((NAME + '.command.json', 'command.json'),
                     (NAME + '.launcher.log', 'launcher.log'),
                     ('run-' + NAME + '.py', 'run.py')):
    shutil.copy2(ROOT / source, OUT / dest)
shutil.copy2(__file__, OUT / 'audit.py')
print(json.dumps(summary, indent=2))
