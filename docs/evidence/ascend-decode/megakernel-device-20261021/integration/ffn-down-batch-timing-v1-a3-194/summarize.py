from pathlib import Path
import hashlib
import json
import shutil
import statistics

root = Path('/tmp/tilert-megakernel-validation-20261021')
repo = Path('/home/shanshan/repo/tile-ai/TileRT-Ascend')
name = 'ffn-down-batch-timing-v1'
out = repo / 'docs/evidence/ascend-decode/megakernel-device-20261021/integration' / (name + '-a3-194')
out.mkdir(exist_ok=True)
data = (root / (name + '.result.json')).read_bytes()
if (out / 'result.json').exists():
    assert (out / 'result.json').read_bytes() == data
else:
    (out / 'result.json').write_bytes(data)
r = json.loads(data)
assert r['pass'] and r['arm_names']['fused'] == 'down_batch'
assert {key: r['arguments'][key] for key in ('rows', 'blocks', 'banks', 'calls', 'iterations', 'samples')} == {'rows': 12, 'blocks': 24, 'banks': 16, 'calls': 16, 'iterations': 30, 'samples': 30}
assert len(r['arenas']) == 2
candidate = r['arguments']['candidate_library']
assert r['libraries'][r['arguments']['library']] == '3673c2a73b2f4b5a4a93b3c2543a4aca35b45773d6a53fc043f27e59a2482655'
assert r['libraries'][candidate] == 'c83c9cb66f9109b810023d95794a4907734e73f66b9526ffd0c3b5987e63d984'
arenas = []
for a in r['arenas']:
    assert a['pass'] and len(a['samples']) == 30
    assert len(a['prechecks']) == 256 and len(a['postchecks']) == 64
    checks = [v for c in a['prechecks'] + a['postchecks'] for v in c['checks'].values()]
    assert len(checks) == 9600 and all(v['bitwise_equal'] for v in checks)
    assert all(s['post_event_tail_byte_equal'] for s in a['samples'])
    savings = [s['per_call_us']['separate'] - s['per_call_us']['fused'] for s in a['samples']]
    for s, value in zip(a['samples'], savings):
        assert abs(s['paired_saving_us'] - value) < 1e-9
    for arm in ('fused', 'separate'):
        median = statistics.median(s['per_call_us'][arm] for s in a['samples'])
        assert abs(median - a['summary_us'][arm]['p50']) < 1e-9
    assert abs(statistics.median(savings) - a['paired_saving_us']['p50']) < 1e-9
    assert sum(value > 0 for value in savings) == a['paired_wins']
    assert a['timed_route']['active_experts'] == 65
    arenas.append({'owner': a['owner'], 'active_experts': 65,
                   'candidate_us': a['summary_us']['fused'],
                   'base_us': a['summary_us']['separate'],
                   'paired_saving_us': a['paired_saving_us'],
                   'paired_wins': a['paired_wins'], 'pairs': 30})
assert all(a['paired_wins'] == 0 for a in arenas)
summary = {'scope': r['scope'], 'candidate': 'down_batch', 'host': 'a3-194',
           'physical_die': 4, 'rows': 12, 'blocks': 24, 'weight_banks': 16,
           'calls_per_graph': 16, 'graph_replays_per_event': 30,
           'pairs_per_arena': 30, 'raw_byte_boundary_checks': 19200,
           'post_event_tail_checks': 360, 'pass': True, 'performance_selected': False,
           'scratch_matching': r['scratch_matching'],
           'fresh_write_gate': r['fresh_write_gate'],
           'extra_storage': r['extra_storage'],
           'minimum_logical_active_matrix_bytes': r['logical_minimum_active_matrix_bytes'],
           'payload_interpretation': r['logical_payload_interpretation'],
           'candidate_library': {'path': candidate, 'sha256': r['libraries'][candidate]},
           'sources': r['sources'], 'arenas': arenas,
           'report_sha256': hashlib.sha256(data).hexdigest(),
           'interpretation': 'The batched Down overlap is rejected: all 60 pairs are slower than original3673. Local synthetic FFN only, without TP collectives or model TPOT.'}
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
for source, target in ((name + '.command.json', 'command.json'),
                       (name + '.launcher.log', 'launcher.log'),
                       ('run-' + name + '.py', 'run.py'),
                       ('archive-' + name + '.py', 'summarize.py')):
    shutil.copy2(root / source, out / target)
print(json.dumps({'pass': True, 'report_sha256': summary['report_sha256'],
                  'arenas': arenas}, indent=2))
