from pathlib import Path
import hashlib
import json
import shutil
import statistics

root = Path('/tmp/tilert-megakernel-validation-20261021')
repo = Path('/home/shanshan/repo/tile-ai/TileRT-Ascend')
name = 'ffn-norm-join-timing-v1'
out = repo / 'docs/evidence/ascend-decode/megakernel-device-20261021/integration' / (name + '-a3-194')
out.mkdir(exist_ok=True)
data = (root / (name + '.result.json')).read_bytes()
if (out / 'result.json').exists():
    assert (out / 'result.json').read_bytes() == data
else:
    (out / 'result.json').write_bytes(data)
r = json.loads(data)
assert r['pass'] and r['arm_names']['fused'] == 'norm_join'
assert {key: r['arguments'][key] for key in ('rows', 'blocks', 'banks', 'calls', 'iterations', 'samples')} == {'rows': 12, 'blocks': 24, 'banks': 16, 'calls': 16, 'iterations': 30, 'samples': 30}
assert len(r['arenas']) == 2
candidate = r['arguments']['candidate_library']
assert r['libraries'][r['arguments']['active_library']] == '381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2'
assert r['arm_names']['separate'] == 'original_shared_gu_static_parent'
assert r['libraries'][candidate] == 'ddd2f7b94bef711b09b78a90315d6ba8a0f77fb349ccb56d5859255d11782303'
arenas = []
for a in r['arenas']:
    assert a['pass'] and len(a['samples']) == 30
    assert len(a['prechecks']) == 256 and len(a['postchecks']) == 64
    checks = [v for c in a['prechecks'] + a['postchecks'] for v in c['checks'].values()]
    assert len(checks) == 9600 and all(v['bitwise_equal'] for v in checks)
    assert all(s['post_event_tail_byte_equal'] for s in a['samples'])
    assert len(a['extra_checks']) == 348
    assert all(c['checks']['bitwise_equal'] for c in a['extra_checks'])
    assert {arm: sum(c['arm'] == arm for c in a['extra_checks']) for arm in ('fused','separate')} == {'fused': 174, 'separate': 174}
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
                   'parent_us': a['summary_us']['separate'],
                   'paired_saving_us': a['paired_saving_us'],
                   'paired_wins': a['paired_wins'], 'pairs': 30})
summary = {'scope': r['scope'], 'candidate': 'norm_join', 'host': 'a3-194',
           'physical_die': 4, 'rows': 12, 'blocks': 24, 'weight_banks': 16,
           'calls_per_graph': 16, 'graph_replays_per_event': 30,
           'pairs_per_arena': 30, 'raw_byte_boundary_checks': 19200,
           'post_event_tail_checks': 360, 'pass': True,
           'performance_pass': all(a['paired_wins'] == 30 for a in arenas),
           'promotion': 'not selected; both arenas regress against the static-GU parent',
           'scratch_matching': r['scratch_matching'],
           'fresh_write_gate': r['fresh_write_gate'],
           'active_table_checks': 696,
           'active_table_gate': 'Both arms replace all 264 INT32 words from identical poison',
           'minimum_logical_active_matrix_bytes': r['logical_minimum_active_matrix_bytes'],
           'payload_interpretation': r['logical_payload_interpretation'],
           'candidate_library': {'path': candidate, 'sha256': r['libraries'][candidate]},
           'sources': r['sources'], 'arenas': arenas,
           'report_sha256': hashlib.sha256(data).hexdigest(),
           'interpretation': 'Positive paired saving means private norm final-join removal is faster than the original-shared static-GU parent with the same corrected activation arithmetic. Local M12/24 E65 FFN only, with synthetic weights and no TP collectives or model TPOT.'}
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
for source, target in ((name + '.command.json', 'command.json'),
                       (name + '.launcher.log', 'launcher.log'),
                       ('run-' + name + '.py', 'run.py'),
                       ('archive-' + name + '.py', 'summarize.py')):
    shutil.copy2(root / source, out / target)
print(json.dumps({'pass': True, 'report_sha256': summary['report_sha256'],
                  'arenas': arenas}, indent=2))
