from pathlib import Path
import argparse
import hashlib
import json
import shutil
import statistics
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('kind', choices=('profile', 'active'))
args = parser.parse_args()
repo = Path('/home/shanshan/repo/tile-ai/TileRT-Ascend')
root = Path('/tmp/tilert-megakernel-validation-20261021')
name = 'ffn-' + args.kind + '-timing-v3'
out = repo / 'docs/evidence/ascend-decode/megakernel-device-20261021/integration' / (name + '-a3-194')
out.mkdir(exist_ok=True)
ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes', 'a3-194']
data = subprocess.check_output(ssh + ['cat', '/home/shanshan/tilert-megakernel-20261021/logs/' + name + '/result.json'])
if (out / 'result.json').exists():
    assert (out / 'result.json').read_bytes() == data
else:
    (out / 'result.json').write_bytes(data)
r = json.loads(data)
assert r['pass'] and r['arm_names']['fused'] == args.kind
assert r['arguments']['rows'] == 12 and r['arguments']['blocks'] == 24
assert r['arguments']['banks'] == 16 and r['arguments']['calls'] == 16
assert r['arguments']['iterations'] == 30
assert len(r['arenas']) == 2
arenas = []
for a in r['arenas']:
    assert a['pass'] and len(a['samples']) == 30
    assert len(a['prechecks']) == 256 and len(a['postchecks']) == 64
    checks = [v for c in a['prechecks'] + a['postchecks'] for v in c['checks'].values()]
    assert len(checks) == 9600 and all(v['bitwise_equal'] for v in checks)
    assert len(a['extra_checks']) == 174
    for c in a['extra_checks']:
        if args.kind == 'profile':
            assert c['checks']['valid_worker_traces'] == 72
        else:
            assert c['checks']['bitwise_equal']
    assert all(s['post_event_tail_byte_equal'] for s in a['samples'])
    savings = [s['per_call_us']['separate'] - s['per_call_us']['fused'] for s in a['samples']]
    for s, value in zip(a['samples'], savings):
        assert abs(s['paired_saving_us'] - value) < 1e-9
    medians = {arm: statistics.median(s['per_call_us'][arm] for s in a['samples'])
               for arm in ('fused', 'separate')}
    for arm, median in medians.items():
        assert abs(median - a['summary_us'][arm]['p50']) < 1e-9
    paired_median = statistics.median(savings)
    assert abs(paired_median - a['paired_saving_us']['p50']) < 1e-9
    assert sum(value > 0 for value in savings) == a['paired_wins']
    arenas.append({'owner': a['owner'], 'active_experts': a['timed_route']['active_experts'],
                   'candidate_us': a['summary_us']['fused'],
                   'base_us': a['summary_us']['separate'],
                   'paired_saving_us': a['paired_saving_us'],
                   'paired_wins': a['paired_wins'], 'pairs': 30,
                   'extra_storage': a['extra_storage']})
summary = {
    'scope': r['scope'], 'candidate': args.kind, 'host': 'a3-194', 'physical_die': 4,
    'rows': 12, 'blocks': 24, 'weight_banks': 16, 'calls_per_graph': 16,
    'graph_replays_per_event': 30, 'pairs_per_arena': 30,
    'raw_byte_boundary_checks': 19200, 'post_event_tail_checks': 360,
    'extra_checks': 348, 'extra_check_scope': 'All 72 worker traces' if args.kind == 'profile' else 'Entire INT32[264] active table, including reserved and unused slots',
    'pass': True, 'scratch_matching': r['scratch_matching'],
    'fresh_write_gate': r['fresh_write_gate'],
    'extra_fresh_write_gate': 'Trace zeroed; active table count zero and all other words one before each checked replay and timed warmup',
    'minimum_logical_active_matrix_bytes': r['logical_minimum_active_matrix_bytes'],
    'payload_interpretation': r['logical_payload_interpretation'],
    'candidate_library': r['candidate_library'], 'sources': r['sources'],
    'arenas': arenas, 'report_sha256': hashlib.sha256(data).hexdigest(),
    'interpretation': 'Positive paired saving means candidate is faster. Local FFN only, with synthetic weights and no TP collectives or model TPOT.',
}
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
for source, target in ((name + '.command.json', 'command.json'),
                       (name + '.launcher.log', 'launcher.log'),
                       ('run-ffn-variant-timing-v3.py', 'run.py'),
                       ('archive-ffn-variant-timing-v3.py', 'summarize.py')):
    shutil.copy2(root / source, out / target)
print(json.dumps({'candidate': args.kind, 'pass': True,
                  'arenas': [{key: a[key] for key in ('owner', 'active_experts', 'paired_saving_us', 'paired_wins')}
                             for a in arenas]}, indent=2))
