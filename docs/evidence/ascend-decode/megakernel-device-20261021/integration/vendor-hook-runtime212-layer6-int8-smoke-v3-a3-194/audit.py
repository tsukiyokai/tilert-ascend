"""Archive and audit actual runtime-212 attention reports."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile


ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
EVIDENCE = Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration')
TASK = '/home/shanshan/tilert-megakernel-20261021'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes', 'a3-194']


def load(path):
    return json.loads(path.read_text())


def audit(mode, phase, archive):
    name = f'vendor-hook-runtime212-layer6-{mode}-{phase}-v3'
    output = EVIDENCE / (name + '-a3-194')
    if archive:
        output.mkdir(exist_ok=False)
        command = ['tar', '-czf', '-', '-C', TASK + '/logs/' + name, '.']
        data = subprocess.check_output(SSH + [shlex.join(command)])
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
            tar.extractall(output, filter='data')
        shutil.copy2(ROOT / 'run-runtime212-attention-v3.py', output / 'run.py')
        shutil.copy2(ROOT / (name + '.command.json'), output / 'command.json')
        shutil.copy2(ROOT / (name + '.launcher.log'), output / 'launcher.log')
        shutil.copy2(ROOT / 'runtime212-host-memory-v3.json', output / 'host-memory.json')
        wrapper = subprocess.check_output(SSH + ['cat', TASK + '/vendor_hook_runtime212_run_memory64_tmp.sh'])
        assert hashlib.sha256(wrapper).hexdigest() == '94c9915014a07efa2e946fe44966b97324283886fd0d6548dc51e8f78b335625'
        (output / 'host-launcher.sh').write_bytes(wrapper)

    launcher = load(output / 'summary.json')
    assert launcher['success'] and launcher['exitcodes'] == [0] * 16
    ranks = [load(output / f'rank-{rank:02}.json') for rank in range(16)]
    for rank, report in enumerate(ranks):
        assert report['rank'] == rank and not report.get('error')
        assert report['rank_zero_status']['success']
        assert not report['npu_initialized_before']
        assert report['npu_initialized_after'] == (rank == 0)
        assert all(stage['success'] for stage in report['stages'])
        groups = next(stage for stage in report['stages'] if stage['name'] == 'real_groups')
        assert groups['tp_world_size'] == 16 and groups['tp_rank'] == rank
        assert groups['cpu_rank_sum'] == groups['expected_rank_sum'] == 120
    stages = {item['name']: item for item in ranks[0]['stages']}
    ctor = stages['real_attention_constructor']
    assert ctor['selection']['layer_index'] == 6
    assert ctor['selection']['ffn'] == 'sparse' and ctor['selection']['indexer'] == 'full'
    assert ctor['actual_skip_topk'] is False and ctor['tp_size'] == 16
    assert ctor['reduce_results'] is False
    bind = stages['npu_bind']
    assert bind['device_count'] == 1 and bind['logical_device'] == 0

    report_path = output / ('outer-graph.json' if phase == 'graph' else 'outer-forward.json')
    report = load(report_path)
    assert report['pass'] and report['mode'] == mode
    widths = [1, 2, 12, 32] if phase == 'graph' else [1]
    iterations = 8 if phase == 'graph' else 2
    assert report['widths'] == widths and report['iterations'] == iterations
    assert {(case['rows'], case['iteration']) for case in report['cases']} == {
        (rows, iteration) for rows in widths for iteration in range(iterations)
    }
    assert len(report['cases']) == len(widths) * iterations
    boundaries = {'output', 'topk', 'main_cache', 'rope_cache', 'index_cache'}
    if mode == 'int8':
        boundaries.add('index_scale')
    raw_checks = 0
    replays = 0
    for case in report['cases']:
        assert case['pass']
        if phase == 'graph':
            assert set(case['arms']) == {'vendor', 'native'}
            arms = case['arms'].values()
        else:
            arms = [case]
        for arm in arms:
            assert arm['pass'] and set(arm['checks']) == boundaries
            for check in arm['checks'].values():
                assert check['pass'] and check['bitwise_equal']
                assert check['bitwise_different_bytes'] == 0
                raw_checks += 1
            if phase == 'graph':
                assert arm['python_calls_during_replay'] == 0
                assert arm['native_python_calls_during_replay'] == 0
                assert arm['replay_count'] == case['iteration'] + 1
                replays += 1
    captures = []
    if phase == 'graph':
        captures = report['captures']
        assert len(captures) == 2 * len(widths)
        assert {(item['rows'], item['arm']) for item in captures} == {
            (rows, arm) for rows in widths for arm in ('vendor', 'native')
        }
        assert all(item['capture_counter_delta'] == 1 for item in captures)
        assert all(item['runnable_calls'] == 1 for item in captures)
        assert report['updatable_graph_branch'] is False
    else:
        assert report['collective'] is False and report['graph_capture'] is False
    result = {
        'pass': True,
        'scope': 'Actual layer-6 attention local TP16 shard, synthetic parameters, one NPU; no TP collective or FFN execution.',
        'torch': ranks[0]['torch'],
        'torch_npu': ranks[0]['torch_npu'],
        'mode': mode,
        'widths': widths,
        'iterations': iterations,
        'rank_successes': 16,
        'npu_rank_count': 1,
        'cases': len(report['cases']),
        'captures': len(captures),
        'actual_graph_replays': replays,
        'raw_byte_checks': raw_checks,
        'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'native_libraries': stages['load_native_libraries']['sha256'],
    }
    (output / 'acceptance-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    shutil.copy2(__file__, output / 'audit.py')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('int8', 'bf16'))
    parser.add_argument('phase', choices=('smoke', 'graph'))
    parser.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    audit(args.mode, args.phase, args.archive)
