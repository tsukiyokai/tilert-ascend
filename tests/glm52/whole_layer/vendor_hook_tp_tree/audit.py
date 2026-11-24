"""Audit the frozen three-rank layer reports and actual HCCL launch records."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


MANIFEST = '071bc4e035d559fde4641bee95665f01179eedf7345e78f40adced5b377d908c'
CANDIDATE = '7ec3e8fdd6155d99d98a9b69d2d4b02dc43c9417695f7632ad6559f42ecb96b3'
LIBRARIES = {CANDIDATE,
    '609c973e92b61637f7768ccefb0fbd84b2e13fcfb066b241a41cd5d8376fdd0f',
    '8deff750464e23bb871302c8626b86d37b60c28a00991d550f95dfcfd47ecb6f',
    '64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c',
    '381febf94f918cc7393019b6f99ec8dc38e9a5cc9e63b2d49d5fe483158e3ff2'}
SHARDED = {'self_attn.q_b_proj.weight', 'self_attn.kv_b_proj.weight',
    'self_attn.o_proj.weight', 'mlp.shared_experts.gate_up_proj.weight',
    'mlp.shared_experts.down_proj.weight',
    'mlp.experts.routed_experts.w13_weight', 'mlp.experts.routed_experts.w2_weight'}
COMMON = ('input_hidden', 'input_residual', 'layer6.retained', 'layer7.retained',
          'layer6.ar1', 'layer6.ar2', 'layer7.ar1', 'layer7.ar2')
PARTIALS = ('layer6.attention_partial', 'layer6.ffn_partial',
            'layer7.attention_partial', 'layer7.ffn_partial')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(metric):
    assert metric['pass'] and metric['bitwise_equal']
    assert metric['bitwise_different_bytes'] == 0


def launches(raw):
    events = []
    for path in sorted(raw.rglob('*.log')):
        for number, line in enumerate(path.open(errors='replace'), 1):
            selected = '[SelectAlgfor91093]' in line
            launched = '[INFO]' in line and '[ExecuteKernelLaunchInner]' in line
            if not (selected or launched):
                continue
            process = int(re.search(r'HCCL\((\d+),', line)[1])
            timestamp = re.search(r'\d{4}-\d\d-\d\d-\d\d:\d\d:\d\d\.\d+\.\d+', line)[0]
            events.append((process, timestamp, selected, path, number, line))
    selected_by_process, selections, calls = {}, [], []
    for process, timestamp, selected, path, number, line in sorted(events):
        entry = {'path': str(path.relative_to(raw)), 'line': number,
                 'process': process, 'timestamp': timestamp}
        if selected:
            assert 'algName [AllReduceMeshAivFor91093Executor]' in line
            selected_by_process[process] = entry
            selections.append(entry)
            continue
        assert process in selected_by_process
        fields = dict(re.findall(r'\b([A-Za-z]\w*)\s*\[([^\]]*)\]', line))
        for key, value in {'rankSize': '3', 'dataType': 'float32',
                           'reduceOp': 'sum', 'deterministic': '1'}.items():
            assert fields[key] == value
        rank, tag, count = (int(fields[k]) for k in ('rank', 'tag', 'count'))
        rows = 1 if tag <= 32 else 12
        assert rank in range(3) and 1 <= tag <= 64 and count == rows * 6144
        assert fields['sendbuff'] == fields['recvbuff']
        calls.append({**entry, 'rank': rank, 'tag': tag, 'rows': rows,
                      'fields': fields, 'selection': selected_by_process[process]})
    assert len(selections) == 6 and len(calls) == 192
    assert Counter((c['rank'], c['tag']) for c in calls) == Counter(
        (rank, tag) for rank in range(3) for tag in range(1, 65))
    return {'selections': selections, 'calls': calls}


def audit(raw, fixture, graph):
    assert digest(fixture / 'manifest.json') == MANIFEST
    manifest = read(fixture / 'manifest.json')
    pins = manifest['files_sha256']
    for name, expected in pins.items():
        assert digest(fixture / name) == expected, name
    name = raw.name.removesuffix('-raw')
    directory = raw / name
    driver = read(directory / 'summary.json')
    device_script = 'tp_graph_device.py' if graph else 'tp_device.py'
    assert driver['success'] and driver['exitcodes'] == [0] * 16
    assert driver['script_sha256'] == pins[device_script]
    assert driver['arguments']['physical_devices'] == [4, 6, 7]
    assert driver['arguments']['forward_widths'] == [1, 12]
    assert driver['arguments']['forward_replays'] == 8
    workers = []
    for rank in range(16):
        worker = read(directory / f'rank-{rank:02d}.json')
        assert worker['success'] and worker['rank'] == rank
        assert (worker['torch'], worker['torch_npu']) == ('2.12.0+cu130', '2.12.0')
        assert not worker['npu_initialized_before']
        assert worker['npu_initialized_after'] == (rank < 3)
        assert worker['device_participant'] == (rank < 3)
        assert all(stage['success'] for stage in worker['stages'])
        assert worker['all_rank_statuses'] == [
            {'rank': value, 'success': True} for value in range(16)]
        stages = {row['name']: row for row in worker['stages']}
        assert stages['actual_tp16_gloo_geometry']['cpu_rank_sum'] == 120
        assert stages['actual_tp16_gloo_geometry']['tp_rank'] == rank
        if rank < 3:
            mapping = stages['physical_device_mapping']
            assert mapping['logical_to_physical'] == [4, 6, 7]
            assert mapping['logical_rank'] == rank
            assert mapping['direct_peer_access'] == {
                str(peer): 1 for peer in range(3) if peer != rank}
            group = stages['actual_mc2_and_hccl3_groups']
            assert group['mc2_ranks'] == [rank]
            assert group['oracle_hccl_ranks'] == [0, 1, 2]
            assert group['oracle_rank'] == rank and group['tp_constructor_size'] == 16
            assert not group['tp16_device_collective']
            mode = 'graph' if graph else 'chain'
            assert worker['result'][f'three_rank_component_{mode}_executed']
            assert not worker['result']['full_decoder_forward_executed']
        workers.append(stages)
    for layer in (6, 7):
        rows = [{p['name']: p for p in stage[f'layer{layer}_move_parameters']['parameters']}
                for stage in workers[:3]]
        assert all(row.keys() == rows[0].keys() for row in rows) and SHARDED <= rows[0].keys()
        for name in rows[0]:
            for field in ('shape', 'dtype', 'bytes'):
                assert all(row[name][field] == rows[0][name][field] for row in rows)
            assert len({row[name]['sha256'] for row in rows}) == (3 if name in SHARDED else 1)
            assert all(row[name]['replicated'] == (name not in SHARDED) for row in rows)
    reports, identities, checks, reductions = [], {}, 0, 0
    filename = 'tp-chain-graph.json' if graph else 'tp-chain.json'
    for rank in range(3):
        path = directory / f'device-rank{rank}'
        report = read(path / filename)
        identities[str(rank)] = digest(path / filename)
        assert report['pass'] and report['rank'] == rank and report['ranks'] == 3
        assert report['constructor_tp_size'] == 16 and report['tp16_collectives_executed'] == 0
        assert report['eager'] == (not graph) and report['graph_capture'] == graph
        assert not report['timing'] and not report['original_decoder_forward_called']
        assert report['widths'] == [1, 12] and report['iterations'] == 8
        assert report['blocks'] == 24 and set(report['libraries'].values()) == LIBRARIES
        assert 'BF16.float -> SUM -> BF16' in report['original_hccl_entry']
        for key, source in (('script', 'tp_graph.py' if graph else 'tp_chain.py'),
                            ('parent_phases', 'tp_parent.py'), ('tp_host', 'tp_host.py')):
            assert report[key]['sha256'] == pins[source]
        assert report['actual_decoder']['sha256'] == '96f4f755f62caa85869fb884209e7f8869882ddb6efbd40c930778b27bd3e39d'
        window = report['window']
        assert window['bytes'] == 2359744 and window['capacity'] == 32
        assert window['address'] > 0 and window['stream'] > 0
        if graph:
            assert window['capture_manager_retained']
            assert [c['rows'] for c in report['captures']] == [1, 12]
            for capture in report['captures']:
                assert capture['phase'] == 'captured_not_executed'
                assert capture['runnable_calls'] == capture['capture_counter_delta'] == 1
                assert capture['capture_stream'] == capture['window_stream'] == window['stream']
                assert capture['caller_stream'] == capture['capture_stream']
                assert capture['unchanged_capture_boundaries'] == 113
                assert capture['graph_type'] == 'UpdatableGraph'
                assert capture['workspace_preallocated'] and capture['window_allocated_before_capture']
                assert capture['wrapper']['sha256'] == '8e1c269e331d005f1361acc5cec24402bff7ae39b75e952ea890da7812804955'
                assert capture['capture_context']['sha256'] == 'a6c51c8bf24d676816f0d4450d24520abd6029cd3a1031a2b8d9140eed40407f'
                assert capture['torch_graph_manager']['sha256'] == '7d76706e096c0249dd63a3000a625b956df3841f62a163efb8dbc1bf76b26c6a'
        assert len(report['cases']) == 16
        for width in (1, 12):
            cases = [row for row in report['cases'] if row['rows'] == width]
            assert [row['iteration'] for row in cases] == list(range(8))
            for key in ('consumer.output', 'consumer.retained'):
                assert len({row['outputs'][key]['sha256'] for row in cases}) == 8
                assert all(row['outputs'][key]['finite'] and row['outputs'][key]['nonzero_elements'] > 0 for row in cases)
            for key in ('hidden', 'slot_i64:0', 'slot_i64:2'):
                assert len({row['inputs']['tensors'][key]['sha256'] for row in cases}) == 8
            for case in cases:
                assert case['pass'] and len(case['checks']) == 113
                for value in case['checks'].values():
                    exact(value)
                    checks += 1
                if graph:
                    assert case['explicit_replay'] == case['iteration'] + 1
                    assert case['python_calls_during_replay'] == 0
                    assert case['capture_stream'] == window['stream']
                assert [(r['layer'], r['phase']) for r in case['hccl_ordered_checks']] == [
                    (layer, phase) for layer in (6, 7) for phase in ('ar1', 'ar2')]
                for reduction in case['hccl_ordered_checks']:
                    exact(reduction['check'])
                    assert len(reduction['partials']) == 3
                    assert len({p['sha256'] for p in reduction['partials']}) == 3
                    reductions += 1
                witness = case['rank_witnesses']
                assert [row['rank'] for row in witness] == [0, 1, 2]
                assert all(len({row[key] for row in witness}) == 1 for key in COMMON)
                assert all(len({row[key] for row in witness}) == 3 for key in PARTIALS)
        nz = read(path / 'nz-storage-correction.json')
        assert nz['completed'] and nz['additional_hbm_bytes'] == 0 and len(nz['banks']) == 4
        for bank in nz['banks']:
            assert bank['raw_bytes_equal'] and bank['raw_sha256_before'] == bank['raw_sha256_after']
            assert bank['new_device_allocation_bytes'] == 0
            assert bank['before']['data_ptr'] == bank['after']['data_ptr']
        reports.append(report)
    for cases in zip(*(report['cases'] for report in reports)):
        assert all(row['rank_witnesses'] == cases[0]['rank_witnesses'] for row in cases)
    assert checks == 5424 and reductions == 192
    hccl = launches(raw)
    return {'pass': True, 'scope': 'Three TP16-shaped shards with synthetic parameters',
            'physical_devices': [4, 6, 7], 'rows': [1, 12], 'inputs_per_width': 8,
            'raw_byte_checks': checks, 'hccl_tree_checks': reductions,
            'selector_records': len(hccl['selections']), 'hccl_launch_records': len(hccl['calls']),
            'graph_capture': graph, 'official_captures': 6 if graph else 0,
            'explicit_replays': 48 if graph else 0,
            'capture_unchanged_boundaries': 678 if graph else 0,
            'python_calls_during_replay': 0 if graph else None,
            'checkpoint_loaded': False, 'tp16_device_collectives': 0, 'timing': False,
            'fixture_manifest_sha256': MANIFEST, 'candidate_sha256': CANDIDATE,
            'rank_report_sha256': identities, 'hccl': hccl}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'fixture', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--kind', choices=('eager', 'graph'), required=True)
    args = parser.parse_args()
    result = audit(args.raw, args.fixture, args.kind == 'graph')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'hccl'}))
