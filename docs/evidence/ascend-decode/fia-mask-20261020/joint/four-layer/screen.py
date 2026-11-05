"""One four-layer SFA versus actual-helper shared-mask graph screen."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch


ROOT = Path('/workspace/offline/fia-four-layer-screen-20261020-codex')
CONTRACT = Path('/workspace/offline/fia-four-layer-contract-20261020-codex')
FIXTURE = CONTRACT / 'fixture.pt'
HELPER_SHA = 'c1950e22a6add43b71372a107f1268bf946c9628b42f17baf24782be944b957b'
FIXTURE_SHA = '96c8c5aaa295d5d6755fe65e611b9b801bc5207f338dc33c8ebffb599f0c56ec'
WIDTH, CAPACITY, LAYERS = 8, 9344, 4
QUOTA = 256 * 2**20
torch.set_num_threads(1)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
ROOT.mkdir(exist_ok=True, parents=True)
report = {
    'scope': 'One synthetic four-layer M8/C9344 shared-selection group screen',
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'helper_sha256': hashlib.sha256((CONTRACT / 'cann_attention.py').read_bytes()).hexdigest(),
    'fixture_sha256': hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
    'npu_launched': False, 'device': 0, 'quota_bytes': QUOTA,
    'quota_scope': 'Torch allocator; excludes runtime/context allocations',
    'attention_calls_attempted': {'eager': 0, 'rebuilt_reference': 0,
                                 'capture': 0, 'validation': 0, 'measured': 0},
    'shape': {'M': WIDTH, 'capacity': CAPACITY, 'layers': LAYERS,
              'heads': 4, 'topk': 2048},
    'groups_per_graph': 1, 'attention_calls_per_group': 4,
    'timing_scope': ('One graph replay per arm and pair, thirty AB/BA pairs. '
                     'Each group contains four different query/KV layers. '
                     'Shared path includes one GE, actual selection_mask, and '
                     'four actual shared_kv_fia calls. Static columns and prepared '
                     'threshold metadata are reused. Wall includes events and '
                     'stream synchronization; inputs and addresses are hot.'),
    'model_tpot_measured': False, 'decoder_precision_verified': False,
    'production_reuse_eligibility_verified': False,
}
outputs, graphs, samples = {}, {}, []
phase = 'eager'


def difference(actual, reference):
    delta = actual.double() - reference.double()
    return {'bit_equal': torch.equal(actual, reference),
            'finite': bool(torch.isfinite(actual).all()),
            'max_abs': delta.abs().max().item(),
            'nrmse': (delta.norm() / reference.double().norm()).item()}


def checked_cpu(values):
    result = [value.cpu().clone() for value in values]
    assert len(result) == LAYERS
    assert all(bool(torch.isfinite(value).all()) and int(value.count_nonzero()) > 0
               for value in result)
    return result


try:
    assert report['helper_sha256'] == HELPER_SHA
    assert report['fixture_sha256'] == FIXTURE_SHA
    (ROOT / 'cann_attention.py').write_bytes((CONTRACT / 'cann_attention.py').read_bytes())
    spec = importlib.util.spec_from_file_location('actual_cann_attention', ROOT / 'cann_attention.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    case = torch.load(FIXTURE, map_location='cpu', weights_only=True)
    cpu_layers = case['layers']
    assert len(cpu_layers) == LAYERS
    assert len({layer['q'].data_ptr() for layer in cpu_layers}) == LAYERS
    assert len({layer['kv'][0].data_ptr() for layer in cpu_layers}) == LAYERS
    assert all(bool(torch.isfinite(value).all()) for layer in cpu_layers
               for value in [layer['q'], layer['qr'], *layer['kv']])
    states = {
        state: {'ids': case['ids_' + ('B' if state == 'B' else 'A')],
                'thresholds': case['thresholds_' + ('B' if state == 'B' else 'A')],
                'end': torch.tensor([case['written_B' if state == 'B' else 'written_A']],
                                    dtype=torch.int32)}
        for state in ['A', 'B', 'reset']}
    report['custom_opp_path'] = os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')
    assert '/sfa-c1-' not in report['custom_opp_path'] and '/bounds/' not in report['custom_opp_path']
    import torch_npu
    from vllm_ascend.utils import enable_custom_op

    torch_npu.npu.set_device(0)
    total = torch_npu.npu.get_device_properties(0).total_memory
    torch_npu.npu.set_per_process_memory_fraction(QUOTA / total)
    assert enable_custom_op()
    report['schemas'] = {
        'sfa': str(torch.ops._C_ascend.npu_sparse_flash_attention.default._schema),
        'fia': str(torch.ops.npu.npu_fused_infer_attention_score_v2.default._schema)}
    (ROOT / 'schema.json').write_text(json.dumps(report['schemas'], indent=2) + '\n')
    device = torch.device('npu', 0)
    layers = [{'q': layer['q'].to(device), 'qr': layer['qr'].to(device),
               'kv': tuple(value.to(device) for value in layer['kv'])}
              for layer in cpu_layers]
    assert len({layer['q'].data_ptr() for layer in layers}) == LAYERS
    assert len({layer['kv'][0].data_ptr() for layer in layers}) == LAYERS
    report['layer_device_ptrs'] = [
        {name: layer[name].data_ptr() for name in ['q', 'qr']}
        | {'latent': layer['kv'][0].data_ptr(), 'rope': layer['kv'][1].data_ptr()}
        for layer in layers]
    indices = states['A']['ids'].to(device)
    thresholds = states['A']['thresholds'].to(device)
    kv_lengths = states['A']['end'].to(device)
    q_lengths = torch.tensor([WIDTH], dtype=torch.int32, device=device)
    columns = torch.arange(CAPACITY, dtype=torch.int32, device=device).view(1, CAPACITY)
    table = torch.arange(CAPACITY // 128, dtype=torch.int32, device=device).view(1, -1)

    def attention(arm, layer_id, mask=None):
        layer = layers[layer_id]
        report['attention_calls_attempted'][phase] += 1
        report['last_started'] = f'{phase}_{arm}_layer{layer_id}'
        if arm == 'sfa':
            return torch.ops._C_ascend.npu_sparse_flash_attention(
                query=layer['q'], key=layer['kv'][0].view(-1, 128, 1, 512),
                value=layer['kv'][0].view(-1, 128, 1, 512),
                query_rope=layer['qr'], key_rope=layer['kv'][1].view(-1, 128, 1, 64),
                sparse_indices=indices.view(WIDTH, 1, 2048), block_table=table,
                actual_seq_lengths_query=q_lengths, actual_seq_lengths_kv=kv_lengths,
                scale_value=1 / 16, sparse_block_size=1,
                layout_query='TND', layout_kv='PA_BSND', sparse_mode=3,
                attention_mode=2, return_softmax_lse=False,
            )[0]
        return helper.shared_kv_fia(layer['q'], layer['qr'], layer['kv'], mask, 1 / 16)

    def sfa_group():
        return tuple(attention('sfa', layer) for layer in range(LAYERS))

    def shared_group():
        causal_mask = columns >= thresholds.view(WIDTH, 1)
        mask = helper.selection_mask(indices, causal_mask)
        return tuple(attention('shared', layer, mask) for layer in range(LAYERS))

    def rebuilt_group():
        result = []
        for layer in range(LAYERS):
            causal_mask = columns >= thresholds.view(WIDTH, 1)
            mask = helper.selection_mask(indices, causal_mask)
            result.append(attention('rebuilt', layer, mask))
        return tuple(result)

    operations = {'sfa': sfa_group, 'shared': shared_group}
    stream = torch_npu.npu.Stream()

    def set_state(state):
        indices.copy_(states[state]['ids'])
        thresholds.copy_(states[state]['thresholds'])
        kv_lengths.copy_(states[state]['end'])
        stream.synchronize()

    with torch.inference_mode(), torch_npu.npu.stream(stream):
        report['npu_launched'] = True
        for name, operation in operations.items():
            for _ in range(7):
                warm = operation()
            stream.synchronize()
            checked_cpu(warm)
        report['shared_vs_rebuilt_checks'] = []
        for state in ['A', 'B', 'reset']:
            set_state(state)
            for name, operation in operations.items():
                phase = 'eager'
                values = checked_cpu(operation())
                outputs.update({f'eager_{state}_{name}_{layer}': value
                                for layer, value in enumerate(values)})
            phase = 'rebuilt_reference'
            values = checked_cpu(rebuilt_group())
            outputs.update({f'eager_{state}_rebuilt_{layer}': value
                            for layer, value in enumerate(values)})
            for layer in range(LAYERS):
                check = difference(outputs[f'eager_{state}_shared_{layer}'], values[layer])
                report['shared_vs_rebuilt_checks'].append({'state': state, 'layer': layer, **check})
                assert check['bit_equal'], 'Shared mask versus rebuilt FIA mismatch'
        for name in operations:
            for layer in range(LAYERS):
                assert torch.equal(outputs[f'eager_A_{name}_{layer}'], outputs[f'eager_reset_{name}_{layer}'])
                assert not torch.equal(outputs[f'eager_A_{name}_{layer}'], outputs[f'eager_B_{name}_{layer}'])
        phase = 'capture'
        for name, operation in operations.items():
            graph = torch_npu.npu.NPUGraph()
            with torch_npu.npu.graph(graph, stream=stream):
                graph_outputs = operation()
            stream.synchronize()
            graphs[name] = (graph, graph_outputs)
        report['graph_state_checks'] = []
        for state in ['A', 'B', 'reset']:
            set_state(state)
            for name, (graph, graph_outputs) in graphs.items():
                for value in graph_outputs:
                    value.fill_(float('nan'))
                report['last_started'] = f'graph_validation_{state}_{name}'
                report['attention_calls_attempted']['validation'] += LAYERS
                graph.replay()
                stream.synchronize()
                actual = checked_cpu(graph_outputs)
                for layer, value in enumerate(actual):
                    outputs[f'graph_{state}_{name}_{layer}'] = value
                    check = difference(value, outputs[f'eager_{state}_{name}_{layer}'])
                    report['graph_state_checks'].append({'state': state, 'arm': name,
                                                         'layer': layer, **check})
                    assert check['bit_equal'], 'Graph versus eager mismatch'
        report['cross_path_diagnostics'] = {
            f'{state}_layer{layer}': difference(outputs[f'eager_{state}_shared_{layer}'],
                                                outputs[f'eager_{state}_sfa_{layer}'])
            for state in ['A', 'B', 'reset'] for layer in range(LAYERS)}
        start = torch_npu.npu.Event(enable_timing=True)
        end = torch_npu.npu.Event(enable_timing=True)
        for pair in range(30):
            order = ['sfa', 'shared'] if pair % 2 == 0 else ['shared', 'sfa']
            for position, name in enumerate(order):
                report['last_started'] = f'measured_{pair}_{name}'
                report['attention_calls_attempted']['measured'] += LAYERS
                graph = graphs[name][0]
                wall_start = time.perf_counter_ns()
                start.record(stream)
                enqueue_start = time.perf_counter_ns()
                graph.replay()
                enqueue_end = time.perf_counter_ns()
                end.record(stream)
                stream.synchronize()
                wall_end = time.perf_counter_ns()
                group_us = start.elapsed_time(end) * 1000
                wall_us = (wall_end - wall_start) / 1000
                samples.append({'pair': pair, 'position': position, 'arm': name,
                                'groups': 1, 'attention_calls': LAYERS,
                                'event_group_us': group_us, 'wall_group_us': wall_us,
                                'event_amortized_attention_us': group_us / LAYERS,
                                'wall_amortized_attention_us': wall_us / LAYERS,
                                'host_graph_enqueue_us': (enqueue_end - enqueue_start) / 1000})
        for name, (_, graph_outputs) in graphs.items():
            for layer, value in enumerate(checked_cpu(graph_outputs)):
                outputs[f'measured_final_{name}_{layer}'] = value
                assert torch.equal(value, outputs[f'eager_A_{name}_{layer}'])
        report['summary'] = {
            name: {key + '_median': statistics.median(sample[key] for sample in samples
                                                       if sample['arm'] == name)
                   for key in ['event_group_us', 'wall_group_us',
                               'event_amortized_attention_us', 'wall_amortized_attention_us']}
            for name in operations}
        paired = []
        for pair in range(30):
            arms = {sample['arm']: sample for sample in samples if sample['pair'] == pair}
            paired.append({'pair': pair,
                           'shared_over_sfa_event': arms['shared']['event_group_us'] / arms['sfa']['event_group_us'],
                           'shared_minus_sfa_event_group_us': arms['shared']['event_group_us'] - arms['sfa']['event_group_us'],
                           'shared_over_sfa_wall': arms['shared']['wall_group_us'] / arms['sfa']['wall_group_us']})
        report['paired_comparisons'] = paired
        report['paired_event_ratio_median'] = statistics.median(p['shared_over_sfa_event'] for p in paired)
        report['paired_wall_ratio_median'] = statistics.median(p['shared_over_sfa_wall'] for p in paired)
        report['paired_event_delta_group_us_median'] = statistics.median(p['shared_minus_sfa_event_group_us'] for p in paired)
        assert sum(report['attention_calls_attempted'].values()) == 364
        report['status'] = 'single_four_layer_screen_completed'
except Exception:
    report['status'] = 'first_error_stopped'
    report['exception'] = traceback.format_exc()
finally:
    report['samples'] = samples
    torch.save(outputs, ROOT / 'outputs.pt')
    report['outputs_sha256'] = hashlib.sha256((ROOT / 'outputs.pt').read_bytes()).hexdigest()
    libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if any(part in line for part in ['opmaster', 'optiling', 'opapi'])})
    report['loaded_libraries'] = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                  for path in libraries if Path(path).is_file()}
    if report['npu_launched']:
        report['peak_allocated_bytes'] = torch_npu.npu.max_memory_allocated()
        report['peak_reserved_bytes'] = torch_npu.npu.max_memory_reserved()
    (ROOT / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
if report['status'] == 'first_error_stopped':
    sys.exit(1)
