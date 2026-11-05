"""One M8 production SFA versus full joint-BSND bitmap-path graph screen."""

import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch


ROOT = Path('/workspace/offline/fia-joint-m-uint8-view-screen-20261020-codex')
INPUT = Path('/workspace/offline/sfa-c1-bounds-20261019-codex/cpu-fixture/input.pt')
SOURCE_INPUT_SHA = 'd8b00212e619c88383dcb79c487c050fa38b26a29467240e8ac66d7261f5d23a'
CAPACITY, WIDTH, HEADS = 9344, 8, 4
QUOTA = 256 * 2**20
ROOT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
report = {
    'scope': 'One synthetic M8/C9344 uint8-bitmap bool-view paired graph micro-screen',
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'original_input_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest(),
    'device': 0, 'npu_launched': False, 'quota_bytes': QUOTA,
    'quota_scope': 'Torch allocator, excluding runtime/context allocations',
    'attention_calls_attempted': {'eager': 0, 'capture': 0, 'validation': 0, 'measured': 0},
    'shape': {'M': WIDTH, 'capacity': CAPACITY, 'heads': HEADS, 'topk': 2048,
              'latent_dim': 512, 'rope_dim': 64},
    'timing_scope': ('Each graph contains ten full attention calls. Joint includes '
                     'int32-to-int64 cast, ones/fill, scatter, causal comparison, '
                     'mask OR, and FIA. Prepared device thresholds and static '
                     'columns are reused. Same-process AB/BA ordering alternates '
                     'over thirty pairs; inputs and addresses are hot. Wall '
                     'includes event enqueue and stream synchronization.'),
    'model_tpot_measured': False, 'decoder_precision_verified': False,
    'baseline_source_sha256': 'c807b5f1ec922c8c19404b8a25d76451a09267a68ee51006e9393b284211f731',
    'only_operation_change': 'bitmap.view(torch.bool) before OR; uint8 ones, scatter False, long and GE unchanged',
}
outputs = {}
graphs = {}
samples = []


def difference(value, reference):
    delta = value.double() - reference.double()
    return {'bit_equal': torch.equal(value, reference),
            'finite': bool(torch.isfinite(value).all()),
            'max_abs': delta.abs().max().item(),
            'nrmse': (delta.norm() / reference.double().norm()).item()}


def checked_cpu(value):
    actual = value.cpu().clone()
    assert bool(torch.isfinite(actual).all())
    assert int(actual.count_nonzero()) > 0
    return actual


try:
    assert report['original_input_sha256'] == SOURCE_INPUT_SHA
    case = torch.load(INPUT, map_location='cpu', weights_only=True)
    assert case['query'].shape == (WIDTH, HEADS, 512)
    assert case['query_rope'].shape == (WIDTH, HEADS, 64)
    assert case['sparse_indices'].shape == (WIDTH, 1, 2048)
    assert case['sparse_indices'].dtype == torch.int32
    written = int(case['actual_seq_lengths_kv'][0])
    assert written == 8200
    table = case['block_table'][0].long()
    latent = torch.zeros((CAPACITY, 512), dtype=torch.bfloat16)
    rope = torch.zeros((CAPACITY, 64), dtype=torch.bfloat16)
    latent[:written].copy_(case['key'][table].reshape(-1, 512)[:written])
    rope[:written].copy_(case['key_rope'][table].reshape(-1, 64)[:written])
    assert bool(torch.isfinite(latent).all()) and bool(torch.isfinite(rope).all())
    ids = case['sparse_indices'][:, 0].long()
    for row in range(WIDTH):
        assert ids[row].unique().numel() == 2048
        assert bool(((ids[row] >= 0) & (ids[row] < written - WIDTH + row + 1)).all())
    state_cpu = {}
    for state, end, sign in [('A', written, 1), ('B', written - 1, -1),
                             ('reset', written, 1)]:
        state_cpu[state] = {
            'query': case['query'] * sign,
            'end': torch.tensor([end], dtype=torch.int32),
            'thresholds': torch.arange(end - WIDTH + 1, end + 1, dtype=torch.int32),
        }
    report['B_newly_masked_selection_counts'] = [
        int((ids[row] >= int(state_cpu['B']['thresholds'][row])).sum())
        for row in range(WIDTH)]
    payload = {'query': case['query'], 'query_rope': case['query_rope'],
               'latent': latent, 'rope': rope, 'sparse_indices': case['sparse_indices'],
               'actual_seq_lengths_query': case['actual_seq_lengths_query'],
               'state_cpu': state_cpu}
    torch.save(payload, ROOT / 'input.pt')
    report['input_sha256'] = hashlib.sha256((ROOT / 'input.pt').read_bytes()).hexdigest()
    report['custom_opp_path'] = os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')
    assert '/sfa-c1-' not in report['custom_opp_path']
    assert '/bounds/' not in report['custom_opp_path']
    import torch_npu
    from vllm_ascend.utils import enable_custom_op

    torch_npu.npu.set_device(0)
    total = torch_npu.npu.get_device_properties(0).total_memory
    torch_npu.npu.set_per_process_memory_fraction(QUOTA / total)
    assert enable_custom_op(), 'Production sparse attention custom operator unavailable'
    report['schemas'] = {
        'sfa': str(torch.ops._C_ascend.npu_sparse_flash_attention.default._schema),
        'fia': str(torch.ops.npu.npu_fused_infer_attention_score_v2.default._schema),
    }
    (ROOT / 'schema.json').write_text(json.dumps(report['schemas'], indent=2) + '\n')
    device = torch.device('npu', 0)
    query = case['query'].to(device)
    query_rope = case['query_rope'].to(device)
    key, key_rope = latent.to(device), rope.to(device)
    raw_indices = case['sparse_indices'].to(device)
    q_lengths = case['actual_seq_lengths_query'].to(device)
    kv_lengths = state_cpu['A']['end'].to(device)
    thresholds = state_cpu['A']['thresholds'].to(device)
    columns = torch.arange(CAPACITY, dtype=torch.int32, device=device)
    identity_table = torch.arange(CAPACITY // 128, dtype=torch.int32,
                                  device=device).view(1, -1)

    def sparse_attention():
        return torch.ops._C_ascend.npu_sparse_flash_attention(
            query=query, key=key.view(-1, 128, 1, 512),
            value=key.view(-1, 128, 1, 512),
            query_rope=query_rope, key_rope=key_rope.view(-1, 128, 1, 64),
            sparse_indices=raw_indices, block_table=identity_table,
            actual_seq_lengths_query=q_lengths, actual_seq_lengths_kv=kv_lengths,
            scale_value=1 / 16, sparse_block_size=1, layout_query='TND',
            layout_kv='PA_BSND', sparse_mode=3, attention_mode=2,
            return_softmax_lse=False,
        )[0]

    def joint_attention():
        selected = raw_indices[:, 0].long()
        bitmap = torch.ones((WIDTH, CAPACITY), dtype=torch.uint8, device=device)
        bitmap.scatter_(1, selected, False)
        mask = bitmap.view(torch.bool) | (columns.view(1, CAPACITY) >= thresholds.view(WIDTH, 1))
        return torch_npu.npu_fused_infer_attention_score_v2(
            query.view(1, WIDTH, HEADS, 512), key.view(1, CAPACITY, 1, 512),
            key.view(1, CAPACITY, 1, 512),
            query_rope=query_rope.view(1, WIDTH, HEADS, 64),
            key_rope=key_rope.view(1, CAPACITY, 1, 64),
            atten_mask=mask.view(1, 1, WIDTH, CAPACITY),
            num_query_heads=HEADS, num_key_value_heads=1,
            softmax_scale=1 / 16, input_layout='BSND', sparse_mode=0,
            block_table=None, block_size=0,
            actual_seq_qlen=[WIDTH], actual_seq_kvlen=[CAPACITY],
        )[0].view(WIDTH, HEADS, 512)

    operations = {'sfa': sparse_attention, 'joint': joint_attention}
    stream = torch_npu.npu.Stream()

    def set_state(name):
        query.copy_(state_cpu[name]['query'])
        kv_lengths.copy_(state_cpu[name]['end'])
        thresholds.copy_(state_cpu[name]['thresholds'])
        stream.synchronize()

    with torch.inference_mode(), torch_npu.npu.stream(stream):
        report['npu_launched'] = True
        for name, operation in operations.items():
            report['last_started'] = 'warm_' + name
            for _ in range(7):
                report['attention_calls_attempted']['eager'] += 1
                warm = operation()
            stream.synchronize()
            checked_cpu(warm)
        for state in ['A', 'B', 'reset']:
            set_state(state)
            for name, operation in operations.items():
                report['last_started'] = 'eager_' + state + '_' + name
                report['attention_calls_attempted']['eager'] += 1
                outputs['eager_' + state + '_' + name] = checked_cpu(operation())
        for name in operations:
            assert torch.equal(outputs['eager_A_' + name], outputs['eager_reset_' + name])
            assert not torch.equal(outputs['eager_A_' + name], outputs['eager_B_' + name])
        baseline_path = Path('/workspace/offline/fia-joint-m-screen-20261020-codex/outputs.pt')
        baseline_outputs = torch.load(baseline_path, map_location='cpu', weights_only=True)
        report['bool_baseline_outputs_sha256'] = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        assert report['bool_baseline_outputs_sha256'] == '87131d54ff5e6e029cd3ac1d2d5e7b73164ad6ebb326073bdad6d2d1c72bca04'
        report['dtype_baseline_eager_checks'] = {}
        for state in ['A', 'B', 'reset']:
            for name in operations:
                output_name = 'eager_' + state + '_' + name
                check = difference(outputs[output_name], baseline_outputs[output_name])
                report['dtype_baseline_eager_checks'][output_name] = check
                assert check['bit_equal'], 'Uint8 versus frozen bool baseline output mismatch'
        for name, operation in operations.items():
            report['last_started'] = 'capture_' + name
            graph = torch_npu.npu.NPUGraph()
            with torch_npu.npu.graph(graph, stream=stream):
                for _ in range(10):
                    report['attention_calls_attempted']['capture'] += 1
                    graph_output = operation()
            stream.synchronize()
            graphs[name] = (graph, graph_output)
        report['graph_state_checks'] = []
        for state in ['A', 'B', 'reset']:
            set_state(state)
            for name, (graph, graph_output) in graphs.items():
                report['last_started'] = 'graph_' + state + '_' + name
                graph_output.fill_(float('nan'))
                report['attention_calls_attempted']['validation'] += 10
                graph.replay()
                stream.synchronize()
                actual = checked_cpu(graph_output)
                outputs['graph_' + state + '_' + name] = actual
                check = difference(actual, outputs['eager_' + state + '_' + name])
                report['graph_state_checks'].append({'state': state, 'arm': name, **check})
                assert check['bit_equal'], 'Graph versus eager output mismatch'
        report['cross_path_diagnostics'] = {
            state: difference(outputs['eager_' + state + '_joint'],
                              outputs['eager_' + state + '_sfa'])
            for state in ['A', 'B', 'reset']}
        start = torch_npu.npu.Event(enable_timing=True)
        end = torch_npu.npu.Event(enable_timing=True)
        for pair in range(30):
            order = ['sfa', 'joint'] if pair % 2 == 0 else ['joint', 'sfa']
            for position, name in enumerate(order):
                report['last_started'] = 'measured_' + str(pair) + '_' + name
                graph = graphs[name][0]
                report['attention_calls_attempted']['measured'] += 10
                wall_start = time.perf_counter_ns()
                start.record(stream)
                enqueue_start = time.perf_counter_ns()
                graph.replay()
                enqueue_end = time.perf_counter_ns()
                end.record(stream)
                stream.synchronize()
                wall_end = time.perf_counter_ns()
                event_ms = start.elapsed_time(end)
                samples.append({'pair': pair, 'position': position, 'arm': name,
                                'calls': 10, 'event_total_ms': event_ms,
                                'event_us_per_call': event_ms * 100,
                                'wall_total_ms': (wall_end - wall_start) / 1e6,
                                'wall_us_per_call': (wall_end - wall_start) / 1e4,
                                'host_graph_enqueue_us': (enqueue_end - enqueue_start) / 1e3})
        for name, (_, graph_output) in graphs.items():
            outputs['measured_final_' + name] = checked_cpu(graph_output)
            assert torch.equal(outputs['measured_final_' + name], outputs['eager_A_' + name])
        report['summary'] = {}
        for name in operations:
            arm = [s for s in samples if s['arm'] == name]
            report['summary'][name] = {
                key + '_median': statistics.median(s[key] for s in arm)
                for key in ['event_us_per_call', 'wall_us_per_call', 'host_graph_enqueue_us']}
        paired = []
        for pair in range(30):
            arms = {s['arm']: s for s in samples if s['pair'] == pair}
            paired.append({'pair': pair,
                           'joint_over_sfa_event': arms['joint']['event_us_per_call'] / arms['sfa']['event_us_per_call'],
                           'joint_minus_sfa_event_us': arms['joint']['event_us_per_call'] - arms['sfa']['event_us_per_call'],
                           'joint_over_sfa_wall': arms['joint']['wall_us_per_call'] / arms['sfa']['wall_us_per_call']})
        report['paired_comparisons'] = paired
        report['paired_event_ratio_median'] = statistics.median(p['joint_over_sfa_event'] for p in paired)
        report['paired_wall_ratio_median'] = statistics.median(p['joint_over_sfa_wall'] for p in paired)
        report['paired_event_delta_us_median'] = statistics.median(p['joint_minus_sfa_event_us'] for p in paired)
        report['status'] = 'single_screen_completed'
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
