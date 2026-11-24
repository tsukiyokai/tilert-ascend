"""One M8 production SFA versus complete joint-BSND bitmap graph profile."""

import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import torch


ROOT = Path('/workspace/offline/fia-joint-m-profile-20261020-codex')
INPUT = Path('/workspace/offline/sfa-c1-bounds-20261019-codex/cpu-fixture/input.pt')
SOURCE_INPUT_SHA = 'd8b00212e619c88383dcb79c487c050fa38b26a29467240e8ac66d7261f5d23a'
CAPACITY, WIDTH, HEADS = 9344, 8, 4
QUOTA = 256 * 2**20
ROOT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
report = {
    'scope': 'One synthetic M8/C9344 complete-path graph attribution profile',
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'original_input_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest(),
    'device': 0, 'npu_launched': False, 'quota_bytes': QUOTA,
    'quota_scope': 'Torch allocator, excluding runtime/context allocations',
    'attention_calls_attempted': {'eager': 0, 'capture': 0, 'validation': 0, 'profiled': 0},
    'shape': {'M': WIDTH, 'capacity': CAPACITY, 'heads': HEADS, 'topk': 2048,
              'latent_dim': 512, 'rope_dim': 64},
    'profile_scope': ('Each pre-existing ten-call graph replays exactly once in '
                      'one CPU/NPU profiler window; no recapture in the window. '
                      'Joint contains cast, ones/fill, scatter, causal comparison, '
                      'mask OR, and FIA. Static columns and prepared thresholds '
                      'are reused. No repeated timing screen.'),
    'screen_source_sha256': '0a0bc242f829daf03abaa2b2d00741d8bf8c68cbb1beacd20fba437664a673c5',
    'profiler_analysis': 'CPU offline analysis after device process exits',
    'model_tpot_measured': False, 'decoder_precision_verified': False,
}
outputs = {}
graphs = {}


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
        bitmap = torch.ones((WIDTH, CAPACITY), dtype=torch.bool, device=device)
        bitmap.scatter_(1, selected, False)
        mask = bitmap | (columns.view(1, CAPACITY) >= thresholds.view(WIDTH, 1))
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
        from torch_npu import profiler
        from torch_npu.npu import mstx

        trace_directory = ROOT / 'trace'
        profile_options = {
            'activities': [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
            'record_shapes': False, 'profile_memory': False, 'with_stack': False,
            'on_trace_ready': profiler.tensorboard_trace_handler(
                str(trace_directory), worker_name='sfa-joint', analyse_flag=False),
            'experimental_config': profiler._ExperimentalConfig(
                profiler_level=profiler.ProfilerLevel.Level1,
                aic_metrics=profiler.AiCMetrics.AiCoreNone, mstx=True),
        }
        stream.synchronize()
        report['profile_replay_order'] = ['sfa', 'joint']
        with profiler.profile(**profile_options):
            for name in report['profile_replay_order']:
                report['last_started'] = 'profile_' + name
                report['attention_calls_attempted']['profiled'] += 10
                identifier = mstx.range_start('profile_' + name + '_ten_calls')
                with torch.autograd.profiler.record_function('profile_' + name):
                    graphs[name][0].replay()
                    stream.synchronize()
                mstx.range_end(identifier)
        for name, (_, graph_output) in graphs.items():
            outputs['profile_final_' + name] = checked_cpu(graph_output)
            assert torch.equal(outputs['profile_final_' + name], outputs['eager_A_' + name])
        report['trace_files'] = [
            {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size}
            for path in sorted(trace_directory.rglob('*')) if path.is_file()]
        assert any('device_' in entry['path'] and '/data/' in entry['path']
                   and entry['bytes'] for entry in report['trace_files'])
        report['status'] = 'single_attribution_profile_completed'

except Exception:
    report['status'] = 'first_error_stopped'
    report['exception'] = traceback.format_exc()
finally:
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
