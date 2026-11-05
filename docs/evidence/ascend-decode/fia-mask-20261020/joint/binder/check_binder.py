"""Functional check of the actual four-role SharedKVAttention binder."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

import torch


ROOT = Path('/workspace/offline/shared-fia-binder-device-20261020-codex')
SOURCE = Path('/workspace/offline/shared-fia-integration-20261020/tree/tilert/models/glm_5_2_ascend/cann_attention.py')
FIXTURE = Path('/workspace/offline/fia-four-layer-contract-20261020-codex/fixture.pt')
REFERENCE = Path('/workspace/offline/fia-four-layer-screen-20261020-codex/outputs.pt')
ROOT.mkdir(exist_ok=True, parents=True)
torch.set_num_threads(1)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
report = {'scope': 'Actual binder four-role A/B/reset functionality; no timing',
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'binder_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'fixture_sha256': hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
          'reference_sha256': hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
          'attention_calls_attempted': {'eager': 0, 'capture': 0, 'replay': 0},
          'npu_launched': False, 'quota_bytes': 256 * 2**20,
          'skip_topk_roles': [False, True, True, True], 'checks': []}
outputs = {}
phase = 'eager'


def original(*args, **kwargs):
    raise AssertionError('Unexpected M8 fallback')


def check(values, state, mode):
    for layer, value in enumerate(values):
        actual = value.cpu().clone()
        outputs[f'{mode}_{state}_{layer}'] = actual
        expected = reference[f'eager_{state}_rebuilt_{layer}']
        finite = bool(torch.isfinite(actual).all())
        equal = torch.equal(actual, expected)
        report['checks'].append({'mode': mode, 'state': state, 'layer': layer,
                                 'finite': finite, 'bit_equal_saved_rebuilt_fia': equal})
        assert finite and equal, 'Binder versus saved rebuilt FIA mismatch'


try:
    assert report['binder_sha256'] == '47503e3cd1b730cfcc3b21f27f8d1709f2bcf003cb72d3537d383d65df3417e6'
    assert report['fixture_sha256'] == '96c8c5aaa295d5d6755fe65e611b9b801bc5207f338dc33c8ebffb599f0c56ec'
    assert report['reference_sha256'] == '14efda2ee619297f9b61458b2e3b39549bbabf55aa4481ce3a808a975163ac31'
    (ROOT / 'cann_attention.py').write_bytes(SOURCE.read_bytes())
    spec = importlib.util.spec_from_file_location('actual_binder', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case = torch.load(FIXTURE, map_location='cpu', weights_only=True)
    reference = torch.load(REFERENCE, map_location='cpu', weights_only=True)
    report['custom_opp_path'] = os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')
    assert '/sfa-c1-' not in report['custom_opp_path'] and '/bounds/' not in report['custom_opp_path']
    import torch_npu

    torch_npu.npu.set_device(0)
    total = torch_npu.npu.get_device_properties(0).total_memory
    torch_npu.npu.set_per_process_memory_fraction(256 * 2**20 / total)
    report['fia_schema'] = str(torch.ops.npu.npu_fused_infer_attention_score_v2.default._schema)
    device = torch.device('npu', 0)
    layers = [{'q': item['q'].to(device), 'qr': item['qr'].to(device),
               'kv': tuple(value.to(device) for value in item['kv'])}
              for item in case['layers']]
    assert len({item['kv'][0].data_ptr() for item in layers}) == 4
    indices = case['ids_A'].to(device).view(8, 1, 2048)
    metadata = SimpleNamespace(slot_mapping=(case['thresholds_A'] - 1).to(device))
    query_lengths = torch.tensor([8], dtype=torch.int32, device=device)
    key_lengths = torch.tensor([8200], dtype=torch.int32, device=device)
    binder = module.SharedKVAttention(8, 9344, device)
    implementations = [SimpleNamespace(skip_topk=role, scale=1 / 16,
                                       _execute_sparse_flash_attention_process=original)
                       for role in report['skip_topk_roles']]
    for impl in implementations:
        binder.bind(impl)
    stream = torch_npu.npu.Stream()

    def set_state(state):
        suffix = 'B' if state == 'B' else 'A'
        indices.copy_(case['ids_' + suffix].view(8, 1, 2048))
        metadata.slot_mapping.copy_(case['thresholds_' + suffix] - 1)
        key_lengths.copy_(torch.tensor([case['written_' + suffix]], dtype=torch.int32))
        stream.synchronize()

    def group():
        values = []
        for layer, impl in enumerate(implementations):
            item = layers[layer]
            report['attention_calls_attempted'][phase] += 1
            report['last_started'] = f'{phase}_layer{layer}'
            values.append(impl._execute_sparse_flash_attention_process(
                item['q'], item['qr'], item['kv'], indices, metadata,
                query_lengths, key_lengths, block_table=None))
        return values

    with torch.inference_mode(), torch_npu.npu.stream(stream):
        report['npu_launched'] = True
        for state in ['A', 'B', 'reset']:
            set_state(state)
            check(group(), state, 'eager')
        phase = 'capture'
        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph, stream=stream):
            graph_outputs = group()
        stream.synchronize()
        for state in ['A', 'B', 'reset']:
            set_state(state)
            for value in graph_outputs:
                value.fill_(float('nan'))
            report['last_started'] = f'replay_{state}'
            report['attention_calls_attempted']['replay'] += 4
            graph.replay()
            stream.synchronize()
            check(graph_outputs, state, 'graph')
        assert sum(report['attention_calls_attempted'].values()) == 28
        report['status'] = 'binder_functionality_passed'
except Exception:
    report['status'] = 'first_error_stopped'
    report['exception'] = traceback.format_exc()
finally:
    torch.save(outputs, ROOT / 'outputs.pt')
    report['outputs_sha256'] = hashlib.sha256((ROOT / 'outputs.pt').read_bytes()).hexdigest()
    if report['npu_launched']:
        report['peak_allocated_bytes'] = torch_npu.npu.max_memory_allocated()
        report['peak_reserved_bytes'] = torch_npu.npu.max_memory_reserved()
    libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if any(part in line for part in ['optiling', 'opmaster', 'opapi'])})
    report['loaded_libraries'] = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                  for path in libraries if Path(path).is_file()}
    (ROOT / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
if report['status'] == 'first_error_stopped':
    sys.exit(1)
