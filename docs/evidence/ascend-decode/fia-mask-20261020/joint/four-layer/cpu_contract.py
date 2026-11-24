"""CPU mask and call-contract checks of the frozen production helper source."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


ROOT = Path('/workspace/offline/fia-four-layer-contract-20261020-codex')
INPUT = Path('/workspace/offline/fia-joint-m-uint8-view-screen-20261020-codex/input.pt')
SOURCE = ROOT / 'cann_attention.py'
torch.set_num_threads(1)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
spec = importlib.util.spec_from_file_location('actual_cann_attention', SOURCE)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
case = torch.load(INPUT, map_location='cpu', weights_only=True)
width, capacity, written = 8, 9344, 8200
columns = torch.arange(capacity, dtype=torch.int32).view(1, -1)
ids_a = case['sparse_indices'][:, 0].clone()
thresholds_a = case['state_cpu']['A']['thresholds'].clone()
thresholds_b = thresholds_a - 1
ids_b = ids_a.clone()
for row in range(width):
    threshold = int(thresholds_b[row])
    original = ids_a[row].tolist()
    selected = set(original)
    valid = [token for token in original if token < threshold]
    new_past = next(token for token in range(threshold) if token not in selected)
    ids_b[row] = torch.tensor(valid[:2046] + [new_past, threshold], dtype=torch.int32)
    assert ids_b[row].unique().numel() == 2048
    assert ids_b[row, -1] == threshold
    assert not torch.equal(ids_a[row], ids_b[row])
layers = []
for layer in range(4):
    latent = torch.zeros_like(case['latent'])
    rope = torch.zeros_like(case['rope'])
    latent[:written] = case['latent'][:written].roll(layer, dims=0)
    rope[:written] = case['rope'][:written].roll(layer, dims=0)
    layers.append({'q': case['query'].roll(layer, dims=-1).clone(),
                   'qr': case['query_rope'].roll(layer, dims=-1).clone(),
                   'kv': (latent, rope)})
for layer in layers:
    assert all(bool(torch.isfinite(tensor).all()) for tensor in layer['kv'])
    assert all(int(tensor[written:].count_nonzero()) == 0 for tensor in layer['kv'])
torch.save({'layers': layers, 'ids_A': ids_a, 'ids_B': ids_b,
            'thresholds_A': thresholds_a, 'thresholds_B': thresholds_b,
            'written_A': written, 'written_B': written - 1}, ROOT / 'fixture.pt')


def tensor_sha(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


layer_ids = {layer['q'].data_ptr(): i for i, layer in enumerate(layers)}
assert len(layer_ids) == 4
assert len({layer['kv'][0].data_ptr() for layer in layers}) == 4
assert len({tensor_sha(layer['kv'][0]) for layer in layers}) == 4
assert len({tensor_sha(layer['q']) for layer in layers}) == 4
records = []


def record_fia(query, key, value, **kwargs):
    layer = layer_ids[query.data_ptr()]
    expected = layers[layer]
    assert query.shape == (1, width, 4, 512)
    assert key.shape == (1, capacity, 1, 512)
    assert key.data_ptr() == expected['kv'][0].data_ptr()
    assert value is key
    assert kwargs['query_rope'].data_ptr() == expected['qr'].data_ptr()
    assert kwargs['key_rope'].data_ptr() == expected['kv'][1].data_ptr()
    assert kwargs['atten_mask'].shape == (1, 1, width, capacity)
    assert kwargs['atten_mask'].dtype == torch.bool
    assert kwargs['input_layout'] == 'BSND' and kwargs['sparse_mode'] == 0
    assert kwargs['actual_seq_qlen'] == [width]
    assert kwargs['actual_seq_kvlen'] == [capacity]
    assert kwargs['num_query_heads'] == 4 and kwargs['num_key_value_heads'] == 1
    assert kwargs['softmax_scale'] == 1 / 16
    records.append({'layer': layer, 'query_ptr': query.data_ptr(),
                    'key_ptr': key.data_ptr(),
                    'mask_sha256': tensor_sha(kwargs['atten_mask']),
                    'mask_ptr': kwargs['atten_mask'].data_ptr()})
    return torch.zeros_like(query), None


# This records the real helper's call arguments; it does not emulate CANN math.
sys.modules['torch_npu'] = SimpleNamespace(
    npu_fused_infer_attention_score_v2=record_fia)
ids = ids_a.clone()
thresholds = thresholds_a.clone()
states = []
mask_a = helper.selection_mask(ids_a, columns >= thresholds_a.view(width, 1))
for name, state_ids, state_thresholds in [('A', ids_a, thresholds_a),
                                         ('B', ids_b, thresholds_b),
                                         ('reset', ids_a, thresholds_a)]:
    ids.copy_(state_ids)
    thresholds.copy_(state_thresholds)
    causal = columns >= thresholds.view(width, 1)
    mask = helper.selection_mask(ids, causal)
    for row in range(width):
        selected = set(ids[row].tolist())
        expected = torch.tensor([token not in selected or token >= int(thresholds[row])
                                 for token in range(capacity)], dtype=torch.bool)
        assert torch.equal(mask[row], expected)
    records.clear()
    for layer in layers:
        helper.shared_kv_fia(layer['q'], layer['qr'], layer['kv'], mask, 1 / 16)
    shared = list(records)
    assert len({record['mask_ptr'] for record in shared}) == 1
    records.clear()
    rebuilt = []
    for layer in layers:
        fresh = helper.selection_mask(ids, columns >= thresholds.view(width, 1))
        rebuilt.append(fresh)
        helper.shared_kv_fia(layer['q'], layer['qr'], layer['kv'], fresh, 1 / 16)
    repeated = list(records)
    assert [{k: v for k, v in item.items() if k != 'mask_ptr'} for item in shared] == [
        {k: v for k, v in item.items() if k != 'mask_ptr'} for item in repeated]
    states.append({'name': name, 'mask_sha256': tensor_sha(mask),
                   'excluded_per_row': mask.sum(1).tolist(),
                   'call_arguments_match_four_rebuilds': True,
                   'shared_mask_same_ptr_four_consumers': True})
    if name == 'B':
        stale_threshold = helper.selection_mask(ids_b, columns >= thresholds_a.view(width, 1))
        stale_indices = helper.selection_mask(ids_a, causal)
        assert all(not bool(stale_threshold[row, int(thresholds_b[row])])
                   and bool(mask[row, int(thresholds_b[row])]) for row in range(width))
        assert bool((mask != stale_indices).any(1).all())
        assert bool((mask != mask_a).any(1).all())
assert states[0]['mask_sha256'] == states[2]['mask_sha256']
assert states[0]['mask_sha256'] != states[1]['mask_sha256']
report = {
    'scope': 'CPU mask semantics and real-helper API arguments, no CANN emulation',
    'npu_launched': False,
    'helper_source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'cpu_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'input_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest(),
    'fixture_sha256': hashlib.sha256((ROOT / 'fixture.pt').read_bytes()).hexdigest(),
    'finite_cache_with_zero_unwritten_tail': True,
    'states': states, 'stale_threshold_counterexamples': 8,
    'stale_indices_counterexample_rows': 8,
    'layers': [{'query_sha256': tensor_sha(layer['q']),
                'latent_sha256': tensor_sha(layer['kv'][0]),
                'rope_sha256': tensor_sha(layer['kv'][1])} for layer in layers],
    'four_layer_q_kv_payload_bytes': sum(
        tensor.numel() * tensor.element_size()
        for layer in layers for tensor in [layer['q'], layer['qr'], *layer['kv']]),
    'limit': 'Device bitwise equivalence, graph reset, memory and timing remain untested',
}
(ROOT / 'cpu-result.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
