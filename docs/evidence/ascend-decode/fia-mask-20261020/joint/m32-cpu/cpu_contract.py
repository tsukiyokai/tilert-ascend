"""Server CPU contracts for the actual shared-FIA helpers at M32."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import torch


ROOT = Path('/workspace/offline/fia-m32-contract-20261020-codex')
INPUT = Path('/workspace/offline/fia-four-layer-contract-20261020-codex/fixture.pt')
HELPER = Path('/workspace/offline/shared-fia-integration-20261020/tree/tilert/models/glm_5_2_ascend/cann_attention.py')
EXPECTED_HELPER = '47503e3cd1b730cfcc3b21f27f8d1709f2bcf003cb72d3537d383d65df3417e6'
WIDTH, CAPACITY, WRITTEN = 32, 9344, 8224


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


torch.set_num_threads(1)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
ROOT.mkdir(parents=True, exist_ok=True)
assert file_sha(HELPER) == EXPECTED_HELPER
shutil.copyfile(HELPER, ROOT / 'cann_attention.py')
spec = importlib.util.spec_from_file_location('actual_cann_attention', ROOT / 'cann_attention.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
old = torch.load(INPUT, map_location='cpu', weights_only=True)
assert file_sha(INPUT) == '96c8c5aaa295d5d6755fe65e611b9b801bc5207f338dc33c8ebffb599f0c56ec'
columns = torch.arange(CAPACITY, dtype=torch.int32).view(1, -1)
thresholds_a = torch.arange(8193, WRITTEN + 1, dtype=torch.int32)
thresholds_b = thresholds_a - 1
ids_a, ids_b = [], []
for row in range(WIDTH):
    threshold = int(thresholds_a[row])
    # Keep 2047 prior IDs and include this row's current token explicitly.
    prior = [token for token in old['ids_A'][row % 8].tolist()
             if token < threshold - 1]
    selected_a = prior[:2047] + [threshold - 1]
    assert len(selected_a) == len(set(selected_a)) == 2048
    selected = set(selected_a)
    new_past = next(token for token in range(threshold - 1) if token not in selected)
    # B changes membership and makes the last selected ID future at once.
    selected_b = selected_a[:2046] + [new_past, threshold - 1]
    assert len(selected_b) == len(set(selected_b)) == 2048
    ids_a.append(selected_a)
    ids_b.append(selected_b)
ids_a = torch.tensor(ids_a, dtype=torch.int32)
ids_b = torch.tensor(ids_b, dtype=torch.int32)
layers = []
for source in old['layers']:
    latent = torch.zeros_like(source['kv'][0])
    rope = torch.zeros_like(source['kv'][1])
    for target, previous in zip((latent, rope), source['kv']):
        target[:8200] = previous[:8200]
        target[8200:WRITTEN] = previous[8176:8200]
    layers.append({
        'q': torch.cat([source['q'].roll(group, dims=-1) for group in range(4)]),
        'qr': torch.cat([source['qr'].roll(group, dims=-1) for group in range(4)]),
        'kv': (latent, rope),
    })
for layer in layers:
    assert all(bool(torch.isfinite(value).all())
               for value in [layer['q'], layer['qr'], *layer['kv']])
    assert all(int(value[WRITTEN:].count_nonzero()) == 0 for value in layer['kv'])
for name in ('q', 'qr'):
    assert len({layer[name].data_ptr() for layer in layers}) == 4
    assert len({tensor_sha(layer[name]) for layer in layers}) == 4
for index in (0, 1):
    assert len({layer['kv'][index].data_ptr() for layer in layers}) == 4
    assert len({tensor_sha(layer['kv'][index]) for layer in layers}) == 4
assert bool(((ids_a >= 0) & (ids_a < CAPACITY)).all())
assert bool(((ids_b >= 0) & (ids_b < CAPACITY)).all())
fixture = {'layers': layers, 'ids_A': ids_a, 'ids_B': ids_b,
           'thresholds_A': thresholds_a, 'thresholds_B': thresholds_b,
           'written_A': WRITTEN, 'written_B': WRITTEN - 1,
           'zero_tail_begin': WRITTEN}
torch.save(fixture, ROOT / 'fixture.pt')
records = []
layer_ids = {layer['q'].data_ptr(): index for index, layer in enumerate(layers)}


def record_fia(query, key, value, **kwargs):
    index = layer_ids[query.data_ptr()]
    layer = layers[index]
    assert query.shape == (1, WIDTH, 4, 512)
    assert key.shape == (1, CAPACITY, 1, 512) and value is key
    assert key.data_ptr() == layer['kv'][0].data_ptr()
    assert kwargs['query_rope'].shape == (1, WIDTH, 4, 64)
    assert kwargs['query_rope'].data_ptr() == layer['qr'].data_ptr()
    assert kwargs['key_rope'].shape == (1, CAPACITY, 1, 64)
    assert kwargs['key_rope'].data_ptr() == layer['kv'][1].data_ptr()
    mask = kwargs['atten_mask']
    assert mask.shape == (1, 1, WIDTH, CAPACITY) and mask.dtype == torch.bool
    assert kwargs['input_layout'] == 'BSND' and kwargs['sparse_mode'] == 0
    assert kwargs['actual_seq_qlen'] == [WIDTH]
    assert kwargs['actual_seq_kvlen'] == [CAPACITY]
    assert kwargs['num_query_heads'] == 4 and kwargs['num_key_value_heads'] == 1
    assert kwargs['softmax_scale'] == 1 / 16
    records.append({'layer': index, 'mask_sha256': tensor_sha(mask),
                    'mask_ptr': mask.data_ptr()})
    return torch.zeros_like(query), None


def unexpected_fallback(*args, **kwargs):
    raise AssertionError('The matching-width call must use the actual binder')


# Record the real helper call. No device backend or CANN math is substituted.
sys.modules['torch_npu'] = SimpleNamespace(npu_fused_infer_attention_score_v2=record_fia)
binder = helper.SharedKVAttention(WIDTH, CAPACITY, 'cpu')
impls = [SimpleNamespace(skip_topk=(index != 0), scale=1 / 16,
                         _execute_sparse_flash_attention_process=unexpected_fallback)
         for index in range(4)]
for impl in impls:
    binder.bind(impl)
ids = ids_a.clone()
slots = thresholds_a.clone() - 1
metadata = SimpleNamespace(slot_mapping=slots)
states, masks = [], {}
for name, state_ids, thresholds in [('A', ids_a, thresholds_a),
                                     ('B', ids_b, thresholds_b),
                                     ('reset', ids_a, thresholds_a)]:
    ids.copy_(state_ids)
    slots.copy_(thresholds - 1)
    expected_rows = []
    for row_ids, threshold in zip(ids, thresholds):
        selected_set = set(row_ids.tolist())
        expected_rows.append([token not in selected_set or token >= int(threshold)
                              for token in range(CAPACITY)])
    expected = torch.tensor(expected_rows, dtype=torch.bool)
    mask = helper.selection_mask(ids, columns >= thresholds.view(WIDTH, 1))
    assert torch.equal(mask, expected)
    masks[name] = mask
    records.clear()
    for layer, impl in zip(layers, impls):
        impl._execute_sparse_flash_attention_process(
            layer['q'], layer['qr'], layer['kv'], ids.view(WIDTH, 1, 2048),
            metadata, [WIDTH], [CAPACITY])
    shared = list(records)
    assert torch.equal(binder.mask, expected)
    assert len(shared) == 4 and len({record['mask_ptr'] for record in shared}) == 1
    records.clear()
    for layer in layers:
        fresh = helper.selection_mask(ids, columns >= thresholds.view(WIDTH, 1))
        helper.shared_kv_fia(layer['q'], layer['qr'], layer['kv'], fresh, 1 / 16)
    assert [(item['layer'], item['mask_sha256']) for item in shared] == [
        (item['layer'], item['mask_sha256']) for item in records]
    states.append({'state': name, 'mask_sha256': tensor_sha(mask),
                   'valid_selected_per_row': (~mask).sum(1).tolist(),
                   'all_bits_match_independent_set': True,
                   'actual_binder_matches_four_rebuild_calls': True,
                   'one_mask_pointer_four_distinct_q_kv': True})
stale_ids = helper.selection_mask(ids_a, columns >= thresholds_b.view(WIDTH, 1))
stale_thresholds = helper.selection_mask(ids_b, columns >= thresholds_a.view(WIDTH, 1))
stale_ids_counts = (stale_ids != masks['B']).sum(1)
stale_threshold_counts = (stale_thresholds != masks['B']).sum(1)
assert bool((stale_ids_counts > 0).all()) and bool((stale_threshold_counts > 0).all())
assert torch.equal(masks['A'], masks['reset'])
assert bool((masks['A'] != masks['B']).any(1).all())

# A mathematical FP64 equality check for one query; this is not CANN emulation.
row = WIDTH - 1
layer = layers[0]
query = layer['q'][row].double()
query_rope = layer['qr'][row].double()
latent, rope = [value.double() for value in layer['kv']]
scores = (query @ latent.T + query_rope @ rope.T) / 16
probability = scores.masked_fill(masks['B'][row].view(1, -1), -torch.inf).softmax(-1)
dense = probability @ latent
selected = ids_b[row][ids_b[row] < thresholds_b[row]].long()
selected_scores = (query @ latent[selected].T + query_rope @ rope[selected].T) / 16
selected_output = selected_scores.softmax(-1) @ latent[selected]
torch.testing.assert_close(dense, selected_output, rtol=1e-12, atol=1e-13)
future = int(thresholds_b[row])
assert future in ids_b[row].tolist() and bool(masks['B'][row, future])
perturbed_latent = latent.clone()
perturbed_rope = rope.clone()
perturbed_latent[future] += 7
perturbed_rope[future] += 5
perturbed_scores = (query @ perturbed_latent.T + query_rope @ perturbed_rope.T) / 16
perturbed_probability = perturbed_scores.masked_fill(
    masks['B'][row].view(1, -1), -torch.inf).softmax(-1)
perturbed = perturbed_probability @ perturbed_latent
assert torch.equal(dense, perturbed)
report = {
    'scope': 'CPU M32 actual-helper/binder API and set semantics; FP64 algebra, not CANN emulation',
    'npu_launched': False, 'torch_version': torch.__version__,
    'cpu_source_sha256': file_sha(Path(__file__)),
    'helper_source_sha256': file_sha(ROOT / 'cann_attention.py'),
    'input_sha256': file_sha(INPUT), 'fixture_sha256': file_sha(ROOT / 'fixture.pt'),
    'width': WIDTH, 'capacity': CAPACITY, 'layers': 4,
    'derivation': 'M8 Q/QR expanded in four channel-roll groups; each independent KV keeps first8200 rows, copies rows8176:8200 into8200:8224, zeros8224:9344; A selected lists replace one old ID with each query current slot; B changes selected membership and decrements every causal threshold',
    'unique_2048_in_range_each_state': True,
    'all_query_kv_finite': True, 'zero_tail_begin': WRITTEN,
    'logical_B_tail': 'row8223 is initialized finite history from A and masked; never uninitialized',
    'states': states,
    'stale_indices_counterexample_rows': int((stale_ids_counts > 0).sum()),
    'stale_indices_changed_bits_per_row': stale_ids_counts.tolist(),
    'stale_threshold_counterexample_rows': int((stale_threshold_counts > 0).sum()),
    'stale_threshold_changed_bits_per_row': stale_threshold_counts.tolist(),
    'fp64': {'layer': 0, 'query_row': row, 'heads': 4, 'state': 'B',
             'selected_valid_rows': int(selected.numel()),
             'dense_vs_selected_max_abs': float((dense - selected_output).abs().max()),
             'future_selected_id': future, 'finite_future_perturb_bit_equal': True},
    'four_layer_q_kv_payload_bytes': sum(value.numel() * value.element_size()
        for layer in layers for value in [layer['q'], layer['qr'], *layer['kv']]),
    'fixture_bytes': (ROOT / 'fixture.pt').stat().st_size,
    'limits': ['Production shared_fia remains M8-only; this parameterized helper contract does not enable M32',
               'No device API, numerical CANN comparison, graph memory or timing has been tested',
               'This synthetic extension preserves CPU contracts and does not replay a recorded M32 model state'],
}
(ROOT / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
