"""Replay the mask-sharing invariant on CPU using the installed role rule."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path('/workspace/offline/fia-mask-sharing-source-20261020-codex')
ROOT.mkdir(parents=True, exist_ok=True)
SOURCE_ROOT = Path('/workspace/src/vllm-ascend-unified/vllm_ascend')
config_path = Path('/workspace/models/GLM-5.2-w4a8c8/config.json')
config = json.loads(config_path.read_text())
utils_path = SOURCE_ROOT / 'utils.py'
parsed = ast.parse(utils_path.read_text())
function = next(node for node in parsed.body
                if isinstance(node, ast.FunctionDef)
                and node.name == 'should_reuse_topk')
module = ast.Module(body=[function], type_ignores=[])
namespace = {'Any': object}
exec(compile(ast.fix_missing_locations(module), str(utils_path), 'exec'), namespace)
reuse = namespace['should_reuse_topk']
roles = [not reuse(SimpleNamespace(**config), layer) for layer in range(78)]
producers = [layer for layer, producer in enumerate(roles) if producer]
assert producers == [i for i, kind in enumerate(config['indexer_types']) if kind == 'full']
assert producers == [0, 1, *range(2, 78, 4)]
assert all(roles[:3]) and config['first_k_dense_replace'] == 3

def groups_from_roles(flags):
    groups = []
    for layer, producer in enumerate(flags):
        if producer:
            groups.append([])
        assert groups, 'Consumer cannot precede the first producer'
        groups[-1].append(layer)
    return groups

groups = groups_from_roles(roles)
try:
    groups_from_roles([False, *roles[1:]])
except AssertionError:
    rejects_consumer_first = True
else:
    raise AssertionError('Invalid leading consumer was accepted')
fixture = Path('/workspace/offline/fia-joint-m-screen-20261020-codex/input.pt')
case = torch.load(fixture, map_location='cpu', weights_only=True)
base_ids = case['sparse_indices'][:, 0].long()
base_thresholds = case['state_cpu']['A']['thresholds']
columns = torch.arange(9344).view(1, -1)
shared_ids = torch.empty_like(base_ids)
shared_mask = torch.ones((8, 9344), dtype=torch.bool)

def mask(ids, thresholds):
    result = torch.ones((8, 9344), dtype=torch.bool)
    result.scatter_(1, ids, False)
    return result | (columns >= thresholds.view(8, 1))

def digest(value):
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()

states = []
last_previous_forward = None
pointer_cache = None
witnesses = {}
for name, position_delta, selection_delta in [('A', 0, 0), ('B', -1, 7), ('reset', 0, 0)]:
    thresholds = base_thresholds + position_delta
    producer_count = 0
    consumption_count = 0
    hashes = []
    for layer, producer in enumerate(roles):
        if producer:
            ids = (base_ids + 31 * (producer_count + 1) + selection_delta) % 9344
            assert all(ids[row].unique().numel() == 2048 for row in range(8))
            shared_ids.copy_(ids)
            expected = mask(ids, thresholds)
            shared_mask.copy_(expected)
            producer_count += 1
            hashes.append(digest(shared_mask))
            if name == 'A' and pointer_cache is None:
                pointer_cache = expected.clone()
            elif name == 'A' and 'pointer_only_cache' not in witnesses:
                assert not torch.equal(pointer_cache, expected)
                witnesses['pointer_only_cache'] = {'first_bad_layer': layer,
                                                  'same_index_storage': True}
            if name == 'B' and layer == 0:
                assert not torch.equal(last_previous_forward, expected)
                witnesses['reuse_mask_across_forwards'] = {'first_bad_layer': 0,
                                                          'forward': 'B'}
        reference = mask(shared_ids, thresholds)
        assert torch.equal(shared_mask, reference)
        consumption_count += 1
    assert producer_count == 21 and consumption_count == 78
    assert len(set(hashes)) == 21
    states.append({'name': name, 'mask_builds': producer_count,
                   'mask_consumptions': consumption_count,
                   'producer_mask_sha256': hashes})
    last_previous_forward = shared_mask.clone()
assert states[0]['producer_mask_sha256'] == states[2]['producer_mask_sha256']
assert states[0]['producer_mask_sha256'] != states[1]['producer_mask_sha256']
report = {'scope': 'CPU state-machine and bitmap equality only; no model instantiation, NPU import, NPU launch or performance claim',
          'checkpoint_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
          'utils_sha256': hashlib.sha256(utils_path.read_bytes()).hexdigest(),
          'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
          'shape': [8, 9344], 'topk': 2048, 'producers': producers,
          'groups': groups, 'dense_mlp_layers_are_producers': [0, 1, 2],
          'rejects_consumer_before_producer': rejects_consumer_first,
          'states': states, 'counterexamples': witnesses,
          'graph_replay_contract': 'The captured graph must retain every producer mask-build task on every replay; Python cache-hit state cannot suppress captured writes',
          'storage_contract': 'One shared mask is valid only while same-stream forward consumers finish before the next producer overwrites it'}
(ROOT / 'cpu-lifecycle-result.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'states'}, indent=2))
print('states:', [(s['name'], s['mask_builds'], s['mask_consumptions']) for s in states])
