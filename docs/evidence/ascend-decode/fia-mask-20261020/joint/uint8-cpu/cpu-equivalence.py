"""Check frozen bitmap representations on CPU; perform no device operation."""

from pathlib import Path
import hashlib
import json
import torch

root = Path('/workspace/offline/fia-bitmap-cpu-20261020-codex')
root.mkdir(parents=True, exist_ok=True)
fixture = Path('/workspace/offline/fia-joint-m-screen-20261020-codex/input.pt')
state = torch.load(fixture, map_location='cpu', weights_only=True)
ids = state['sparse_indices'][:, 0].long()
capacity = int(state['latent'].shape[0])
width = int(ids.shape[0])
assert (width, capacity) == (8, 9344)
assert all(ids[r].unique().numel() == 2048 for r in range(width))
assert bool(((ids >= 0) & (ids < capacity)).all())
columns = torch.arange(capacity, dtype=torch.int32).view(1, -1)
threshold_a = state['state_cpu']['A']['thresholds']
report = {
    'scope': ('Server CPU equality only; CPU scatter uses long index and '
              'does not validate NPU int32 dispatch or dtype-view support'),
    'fixture': str(fixture),
    'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
    'index_dtype_original': str(state['sparse_indices'].dtype),
    'cpu_scatter_index_dtype': str(ids.dtype),
    'shape': [width, capacity], 'topk': 2048,
    'torch_version': torch.__version__, 'npu_launched': False, 'cases': [],
}
for name, thresholds in [('A', threshold_a),
                         ('causal_threshold_minus_one', threshold_a - 1)]:
    original = torch.ones((width, capacity), dtype=torch.bool)
    original.scatter_(1, ids, False)
    original = original | (columns >= thresholds.view(width, 1))
    full_u8 = torch.ones((width, capacity), dtype=torch.uint8)
    full_u8.scatter_(1, ids, False)
    full_u8 = full_u8 | (columns >= thresholds.view(width, 1))
    updates = (ids >= thresholds.view(width, 1)).to(torch.uint8)
    selected_u8 = torch.ones((width, capacity), dtype=torch.uint8)
    selected_u8.scatter_(1, ids, updates)
    reference_bytes = original.view(torch.uint8)
    assert torch.equal(reference_bytes, full_u8)
    assert torch.equal(reference_bytes, selected_u8)
    assert torch.equal(full_u8.view(torch.bool), original)
    assert full_u8.view(torch.bool).data_ptr() == full_u8.data_ptr()
    report['cases'].append({
        'name': name, 'thresholds': thresholds.tolist(),
        'full_u8_byte_equal': True,
        'selected_updates_u8_byte_equal': True,
        'full_u8_change': 'Only bitmap ones dtype bool to uint8; scatter False, GE bool, OR unchanged',
        'bool_view_alias_cpu': True,
        'excluded_per_row': original.sum(dim=1).tolist(),
        'selected_excluded_per_row': updates.sum(dim=1).tolist(),
        'mask_sha256': hashlib.sha256(reference_bytes.numpy().tobytes()).hexdigest(),
        'output_values': full_u8.unique().tolist(),
    })
report['perturb_changes_mask'] = (
    report['cases'][0]['mask_sha256'] != report['cases'][1]['mask_sha256'])
(root / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
