"""CPU equivalence for unique full top-k selection and a joint-M dense mask."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import torch

torch.set_num_threads(1)
source = Path('/workspace/offline/sfa-c1-bounds-20261019-codex/cpu-fixture/input.pt')
case = torch.load(source, map_location='cpu', weights_only=True)
query, query_rope = case['query'].double(), case['query_rope'].double()
indices = case['sparse_indices'][:, 0].long()
width, heads, _ = query.shape
capacity = 9344
written = int(case['actual_seq_lengths_kv'][0])
table = case['block_table'][0].long()
logical = case['key'][table].reshape(-1, 512)
logical_rope = case['key_rope'][table].reshape(-1, 64)
latent = torch.zeros(capacity, 512, dtype=torch.float64)
rope = torch.zeros(capacity, 64, dtype=torch.float64)
latent[:written].copy_(logical[:written])
rope[:written].copy_(logical_rope[:written])
thresholds = written - width + torch.arange(width) + 1
for row in range(width):
    assert indices[row].unique().numel() == 2048
    assert bool(((indices[row] >= 0) & (indices[row] < thresholds[row])).all())
mask = torch.ones(width, capacity, dtype=torch.bool)
mask.scatter_(1, indices, False)


def dense(keys, key_rope):
    scores = torch.einsum('mhd,nd->mhn', query, keys)
    scores += torch.einsum('mhr,nr->mhn', query_rope, key_rope)
    scores /= 16
    scores.masked_fill_(mask[:, None, :], float('-inf'))
    return scores.softmax(-1) @ keys


reference = []
for row in range(width):
    keys, key_rope = latent[indices[row]], rope[indices[row]]
    scores = (query[row] @ keys.T + query_rope[row] @ key_rope.T) / 16
    reference.append(scores.softmax(-1) @ keys)
reference = torch.stack(reference)
output = dense(latent, rope)
torch.testing.assert_close(output, reference, atol=1e-12, rtol=1e-12)
changed_latent, changed_rope = latent.clone(), rope.clone()
changed_latent[written].fill_(1e6)
changed_rope[written].fill_(-1e6)
finite_changed = dense(changed_latent, changed_rope)
assert torch.equal(output, finite_changed)
changed_latent[written].fill_(float('nan'))
changed_rope[written].fill_(float('nan'))
nan_changed = dense(changed_latent, changed_rope)
assert not torch.isfinite(nan_changed).all()
changed_latent[written].zero_()
changed_rope[written].zero_()
reset = dense(changed_latent, changed_rope)
assert torch.equal(output, reset)
result = {
    'scope': 'CPU FP64 identity on one frozen M8 fixture; no joint-M NPU execution.',
    'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'shape': {'M': width, 'heads': heads, 'D': 512, 'R': 64,
              'selected_tokens_per_query': 2048, 'written_tokens': written,
              'capacity': capacity, 'bool_mask': [1, 1, width, capacity]},
    'masked_dense_vs_gathered_fp64_max_abs': (output - reference).abs().max().item(),
    'fp64_algebra_tolerance': {'atol': 1e-12, 'rtol': 1e-12,
                              'scope': 'CPU identity; no decoder threshold change'},
    'unselected_finite_slot_perturbation_bit_equal': torch.equal(output, finite_changed),
    'unselected_nan_slot_nonfinite_outputs': int((~torch.isfinite(nan_changed)).sum()),
    'zero_tail_reset_bit_equal': torch.equal(output, reset),
    'byte_arithmetic': {
        'mask_bool_bytes': width * capacity,
        'avoided_per_query_gather_buffers_bf16_bytes': width * 2048 * 576 * 2,
        'existing_full_cache_latent_rope_bf16_bytes': capacity * 576 * 2,
        'dense_to_selected_qk_and_pv_work_ratio': capacity / 2048,
        'scope': 'Shapes and work counts only, not HBM traffic or time.',
    },
    'unique_selection_requirement': {
        'duplicate_counterexample': {'ids': [0, 0, 1], 'q': 0, 'values': [0, 2],
                                     'sparse_mean': 2 / 3, 'set_mask_dense_mean': 1.0},
    },
}
destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)
(destination / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
