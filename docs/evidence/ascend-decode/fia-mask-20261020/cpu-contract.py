"""FP64 gather-mask semantics on a frozen MLA fixture; CPU only."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import torch

torch.set_num_threads(1)


def attend(query, query_rope, latent, rope, indices, thresholds):
    invalid = (indices < 0) | (indices >= thresholds[:, None])
    safe_indices = indices.masked_fill(invalid, 0)
    keys = latent[safe_indices]
    key_rope = rope[safe_indices]
    scores = query @ keys.transpose(-1, -2)
    scores += query_rope @ key_rope.transpose(-1, -2)
    scores *= 1 / 16
    scores.masked_fill_(invalid[:, None, :], float('-inf'))
    return scores.softmax(-1) @ keys, invalid


def main():
    source = Path('/workspace/offline/sfa-c1-bounds-20261019-codex/'
                  'cpu-fixture/input.pt')
    case = torch.load(source, map_location='cpu', weights_only=True)
    table = case['block_table'][0].long()
    latent = case['key'][table].reshape(-1, 512).double()
    rope = case['key_rope'][table].reshape(-1, 64).double()
    query, query_rope = case['query'].double(), case['query_rope'].double()
    indices = case['sparse_indices'][:, 0].long().clone()
    width = query.shape[0]
    thresholds = (case['actual_seq_lengths_kv'][0].long()
                  - case['actual_seq_lengths_query'][0].long()
                  + torch.arange(width) + 1)
    for row in range(width):
        indices[row, 0] = thresholds[row]
        indices[row, 1] = thresholds[row] + 1
        indices[row, -(row + 1):] = -1
    masked, invalid = attend(
        query, query_rope, latent, rope, indices, thresholds,
    )
    filtered = []
    for row in range(width):
        selected = indices[row, ~invalid[row]]
        keys, key_rope = latent[selected], rope[selected]
        scores = (query[row] @ keys.T + query_rope[row] @ key_rope.T) / 16
        filtered.append(scores.softmax(-1) @ keys)
    filtered = torch.stack(filtered)
    max_abs = (masked - filtered).abs().max().item()
    torch.testing.assert_close(masked, filtered, atol=1e-12, rtol=1e-12)

    perturbations = {}
    future = indices[0, :2]
    for name, value in [('finite_large', 1e6), ('nan', float('nan'))]:
        changed_latent, changed_rope = latent.clone(), rope.clone()
        changed_latent[future] = value
        changed_rope[future] = value
        changed, _ = attend(
            query[:1], query_rope[:1], changed_latent, changed_rope,
            indices[:1], thresholds[:1],
        )
        equal = torch.equal(masked[:1], changed)
        assert equal
        perturbations[name] = {'row0_bit_equal': equal}
    mask_shape = list(invalid.view(width, 1, 1, 2048).shape)
    result = {
        'scope': 'CPU FP64 algebra and counterfactual only; no NPU call.',
        'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'input_path': str(source),
        'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'shapes': {'query': list(query.shape), 'query_rope': list(query_rope.shape),
                   'indices': list(indices.shape), 'bool_mask': mask_shape},
        'thresholds': thresholds.tolist(),
        'injected_future_tokens_per_row': 2,
        'invalid_counts': invalid.sum(-1).tolist(),
        'masked_vs_filtered_fp64_max_abs': max_abs,
        'fp64_algebra_tolerance': {'atol': 1e-12, 'rtol': 1e-12,
                                  'scope': 'CPU identity only; no decoder threshold change'},
        'future_cache_perturbation': perturbations,
        'minimal_counterexample': {
            'q': 0, 'ids': [0, 2, 3, -1], 'causal_length': 3,
            'values_at_ids0_2_3': [0, 2, 100],
            'unmasked_negative_clamped_mean': 25.5,
            'masked_filtered_mean': 1.0,
            'unmasked_if_future_value1000': 250.5,
        },
        'contracts': [
            'Construct invalid from original token IDs before rewriting gather addresses.',
            'Replace every invalid ID with a known initialized causal slot, such as0; physical-range clamp alone does not exclude allocated future slots or NaNs.',
            'Require trailing -1 padding to match SFA termination; an interior -1 followed by valid IDs changes semantics under elementwise masking.',
            'Keep gathered FIA actual_seq_kvlen at2048; valid entries are not necessarily a contiguous prefix. The token-domain threshold is used only to build the mask.',
            'Require at least one valid token per row here. An all-masked row needs its explicit zero-output contract checked separately.',
        ],
    }
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
