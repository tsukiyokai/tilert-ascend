from pathlib import Path
import hashlib
import json
import torch
root = Path('/workspace/offline/fia-bitmap-cpu-20261020-codex')
fixture = Path('/workspace/offline/fia-joint-m-screen-20261020-codex/input.pt')
case = torch.load(fixture, map_location='cpu', weights_only=True)
ids = case['sparse_indices'][:, 0].long()
columns = torch.arange(9344, dtype=torch.int32).view(1, -1)
results = []
for perturb in [0, -1]:
    threshold = case['state_cpu']['A']['thresholds'] + perturb
    causal = columns >= threshold.view(8, 1)
    reference = torch.ones((8, 9344), dtype=torch.bool)
    reference.scatter_(1, ids, False)
    reference = reference | causal
    bitmap = torch.ones((8, 9344), dtype=torch.uint8)
    bitmap.scatter_(1, ids, False)
    view = bitmap.view(torch.bool)
    actual = view | causal
    assert view.data_ptr() == bitmap.data_ptr()
    assert torch.equal(actual, reference)
    results.append({'perturb': perturb, 'bit_equal': True,
                    'view_shares_data_ptr': True,
                    'actual_dtype': str(actual.dtype),
                    'mixed_or_dtype': str((bitmap | causal).dtype),
                    'mask_sha256': hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()).hexdigest()})
report = {'scope': 'Server CPU metadata-view and mask math only; no NPU launch or capture assertion',
          'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
          'cases': results}
(root / 'view-before-or-result.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
