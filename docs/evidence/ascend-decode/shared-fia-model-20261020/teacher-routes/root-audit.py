"""Read the saved route IDs independently; no device work."""
import hashlib
import json
import os
from pathlib import Path
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import torch

root = Path('/workspace/experiments/teacher-input-router-20261020')
torch.set_num_threads(1)
assert json.loads((root / 'routes-retry01-exit.json').read_text())['returncode'] == 0
record = json.loads((root / 'routes-result.json').read_text())
assert record['status'] == 'pass'
assert record['binding'] == json.loads((root / 'norm-result.json').read_text())['binding']
path = root / 'teacher-routes.pt'
assert hashlib.sha256(path.read_bytes()).hexdigest() == 'cad3492150b53d5d631f0bd96d72c3e1fd7e69614c78d8cccd513832609eb348'
raw = torch.load(path, map_location='cpu', weights_only=True)['routes'].tolist()
assert len(raw) == 75
result = {}
for width in (8, 16, 32):
    unions, hottest, hot = [], [], []
    for layer in raw:
        assert len(layer) == 32 and all(len(set(row)) == 8 and min(row) >= 0 and max(row) < 256 for row in layer)
        for offset in range(0, 32, width):
            counts = {}
            for row in layer[offset:offset + width]:
                for expert in row:
                    counts[expert] = counts.get(expert, 0) + 1
            unions.append(len(counts))
            hottest.append(max(counts.values()))
            hot.append(sum(n > 16 for n in counts.values()))
    result[str(width)] = {'windows': len(unions), 'union_mean': sum(unions) / len(unions), 'le16_windows': sum(n <= 16 for n in hottest), 'active_total': sum(unions), 'hot_total': sum(hot)}
result['npu_initialized'] = torch.npu.is_initialized() if hasattr(torch, 'npu') else False
assert not result['npu_initialized']
(root / 'root-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
