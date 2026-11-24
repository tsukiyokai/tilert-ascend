"""Check the release overlay keeps every factory GroupedMatmul variant."""
import hashlib
import json
from pathlib import Path
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import torch
import torch_npu

root = Path('/workspace/experiments/gmm2-compact-20261020')
kernel = root / 'release-build-v2/overlay/vendors/tilert_gmm2/op_impl/ai_core/tbe/kernel'
config = json.loads((kernel / 'config/ascend910_93/grouped_matmul.json').read_text())
for item in config['binList']:
    descriptor = kernel / item['binInfo']['jsonFilePath']
    details = json.loads(descriptor.read_text())
    binary = descriptor.with_suffix('.o')
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == details['sha256']
torch.npu.set_device('npu:15')
x = torch.arange(256, dtype=torch.float32).reshape(2, 128).bfloat16().npu()
w = torch.eye(128, 64, dtype=torch.bfloat16).unsqueeze(0).npu()
groups = torch.tensor([2], dtype=torch.int64, device='npu:15')
y = torch_npu.npu_grouped_matmul(
    x=[x], weight=[w], group_list=groups, split_item=2, group_type=0,
    group_list_type=1, output_dtype=torch.bfloat16)[0]
torch.npu.synchronize()
exact = torch.equal(y.cpu(), x[:, :64].cpu())
assert exact
report = {'registry_variants': len(config['binList']),
          'all_descriptor_binary_hashes_match': True,
          'bf16_nd_identity_projection_exact': exact,
          'peak_reserved_bytes': torch.npu.max_memory_reserved()}
(root / 'coverage.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
