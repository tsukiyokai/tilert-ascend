"""A3 numerical witness for token-index versus block-index SFA units."""

import hashlib
import json
from pathlib import Path

import torch
import torch_npu
import vllm_ascend.vllm_ascend_C as extension

torch.set_num_threads(1)
torch.npu.set_device(0)
properties = torch.npu.get_device_properties(0)
assert properties.name.upper() in ('ASCEND910_9382', 'ASCEND910_9392')
torch.npu.set_per_process_memory_fraction(2 * 1024**3 / properties.total_memory, 0)
directory = Path('/evidence/sfa-block-witness')
directory.mkdir(exist_ok=False)
key = torch.zeros((65, 128, 1, 512), dtype=torch.bfloat16)
key.view(-1, 512)[:512].fill_(1)
key.view(-1, 512)[512:8192].fill_(2)
key.view(-1, 512)[8192].fill_(4)
indices = torch.full((1, 1, 2048), -1, dtype=torch.int32)
indices[0, 0, :512] = torch.arange(512, dtype=torch.int32)
inputs = {'query': torch.zeros((1, 4, 512), dtype=torch.bfloat16),
          'key': key, 'value': key, 'sparse_indices': indices,
          'query_rope': torch.zeros((1, 4, 64), dtype=torch.bfloat16),
          'key_rope': torch.zeros((65, 128, 1, 64), dtype=torch.bfloat16),
          'block_table': torch.arange(65, dtype=torch.int32)[None],
          'actual_seq_lengths_query': torch.tensor([1], dtype=torch.int32),
          'actual_seq_lengths_kv': torch.tensor([8193], dtype=torch.int32)}
device_inputs = {name: value.to('npu:0') for name, value in inputs.items()}
outputs, records = [], []
for block, expected in ((1, 1.0), (16, 1.9375), (1, 1.0)):
    result = torch.ops._C_ascend.npu_sparse_flash_attention(
        **device_inputs, scale_value=1/16, sparse_block_size=block,
        layout_query='TND', layout_kv='PA_BSND', sparse_mode=3,
        attention_mode=2, return_softmax_lse=False)
    actual = result[0].detach().cpu().clone()
    outputs.append(actual)
    records.append({'sparse_block_size': block, 'expected': expected,
                    'finite': bool(torch.isfinite(actual).all()),
                    'actual_values': actual.unique().tolist()[:12],
                    'exact': torch.equal(actual, torch.full_like(actual, expected))})
artifact = directory / 'actual.pt'
torch.save({'inputs': inputs, 'outputs': outputs}, artifact)
report = {'scope': 'One-die A3 SFA index-unit numerical witness; no latency measurement',
          'sku': properties.name, 'device': 0, 'records': records,
          'repeat_exact': torch.equal(outputs[0], outputs[2]),
          'extension': extension.__file__,
          'extension_sha256': hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
          'operator_schema': str(torch.ops._C_ascend.npu_sparse_flash_attention.default._schema),
          'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'precision_verified': False, 'performance_measured': False}
report['witness_passed'] = report['repeat_exact'] and all(item['exact'] for item in records)
(directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(int(not report['witness_passed']))
