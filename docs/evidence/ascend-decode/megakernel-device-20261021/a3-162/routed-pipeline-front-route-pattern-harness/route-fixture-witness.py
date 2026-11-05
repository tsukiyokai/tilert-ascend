"""Record equal input bytes for two route patterns on the target CPU."""
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch_npu

sys.path.insert(0, '/workspace/tests/glm52/routed_int4')
import check_pipeline_front as harness

assert not torch.npu.is_initialized()
torch.set_num_threads(1)
result = {'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
          'npu_initialized_before': torch.npu.is_initialized(),
          'harness_sha256': hashlib.sha256(Path(harness.__file__).read_bytes()).hexdigest(),
          'cases': []}
for width in (12, 32):
    mixed = harness.input_case(torch, width, 3, 20261021, 'mixed')
    shared = harness.input_case(torch, width, 3, 20261021, 'shared8')
    assert all(torch.equal(mixed[index].view(torch.uint8), shared[index].view(torch.uint8))
               for index in (0, 1, 3))
    result['cases'].append({
        'width': width,
        'mixed_experts': mixed[2].unique().numel(),
        'shared8_experts': shared[2].unique().numel(),
        'same_input_sha256': {
            name: harness.tensor_sha256(torch, mixed[index])
            for index, name in ((0, 'routed'), (1, 'shared'), (3, 'probabilities'))
        },
        'mixed_choices_sha256': harness.tensor_sha256(torch, mixed[2]),
        'shared8_choices_sha256': harness.tensor_sha256(torch, shared[2]),
    })
result['npu_initialized_after'] = torch.npu.is_initialized()
assert result['npu_initialized_after'] is False
output = Path('/workspace/logs/routed-pipeline-front-route-fixtures.json')
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
