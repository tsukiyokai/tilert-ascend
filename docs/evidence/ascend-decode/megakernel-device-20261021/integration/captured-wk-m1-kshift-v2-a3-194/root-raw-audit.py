import hashlib
import json
from pathlib import Path
import torch

root = Path('/tmp/tilert-megakernel-validation-20261021')
raw_path = root/'captured-wk-m1-kshift-v2-raw/wk-kshift-raw-natural-r0.pt'
capture_path = root/'vendor-hook-decoder212-graph-index-int8-m1-v3-raw/report/decoder-graph-index-diagnostic.pt'
raw = torch.load(raw_path, map_location='cpu', weights_only=False)
capture = torch.load(capture_path, map_location='cpu', weights_only=False)
result = {'scope': 'Raw first natural M1 probe call before arithmetic acceptance',
          'raw_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
          'capture_sha256': hashlib.sha256(capture_path.read_bytes()).hexdigest(),
          'outputs': {}, 'arithmetic_control_accepted': False,
          'write_coverage': 'Undetermined: arithmetic can produce the same canonical NaN as prefill'}
for key, reference, integer, poison in (
        ('accumulators', 'key_accumulators', torch.int32, 0x7fffffff),
        ('projection', 'key_projection', torch.int16, 0x7fff)):
    value = raw[key]
    expected = capture['native'][reference]
    assert tuple(value.shape) == (16, 160)
    assert torch.isfinite(value[0,:80]).all()
    assert (value[0,80:].contiguous().view(integer) == poison).all()
    assert (value[1:].contiguous().view(torch.uint8) == 0xa5).all()
    assert torch.equal(raw[key+'_finite'], torch.isfinite(value))
    prefix = value[:1,:80].contiguous().view(torch.uint8)
    reference_prefix = expected[:,:80].contiguous().view(torch.uint8)
    result['outputs'][key] = {
        'finite_columns': [0,79], 'equal_to_prefill_bits_columns': [80,159],
        'poison_bits': hex(poison), 'guard_different_bytes': 0,
        'first80_different_bytes': int((prefix != reference_prefix).sum()),
        'witness25_actual': float(value[0,25]),
        'witness25_captured_native': float(expected[0,25]),
    }
out=Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/captured-wk-m1-kshift-v2-a3-194')
(out/'root-raw-audit.json').write_text(json.dumps(result,indent=2)+'\n')
(out/'root-raw-audit.py').write_text(Path(__file__).read_text())
print(json.dumps(result['outputs'],indent=2))
