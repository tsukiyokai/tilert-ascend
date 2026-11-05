import hashlib
import json
from pathlib import Path
import torch

root = Path('/tmp/tilert-megakernel-validation-20261021')
raw_root = root/'captured-wk-m1-kshift-v3-raw'
capture_path = root/'vendor-hook-decoder212-graph-index-int8-m1-v3-raw/report/decoder-graph-index-diagnostic.pt'
assert hashlib.sha256(capture_path.read_bytes()).hexdigest() == 'a95e0c289c7e8b4f0808f185aecffc8739e602d6a72896894baf79c640e54c41'
capture = torch.load(capture_path, map_location='cpu', weights_only=False)
report = json.loads((raw_root/'wk-kshift.json').read_text())
records = []
for call in report['raw_calls']:
    path = raw_root/Path(call['dump']['path']).name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == call['dump']['sha256']
    raw = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('accumulators', 'projection'):
        value = raw[key]
        assert tuple(value.shape) == (16, 160)
        assert torch.isfinite(value[0]).all()
        assert (value[1:].contiguous().view(torch.uint8) == 0xa5).all()
        assert torch.equal(raw[key+'_finite'], torch.isfinite(value))
    acc, bf16 = raw['accumulators'][:1], raw['projection'][:1]
    equal = lambda a,b: torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    assert equal(acc.to(torch.bfloat16), bf16)
    if call['arm'] == 'natural':
        assert equal(acc, capture['native']['key_accumulators'])
        assert equal(bf16, capture['native']['key_projection'])
        assert int((bf16 != capture['vendor']['key_projection']).sum()) == 1
    else:
        assert equal(bf16, capture['vendor']['key_projection'])
        assert acc[0,25].item() == -0.3349609375
        assert bf16[0,25].item() == -0.3359375
    records.append({'arm':call['arm'],'repeat':call['repeat'],'pass':True,
                    'raw_sha256':call['dump']['sha256'],
                    'witness_fp32':acc[0,25].item(),'witness_bf16':bf16[0,25].item()})
assert len(records) == 6
result = {'pass':True,'scope':'Independent CPU audit of six retained device outputs',
          'natural_reproduces_all_160_captured_native_fp32_and_bf16':True,
          'shifted_reproduces_all_160_captured_vendor_bf16':True,
          'fifteen_guard_rows_unchanged':True,'direct_bf16_matches_fp32_rounding':True,
          'calls':records,'capture_sha256':hashlib.sha256(capture_path.read_bytes()).hexdigest()}
out = Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration/captured-wk-m1-kshift-v3-a3-194')
(out/'root-raw-audit.json').write_text(json.dumps(result,indent=2)+'\n')
(out/'root-raw-audit.py').write_text(Path(__file__).read_text())
print(json.dumps(result,indent=2))
