import hashlib
import json
from pathlib import Path
import shutil

import torch

ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
NAME = 'vendor-hook-decoder212-ffn-shared-m32-v1'
OUT = Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration') / (NAME + '-a3-194')
raw = ROOT / (NAME + '-raw') / 'report'
report = json.loads((OUT / 'decoder-shared-diagnostic.json').read_text())
path = raw / 'decoder-shared-diagnostic.pt'
assert hashlib.sha256(path.read_bytes()).hexdigest() == report['artifact_sha256']
captured = torch.load(path, weights_only=True, map_location='cpu')
parent_path = ROOT / 'vendor-hook-decoder212-ffn-int8-wide-v2-raw/report/decoder-ffn-layer6-m32-r2.pt'
assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == report['parent_failure_sha256']
parent = torch.load(parent_path, weights_only=True, map_location='cpu')
current = torch.load(raw / 'decoder-ffn-layer6-m32-r2.pt', weights_only=True, map_location='cpu')
def byte_equal(a, b):
    return (a.dtype == b.dtype and a.numel() == b.numel()
            and torch.equal(a.contiguous().view(torch.uint8).reshape(-1),
                            b.contiguous().view(torch.uint8).reshape(-1)))
assert len(report['parent_reproduced']) == 16 and all(report['parent_reproduced'].values())
for key in ('hidden', 'residual'):
    assert byte_equal(parent[key], current[key])
for arm in ('reference', 'native'):
    assert set(parent[arm]) == set(current[arm]) and len(parent[arm]) == 7
    for key in parent[arm]:
        assert byte_equal(parent[arm][key], current[arm][key]), (arm, key)
assert report['calls'] == {'prepare': 1, 'activate': 1, 'down': 1}
assert report['diagnostic_complete'] and not report['timing']
assert not report['original_arithmetic_replaced']
native, reference = captured['native'], captured['reference']
for name in ('normalized', 'codes', 'input_scale', 'gate_int32', 'middle'):
    assert byte_equal(native[name], reference[name]), name
gate = reference['codes'].int() @ reference['gate_weight'].int()
down = reference['middle'].int() @ reference['down_weight'].int()
assert torch.equal(gate, reference['gate_int32'])
assert torch.equal(down, native['down_int32'])
assert reference['down_scale'].dtype == torch.bfloat16
assert reference['middle_scale'].dtype == torch.float32
native_scale = native['middle_scale'].reshape(32)
official_scale = reference['middle_scale'].reshape(32)
rows = (native_scale.view(torch.int32) != official_scale.view(torch.int32)).nonzero().reshape(-1)
assert rows.tolist() == [0, 1, 20]
witnesses = []
for row in rows.tolist():
    a, b = native_scale[row], official_scale[row]
    assert int(b.view(torch.int32)) - int(a.view(torch.int32)) == 1
    witnesses.append({'row': row, 'native': float(a), 'official': float(b),
                      'native_bits': hex(int(a.view(torch.int32))),
                      'official_bits': hex(int(b.view(torch.int32)))})
orders = {}
for name, scale in (('native', native_scale), ('official', official_scale)):
    token = scale.reshape(32, 1)
    channel = reference['down_scale'].float().reshape(1, 6144)
    product = down.float()
    values = {'combined': product * (channel * token),
              'channel_then_token': (product * channel) * token,
              'token_then_channel': (product * token) * channel}
    for order, value in values.items():
        rounded = value.bfloat16()
        orders[f'{name}.{order}'] = {
            arm: int((rounded.view(torch.int16) != target['output'].view(torch.int16)).sum())
            for arm, target in (('native', native), ('official', reference))}
assert orders['native.combined'] == {'native': 0, 'official': 2}
assert orders['official.combined'] == {'native': 2, 'official': 0}
summary = {
    'diagnostic_complete': True, 'original_ffn_acceptance': False,
    'scope': report['scope'], 'parent_input_output_byte_checks': 16,
    'first_differing_retained_boundary': 'shared middle_scale',
    'equal_boundaries': ['normalized', 'codes', 'input_scale', 'gate_int32', 'middle'],
    'middle_scale_witnesses': witnesses,
    'original_gu_int32_matches_exact_cpu_integer_dot': True,
    'native_down_int32_matches_exact_cpu_integer_dot': True,
    'epilogue_cpu_differential': orders,
    'interpretation': 'Replacing only the three middle scales in the combined-scales epilogue reproduces every original BF16 output; the activation-scale producer remains to be reconciled.',
    'library_sha256': '8430ef17d068ce5a1cdbe4915ba189ca38c666322d8726a42091f8acb3b36f84',
    'report_sha256': hashlib.sha256((OUT / 'decoder-shared-diagnostic.json').read_bytes()).hexdigest(),
    'artifact_sha256': report['artifact_sha256'],
    'original_shared_source_sha256': '2892f5523cdbb3aba209654f2c9e0a8f5090406881dd94202a06b87ccc5f358a',
    'physical_dies': 1, 'physical_die': 4, 'synthetic_parameters': True,
    'tp_collectives_executed': 0, 'timing': False}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
shutil.copy2(__file__, OUT / 'audit.py')
print(json.dumps(summary, indent=2))
