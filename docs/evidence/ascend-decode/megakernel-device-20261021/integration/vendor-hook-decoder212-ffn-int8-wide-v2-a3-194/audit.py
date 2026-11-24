import hashlib
import json
from pathlib import Path
import shutil

import torch

ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
NAME = 'vendor-hook-decoder212-ffn-int8-wide-v2'
OUT = Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration') / (NAME + '-a3-194')
report = json.loads((OUT / 'decoder-ffn.json').read_text())
correction = json.loads((OUT / 'nz-storage-correction.json').read_text())
assert correction['completed'] and correction['additional_hbm_bytes'] == 0
assert len(correction['banks']) == 4
for bank in correction['banks']:
    assert bank['raw_bytes_equal'] and bank['new_device_allocation_bytes'] == 0
    assert bank['before']['data_ptr'] == bank['after']['data_ptr']
    assert bank['raw_sha256_before'] == bank['raw_sha256_after']
assert not report['pass'] and report['widths'] == [1, 2, 12, 32]
assert report['iterations'] == 8 and report['native_graph_replay']
assert report['library_sha256'] == '8430ef17d068ce5a1cdbe4915ba189ca38c666322d8726a42091f8acb3b36f84'
assert len(report['cases']) == 27
expected = [(m, i) for m in (1, 2, 12) for i in range(8)] + [(32, i) for i in range(3)]
failures = []
boundaries = {'normalized', 'retained', 'logits', 'shared', 'routed', 'combined', 'pre_ar2'}
for case, (rows, iteration) in zip(report['cases'], expected):
    assert (case['layer'], case['rows'], case['iteration']) == (6, rows, iteration)
    assert set(case['checks']) == boundaries
    for name, metric in case['checks'].items():
        if not metric['bitwise_equal']:
            failures.append((rows, iteration, name, metric))
        else:
            assert metric['pass'] and metric['bitwise_different_bytes'] == 0
assert len(failures) == 1 and failures[0][:3] == (32, 2, 'shared')
dump = ROOT / (NAME + '-raw') / 'report/decoder-ffn-layer6-m32-r2.pt'
captured = torch.load(dump, weights_only=True, map_location='cpu')
coordinates = []
for name in boundaries:
    actual, expected_value = captured['native'][name], captured['reference'][name]
    mismatch = (actual.view(torch.int16 if actual.dtype == torch.bfloat16 else torch.int32)
                != expected_value.view(torch.int16 if expected_value.dtype == torch.bfloat16 else torch.int32)).nonzero()
    if name != 'shared':
        assert mismatch.numel() == 0, name
    else:
        assert mismatch.tolist() == [[20, 2893], [20, 5220]]
        coordinates = [
            {'coordinate': point.tolist(), 'native': actual[tuple(point)].item(),
             'reference': expected_value[tuple(point)].item()}
            for point in mismatch]
input_hashes = {
    name: hashlib.sha256(captured[name].view(torch.uint8).numpy().tobytes()).hexdigest()
    for name in ('hidden', 'residual')}
ranks = [json.loads((OUT / f'rank-{rank:02d}.json').read_text()) for rank in range(16)]
assert all(not rank['rank_zero_status']['success'] for rank in ranks)
assert all(rank['npu_initialized_after'] == (rank['rank'] == 0) for rank in ranks)
summary = {
    'pass': False, 'failure_scope': 'Original FFN shared branch, layer6 M32 iteration2',
    'raw_byte_boundary_checks': 189, 'raw_byte_boundary_passes': 188,
    'complete_cases_passed': 26, 'first_failing_case': {'layer': 6, 'rows': 32, 'iteration': 2},
    'shared_mismatch': coordinates, 'shared_metric': failures[0][3],
    'routed_and_combined_and_pre_ar2_byte_equal': True,
    'completed_layer6_inputs': {'M1': 8, 'M2': 8, 'M12': 8, 'M32': 2},
    'layer7_wide_gate_executed': False, 'graph_replay': True,
    'candidate_library_sha256': report['library_sha256'],
    'original_weights_and_nz_bytes_preserved': True,
    'captured_input_sha256': input_hashes, 'synthetic_parameters': True,
    'attention_executed': False, 'tp_collectives_executed': 0,
    'physical_dies': 1, 'npu_ranks': [0],
    'artifact_sha256': {
        'decoder-ffn.json': hashlib.sha256((OUT / 'decoder-ffn.json').read_bytes()).hexdigest(),
        'failure_dump': hashlib.sha256(dump.read_bytes()).hexdigest()},
    'next_gate': 'Capture original shared activation and native payload on the exact failing input.'}
(OUT / 'failure-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
shutil.copy2(__file__, OUT / 'audit.py')
print(json.dumps(summary, indent=2))
