import hashlib
import json
from pathlib import Path
import shutil

import torch

ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
NAME = 'captured-wk-m12-kshift-v1'
OUT = Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration') / (NAME + '-a3-194')
report_path = OUT / 'wk-kshift.json'
report = json.loads(report_path.read_text())
assert report['pass'] and report['completed'] and not report['cpu_only']
assert report['library']['sha256'] == 'b84efa3b74fd87881d245988935a42f281291de00632984fd283c4f72e55e86b'
assert report['production_reproduction']['bitwise_equal']
assert len(report['cases']) == 6
for case in report['cases']:
    assert case['four_guard_rows_unchanged']
    assert case['direct_bf16_vs_fp32_cast']['bitwise_equal']
    if case['arm'] == 'natural':
        assert case['captured_native_accumulators']['bitwise_equal']
        assert case['captured_native_projection']['bitwise_equal']
        assert case['captured_vendor_projection']['different_elements'] == 2
    else:
        assert case['captured_vendor_projection']['bitwise_equal']
    if case['repeat']:
        assert case['repeatability_accumulators']['bitwise_equal']
        assert case['repeatability_projection']['bitwise_equal']
assert {(case['arm'], case['repeat']) for case in report['cases']} == {
    (arm, repeat) for arm in ('natural', 'shifted') for repeat in range(3)}
path = ROOT / (NAME + '-raw') / 'wk-kshift-outputs.pt'
outputs = torch.load(path, weights_only=True, map_location='cpu')
parent_path = ROOT / 'vendor-hook-decoder212-index-int8-m12-v1-raw/report/decoder-index-diagnostic.pt'
assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == report['capture']['sha256']
parent = torch.load(parent_path, weights_only=False, map_location='cpu')
for name, source in (('accumulators', 'key_accumulators'), ('projection', 'key_projection')):
    assert torch.equal(outputs['natural'][name].view(torch.uint8), parent['native'][source].view(torch.uint8))
assert torch.equal(outputs['shifted']['projection'].view(torch.uint8),
                   parent['vendor']['key_projection'].view(torch.uint8))
witnesses = [case['witnesses'] for case in report['cases'] if case['arm'] == 'shifted']
assert witnesses[0] == witnesses[1] == witnesses[2]
summary = {
    'pass': True, 'scope': report['scope'], 'geometry': report['geometry'],
    'original_f_linear_reproduced': True,
    'natural_control_full_fp32_and_bf16_byte_equal': True,
    'shifted_full_vendor_bf16_byte_equal': True,
    'interleaved_repeats_per_arm': 3, 'elements_per_arm': 1920,
    'four_guard_rows_unchanged_per_call': True,
    'direct_bf16_matches_same_l0c_fp32_rne_cast': True,
    'shifted_witnesses': witnesses[0],
    'candidate_library_sha256': report['library']['sha256'],
    'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
    'raw_outputs_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'physical_dies': 1, 'physical_die': 4, 'timing': False,
    'full_indexed_attention_executed': False,
    'next_gate': 'Integrate the measured M12-only K order into indexed attention and replay the actual Decoder.'}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
shutil.copy2(__file__, OUT / 'audit.py')
print(json.dumps(summary, indent=2))
