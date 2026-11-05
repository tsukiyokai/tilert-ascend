import hashlib
import itertools
import json
from pathlib import Path
import shutil

root = Path('/tmp/tilert-megakernel-validation-20261021')
repo = Path('/home/shanshan/repo/tile-ai/TileRT-Ascend')
dest = repo / 'docs/evidence/ascend-decode/megakernel-device-20261021/integration/ffn-dual-quant-probe-v1-a3-194'
dest.mkdir(exist_ok=False)
summary = {'scope': 'Direct shared/routed input quantization and immediate consumers', 'host': 'a3-194', 'physical_die': 4, 'timing': 'not measured', 'arms': {}}
fixtures = {'positive_zero', 'negative_zero', 'mixed_signed_zero', 'distinct_scale_witness', 'finite_large', 'finite_tiny', 'positive_infinity', 'signed_infinity', 'quiet_nan_payloads', 'signaling_nan_payloads', 'mixed_nonfinite', 'finite_mixed'}
for arm, widths, blocks in [('m12', [12], [24]), ('wide', [1, 2, 12, 32], [16, 24])]:
 name = 'ffn-dual-quant-probe-' + arm + '-v1'
 path = root / name / 'report.json'
 report = json.loads(path.read_text())
 cases = report['cases']
 geometry = len(widths) * len(blocks)
 assert report['pass'] and len(cases) == 24 * geometry
 assert report['captures'] == 2 * geometry and report['replays'] == 48 * geometry
 assert report['candidate']['library_sha256'] == '715060c0c50a275855eb757cd85128a6c5586deeaa7757dc2945b92bee64d69b'
 assert report['probe']['library_sha256'] == '41bc2af880217a0520e6dfc8764de6068e1e570bc08ea14e2349333549126c65'
 actual = {(v['rows'], v['blocks'], v['fixture'], v['repeat']) for v in cases}
 assert actual == set(itertools.product(widths, blocks, fixtures, [0, 1]))
 count = 0
 for case in cases:
  assert case['pass'] and len(case['checks']) == (14 if case['repeat'] == 0 else 22)
  for check in case['checks'].values():
   assert check['bitwise_equal'] and check['different_bytes'] == 0
   count += 1
 assert count == 432 * geometry
 target = dest / arm
 target.mkdir()
 shutil.copy2(path, target / 'report.json')
 for suffix in ('command.json', 'launcher.log'):
  shutil.copy2(root / (name + '.' + suffix), target / suffix)
 summary['arms'][arm] = {'rows': widths, 'blocks': blocks, 'cases': len(cases), 'captures': report['captures'], 'replays': report['replays'], 'raw_byte_checks': count, 'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'pass': True}
for name in ('run-ffn-dual-quant-probe-v1.py', 'archive-ffn-dual-quant-probe-v1.py'):
 shutil.copy2(root / name, dest / name)
shutil.copy2(root / 'ffn-dual-quant-probe-build-v1/probe/check.py', dest / 'check.py')
for name in ('ffn-dual-quant-build-v1', 'ffn-dual-quant-probe-build-v1'):
 shutil.copy2(root / name / 'manifest.json', dest / (name + '-manifest.json'))
(dest / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
