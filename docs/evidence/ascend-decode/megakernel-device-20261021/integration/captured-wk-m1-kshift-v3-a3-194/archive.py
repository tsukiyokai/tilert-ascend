import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

root = Path('/tmp/tilert-megakernel-validation-20261021')
name = 'captured-wk-m1-kshift-v3'
raw = root / (name + '-raw')
raw.mkdir()
data = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes',
    'a3-194', shlex.join(['tar', '-czf', '-', '-C', '/home/shanshan/tilert-megakernel-20261021/logs/' + name, '.'])])
with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
    archive.extractall(raw, filter='data')
out = Path('docs/evidence/ascend-decode/megakernel-device-20261021/integration') / (name + '-a3-194')
out.mkdir()
identities = {}
for path in raw.rglob('*'):
    if path.is_file():
        rel = path.relative_to(raw)
        identities[str(rel)] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
        if path.suffix in ('.json', '.jsonl', '.log', '.txt'):
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
r = json.loads((raw/'wk-kshift.json').read_text())
assert r['pass'] and r['completed'] and len(r['cases']) == 6
assert r['production_reproduction']['bitwise_equal']
assert r['library']['sha256'] == '8493189945cd922ac2cff9a7653bccd1b5ad03bcf7574ea1278ef91fd44594a8'
assert {(c['arm'],c['repeat']) for c in r['cases']} == {(a,i) for a in ('natural','shifted') for i in range(3)}
for case in r['cases']:
    assert case['fifteen_guard_rows_unchanged']
    assert case['direct_bf16_vs_fp32_cast']['bitwise_equal']
    if case['arm'] == 'natural':
        assert case['captured_native_accumulators']['bitwise_equal']
        assert case['captured_native_projection']['bitwise_equal']
        assert case['captured_vendor_projection']['different_elements'] == 1
    else:
        assert case['captured_vendor_projection']['bitwise_equal']
        assert case['witnesses'][0]['accumulator']['bits'] == '0xbeab8000'
        assert case['witnesses'][0]['projection']['bits'] == '0xbeac'
for call in r['raw_calls']:
    path=raw/Path(call['dump']['path']).name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == call['dump']['sha256']
    assert all(not v['nonfinite_active_coordinates'] and v['guard_different_bytes']==0 for v in call['outputs'].values())
summary = {'pass': True, 'completed': True, 'exit': 0,
           'scope': r['scope'], 'original_F_linear_reproduced': True,
           'accepted_natural_or_shifted_cases': 6,
           'natural_control': 'All 160 FP32 accumulators and BF16 outputs match captured native in three repeats',
           'shifted_result': 'All 160 BF16 outputs match captured vendor in three repeats',
           'witness_0_25': {'shifted_fp32': -0.3349609375, 'shifted_fp32_bits': '0xbeab8000', 'shifted_bf16': -0.3359375, 'shifted_bf16_bits': '0xbeac'},
           'fifteen_guard_rows_unchanged': True,
           'raw_outputs_retained': True, 'raw_calls': r['raw_calls'],
           'library_sha256': r['library']['sha256'],
           'report_sha256': hashlib.sha256((raw/'wk-kshift.json').read_bytes()).hexdigest(),
           'physical_die': 4, 'host': 'a3-194',
           'raw_root': str(raw), 'raw_files': identities}
(out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
for source, target in ((name+'.command.json','command.json'),(name+'.launcher.log','launcher.log'),
                       ('run-wk-m1-kshift-v3.py','run.py')):
    shutil.copy2(root/source,out/target)
shutil.copy2(__file__,out/'archive.py')
print(json.dumps({key:summary[key] for key in ('pass','natural_control','shifted_result','report_sha256')},indent=2))
