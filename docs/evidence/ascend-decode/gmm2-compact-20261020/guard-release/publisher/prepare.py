"""Compile and execute the actual release publisher on CPU tensor storage."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
ROOT = Path(__file__).resolve().parent
BASE = Path('/workspace/experiments/gmm2-compact-20261020')
HEADER = BASE / 'guard-release-build/source/ops_transformer/ascendc/grouped_matmul/op_kernel/grouped_matmul_active_groups_a8w4.h'
text = HEADER.read_text()
start = text.index('    __aicore__ inline void Publish(')
end = text.index('    __aicore__ inline void Invalidate()', start)
body = text[start:end]
assert 'splitExperts * 8 <= active' in body and 'end == 256 && active >= 64' in body
cpp = ROOT / 'publisher.cpp'
cpp.write_text((ROOT / 'stubs.cpp').read_text() + body + (ROOT / 'driver.cpp').read_text())
raw = json.loads((BASE / 'guard-counts-screen/counts.json').read_text())
expected = [False, False, False, True, True, False, False, False]
cases = [(record['name'], flag, record['counts'])
         for record, flag in zip(raw['cases'], expected, strict=True)]


def make_counts(active, hot, hot_rows, total=256):
    values = [hot_rows] * hot + [1] * (active - hot)
    remaining = total - sum(values)
    for index in range(hot, active):
        take = min(15, remaining)
        values[index] += take
        remaining -= take
    assert remaining == 0 and min(values) > 0
    counts = [0] * 256
    for index, value in enumerate(values):
        counts[(37 * index + 11) % 256] = value
    return counts


cases.extend([
    ('old_small_m8', True, [1] * 8 + [0] * 248),
    ('old_small_a16_m256', True, [16] * 16 + [0] * 240),
    ('active63_hot7', False, make_counts(63, 7, 25)),
    ('active64_hot7', True, make_counts(64, 7, 25)),
    ('end255_active64_hot7', False, make_counts(64, 7, 25, 255)),
    ('max32_active64_hot1', True, make_counts(64, 1, 32)),
    ('max33_active64_hot1', False, make_counts(64, 1, 33)),
])
for index, label in ((4, 'dynamic_A'), (5, 'dynamic_B'), (4, 'dynamic_reset_A')):
    _, flag, counts = cases[index]
    cases.append((label, flag, counts))
payload = str(len(cases)) + '\n' + '\n'.join(
    f'{name} {int(flag)} ' + ' '.join(map(str, counts)) for name, flag, counts in cases) + '\n'
(ROOT / 'cases.json').write_text(json.dumps(cases) + '\n')
subprocess.run(['g++', '-std=c++17', '-O2', str(cpp), '-o', str(ROOT / 'publisher')], check=True)
run = subprocess.run([str(ROOT / 'publisher')], input=payload, text=True, capture_output=True, check=True)
checks = [json.loads(line) for line in run.stdout.splitlines()]
assert len(checks) == len(cases)
files = [HEADER, ROOT / 'prepare.py', ROOT / 'stubs.cpp', ROOT / 'driver.cpp', cpp, ROOT / 'cases.json']
result = {'status': 'pass', 'npu_launched': False, 'actual_cpp_publish_executed': True,
          'scope': 'Publisher guard, header, compact records and padding only; flags/DataCopy are CPU storage stubs, not asynchronous or Graph validation.',
          'cases': checks, 'publisher_body_sha256': hashlib.sha256(body.encode()).hexdigest(),
          'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
(ROOT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': 'pass', 'cases': len(checks), 'actual_cpp_publish_executed': True}))
