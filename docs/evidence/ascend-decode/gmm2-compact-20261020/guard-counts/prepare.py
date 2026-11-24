"""Construct count distributions around a one-eighth reread budget."""

import hashlib
import json
from pathlib import Path
import runpy


ROOT = Path('/workspace/experiments/gmm2-compact-20261020')
OUT = ROOT / 'guard-counts-screen'
helper = ROOT / 'retile-cpu-agent/audit.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest() == (
    '13b90f39649c404e337465e183d353c97d6fef6e55af646dd81d3d820b413d91'
)
check = runpy.run_path(str(helper), run_name='schedule_helper')['check']
cases = []
for active, hot, hot_rows in (
    (16, 2, 32), (32, 4, 17), (32, 4, 32),
    (64, 8, 17), (64, 8, 25), (64, 9, 17), (32, 8, 17),
    (16, 8, 30),
):
    counts = [hot_rows] * hot + [1] * (active - hot)
    remaining = 256 - sum(counts)
    for index in range(remaining):
        counts[hot + index % (active - hot)] += 1
    assert max(counts[hot:]) <= 16
    scattered = [0] * 256
    for index, count in enumerate(counts):
        scattered[index * 251 % 256] = count
    assert sum(scattered) == 256
    assert sum(c > 16 for c in scattered) == hot
    compact = check(scattered, 64, 512)
    wide = check(scattered, 32, 1024)
    cases.append({
        'name': f'active{active}-hot{hot}-rows{hot_rows}',
        'counts': scattered, 'active': active, 'hot': hot,
        'maximum': max(scattered),
        'compact': {'tasks': compact['tiles']},
        'wide': {'tasks': wide['tiles']},
        'cpu_schedule': {'compact': compact, 'wide': wide},
    })
result = {
    'scope': 'Synthetic M32 top8 COUNT distributions, total rows 256',
    'npu_launched': False, 'cases': cases,
    'purpose': 'Test concentration boundaries before choosing a guard',
}
(OUT / 'counts.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps([{
    key: value for key, value in case.items()
    if key not in ('counts', 'cpu_schedule')
} for case in cases], indent=2))
