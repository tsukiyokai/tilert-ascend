from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

root = Path('/work')
expected_path = root / 'receipts/checkpoint-expected.json'
expected = json.loads(expected_path.read_text())['files']
model = Path('/models/glm52')
for item in expected:
    path = model / item['name']
    if not path.is_file() or path.stat().st_size != item['bytes']:
        raise ValueError('Consumer view missing or wrong size: ' + item['name'])


def digest(item):
    path = model / item['name']
    hashed = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 2**20):
            hashed.update(block)
    actual = hashed.hexdigest()
    return {'name': item['name'], 'bytes': path.stat().st_size,
            'sha256': actual, 'expected_sha256': item['expected_sha256'],
            'pass': actual == item['expected_sha256']}


started = time.monotonic()
files = []
with ThreadPoolExecutor(max_workers=8) as pool:
    pending = [pool.submit(digest, item) for item in expected]
    for future in as_completed(pending):
        files.append(future.result())
        if len(files) % 16 == 0 or len(files) == len(expected):
            print('files_complete', len(files), 'of', len(expected), flush=True)
receipt = {'scope': 'Actual readonly checkpoint bytes and CPU page-cache preparation before NPU allocation',
           'observed_at': datetime.now(timezone.utc).isoformat(),
           'seconds': time.monotonic() - started,
           'files': sorted(files, key=lambda value: value['name']),
           'pass': all(value['pass'] for value in files),
           'npu_executed': False, 'performance_measured': False,
           'hash_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root / 'receipts/checkpoint-new-consumer.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({key: value for key, value in receipt.items() if key != 'files'}))
raise SystemExit(0 if receipt['pass'] else 1)
