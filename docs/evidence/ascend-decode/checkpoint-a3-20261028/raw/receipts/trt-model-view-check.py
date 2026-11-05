"""Check all 109 canonical names and bytes in the consumer namespace."""
import hashlib
import json
from pathlib import Path

root = Path('/models/glm52')
receipts = Path('/work/receipts/checkpoint')
official_path = receipts / 'official-files.json'
raw = official_path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'fd225ae2ee98270f62feab18f7f75fb87e7c3bbb00e811bfb455b7155caece5f'
expected = {entry['Path']: entry for entry in json.loads(raw)['Data']['Files']
            if entry['Type'] == 'blob'}
assert len(expected) == 109
core = json.loads((receipts / 'summary.json').read_text())
assert core['pass'] and core['upstream_weight_hashes_verified']
prior = {entry['name']: entry for entry in core['files']}
rows = []
for name, entry in sorted(expected.items()):
    path = root / name
    before = path.stat()
    assert before.st_size == entry['Size'], name
    if name in prior:
        actual = prior[name]['sha256']
        assert prior[name]['bytes'] == before.st_size
        assert prior[name]['header_and_index_pass']
        method = 'actual full SHA from current staged verification interval'
    else:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        actual = digest.hexdigest()
        method = 'actual consumer-namespace full SHA'
    after = path.stat()
    assert (before.st_size, before.st_ino, before.st_mtime_ns) == (
        after.st_size, after.st_ino, after.st_mtime_ns), name
    assert actual == entry['Sha256'], name
    rows.append({'name': name, 'bytes': before.st_size, 'sha256': actual,
                 'expected_sha256': entry['Sha256'], 'hash_matches': True,
                 'method': method, 'resolved_path': str(path.resolve())})
report = {'scope': '109-file canonical consumer view and full SHA provenance; no model inference',
          'pass': True, 'files': rows, 'bytes': sum(item['bytes'] for item in rows),
          'full_core_shards': len(prior), 'all_109_hashes_match': True,
          'core_report_sha256': hashlib.sha256((receipts / 'summary.json').read_bytes()).hexdigest(),
          'official_files_sha256': hashlib.sha256(raw).hexdigest(),
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
assert report['bytes'] == 405008283962
(receipts / 'consumer-view.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ('pass', 'bytes', 'full_core_shards', 'all_109_hashes_match')}))
