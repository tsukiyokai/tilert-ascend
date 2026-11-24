"""Check typed host records against the compiler-derived six-field change."""

import hashlib
import json
from pathlib import Path
import struct

from audit import extract_plans


ROOT = Path(__file__).resolve().parent
PARENT_SHA = 'c207230aafb11e2c2f825cd9a2ea70147ee7733046b8b2957593c9c2bc3e8009'
CANDIDATE_SHA = 'c2b73af9d605ad8e64ea551f288d188a0cf636828a5a0471ea7c63d616f50999'
CHANGED_FIELDS = ('baseK', 'depthA1', 'depthB1', 'stepKa', 'stepKb', 'dbL0B')


def main():
    plans = extract_plans((ROOT / 'probe.o').read_bytes())
    rows = [json.loads(line) for line in (ROOT / 'runtime-records.jsonl').read_text().splitlines()]
    offsets, = [row['field_offsets'] for row in rows if 'field_offsets' in row]
    assert set(offsets) == {*CHANGED_FIELDS, 'transLength'}
    metadata = json.loads((ROOT / 'runtime-environment.json').read_text())
    assert metadata['devices'] == [] and metadata['host_only']
    assert hashlib.sha256((ROOT / 'runtime_record.cpp').read_bytes()).hexdigest() == metadata['source_sha256']
    for filename in ('command.json', 'runtime-command.json'):
        command = json.loads((ROOT / filename).read_text())
        assert not any(value == '--device' or value.startswith('--device=') for value in command)
    records = []
    for layout, suffix in ((0, 'nd'), (1, 'nz')):
        parent, = [row for row in rows if row.get('label') == 'parent' and row['layout'] == layout]
        proposed, = [row for row in rows if row.get('label') == 'proposed' and row['layout'] == layout]
        old = bytes.fromhex(parent['raw_hex'])
        new = bytes.fromhex(proposed['raw_hex'])
        assert len(old) == len(new) == 200
        assert hashlib.sha256(old).hexdigest() == PARENT_SHA
        assert hashlib.sha256(new).hexdigest() == CANDIDATE_SHA
        reconstructed = bytearray(old)
        before = plans['n1024_' + suffix + '_fields']
        after = plans['k128_' + suffix + '_fields']
        changes = {name for name in before if before[name] != after[name]}
        assert changes == set(CHANGED_FIELDS)
        for field in CHANGED_FIELDS:
            assert struct.unpack_from('<i', old, offsets[field])[0] == before[field]
            assert struct.unpack_from('<i', new, offsets[field])[0] == after[field]
            struct.pack_into('<i', reconstructed, offsets[field], after[field])
        assert bytes(reconstructed) == new
        assert struct.unpack_from('<i', new, offsets['transLength'])[0] == 0
        records.append({'layout': suffix, 'bytes': len(new),
                        'parent_sha256': PARENT_SHA, 'proposed_sha256': CANDIDATE_SHA})
    result = {
        'pass': True, 'device_execution': False,
        'scope': 'Installed host API and public TCubeTiling named-field assignment',
        'records': records, 'changed_fields': list(CHANGED_FIELDS),
        'all_other_bytes_preserved': True, 'runtime_transLength': 0,
        'static_transLength_in_both_controls_and_candidate': 131072,
        'numerical_acceptance': 'not run', 'performance_acceptance': 'not run',
    }
    (ROOT / 'runtime-record-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
