"""Read the installed compiler's six constant plans from the actual ELF object."""

import hashlib
import json
from pathlib import Path
import struct


FIELDS = (
    'M N Ka Kb singleCoreM singleCoreN singleCoreK baseM baseN baseK '
    'depthA1 depthB1 stepM stepN stepKa stepKb iterateOrder shareL1Size '
    'shareL0CSize shareUbSize dbL0A dbL0B dbL0C transLength shareMode isBias'
).split()
ROOT = Path(__file__).resolve().parent


def extract_plans(data):
    if data[:6] != b'\x7fELF\x02\x01':
        raise ValueError('Expected a little-endian ELF64 object')
    shoff = struct.unpack_from('<Q', data, 40)[0]
    shsize, shnum, shstr = struct.unpack_from('<HHH', data, 58)
    if shsize != 64:
        raise ValueError('Unexpected ELF section header size')
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, shoff + i * shsize)
                for i in range(shnum)]

    def contents(section):
        return data[section[4]:section[4] + section[5]]

    names = contents(sections[shstr])
    section_names = [names[s[0]:].split(b'\0', 1)[0].decode()
                     for s in sections]
    symbols = sections[section_names.index('.symtab')]
    strings = contents(sections[symbols[6]])
    results = {}
    for offset in range(symbols[4], symbols[4] + symbols[5], symbols[9]):
        name, _, _, index, value, size = struct.unpack_from('<IBBHQQ', data, offset)
        name = strings[name:].split(b'\0', 1)[0].decode()
        if not name.endswith('_fields'):
            continue
        if size != 4 * len(FIELDS):
            raise ValueError('Unexpected constant plan size: ' + name)
        payload = contents(sections[index])[value:value + size]
        results[name] = dict(zip(FIELDS, struct.unpack('<26i', payload), strict=True))
    if len(results) != 6:
        raise ValueError('Expected six independently emitted plan symbols')
    return results


def main():
    plans = extract_plans((ROOT / 'probe.o').read_bytes())
    parent = ROOT.parent / 'ffn-down-n1024-build-v2'
    runtime = {}
    for label, filename in [('n512', 'parent-tilings.jsonl'),
                            ('n1024', 'candidate-tilings.jsonl')]:
        records = [json.loads(line) for line in (parent / filename).read_text().splitlines()]
        for layout, suffix in [(0, 'nd'), (1, 'nz')]:
            record, = [row for row in records if row['projection'] == 1
                       and row['layout'] == layout]
            assert record['status'] == 0
            plan = plans[label + '_' + suffix + '_fields']
            for field in FIELDS[:23]:
                assert plan[field] == record[field], (label, suffix, field)
            runtime[label + '_' + suffix] = hashlib.sha256(
                bytes.fromhex(record['raw_hex'])).hexdigest()
    changes = {'baseK': [64, 128], 'depthA1': [4, 2], 'depthB1': [12, 6],
               'stepKa': [2, 1], 'stepKb': [2, 1], 'dbL0B': [2, 1]}
    for suffix in ('nd', 'nz'):
        before, after = plans['n1024_' + suffix + '_fields'], plans['k128_' + suffix + '_fields']
        actual = {field: [before[field], after[field]] for field in FIELDS
                  if before[field] != after[field]}
        assert actual == changes, actual
        assert after['baseM'] * after['baseK'] // 2 <= 65536
        assert after['baseN'] * after['baseK'] // 2 == 65536
        assert after['baseM'] * after['baseN'] * 4 == 131072
        assert after['depthA1'] * after['baseM'] * after['baseK'] // 2 == 4096
        assert after['depthB1'] * after['baseN'] * after['baseK'] // 2 == 393216
    assert json.loads((ROOT / 'devices.json').read_text()) == []
    summary = {
        'pass': True, 'scope': 'Installed CANN 9.2 compile-time tiling API and resource arithmetic',
        'device_execution': False, 'mapped_npu_devices': [],
        'nd_nz_cases': 6, 'known_parent_fields_checked': 4 * 23,
        'known_parent_runtime_records': runtime,
        'candidate_named_field_changes': changes,
        'candidate_l1_bytes': 397312, 'candidate_l0b_bytes': 65536,
        'candidate_l0c_bytes': 131072,
        'static_api_transLength': 131072,
        'runtime_record_contract': 'Preserve the accepted runtime parent record except the six named fields; compile-time transLength is also 131072 in both accepted control plans',
        'compiler_object_sha256': hashlib.sha256((ROOT / 'probe.o').read_bytes()).hexdigest(),
        'probe_source_sha256': hashlib.sha256((ROOT / 'probe.cpp').read_bytes()).hexdigest(),
        'numerical_acceptance': 'not run', 'performance_acceptance': 'not run',
    }
    (ROOT / 'fields.json').write_text(json.dumps(plans, indent=2) + '\n')
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
