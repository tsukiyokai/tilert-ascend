"""Audit saved tree-reduction bits, HCCL launches and frozen identities."""

import ast
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct


ROOT = Path('/tmp/tilert-megakernel-validation-20261021')
NAME = 'communication-tree-three-rank-v1'
RAW = ROOT / (NAME + '-raw')
FIXTURE = ROOT / 'communication-tree-device-v1'
OUTPUT = Path('/home/shanshan/repo/tile-ai/TileRT-Ascend/docs/evidence/'
              'ascend-decode/megakernel-device-20261021/integration') / (NAME + '-a3-194')
MANIFEST = '18d57bafa45fb629965916d0d380e014686dd322c575c469ee6327caa4cc98e8'
ALGORITHM = 'AllReduceMeshAivFor91093Executor'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def f32(value):
    try:
        return struct.unpack('<f', struct.pack('<f', value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def f32_bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def bf16_value(value):
    return struct.unpack('<f', struct.pack('<I', value << 16))[0]


def bf16_bits(value):
    bits = f32_bits(value)
    if bits & 0x7F800000 == 0x7F800000:
        return bits >> 16
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def first_32(values):
    return [values[index % len(values)] for index in range(32)]


def comparison(actual, expected, rows):
    assert len(actual) == len(expected)
    repeats = rows * 6144 // len(actual)
    assert repeats * len(actual) == rows * 6144
    raw_actual = struct.pack('<' + 'H' * len(actual), *actual)
    raw_expected = struct.pack('<' + 'H' * len(expected), *expected)
    values = [(bf16_value(a), bf16_value(b)) for a, b in zip(actual, expected)]
    finite = [(a, b) for a, b in values if math.isfinite(a) and math.isfinite(b)]
    return {
        'bitwise_equal': actual == expected,
        'different_bytes': repeats * sum(a != b for a, b in zip(raw_actual, raw_expected)),
        'finite_elements': len(finite) * repeats,
        'actual_nonfinite': repeats * sum(not math.isfinite(a) for a, _ in values),
        'expected_nonfinite': repeats * sum(not math.isfinite(b) for _, b in values),
        'finite_max_absolute': max((abs(a-b) for a, b in finite), default=None),
        'first_32_actual_bits': first_32(actual),
        'first_32_expected_bits': first_32(expected),
    }


assert digest(FIXTURE / 'manifest.json') == MANIFEST
pins = dict(line.split('  ', 1)[::-1] for line in
            (FIXTURE / 'frozen-inputs.sha256').read_text().splitlines())
assert len(pins) == 43
for filename, sha in pins.items():
    assert digest(FIXTURE / filename) == sha, filename
module = ast.parse((FIXTURE / 'communication_tree_devicecheck.py').read_text())
edges = next(ast.literal_eval(node.value) for node in module.body
             if isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id == 'EDGE_COLUMNS'
                 for target in node.targets))
assert len(edges) == 32
manifest = json.loads((FIXTURE / 'manifest.json').read_text())
cases, identities = [], []
for rank in range(3):
    path = RAW / NAME / f'rank-{rank}.json'
    report = json.loads(path.read_text())
    assert report['rank'] == rank and report['ranks'] == 3
    assert report['pass'] and report['order_pass'] and report['edge_hccl_equal']
    assert (report['torch'], report['torch_npu']) == ('2.12.0+cu130', '2.12.0')
    for key, filename in [('parent_sha256', 'parent_probe.so'),
                          ('candidate_sha256', 'candidate_probe.so'),
                          ('driver_sha256', 'communication_tree_devicecheck.py')]:
        assert report[key] == pins[filename]
    assert report['device_mapping'] == {
        'logical_to_physical': [4, 6, 7], 'logical_rank': rank,
        'direct_peer_access': {str(peer): 1 for peer in range(3) if peer != rank}}
    env = report['environment']
    assert env['HCCL_OP_EXPANSION_MODE'] == 'AIV'
    assert env['HCCL_DETERMINISTIC'] == 'true'
    assert env['HCCL_ALGO'] is None and env['HCCL_ALGO_CONFIG'] is None
    assert report['cpu_contract']['pass']
    assert [shape['rows'] for shape in report['shapes']] == [1, 12]
    identities.append({'rank': rank, 'report_sha256': digest(path),
                       'loaded_libraries': report['loaded_libraries']})
    for shape in report['shapes']:
        rows = shape['rows']
        assert shape['order_pass'] and shape['edge_hccl_equal']
        assert shape['all_consumers_pass']
        parent_addresses, candidate_addresses = (
            shape[name + '_addresses'] for name in ('parent', 'candidate'))
        assert all(parent_addresses[key] == candidate_addresses[key]
                   for key in ('partial', 'weight'))
        addresses = [item[key] for item in (parent_addresses, candidate_addresses)
                     for key in ('first', 'output', 'consumer')]
        assert len(set(addresses)) == 6
        assert not set(addresses) & {parent_addresses['partial'], parent_addresses['weight']}
        assert [(case['family'], case['iteration']) for case in shape['checks']] == [
            (family, iteration) for family in ('order', 'edges') for iteration in range(6)]
        for case in shape['checks']:
            family, iteration = case['family'], case['iteration']
            assert case['mode'] == ('eager' if iteration < 2 else 'graph')
            if family == 'order':
                scale = 2.0 ** (iteration - 2)
                inputs = [[value * scale for value in values] for values in
                          ((1, 1, 1), (2.0 ** -25, -1, 2), (-1, 2.0 ** -25, 3))]
            else:
                columns = list(edges[-iteration:] + edges[:-iteration]) if iteration else list(edges)
                inputs = [[bf16_value(values[peer]) for _, values in columns]
                          for peer in range(3)]
            serial_float = [f32(f32(a + b) + c) for a, b, c in zip(*inputs)]
            tree_float = [f32(f32(a + c) + b) for a, b, c in zip(*inputs)]
            serial, tree = ([bf16_bits(value) for value in values]
                            for values in (serial_float, tree_float))
            source = [bf16_bits(value) for value in inputs[rank]]
            assert case['hccl_input_first_32_fp32_bits'] == first_32(
                [f32_bits(value) for value in inputs[rank]])
            assert case['hccl_sum_first_32_fp32_bits'] == first_32(
                [f32_bits(value) for value in tree_float])
            expected = {
                'input_unchanged': (source, source),
                'parent_vs_serial': (serial, serial),
                'parent_consumer': (serial, serial),
                'candidate_vs_tree': (tree, tree),
                'candidate_consumer': (tree, tree),
                'hccl_vs_tree': (tree, tree),
                'candidate_vs_hccl': (tree, tree),
                'parent_vs_hccl': (serial, tree),
            }
            for key, (actual, reference) in expected.items():
                assert case[key] == comparison(actual, reference, rows), (
                    rank, rows, family, iteration, key)
            cases.append({'rank': rank, 'rows': rows, 'family': family,
                          'iteration': iteration, 'mode': case['mode'],
                          'candidate_matches_hccl': True,
                          'parent_differs_from_hccl': serial != tree})
assert len(cases) == 72
assert all(item['loaded_libraries'] == identities[0]['loaded_libraries'] for item in identities)
for filename, expected in manifest['shmem_libraries'].items():
    assert identities[0]['loaded_libraries']['/shmem/' + filename] == expected
assert identities[0]['loaded_libraries'][
    '/opt/ascend/cann-9.2.0/aarch64-linux/lib64/libhccl_alg.so'] == (
        '9662caf979f889655d98278805fae701d29040b0b87a2fc21702ba11633c2956')

launches, selections, excerpts = [], [], []
for path in sorted(RAW.rglob('*.log')):
    current = None
    for number, line in enumerate(path.open(), 1):
        selected = 'SelectAlgfor91093' in line
        launched = '[INFO]' in line and '[ExecuteKernelLaunchInner]' in line
        if not (selected or launched):
            continue
        location = {'path': str(path.relative_to(RAW)), 'line': number}
        excerpts.append(f"{location['path']}:{number}\n{line}")
        process = int(re.search(r'HCCL\((\d+),', line)[1])
        if selected:
            assert f'algName [{ALGORITHM}]' in line
            current = {**location, 'algorithm': ALGORITHM, 'process': process}
            selections.append(current)
            continue
        assert current is not None and process == current['process']
        fields = dict(re.findall(r'\b([A-Za-z]\w*)\s*\[([^\]]*)\]', line))
        for key, value in {'rankSize': '3', 'dataType': 'float32',
                           'reduceOp': 'sum', 'deterministic': '1'}.items():
            assert fields[key] == value
        assert fields['sendbuff'] == fields['recvbuff']
        rank, tag, count = (int(fields[key]) for key in ('rank', 'tag', 'count'))
        rows = 1 if tag <= 12 else 12
        assert 0 <= rank < 3 and 1 <= tag <= 24 and count == rows * 6144
        launches.append({**location, 'rank': rank, 'tag': tag, 'rows': rows,
                         'family': 'order' if (tag-1) % 12 < 6 else 'edges',
                         'iteration': (tag-1) % 6, 'fields': fields, 'selection': current})
assert len(selections) == 6 and len(launches) == 72
assert {(item['rank'], item['tag']) for item in launches} == {
    (rank, tag) for rank in range(3) for tag in range(1, 25)}
assert Counter((item['rank'], item['rows']) for item in launches) == {
    (rank, rows): 12 for rank in range(3) for rows in (1, 12)}
assert set(Counter((item['selection']['path'], item['selection']['line'])
                   for item in launches).values()) == {12}

OUTPUT.mkdir(exist_ok=False)
raw_manifest = []
for path in sorted(RAW.rglob('*')):
    if not path.is_file():
        continue
    target = OUTPUT / 'raw' / path.relative_to(RAW)
    target.parent.mkdir(parents=True, exist_ok=True)
    original_sha = digest(path)
    if path.suffix == '.log':
        target = target.with_suffix('.log.gz')
        with path.open('rb') as source, target.open('wb') as destination:
            with gzip.GzipFile(filename='', mode='wb', mtime=0, fileobj=destination) as output:
                shutil.copyfileobj(source, output)
        with gzip.open(target, 'rb') as source:
            assert hashlib.sha256(source.read()).hexdigest() == original_sha
    else:
        shutil.copy2(path, target)
        assert digest(target) == original_sha
    raw_manifest.append({'source': str(path.relative_to(RAW)),
                         'original_sha256': original_sha,
                         'original_bytes': path.stat().st_size,
                         'archive': str(target.relative_to(OUTPUT)),
                         'archive_sha256': digest(target),
                         'archive_bytes': target.stat().st_size})
assert len(raw_manifest) == 20
assert sum(item['original_bytes'] for item in raw_manifest) == 93268732
for filename in pins:
    if filename.endswith('.so'):
        continue
    target = OUTPUT / 'source' / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FIXTURE / filename, target)
for filename in [NAME + '.command.json', NAME + '.launcher.log', NAME + '.admission.jsonl',
                 'run-' + NAME + '.py', 'communication-tree-device-handoff-v1.json']:
    shutil.copy2(ROOT / filename, OUTPUT / filename)
shutil.copy2(__file__, OUTPUT / 'audit.py')
save('case-audit.json', cases)
save('launch-audit.json', {'selections': selections, 'launches': launches})
save('raw-manifest.json', raw_manifest)
(OUTPUT / 'hccl-selection-and-launch.log').write_text('\n'.join(excerpts))
summary = {
    'audit_pass': True, 'candidate_matches_hccl': True,
    'scope': 'First reduction and immediate consumer; serial/tree/HCCL on synthetic BF16 inputs',
    'ranks': 3, 'physical_dies': [4, 6, 7], 'rows': [1, 12], 'blocks': 24,
    'cases': 72, 'order_cases': 36, 'edge_cases': 36,
    'edge_columns': 32, 'eager_cases': 24, 'graph_cases': 48,
    'graph_captures_across_arms': 12, 'explicit_graph_replays_across_arms': 96,
    'full_byte_comparisons': 576, 'candidate_hccl_equal': 72,
    'parent_hccl_different': 72, 'input_unchanged': 72,
    'immediate_consumers_equal': 144, 'hccl_fp32_sum_prefixes_reconstructed': 72,
    'algorithm': ALGORITHM, 'deterministic': True,
    'selector_lines': 6, 'info_launch_lines': 72,
    'fixture_manifest_sha256': MANIFEST,
    'parent_sha256': pins['parent_probe.so'],
    'candidate_sha256': pins['candidate_probe.so'],
    'driver_sha256': pins['communication_tree_devicecheck.py'],
    'identity_by_rank': identities, 'raw_files': len(raw_manifest),
    'raw_bytes': sum(item['original_bytes'] for item in raw_manifest),
    'archive_bytes': sum(item['archive_bytes'] for item in raw_manifest),
    'limits': ['Inputs are finite BF16; some sums overflow. NaN inputs are not covered.',
               'Only the first reduction and its consumer are compared; the second and CUBE execute.',
               'Full device tensors are not saved; scalar fixtures reconstruct recorded byte metrics and prefixes.',
               'This private three-rank probe is not a whole-layer, TP16, or timing result.'],
}
save('summary.json', summary)
print(json.dumps({key: summary[key] for key in (
    'audit_pass', 'candidate_matches_hccl', 'cases', 'edge_cases',
    'selector_lines', 'info_launch_lines', 'raw_bytes', 'archive_bytes')}, indent=2))
