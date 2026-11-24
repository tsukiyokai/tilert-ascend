"""Validate safetensors physical lengths, index coverage and actual SHA256."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import struct
import time

SIZES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1,
         'U16': 2, 'I16': 2, 'F16': 2, 'BF16': 2,
         'U32': 4, 'I32': 4, 'F32': 4, 'U64': 8, 'I64': 8, 'F64': 8}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--official-files', type=Path, required=True)
parser.add_argument('--workers', type=int, default=8)
args = parser.parse_args()
started = time.time()
index = json.loads((args.model / 'quant_model_weights.safetensors.index.json').read_text())
weight_map = index['weight_map']
files = sorted(set(weight_map.values()))
official_bytes = args.official_files.read_bytes()
assert hashlib.sha256(official_bytes).hexdigest() == 'fd225ae2ee98270f62feab18f7f75fb87e7c3bbb00e811bfb455b7155caece5f'
expected_files = {item['Path']: item for item in json.loads(official_bytes)['Data']['Files'] if item['Type'] == 'blob'}
resumed = []
args.output.mkdir(parents=True, exist_ok=True)


def inspect(name):
    path = args.model / name
    before = path.stat()
    size = before.st_size
    receipt = args.output / (name + '.json')
    if receipt.is_file():
        previous = json.loads(receipt.read_text())
        if (previous['sha256'] == expected_files[name]['Sha256']
                and previous['bytes'] == size == expected_files[name]['Size']
                and before.st_mtime_ns <= receipt.stat().st_mtime_ns
                and previous['header_and_index_pass']):
            resumed.append(name)
            return previous
    with path.open('rb') as handle:
        header_length = struct.unpack('<Q', handle.read(8))[0]
        if not 1 <= header_length <= min(64 * 1024 * 1024, size - 8):
            raise ValueError(f'Invalid header length: {name}')
        raw = handle.read(header_length)
        header = json.loads(raw)
        ranges = []
        names = []
        for key, tensor in header.items():
            if key == '__metadata__':
                continue
            begin, end = tensor['data_offsets']
            expected = math.prod(tensor['shape']) * SIZES[tensor['dtype']]
            if begin < 0 or end - begin != expected:
                raise ValueError(f'Invalid tensor byte extent: {name}/{key}')
            if weight_map.get(key) != name:
                raise ValueError(f'Index differs from actual tensor: {name}/{key}')
            ranges.append((begin, end))
            names.append(key)
        previous = 0
        for begin, end in sorted(ranges):
            if begin != previous or end < begin:
                raise ValueError(f'Non-contiguous/overlapping offsets: {name}')
            previous = end
        if previous + header_length + 8 != size:
            raise ValueError(f'Truncated or trailing data: {name}')
        if set(names) != {key for key, shard in weight_map.items() if shard == name}:
            raise ValueError(f'Missing indexed tensors: {name}')
        handle.seek(0)
        sha = hashlib.sha256()
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            sha.update(block)
    after = path.stat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), name
    assert size == expected_files[name]['Size'] and sha.hexdigest() == expected_files[name]['Sha256'], name
    result = {'stable_stat': True, 'mtime_ns': after.st_mtime_ns, 'name': name, 'resolved_path': str(path.resolve()), 'bytes': size,
              'sha256': sha.hexdigest(), 'header_sha256': hashlib.sha256(raw).hexdigest(),
              'tensors': len(names), 'header_and_index_pass': True}
    (args.output / (name + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(name, 'pass', size, flush=True)
    return result


with ThreadPoolExecutor(max_workers=args.workers) as pool:
    reports = list(pool.map(inspect, files))
summary = {'scope': 'actual file integrity and provenance; no model numerical gate',
           'indexed_shards': len(files), 'indexed_tensors': len(weight_map),
           'bytes': sum(item['bytes'] for item in reports), 'files': reports,
           'elapsed_seconds': time.time() - started, 'pass': True,
           'upstream_weight_hashes_verified': True, 'resumed_current_run_files': sorted(resumed),
           'official_files_sha256': hashlib.sha256(official_bytes).hexdigest(),
           'prior_validator_sha256': '6e511428a1bd422e9fa6eb179967389c1db55aff663de9fb96b5631a7d52bc79',
           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
