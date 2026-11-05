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
args = parser.parse_args()
started = time.time()
index = json.loads((args.model / 'quant_model_weights.safetensors.index.json').read_text())
weight_map = index['weight_map']
files = sorted(set(weight_map.values()))
args.output.mkdir(parents=True, exist_ok=True)


def inspect(name):
    path = args.model / name
    size = path.stat().st_size
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
    result = {'name': name, 'resolved_path': str(path.resolve()), 'bytes': size,
              'sha256': sha.hexdigest(), 'header_sha256': hashlib.sha256(raw).hexdigest(),
              'tensors': len(names), 'header_and_index_pass': True}
    (args.output / (name + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(name, 'pass', size, flush=True)
    return result


with ThreadPoolExecutor(max_workers=2) as pool:
    reports = list(pool.map(inspect, files))
summary = {'scope': 'actual file integrity and provenance; no model numerical gate',
           'indexed_shards': len(files), 'indexed_tensors': len(weight_map),
           'bytes': sum(item['bytes'] for item in reports), 'files': reports,
           'elapsed_seconds': time.time() - started, 'pass': True,
           'upstream_weight_hashes_verified': False,
           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
