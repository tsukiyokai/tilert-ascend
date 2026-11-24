"""Recompute the rejected pilot from compressed raw rank reports."""
import gzip
import json
from pathlib import Path
import shutil
import sys
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'raw/source-pilot'))
import device_bench


def main():
    source = ROOT / 'pilot/tp16-pilot-v1'
    expected = json.loads((source / 'summary.json').read_text())
    with tempfile.TemporaryDirectory(prefix='trt-pilot-readback-') as directory:
        output = Path(directory) / 'pilot'
        shutil.copytree(source, output)
        for archive in output.glob('world-*.tar.zst'):
            subprocess.run(['tar', '--zstd', '-xf', str(archive), '-C', str(output)], check=True)
        for path in output.rglob('*.gz'):
            path.with_suffix('').write_bytes(gzip.decompress(path.read_bytes()))
        actual = device_bench.summarize(output)
        assert actual == expected
    return {'raw_statistics_reproduce': True, 'world_pairs': actual['world_pairs'],
            'accepted': actual['accepted'], 'pilot_only': actual['pilot_only'],
            'launch_skew_valid': {rows: value['launch_skew_valid']
                                  for rows, value in actual['rows'].items()},
            'scope': 'continuous 32-probe averages; submitted envelope invalid; no performance win'}


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
