"""Reproduce the recorded TP2/TP4 numerical and source-identity checks."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'raw'
PREPARED = RAW / 'build/rsag-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_report(path):
    content = path.read_bytes() if path.is_file() else gzip.decompress(
        path.with_suffix('.json.gz').read_bytes())
    return json.loads(content)


def main():
    manifest = json.loads((PREPARED / 'manifest.json').read_text())
    build = json.loads((PREPARED / 'build-report.json').read_text())
    assert build['prepared_manifest_sha256'] == digest(PREPARED / 'manifest.json')
    for relative, expected in manifest['sources'].items():
        assert digest(PREPARED / relative) == expected, relative
    source = RAW / 'source/experiments/communication-rsag'
    for relative, expected in manifest['experiment_sources'].items():
        assert digest(source / relative) == expected, relative
    records, cross_arm = [], 0
    for ranks in (2, 4):
        reports = {}
        for arm in ('control', 'candidate'):
            for rank in range(ranks):
                path = RAW / f'receipts/rsag/tp{ranks}/{arm}/ranks/{arm}-rank-{rank}.json'
                report = load_report(path)
                assert report['pass'] and report['rank'] == rank and report['ranks'] == ranks
                assert report['manifest_sha256'] == digest(PREPARED / 'manifest.json')
                assert report['build_report_sha256'] == digest(PREPARED / 'build-report.json')
                assert report['library_sha256'] == build['arms'][arm]['library']['sha256']
                assert report['driver_sha256'] == manifest['experiment_sources']['device_check.py']
                count = 0
                for shape in report['shapes']:
                    assert shape['pass']
                    for check in shape['checks']:
                        assert check['guard_regions_intact']
                        assert all(check[name]['bitwise_equal'] for name in (
                            'input', 'weight', 'first', 'output'))
                        count += 1
                reports[arm, rank] = report
                records.append({'ranks': ranks, 'arm': arm, 'rank': rank, 'checks': count})
        for rank in range(ranks):
            for control, candidate in zip(reports['control', rank]['shapes'],
                                          reports['candidate', rank]['shapes']):
                assert control['rows'] == candidate['rows']
                assert len(control['checks']) == len(candidate['checks'])
                for left, right in zip(control['checks'], candidate['checks']):
                    assert (left['iteration'], left['mode']) == (right['iteration'], right['mode'])
                    for name in ('input', 'weight', 'first', 'output'):
                        assert left[name]['actual_sha256'] == right[name]['actual_sha256']
                    cross_arm += 1
    return {'scope': 'TP2/TP4 synthetic complete probe numerical gate; no timing',
            'pass': True, 'rank_reports': records,
            'numerical_records': sum(item['checks'] for item in records),
            'cross_arm_records': cross_arm,
            'performance_measured': False, 'tp16_verified': False,
            'hccl_tree_equivalence_verified': False}


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
