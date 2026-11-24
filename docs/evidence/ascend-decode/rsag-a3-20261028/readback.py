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
    records, cross_arm, special_pairs = [], 0, 0
    observations = []
    for phase, ranks in (("tp2", 2), ("tp4", 4), ("hccl-tp4", 4), ("hccl-tp16", 16)):
        reports = {}
        for arm in ('control', 'candidate'):
            for rank in range(ranks):
                path = RAW / f'receipts/rsag/{phase}/{arm}/ranks/{arm}-rank-{rank}.json'
                report = load_report(path)
                assert report['pass'] and report['rank'] == rank and report['ranks'] == ranks
                identity = report.get('binding', report)
                assert identity['manifest_sha256'] == digest(PREPARED / 'manifest.json')
                assert identity['build_report_sha256'] == digest(PREPARED / 'build-report.json')
                assert identity['library_sha256'] == build['arms'][arm]['library']['sha256']
                driver = (RAW / 'source-hccl/device_check.py') if phase.startswith('hccl-') else (source / 'device_check.py')
                assert report['driver_sha256'] == digest(driver)
                assert any(item['sha256'] == build['shmem_library']['sha256']
                           for item in report['loaded_shmem'])
                count = 0
                for shape in report['shapes']:
                    assert shape['pass']
                    for check in shape['checks']:
                        assert check['guard_regions_intact']
                        assert all(check[name]['bitwise_equal'] for name in (
                            'input', 'weight', 'first', 'output'))
                        count += 1
                reports[arm, rank] = report
                records.append({'phase': phase, 'ranks': ranks, 'arm': arm, 'rank': rank, 'checks': count})
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
            if phase.startswith('hccl-'):
                for control, candidate in zip(reports['control', rank]['hccl']['shapes'],
                                              reports['candidate', rank]['hccl']['shapes']):
                    assert control['rows'] == candidate['rows']
                    for left, right in zip(control['checks'], candidate['checks']):
                        assert (left['family'], left['iteration']) == (right['family'], right['iteration'])
                        for name in ('native_first_vs_tree', 'native_output_vs_tree'):
                            assert left[name]['actual_sha256'] == right[name]['actual_sha256']
                        special_pairs += 1
                if rank == 0:
                    for shape in reports['candidate', rank]['hccl']['shapes']:
                        for family in ('witness', 'finite', 'ties', 'edges'):
                            checks = [c for c in shape['checks'] if c['family'] == family]
                            observations.append({'phase': phase, 'rows': shape['rows'],
                                                 'family': family, 'checks': len(checks),
                                **{name: sum(c[name]['bitwise_equal'] for c in checks) for name in (
                                    'native_first_vs_tree', 'native_output_vs_tree',
                                    'native_first_vs_hccl', 'hccl_first_vs_tree', 'hccl_first_vs_serial')}})
    return {'scope': 'TP2/TP4/TP16 default numerical gate; special fixtures observational; no timing',
            'pass': True, 'rank_reports': records,
            'numerical_records': sum(item['checks'] for item in records),
            'cross_arm_records': cross_arm,
            'special_cross_arm_records': special_pairs, 'observations': observations,
            'performance_measured': False, 'tp16_default_numerical_gate_verified': True,
            'hccl_tree_equivalence_verified': False}


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
