"""Read every compressed byte and check the complete saved check records."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path


PHASES = ('A0', 'B', 'A1')
NEGATIVES = {
    'missing_field', 'duplicate_field', 'cross_replay_identity',
    'cross_replay_tensor_with_relabelled_identity', 'missing_row_identity',
    'missing_tensor_row', 'fp32_logits_cast_bf16', 'hidden_wrong',
    'route_wrong', 'index_wrong', 'cache_wrong', 'token_ids_wrong',
    'nonfinite', 'wrong_width', 'wrong_fp32_head',
    'wrong_checkpoint_sha256', 'wrong_source_sha256',
    'wrong_weight_arithmetic_sha256', 'wrong_input_sha256',
}
NUMERICAL_NEGATIVES = {
    'cross_replay_tensor_with_relabelled_identity',
    'fp32_logits_cast_bf16', 'hidden_wrong', 'route_wrong', 'index_wrong',
    'cache_wrong', 'token_ids_wrong', 'nonfinite',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decoded_digest(path):
    checksum, size = hashlib.sha256(), 0
    with gzip.open(path, 'rb') as source:
        while block := source.read(1024 * 1024):
            checksum.update(block)
            size += len(block)
    return checksum.hexdigest(), size


def read_json(path):
    with gzip.open(path, 'rt') as source:
        return json.load(source)


def verify_archive(root, directory, manifest):
    entries = {}
    for entry in manifest['files']:
        path = directory / entry['path']
        require(entry['path'] not in entries, 'Duplicate archive path')
        require(digest(path) == entry['compressed_sha256'],
                'Compressed hash differs: ' + entry['path'])
        checksum, size = decoded_digest(path)
        require((checksum, size) == (entry['raw_sha256'], entry['raw_bytes']),
                'Decompressed bytes differ: ' + entry['path'])
        entries[entry['path']] = entry
    require(set(entries) == {
        str(path.relative_to(directory)) for path in directory.rglob('*.gz')
    }, 'Compressed file inventory differs')
    for relative, checksum in manifest['frozen_sources'].items():
        require(digest(root / relative) == checksum,
                'Frozen source differs: ' + relative)
    for entry in manifest['observer_source_files']:
        require(digest(root / entry['path']) == entry['sha256'],
                'Archived observer source differs')
    return entries


def verify_report(report, entries, manifest):
    require(report['status'] == 'pass' and report['source_stable'] is True,
            'Incomplete original run')
    require(report['width'] == 32 and report['field_count'] == 628,
            'Expected 32 rows and 628 fields')
    require(report['precision_verified'] is False
            and report['formal_precision_gate_passed'] is False
            and report['npu_executed'] is False
            and report['performance_measured'] is False,
            'CPU scope was changed')
    cases = {case['name']: case for case in report['cases']}
    expected = {*PHASES, *(f'reset_exact.row{row}' for row in range(32)),
                *NEGATIVES, 'disk_frame_owns_storage'}
    require(len(cases) == len(report['cases']) and set(cases) == expected,
            'Case inventory differs')
    require(all(case['passed'] is True for case in cases.values()),
            'An experiment case failed')
    for phase in PHASES:
        require(cases[phase]['accepted'] is True
                and cases[phase]['check_count'] == 25216,
                'Incomplete accepted phase')
        require(entries[f'frames-{phase}.pt.gz']['raw_sha256']
                == cases[phase]['frame_sha256'], 'Frame hash differs')
        hashes = cases[phase]['trace_sha256']
        require(set(hashes) == {f'step-{row:05d}.pt' for row in range(32)},
                'HF teacher row inventory differs')
        for name, checksum in hashes.items():
            require(entries[f'{phase}/{name}.gz']['raw_sha256'] == checksum,
                    'HF Tensor artifact differs')
    for name in NEGATIVES:
        require(cases[name]['accepted'] is False, 'Negative case was accepted')
        if name in NUMERICAL_NEGATIVES:
            require(cases[name]['failures']
                    and cases[name]['check_count'] == 25216,
                    'Incomplete numerical rejection')
        else:
            require(isinstance(cases[name].get('error'), str)
                    and bool(cases[name]['error']),
                    'Incomplete structural rejection')
    for name, checksum in report['checkpoint_sha256'].items():
        require(entries[f'checkpoint/{name}.gz']['raw_sha256'] == checksum,
                'Checkpoint payload differs')
    require(entries['strict-checks.jsonl.gz']['raw_sha256']
            == report['strict_checks_sha256'], 'Strict record hash differs')
    project_sources = {entry['origin'] for entry in entries.values()
                       if entry['path'].startswith('sources/project/')}
    require(project_sources <= report['source_sha256'].keys(),
            'Required project source identity is missing')
    private_sources = [Path(origin).name for origin in report['source_sha256']
                       if origin not in project_sources]
    require(len(private_sources) == len(manifest['frozen_sources'])
            and set(private_sources) == {
                Path(name).name for name in manifest['frozen_sources']
            }, 'Required frozen source inventory differs')
    for origin, checksum in report['source_sha256'].items():
        copied = [entry for entry in entries.values()
                  if entry['origin'] == origin]
        if copied:
            require(len(copied) == 1 and copied[0]['raw_sha256'] == checksum,
                    'Bound source snapshot differs')
        else:
            frozen = [value for name, value in
                      manifest['frozen_sources'].items()
                      if Path(name).name == Path(origin).name]
            require(frozen == [checksum], 'Frozen run source binding differs')
    hf_sources = {entry['origin'] for entry in entries.values()
                  if entry['path'].startswith('sources/transformers/')}
    require(set(report['installed_hf_source_sha256']) == hf_sources
            and len(hf_sources) == 4, 'Installed HF source inventory differs')
    for origin, checksum in report['installed_hf_source_sha256'].items():
        copied = [entry for entry in entries.values()
                  if entry['origin'] == origin]
        require(len(copied) == 1 and copied[0]['raw_sha256'] == checksum,
                'Installed HF source snapshot differs')
    loaded = report['loaded_reference_sources']
    require(set(loaded) == {'HfReference', 'ReferenceWeights'},
            'Loaded reference classes differ')
    for name, suffix in (
        ('HfReference', '/tilert/models/glm_5_2_ascend/reference.py'),
        ('ReferenceWeights',
         '/tilert/models/glm_5_2_ascend/reference_weights.py'),
    ):
        origins = [origin for origin in project_sources
                   if origin.endswith(suffix)]
        require(len(origins) == 1 and loaded[name]['path'] == origins[0]
                and loaded[name]['sha256']
                == report['source_sha256'][origins[0]],
                'Loaded reference class source differs')
    state = report['model_state_checks']
    require(len(state) == 99, 'Expected 99 model state checks')
    state_names = {
        **{phase + '.reset_lengths': 0 for phase in PHASES},
        **{f'{phase}.row{row}.cache_lengths': row + 1
           for phase in PHASES for row in range(32)},
    }
    require({check['name'] for check in state} == state_names.keys(),
            'Repeated or missing cache state check')
    for check in state:
        position = check['expected_position']
        require(type(position) is int
                and position == state_names[check['name']]
                and check['passed'] is True
                and check['actual_position'] == position
                and len(check['mla_lengths']) == 78
                and all(value == position for value in check['mla_lengths'])
                and check['index_lengths'] == [
                    position if layer < 3 or (layer - 2) % 4 == 0 else None
                    for layer in range(78)
                ], 'Saved cache length evidence differs')
    require(len(report['scalar_oracle_checks']) == 629
            and all(check['passed'] is True
                    for check in report['scalar_oracle_checks']),
            'Scalar oracle evidence differs')
    require(len({(check['replay_id'], check['name'])
                 for check in report['scalar_oracle_checks']}) == 629,
            'Repeated scalar oracle check')
    return cases


def verify_strict_records(directory, cases):
    counts, rows, failed = Counter(), defaultdict(set), defaultdict(list)
    total, passed = 0, 0
    with gzip.open(directory / 'strict-checks.jsonl.gz', 'rt') as source:
        for line in source:
            record = json.loads(line)
            label = record['label']
            group, row = label.rsplit('.row', 1)
            require(int(row) in range(32) and row not in rows[group],
                    'Repeated or invalid row in strict records')
            rows[group].add(row)
            checks = record['checks']
            require(len({check['name'] for check in checks}) == len(checks),
                    'Duplicate check name')
            require(all(type(check['passed']) is bool for check in checks),
                    'Invalid check result')
            counts[group] += len(checks)
            total += len(checks)
            passed += sum(check['passed'] for check in checks)
            failed[group].extend(check for check in checks
                                 if not check['passed'])
            if group == 'reset':
                require(all(check['exact'] is True for check in checks
                            if 'exact' in check), 'Reset exactness differs')
    for group in (*PHASES, 'reset'):
        require(rows[group] == {str(row) for row in range(32)}
                and not failed[group], 'Positive/reset rows are incomplete')
    for group in counts:
        if group in PHASES:
            expected_count = cases[group]['check_count']
        elif group == 'reset':
            expected_count = sum(cases[f'reset_exact.row{row}']['check_count']
                                 for row in range(32))
        else:
            require(group in NEGATIVES, 'Unknown strict record group')
            expected_count = cases[group]['check_count']
            require(failed[group] == cases[group]['failures'],
                    'Raw negative failures differ from report')
        require(counts[group] == expected_count, 'Strict check count differs')
    require(set(counts) == {*PHASES, 'reset', *NUMERICAL_NEGATIVES},
            'Strict record group inventory differs')
    require(total == 297632 and total - passed == 256,
            'Strict check totals differ')
    return {'total': total, 'passed': passed,
            'failed_expected_negative': total - passed,
            'unexpected_failed': 0, 'checks_by_group': dict(counts),
            'rows_by_group': {key: len(value) for key, value in rows.items()}}


def verify_observers(directory, manifest):
    outcomes = []
    for suffix, expected in (
        ('', [False, False, True, True]), ('-default', [True] * 4),
    ):
        name = 'tilert-graph-observer-traceability' + suffix + '-20261028'
        path = directory / 'observer' / (name + '.json.gz')
        report = read_json(path)
        for origin, checksum in report['source_sha256'].items():
            archived = [entry['sha256'] for entry in
                        manifest['observer_source_files']
                        if entry['origin'] == origin]
            if not archived:
                archived = [value for relative, value in
                            manifest['frozen_sources'].items()
                            if Path(relative).name == Path(origin).name]
            require(archived == [checksum], 'Observer source binding differs')
        observed = [case['passed'] for case in report['cases']]
        require(observed == expected, 'Observer trap/control differs')
        for case, should_survive in zip(report['cases'], expected):
            require(len(case['calls']) == (2 if should_survive else 0),
                    'CPU observer execution differs')
            require(case['outputs'] == [[3., 5.], [7., 9.]],
                    'Observer changed CPU model outputs')
            if should_survive:
                require([call['values'] for call in case['calls']]
                        == [[2., 4.], [6., 8.]], 'CPU observations differ')
        outcomes.append({'report': str(path), 'expected': expected,
                         'observed': observed, 'passed': True})
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    directory = (
        args.root / 'docs/evidence/ascend-decode/graph-frame32-20261028'
    )
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    entries = verify_archive(args.root, directory, manifest)
    report = read_json(directory / 'report.json.gz')
    cases = verify_report(report, entries, manifest)
    result = {
        'status': 'pass', 'audit_source_sha256': digest(Path(__file__)),
        'manifest_sha256': digest(manifest_path),
        'archive_files': len(entries),
        'raw_bytes': sum(entry['raw_bytes'] for entry in entries.values()),
        'compressed_bytes': sum(entry['compressed_bytes']
                                for entry in entries.values()),
        'cases': len(cases), 'rejected_negative_cases': len(NEGATIVES),
        'real_hf_steps': 96, 'state_checks': 99, 'scalar_oracle_checks': 629,
        'strict_records': verify_strict_records(directory, cases),
        'observer_controls': verify_observers(directory, manifest),
        'precision_verified': False, 'npu_executed': False,
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
