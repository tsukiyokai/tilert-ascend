"""Check saved CPU tensors with the current receiver; never run HF."""

import argparse
import ast
import gzip
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path

import torch

from receiver import decode_frame, reference_value


DIRECTORY = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    with gzip.open(path, 'rt') as source:
        return json.load(source)


def load_tensor(path):
    with gzip.open(path, 'rb') as source:
        return torch.load(io.BytesIO(source.read()), map_location='cpu',
                          weights_only=True)


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_header(phase, report):
    tokens = [2 if row % 3 == 1 else 1 for row in range(32)]
    if phase == 'B':
        tokens = [1 if token == 2 else 2 for token in tokens]
    inputs = {'tokens': tokens, 'positions': list(range(32)),
              'committed_positions': list(range(1, 33))}

    def object_digest(value):
        text = json.dumps(value, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(text.encode()).hexdigest()

    case = next(case for case in report['cases'] if case['name'] == phase)
    return {
        'origin': 'real_cpu_hf_tensor_frame_staging', 'replay_id': phase,
        'width': 32, 'layer_count': 78, 'fp32_head': True,
        'mla_preprocess': 'native', 'inputs': inputs,
        'input_sha256': object_digest(inputs),
        'source_sha256': report['source_sha256'],
        'checkpoint_sha256': report['checkpoint_sha256'],
        'trace_sha256': case['trace_sha256'],
        'weight_arithmetic_sha256': object_digest(report['weight_arithmetic']),
        'loaded_reference_sources': report['loaded_reference_sources'],
        'installed_hf_source_sha256': report['installed_hf_source_sha256'],
    }


def scalar_oracle(directory):
    paths = {
        'collector': directory / 'sources/project/tests/glm52/'
        'run_hf_reference.py.gz',
        'fixture': directory / 'sources/project/tests/glm52/'
        'check_reference_runtime.py.gz',
    }
    namespace = {'torch': torch, 'math': math}
    for name, selected in (
        ('collector', {'validate_trace'}),
        ('fixture', {'check', 'check_trace'}),
    ):
        with gzip.open(paths[name], 'rt') as source:
            tree = ast.parse(source.read())
        functions = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name in selected]
        if {function.name for function in functions} != selected:
            raise ValueError('Archived scalar oracle source is incomplete')
        code = ast.Module(body=functions, type_ignores=[])
        exec(compile(code, str(paths[name]), 'exec'), namespace)
    return namespace['check_trace']


def reject(name, package, header, references):
    try:
        decode_frame(package, header, references)
    except (ValueError, TypeError, KeyError) as error:
        return {'name': name, 'rejected': True,
                'error': f'{type(error).__name__}: {error}'}
    raise AssertionError('Current receiver accepted ' + name)


def verify(directory):
    torch.set_num_threads(1)
    report = load_json(directory / 'report.json.gz')
    oracle = scalar_oracle(directory)
    packages, references, decoded = {}, {}, {}
    scalar_checks = []
    field_rows = 0
    for phase in ('A0', 'B', 'A1'):
        rows = [load_tensor(directory / phase / f'step-{row:05d}.pt.gz')
                for row in range(32)]
        package = load_tensor(directory / f'frames-{phase}.pt.gz')
        header = expected_header(phase, report)
        received = decode_frame(package, header, rows)
        if len(received) != 32:
            raise AssertionError('Receiver did not return 32 rows')
        for row, (actual, expected) in enumerate(zip(received, rows)):
            checks = []
            oracle(expected, True, checks)
            scalar_checks.extend({**check, 'replay_id': phase}
                                 for check in checks)
            for frame in package['frames']:
                tag = frame['tag']
                left = (actual[tag] if tag in ('last_hidden', 'token_ids')
                        else reference_value(actual, tag))
                right = reference_value(expected, tag)
                field_rows += 1
                if left is None or right is None:
                    equal = left is None and right is None
                else:
                    equal = left.dtype == right.dtype and torch.equal(
                        left, right,
                    )
                if not equal:
                    raise AssertionError(
                        f'Frame copy differs: {phase}/{row}/{tag}',
                    )
        packages[phase], references[phase], decoded[phase] = (
            package, rows, received,
        )
    if (scalar_checks != report['scalar_oracle_checks']
            or not all(check['passed'] for check in scalar_checks)):
        raise AssertionError('Scalar oracle readback differs')
    package, rows = packages['A0'], references['A0']
    header = expected_header('A0', report)
    frozen = DIRECTORY / 'frozen/tilert-cann-frame32-experiment-20261028.py'
    legacy = module_at('frozen_frame_decoder', frozen)
    legacy_empty = legacy.compare_package('empty_reference', package, header,
                                          [], None, io.StringIO())
    if legacy_empty != {'name': 'empty_reference', 'accepted': True,
                        'check_count': 0, 'failures': []}:
        raise AssertionError('Frozen empty-reference counterexample changed')
    cases = [reject('empty_reference', package, header, []),
             reject('one_reference_row', package, header, rows[:1]),
             reject('31_reference_rows', package, header, rows[:31]),
             reject('33_reference_rows', package, header, [*rows, rows[-1]])]
    aliases = []
    for key, value in (('width', 32.0), ('layer_count', 78.0),
                       ('fp32_head', 1)):
        changed = {**package, 'header': {**header, key: value}}
        aliases.append({'field': key,
                        'legacy_header_equal': changed['header'] == header})
        cases.append(reject(key + '_type_alias', changed, header, rows))
    inputs = {**header['inputs'],
              'tokens': [True, *header['inputs']['tokens'][1:]]}
    changed = {**package, 'header': {**header, 'inputs': inputs}}
    cases.append(reject('bool_input_token', changed, header, rows))
    for label, values in (
        ('float_row_ids', [float(row) for row in range(32)]),
        ('bool_row_id', [False, *range(1, 32)]),
    ):
        changed = {**package, 'frames': list(package['frames'])}
        changed['frames'][0] = {**changed['frames'][0], 'row_ids': values}
        cases.append(reject(label, changed, header, rows))
    changed = {**package, 'frames': list(package['frames'])}
    index = next(index for index, frame in enumerate(changed['frames'])
                 if frame['tag'] == 'indices.0')
    changed['frames'][index] = {
        **changed['frames'][index],
        'lengths': [float(row + 1) for row in range(32)],
    }
    cases.append(reject('float_index_lengths', changed, header, rows))
    audit = module_at('frame_archive_audit', DIRECTORY / 'audit.py')
    manifest = json.loads((directory / 'manifest.json').read_text())
    entries = {entry['path']: entry for entry in manifest['files']}
    for key, value in (
        ('source_sha256', {}), ('installed_hf_source_sha256', {}),
        ('loaded_reference_sources', {}),
        ('model_state_checks', [report['model_state_checks'][0]] * 99),
    ):
        changed_report = {**report, key: value}
        try:
            audit.verify_report(changed_report, entries, manifest)
        except (ValueError, KeyError, TypeError) as error:
            cases.append({'name': key + '_receipt_missing_or_repeated',
                          'rejected': True, 'error': str(error)})
        else:
            raise AssertionError('Audit admitted incomplete ' + key)
    before = decoded['A0'][0]['hidden_states'][0].clone()
    field = next(frame for frame in package['frames']
                 if frame['tag'] == 'hidden_states.0')
    field['tensor'].zero_()
    if not torch.equal(decoded['A0'][0]['hidden_states'][0], before):
        raise AssertionError('Receiver returned frame storage aliases')
    return {
        'status': 'pass', 'real_saved_hf_steps_read': 96,
        'received_frames': 3, 'received_rows': 96,
        'exact_field_rows_checked': field_rows,
        'scalar_oracles_recomputed': len(scalar_checks),
        'legacy_empty_reference': legacy_empty, 'legacy_type_aliases': aliases,
        'rejected_cases': cases, 'owned_row_snapshots': True,
        'hf_constructed': False, 'hf_forward_executed': False,
        'precision_verified': False, 'npu_executed': False,
        'data_manifest_sha256': digest(directory / 'manifest.json'),
        'source_sha256': {
            str(path.relative_to(DIRECTORY)): digest(path) for path in (
                Path(__file__).resolve(), DIRECTORY / 'receiver.py',
                DIRECTORY / 'audit.py', frozen,
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=DIRECTORY.parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = (
        args.root / 'docs/evidence/ascend-decode/graph-frame32-20261028'
    )
    result = verify(directory)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
