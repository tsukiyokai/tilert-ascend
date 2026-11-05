"""Probe observer compilation and strict 32-row CPU HF frame staging.

The model is the real installed HF implementation with the repository's
small FLOAT fixture. This program does not execute an NPU or certify the
production checkpoint, an 8K prefix, or the optimized performance graph.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import traceback


WIDTH = 32
LAYER_COUNT = 78
COUNTS = {'hidden_states': 79, 'layer_logits': 78,
          'indices': 78, 'routes': 78}
CACHE_FIELDS = ('latent', 'rope', 'index', 'index_scale')
PROBES = (
    Path('/tmp/tilert-graph-observer-traceability-20261028.py'),
    Path('/tmp/tilert-graph-observer-traceability-default-20261028.py'),
)
PROJECT_SOURCES = (
    'tests/glm52/check_reference_runtime.py',
    'tests/glm52/run_hf_reference.py',
    'tests/glm52/run_native_decode.py',
    'tests/glm52/prefix_io.py',
    'tilert/models/glm_5_2_ascend/reference.py',
    'tilert/models/glm_5_2_ascend/reference_weights.py',
    'tilert/models/glm_5_2_ascend/ops/indexer.py',
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(encoded.encode()).hexdigest()


def source_hashes(root):
    paths = [Path(__file__).resolve(), *PROBES]
    paths.extend(root / source for source in PROJECT_SOURCES)
    paths.extend((root / 'docs/evidence/ascend-decode/c8-arch22').glob('*.h'))
    return {str(path.resolve()): digest(path) for path in sorted(paths)}


def field_tags():
    result = ['logits', 'last_hidden', 'token_ids']
    result.extend(f'{field}.{layer}' for field, count in COUNTS.items()
                  for layer in range(count))
    result.extend(f'cache_rows.{layer}.{field}'
                  for layer in range(LAYER_COUNT) for field in CACHE_FIELDS)
    return result


def value_at(trace, tag):
    if tag == 'last_hidden':
        return trace['hidden_states'][-1]
    if tag == 'token_ids':
        return trace['logits'].argmax(-1)
    if tag == 'logits':
        return trace['logits']
    parts = tag.split('.')
    value = trace[parts[0]][int(parts[1])]
    return value[parts[2]] if len(parts) == 3 else value


def frame_header(replay_id, traces, sources, checkpoint, trace_hashes):
    inputs = {
        'tokens': [trace['token'] for trace in traces],
        'positions': [trace['position'] for trace in traces],
        'committed_positions': [trace['committed_position']
                                for trace in traces],
    }
    if (len(traces) != WIDTH or inputs['positions'] != list(range(WIDTH))
            or inputs['committed_positions'] != list(range(1, WIDTH + 1))):
        raise ValueError('Expected a cold 32-row teacher sequence')
    return {
        'origin': 'real_cpu_hf_tensor_frame_staging', 'replay_id': replay_id,
        'width': WIDTH, 'layer_count': LAYER_COUNT, 'fp32_head': True,
        'mla_preprocess': 'native', 'inputs': inputs,
        'input_sha256': object_digest(inputs), 'source_sha256': sources,
        'checkpoint_sha256': checkpoint, 'trace_sha256': trace_hashes,
    }


def pack_frames(header, traces):
    frames = []
    for tag in field_tags():
        values = [value_at(trace, tag) for trace in traces]
        lengths = None
        if all(value is None for value in values):
            tensor = None
        elif any(value is None for value in values):
            raise ValueError('Mixed None and Tensor rows: ' + tag)
        elif tag.startswith('indices.'):
            lengths = [value.numel() for value in values]
            tensor = torch.full((WIDTH, max(lengths)), -1,
                                dtype=values[0].dtype)
            for row, value in enumerate(values):
                tensor[row, :value.numel()].copy_(value.reshape(-1))
        elif tag.startswith(('cache_rows.', 'routes.')):
            tensor = torch.stack(values).clone()
        else:
            tensor = torch.cat(values).clone()
        frames.append({
            'tag': tag, 'replay_id': header['replay_id'],
            'input_sha256': header['input_sha256'],
            'row_ids': list(range(WIDTH)), 'lengths': lengths,
            'tensor': tensor,
        })
    return {'header': header, 'frames': frames}


def unpack_frames(package, expected_header, expected):
    if package['header'] != expected_header:
        raise ValueError('Frame header identity differs')
    frames = {}
    for frame in package['frames']:
        tag = frame['tag']
        if tag in frames:
            raise ValueError('Duplicate field: ' + tag)
        if (frame['replay_id'] != expected_header['replay_id']
                or frame['input_sha256'] != expected_header['input_sha256']
                or frame['row_ids'] != list(range(WIDTH))):
            raise ValueError('Cross-replay field or missing row: ' + tag)
        frames[tag] = frame
    if set(frames) != set(field_tags()):
        raise ValueError('Complete 628-field coverage is required')
    values = []
    for row, reference in enumerate(expected):
        trace = {key: reference[key] for key in
                 ('token', 'position', 'committed_position')}
        trace.update({field: [None] * count
                      for field, count in COUNTS.items()})
        trace['cache_rows'] = [{} for _ in range(LAYER_COUNT)]
        for tag, frame in frames.items():
            expected_value = value_at(reference, tag)
            tensor = frame['tensor']
            if tensor is None:
                if expected_value is not None or frame['lengths'] is not None:
                    raise ValueError('Unexpected None field: ' + tag)
                value = None
            else:
                if (not isinstance(tensor, torch.Tensor) or tensor.ndim < 1
                        or tensor.shape[0] != WIDTH):
                    raise ValueError('Every tensor must carry 32 rows: ' + tag)
                if expected_value is None:
                    raise ValueError('Unexpected tensor field: ' + tag)
                if tag.startswith('indices.'):
                    lengths = [value_at(item, tag).numel()
                               for item in expected]
                    if (frame['lengths'] != lengths
                            or tensor.shape != (WIDTH, max(lengths))):
                        raise ValueError('Index lengths differ: ' + tag)
                    length = lengths[row]
                    if not bool((tensor[row, length:] == -1).all()):
                        raise ValueError('Index padding was overwritten')
                    value = tensor[row, :length].reshape(expected_value.shape)
                else:
                    if frame['lengths'] is not None:
                        raise ValueError('Unexpected ragged lengths: ' + tag)
                    value = tensor[row]
                    if not tag.startswith(('cache_rows.', 'routes.')):
                        value = value.unsqueeze(0)
                    if value.shape != expected_value.shape:
                        raise ValueError('Tensor row shape differs: ' + tag)
            if tag in ('logits', 'last_hidden', 'token_ids'):
                trace[tag] = value
            else:
                parts = tag.split('.')
                target = trace[parts[0]]
                if len(parts) == 3:
                    target[int(parts[1])][parts[2]] = value
                else:
                    target[int(parts[1])] = value
        values.append(trace)
    return values


def compare_package(label, package, header, expected, comparison, raw):
    traces = unpack_frames(package, header, expected)
    failures = []
    count = 0
    for row, (actual, reference) in enumerate(zip(traces, expected)):
        checks = []
        prefix = f'{label}.row{row}'
        comparison.compare_trace(prefix, actual, reference, checks)
        comparison.compare_tensor(
            prefix + '.last_hidden', actual['last_hidden'],
            reference['hidden_states'][-1], checks,
        )
        comparison.compare_tensor(
            prefix + '.token_ids', actual['token_ids'],
            reference['logits'].argmax(-1), checks, exact=True,
        )
        count += len(checks)
        failures.extend(check for check in checks if not check['passed'])
        raw.write(json.dumps({'label': prefix, 'checks': checks}) + '\n')
    return {'name': label, 'accepted': not failures, 'check_count': count,
            'failures': failures}


def check_negative(label, package, header, expected, comparison, raw):
    try:
        outcome = compare_package(label, package, header, expected,
                                  comparison, raw)
        outcome['passed'] = not outcome['accepted']
        return outcome
    except (ValueError, TypeError, KeyError) as error:
        return {'name': label, 'passed': True, 'accepted': False,
                'error': f'{type(error).__name__}: {error}'}


def frame_index(package, tag):
    return next(index for index, frame in enumerate(package['frames'])
                if frame['tag'] == tag)


def mutate_field(package, tag, transform):
    changed = {'header': package['header'], 'frames': list(package['frames'])}
    index = frame_index(changed, tag)
    frame = {**changed['frames'][index]}
    frame['tensor'] = transform(frame['tensor'].clone())
    changed['frames'][index] = frame
    return changed


def run_negatives(package, other, header, expected, comparison, raw):
    cases = []
    missing = {'header': header, 'frames': package['frames'][:-1]}
    duplicate = {'header': header,
                 'frames': [*package['frames'], package['frames'][0]]}
    crossing = {'header': header, 'frames': list(package['frames'])}
    index = frame_index(crossing, 'hidden_states.77')
    crossing['frames'][index] = other['frames'][index]
    disguised = {'header': header, 'frames': list(package['frames'])}
    disguised['frames'][index] = {
        **package['frames'][index],
        'tensor': other['frames'][index]['tensor'].clone(),
    }
    bad_rows = {'header': header, 'frames': list(package['frames'])}
    bad_rows['frames'][index] = {**package['frames'][index],
                                'row_ids': list(range(WIDTH - 1))}
    changes = {
        'missing_field': missing, 'duplicate_field': duplicate,
        'cross_replay_identity': crossing,
        'cross_replay_tensor_with_relabelled_identity': disguised,
        'missing_row_identity': bad_rows,
        'missing_tensor_row': mutate_field(
            package, 'logits', lambda value: value[:-1],
        ),
        'fp32_logits_cast_bf16': mutate_field(
            package, 'logits', lambda value: value.bfloat16(),
        ),
        'hidden_wrong': mutate_field(
            package, 'hidden_states.77', lambda value: value * 2,
        ),
        'route_wrong': mutate_field(
            package, 'routes.3', lambda value: value * 0,
        ),
        'index_wrong': mutate_field(
            package, 'indices.0', lambda value: value + (value >= 0),
        ),
        'cache_wrong': mutate_field(
            package, 'cache_rows.0.latent', lambda value: value + 1,
        ),
        'token_ids_wrong': mutate_field(
            package, 'token_ids', lambda value: value + 1,
        ),
        'nonfinite': mutate_field(
            package, 'logits', lambda value: value * float('nan'),
        ),
    }
    for label, changed in changes.items():
        cases.append(check_negative(label, changed, header, expected,
                                    comparison, raw))
    for key, value in (
        ('width', WIDTH - 1), ('fp32_head', False),
        ('checkpoint_sha256', {'wrong_weights': '0' * 64}),
        ('source_sha256', {'wrong_source': '0' * 64}),
        ('weight_arithmetic_sha256', '0' * 64),
        ('input_sha256', '0' * 64),
    ):
        changed = {'header': {**header, key: value},
                   'frames': package['frames']}
        cases.append(check_negative('wrong_' + key, changed, header, expected,
                                    comparison, raw))
    return cases


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(args, report):
    global torch
    import torch
    import transformers

    torch.set_num_threads(1)
    sys.path[:0] = [str(args.root), str(args.root / 'tests/glm52')]
    fixture = load_module('frame32_fixture', args.root / PROJECT_SOURCES[0])
    collector = load_module('frame32_collector',
                            args.root / PROJECT_SOURCES[1])
    comparison = load_module('frame32_comparison',
                             args.root / PROJECT_SOURCES[2])
    from tilert.models.glm_5_2_ascend.reference import HfReference
    from tilert.models.glm_5_2_ascend.reference_weights import ReferenceWeights

    report['versions'] = {'torch': torch.__version__,
                          'transformers': transformers.__version__}
    report['loaded_reference_sources'] = {
        cls.__name__: {'path': str(Path(inspect.getfile(cls)).resolve()),
                       'sha256': digest(Path(inspect.getfile(cls)))}
        for cls in (HfReference, ReferenceWeights)
    }
    for cls, name in (
        (HfReference, 'tilert/models/glm_5_2_ascend/reference.py'),
        (ReferenceWeights,
         'tilert/models/glm_5_2_ascend/reference_weights.py'),
    ):
        actual = report['loaded_reference_sources'][cls.__name__]
        expected_path = str((args.root / name).resolve())
        if (actual['path'] != expected_path or actual['sha256']
                != report['source_sha256'][expected_path]):
            raise ValueError('Loaded reference implementation differs')
    from transformers import cache_utils
    from transformers.models.glm_moe_dsa import configuration_glm_moe_dsa
    from transformers.utils import output_capturing

    report['installed_hf_source_sha256'] = {
        str(path.resolve()): digest(path) for path in (
            Path(inspect.getfile(transformers.GlmMoeDsaForCausalLM)),
            Path(configuration_glm_moe_dsa.__file__),
            Path(cache_utils.__file__), Path(output_capturing.__file__),
        )
    }
    checkpoint = args.output / 'checkpoint'
    checkpoint.mkdir()
    fixture.make_checkpoint(checkpoint)
    checkpoint_hashes = {path.name: digest(path)
                         for path in sorted(checkpoint.iterdir())}
    report['checkpoint_sha256'] = checkpoint_hashes
    c8_directory = args.root / 'docs/evidence/ascend-decode/c8-arch22'
    engine = HfReference(
        checkpoint, mla_preprocess='native', fp32_head=True,
        c8_source_directory=c8_directory,
    )
    report['weight_arithmetic'] = engine.weights.contract
    head_dtype = engine.weights.contract['lm_head']['projection_output_dtype']
    if head_dtype != 'float32':
        raise ValueError('Expected the real FP32 head arithmetic path')
    packages, truths = {}, {}
    teacher_a = [2 if row % 3 == 1 else 1 for row in range(WIDTH)]
    teacher_b = [1 if token == 2 else 2 for token in teacher_a]
    raw_path = args.output / 'strict-checks.jsonl'
    report['cases'] = []
    report['model_state_checks'] = []
    report['scalar_oracle_checks'] = []
    with raw_path.open('w') as raw:
        for replay_id, teacher in (('A0', teacher_a), ('B', teacher_b),
                                   ('A1', teacher_a)):
            engine.reset()
            state_report = {'checks': []}
            collector.check_lengths(engine, 0, state_report,
                                    replay_id + '.reset_lengths')
            if not state_report['checks'][-1]['passed']:
                raise ValueError('HF reset did not clear every cache')
            report['model_state_checks'].extend(state_report['checks'])
            directory = args.output / replay_id
            directory.mkdir()
            traces, hashes = [], {}
            for row, token in enumerate(teacher):
                trace = engine.step(token)
                if (trace['token'] != token or trace['position'] != row
                        or trace['committed_position'] != row + 1):
                    raise ValueError('HF trace differs from requested input')
                collector.validate_trace(trace, fp32_head=True)
                state_report = {'checks': []}
                collector.check_lengths(engine, row + 1, state_report,
                                        f'{replay_id}.row{row}.cache_lengths')
                if not state_report['checks'][-1]['passed']:
                    raise ValueError('HF layer cache lengths differ')
                report['model_state_checks'].extend(state_report['checks'])
                scalar_checks = []
                fixture.check_trace(trace, True, scalar_checks)
                if not all(check['passed'] for check in scalar_checks):
                    raise ValueError('Scalar fixture oracle failed')
                report['scalar_oracle_checks'].extend(
                    {**check, 'replay_id': replay_id}
                    for check in scalar_checks
                )
                path = directory / f'step-{row:05d}.pt'
                torch.save(trace, path)
                hashes[path.name] = digest(path)
                traces.append(torch.load(path, weights_only=True))
                if (row + 1) % 8 == 0:
                    print(f'{replay_id}: {row + 1}/{WIDTH}', flush=True)
            header = frame_header(
                replay_id, traces, report['source_sha256'],
                checkpoint_hashes, hashes,
            )
            arithmetic_hash = object_digest(engine.weights.contract)
            hf_sources = report['installed_hf_source_sha256']
            header.update(
                weight_arithmetic_sha256=arithmetic_hash,
                loaded_reference_sources=report['loaded_reference_sources'],
                installed_hf_source_sha256=hf_sources,
            )
            package = pack_frames(header, traces)
            path = args.output / f'frames-{replay_id}.pt'
            torch.save(package, path)
            package = torch.load(path, weights_only=True)
            packages[replay_id], truths[replay_id] = package, traces
            outcome = compare_package(replay_id, package, header, traces,
                                      comparison, raw)
            outcome['passed'] = outcome['accepted']
            outcome.update(frame_sha256=digest(path), trace_sha256=hashes)
            report['cases'].append(outcome)
        reset_pairs = zip(truths['A1'], truths['A0'])
        for row, (actual, expected) in enumerate(reset_pairs):
            checks = []
            comparison.compare_trace(f'reset.row{row}', actual, expected,
                                     checks, exact=True)
            raw.write(json.dumps({'label': f'reset.row{row}',
                                  'checks': checks}) + '\n')
            report['cases'].append({
                'name': f'reset_exact.row{row}',
                'passed': all(check['passed'] for check in checks),
                'check_count': len(checks),
                'failures': [check for check in checks if not check['passed']],
            })
        package = packages['A0']
        report['cases'].extend(run_negatives(
            package, packages['B'], package['header'], truths['A0'],
            comparison, raw,
        ))
        before = package['frames'][frame_index(package, 'hidden_states.0')]
        owned_before = before['tensor'].clone()
        truths['A0'][0]['hidden_states'][0].zero_()
        report['cases'].append({
            'name': 'disk_frame_owns_storage',
            'passed': torch.equal(before['tensor'], owned_before),
        })
    report['strict_checks_sha256'] = digest(raw_path)
    report['source_stable'] = (
        source_hashes(args.root) == report['source_sha256']
    )
    report['status'] = ('pass' if report['source_stable'] and all(
        case['passed'] for case in report['cases']
    ) else 'fail')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(
        '/Users/shanshan/repo/tile-ai/TileRT-Ascend',
    ))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    args.root, args.output = args.root.resolve(), args.output.resolve()
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'ready',
        'command': [sys.executable, *sys.argv], 'width': WIDTH,
        'field_count': len(field_tags()),
        'source_sha256': source_hashes(args.root),
        'precision_verified': False, 'formal_precision_gate_passed': False,
        'npu_executed': False, 'performance_measured': False,
        'scope': 'Real 78-layer CPU HF FLOAT fixture, native MLA, FP32 head; '
                 '32 cold teacher rows with A/B/A reset. Complete staging '
                 'and original strict thresholds; no production checkpoint, '
                 '8K prefix, NPU callback or compiled SuperKernel claim.',
    }
    if args.check_only:
        print(json.dumps(report))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'report.json'
    try:
        for index, probe in enumerate(PROBES):
            process = subprocess.run([sys.executable, str(probe)], check=True,
                                     text=True, capture_output=True)
            summary = json.loads(process.stdout.strip().splitlines()[-1])
            probe_path = Path(summary['output'])
            probe_report = json.loads(probe_path.read_text())
            expected = [False, False, True, True] if index == 0 else [True] * 4
            observed = [case['passed'] for case in probe_report['cases']]
            if observed != expected:
                raise ValueError('Observer retention result differs')
            report.setdefault('observer_cpu_probes', []).append({
                'report': str(probe_path), 'sha256': digest(probe_path),
                'expected': expected, 'observed': observed,
            })
        report['status'] = 'running'
        path.write_text(json.dumps(report, indent=2) + '\n')
        run(args, report)
    except Exception:
        report.update(status='error', error=traceback.format_exc())
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'report': str(path), 'status': report['status']}))
    raise SystemExit(report['status'] != 'pass')


if __name__ == '__main__':
    main()
