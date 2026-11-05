"""Audit saved node-GC comparisons without rerunning their performance.

The shared native auditor supplies the independent packed oracle, call/GC
accounting and complete distributions. This entry binds the two GC builds,
their numerical suites and the saved constructor-lifetime matrix.
"""

import argparse
import ast
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import traceback


ROOT = Path(__file__).resolve().parents[2]
HELPER = Path(__file__).resolve().parents[1] / 'lookup-native/audit.py'
SPEC = importlib.util.spec_from_file_location('native_lookup_auditor', HELPER)
shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shared)
require = shared.require
BINARIES = {
    'baseline': shared.BINARY_SHA256,
    'candidate':
        '37f5e5f88deb5a2789c8f0053a58d1b31d7a2512d128940f75256b596c579fe0',
}
METRICS = ('setup_ms', 'steady_ms', 'total_ms', 'teardown_ms', 'propose_us',
           'commit_us', 'gc_overlap_ms', 'collection_count',
           'observed_collection_count')


def input_path(path, archives):
    origin = str(path).replace('/private/tmp/', '/tmp/')
    for archive in archives:
        manifest_path = archive / 'manifest.json'
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        matches = [archive / entry['path'] for entry in manifest['files']
                   if entry['origin'].replace('/private/tmp/', '/tmp/')
                   == origin]
        require(len(matches) <= 1, 'Archive input is ambiguous: ' + origin)
        if matches:
            return matches[0]
    raise ValueError('Archived input is absent: ' + origin)


def build_audit(arm, path, artifacts, root):
    data = shared.unwrap(path.read_bytes())
    build = json.loads(data)
    require(build['status'] == 'pass'
            and build['binary_sha256'] == BINARIES[arm],
            'Unexpected fixed binary: ' + arm)
    artifacts.revision = build['git_head']
    binary = artifacts.get(build['binary'], BINARIES[arm])
    identity = build['native_identity']
    require(identity.get('acyclic_nodes', False) == (arm == 'candidate')
            and 'lowering_sha256' not in identity
            and build['isolated_load'] == {
                **identity, 'loaded_binary_sha256': BINARIES[arm],
            }, 'Unexpected GC build identity')
    builder = artifacts.get(str(root / 'experiments/lookup-native/build.py'),
                            build['builder_sha256'])
    guard = next(node for node in ast.parse(builder).body
                 if isinstance(node, ast.FunctionDef)
                 and node.name == 'native_guard')
    namespace = {}
    exec(compile(ast.Module(body=[guard], type_ignores=[]),
                 '<bound-native-guard>', 'exec'), namespace)
    sources = {name: artifacts.get(str(Path(build['binary']).with_name(name)),
                                   checksum) for name, checksum in
               identity['source_sha256'].items()}
    require(set(sources) == {'lookup_draft.py', 'lookup_draft.pxd'},
            'Unexpected Python/type source inventory')
    original = sources['lookup_draft.py'].decode()
    lowered = original
    if arm == 'candidate':
        require(original.count('class _Node:') == 1,
                'Expected one node definition')
        lowered = original.replace(
            'class _Node:',
            'import cython\n\n\n@cython.no_gc\n@cython.cclass\nclass _Node:',
            1,
        )
    generated_input = artifacts.get(
        str(Path(build['generated_c']).with_suffix('.py')),
        build['generated_input_sha256'],
    )
    require(generated_input.decode() == lowered
            + namespace['native_guard'](identity),
            'Generated input changes more than the GC lowering/identity')
    generated = artifacts.get(build['generated_c'],
                              build['generated_c_sha256']).decode()
    flags = {}
    for name in ('_Node', 'LookupDrafter'):
        pattern = (r'static PyTypeObject __pyx_type_\w+_' + name
                   + r'\s*=\s*\{(.*?)\n\};')
        match = re.search(pattern, generated, re.S)
        require(match is not None, 'Generated type definition is absent')
        flags[name] = 'Py_TPFLAGS_HAVE_GC' in match[1]
    require(flags == {'_Node': arm == 'baseline', 'LookupDrafter': True}
            and all('Py_ssize_t ' + field + ';' in generated
                    for field in ('hi', 'length', 'latest')),
            'Native GC or coordinate declarations differ')
    return build, sources, binary, {
        'receipt_sha256': shared.sha256(data),
        'binary_sha256': BINARIES[arm], 'type_gc_flags': flags,
        'builder_sha256': build['builder_sha256'],
        'generated_input_sha256': shared.sha256(generated_input),
        'generated_c_sha256': build['generated_c_sha256'],
        'prototype_lowering_is_formal_input': False,
    }


def native_identity(build):
    filename = build['binary']
    sources = {filename: build['binary_sha256']}
    sources.update({str(Path(filename).with_name(name)): checksum
                    for name, checksum in
                    build['native_identity']['source_sha256'].items()})
    return {'module': shared.MODULE, 'class': 'LookupDrafter',
            'kind': 'cython',
            'file': filename, 'file_sha256': build['binary_sha256'],
            'source_sha256': sources, 'build': build['isolated_load']}


def inspect_pair(builds, sources, binaries):
    script = '''import gc,hashlib,importlib.util,json,pathlib,sys,weakref
modules={}
for arm,filename in zip(("baseline","candidate"),sys.argv[1:]):
 name="tilert.models.glm_5_2_ascend.lookup_draft"
 spec=importlib.util.spec_from_file_location(name,filename)
 module=importlib.util.module_from_spec(spec)
 sys.modules[name]=module
 spec.loader.exec_module(module)
 modules[arm]=module
assert modules["baseline"] is not modules["candidate"]
assert modules["baseline"]._Node is not modules["candidate"]._Node
assert (modules["baseline"].LookupDrafter
 is not modules["candidate"].LookupDrafter)
result={}
for arm,module in modules.items():
 node=module._Node(0,0,-1)
 assert gc.is_tracked(node) is (arm=="baseline")
 drafter=module.LookupDrafter([1,2,7,9,2,8,2],3,1)
 assert gc.is_tracked(drafter) and drafter.propose(1)==[8]
 assert type(drafter._root) is module._Node
 assert "__pyx_vtable__" in vars(module.LookupDrafter)
 reference=weakref.ref(drafter)
 drafter.cycle=drafter
 del drafter
 gc.collect()
 assert reference() is None
 result[arm]={"identity":module._NATIVE_BUILD,
  "node_tracked":gc.is_tracked(node),"drafter_cycle_released":True}
assert "torch" not in sys.modules and "Cython" not in sys.modules
print(json.dumps({"distinct_modules_and_types":True,"arms":result}))
'''
    with TemporaryDirectory(prefix='lookup-gc-audit-') as temporary:
        filenames = []
        for arm in shared.ARMS:
            directory = Path(temporary) / arm
            directory.mkdir()
            path = directory / Path(builds[arm]['binary']).name
            path.write_bytes(binaries[arm])
            filenames.append(str(path))
            for name, value in sources[arm].items():
                (directory / name).write_bytes(value)
        process = subprocess.run([sys.executable, '-I', '-S', '-c', script,
                                  *filenames], check=True, text=True,
                                 capture_output=True)
    result = json.loads(process.stdout)
    for arm in shared.ARMS:
        require(result['arms'][arm]['identity']
                == builds[arm]['isolated_load'],
                'Reloaded module differs from the measured identity')
    return {'reloaded_native': True, **result}


def lifetime_audit(path, summaries, artifacts, root, reference_hash):
    data = shared.unwrap(path.read_bytes())
    saved = json.loads(data)
    require(saved['status'] == 'pass'
            and saved['binary_sha256'] == BINARIES
            and saved['build_receipts_sha256'] == {
                arm: value['receipt_sha256']
                for arm, value in summaries.items()
            } and saved['reference_decoded_sha256'] == reference_hash,
            'Lifetime evidence does not bind both final binaries')
    artifacts.get(str(root / 'experiments/lookup-gc/verify.py'),
                  saved['verifier_sha256'])
    names = (*shared.WORKLOADS, 'deep_repeated_tail')
    matrix = [(arm, name, cycle) for arm in shared.ARMS for name in names
              for cycle in (False, True)]
    require([(row['arm'], row['workload'], row['cycle_on_drafter'])
             for row in saved['cases']] == matrix,
            'Lifetime matrix is incomplete or duplicated')
    for row in saved['cases']:
        tracked = row['arm'] == 'baseline'
        require(type(row['cycle_on_drafter']) is bool and row['nodes'] > 0
                and row['tracked_nodes'] == (row['nodes'] if tracked else 0)
                and (tracked or row['tracked_child_maps'] == 0)
                and row['constructor_live_bytes'] > 0
                and row['constructor_remaining_bytes'] == 0
                and row['weakref_cleared'] is True,
                'Lifetime release/collection evidence differs')
    for baseline, candidate in zip(saved['cases'][:10], saved['cases'][10:]):
        require(all(baseline[key] == candidate[key] for key in (
            'workload', 'cycle_on_drafter', 'nodes', 'maximum_depth',
        )), 'GC lowering changed the saved tree census')
    return {'receipt_sha256': shared.sha256(data), 'cases': saved['cases'],
            'verifier_sha256': saved['verifier_sha256']}


def panel_audit(panel, path, raw_path, artifacts, cache, summaries, builds,
                checks_hash):
    data = shared.unwrap(path.read_bytes())
    report = json.loads(data)
    require(report['status'] == 'pass' and report['source_stable'] is True
            and report['head_stable'] is True
            and report['screening_only'] is False
            and report['npu_executed'] is False
            and report['gc_start'] == panel,
            'Formal panel completion or scope differs')
    require(report['parameters'] == {
        'width': 32, 'longest': 32, 'shortest': 2, 'prompt_length': 8192,
        'output_length': 1024, 'samples': 30, 'warmup': 2, 'gc_start': panel,
        'modes': list(shared.MODES), 'gc_thresholds': [700, 10, 10],
    } and report['same_python_and_pxd'] is True,
        'Formal panel parameters differ')
    for arm in shared.ARMS:
        require(report['native_builds'][arm] == {
            'receipt_sha256': summaries[arm]['receipt_sha256'],
            'binary': builds[arm]['binary'], 'binary_sha256': BINARIES[arm],
            'loaded_identity': builds[arm]['isolated_load'],
            'node_tracked': arm == 'baseline', 'drafter_tracked': True,
        }, 'Panel build/load/GC identity differs')
    for origin, checksum in report['source_sha256'].items():
        artifacts.get(origin, checksum)
    require(len(report['source_sha256']) == 19
            and [value for name, value in report['source_sha256'].items()
                 if name.endswith('/checks.json')] == [checks_hash]
            and [value for name, value in report['source_sha256'].items()
                 if name.endswith('/build.json')] == [
                     summaries[arm]['receipt_sha256'] for arm in shared.ARMS
                 ], 'Panel build/checks input binding differs')
    reference_paths = [name for name in report['source_sha256']
                       if name.endswith('rfind-cpu.json.gz')]
    require(len(reference_paths) == 1, 'Frozen reference identity is absent')
    name = reference_paths[0]
    reference_data = shared.unwrap(artifacts.get(
        name, report['source_sha256'][name],
    ))
    reference_hash = shared.sha256(reference_data)
    require(reference_hash == report['reference_decoded_sha256'],
            'Frozen reference decoded bytes differ')
    reference = json.loads(reference_data)
    require(reference['status'] == 'pass'
            and [case['name'] for case in report['workloads']]
            == list(shared.WORKLOADS), 'Workload inventory differs')
    signatures = {}
    for case, frozen in zip(report['workloads'], reference['results']):
        require(all(case[key] == frozen[key] for key in
                    ('name', 'prompt_token_ids', 'output_token_ids')),
                'Frozen workload tokens differ')
        key = shared.json_hash([case['prompt_token_ids'],
                                case['output_token_ids']])
        if key not in cache:
            cache[key] = shared.oracle(case)
        rounds, signature = cache[key]
        require(signature == case['oracle_signature_sha256']
                == frozen['oracle']['signature_sha256'],
                'Independent oracle signature differs')
        signatures[case['name']] = rounds, signature
    raw_path = raw_path or Path(report['raw_samples'])
    raw = shared.unwrap(raw_path.read_bytes())
    require(shared.sha256(raw) == report['raw_samples_sha256'],
            'Raw sample bytes differ')
    lines, order = raw.splitlines(), list(shared.schedule())
    require(len(lines) == len(order) == report['raw_sample_count'] == 768,
            'Formal sample matrix is incomplete')
    groups = defaultdict(lambda: {metric: [] for metric in METRICS})
    pairs, gc_groups = defaultdict(dict), defaultdict(shared.new_gc_group)
    round_count = events = warmup = 0
    event_counter_differences = []
    for ordinal, (line, expected) in enumerate(zip(lines, order)):
        row = json.loads(line)
        workload, sample, mode, arm, position, permutation, modes = expected
        require(all(type(row[key]) is int for key in
                    ('sample', 'arm_order', 'ordinal'))
                and type(row['warmup']) is bool
                and (row['workload'], row['sample'], row['mode'], row['arm'],
                     row['arm_order'], row['arm_permutation'],
                     row['mode_order'],
                     row['ordinal'], row['warmup']) == (
                    workload, sample, mode, arm, position, list(permutation),
                    list(modes), ordinal, sample < 0),
                'Sample order/identity differs at ' + str(ordinal))
        require((row['pre_collected'] is None) == (panel == 'carry'),
                'Fresh/carry collection boundary differs')
        rounds, signature = signatures[workload]
        deltas, collections, overlap = shared.check_row(row, rounds, signature)
        round_count += len(rounds)
        events += len(row['gc_events'])
        warmup += row['warmup']
        if mode == 'gc_on_observed':
            observed = Counter(item['generation'] for item in collections)
            counters = [delta['collections'] for delta in deltas]
            if counters != [observed[generation] for generation in range(3)]:
                event_counter_differences.append({
                    'ordinal': ordinal, 'counter_deltas': counters,
                    'observed': [observed[generation]
                                 for generation in range(3)],
                })
        if not row['warmup']:
            shared.accumulate(groups, pairs, gc_groups, row, deltas,
                              collections, overlap)
    results, gc_results, nonwins = shared.summarize_groups(
        groups, pairs, gc_groups)
    shared.same(results, report['results'], panel)
    return {
        'panel': panel, 'report_sha256': shared.sha256(data),
        'raw_sha256': shared.sha256(raw), 'sample_rows': len(lines),
        'warmup_rows': warmup, 'measured_rows': len(lines) - warmup,
        'rounds_checked': round_count, 'gc_events_checked': events,
        'reference_decoded_sha256': reference_hash,
        'signatures': {name: {'sha256': value[1], 'rounds': len(value[0])}
                       for name, value in signatures.items()},
        'summary_matches': True, 'results': results,
        'gc_statistics': gc_results, 'paired_nonwins': nonwins,
        'event_counter_differences': event_counter_differences,
        'reported_head': report['git_head'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    for panel in ('fresh', 'carry'):
        parser.add_argument('--' + panel, type=Path, default=Path(
            '/tmp/tilert-lookup-gc-final-' + panel + '-20261028.json'))
        parser.add_argument('--' + panel + '-raw', type=Path)
    for name, value in {
        'baseline-build':
            '/tmp/tilert-lookup-native-final-20261028/build.json',
        'candidate-build':
            '/tmp/tilert-lookup-node-gc-off-final-20261028/build.json',
        'baseline-checks':
            '/tmp/tilert-lookup-native-final-checks-20261028/checks.json',
        'candidate-checks':
            '/tmp/tilert-lookup-node-gc-off-final-checks-20261028/checks.json',
        'lifetime': '/tmp/tilert-lookup-node-gc-final-lifetime-20261028.json',
    }.items():
        parser.add_argument('--' + name, type=Path, default=Path(value))
    parser.add_argument('--artifacts', type=Path, action='append', default=[])
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--skip-live-load', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'status': 'running', 'scope': 'Saved native node-GC CPU audit',
              'performance_rerun': False, 'npu_executed': False,
              'command': [sys.executable, *sys.argv]}
    monitored = [Path(__file__), HELPER,
                 *(args.root / 'experiments/lookup-gc' / name
                   for name in ('verify.py', 'probe.py')),
                 args.root / 'experiments/lookup-native/build.py',
                 *(args.root / 'tilert/models/glm_5_2_ascend' / name
                   for name in ('lookup_draft.py', 'lookup_draft.pxd'))]
    report['audit_sources_sha256'] = {
        str(path): shared.sha256(path.read_bytes()) for path in monitored
    }
    try:
        old_archive = args.root / (
            'docs/evidence/ascend-decode/lookup-native-20261028'
        )
        if args.archive:
            args.archive = args.archive.resolve()
            args.artifacts.insert(0, args.archive)
            for name in ('fresh', 'carry', 'baseline_build', 'candidate_build',
                         'baseline_checks', 'candidate_checks', 'lifetime'):
                setattr(args, name, input_path(
                    getattr(args, name), [*args.artifacts, old_archive],
                ))
            for panel in ('fresh', 'carry'):
                raw = getattr(args, panel + '_raw') or Path(
                    shared.read_json(getattr(args, panel))['raw_samples'])
                setattr(args, panel + '_raw', input_path(
                    raw, [*args.artifacts, old_archive],
                ))
        artifacts = shared.Artifacts([*args.artifacts, old_archive], args.root,
                                     historical=bool(args.archive))
        builds, sources, binaries, summaries = {}, {}, {}, {}
        for arm in shared.ARMS:
            outcome = build_audit(arm, getattr(args, arm + '_build'),
                                  artifacts, args.root)
            builds[arm], sources[arm], binaries[arm], summaries[arm] = outcome
        require(builds['baseline']['native_identity'] == {
            key: value for key, value in builds['candidate']['native_identity']
            .items() if key != 'acyclic_nodes'
        }, 'GC arms differ in arithmetic/type/compiler identity')
        require(all(builds['baseline'][field] == builds['candidate'][field]
                    for field in ('compiler_version', 'compiler_config',
                                  'compiler_environment')),
                'GC arms use different compiler settings')
        report['builds'] = summaries
        report['native_inspection'] = (
            {'reloaded_native': False,
             'reason': 'Explicit ABI/byte-only audit'}
            if args.skip_live_load else inspect_pair(builds, sources, binaries)
        )
        report['regression_receipts'] = {}
        for arm in shared.ARMS:
            checksum, suites = shared.verify_checks(
                getattr(args, arm + '_checks'),
                summaries[arm]['receipt_sha256'],
                native_identity(builds[arm]), artifacts, args.root,
            )
            report['regression_receipts'][arm] = {
                'receipt_sha256': checksum, 'suites': suites,
            }
        cache = {}
        report['panels'] = [panel_audit(
            panel, getattr(args, panel), getattr(args, panel + '_raw'),
            artifacts,
            cache, summaries, builds,
            report['regression_receipts']['candidate']['receipt_sha256'],
        ) for panel in ('fresh', 'carry')]
        report['lifetime'] = lifetime_audit(
            args.lifetime, summaries, artifacts, args.root,
            report['panels'][0]['reference_decoded_sha256'],
        )
        report['verified_artifacts'] = [
            {'origin': origin, 'sha256': checksum, 'path': str(value[0])}
            for (origin, checksum), value in artifacts.cache.items()
        ]
        report['archive_integrity'] = artifacts.archive_integrity
        report['audit_sources_stable'] = all(
            shared.sha256(Path(path).read_bytes()) == checksum
            for path, checksum in report['audit_sources_sha256'].items()
        )
        require(report['audit_sources_stable'], 'Audited sources changed')
        report['status'] = 'pass'
    except Exception:
        report.update(status='fail', error=traceback.format_exc())
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)
                           + '\n')
    print(json.dumps({'status': report['status'], 'output': str(args.output),
                      'error': report.get('error')}), flush=True)
    return int(report['status'] != 'pass')


if __name__ == '__main__':
    raise SystemExit(main())
