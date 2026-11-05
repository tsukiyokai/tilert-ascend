"""Independently audit saved native lookup samples without timing lookup.

Input reports, raw samples and build artifacts may be raw, gzip or xz.
An aligned packed reverse search defines the oracle; production lookup and
the benchmark's summary functions are never called to establish equality.
"""

import argparse
import ast
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import lzma
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import traceback


BINARY_SHA256 = (
    'f03d6bf62929b1f97f4bfe65576f49deee48b6f5f4935cea383bd3ec0294a6aa'
)
WORKLOADS = ('exact_copy', 'copy_5pct_edits', 'ambiguous_bigram',
             'random_no_match')
MODES = ('gc_on_plain', 'gc_on_observed', 'gc_off_observed')
ARMS = ('baseline', 'candidate')
PHASES = ('driver', 'setup', 'propose', 'commit', 'teardown')
MODULE = 'tilert.models.glm_5_2_ascend.lookup_draft'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def unwrap_once(value):
    if value.startswith(b'\x1f\x8b'):
        return gzip.decompress(value)
    if value.startswith(b'\xfd7zXZ\x00'):
        return lzma.decompress(value)
    return value


def unwrap(value):
    while True:
        decoded = unwrap_once(value)
        if decoded == value:
            return value
        value = decoded


def read_json(path):
    return json.loads(unwrap(path.read_bytes()))


def json_hash(value, compact=False):
    options = {'sort_keys': True, 'separators': (',', ':')} if compact else {}
    return sha256(json.dumps(value, **options).encode())


class Artifacts:
    def __init__(self, roots, root, historical=False):
        self.roots = roots
        self.root = root
        self.historical = historical
        self.revision = None
        self.receipts = {}
        self.cache = {}
        self.by_hash = defaultdict(list)
        self.archive_integrity = []
        for directory in roots:
            manifest_path = directory / 'manifest.json'
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text())
            count, raw_bytes = 0, 0
            for entry in manifest['files']:
                path = directory / entry['path']
                stored = path.read_bytes()
                require(sha256(stored) == entry['compressed_sha256'],
                        'Archive compressed bytes differ: ' + str(path))
                decoded = unwrap_once(stored)
                require(sha256(decoded) == entry['raw_sha256']
                        and len(decoded) == entry['raw_bytes'],
                        'Archive raw bytes differ: ' + str(path))
                self.by_hash[entry['raw_sha256']].append(path)
                self.by_hash[entry['compressed_sha256']].append(path)
                count += 1
                raw_bytes += len(decoded)
            self.archive_integrity.append({
                'manifest': str(manifest_path),
                'manifest_sha256': sha256(manifest_path.read_bytes()),
                'files_checked': count, 'raw_bytes_checked': raw_bytes,
            })

    def relative_source(self, origin):
        for directory in ('tilert', 'tests', 'experiments', 'docs'):
            marker = '/' + directory + '/'
            if marker in origin:
                return directory + '/' + origin.split(marker, 1)[1]
        return None

    def get(self, origin, expected):
        key = origin, expected
        if key in self.cache:
            path, value = self.cache[key]
        else:
            candidates = list(self.by_hash[expected])
            relative = self.relative_source(origin)
            if relative:
                candidates.append(self.root / relative)
            if not self.historical or not origin.startswith(
                    ('/tmp/', '/private/tmp/')):
                candidates.append(Path(origin))
            for root in self.roots:
                for suffix in ('', '.gz', '.xz'):
                    candidates.extend(root.rglob(Path(origin).name + suffix))
            found = None
            for path in dict.fromkeys(candidates):
                if not path.is_file():
                    continue
                value = path.read_bytes()
                while True:
                    if sha256(value) == expected:
                        found = path, value
                        break
                    decoded = unwrap_once(value)
                    if decoded == value:
                        break
                    value = decoded
                if found:
                    break
            if (found is None and self.historical and relative
                    and self.revision):
                process = subprocess.run(
                    ['git', 'show', self.revision + ':' + relative],
                    cwd=self.root, capture_output=True,
                )
                if (process.returncode == 0
                        and sha256(process.stdout) == expected):
                    found = ('git:' + self.revision + ':' + relative,
                             process.stdout)
            require(found is not None,
                    'Missing or changed artifact: ' + origin)
            path, value = found
            self.cache[key] = path, value
        self.receipts[origin] = {'path': str(path), 'sha256': expected}
        return value


def packed_proposal(context, packed, count=31):
    """Search longest suffix first, then the latest aligned old occurrence."""
    for length in range(min(32, len(context) - 1), 1, -1):
        pattern = struct.pack('<' + 'I' * length, *context[-length:])
        limit = (len(context) - 1) * 4
        while True:
            location = packed.rfind(pattern, 0, limit)
            if location < 0:
                break
            if location % 4 == 0:
                end = location // 4 + length
                return context[end:end + count]
            # Exclude this start while retaining earlier overlapping matches.
            limit = location + len(pattern) - 1
    return []


def oracle(case):
    prompt, output = case['prompt_token_ids'], case['output_token_ids']
    require(len(prompt) == 8192 and len(output) == 1024,
            'Workload lengths differ')
    require(all(type(token) is int and 0 <= token < 2**32
                for token in (*prompt, *output)), 'Unsupported packed token')
    context = [*prompt, output[0]]
    packed = bytearray(struct.pack('<' + 'I' * len(context), *context))
    position, rounds = 1, []
    while position < len(output):
        proposal = packed_proposal(context, packed)
        padded = proposal + [output[position - 1]] * (31 - len(proposal))
        matched = 0
        for left, right in zip(padded, output[position:]):
            if left != right:
                break
            matched += 1
        accepted = min(matched + 1, len(output) - position)
        rounds.append([position, proposal, accepted])
        committed = output[position:position + accepted]
        context.extend(committed)
        packed.extend(struct.pack('<' + 'I' * len(committed), *committed))
        position += accepted
    return rounds, json_hash(rounds)


def distribution(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    count = len(values)
    middle = count // 2
    median = (values[middle] if count % 2
              else (values[middle - 1] + values[middle]) / 2)
    mean = math.fsum(values) / count
    deviation = math.sqrt(math.fsum((value - mean)**2
                                    for value in values) / count)
    return {
        'count': count, 'p50': median,
        'p90': values[(count * 90 + 99) // 100 - 1],
        'p95': values[(count * 95 + 99) // 100 - 1],
        'p99': values[(count * 99 + 99) // 100 - 1],
        'max': values[-1], 'std': deviation,
    }


def same(actual, expected, label='summary'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(),
                label + ': key inventory differs')
        for key in expected:
            same(actual[key], expected[key], label + '.' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected),
                label + ': length differs')
        for index, (left, right) in enumerate(zip(actual, expected)):
            same(left, right, label + '.' + str(index))
    elif type(expected) in (int, float):
        require(type(actual) in (int, float)
                and math.isclose(actual, expected, rel_tol=1e-12,
                                 abs_tol=1e-12), label + ': number differs')
    else:
        require(actual == expected, label + ': value differs')


def check_gc(row, intervals):
    events = row['gc_events']
    require(len(events) % 2 == 0, 'Unpaired GC event')
    collections = []
    for first, last in zip(events[::2], events[1::2]):
        require(first['event'] == 'start' and last['event'] == 'stop'
                and first['generation'] == last['generation']
                and first['generation'] in range(3), 'GC event order differs')
        require(first['entered_ns'] <= first['left_ns']
                <= last['entered_ns'] <= last['left_ns'],
                'GC timestamps are not ordered')
        collections.append({
            'generation': first['generation'], 'phase': first['phase'],
            'stop_phase': last['phase'], 'position': first['position'],
            'start_ns': first['entered_ns'], 'stop_ns': last['entered_ns'],
            'observed_interval_ns': last['entered_ns'] - first['entered_ns'],
            'between_our_callbacks_ns': last['entered_ns'] - first['left_ns'],
            'our_callback_ns': first['left_ns'] - first['entered_ns']
            + last['left_ns'] - last['entered_ns'],
            'collected': last['collected'],
            'uncollectable': last['uncollectable'],
        })
    require(collections == row['gc_collections'], 'GC pairing differs')
    overlap = dict.fromkeys(PHASES, 0)
    for collection in collections:
        for phase, _, begin, end in intervals:
            overlap[phase] += max(0, min(end, collection['stop_ns'])
                                  - max(begin, collection['start_ns']))
    observed = sum(item['observed_interval_ns'] for item in collections)
    overlap['driver'] = observed - sum(overlap.values())
    require(overlap == row['gc_overlap_ns']
            and all(value >= 0 for value in overlap.values()),
            'GC phase overlap differs')
    deltas = []
    for before, after in zip(row['gc_stats_before'], row['gc_stats_after']):
        delta = {key: after[key] - before[key] for key in
                 ('collections', 'collected', 'uncollectable')}
        require(all(value >= 0 for value in delta.values()),
                'GC counters decreased')
        deltas.append(delta)
    require(len(deltas) == 3, 'Expected all three GC generations')
    if row['mode'] == 'gc_on_plain':
        require(not events and not collections, 'Plain arm has callbacks')
    if row['mode'] == 'gc_off_observed':
        require(not events and not any(
            delta['collections'] for delta in deltas),
                'Collection observed with GC disabled')
    return deltas, collections, overlap


def check_row(row, expected, signature):
    require(all(type(item['position']) is int
                and type(item['accepted']) is int
                and type(item['proposal']) is list
                and all(type(token) is int for token in item['proposal'])
                for item in row['rounds']), 'Invalid round field type')
    triples = [[item['position'], item['proposal'], item['accepted']]
               for item in row['rounds']]
    require(triples == expected, 'Actual round transcript differs from oracle')
    require(json_hash(triples) == row['signature_sha256'] == signature,
            'Round signature differs')
    require(all(type(item[key]) is int and item[key] >= 0
                for item in row['rounds'] for key in
                ('propose_ns', 'commit_ns')), 'Invalid call duration')
    steady = sum(item['propose_ns'] + item['commit_ns']
                 for item in row['rounds'])
    require(row['steady_ns'] == steady
            and row['total_ns'] == row['setup_ns'] + steady,
            'Call accounting differs')
    expected_report = {
        'drafter': 'lookup', 'longest_tail': 32, 'shortest_tail': 2,
        'context_tokens': 9216, 'rounds': len(expected),
        'tail_hits': sum(bool(item[1]) for item in expected),
    }
    require(row['final_report'] == expected_report,
            'Public lookup report differs')
    intervals = row['phase_intervals']
    require(len(intervals) == 2 * len(expected) + 2,
            'Incomplete call intervals')
    require(intervals[0][:2] == ['setup', -1]
            and intervals[-1][:2] == ['teardown', -1],
            'Boundary phases differ')
    require(intervals[0][3] - intervals[0][2] == row['setup_ns']
            and intervals[-1][3] - intervals[-1][2] == row['teardown_ns'],
            'Boundary duration differs')
    for index, item in enumerate(row['rounds']):
        for offset, phase in ((1, 'propose'), (2, 'commit')):
            label, position, begin, end = intervals[index * 2 + offset]
            require(label == phase and position == item['position']
                    and end - begin == item[phase + '_ns'],
                    'Per-call interval differs')
    previous = 0
    for _, _, begin, end in intervals:
        require(type(begin) is int and type(end) is int
                and previous <= begin <= end, 'Call intervals overlap')
        previous = end
    require(previous <= row['observed_wall_ns'] and row['observed_wall_ns']
            >= row['total_ns'] + row['teardown_ns'],
            'Wall accounting differs')
    return check_gc(row, intervals)


def schedule():
    for workload in WORKLOADS:
        for sample in range(-2, 30):
            shift = sample % 3
            modes = (*MODES[shift:], *MODES[:shift])
            for mode in modes:
                order = ARMS if (sample + MODES.index(mode)) % 2 == 0 else (
                    ARMS[::-1])
                for position, arm in enumerate(order):
                    yield workload, sample, mode, arm, position, order, modes


def native_inspection(binary, sources, root):
    script = '''import hashlib, importlib.util, json, pathlib, sys
name, filename = sys.argv[1:]
spec = importlib.util.spec_from_file_location(name, filename)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
cls = module.LookupDrafter
assert "__pyx_vtable__" in vars(cls)
assert "context" in vars(cls) and "hi" in vars(module._Node)
assert cls([1,2,7,9,2,8,2],3,1).propose(1) == [8]
node = module._Node(1,1,1)
try:
    node.hi = "not-an-index"
except TypeError:
    coordinate_type_checked = True
else:
    raise AssertionError("Node coordinate stayed a Python object")
assert "torch" not in sys.modules and "Cython" not in sys.modules
print(json.dumps({"identity":module._NATIVE_BUILD,
 "file":str(pathlib.Path(module.__file__).resolve()),
 "binary_sha256":hashlib.sha256(
  pathlib.Path(filename).read_bytes()).hexdigest(),
 "method_type":type(cls.propose).__name__,
 "coordinate_type_checked":coordinate_type_checked,
 "node_members":{key:type(vars(module._Node)[key]).__name__
  for key in ("hi","length","latest","children")},
 "lookup_context_member":type(vars(cls)["context"]).__name__}))
'''
    with TemporaryDirectory(prefix='native-lookup-audit-') as temporary:
        directory = Path(temporary)
        filename = directory / Path(binary[0]).name
        filename = filename.with_name(filename.name.removesuffix('.gz')
                                      .removesuffix('.xz'))
        filename.write_bytes(binary[1])
        for name, content in sources.items():
            (directory / name).write_bytes(content)
        process = subprocess.run([sys.executable, '-I', '-S', '-c', script,
                                  MODULE, str(filename)], text=True,
                                 capture_output=True, check=True)
        result = json.loads(process.stdout)
    default_script = '''import json, sys
sys.path.insert(0,sys.argv[1])
from tilert.pd_vllm.ascend_cache import drafter_metadata
print(json.dumps(drafter_metadata("lookup")))
'''
    default = subprocess.run([sys.executable, '-I', '-S', '-c', default_script,
                              str(root)], text=True, capture_output=True,
                             check=True)
    result['current_default_import'] = json.loads(default.stdout)
    return result


def verify_build(args, artifacts):
    build_bytes = unwrap(args.build.read_bytes())
    build = json.loads(build_bytes)
    require(build['status'] == 'pass'
            and build['binary_sha256'] == BINARY_SHA256,
            'Build does not bind the fixed binary')
    artifacts.revision = build['git_head']
    binary = artifacts.get(build['binary'], BINARY_SHA256)
    require(sha256(binary) == BINARY_SHA256, 'Fixed binary differs')
    identity = build['native_identity']
    artifacts.get(str(args.root / 'experiments/lookup-native/build.py'),
                  build['builder_sha256'])
    require(identity == {key: value for key, value in
                         build['isolated_load'].items()
                         if key != 'loaded_binary_sha256'}
            and build['isolated_load']['loaded_binary_sha256']
            == BINARY_SHA256,
            'Build/load identities differ')
    sources = {}
    package = args.root / 'tilert/models/glm_5_2_ascend'
    for name, checksum in identity['source_sha256'].items():
        sources[name] = artifacts.get(str(package / name), checksum)
        sibling = str(Path(build['binary']).with_name(name))
        require(artifacts.get(sibling, checksum) == sources[name],
                'Binary sibling sources differ')
    generated = artifacts.get(build['generated_c'],
                              build['generated_c_sha256'])
    generated_input = artifacts.get(
        str(Path(build['generated_c']).with_suffix('.py')),
        build['generated_input_sha256'],
    )
    require(generated_input.startswith(sources['lookup_draft.py']),
            'Compiled input changed algorithm source')
    tree = ast.parse(generated_input.decode())
    embedded = [ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name)
                        and target.id == '_NATIVE_BUILD'
                        for target in node.targets)]
    require(embedded == [identity], 'Embedded build guard identity differs')
    c_text = generated.decode()
    declarations = {
        'node_coordinates': all('Py_ssize_t ' + field + ';' in c_text
                                for field in ('hi', 'length', 'latest')),
        'native_shared': bool(re.search(
            r'static Py_ssize_t __pyx_f_\w+__shared\([^;]+Py_ssize_t',
            c_text)),
        'native_insert': bool(re.search(
            r'static void __pyx_f_\w+__insert\([^;]+Py_ssize_t', c_text)),
        'object_public_counters': all('PyObject *' + field + ';' in c_text
                                      for field in ('longest', 'shortest',
                                                    'hits', 'rounds')),
    }
    require(all(declarations.values()), 'Cython type augmentation is missing')
    if args.skip_live_load:
        live = {'reloaded_native': False,
                'reason': 'Byte/statistics audit; ABI not exercised',
                'historical_isolated_load': build['isolated_load']}
    else:
        live = native_inspection((build['binary'], binary), sources, args.root)
        require(live['identity'] == build['isolated_load']
                and live['binary_sha256'] == BINARY_SHA256,
                'Live extension identity differs')
        live['reloaded_native'] = True
    return build, sha256(build_bytes), {
        'type_declarations': declarations, 'isolated_binary': live,
    }


def verify_checks(path, build_hash, identity, artifacts, root):
    content = unwrap(path.read_bytes())
    checks = json.loads(content)
    require(checks['status'] == 'pass' and checks['source_stable'] is True
            and checks['build_receipt_sha256'] == build_hash,
            'Checks do not bind the fixed build')
    artifacts.get(str(root / 'experiments/lookup-native/check.py'),
                  checks['checker_sha256'])
    require(checks['guards'] == [
        {'mutation': name, 'rejected': True} for name in
        ('before_py', 'before_pxd', 'after_py', 'after_pxd', 'after_binary')
    ], 'Native guard rejection inventory differs')
    require([suite['suite'] for suite in checks['suites']]
            == ['basic', 'http'],
            'Native suite inventory differs')
    outcome = []
    for receipt in checks['suites']:
        report = json.loads(unwrap(artifacts.get(
            receipt['report'], receipt['report_sha256'],
        )))
        expected_count = 72 if receipt['suite'] == 'basic' else 7
        require(receipt['returncode'] == 0 and report['status'] == 'pass'
                and report['source_stable'] is True
                and report['suite'] == receipt['suite']
                and report['tests_run'] == expected_count
                and report['build_receipt_sha256'] == build_hash
                and report['checker_sha256'] == checks['checker_sha256']
                and report['native'] == identity
                and report['npu_executed'] is False,
                'Native suite receipt differs')
        require(len(report['source_sha256']) == 91,
                'Native suite source inventory differs')
        for origin, checksum in report['source_sha256'].items():
            artifacts.get(origin, checksum)
        entry = {'suite': receipt['suite'], 'tests': expected_count,
                 'report_sha256': receipt['report_sha256']}
        if receipt['suite'] == 'basic':
            require(report['bad_compression']['inner_prefix_error_rejected']
                    is True and report['proposals']['proposal_checks'] == 24159
                    and report['proposals']['invariant_prefix_checks']
                    == 217685
                    and len(report['sessions']) == 126,
                    'Native semantic coverage receipt differs')
            for session in report['sessions']:
                require(session['output_token_ids']
                        == session['expected_token_ids']
                        and session['lookup_report']['context_tokens']
                        == session['valid_cache_rows'] + 1,
                        'Saved scalar session differs')
            artifacts.get(str(root / 'experiments/lookup-index/audit.py'),
                          report['oracle_auditor_sha256'])
            entry.update(proposals=report['proposals'], scalar_sessions=126)
        outcome.append(entry)
    return sha256(content), outcome


def sample_identity(row):
    return {key: row[key] for key in ('workload', 'mode', 'arm', 'sample')}


def tail_record(row, item, phase, deltas, collections):
    interval = next(interval for interval in row['phase_intervals']
                    if interval[:2] == [phase, item['position']])
    observed = row['mode'] != 'gc_on_plain'
    overlaps = [{**collection, 'overlap_ns': max(
        0, min(interval[3], collection['stop_ns'])
        - max(interval[2], collection['start_ns']),
    )} for collection in collections]
    overlaps = [collection for collection in overlaps
                if collection['overlap_ns']]
    return {
        **sample_identity(row), 'position': item['position'],
        'accepted': item['accepted'], 'proposal_length': len(item['proposal']),
        'duration_ns': item[phase + '_ns'],
        'request_collections_by_generation': [
            delta['collections'] for delta in deltas],
        'direct_gc_overlap_ns': (sum(item['overlap_ns'] for item in overlaps)
                                 if observed else None),
        'direct_gc_collections': overlaps,
    }


def accumulate(groups, pairs, gc_groups, row, deltas, collections, overlap):
    key = row['workload'], row['mode'], row['arm']
    group = groups[key]
    for name in ('setup', 'steady', 'total', 'teardown'):
        group[name + '_ms'].append(row[name + '_ns'] / 1e6)
    for phase in ('propose', 'commit'):
        group[phase + '_us'].extend(item[phase + '_ns'] / 1e3
                                    for item in row['rounds'])
    group['gc_overlap_ms'].append(sum(
        overlap[phase] for phase in ('setup', 'propose', 'commit')) / 1e6)
    group['collection_count'].append(sum(delta['collections']
                                         for delta in deltas))
    group['observed_collection_count'].append(len(collections))
    gc_group = gc_groups[key]
    for generation, delta in enumerate(deltas):
        for field, value in delta.items():
            gc_group['counter_totals'][generation][field] += value
    gc_group['observed_generations'].update(
        item['generation'] for item in collections)
    for phase, value in overlap.items():
        gc_group['overlap_ns'][phase] += value
    for item in collections:
        for field in ('observed_interval_ns', 'between_our_callbacks_ns',
                      'our_callback_ns'):
            gc_group[field].append(item[field])
    for phase in ('propose', 'commit'):
        biggest = max(row['rounds'], key=lambda item: item[phase + '_ns'])
        gc_group['largest_calls'][phase].append(tail_record(
            row, biggest, phase, deltas, collections,
        ))
    pair_key = row['workload'], row['mode'], row['sample']
    require(row['arm'] not in pairs[pair_key], 'Duplicate measured pair arm')
    pairs[pair_key][row['arm']] = {
        field: row[field + '_ns'] for field in ('setup', 'steady', 'total')
    }


def new_gc_group():
    return {
        'counter_totals': [Counter() for _ in range(3)],
        'observed_generations': Counter(), 'overlap_ns': Counter(),
        'observed_interval_ns': [], 'between_our_callbacks_ns': [],
        'our_callback_ns': [], 'largest_calls': {'propose': [], 'commit': []},
    }


def summarize_groups(groups, pairs, gc_groups):
    results, gc_results, paired_losses = [], [], []
    for workload in WORKLOADS:
        for mode in MODES:
            arms = {arm: {name: distribution(values) for name, values in
                          groups[workload, mode, arm].items()} for arm in ARMS}
            ratio, wins = {}, {}
            for field in ('setup', 'steady', 'total'):
                values, outcomes = [], Counter()
                for sample in range(30):
                    pair = pairs[workload, mode, sample]
                    require(set(pair) == set(ARMS), 'Incomplete measured pair')
                    baseline = pair['baseline'][field]
                    candidate = pair['candidate'][field]
                    require(candidate > 0, 'Invalid paired duration')
                    values.append(baseline / candidate)
                    result = ('wins' if candidate < baseline else
                              'losses' if candidate > baseline else 'ties')
                    outcomes[result] += 1
                    if result != 'wins':
                        paired_losses.append({
                            'workload': workload, 'mode': mode,
                            'metric': field, 'sample': sample,
                            'baseline_ns': baseline, 'candidate_ns': candidate,
                            'result': result,
                        })
                ratio[field] = distribution(values)
                wins[field] = {name: outcomes[name]
                               for name in ('wins', 'ties', 'losses')}
            regressions = []
            for metric in ('setup_ms', 'steady_ms', 'total_ms', 'teardown_ms',
                           'propose_us', 'commit_us'):
                for statistic in ('p50', 'p90', 'p95', 'p99', 'max', 'std'):
                    before = arms['baseline'][metric][statistic]
                    after = arms['candidate'][metric][statistic]
                    if after > before:
                        regressions.append({
                            'metric': metric, 'stat': statistic,
                            'baseline': before, 'candidate': after,
                        })
            results.append({
                'workload': workload, 'mode': mode, 'arms': arms,
                'paired_baseline_over_candidate': ratio, 'paired_wins': wins,
                'candidate_regressions': regressions,
            })
            for arm in ARMS:
                group = gc_groups[workload, mode, arm]
                gc_results.append({
                    'workload': workload, 'mode': mode, 'arm': arm,
                    'counter_totals': group['counter_totals'],
                    'observed_generations': dict(
                        group['observed_generations']),
                    'overlap_ns': dict(group['overlap_ns']),
                    'collection_timings': {field: distribution(group[field])
                                           for field in (
                        'observed_interval_ns', 'between_our_callbacks_ns',
                        'our_callback_ns',
                    )},
                    'largest_request_calls': {
                        phase: sorted(values, key=lambda item:
                                      item['duration_ns'], reverse=True)[:5]
                        for phase, values in group['largest_calls'].items()
                    },
                })
    return results, gc_results, paired_losses


def audit_panel(panel, report_path, raw_path, artifacts, oracle_cache,
                build_hash, checks_hash, identity):
    report_data = unwrap(report_path.read_bytes())
    report = json.loads(report_data)
    require(report['status'] == 'pass' and report['source_stable'] is True
            and report['head_stable'] is True and report['gc_start'] == panel
            and report['npu_executed'] is False
            and report['performance_measured'] is True,
            'Panel scope or completion differs')
    parameters = report['parameters']
    require(parameters == {
        'width': 32, 'longest': 32, 'shortest': 2, 'prompt_length': 8192,
        'output_length': 1024, 'samples': 30, 'warmup': 2, 'gc_start': panel,
        'modes': list(MODES), 'gc_thresholds': [700, 10, 10],
    }, 'Panel parameters differ')
    require(json_hash(parameters, compact=True) == report['parameters_sha256'],
            'Parameter digest differs')
    require(report['native']['file_sha256'] == BINARY_SHA256
            and report['native']['build']['loaded_binary_sha256']
            == BINARY_SHA256 and report['native']['kind'] == 'cython'
            and report['baseline']['kind'] == 'python'
            and report['same_algorithm_source'] is True,
            'Panel arm identities differ')
    require(report['native'] == identity
            and report['baseline']['file_sha256']
            == identity['build']['source_sha256']['lookup_draft.py']
            == report['source_sha256'][report['baseline']['file']],
            'Baseline/native source identity differs')
    source_hashes = report['source_sha256']
    require([value for name, value in source_hashes.items()
             if name.endswith('/build.json')] == [build_hash]
            and [value for name, value in source_hashes.items()
                 if name.endswith('/checks.json')] == [checks_hash],
            'Panel build/checks binding differs')
    for origin, checksum in source_hashes.items():
        artifacts.get(origin, checksum)
    cases = report['workloads']
    require([case['name'] for case in cases] == list(WORKLOADS),
            'Panel workload inventory differs')
    signatures = {}
    reference_path = [name for name in source_hashes
                      if name.endswith('rfind-cpu.json.gz')]
    require(len(reference_path) == 1, 'Frozen reference identity is missing')
    reference_bytes = unwrap(artifacts.get(reference_path[0],
                                           source_hashes[reference_path[0]]))
    require(sha256(reference_bytes) == report['reference_decoded_sha256'],
            'Decoded frozen reference differs')
    reference = json.loads(reference_bytes)
    require(reference['status'] == 'pass', 'Frozen reference did not pass')
    for case, frozen in zip(cases, reference['results']):
        require(all(case[key] == frozen[key] for key in
                    ('name', 'prompt_token_ids', 'output_token_ids')),
                'Frozen workload tokens differ')
        key = json_hash([case['prompt_token_ids'], case['output_token_ids']])
        if key not in oracle_cache:
            oracle_cache[key] = oracle(case)
        rounds, signature = oracle_cache[key]
        require(signature == case['oracle_signature_sha256']
                == frozen['oracle']['signature_sha256'],
                'Independent reverse-search signature differs')
        signatures[case['name']] = rounds, signature
    metrics = ('setup_ms', 'steady_ms', 'total_ms', 'teardown_ms',
               'propose_us', 'commit_us', 'gc_overlap_ms', 'collection_count',
               'observed_collection_count')
    groups = defaultdict(lambda: {name: [] for name in metrics})
    pairs, gc_groups = defaultdict(dict), defaultdict(new_gc_group)
    raw = unwrap(raw_path.read_bytes())
    require(sha256(raw) == report['raw_samples_sha256'], 'Raw samples changed')
    expected_order = list(schedule())
    lines = raw.splitlines()
    require(len(lines) == report['raw_sample_count'] == len(expected_order)
            == 768, 'Panel sample matrix is incomplete')
    warmup, round_count, events = 0, 0, 0
    for ordinal, (line, order) in enumerate(zip(lines, expected_order)):
        row = json.loads(line)
        require(all(type(row[key]) is int for key in
                    ('sample', 'arm_order', 'ordinal'))
                and type(row['warmup']) is bool,
                'Invalid sample metadata type')
        workload, sample, mode, arm, arm_order, permutation, modes = order
        require((row['workload'], row['sample'], row['mode'], row['arm'],
                 row['arm_order'], row['arm_permutation'], row['mode_order'],
                 row['ordinal'], row['warmup']) == (
                    workload, sample, mode, arm, arm_order, list(permutation),
                    list(modes), ordinal, sample < 0),
                'Sample permutation or matrix differs at ' + str(ordinal))
        require((row['pre_collected'] is None) == (panel == 'carry'),
                'Fresh/carry precollection differs')
        rounds, signature = signatures[workload]
        deltas, collections, overlap = check_row(row, rounds, signature)
        round_count += len(rounds)
        events += len(row['gc_events'])
        warmup += row['warmup']
        if not row['warmup']:
            accumulate(groups, pairs, gc_groups, row, deltas, collections,
                       overlap)
    results, gc_results, paired_losses = summarize_groups(
        groups, pairs, gc_groups)
    same(results, report['results'], panel)
    return {
        'panel': panel, 'report_sha256': sha256(report_data),
        'raw_sha256': sha256(raw), 'sample_rows': len(lines),
        'warmup_rows': warmup, 'measured_rows': len(lines) - warmup,
        'rounds_checked': round_count, 'gc_events_checked': events,
        'signatures': {name: {'sha256': value[1], 'rounds': len(value[0])}
                       for name, value in signatures.items()},
        'summary_matches': True, 'results': results,
        'gc_statistics': gc_results, 'paired_nonwins': paired_losses,
        'reported_head': report['git_head'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument('--fresh', type=Path, default=Path(
        '/tmp/tilert-lookup-native-fresh-20261028.json',
    ))
    parser.add_argument('--carry', type=Path, default=Path(
        '/tmp/tilert-lookup-native-carry-20261028.json',
    ))
    parser.add_argument('--fresh-raw', type=Path)
    parser.add_argument('--carry-raw', type=Path)
    parser.add_argument('--build', type=Path, default=Path(
        '/tmp/tilert-lookup-native-final-20261028/build.json',
    ))
    parser.add_argument('--checks', type=Path, default=Path(
        '/tmp/tilert-lookup-native-final-checks-20261028/checks.json',
    ))
    parser.add_argument('--artifacts', type=Path, action='append', default=[])
    parser.add_argument('--archive', type=Path,
                        help='Use archived reports, raw samples and artifacts')
    parser.add_argument('--skip-live-load', action='store_true',
                        help='Audit bytes/statistics; do not execute this ABI')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.archive:
        args.archive = args.archive.resolve()
        args.artifacts.insert(0, args.archive)
        args.fresh = args.archive / 'fresh.json.gz'
        args.carry = args.archive / 'carry.json.gz'
        args.fresh_raw = args.archive / 'fresh.samples.jsonl.gz'
        args.carry_raw = args.archive / 'carry.samples.jsonl.gz'
        args.build = args.archive / 'build/build.json.gz'
        args.checks = args.archive / 'checks/checks.json.gz'
    report = {
        'status': 'running',
        'scope': 'Independent saved CPU performance audit',
        'command': [sys.executable, *sys.argv], 'npu_executed': False,
        'performance_rerun': False, 'hf_executed': False,
        'auditor_sha256': sha256(Path(__file__).read_bytes()),
    }
    sources = [Path(__file__).resolve(),
               *(args.root / 'experiments/lookup-native' / name for name in (
                   'build.py', 'check.py', 'probe.py', 'install.py',
               )),
               *(args.root / 'tilert/models/glm_5_2_ascend' / name
                 for name in ('lookup_draft.py', 'lookup_draft.pxd')),
               args.root / 'tilert/pd_vllm/ascend_cache.py']
    report['audited_source_sha256'] = {
        str(path): sha256(path.read_bytes()) for path in sources
    }
    try:
        artifacts = Artifacts(args.artifacts, args.root, bool(args.archive))
        build, build_hash, inspected = verify_build(args, artifacts)
        identity = read_json(args.fresh)['native']
        require(identity['build'] == build['isolated_load'],
                'Fresh panel differs from fixed build')
        checks_hash, checks = verify_checks(args.checks, build_hash, identity,
                                            artifacts, args.root)
        report.update(build_receipt_sha256=build_hash,
                      binary_sha256=BINARY_SHA256,
                      checks_receipt_sha256=checks_hash,
                      native_inspection=inspected, semantic_receipts=checks)
        report['panels'] = []
        oracle_cache = {}
        for panel, path, raw in (
            ('fresh', args.fresh, args.fresh_raw),
            ('carry', args.carry, args.carry_raw),
        ):
            raw = raw or Path(read_json(path)['raw_samples'])
            report['panels'].append(audit_panel(
                panel, path, raw, artifacts, oracle_cache, build_hash,
                checks_hash, identity,
            ))
        report['verified_artifacts'] = artifacts.receipts
        report['archive_integrity'] = artifacts.archive_integrity
        report['auditor_sources_stable'] = all(
            sha256(Path(path).read_bytes()) == checksum for path, checksum in
            report['audited_source_sha256'].items()
        )
        require(report['auditor_sources_stable'], 'Audited sources changed')
        report['status'] = 'pass'
    except Exception:
        report.update(status='fail', error=traceback.format_exc())
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)
                           + '\n')
    print(json.dumps({'status': report['status'], 'output': str(args.output),
                      'error': report.get('error')}))
    return int(report['status'] != 'pass')


if __name__ == '__main__':
    raise SystemExit(main())
