"""Derive a three-rank FP32-HCCL fixture from the accepted TP graph fixture."""

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PARENT_MANIFEST = "546f56c9bc0bb47d67d3848da7acf3d64f66409fde53784b11986aad24d27ea7"
PARENT_LIBRARY = "c2b72e469c3ee80a5fe7d9df4a3cceedc0e65cca95e6f426aea762c213392462"
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace(source, old, new, count=1):
    if source.count(old) != count:
        raise ValueError(f"The frozen source anchor changed: {old!r}")
    return source.replace(old, new)


def replace_function(source, name, replacement):
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one function: {name}")
    node = matches[0]
    lines = source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1]) + replacement + ''.join(lines[node.end_lineno:])


def chain_source(source, candidate_sha):
    source = replace_function(source, 'ordered_sum', '')
    source = replace_function(source, 'check_rank_witnesses', '')
    source = replace(source, 'from tp_parent import separate',
                     'from tp_parent import separate\n'
                     'from tp_tree_contract import ordered_sum, check_rank_witnesses')
    replacements = (
        (PARENT_LIBRARY, candidate_sha, 1),
        ('rank not in (0, 1)', 'rank not in (0, 1, 2)', 1),
        ('get_world_size(args.oracle_hccl) != 2', 'get_world_size(args.oracle_hccl) != 3', 1),
        ('"rank": rank, "ranks": 2', '"rank": rank, "ranks": 3', 1),
        ('peers = [None, None]', 'peers = [None, None, None]', 1),
        ('library.initialize(rank, rank, 2,', 'library.initialize(rank, rank, 3,', 1),
        ('rank, 2, 32, rank, 1573184', 'rank, 3, 32, rank, 2359744', 1),
        ('if rank == iteration % 2:', 'if rank == iteration % 3:', 1),
        ('                            result.copy_(partial)\n',
         '                            fp32 = partial.float()\n', 1),
        ('                                result, op=torch.distributed.ReduceOp.SUM,',
         '                                fp32, op=torch.distributed.ReduceOp.SUM,', 1),
        ('                                group=args.oracle_hccl)\n',
         '                                group=args.oracle_hccl)\n'
         '                            result.copy_(fp32.bfloat16())\n', 1),
    )
    for old, new, count in replacements:
        source = replace(source, old, new, count)
    source = source.replace('Two TP16-shaped local shards', 'Three TP16-shaped local shards')
    source = source.replace('two TP16-shaped local shards', 'three TP16-shaped local shards')
    source = source.replace('two-rank', 'three-rank')
    source = source.replace('Use two C8 shards', 'Use three C8 shards')
    source = source.replace('HCCL2', 'FP32 HCCL3').replace('SHMEM2', 'SHMEM3')
    source = source.replace('group=actual_hccl2', 'group=actual_hccl3; BF16.float -> SUM -> BF16')
    source = source.replace('ordered BF16-input oracle', 'selected FP32 tree oracle')
    return source


def groups_source(source):
    source = source.replace('rank < 2', 'rank < 3').replace('rank >= 2', 'rank >= 3')
    source = replace(source, 'for target in (0, 1):', 'for target in (0, 1, 2):')
    source = replace(source, '[0, 1]', '[0, 1, 2]', count=3)
    source = replace(source, 'get_world_size(device) != 2', 'get_world_size(device) != 3')
    return source.replace('two-rank', 'three-rank').replace('hccl2', 'hccl3')


def parameters_source(source):
    source = replace_function(source, 'check_pair',
        'def check_pair(records):\n'
        '    from tp_tree_contract import check_parameters\n\n'
        '    check_parameters(records, SHARDED_MATRICES)\n')
    source = replace(source, 'rank not in (0, 1)', 'rank not in (0, 1, 2)')
    source = replace(source, 'value.roll(1, 0)', 'value.roll(rank, 0)')
    source = replace(source, 'peers = [None, None]', 'peers = [None, None, None]')
    return source.replace('two TP16-shaped', 'three TP16-shaped').replace(
        'two device participants', 'three device participants')


def device_source(source):
    source = replace_function(source, 'device_mapping',
        'def device_mapping(torch, args):\n'
        '    from communication_order_devicecheck import device_mapping as verify\n\n'
        '    return None, verify(torch, args.physical_devices, args.device_rank)\n')
    source = source.replace('rank < 2', 'rank < 3').replace('rank >= 2', 'rank >= 3')
    source = replace(source, '"mc2_singletons": [[0], [1]], "hccl_oracle": [0, 1]',
                     '"mc2_singletons": [[0], [1], [2]], "hccl_oracle": [0, 1, 2]')
    source = replace(source, 'type=int, nargs=2, required=True', 'type=int, nargs=3, required=True')
    source = replace(source, 'args.physical_devices[0] >= args.physical_devices[1]',
                     'any(left >= right for left, right in\n'
                     '            zip(args.physical_devices, args.physical_devices[1:]))')
    return source.replace('Two controlled', 'Three controlled').replace(
        'two device participants and fourteen', 'three device participants and thirteen').replace(
        'HCCL2', 'FP32 HCCL3').replace('SHMEM2', 'SHMEM3')


def constructor_source(source):
    source = replace(source, 'torch.npu.device_count() != 2', 'torch.npu.device_count() != 3')
    return source.replace('exactly two NPUs', 'exactly three NPUs').replace(
        'two-device', 'three-device').replace('two_rank', 'three_rank')


def nonmember_source(source):
    replacements = [('world_size=3', 'world_size=4'), ('if rank == 2:', 'if rank == 3:'),
                    ('new_group([0, 1]', 'new_group([0, 1, 2]')]
    for old, new in replacements:
        source = replace(source, old, new)
    source = replace(source, 'range(3)', 'range(4)', count=2)
    return source.replace('three-rank CPU', 'four-rank CPU').replace('global rank2', 'global rank3')


def launcher_source(source, deploy_name):
    source = replace(source, '[[ $# -lt 3 ]]', '[[ $# -lt 4 ]]')
    source = replace(source, 'PHYSICAL_SECOND REPORT_NAME', 'PHYSICAL_SECOND PHYSICAL_THIRD REPORT_NAME')
    source = replace(source, 'report_name=$3\nshift 3',
                     'physical_third=$3\nreport_name=$4\nshift 4')
    source = replace(source, '|| ! $report_name =~',
                     '|| ! $physical_third =~ ^[0-9]+$ || ! $report_name =~')
    source = replace(source, 'vendor-hook-tp-graph-v1', deploy_name)
    source = replace(source, '--device "/dev/davinci$physical_second"',
                     '--device "/dev/davinci$physical_second" --device "/dev/davinci$physical_third"')
    source = replace(source, 'env -u ASCEND_DOCKER_RUNTIME',
                     'HCCL_OP_EXPANSION_MODE=AIV HCCL_DETERMINISTIC=true \\\n'
                     '        ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=0 \\\n'
                     '        env -u ASCEND_DOCKER_RUNTIME -u ASCEND_RT_VISIBLE_DEVICES \\\n'
                     '        -u HCCL_ALGO -u HCCL_ALGO_CONFIG')
    source = replace(source, '--physical-devices "$2" "$3"',
                     '--physical-devices "$2" "$3" "$4"')
    source = replace(source, '"${@:4}"', '"${@:5}"')
    source = replace(source, '"$physical_second" "$@"',
                     '"$physical_second" "$physical_third" "$@"')
    return source


def prepare(parent, build, probe, output, deploy_name):
    if digest(parent / 'manifest.json') != PARENT_MANIFEST:
        raise ValueError('Use the accepted two-rank graph fixture')
    pins = dict(line.split('  ', 1)[::-1] for line in
                (parent / 'frozen-inputs.sha256').read_text().splitlines())
    for name, expected in pins.items():
        if digest(parent / name) != expected:
            raise ValueError(f'Parent file changed: {name}')
    build_report = json.loads((build / 'results/build.json').read_text())
    candidate = build / 'results/layer_tp_tree_chain.so'
    candidate_sha = digest(candidate)
    if candidate_sha != build_report['library_sha256']:
        raise ValueError('The tree layer library differs from its build')
    output.mkdir(parents=True, exist_ok=False)
    for name in pins:
        if name == 'manifest.json':
            continue
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, target)
    changes = {}

    def write(name, transformed):
        original = (parent / name).read_text()
        if name.endswith('.py'):
            ast.parse(transformed)
        (output / name).write_text(transformed)
        changes[name] = ''.join(difflib.unified_diff(
            original.splitlines(keepends=True), transformed.splitlines(keepends=True),
            fromfile='parent/' + name, tofile='candidate/' + name))

    for name in ('tp_chain.py', 'tp_graph.py'):
        write(name, chain_source((parent / name).read_text(), candidate_sha))
    write('tp_groups.py', groups_source((parent / 'tp_groups.py').read_text()))
    write('tp_parameters.py', parameters_source((parent / 'tp_parameters.py').read_text()))
    for name in ('tp_device.py', 'tp_graph_device.py'):
        write(name, device_source((parent / name).read_text()))
    for name in ('tp_constructor.py', 'tp_graph_constructor.py'):
        write(name, constructor_source((parent / name).read_text()))
    write('tp_nonmember_cpu.py', nonmember_source((parent / 'tp_nonmember_cpu.py').read_text()))
    for name in ('device_tp_run.sh', 'device_tp_graph_run.sh'):
        launcher = (parent / name).read_text().replace(
            'vendor-hook-tp-chain-v2', 'vendor-hook-tp-graph-v1')
        write(name, launcher_source(launcher, deploy_name))
    host = (build / 'source/host.py').read_text()
    write('tp_host.py', host)
    shutil.copy2(candidate, output / 'layer_tp_chain.so')
    for source, target in [('contract.py', 'tp_tree_contract.py'),
                           ('stage.py', 'stage_tp_tree.py'),
                           ('test_contract.py', 'test_tp_tree.py'),
                           ('cpu_check.py', 'tp_tree_cpu_check.py')]:
        shutil.copy2(HERE / source, output / target)
    for name, expected in {
        'communication_devicecheck.py': '13937a913e17a8a619a315cc7b1a9fbf833f82378c69d19e855151acc2e29255',
        'communication_order_devicecheck.py': '569733a6d85b5e61067697558b1c10faa2fc5241668939a95c1bec7c4882fe70',
    }.items():
        if digest(probe / name) != expected:
            raise ValueError('The accepted physical-device mapper changed')
        shutil.copy2(probe / name, output / name)
    (output / 'three-rank.patch').write_text('\n'.join(changes.values()))
    shutil.copy2(build / 'manifest.json', output / 'tree-build-manifest.json')
    shutil.copy2(build / 'results/build.json', output / 'tree-build-report.json')
    report = json.loads((parent / 'manifest.json').read_text())
    report.update(parent_manifest_sha256=PARENT_MANIFEST,
                  tp_library_sha256=candidate_sha,
                  tp_host_sha256=digest(output / 'tp_host.py'),
                  deployment_directory=deploy_name,
                  scope='Three TP16-shaped shards; FP32 HCCL tree versus fused tree layer',
                  device_execution=False,
                  files_sha256={str(p.relative_to(output)): digest(p)
                                for p in sorted(output.rglob('*')) if p.is_file()})
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    files = {**report['files_sha256'], 'manifest.json': digest(output / 'manifest.json')}
    (output / 'frozen-inputs.sha256').write_text(''.join(
        f'{sha}  {name}\n' for name, sha in sorted(files.items())))
    print(json.dumps({'manifest_sha256': files['manifest.json'], 'files': len(files)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'build', 'probe', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--deploy-name', default='vendor-hook-tp-tree-v1')
    args = parser.parse_args()
    prepare(args.parent.resolve(), args.build.resolve(), args.probe.resolve(),
            args.output.resolve(), args.deploy_name)
