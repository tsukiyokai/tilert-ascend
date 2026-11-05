"""Relocate paths in the frozen CPU experiment and run it."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


NAMES = (
    'tilert-cann-frame32-experiment-20261028.py',
    'tilert-graph-observer-traceability-20261028.py',
    'tilert-graph-observer-traceability-default-20261028.py',
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root, workspace):
    directory = Path(__file__).resolve().parent
    evidence = root / 'docs/evidence/ascend-decode/graph-frame32-20261028'
    manifest = json.loads((evidence / 'manifest.json').read_text())
    vendor = directory / 'observer-source/msprobe'
    configured = {}
    for name in NAMES:
        frozen = directory / 'frozen' / name
        key = str(frozen.relative_to(root))
        if digest(frozen) != manifest['frozen_sources'][key]:
            raise ValueError('Frozen experiment source differs: ' + key)
        text = frozen.read_text()
        replacements = {
            '/tmp/' + other: str(workspace / other) for other in NAMES[1:]
            if '/tmp/' + other in text
        }
        if name == NAMES[1]:
            replacements.update({
                '/Users/shanshan/repo/ascend/msprobe/python/msprobe/pytorch/':
                str(vendor / 'python/msprobe/pytorch') + '/',
                '/Users/shanshan/repo/ascend/msprobe/ccsrc/aclgraph_dump/':
                str(vendor / 'ccsrc/aclgraph_dump') + '/',
            })
        report_name = name.removesuffix('.py') + '.json'
        if '/tmp/' + report_name in text:
            replacements['/tmp/' + report_name] = str(workspace / report_name)
        for previous, replacement in replacements.items():
            if text.count(previous) != 1:
                raise ValueError('Expected one path literal: ' + previous)
            text = text.replace(previous, replacement)
        target = workspace / name
        target.write_text(text)
        configured[name] = {
            'frozen_sha256': digest(frozen), 'path': str(target),
            'configured_sha256': digest(target), 'replacements': replacements,
        }
    provenance = {
        'launcher_sha256': digest(Path(__file__)),
        'scope': 'Only path literals change; frozen evidence stays unchanged',
        'sources': configured,
    }
    (workspace / 'configuration.json').write_text(
        json.dumps(provenance, indent=2) + '\n',
    )
    return workspace / NAMES[0]


def invoke(args, workspace):
    source = prepare(args.root, workspace)
    command = [sys.executable, str(source), '--root', str(args.root),
               '--output', str(args.output)]
    if args.check_only:
        command.append('--check-only')
    return subprocess.call(command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    args.root, args.output = args.root.resolve(), args.output.resolve()
    if args.check_only:
        with TemporaryDirectory(prefix='graph-frame32-sources-') as temporary:
            return invoke(args, Path(temporary))
    if args.output.exists():
        raise FileExistsError('Choose a new output directory')
    workspace = args.output.with_name(args.output.name + '-sources')
    workspace.mkdir(parents=True, exist_ok=False)
    return invoke(args, workspace)


if __name__ == '__main__':
    raise SystemExit(main())
