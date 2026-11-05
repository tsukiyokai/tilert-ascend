"""Load the three-rank whole-layer fixture and test its CPU contracts."""

import argparse
import hashlib
import importlib
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    lines = (root / 'frozen-inputs.sha256').read_text().splitlines()
    for line in lines:
        expected, name = line.split('  ', 1)
        if digest(root / name) != expected:
            raise ValueError(f'A frozen input changed: {name}')
    return len(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('decoder', 'runtime', 'indexed', 'shared', 'shmem', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if list(Path('/dev').glob('davinci*')):
        raise RuntimeError('The CPU gate must not map a davinci device')
    counts = {name: verify(getattr(args, name)) for name in ('decoder', 'runtime')}
    manifest = json.loads((args.decoder / 'manifest.json').read_text())
    shmem = json.loads((args.decoder / 'shmem-libraries.json').read_text())
    for name, expected in shmem.items():
        if digest(args.shmem / name) != expected:
            raise ValueError(f'A SHMEM dependency changed: {name}')
    sys.path[:0] = [str(args.decoder), str(args.runtime / 'tests/glm52/whole_layer')]
    import torch
    import torch_npu
    import local_chain
    import tp_chain
    import tp_graph_constructor
    from local_chain_host import Library
    from tp_host import Library as TPLibrary
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

    if ((torch.__version__, torch_npu.__version__) != ('2.12.0+cu130', '2.12.0')
            or torch.npu.is_initialized()):
        raise RuntimeError('Use the recorded Torch212 image without NPU initialization')
    for path, key in ((args.indexed, 'indexed_library_sha256'),
                      (args.shared, 'shared_attention_library_sha256')):
        if digest(path) != manifest[key]:
            raise ValueError('A frozen attention library changed')
        torch.ops.load_library(str(path))
    local_chain.load_ffn(args.decoder / 'complete_ffn.so')
    local = Library(args.decoder / 'layer_local_chain.so', local_chain.CHAIN_SHA)
    tp = TPLibrary(args.decoder / 'layer_tp_chain.so', tp_chain.TP_SHA)
    suite = unittest.TestSuite()
    for name in ('test_local_chain', 'test_tp_tree'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module(name)))
    graph_contracts = importlib.import_module('test_tp_graph').GraphContracts
    for name in ('test_actual_runnable_checks_window_stream_before_submission',
                 'test_released_window_and_python_fallback_are_rejected',
                 'test_capture_cannot_substitute_equal_valued_inputs',
                 'test_close_resets_graph_before_releasing_entries',
                 'test_capture_and_replay_enforce_fresh_execution_separately'):
        suite.addTest(graph_contracts(name))
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError(log.getvalue())
    from tp_nonmember_cpu import run as run_nonmember
    nonmember = run_nonmember()
    command = [sys.executable, str(args.decoder / 'tp_graph_device.py'),
               '--runtime', str(args.runtime), '--output', '/tmp/tree-not-started',
               '--extension-sha256', 'unused', '--physical-devices', '4', '6', '7',
               '--adapter', str(args.runtime / 'tilert/models/glm_5_2_ascend/attention_native.py'),
               '--indexed-library', str(args.indexed), '--shared-library', str(args.shared),
               '--sfa-library', str(args.runtime / 'native/sfa/vendor_sfa_adapter.so')]
    controls = []
    for extra, expected_code in ((['--help'], 0),
                                  (['--physical-devices', '7', '6', '4'], 2),
                                  (['--forward-widths', '3'], 2),
                                  (['--forward-replays', '0'], 2)):
        run = subprocess.run(command + extra, text=True, capture_output=True, check=False)
        if run.returncode != expected_code:
            raise RuntimeError(f'Unexpected CLI result: {run}')
        controls.append({'arguments': extra, 'exit_code': run.returncode,
                         'stdout': run.stdout, 'stderr': run.stderr})
    for name in ('device_tp_run.sh', 'device_tp_graph_run.sh'):
        subprocess.run(['bash', '-n', str(args.decoder / name)], check=True)
    if torch.npu.is_initialized() or Path('/tmp/tree-not-started').exists():
        raise RuntimeError('The CPU gate entered a device worker')
    report = {
        'pass': True, 'scope': 'Three-rank arithmetic and group contracts; five DSOs; no device',
        'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
        'files_verified': counts, 'manifest_sha256': digest(args.decoder / 'manifest.json'),
        'contracts_run': result.testsRun, 'contract_log': log.getvalue(),
        'actual_nonmember_probe': nonmember, 'cli_controls': controls,
        'local_record_bytes': [local.config_bytes, local.address_bytes],
        'candidate_sha256': tp_chain.TP_SHA,
        'actual_graph_sources': {inspect.getfile(value): digest(Path(inspect.getfile(value)))
                                 for value in (ACLGraphWrapper,
                                               tp_graph_constructor.run_device_rank)},
        'libraries': {str(path): digest(path) for path in (
            args.indexed, args.shared, args.decoder / 'complete_ffn.so',
            args.decoder / 'layer_local_chain.so', args.decoder / 'layer_tp_chain.so')},
        'shmem_libraries': shmem, 'npu_initialized_before': False,
        'npu_initialized_after': False, 'mapped_davinci_devices': [],
        'device_execution': False,
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'pass': True, 'contracts_run': result.testsRun,
                      'npu_initialized_after': False}))


if __name__ == '__main__':
    main()
